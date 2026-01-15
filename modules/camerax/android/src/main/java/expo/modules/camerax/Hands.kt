package expo.modules.camerax

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import com.qualcomm.qti.QnnDelegate
import com.qualcomm.qti.QnnDelegate.Options.BackendType
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * LiteRT/Qualcomm-backed hand pipeline placeholder.
 * Will mirror the MediaPipe hands flow using the mediapipe_hand detector/landmark assets.
 */
class Hands(private val context: Context) {
    companion object {
        private const val TAG = "HandsLiteRt"
        private const val DETECTOR_MODEL = "mediapipe_hand-handdetector.tflite"
        private const val LANDMARK_MODEL = "mediapipe_hand-handlandmarkdetector.tflite"
        private const val DETECTOR_INPUT_SIZE = 256
        private const val LANDMARK_INPUT_SIZE = 256
        // Detection/landmark score gates (mirror Python app; keep tighter detector gate)
        private const val DET_SCORE_THRESHOLD_DET = 0.5f
        private const val DET_SCORE_THRESHOLD_LMK = 0.5f
        // NMS IoU for palm detections (match Python sample)
        private const val NMS_IOU = 0.3f
        // ROI shaping: shift along wrist->middle vector (negative pulls toward fingertips), expand box.
        // Keep the negative offset as requested even though Python uses +0.5.
        private const val ROI_DXY = -0.25f
        private const val ROI_DSCALE = 2.5f
        // IoU match threshold when assigning detections to existing tracks
        private const val MATCH_IOU_THRESHOLD = 0.2f
        // Landmark smoothing (EMA) within a track
        private const val SMOOTH_ALPHA = 0.5f
        // ROI smoothing (EMA) when matching detections to existing tracks
        private const val ROI_SMOOTH_ALPHA = 0f
        // ROI prediction: MediaPipe-style velocity EMA and small cap to limit overshoot
        private const val ROI_PRED_ALPHA = 0.5f
        private const val ROI_PRED_CAP_FRAC = 0.15f
        // Tracking/backoff: consider a track "fresh" for this many ms; skip detector if fresh
        private const val TRACK_FRESH_MS = 100L
        // Drop tracks after this age (unless refreshed)
        private const val TRACK_MAX_AGE_MS = 1200L
        // Run detector every N frames when not forced (set to 1 to run every frame)
        private const val DETECTOR_PERIOD = 1
        private const val ROT_OFFSET = (Math.PI.toFloat() / 2f)
        private const val DEFAULT_HAND_ANCHORS = "anchors_palm.npy"
        // keypoints for rotation: wrist center idx 0, middle finger base idx 2 (from model.py)
        private const val KP_ROT_START = 0
        private const val KP_ROT_END = 2
        // 21 landmarks expected (to mirror Python MediaPipe hand landmark layout)
        private const val NUM_LANDMARKS = 21
    }

    private var detector: Interpreter? = null
    private var landmark: Interpreter? = null
    // Delegates now managed by DelegateManager.kt
    private var detectorCoordsPerAnchor = 0
    private var detectorNumAnchors = 0
    private var anchors: FloatArray? = null
    private var handClassifier: Interpreter? = null
    // Reusable buffers to cut allocations
    private var detectorTensorImage: TensorImage? = null
    private var detectorCoordsBuf: ByteBuffer? = null
    private var detectorScoresBuf: ByteBuffer? = null
    private var lmScoreBuf: ByteBuffer? = null
    private var lmLrBuf: ByteBuffer? = null
    private var lmLmkBuf: ByteBuffer? = null
    private var cropBitmaps = arrayOfNulls<Bitmap>(2)
    private val warpMatrix = Matrix()

    init {
        setupDetector()
        setupLandmark()
        try {
            MediapipeLiteRtUtil.loadNpyFloatArray(context.assets, DEFAULT_HAND_ANCHORS)?.let {
                anchors = it
                Log.i(TAG, "Loaded hand anchors (${it.size} floats)")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to load hand anchors: ${t.message}")
        }
        detector?.let {
            Log.i(TAG, "Hand detector input shape=${it.getInputTensor(0).shape().contentToString()} type=${it.getInputTensor(0).dataType()}")
        }
        landmark?.let {
            Log.i(TAG, "Hand landmark input shape=${it.getInputTensor(0).shape().contentToString()} type=${it.getInputTensor(0).dataType()}")
        }
    }

    fun close() {
        try { detector?.close() } catch (_: Throwable) {}
        try { landmark?.close() } catch (_: Throwable) {}
        // Note: Delegates are now managed by DelegateManager.kt
    }

    data class HandResult(
        val score: Float,
        val handedness: Boolean, // true = right, false = left
        val landmarks: List<Pair<Float, Float>>,
        val roi: FloatArray? = null,
        val prediction: String = ""  // Classification result (e.g., "Prediction: 0 (Confidence: 0.85)")
    )

    fun detectAndLandmark(
        bitmap: Bitmap,
        poseHints: List<List<Pair<Float, Float>>>? = null,
        isFrontCamera: Boolean = false
    ): List<HandResult> {
        Log.d(TAG, "hand detectAndLandmark frame=${bitmap.width}x${bitmap.height}")
        val det = detector ?: return emptyList()
        val prep = MediapipeLiteRtUtil.resizePadTo(bitmap, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE)
        val input = buildInputBuffer(prep.bitmap, det.getInputTensor(0))

        val coordsTensor = det.getOutputTensor(0)
        val scoresTensor = det.getOutputTensor(1)
        val (coords, scores) = runDetector(det, input, coordsTensor, scoresTensor)

        val lmInterp = landmark ?: return emptyList()
        val results = mutableListOf<HandResult>()
        val poseHint = poseHints?.firstOrNull()
        val anchorsLocal = anchors ?: run {
            Log.w(TAG, "Anchors missing; skipping hand detection")
            return results
        }
        val detections = MediapipeLiteRtUtil.decodeWithAnchors(
            coords = coords,
            scores = scores,
            anchors = anchorsLocal,
            numAnchors = detectorNumAnchors,
            coordsPerAnchor = detectorCoordsPerAnchor,
            scoreThreshold = DET_SCORE_THRESHOLD_DET,
            inputW = DETECTOR_INPUT_SIZE,
            inputH = DETECTOR_INPUT_SIZE
        )
        Log.d(TAG, "hand detector raw dets=${detections.size}")
        val nms = MediapipeLiteRtUtil.nms(detections, NMS_IOU)
        val limited = nms.sortedByDescending { it.score }.take(2)
        Log.d(TAG, "hand detector nms dets=${nms.size} capped=${limited.size}")

        for (detObjRaw in limited) {
            // Map detection back to original normalized coords.
            val x0 = (detObjRaw.x0 * prep.targetWidth - prep.padX) / prep.scale
            val y0 = (detObjRaw.y0 * prep.targetHeight - prep.padY) / prep.scale
            val x1 = (detObjRaw.x1 * prep.targetWidth - prep.padX) / prep.scale
            val y1 = (detObjRaw.y1 * prep.targetHeight - prep.padY) / prep.scale
            val mappedKp = detObjRaw.keypoints?.let { kps ->
                val out = FloatArray(kps.size)
                var i = 0
                while (i < kps.size) {
                    val kx = (kps[i] * prep.targetWidth - prep.padX) / prep.scale
                    val ky = (kps[i + 1] * prep.targetHeight - prep.padY) / prep.scale
                    out[i] = kx / prep.originalWidth
                    out[i + 1] = ky / prep.originalHeight
                    i += 2
                }
                out
            }
            val detObj = detObjRaw.copy(
                x0 = (x0 / prep.originalWidth),
                y0 = (y0 / prep.originalHeight),
                x1 = (x1 / prep.originalWidth),
                y1 = (y1 / prep.originalHeight),
                keypoints = mappedKp
            )
            // Build ROI corners from keypoints and box.
            val roi = computeHandRoi(detObj, bitmap.width, bitmap.height) ?: continue
            val (forward, inverse) = MediapipeLiteRtUtil.buildAffineFromRoi(
                roi,
                LANDMARK_INPUT_SIZE,
                LANDMARK_INPUT_SIZE
            )
            val crop = reuseWarp(bitmap, forward, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE, 1)
            val lmInput = buildInputBuffer(crop, lmInterp.getInputTensor(0))
            val out = runLandmark(lmInterp, lmInput, lmInterp.getOutputTensor(0), lmInterp.getOutputTensor(1), lmInterp.getOutputTensor(2))
            val score = out.first
            val lmk = out.third
            if (score < DET_SCORE_THRESHOLD_LMK || lmk.isEmpty()) {
                continue
            }
            val mapped = FloatArray(lmk.size)
            inverse?.mapPoints(mapped, lmk)
            val pairs = mutableListOf<Pair<Float, Float>>()
            var i = 0
            while (i < mapped.size) {
                pairs.add(Pair(mapped[i], mapped[i + 1]))
                i += 2
            }
            val handedModel = out.second
            val handedPose = poseHint?.let { inferHandednessFromPose(pairs, it) }
            val handed = handedPose ?: handedModel
            // Classify bow hand - for front camera the bow hand appears as "left" due to mirroring
            // so we classify if: (front camera AND left) OR (back camera AND right)
            val isBowHand = if (isFrontCamera) !handed else handed
            val prediction = if (isBowHand) {
                classifyHand(pairs, isFrontCamera)
            } else {
                ""
            }
            Log.d("classification", "Hand detection - side=${if (handed) "right" else "left"}, isBowHand=$isBowHand, score=$score, prediction=$prediction")
            results.add(HandResult(score, handed, pairs, roi, prediction))
        }
        Log.d(TAG, "hand landmarks count=${results.size}")
        return results
    }

    private fun computeHandRoi(det: MediapipeLiteRtUtil.Detection, imgW: Int, imgH: Int): FloatArray? {
        val kps = det.keypoints ?: return null
        val kpCount = kps.size / 2
        if (kpCount <= KP_ROT_END) return null

        val sx = kps[KP_ROT_START * 2]
        val sy = kps[KP_ROT_START * 2 + 1]
        val ex = kps[KP_ROT_END * 2]
        val ey = kps[KP_ROT_END * 2 + 1]
        // MediaPipe uses wrist -> middle-finger-base vector for orientation.
        // Match MediaPipe vector direction (start - end) so the rotation/offset aligns with the reference Python app.
        val dx = sx - ex
        val dy = sy - ey
        // MediaPipe (y-down): rotation = atan2(dy, dx) - 0.5*pi.
        val angle = kotlin.math.atan2(dy, dx) - ROT_OFFSET
        val angleDeg = Math.toDegrees(angle.toDouble())
        Log.d(TAG, "ROI rot wrist->mid dx=$dx dy=$dy angleDeg=$angleDeg sx=$sx sy=$sy ex=$ex ey=$ey")

        // Box center/size from detector box.
        val x0 = det.x0
        val y0 = det.y0
        val x1 = det.x1
        val y1 = det.y1
        val xc = (x0 + x1) / 2f
        val yc = (y0 + y1) / 2f
        val wPix = (x1 - x0) * ROI_DSCALE * imgW
        val hPix = (y1 - y0) * ROI_DSCALE * imgH

        // Apply directional offset along rotation vector.
        val dxPix = dx * imgW
        val dyPix = dy * imgH
        val vecLen = kotlin.math.hypot(dxPix, dyPix)
        val offset = if (vecLen > 1e-6f) ROI_DXY * wPix else 0f
        val xcPix = xc * imgW + offset * (if (vecLen > 1e-6f) dxPix / vecLen else 0f)
        val ycPix = yc * imgH + offset * (if (vecLen > 1e-6f) dyPix / vecLen else 0f)

        val cosA = kotlin.math.cos(angle)
        val sinA = kotlin.math.sin(angle)
        val hw = wPix / 2f
        val hh = hPix / 2f
        // Order TL, BL, TR, BR to match MediaPipe rotated rect and affine mapping.
        val pts = floatArrayOf(
            -hw, -hh,
            -hw, hh,
            hw, -hh,
            hw, hh
        )
        for (i in 0 until 4) {
            val x = pts[i * 2]
            val y = pts[i * 2 + 1]
            pts[i * 2] = x * cosA - y * sinA + xcPix
            pts[i * 2 + 1] = x * sinA + y * cosA + ycPix
        }
        Log.d(TAG, "ROI pts=${pts.contentToString()} center=($xcPix,$ycPix) wh=($wPix,$hPix) img=($imgW,$imgH)")
        return pts
    }

    // Infer handedness using pose wrists: choose the closer wrist to the hand wrist (landmark 0).
    private fun inferHandednessFromPose(handLandmarks: List<Pair<Float, Float>>, pose: List<Pair<Float, Float>>): Boolean? {
        val wrist = handLandmarks.getOrNull(0) ?: return null
        val leftPose = pose.getOrNull(15) // MediaPipe left wrist
        val rightPose = pose.getOrNull(16) // MediaPipe right wrist
        if (leftPose == null && rightPose == null) return null
        val dLeft = leftPose?.let { d2(it, wrist) }
        val dRight = rightPose?.let { d2(it, wrist) }
        return when {
            dLeft != null && dRight != null -> dRight < dLeft // closer to right wrist => right hand
            dRight != null -> true
            dLeft != null -> false
            else -> null
        }
    }

    private fun d2(a: Pair<Float, Float>, b: Pair<Float, Float>): Float {
        val dx = a.first - b.first
        val dy = a.second - b.second
        return dx * dx + dy * dy
    }

    private fun setupDetector() {
        // Delegation moved to DelegateManager.kt
        detector = DelegateManager.createInterpreter(context, DETECTOR_MODEL, "Hands")
        val out0 = detector?.getOutputTensor(0)
        detectorCoordsPerAnchor = out0?.shape()?.lastOrNull() ?: 0
        detectorNumAnchors = out0?.shape()?.getOrNull(out0.shape().size - 2) ?: 0
    }

    private fun setupLandmark() {
        // Delegation moved to DelegateManager.kt
        landmark = DelegateManager.createInterpreter(context, LANDMARK_MODEL, "HandsLM")
    }

    // COMMENTED OUT - moved to DelegateManager.kt
    // private fun buildInterpreterOptions(choice: DelegateChoice, name: String): Interpreter.Options { ... }
    // private fun tryLoadQnnAndPickSkelDir(): String? { ... }

    private fun buildInputBuffer(bitmap: Bitmap, inputTensor: Tensor): Any {
        val shape = inputTensor.shape()
        val isNhwc = shape.size == 4 && shape[3] == 3
        val isNchw = shape.size == 4 && shape[1] == 3
        return when (inputTensor.dataType()) {
            DataType.FLOAT32 -> {
                if (isNhwc) {
                    val imageProcessor = ImageProcessor.Builder()
                        .add(NormalizeOp(0f, 255f))
                        .build()
                    val tensorImage = detectorTensorImage ?: TensorImage(DataType.FLOAT32).also { detectorTensorImage = it }
                    tensorImage.load(bitmap)
                    val processed = imageProcessor.process(tensorImage)
                    processed.buffer
                } else {
                    val buffer = ByteBuffer.allocateDirect(4 * inputTensor.numElements()).order(ByteOrder.nativeOrder())
                    val pixels = IntArray(bitmap.width * bitmap.height)
                    bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                    for (c in 0 until 3) {
                        for (p in pixels) {
                            val v = when (c) {
                                0 -> (p shr 16) and 0xFF
                                1 -> (p shr 8) and 0xFF
                                else -> p and 0xFF
                            }
                            buffer.putFloat(v / 255f)
                        }
                    }
                    buffer
                }
            }
            DataType.UINT8 -> {
                if (isNhwc) {
                    val tensorImage = TensorImage(DataType.UINT8)
                    tensorImage.load(bitmap)
                    tensorImage.buffer
                } else {
                    val buffer = ByteBuffer.allocateDirect(inputTensor.numElements()).order(ByteOrder.nativeOrder())
                    val pixels = IntArray(bitmap.width * bitmap.height)
                    bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                    for (c in 0 until 3) {
                        for (p in pixels) {
                            val v = when (c) {
                                0 -> (p shr 16) and 0xFF
                                1 -> (p shr 8) and 0xFF
                                else -> p and 0xFF
                            }
                            buffer.put(v.toByte())
                        }
                    }
                    buffer
                }
            }
            DataType.INT8 -> {
                val quant = inputTensor.quantizationParams()
                val scale = quant.scale
                val zeroPoint = quant.zeroPoint
                val buffer = ByteBuffer.allocateDirect(inputTensor.numElements()).order(ByteOrder.nativeOrder())
                val pixels = IntArray(bitmap.width * bitmap.height)
                bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                val pushQuant: (Int) -> Unit = { p ->
                    val real = p / 255f
                    val q = (real / scale + zeroPoint).toInt().coerceIn(-128, 127)
                    buffer.put(q.toByte())
                }
                if (isNhwc) {
                    for (p in pixels) {
                        pushQuant((p shr 16) and 0xFF)
                        pushQuant((p shr 8) and 0xFF)
                        pushQuant(p and 0xFF)
                    }
                } else {
                    for (c in 0 until 3) {
                        for (p in pixels) {
                            val v = when (c) {
                                0 -> (p shr 16) and 0xFF
                                1 -> (p shr 8) and 0xFF
                                else -> p and 0xFF
                            }
                            pushQuant(v)
                        }
                    }
                }
                buffer
            }
            else -> throw IllegalStateException("Unsupported input type ${inputTensor.dataType()}")
        }
    }

    private fun allocateOutput(tensor: Tensor): Pair<DataType, Any> {
        val dt = tensor.dataType()
        val numBytes = tensor.numBytes()
        return when (dt) {
            DataType.FLOAT32 -> Pair(dt, FloatArray(tensor.numElements()))
            DataType.UINT8, DataType.INT8 -> Pair(dt, ByteArray(numBytes))
            else -> throw IllegalStateException("Unsupported output type $dt")
        }
    }

    private fun dequantizeOutput(
        tensor: Tensor,
        dataType: DataType,
        raw: Any
    ): FloatArray {
        val numElems = tensor.numElements()
        return when (dataType) {
            DataType.FLOAT32 -> raw as FloatArray
            DataType.UINT8 -> {
                val scale = tensor.quantizationParams().scale
                val zero = tensor.quantizationParams().zeroPoint
                val arr = raw as ByteArray
                FloatArray(numElems) { idx ->
                    val v = arr[idx].toInt() and 0xFF
                    scale * (v - zero)
                }
            }
            DataType.INT8 -> {
                val scale = tensor.quantizationParams().scale
                val zero = tensor.quantizationParams().zeroPoint
                val arr = raw as ByteArray
                FloatArray(numElems) { idx ->
                    val v = arr[idx].toInt()
                    scale * (v - zero)
                }
            }
            else -> throw IllegalStateException("Unsupported type $dataType")
        }
    }

    // Reuse warp into preallocated bitmaps to cut down allocations.
    private fun reuseWarp(src: Bitmap, matrix: Matrix, dstW: Int, dstH: Int, slot: Int): Bitmap {
        val idx = slot.coerceIn(0, cropBitmaps.size - 1)
        val existing = cropBitmaps[idx]
        val out = if (existing == null || existing.width != dstW || existing.height != dstH) {
            Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888).also { cropBitmaps[idx] = it }
        } else existing
        warpMatrix.reset()
        warpMatrix.set(matrix)
        val canvas = Canvas(out)
        canvas.drawBitmap(src, warpMatrix, null)
        return out
    }

    private fun runDetector(
        interpreter: Interpreter,
        input: Any,
        coordsTensor: Tensor,
        scoresTensor: Tensor
    ): Pair<FloatArray, FloatArray> {
        val t0 = System.currentTimeMillis()
        if (detectorCoordsBuf == null || detectorCoordsBuf!!.capacity() < coordsTensor.numBytes()) {
            detectorCoordsBuf = ByteBuffer.allocateDirect(coordsTensor.numBytes()).order(ByteOrder.nativeOrder())
        }
        if (detectorScoresBuf == null || detectorScoresBuf!!.capacity() < scoresTensor.numBytes()) {
            detectorScoresBuf = ByteBuffer.allocateDirect(scoresTensor.numBytes()).order(ByteOrder.nativeOrder())
        }
        val coordsBuf = detectorCoordsBuf!!
        val scoresBuf = detectorScoresBuf!!
        coordsBuf.clear(); scoresBuf.clear()
        val outputs: MutableMap<Int, Any> = hashMapOf(0 to coordsBuf, 1 to scoresBuf)
        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
        val inferenceTime = System.currentTimeMillis() - t0
        val coords = dequantizeBuffer(coordsTensor, coordsBuf)
        val rawScores = dequantizeBuffer(scoresTensor, scoresBuf)
        val activatedScores = FloatArray(rawScores.size) { idx ->
            val v = rawScores[idx].coerceIn(-100f, 100f)
            1f / (1f + exp(-v))
        }
        if (rawScores.isNotEmpty()) {
            val rMin = rawScores.minOrNull() ?: 0f
            val rMax = rawScores.maxOrNull() ?: 0f
            val aMin = activatedScores.minOrNull() ?: 0f
            val aMax = activatedScores.maxOrNull() ?: 0f
            Log.d(TAG, "hand detector scores raw min=$rMin max=$rMax activated min=$aMin max=$aMax")
        }
        Log.d("Inference Metrics", "Hand detector inference=${inferenceTime}ms")
        return Pair(coords, activatedScores)
    }

    private fun runLandmark(
        interpreter: Interpreter,
        input: Any,
        scoreTensor: Tensor,
        lrTensor: Tensor,
        lmkTensor: Tensor
    ): Triple<Float, Boolean, FloatArray> {
        if (lmScoreBuf == null || lmScoreBuf!!.capacity() < scoreTensor.numBytes()) {
            lmScoreBuf = ByteBuffer.allocateDirect(scoreTensor.numBytes()).order(ByteOrder.nativeOrder())
        }
        if (lmLrBuf == null || lmLrBuf!!.capacity() < lrTensor.numBytes()) {
            lmLrBuf = ByteBuffer.allocateDirect(lrTensor.numBytes()).order(ByteOrder.nativeOrder())
        }
        if (lmLmkBuf == null || lmLmkBuf!!.capacity() < lmkTensor.numBytes()) {
            lmLmkBuf = ByteBuffer.allocateDirect(lmkTensor.numBytes()).order(ByteOrder.nativeOrder())
        }
        val scoreBuf = lmScoreBuf!!.apply { clear() }
        val lrBuf = lmLrBuf!!.apply { clear() }
        val lmkBuf = lmLmkBuf!!.apply { clear() }
        val outputs: MutableMap<Int, Any> = hashMapOf(
            0 to scoreBuf,
            1 to lrBuf,
            2 to lmkBuf
        )
        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
        val scoreArr = dequantizeBuffer(scoreTensor, scoreBuf)
        val lrArr = dequantizeBuffer(lrTensor, lrBuf)
        val lmkArr = dequantizeBuffer(lmkTensor, lmkBuf)

        val numDims = lmkTensor.shape().lastOrNull() ?: 0
        val numPts = lmkTensor.shape().getOrNull(lmkTensor.shape().size - 2) ?: 0
        val coords = FloatArray(numPts * 2)
        for (i in 0 until numPts) {
            coords[i * 2] = lmkArr[i * numDims] * LANDMARK_INPUT_SIZE
            coords[i * 2 + 1] = lmkArr[i * numDims + 1] * LANDMARK_INPUT_SIZE
        }
        val score = scoreArr.firstOrNull() ?: 0f
        val handed = (lrArr.firstOrNull() ?: 0f) >= 0.5f
        return Triple(score, handed, coords)
    }

    private fun dequantizeBuffer(tensor: Tensor, buffer: ByteBuffer): FloatArray {
        buffer.rewind()
        val numElems = tensor.numElements()
        return when (tensor.dataType()) {
            DataType.FLOAT32 -> {
                val out = FloatArray(numElems)
                buffer.asFloatBuffer().get(out)
                out
            }
            DataType.UINT8 -> {
                val scale = tensor.quantizationParams().scale
                val zero = tensor.quantizationParams().zeroPoint
                val out = FloatArray(numElems)
                var i = 0
                while (i < numElems) {
                    val v = buffer.get().toInt() and 0xFF
                    out[i] = scale * (v - zero)
                    i++
                }
                out
            }
            DataType.INT8 -> {
                val scale = tensor.quantizationParams().scale
                val zero = tensor.quantizationParams().zeroPoint
                val out = FloatArray(numElems)
                var i = 0
                while (i < numElems) {
                    val v = buffer.get().toInt()
                    out[i] = scale * (v - zero)
                    i++
                }
                out
            }
            else -> throw IllegalStateException("Unsupported type ${tensor.dataType()}")
        }
    }

    /**
     * Classify hand posture using wrist-relative normalized coordinates.
     * Matches HandLandmarkerHelper.extractHandCoordinates + runTFLiteInference logic.
     *
     * @param landmarks 21 hand landmarks as (x, y) pairs
     * @param isFrontCamera true if using front camera (affects X flip)
     * @return Prediction string like "Prediction: 0 (Confidence: 0.85)" or empty on error
     */
    private fun classifyHand(landmarks: List<Pair<Float, Float>>, isFrontCamera: Boolean): String {
        try {
            if (handClassifier == null) {
                val model = FileUtil.loadMappedFile(context, "keypoint_classifier_FINAL.tflite")
                handClassifier = Interpreter(model)
                Log.d(TAG, "Loaded hand classifier")
            }
            if (landmarks.size < 21) return ""

            // Extract wrist-relative coordinates (matches HandLandmarkerHelper.extractHandCoordinates)
            val originX = landmarks[0].first
            val originY = landmarks[0].second
            val coords = FloatArray(42)
            var maxAbsValue = 0f

            for ((j, lm) in landmarks.withIndex()) {
                var relX = lm.first - originX
                val relY = lm.second - originY

                // Flip X for back camera (matches HandLandmarkerHelper)
                if (!isFrontCamera) {
                    relX *= -1
                }

                coords[j * 2] = relX
                coords[j * 2 + 1] = relY
                maxAbsValue = maxOf(maxAbsValue, kotlin.math.abs(relX), kotlin.math.abs(relY))
            }

            // Normalize by max absolute value (matches HandLandmarkerHelper)
            if (maxAbsValue > 0f) {
                for (i in coords.indices) {
                    coords[i] /= maxAbsValue
                }
            }

            // Run classifier
            val output = Array(1) { FloatArray(4) }
            val t0 = android.os.SystemClock.uptimeMillis()
            handClassifier?.run(arrayOf(coords), output)
            val t1 = android.os.SystemClock.uptimeMillis()

            val results = output[0]

            // Apply supination penalty (matches HandLandmarkerHelper)
            val supinationIdx = 1
            results[supinationIdx] *= 0.7f

            val maxIdx = results.indices.maxByOrNull { results[it] } ?: 0
            val confidence = results[maxIdx]

            // Low-confidence supination → report as class 0 (matches HandLandmarkerHelper)
            if (maxIdx == supinationIdx && confidence < 0.60f) {
                Log.d(TAG, "Hand class=0 (supination below threshold) conf=$confidence time=${t1 - t0}ms")
                return "Prediction: 0 (Confidence: %.2f)".format(confidence)
            }

            Log.d(TAG, "Hand class=$maxIdx conf=$confidence time=${t1 - t0}ms")
            return "Prediction: $maxIdx (Confidence: %.2f)".format(confidence)
        } catch (t: Throwable) {
            Log.w(TAG, "Hand classifier failed: ${t.message}")
            return ""
        }
    }
}
