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
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * LiteRT/Qualcomm-backed pose pipeline placeholder.
 * Will mirror the MediaPipe pose flow using the mediapipe_pose detector/landmark assets.
 */
class Pose(private val context: Context) {
    companion object {
        private const val TAG = "PoseLiteRt"
        // Model selection: w8a8 for NPU (HTP), float for GPU/CPU
        private const val DETECTOR_MODEL_W8A8 = "mediapipe_pose-posedetector-w8a8.tflite"
        private const val LANDMARK_MODEL_W8A8 = "mediapipe_pose-poselandmarkdetector-w8a8.tflite"
        private const val DETECTOR_MODEL_FLOAT = "mediapipe_pose-posedetector-float.tflite"
        private const val LANDMARK_MODEL_FLOAT = "mediapipe_pose-poselandmarkdetector-float.tflite"
        // Detector input dims from model card (3x128x128)
        private const val DETECTOR_INPUT_SIZE = 128
        private const val LANDMARK_INPUT_SIZE = 256
        private const val DET_SCORE_THRESHOLD = 0.2f
        private const val LM_SCORE_THRESHOLD = 0.5f
        private const val NMS_IOU = 0.5f
        private const val ROI_SCALE = 1.75f
        private const val POSE_ROI_DXY = 0f
        private const val DEFAULT_POSE_ANCHORS = "anchors_pose.npy"

        // Pose landmark connections (subset from model.py)
        val POSE_CONNECTIONS = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 7,
            0 to 4, 4 to 5, 5 to 6, 6 to 8,
            11 to 13, 13 to 15, 15 to 17, 17 to 19,
            19 to 15, 15 to 21, 12 to 14, 14 to 16,
            16 to 18, 18 to 20, 20 to 16, 16 to 22,
            11 to 12, 12 to 24, 24 to 23, 23 to 11
        )
    }

    private var detector: Interpreter? = null
    private var landmark: Interpreter? = null
    private var poseClassifier: Interpreter? = null
    // Delegates now managed by DelegateManager.kt
    private var detectorInputType: DataType = DataType.FLOAT32
    private var detectorCoordsPerAnchor = 0
    private var detectorNumAnchors = 0
    private var anchors: FloatArray? = null
    private var lastRoi: FloatArray? = null
    private var lastLandmarksArr: FloatArray? = null
    // Reusable buffers to cut allocations
    private var detectorTensorImage: TensorImage? = null
    private var coords0Buf: ByteBuffer? = null
    private var coords1Buf: ByteBuffer? = null
    private var scores0Buf: ByteBuffer? = null
    private var scores1Buf: ByteBuffer? = null
    private var coordsBuf: ByteBuffer? = null
    private var scoresBuf: ByteBuffer? = null
    private var lmScoreBuf: ByteBuffer? = null
    private var lmLmkBuf: ByteBuffer? = null
    private var cropBitmap: Bitmap? = null
    private val warpMatrix = Matrix()

    init {
        setupDetector()
        setupLandmark()
        // Attempt to load default anchors from assets.
        try {
            MediapipeLiteRtUtil.loadNpyFloatArray(context.assets, DEFAULT_POSE_ANCHORS)?.let {
                anchors = it
                Log.i(TAG, "Loaded anchors from $DEFAULT_POSE_ANCHORS (${it.size} floats)")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to load default pose anchors: ${t.message}")
        }
    }

    /**
     * Provide anchors for pose detector decoding. Expected flatten [numAnchors * 4] (x,y,w,h).
     */
    fun setAnchors(anchorArray: FloatArray) {
        anchors = anchorArray
    }

    fun close() {
        try { detector?.close() } catch (_: Throwable) {}
        try { landmark?.close() } catch (_: Throwable) {}
        // Note: Delegates are now managed by DelegateManager.kt
    }

    /**
     * Run detector (steps 1-3): preprocess -> infer -> decode anchors -> NMS.
     */
    fun detectPoses(bitmap: Bitmap): List<MediapipeLiteRtUtil.Detection> {
        val det = detector ?: return emptyList()
        val prep = MediapipeLiteRtUtil.resizePadTo(bitmap, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE)
        val inputBuffer = buildInputBuffer(prep.bitmap, det.getInputTensor(0))

        val coordsTensor = det.getOutputTensor(0)
        val scoresTensor = det.getOutputTensor(1)
        val coordsData = runDetector(det, inputBuffer, coordsTensor, scoresTensor)

        val anchorsLocal = anchors
        if (anchorsLocal == null) {
            Log.w(TAG, "Anchors not set; cannot decode detections.")
            return emptyList()
        }

        val detections = MediapipeLiteRtUtil.decodeWithAnchors(
            coords = coordsData.first,
            scores = coordsData.second,
            anchors = anchorsLocal,
            numAnchors = detectorNumAnchors,
            coordsPerAnchor = detectorCoordsPerAnchor,
            scoreThreshold = DET_SCORE_THRESHOLD,
            inputW = DETECTOR_INPUT_SIZE,
            inputH = DETECTOR_INPUT_SIZE
        )

        Log.d(TAG, "pose detector raw dets=${detections.size}")
        val nms = MediapipeLiteRtUtil.nms(detections, NMS_IOU)
        val limited = nms.sortedByDescending { it.score }.take(3)
        Log.d(TAG, "pose detector nms dets=${nms.size} capped=${limited.size}")
        return limited.map { detObj ->
            // Map back to original image coordinates.
            val x0 = (detObj.x0 * prep.targetWidth - prep.padX) / prep.scale
            val y0 = (detObj.y0 * prep.targetHeight - prep.padY) / prep.scale
            val x1 = (detObj.x1 * prep.targetWidth - prep.padX) / prep.scale
            val y1 = (detObj.y1 * prep.targetHeight - prep.padY) / prep.scale

            val mappedKp = detObj.keypoints?.let { kps ->
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
            detObj.copy(
                x0 = x0 / prep.originalWidth,
                y0 = y0 / prep.originalHeight,
                x1 = x1 / prep.originalWidth,
                y1 = y1 / prep.originalHeight,
                keypoints = mappedKp
            )
        }
    }

    data class PoseLandmarks(
        val score: Float,
        val landmarks: List<Pair<Float, Float>>,
        val roi: FloatArray? = null,
        val detKeypoints: FloatArray? = null, // normalized detector keypoints (x0,y0,...)
        val prediction: String = ""  // Classification result (e.g., "Prediction: 0 (Confidence: 0.85)")
    )

    /**
     * Full pose path: detect -> ROI -> landmark -> map back.
     */
    fun detectAndLandmark(bitmap: Bitmap, isFrontCamera: Boolean = false): List<PoseLandmarks> {
        Log.d(TAG, "pose detectAndLandmark frame=${bitmap.width}x${bitmap.height}")
        val lmInterp = landmark ?: return emptyList()
        // Step 1: try reuse last ROI (with smoothing) before running detector.
        lastRoi?.let { prevRoi ->
            val reuse = prevRoi
            val (forward, inverse) = MediapipeLiteRtUtil.buildAffineFromRoi(
                reuse,
                LANDMARK_INPUT_SIZE,
                LANDMARK_INPUT_SIZE
            )
            val crop = reuseWarp(bitmap, forward, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE)
            val input = buildInputBuffer(crop, lmInterp.getInputTensor(0))
            val scoreTensor = lmInterp.getOutputTensor(0)
            val lmkTensor = lmInterp.getOutputTensor(1)
            val out = runLandmark(lmInterp, input, scoreTensor, lmkTensor)
            if (out.first >= LM_SCORE_THRESHOLD && out.second.isNotEmpty()) {
                val mapped = FloatArray(out.second.size)
                inverse?.mapPoints(mapped, out.second)
                lastLandmarksArr?.let { prev ->
                    if (prev.size == mapped.size) {
                        for (i in mapped.indices) {
                            mapped[i] = 0.5f * prev[i] + 0.5f * mapped[i]
                        }
                    }
                }
                lastLandmarksArr = mapped.copyOf()
                val pairs = mutableListOf<Pair<Float, Float>>()
                var i = 0
                while (i < mapped.size) {
                    pairs.add(Pair(mapped[i], mapped[i + 1]))
                    i += 2
                }
                val prediction = classifyPose(pairs, bitmap.width, bitmap.height, isFrontCamera)
                Log.d("classification", "Pose detection - score=${out.first}, points=${pairs.size}, prediction=$prediction (reuse)")
                return listOf(PoseLandmarks(out.first, pairs, reuse, null, prediction))
            }
        }
        val detections = detectPoses(bitmap)
        if (detections.isEmpty()) {
            Log.d(TAG, "pose landmarks count=0")
            return emptyList()
        }
        val results = mutableListOf<PoseLandmarks>()
        for (det in detections) {
            val roiCorners = MediapipeLiteRtUtil.computePoseRoiCorners(
                det,
                bitmap.width,
                bitmap.height,
                boxScale = ROI_SCALE,
                keypointStartIdx = 2,
                keypointEndIdx = 3,
                dxy = POSE_ROI_DXY
            ) ?: continue
            val smoothedRoi = lastRoi?.let { prev ->
                if (prev.size == roiCorners.size) {
                    FloatArray(prev.size) { idx ->
                        0.5f * prev[idx] + 0.5f * roiCorners[idx]
                    }
                } else roiCorners
            } ?: roiCorners
            val (forward, inverse) = MediapipeLiteRtUtil.buildAffineFromRoi(
                smoothedRoi,
                LANDMARK_INPUT_SIZE,
                LANDMARK_INPUT_SIZE
            )
            val crop = reuseWarp(bitmap, forward, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE)
            val input = buildInputBuffer(crop, lmInterp.getInputTensor(0))
            val scoreTensor = lmInterp.getOutputTensor(0)
            val lmkTensor = lmInterp.getOutputTensor(1)
            val out = runLandmark(lmInterp, input, scoreTensor, lmkTensor)
            if (out.first < LM_SCORE_THRESHOLD || out.second.isEmpty()) {
                Log.d(TAG, "pose landmarks skip (low score ${out.first})")
                continue
            }
            val mapped = FloatArray(out.second.size)
            inverse?.mapPoints(mapped, out.second)
            val pairs = mutableListOf<Pair<Float, Float>>()
            var i = 0
            while (i < mapped.size) {
                pairs.add(Pair(mapped[i], mapped[i + 1]))
                i += 2
            }
            val prediction = classifyPose(pairs, bitmap.width, bitmap.height, isFrontCamera)
            Log.d("classification", "Pose detection - score=${out.first}, points=${pairs.size}, prediction=$prediction (detector)")
            lastRoi = smoothedRoi.copyOf()
            lastLandmarksArr = mapped.copyOf()
            results.add(PoseLandmarks(out.first, pairs, smoothedRoi, det.keypoints, prediction))
            if (results.isNotEmpty()) break
        }
        Log.d(TAG, "pose landmarks count=${results.size}")
        return results
    }

    private fun clamp01(v: Float): Float = max(0f, min(1f, v))

    /**
     * Classify pose (elbow position) using 9-float input.
     * Matches HandLandmarkerHelper.extractPoseCoordinates + runTFLitePoseInference logic.
     *
     * NOTE: Pose.kt landmarks are 2D only (no z coordinate). We use z=0.
     * This may reduce accuracy slightly compared to MediaPipe's 3D landmarks,
     * but should work for basic "elbow high/low" detection.
     *
     * @param landmarks 33 pose landmarks as (x, y) pairs IN PIXEL COORDINATES
     * @param imageWidth width of the source image (for normalization)
     * @param imageHeight height of the source image (for normalization)
     * @param isFrontCamera true if using front camera (affects arm selection)
     * @return Prediction string like "Prediction: 0 (Confidence: 0.85)" or empty on error
     */
    private fun classifyPose(
        landmarks: List<Pair<Float, Float>>,
        imageWidth: Int,
        imageHeight: Int,
        isFrontCamera: Boolean
    ): String {
        try {
            if (poseClassifier == null) {
                val model = FileUtil.loadMappedFile(context, "keypoint_classifier (1).tflite")
                poseClassifier = Interpreter(model)
                Log.d(TAG, "Loaded pose classifier")
            }

            // Select arm landmarks based on camera (matches HandLandmarkerHelper)
            // Front camera: left arm (11, 13, 15) - player's right arm appears on left of image
            // Back camera: right arm (12, 14, 16)
            val (shoulderIdx, elbowIdx, handIdx) = if (isFrontCamera) {
                Triple(11, 13, 15)
            } else {
                Triple(12, 14, 16)
            }

            if (landmarks.size <= maxOf(shoulderIdx, elbowIdx, handIdx)) {
                return ""
            }

            val shoulder = landmarks[shoulderIdx]
            val elbow = landmarks[elbowIdx]
            val hand = landmarks[handIdx]

            // Convert pixel coordinates to normalized 0-1 coordinates
            // (matches MediaPipe's normalized landmark format)
            val shoulderNormX = shoulder.first / imageWidth
            val shoulderNormY = shoulder.second / imageHeight
            val elbowNormX = elbow.first / imageWidth
            val elbowNormY = elbow.second / imageHeight
            val handNormX = hand.first / imageWidth
            val handNormY = hand.second / imageHeight

            // Apply camera flip for back camera (matches HandLandmarkerHelper)
            val shoulderX = if (isFrontCamera) shoulderNormX else 1f - shoulderNormX
            val shoulderY = shoulderNormY
            val elbowX = if (isFrontCamera) elbowNormX else 1f - elbowNormX
            val elbowY = elbowNormY
            val handX = if (isFrontCamera) handNormX else 1f - handNormX
            val handY = handNormY

            // Compute vectors (z=0 for 2D landmarks)
            val seVec = floatArrayOf(
                shoulderX - elbowX,
                shoulderY - elbowY,
                0f  // z=0 (2D landmarks)
            )
            val heVec = floatArrayOf(
                handX - elbowX,
                handY - elbowY,
                0f  // z=0 (2D landmarks)
            )

            // Compute distances
            val seDist = sqrt(seVec[0] * seVec[0] + seVec[1] * seVec[1] + seVec[2] * seVec[2])
            val heDist = sqrt(heVec[0] * heVec[0] + heVec[1] * heVec[1] + heVec[2] * heVec[2])

            if (seDist <= 0f || heDist <= 0f) return ""

            // Compute angle (theta)
            val dotProduct = seVec[0] * heVec[0] + seVec[1] * heVec[1] + seVec[2] * heVec[2]
            val cosTheta = (dotProduct / (seDist * heDist)).coerceIn(-1f, 1f)
            val theta = acos(cosTheta)

            // Build 9-float input (matches HandLandmarkerHelper.extractPoseCoordinates)
            val coords = floatArrayOf(
                seVec[0] / seDist, seVec[1] / seDist, seVec[2] / seDist,  // normalized shoulder-elbow vector
                heVec[0] / heDist, heVec[1] / heDist, heVec[2] / heDist,  // normalized hand-elbow vector
                theta,      // angle in radians
                seDist,     // shoulder-elbow distance (now in normalized coords ~0.1-0.3)
                heDist      // hand-elbow distance (now in normalized coords ~0.1-0.3)
            )

            Log.d(TAG, "Pose classifier input: se=(${coords[0]},${coords[1]},${coords[2]}) he=(${coords[3]},${coords[4]},${coords[5]}) theta=${coords[6]} seDist=${coords[7]} heDist=${coords[8]}")

            // Run classifier
            val output = Array(1) { FloatArray(3) }
            val t0 = android.os.SystemClock.uptimeMillis()
            poseClassifier?.run(arrayOf(coords), output)
            val t1 = android.os.SystemClock.uptimeMillis()

            val results = output[0]
            val maxIdx = results.indices.maxByOrNull { results[it] } ?: 0
            val confidence = results[maxIdx]

            Log.d(TAG, "Pose class=$maxIdx conf=$confidence time=${t1 - t0}ms")
            return "Prediction: $maxIdx (Confidence: %.2f)".format(confidence)
        } catch (t: Throwable) {
            Log.w(TAG, "Pose classifier failed: ${t.message}")
            return ""
        }
    }

    private fun setupDetector() {
        // Select model based on delegate: w8a8 for NPU, float for GPU/CPU
        val modelPath = if (DelegateManager.isHtp()) DETECTOR_MODEL_W8A8 else DETECTOR_MODEL_FLOAT
        Log.i(TAG, "Pose detector model=$modelPath")
        detector = DelegateManager.createInterpreter(context, modelPath, "Pose")
        detectorInputType = detector?.getInputTensor(0)?.dataType() ?: DataType.FLOAT32
    }

    private fun setupLandmark() {
        // Select model based on delegate: w8a8 for NPU, float for GPU/CPU
        val modelPath = if (DelegateManager.isHtp()) LANDMARK_MODEL_W8A8 else LANDMARK_MODEL_FLOAT
        Log.i(TAG, "Pose landmark model=$modelPath")
        landmark = DelegateManager.createInterpreter(context, modelPath, "PoseLM")
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
                    // Manual NCHW packing normalized to [0,1]
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
                // Quantize to int8 using input scale/zero point.
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

            else -> {
                throw IllegalStateException("Unsupported input type ${inputTensor.dataType()}")
            }
        }
    }

    private fun runDetector(
        interpreter: Interpreter,
        input: Any,
        coordsTensor: Tensor,
        scoresTensor: Tensor
    ): Pair<FloatArray, FloatArray> {
        val t0 = System.currentTimeMillis()
        return if (interpreter.outputTensorCount >= 4) {
            val coords0 = interpreter.getOutputTensor(0)
            val coords1 = interpreter.getOutputTensor(1)
            val scores0 = interpreter.getOutputTensor(2)
            val scores1 = interpreter.getOutputTensor(3)

            if (coords0Buf == null || coords0Buf!!.capacity() < coords0.numBytes()) {
                coords0Buf = ByteBuffer.allocateDirect(coords0.numBytes()).order(ByteOrder.nativeOrder())
            }
            if (coords1Buf == null || coords1Buf!!.capacity() < coords1.numBytes()) {
                coords1Buf = ByteBuffer.allocateDirect(coords1.numBytes()).order(ByteOrder.nativeOrder())
            }
            if (scores0Buf == null || scores0Buf!!.capacity() < scores0.numBytes()) {
                scores0Buf = ByteBuffer.allocateDirect(scores0.numBytes()).order(ByteOrder.nativeOrder())
            }
            if (scores1Buf == null || scores1Buf!!.capacity() < scores1.numBytes()) {
                scores1Buf = ByteBuffer.allocateDirect(scores1.numBytes()).order(ByteOrder.nativeOrder())
            }
            val c0Buf = coords0Buf!!.apply { clear() }
            val c1Buf = coords1Buf!!.apply { clear() }
            val s0Buf = scores0Buf!!.apply { clear() }
            val s1Buf = scores1Buf!!.apply { clear() }
            val outputs: MutableMap<Int, Any> = hashMapOf(
                0 to c0Buf,
                1 to c1Buf,
                2 to s0Buf,
                3 to s1Buf
            )
            interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            val inferenceTime = System.currentTimeMillis() - t0

            val c0 = dequantizeBuffer(coords0, c0Buf)
            val c1 = dequantizeBuffer(coords1, c1Buf)
            val s0Raw = dequantizeBuffer(scores0, s0Buf)
            val s1Raw = dequantizeBuffer(scores1, s1Buf)

            val a0 = coords0.shape().getOrNull(coords0.shape().size - 2) ?: 0
            val a1 = coords1.shape().getOrNull(coords1.shape().size - 2) ?: 0
            detectorCoordsPerAnchor = coords0.shape().lastOrNull() ?: detectorCoordsPerAnchor
            detectorNumAnchors = a0 + a1

            val coordsCombined = FloatArray((a0 + a1) * detectorCoordsPerAnchor)
            if (a0 > 0) System.arraycopy(c0, 0, coordsCombined, 0, min(c0.size, coordsCombined.size))
            if (a1 > 0) System.arraycopy(c1, 0, coordsCombined, a0 * detectorCoordsPerAnchor, min(c1.size, coordsCombined.size - a0 * detectorCoordsPerAnchor))

            val scoresCombinedRaw = FloatArray(a0 + a1)
            if (a0 > 0) System.arraycopy(s0Raw, 0, scoresCombinedRaw, 0, min(s0Raw.size, scoresCombinedRaw.size))
            if (a1 > 0) System.arraycopy(s1Raw, 0, scoresCombinedRaw, a0, min(s1Raw.size, scoresCombinedRaw.size - a0))
            val scoresCombined = FloatArray(scoresCombinedRaw.size) { idx ->
                val v = scoresCombinedRaw[idx].coerceIn(-100f, 100f)
                1f / (1f + exp(-v))
            }
            if (scoresCombinedRaw.isNotEmpty()) {
                val rMin = scoresCombinedRaw.minOrNull() ?: 0f
                val rMax = scoresCombinedRaw.maxOrNull() ?: 0f
                val aMin = scoresCombined.minOrNull() ?: 0f
                val aMax = scoresCombined.maxOrNull() ?: 0f
                Log.d(TAG, "pose detector scores raw min=$rMin max=$rMax activated min=$aMin max=$aMax")
            }
            Log.d("Inference Metrics", "Pose detector inference=${inferenceTime}ms")

            Pair(coordsCombined, scoresCombined)
        } else {
            if (coordsBuf == null || coordsBuf!!.capacity() < coordsTensor.numBytes()) {
                coordsBuf = ByteBuffer.allocateDirect(coordsTensor.numBytes()).order(ByteOrder.nativeOrder())
            }
            if (scoresBuf == null || scoresBuf!!.capacity() < scoresTensor.numBytes()) {
                scoresBuf = ByteBuffer.allocateDirect(scoresTensor.numBytes()).order(ByteOrder.nativeOrder())
            }
            val coordsBuf = coordsBuf!!.apply { clear() }
            val scoresBuf = scoresBuf!!.apply { clear() }
            val outputs: MutableMap<Int, Any> = hashMapOf(0 to coordsBuf, 1 to scoresBuf)
            interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            val inferenceTime = System.currentTimeMillis() - t0
            val coords = dequantizeBuffer(coordsTensor, coordsBuf)
            val rawScores = dequantizeBuffer(scoresTensor, scoresBuf)
            val scores = FloatArray(rawScores.size) { idx ->
                val v = rawScores[idx].coerceIn(-100f, 100f)
                1f / (1f + exp(-v))
            }
            detectorCoordsPerAnchor = coordsTensor.shape().lastOrNull() ?: detectorCoordsPerAnchor
            detectorNumAnchors = coordsTensor.shape().getOrNull(coordsTensor.shape().size - 2) ?: detectorNumAnchors
            if (rawScores.isNotEmpty()) {
                val rMin = rawScores.minOrNull() ?: 0f
                val rMax = rawScores.maxOrNull() ?: 0f
                val aMin = scores.minOrNull() ?: 0f
                val aMax = scores.maxOrNull() ?: 0f
                Log.d(TAG, "pose detector scores raw min=$rMin max=$rMax activated min=$aMin max=$aMax")
            }
            Log.d("Inference Metrics", "Pose detector inference=${inferenceTime}ms")
            Pair(coords, scores)
        }
    }

    private fun runLandmark(
        interpreter: Interpreter,
        input: Any,
        scoreTensor: Tensor,
        lmkTensor: Tensor
    ): Pair<Float, FloatArray> {
        if (lmScoreBuf == null || lmScoreBuf!!.capacity() < scoreTensor.numBytes()) {
            lmScoreBuf = ByteBuffer.allocateDirect(scoreTensor.numBytes()).order(ByteOrder.nativeOrder())
        }
        if (lmLmkBuf == null || lmLmkBuf!!.capacity() < lmkTensor.numBytes()) {
            lmLmkBuf = ByteBuffer.allocateDirect(lmkTensor.numBytes()).order(ByteOrder.nativeOrder())
        }
        val scoreBuf = lmScoreBuf!!.apply { clear() }
        val lmkBuf = lmLmkBuf!!.apply { clear() }
        val outputs: MutableMap<Int, Any> = hashMapOf(0 to scoreBuf, 1 to lmkBuf)
        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
        val scoreArr = dequantizeBuffer(scoreTensor, scoreBuf)
        val lmkArr = dequantizeBuffer(lmkTensor, lmkBuf)

        // Landmarks are flattened; shape is [1, num, dims]. Assume dims >=2.
        val numDims = lmkTensor.shape().lastOrNull() ?: 0
        val numPts = lmkTensor.shape().getOrNull(lmkTensor.shape().size - 2) ?: 0
        val coords = FloatArray(numPts * 2)
        for (i in 0 until numPts) {
            coords[i * 2] = lmkArr[i * numDims] * LANDMARK_INPUT_SIZE
            coords[i * 2 + 1] = lmkArr[i * numDims + 1] * LANDMARK_INPUT_SIZE
        }
        val score = scoreArr.firstOrNull() ?: 0f
        return Pair(score, coords)
    }

    /**
     * Draw pose landmarks on a bitmap (in place) using detected landmarks (pixel coordinates).
     */
    fun drawPose(bitmap: Bitmap, poses: List<PoseLandmarks>): Bitmap {
        val canvas = android.graphics.Canvas(bitmap)
        val linePaint = android.graphics.Paint().apply {
            color = android.graphics.Color.GREEN
            strokeWidth = 6f
            style = android.graphics.Paint.Style.STROKE
            isAntiAlias = true
        }
        val pointPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.CYAN
            strokeWidth = 8f
            style = android.graphics.Paint.Style.FILL
            isAntiAlias = true
        }
        poses.forEach { pose ->
            drawPoseRoi(bitmap, pose.roi, android.graphics.Color.GREEN)
            val pts = pose.landmarks
            POSE_CONNECTIONS.forEach { (a, b) ->
                if (a < pts.size && b < pts.size) {
                    val pa = pts[a]; val pb = pts[b]
                    canvas.drawLine(pa.first, pa.second, pb.first, pb.second, linePaint)
                }
            }
            pts.forEach { p ->
                canvas.drawCircle(p.first, p.second, 6f, pointPaint)
            }
        }
        return bitmap
    }

    /**
     * Draw a pose ROI rotated-rect (tl, bl, tr, br order) on a bitmap.
     */
    fun drawPoseRoi(bitmap: Bitmap, roi: FloatArray?, color: Int = android.graphics.Color.GREEN): Bitmap {
        if (roi == null || roi.size < 8) return bitmap
        // Usage: drawPoseRoi(bitmap, roiCorners, android.graphics.Color.GREEN) where roiCorners
        // comes from computePoseRoiCorners and is ordered TL, BL, TR, BR.
        val canvas = android.graphics.Canvas(bitmap)
        val roiPaint = android.graphics.Paint().apply {
            this.color = color
            strokeWidth = 4f
            style = android.graphics.Paint.Style.STROKE
            isAntiAlias = true
        }
        val pts = floatArrayOf(
            roi[0], roi[1], roi[2], roi[3],
            roi[0], roi[1], roi[4], roi[5],
            roi[2], roi[3], roi[6], roi[7],
            roi[4], roi[5], roi[6], roi[7]
        )
        canvas.drawLines(pts, roiPaint)
        return bitmap
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

    // Reuse warp into a preallocated bitmap to reduce allocations.
    private fun reuseWarp(src: Bitmap, matrix: Matrix, dstW: Int, dstH: Int): Bitmap {
        val out = if (cropBitmap == null || cropBitmap?.width != dstW || cropBitmap?.height != dstH) {
            Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888).also { cropBitmap = it }
        } else {
            cropBitmap!!
        }
        warpMatrix.reset()
        warpMatrix.set(matrix)
        val canvas = Canvas(out)
        canvas.drawBitmap(src, warpMatrix, null)
        return out
    }
}
