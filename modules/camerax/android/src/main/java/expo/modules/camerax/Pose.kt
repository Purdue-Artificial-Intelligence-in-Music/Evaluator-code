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
 * LiteRT/Qualcomm-backed pose pipeline placeholder.
 * Will mirror the MediaPipe pose flow using the mediapipe_pose detector/landmark assets.
 */
class Pose(private val context: Context) {
    companion object {
        private const val TAG = "PoseLiteRt"
        private const val DETECTOR_MODEL = "mediapipe_pose-posedetector-w8a8.tflite"
        private const val LANDMARK_MODEL = "mediapipe_pose-poselandmarkdetector-w8a8.tflite"
        // Detector input dims from model card (3x128x128)
        private const val DETECTOR_INPUT_SIZE = 128
        private const val LANDMARK_INPUT_SIZE = 256
        private const val DET_SCORE_THRESHOLD = 0.6f
        private const val NMS_IOU = 0.3f
        private const val ROI_SCALE = 1.5f
        private const val DEFAULT_POSE_ANCHORS = "anchors_pose.npy"
        private const val SMOOTH_ALPHA = 0.5f

        private enum class DelegateChoice { HTP, GPU, CPU }

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
    private var qnnDelegate: QnnDelegate? = null
    private var gpuDelegate: GpuDelegate? = null
    private var detectorDelegateChoice: DelegateChoice = DelegateChoice.HTP
    private var landmarkDelegateChoice: DelegateChoice = DelegateChoice.HTP
    private var detectorInputType: DataType = DataType.FLOAT32
    private var detectorCoordsPerAnchor = 0
    private var detectorNumAnchors = 0
    private var anchors: FloatArray? = null
    private var track: PoseTrack? = null
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
    private var tensorImageUint8: TensorImage? = null
    private var tensorImageInt8: TensorImage? = null
    private var pixelBuffer: IntArray? = null
    private var nchwFloatBuffer: ByteBuffer? = null
    private var nchwUint8Buffer: ByteBuffer? = null
    private var nchwInt8Buffer: ByteBuffer? = null
    private var imageProcessorFloat: ImageProcessor? = null
    private var detectorPadBitmap: Bitmap? = null
    private var detectorPadCanvas: Canvas? = null
    private val detectorPadMatrix = Matrix()

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
        try { qnnDelegate?.close() } catch (_: Throwable) {}
        try { gpuDelegate?.close() } catch (_: Throwable) {}
    }

    /**
     * Run detector (steps 1-3): preprocess -> infer -> decode anchors -> NMS.
     */
    fun detectPoses(bitmap: Bitmap): List<MediapipeLiteRtUtil.Detection> {
        val det = detector ?: return emptyList()
        val prep = resizePadReuse(bitmap, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE)
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
        val limited = nms.sortedByDescending { it.score }.take(1)
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
                x0 = clamp01(x0 / prep.originalWidth),
                y0 = clamp01(y0 / prep.originalHeight),
                x1 = clamp01(x1 / prep.originalWidth),
                y1 = clamp01(y1 / prep.originalHeight),
                keypoints = mappedKp
            )
        }
    }

    data class PoseLandmarks(
        val score: Float,
        val landmarks: List<Pair<Float, Float>>
    )

    private data class PoseTrack(
        var roi: FloatArray,
        var lastLandmarks: FloatArray?,
        var lastUpdate: Long
    )

    /**
     * Full pose path: detect -> ROI -> landmark -> map back.
     */
    fun detectAndLandmark(bitmap: Bitmap): List<PoseLandmarks> {
        Log.d(TAG, "pose detectAndLandmark frame=${bitmap.width}x${bitmap.height}")
        val lmInterp = landmark ?: return emptyList()
        val results = mutableListOf<PoseLandmarks>()
        val nowMs = System.currentTimeMillis()

        // Step 1: try existing track.
        track?.let { tr ->
            if (nowMs - tr.lastUpdate < 800) {
                val (forward, inverse) = MediapipeLiteRtUtil.buildAffineFromRoi(
                    tr.roi,
                    LANDMARK_INPUT_SIZE,
                    LANDMARK_INPUT_SIZE
                )
            val crop = reuseWarp(bitmap, forward, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE)
            val input = buildInputBuffer(crop, lmInterp.getInputTensor(0))
                val scoreTensor = lmInterp.getOutputTensor(0)
                val lmkTensor = lmInterp.getOutputTensor(1)
                val out = runLandmark(lmInterp, input, scoreTensor, lmkTensor)
                if (out.first >= DET_SCORE_THRESHOLD && out.second.isNotEmpty()) {
                    val mapped = FloatArray(out.second.size)
                    inverse?.mapPoints(mapped, out.second)
                    tr.lastLandmarks?.let { prev ->
                        if (prev.size == mapped.size) {
                            for (i in mapped.indices) {
                                mapped[i] = SMOOTH_ALPHA * prev[i] + (1f - SMOOTH_ALPHA) * mapped[i]
                            }
                        }
                    }
                    tr.lastLandmarks = mapped.copyOf()
                    tr.lastUpdate = nowMs
                    val pairs = mutableListOf<Pair<Float, Float>>()
                    var i = 0
                    while (i < mapped.size) {
                        pairs.add(Pair(mapped[i], mapped[i + 1]))
                        i += 2
                    }
                    Log.d("classifcation", "pose score=${out.first} points=${pairs.size} (track)")
                    results.add(PoseLandmarks(out.first, pairs))
                }
            }
        }

        // Step 2: detector to refresh/create track if needed.
        val detections = detectPoses(bitmap)
        if (detections.isNotEmpty()) {
            val det = detections.first()
            val roiCorners = MediapipeLiteRtUtil.computePoseRoiCorners(
                det,
                bitmap.width,
                bitmap.height,
                boxScale = ROI_SCALE
            )
            if (roiCorners != null) {
                val matched = track?.let { if (poseIou(it.roi, roiCorners) > 0.3f) it else null }
                val tr = matched ?: PoseTrack(roiCorners, null, nowMs).also { track = it }
                tr.roi = roiCorners
                val (forward, inverse) = MediapipeLiteRtUtil.buildAffineFromRoi(
                    roiCorners,
                    LANDMARK_INPUT_SIZE,
                    LANDMARK_INPUT_SIZE
                )
                val crop = reuseWarp(bitmap, forward, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE)
                val input = buildInputBuffer(crop, lmInterp.getInputTensor(0))
                val scoreTensor = lmInterp.getOutputTensor(0)
                val lmkTensor = lmInterp.getOutputTensor(1)
                val out = runLandmark(lmInterp, input, scoreTensor, lmkTensor)
                if (out.first >= DET_SCORE_THRESHOLD && out.second.isNotEmpty()) {
                    val mapped = FloatArray(out.second.size)
                    inverse?.mapPoints(mapped, out.second)
                    tr.lastLandmarks?.let { prev ->
                        if (prev.size == mapped.size) {
                            for (i in mapped.indices) {
                                mapped[i] = SMOOTH_ALPHA * prev[i] + (1f - SMOOTH_ALPHA) * mapped[i]
                            }
                        }
                    }
                    tr.lastLandmarks = mapped.copyOf()
                    tr.lastUpdate = nowMs
                    val pairs = mutableListOf<Pair<Float, Float>>()
                    var i = 0
                    while (i < mapped.size) {
                        pairs.add(Pair(mapped[i], mapped[i + 1]))
                        i += 2
                    }
                    Log.d("classifcation", "pose score=${out.first} points=${pairs.size} (detector)")
                    if (results.isEmpty()) {
                        results.add(PoseLandmarks(out.first, pairs))
                    } else {
                        results[0] = PoseLandmarks(out.first, pairs)
                    }
                }
            }
        }
        track?.let { if (nowMs - it.lastUpdate > 1500) track = null }
        Log.d(TAG, "pose landmarks count=${results.size}")
        return results
    }

    private fun clamp01(v: Float): Float = max(0f, min(1f, v))

    private fun setupDetector() {
        val model = FileUtil.loadMappedFile(context, DETECTOR_MODEL)
        val opts = buildInterpreterOptions(detectorDelegateChoice, "pose detector")
        detector = Interpreter(model, opts)
        detectorInputType = detector?.getInputTensor(0)?.dataType() ?: DataType.FLOAT32
    }

    private fun setupLandmark() {
        val opts = buildInterpreterOptions(landmarkDelegateChoice, "pose landmark")
        val model = FileUtil.loadMappedFile(context, LANDMARK_MODEL)
        landmark = Interpreter(model, opts)
    }

    private fun buildInterpreterOptions(choice: DelegateChoice, name: String): Interpreter.Options {
        val opts = Interpreter.Options()
        var selected: String? = null
        when (choice) {
            DelegateChoice.HTP -> {
                val skelDir = tryLoadQnnAndPickSkelDir()
                if (skelDir != null) {
                    try {
                        val qOpts = QnnDelegate.Options().apply {
                            setBackendType(BackendType.HTP_BACKEND)
                            setSkelLibraryDir(skelDir)
                        }
                        qnnDelegate = QnnDelegate(qOpts)
                        opts.addDelegate(qnnDelegate)
                        selected = "HTP"
                    } catch (t: Throwable) {
                        Log.w("CheckDel", "$name: QNN delegate unavailable: ${t.message}")
                    }
                }
                if (selected == null) {
                    try {
                        val cl = CompatibilityList()
                        if (cl.isDelegateSupportedOnThisDevice) {
                            gpuDelegate = GpuDelegate(cl.bestOptionsForThisDevice)
                            opts.addDelegate(gpuDelegate)
                            selected = "GPU"
                        }
                    } catch (t: Throwable) {
                        Log.w("CheckDel", "$name: GPU delegate unavailable: ${t.message}")
                    }
                }
            }
            DelegateChoice.GPU -> {
                try {
                    val cl = CompatibilityList()
                    if (cl.isDelegateSupportedOnThisDevice) {
                        gpuDelegate = GpuDelegate(cl.bestOptionsForThisDevice)
                        opts.addDelegate(gpuDelegate)
                        selected = "GPU"
                    }
                } catch (t: Throwable) {
                    Log.w("CheckDel", "$name: GPU delegate unavailable: ${t.message}")
                }
            }
            DelegateChoice.CPU -> {
                // fall through
            }
        }
        if (selected == null) {
            try { opts.setUseXNNPACK(true) } catch (_: Throwable) {}
            opts.setNumThreads(4)
            selected = "CPU"
        }
        Log.i("CheckDel", String.format(Locale.US, "%s delegate=%s", name, selected))
        return opts
    }

    private fun tryLoadQnnAndPickSkelDir(): String? {
        val mustLoad = listOf("QnnSystem", "QnnHtp", "QnnHtpPrepare")
        for (name in mustLoad) {
            try { System.loadLibrary(name) }
            catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "QNN: failed to load $name: ${e.message}. nativeLibDir=${context.applicationInfo.nativeLibraryDir}")
                return null
            }
        }
        val base = context.applicationInfo.nativeLibraryDir
        val skels = listOf(
            "libQnnHtpV79Skel.so",
            "libQnnHtpV75Skel.so",
            "libQnnHtpV73Skel.so",
            "libQnnHtpV69Skel.so"
        )
        val chosen = skels.firstOrNull { File("$base/$it").exists() }
        if (chosen == null) {
            Log.w(TAG, "QNN: no HTP skel found under $base")
            return null
        }
        Log.i(TAG, "QNN: using skel=$chosen in $base")
        return base
    }

    // Letterbox resize using a reusable padded bitmap/canvas to avoid per-frame allocations.
    private fun resizePadReuse(
        src: Bitmap,
        dstWidth: Int,
        dstHeight: Int
    ): MediapipeLiteRtUtil.ResizeResult {
        val srcW = src.width
        val srcH = src.height
        val scale = min(dstWidth.toFloat() / srcW, dstHeight.toFloat() / srcH)
        val scaledW = (srcW * scale).toInt()
        val scaledH = (srcH * scale).toInt()
        val padX = (dstWidth - scaledW) / 2
        val padY = (dstHeight - scaledH) / 2

        val padded = detectorPadBitmap
            ?.takeIf { it.width == dstWidth && it.height == dstHeight }
            ?: Bitmap.createBitmap(dstWidth, dstHeight, Bitmap.Config.ARGB_8888).also {
                detectorPadBitmap = it
                detectorPadCanvas = Canvas(it)
            }
        val canvas = detectorPadCanvas ?: Canvas(padded).also { detectorPadCanvas = it }
        detectorPadMatrix.reset()
        detectorPadMatrix.setScale(scale, scale)
        detectorPadMatrix.postTranslate(padX.toFloat(), padY.toFloat())
        canvas.drawColor(0x00000000)
        canvas.drawBitmap(src, detectorPadMatrix, null)

        return MediapipeLiteRtUtil.ResizeResult(
            bitmap = padded,
            scale = scale,
            padX = padX,
            padY = padY,
            targetWidth = dstWidth,
            targetHeight = dstHeight,
            originalWidth = srcW,
            originalHeight = srcH
        )
    }

    private fun buildInputBuffer(bitmap: Bitmap, inputTensor: Tensor): Any {
        val shape = inputTensor.shape()
        val isNhwc = shape.size == 4 && shape[3] == 3
        val isNchw = shape.size == 4 && shape[1] == 3
        val numBytes = inputTensor.numBytes()

        return when (inputTensor.dataType()) {
            DataType.FLOAT32 -> {
                if (isNhwc) {
                    val imageProcessor = imageProcessorFloat ?: ImageProcessor.Builder()
                        .add(NormalizeOp(0f, 255f))
                        .build().also { imageProcessorFloat = it }
                    val tensorImage = detectorTensorImage ?: TensorImage(DataType.FLOAT32).also { detectorTensorImage = it }
                    tensorImage.load(bitmap)
                    val processed = imageProcessor.process(tensorImage)
                    processed.buffer
                } else {
                    val buffer = nchwFloatBuffer?.takeIf { it.capacity() >= numBytes } ?: ByteBuffer
                        .allocateDirect(numBytes)
                        .order(ByteOrder.nativeOrder())
                        .also { nchwFloatBuffer = it }
                    buffer.clear()
                    val pixels = pixelBuffer?.takeIf { it.size >= bitmap.width * bitmap.height }
                        ?: IntArray(bitmap.width * bitmap.height).also { pixelBuffer = it }
                    bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                    for (c in 0 until 3) {
                        var idx = 0
                        while (idx < pixels.size) {
                            val p = pixels[idx]
                            val v = when (c) {
                                0 -> (p shr 16) and 0xFF
                                1 -> (p shr 8) and 0xFF
                                else -> p and 0xFF
                            }
                            buffer.putFloat(v / 255f)
                            idx++
                        }
                    }
                    buffer.rewind()
                    buffer
                }
            }

            DataType.UINT8 -> {
                if (isNhwc) {
                    val tensorImage = tensorImageUint8 ?: TensorImage(DataType.UINT8).also { tensorImageUint8 = it }
                    tensorImage.load(bitmap)
                    tensorImage.buffer
                } else {
                    val buffer = nchwUint8Buffer?.takeIf { it.capacity() >= numBytes } ?: ByteBuffer
                        .allocateDirect(numBytes)
                        .order(ByteOrder.nativeOrder())
                        .also { nchwUint8Buffer = it }
                    buffer.clear()
                    val pixels = pixelBuffer?.takeIf { it.size >= bitmap.width * bitmap.height }
                        ?: IntArray(bitmap.width * bitmap.height).also { pixelBuffer = it }
                    bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                    for (c in 0 until 3) {
                        var idx = 0
                        while (idx < pixels.size) {
                            val p = pixels[idx]
                            val v = when (c) {
                                0 -> (p shr 16) and 0xFF
                                1 -> (p shr 8) and 0xFF
                                else -> p and 0xFF
                            }
                            buffer.put(v.toByte())
                            idx++
                        }
                    }
                    buffer.rewind()
                    buffer
                }
            }

            DataType.INT8 -> {
                val quant = inputTensor.quantizationParams()
                val scale = quant.scale
                val zeroPoint = quant.zeroPoint
                val buffer = nchwInt8Buffer?.takeIf { it.capacity() >= numBytes } ?: ByteBuffer
                    .allocateDirect(numBytes)
                    .order(ByteOrder.nativeOrder())
                    .also { nchwInt8Buffer = it }
                buffer.clear()
                val pixels = pixelBuffer?.takeIf { it.size >= bitmap.width * bitmap.height }
                    ?: IntArray(bitmap.width * bitmap.height).also { pixelBuffer = it }
                bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

                val pushQuant: (Int) -> Unit = { p ->
                    val real = p / 255f
                    val q = (real / scale + zeroPoint).toInt().coerceIn(-128, 127)
                    buffer.put(q.toByte())
                }

                if (isNhwc) {
                    var idx = 0
                    while (idx < pixels.size) {
                        val p = pixels[idx]
                        pushQuant((p shr 16) and 0xFF)
                        pushQuant((p shr 8) and 0xFF)
                        pushQuant(p and 0xFF)
                        idx++
                    }
                } else {
                    for (c in 0 until 3) {
                        var idx = 0
                        while (idx < pixels.size) {
                            val p = pixels[idx]
                            val v = when (c) {
                                0 -> (p shr 16) and 0xFF
                                1 -> (p shr 8) and 0xFF
                                else -> p and 0xFF
                            }
                            pushQuant(v)
                            idx++
                        }
                    }
                }

                buffer.rewind()
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

    private fun poseIou(roiA: FloatArray, roiB: FloatArray): Float {
        if (roiA.size < 8 || roiB.size < 8) return 0f
        val aMinX = min(min(roiA[0], roiA[2]), min(roiA[4], roiA[6]))
        val aMaxX = max(max(roiA[0], roiA[2]), max(roiA[4], roiA[6]))
        val aMinY = min(min(roiA[1], roiA[3]), min(roiA[5], roiA[7]))
        val aMaxY = max(max(roiA[1], roiA[3]), max(roiA[5], roiA[7]))
        val bMinX = min(min(roiB[0], roiB[2]), min(roiB[4], roiB[6]))
        val bMaxX = max(max(roiB[0], roiB[2]), max(roiB[4], roiB[6]))
        val bMinY = min(min(roiB[1], roiB[3]), min(roiB[5], roiB[7]))
        val bMaxY = max(max(roiB[1], roiB[3]), max(roiB[5], roiB[7]))
        val interX0 = max(aMinX, bMinX)
        val interY0 = max(aMinY, bMinY)
        val interX1 = min(aMaxX, bMaxX)
        val interY1 = min(aMaxY, bMaxY)
        val interW = max(0f, interX1 - interX0)
        val interH = max(0f, interY1 - interY0)
        val inter = interW * interH
        val areaA = (aMaxX - aMinX) * (aMaxY - aMinY)
        val areaB = (bMaxX - bMinX) * (bMaxY - bMinY)
        val union = areaA + areaB - inter
        return if (union > 0f) inter / union else 0f
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
