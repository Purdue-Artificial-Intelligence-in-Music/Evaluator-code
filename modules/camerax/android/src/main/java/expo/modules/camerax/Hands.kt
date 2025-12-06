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
        // Detection/landmark score gates (detector lower to tolerate blur; landmark higher to filter noise)
        private const val DET_SCORE_THRESHOLD_DET = 0.4f
        private const val DET_SCORE_THRESHOLD_LMK = 0.6f
        // NMS IoU for palm detections (lower suppresses more overlaps; higher allows close hands)
        private const val NMS_IOU = 0.1f
        // ROI shaping: shift along wrist->middle vector (negative pulls toward fingertips), expand box
        private const val ROI_DXY = -0.2f
        private const val ROI_DSCALE = 2.5f
        // IoU match threshold when assigning detections to existing tracks
        private const val MATCH_IOU_THRESHOLD = 0.2f
        // Landmark smoothing (EMA) within a track
        private const val SMOOTH_ALPHA = 0.5f
        // ROI smoothing (EMA) when matching detections to existing tracks
        private const val ROI_SMOOTH_ALPHA = 0f
        // Tracking/backoff: consider a track "fresh" for this many ms; skip detector if fresh
        private const val TRACK_FRESH_MS = 0L
        // Drop tracks after this age (unless refreshed)
        private const val TRACK_MAX_AGE_MS = 1200L
        // Run detector every N frames when not forced (set to 1 to run every frame)
        private const val DETECTOR_PERIOD = 1
        private const val ROT_OFFSET = (Math.PI.toFloat() / 2f)
        private const val DEFAULT_HAND_ANCHORS = "anchors_palm.npy"
        // keypoints for rotation: wrist center idx 0, middle finger base idx 2 (from model.py)
        private const val KP_ROT_START = 0
        private const val KP_ROT_END = 2
        // 21 landmarks expected
        private const val NUM_LANDMARKS = 21

        private enum class DelegateChoice { HTP, GPU, CPU }
    }

    private var detector: Interpreter? = null
    private var landmark: Interpreter? = null
    private var qnnDelegate: QnnDelegate? = null
    private var gpuDelegate: GpuDelegate? = null
    private var detectorDelegateChoice: DelegateChoice = DelegateChoice.HTP
    private var landmarkDelegateChoice: DelegateChoice = DelegateChoice.HTP
    private var detectorCoordsPerAnchor = 0
    private var detectorNumAnchors = 0
    private var anchors: FloatArray? = null
    private var lastPoseLandmarks: List<Pair<Float, Float>>? = null
    var usePoseHandedness: Boolean = true
    var singleBodyMode: Boolean = true
    var bowHandIsRight: Boolean = true
    private var handClassifier: Interpreter? = null
    private val tracks = mutableListOf<HandTrack>()
    private var nextTrackId = 0
    private var frameCount = 0
    // Reusable buffers to cut allocations
    private var detectorTensorImage: TensorImage? = null
    private var detectorCoordsBuf: ByteBuffer? = null
    private var detectorScoresBuf: ByteBuffer? = null
    private var lmScoreBuf: ByteBuffer? = null
    private var lmLrBuf: ByteBuffer? = null
    private var lmLmkBuf: ByteBuffer? = null
    private var cropBitmaps = arrayOfNulls<Bitmap>(2)
    private val warpMatrix = Matrix()
    private var detectorPadBitmap: Bitmap? = null
    private var detectorPadCanvas: Canvas? = null
    private val detectorPadMatrix = Matrix()
    private var tensorImageUint8: TensorImage? = null
    private var tensorImageInt8: TensorImage? = null
    private var pixelBuffer: IntArray? = null
    private var nchwFloatBuffer: ByteBuffer? = null
    private var nchwUint8Buffer: ByteBuffer? = null
    private var nchwInt8Buffer: ByteBuffer? = null
    private var imageProcessorFloat: ImageProcessor? = null

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
        try { qnnDelegate?.close() } catch (_: Throwable) {}
        try { gpuDelegate?.close() } catch (_: Throwable) {}
    }

    data class HandResult(
        val score: Float,
        val handedness: Boolean, // true = right, false = left
        val landmarks: List<Pair<Float, Float>>,
        val roi: FloatArray? = null
    )

    private data class HandTrack(
        val id: Int,
        var roi: FloatArray,
        var lastLandmarks: FloatArray?,
        var lastUpdate: Long,
        var lastScore: Float,
        var lowStreak: Int = 0
    )

    fun detectAndLandmark(
        bitmap: Bitmap,
        poseHints: List<List<Pair<Float, Float>>>? = null
    ): List<HandResult> {
        Log.d(TAG, "hand detectAndLandmark frame=${bitmap.width}x${bitmap.height}")
        if (poseHints != null && poseHints.isNotEmpty()) {
            lastPoseLandmarks = poseHints.firstOrNull()
        }
        frameCount++
        val det = detector ?: return emptyList()
        val prep = resizePadReuse(bitmap, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE)
        val input = buildInputBuffer(prep.bitmap, det.getInputTensor(0))

        val coordsTensor = det.getOutputTensor(0)
        val scoresTensor = det.getOutputTensor(1)
        val (coords, scores) = runDetector(det, input, coordsTensor, scoresTensor)

        val lmInterp = landmark ?: return emptyList()
        val results = mutableListOf<HandResult>()
        val nowMs = System.currentTimeMillis()

        // Step 1: reuse active tracks (landmark-only pass).
        val activeTracks = tracks.filter { nowMs - it.lastUpdate < TRACK_MAX_AGE_MS }
        val usedIds = mutableSetOf<Int>()
        activeTracks.forEach { track ->
            val (forward, inverse) = MediapipeLiteRtUtil.buildAffineFromRoi(
                track.roi,
                LANDMARK_INPUT_SIZE,
                LANDMARK_INPUT_SIZE
            )
            val crop = reuseWarp(bitmap, forward, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE, 0)
            val lmInput = buildInputBuffer(crop, lmInterp.getInputTensor(0))
            val out = runLandmark(lmInterp, lmInput, lmInterp.getOutputTensor(0), lmInterp.getOutputTensor(1), lmInterp.getOutputTensor(2))
            val score = out.first
            val lmk = out.third
            if (score < DET_SCORE_THRESHOLD_LMK || lmk.isEmpty()) {
                track.lowStreak += 1
                track.lastScore = score
                track.lastUpdate = nowMs
                return@forEach
            }
            val mapped = FloatArray(lmk.size)
            inverse?.mapPoints(mapped, lmk)
            track.lastLandmarks?.let { prev ->
                if (prev.size == mapped.size) {
                    for (i in mapped.indices) {
                        mapped[i] = SMOOTH_ALPHA * prev[i] + (1f - SMOOTH_ALPHA) * mapped[i]
                    }
                }
            }
            track.lastLandmarks = mapped.copyOf()
            track.lastUpdate = nowMs
            track.lastScore = score
            track.lowStreak = 0
            val pairs = mutableListOf<Pair<Float, Float>>()
            var i = 0
            while (i < mapped.size) {
                pairs.add(Pair(mapped[i], mapped[i + 1]))
                i += 2
            }
            val handed = out.second
            Log.d("classifcation", "hand track=${track.id} side=${if (handed) "right" else "left"} score=$score (track reuse)")
            if (handed == bowHandIsRight) {
                runHandClassifier(pairs)
            }
            results.add(HandResult(score, handed, pairs, track.roi))
            usedIds.add(track.id)
        }

        // Step 2: detector pass to refresh/spawn tracks.
        val hasFreshTrack = activeTracks.any { nowMs - it.lastUpdate < TRACK_FRESH_MS && it.lastScore >= DET_SCORE_THRESHOLD_LMK }
        val shouldRunDetector = !hasFreshTrack || (frameCount % DETECTOR_PERIOD == 0)
        if (!shouldRunDetector) {
            tracks.removeAll { nowMs - it.lastUpdate > TRACK_MAX_AGE_MS }
            Log.d(TAG, "hand landmarks count=${results.size}")
            return results
        }
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
                x0 = (x0 / prep.originalWidth).coerceIn(0f, 1f),
                y0 = (y0 / prep.originalHeight).coerceIn(0f, 1f),
                x1 = (x1 / prep.originalWidth).coerceIn(0f, 1f),
                y1 = (y1 / prep.originalHeight).coerceIn(0f, 1f),
                keypoints = mappedKp
            )
            // Build ROI corners from keypoints and box.
            val roi = computeHandRoi(detObj, bitmap.width, bitmap.height) ?: continue
            val matched = tracks.maxByOrNull { iou(it.roi, roi) }
            val track = if (matched != null && iou(matched.roi, roi) > MATCH_IOU_THRESHOLD) {
                matched.roi = smoothRoi(matched.roi, roi)
                matched
            } else {
                if (tracks.size >= 2) continue
                HandTrack(nextTrackId++, roi, null, nowMs, 0f).also { tracks.add(it) }
            }
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
                track.lowStreak += 1
                track.lastScore = score
                track.lastUpdate = nowMs
                continue
            }
            val mapped = FloatArray(lmk.size)
            inverse?.mapPoints(mapped, lmk)
            track.lastLandmarks?.let { prev ->
                if (prev.size == mapped.size) {
                    for (i in mapped.indices) {
                        mapped[i] = SMOOTH_ALPHA * prev[i] + (1f - SMOOTH_ALPHA) * mapped[i]
                    }
                }
            }
            track.lastLandmarks = mapped.copyOf()
            track.lastUpdate = nowMs
            track.lastScore = score
            track.lowStreak = 0
            val pairs = mutableListOf<Pair<Float, Float>>()
            var i = 0
            while (i < mapped.size) {
                pairs.add(Pair(mapped[i], mapped[i + 1]))
                i += 2
            }
            val handed = out.second
            Log.d("classifcation", "hand track=${track.id} side=${if (handed) "right" else "left"} score=$score (detector)")
            if (handed == bowHandIsRight) {
                runHandClassifier(pairs)
            }
            if (!usedIds.contains(track.id)) {
                results.add(HandResult(score, handed, pairs, roi))
                usedIds.add(track.id)
            }
        }
        tracks.removeAll { nowMs - it.lastUpdate > TRACK_MAX_AGE_MS || it.lowStreak >= 2 }
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

    private fun setupDetector() {
        val opts = buildInterpreterOptions(detectorDelegateChoice, "hand detector")
        val model = FileUtil.loadMappedFile(context, DETECTOR_MODEL)
        detector = Interpreter(model, opts)
        val out0 = detector?.getOutputTensor(0)
        detectorCoordsPerAnchor = out0?.shape()?.lastOrNull() ?: 0
        detectorNumAnchors = out0?.shape()?.getOrNull(out0.shape().size - 2) ?: 0
    }

    private fun setupLandmark() {
        val opts = buildInterpreterOptions(landmarkDelegateChoice, "hand landmark")
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

    // Use pose wrist/index hints to assign side; fallback to model if pose unavailable.
    private fun assignHandByPose(handLandmarks: List<Pair<Float, Float>>, fallback: Boolean): Boolean {
        val pose = lastPoseLandmarks
        if (!usePoseHandedness || pose == null || pose.isEmpty()) return fallback
        val leftIdx = listOf(15, 19)
        val rightIdx = listOf(16, 20)
        val left = meanPair(pose, leftIdx)
        val right = meanPair(pose, rightIdx)
        if (left == null && right == null) return fallback
        val wrist = handLandmarks.getOrNull(0) ?: return fallback
        val distL = left?.let { d2(wrist, it) }
        val distR = right?.let { d2(wrist, it) }
        return when {
            distL != null && distR != null -> distR <= distL
            distR != null -> true
            distL != null -> false
            else -> fallback
        }
    }

    private fun meanPair(pts: List<Pair<Float, Float>>, idxs: List<Int>): Pair<Float, Float>? {
        val valid = idxs.mapNotNull { pts.getOrNull(it) }
        if (valid.isEmpty()) return null
        val sx = valid.sumOf { it.first.toDouble() }.toFloat()
        val sy = valid.sumOf { it.second.toDouble() }.toFloat()
        val n = valid.size.toFloat()
        return Pair(sx / n, sy / n)
    }

    private fun d2(a: Pair<Float, Float>, b: Pair<Float, Float>): Float {
        val dx = a.first - b.first
        val dy = a.second - b.second
        return dx * dx + dy * dy
    }

    private fun iou(roiA: FloatArray, roiB: FloatArray): Float {
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

    private fun smoothRoi(prev: FloatArray, cur: FloatArray): FloatArray {
        if (prev.size != cur.size) return cur
        val out = FloatArray(prev.size)
        for (i in prev.indices) {
            out[i] = ROI_SMOOTH_ALPHA * prev[i] + (1f - ROI_SMOOTH_ALPHA) * cur[i]
        }
        return out
    }

    private fun runHandClassifier(landmarks: List<Pair<Float, Float>>) {
        try {
            if (handClassifier == null) {
                val model = FileUtil.loadMappedFile(context, "keypoint_classifier_FINAL.tflite")
                handClassifier = Interpreter(model)
            }
            if (landmarks.size < 21) return
            val xs = landmarks.map { it.first }
            val ys = landmarks.map { it.second }
            val minX = xs.minOrNull() ?: return
            val maxX = xs.maxOrNull() ?: return
            val minY = ys.minOrNull() ?: return
            val maxY = ys.maxOrNull() ?: return
            val scaleX = if (maxX > minX) maxX - minX else 1f
            val scaleY = if (maxY > minY) maxY - minY else 1f
            val input = FloatArray(42)
            var i = 0
            landmarks.forEach { (x, y) ->
                input[i++] = (x - minX) / scaleX
                input[i++] = (y - minY) / scaleY
            }
            val output = Array(1) { FloatArray(4) }
            val t0 = android.os.SystemClock.uptimeMillis()
            handClassifier?.run(arrayOf(input), output)
            val t1 = android.os.SystemClock.uptimeMillis()
            val scores = output[0]
            val idx = scores.indices.maxByOrNull { scores[it] } ?: -1
            val conf = if (idx >= 0) scores[idx] else 0f
            Log.d("classifcation", "bow hand class=$idx conf=$conf time=${t1 - t0}ms")
        } catch (t: Throwable) {
            Log.w("classifcation", "hand classifier failed: ${t.message}")
        }
    }
}
