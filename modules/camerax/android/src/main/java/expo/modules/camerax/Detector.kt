package expo.modules.camerax

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import com.qualcomm.qti.QnnDelegate
import com.qualcomm.qti.QnnDelegate.Options.BackendType
import java.io.File
import java.util.concurrent.CountDownLatch
import kotlin.math.*

class Detector(
    private val context: Context,
    private val listener: DetectorListener? = null
) {

    // ==== TFLite + delegates ====
    private var interpreter: Interpreter
    private var qnnDelegate: QnnDelegate? = null
    private var tfliteGpu: GpuDelegate? = null

    // ==== Tensor shapes ====
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    val modelReadyLatch = CountDownLatch(1)

    // ==== State for classification ====
    private var bowRepeat = 0
    private var stringRepeat = 0
    private var bowPoints: List<Point>? = null
    private var stringPoints: List<Point>? = null
    private var yLocked = false
    private var yAvg: MutableList<Double>? = null
    private var frameCounter = 0
    private var stringYCoordHeights: MutableList<List<Int>> = mutableListOf()
    private val numWaitFrames = 5

    private fun Double.f1() = String.format("%.1f", this)
    private fun Double.f3() = String.format("%.3f", this)
    private fun Float.f2() = String.format("%.2f", this)

    private fun fmtQuad(name: String, q: List<Point>?): String {
        return if (q == null || q.size < 4) {
            "$name: null"
        } else {
            val s = q.joinToString(",") { "(${it.x.f1()}, ${it.y.f1()})" }
            "$name: [$s]"
        }
    }
    // Preprocessing
    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STD))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    // ==== QNN helpers ====
    private fun nativeLibDir(): String = context.applicationInfo.nativeLibraryDir

    /** Load QNN host libs and return a dir that contains one of the HTP skel .so files. */
    private fun tryLoadQnnAndPickSkelDir(): String? {
        val mustLoad = listOf("QnnSystem", "QnnHtp", "QnnHtpPrepare")
        for (name in mustLoad) {
            try { System.loadLibrary(name) }
            catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "QNN: failed to load $name: ${e.message}. nativeLibDir=${nativeLibDir()}")
                return null
            }
        }
        val base = nativeLibDir()
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

    companion object {
        private const val TAG = "CheckDel"
        private const val MODEL_ASSET = "nanoV2.tflite" // <-- set your model file name here
        private const val INPUT_MEAN = 0f
        private const val INPUT_STD = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.10f
    }

    init {
        interpreter = createInterpreterWithFallbacks(context)

        // Cache tensor shapes (NHWC or NCHW)
        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape = interpreter.getOutputTensor(0)?.shape()

        if (inputShape != null && inputShape.size >= 4) {
            if (inputShape[1] == 3) {           // NCHW: [1,3,H,W]
                tensorWidth = inputShape[3]
                tensorHeight = inputShape[2]
            } else {                            // NHWC: [1,H,W,3]
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[1]
            }
        }

        if (outputShape != null && outputShape.size == 3) {
            numChannel = outputShape[1]
            numElements = outputShape[2]
        }

        modelReadyLatch.countDown()
    }

    private fun createInterpreterWithFallbacks(context: Context): Interpreter {
        val model = FileUtil.loadMappedFile(context, MODEL_ASSET)
        val options = Interpreter.Options()

        // 1) Qualcomm NPU (QNN/HTP)
        val skelDir = tryLoadQnnAndPickSkelDir()
        if (skelDir != null) {
            try {
                val qOpts = QnnDelegate.Options().apply {
                    setBackendType(BackendType.HTP_BACKEND)
                    setSkelLibraryDir(skelDir)
                }
                qnnDelegate = QnnDelegate(qOpts)
                options.addDelegate(qnnDelegate)
                Log.i(TAG, "Using Qualcomm QNN delegate (HTP/NPU)")
                return Interpreter(model, options)
            } catch (t: Throwable) {
                Log.w(TAG, "QNN delegate unavailable: ${t.message}")
            }
        }

        // 2) GPU
        try {
            val cl = CompatibilityList()
            if (cl.isDelegateSupportedOnThisDevice) {
                tfliteGpu = GpuDelegate(cl.bestOptionsForThisDevice)
                options.addDelegate(tfliteGpu)
                Log.i(TAG, "Using TFLite GPU delegate")
                return Interpreter(model, options)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "GPU delegate unavailable: ${t.message}")
        }

        // 3) CPU/XNNPACK
        Log.i(TAG, "Falling back to CPU/XNNPACK")
        try { options.setUseXNNPACK(true) } catch (_: Throwable) {}
        options.setNumThreads(4)
        return Interpreter(model, options)
    }

    fun close() {
        try { interpreter.close() } catch (_: Throwable) {}
        try { tfliteGpu?.close() } catch (_: Throwable) {}
        try { qnnDelegate?.close() } catch (_: Throwable) {}
    }

    // ===== Data types =====
    data class YoloResults(
        var bowResults: MutableList<Point>?,
        var stringResults: MutableList<Point>?
    )

    data class Point(var x: Double, var y: Double)

    data class OrientedBoundingBox(
        val x: Float,
        val y: Float,
        val height: Float,
        val width: Float,
        val conf: Float,
        val cls: Int,
        val angle: Float
    )

    data class returnBow(
        var classification: Int?,
        var bow: List<Point>?,
        var string: List<Point>?,
        var angle: Int?
    )

    // ===== Inference =====
    fun detect(frame: Bitmap): YoloResults {
        val results = YoloResults(null, null)
        if (tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0) {
            Log.e(TAG, "MODEL ERROR: invalid tensor shapes")
            return results
        }

        val resized = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)
        val tensorImage = TensorImage(INPUT_IMAGE_TYPE).also { it.load(resized) }
        val processed = imageProcessor.process(tensorImage)
        val imageBuffer = processed.buffer
        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)

        val t0 = SystemClock.uptimeMillis()
        interpreter.run(imageBuffer, output.buffer)
        val inferMs = SystemClock.uptimeMillis() - t0
        Log.d(TAG, "inference ${inferMs}ms")

        // Handle [1, 7, N] or [1, N, 7]
        val outShape = interpreter.getOutputTensor(0).shape()
        val raw = output.floatArray

// detect layout, compute (C, N) and possibly transpose
        val (C, N, parsed) = if (outShape.size == 3 && outShape[1] == 8400 && outShape[2] == 7) {
            Triple(7, 8400, transposeN7To7N(raw, N = outShape[1], C = outShape[2]))
        } else {
            // assume [1, 7, N]
            Triple(outShape[1], outShape[2], raw)
        }

        val boxes = newBestBox(parsed, N)
        Log.d(
            "CheckDelBox",
            buildString {
                append("POSTPROCESS BOXES (${boxes.size}) →\n")
                boxes.take(5).forEachIndexed { i, b ->
                    append(
                        "[$i] cls=${b.cls}, conf=${b.conf.f2()}, " +
                                "x=${b.x.f2()}, y=${b.y.f2()}, w=${b.width.f2()}, h=${b.height.f2()}, ang=${b.angle.f2()}\n"
                    )
                }
                if (boxes.size > 5) append("... (${boxes.size - 5} more)\n")
            }
        )
        val imgW = frame.width.toFloat()
        val imgH = frame.height.toFloat()
        val sX = imgW / tensorWidth.toFloat()
        val sY = imgH / tensorHeight.toFloat()

        // Treat small (<=1.5) as normalized; else tensor-space
        val isNorm = boxes.isNotEmpty() &&
                boxes.maxOf { max(max(it.x, it.y), max(it.width, it.height)) } <= 1.5f

        fun toImgX(x: Float) = if (isNorm) x * imgW else x * sX
        fun toImgY(y: Float) = if (isNorm) y * imgH else y * sY
        fun toImgW(w: Float) = if (isNorm) w * imgW else w * sX
        fun toImgH(h: Float) = if (isNorm) h * imgH else h * sY

//        val isNorm = boxes.isNotEmpty() &&
//                boxes.maxOf { max(max(it.x, it.y), max(it.width, it.height)) } <= 1.5f
//
//        fun toNormX(x: Float) = if (isNorm) x else x / tensorWidth.toFloat()
//        fun toNormY(y: Float) = if (isNorm) y else y / tensorHeight.toFloat()
//        fun toNormW(w: Float) = if (isNorm) w else w / tensorWidth.toFloat()
//        fun toNormH(h: Float) = if (isNorm) h else h / tensorHeight.toFloat()
        var bowConf = 0f
        var stringConf = 0f

        for (b in boxes) {
            val cx = b.x
            val cy = b.y
            val ww = b.width
            val hh = b.height

            if (b.cls == 0 && b.conf > bowConf) {
                // Bow: subtract π/2 to align long side horizontally (same convention as your old file)
                val bowAngle = b.angle

                results.bowResults = rotatedRectToPoints(cx, cy, ww, hh, bowAngle).toMutableList()
                bowConf = b.conf
            } else if (b.cls == 1 && b.conf > stringConf) {
                // String: use angle as-is
                results.stringResults = sortStringPoints(
                    rotatedRectToPoints(cx, cy, ww, hh, b.angle).toMutableList()
                )
                stringConf = b.conf
            }
        }

        if (results.bowResults == null && results.stringResults == null) {
            listener?.noDetect()
        } else {
            listener?.detected(results, frame.width, frame.height)
        }
        return results
    }

    // Transpose flattened [1, 8400, 7] -> [1, 7, 8400]
    private fun transposeN7To7N(src: FloatArray, N: Int = 8400, C: Int = 7): FloatArray {
        val dst = FloatArray(C * N)
        var n = 0
        while (n < N) {
            val base = n * C
            var c = 0
            while (c < C) {
                dst[c * N + n] = src[base + c]
                c++
            }
            n++
        }
        return dst
    }

    private fun newBestBox(array: FloatArray, N: Int): List<OrientedBoundingBox> {
        val out = mutableListOf<OrientedBoundingBox>()
        // loop over anchors/proposals
        for (r in 0 until N) {
            val stringCnf = array[5 * N + r]
            val bowCnf    = array[4 * N + r]
            val cls = if (stringCnf > bowCnf) 1 else 0
            val cnf = max(stringCnf, bowCnf)
            if (cnf > CONFIDENCE_THRESHOLD) {
                val x = array[0 * N + r]
                val y = array[1 * N + r]
                val h = array[2 * N + r]
                val w = array[3 * N + r]
                val angle = array[6 * N + r]
                out.add(OrientedBoundingBox(x, y, h, w, cnf, cls, angle))
            }
        }
        return out
    }

    // ===== Geometry & classification =====
    private fun rotatedRectToPoints(
        cx: Float, cy: Float, w: Float, h: Float, angleRad: Float
    ): List<Point> {
        val halfW = w / 2f
        val halfH = h / 2f
        val c = cos(angleRad)
        val s = sin(angleRad)
        val corners = listOf(
            -halfW to -halfH, halfW to -halfH,
            halfW to  halfH, -halfW to  halfH
        )
        return corners.map { (x, y) ->
            val xRot = x * c - y * s + cx
            val yRot = x * s + y * c + cy
            Point(xRot.toDouble(), yRot.toDouble())
        }
    }

    fun updatePoints(stringBox: MutableList<Point>, bowBox: MutableList<Point>) {
        bowPoints = bowBox
        if (!yLocked) {
            stringPoints = stringBox
        } else if (yAvg != null) {
            stringPoints = sortStringPoints(stringBox)
        }
    }

    fun sortStringPoints(pts: MutableList<Point>): MutableList<Point> {
        val sorted = pts.sortedBy { it.y }
        val top = sorted.take(2).sortedBy { it.x }
        val bottom = sorted.drop(2).sortedByDescending { it.x }
        return (top + bottom).toMutableList()
    }

    fun getMidline(): MutableList<Double> {
        fun dist(a: Point, b: Point) =
            (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)

        val d = listOf(
            dist(bowPoints!![0], bowPoints!![1]),
            dist(bowPoints!![1], bowPoints!![2]),
            dist(bowPoints!![2], bowPoints!![3]),
            dist(bowPoints!![3], bowPoints!![0])
        )
        val i = d.indexOf(d.minOrNull())

        val (pair1, pair2) = when (i) {
            0 -> bowPoints!![0] to bowPoints!![1] to (bowPoints!![2] to bowPoints!![3])
            1 -> bowPoints!![1] to bowPoints!![2] to (bowPoints!![3] to bowPoints!![0])
            2 -> bowPoints!![2] to bowPoints!![3] to (bowPoints!![0] to bowPoints!![1])
            else -> bowPoints!![3] to bowPoints!![0] to (bowPoints!![1] to bowPoints!![2])
        }.let { (a, b) ->
            Pair(a, b)
        }

        val mid1 = listOf((pair1.first.x + pair1.second.x) / 2, (pair1.first.y + pair1.second.y) / 2)
        val mid2 = listOf((pair2.first.x + pair2.second.x) / 2, (pair2.first.y + pair2.second.y) / 2)

        val dy = mid1[1] - mid2[1]
        val dx = mid1[0] - mid2[0]

        return if (dx == 0.0) {
            mutableListOf(Double.POSITIVE_INFINITY, mid1[0])
        } else {
            val m = dy / dx
            val b = mid1[1] - m * mid1[0]
            mutableListOf(m, b)
        }
    }

    private fun getVerticalLines(): MutableList<MutableList<Double>> {
        val tl = stringPoints!![0]
        val tr = stringPoints!![1]
        val br = stringPoints!![2]
        val bl = stringPoints!![3]

        fun line(p1: Point, p2: Point): MutableList<Double> {
            val dx = p1.x - p2.x
            return if (dx == 0.0) {
                mutableListOf(Double.POSITIVE_INFINITY, -1.0, p1.y, p2.y)
            } else {
                val m = (p1.y - p2.y) / dx
                val b = p1.y - m * p1.x
                mutableListOf(m, b, p1.y, p2.y)
            }
        }
        return mutableListOf(line(tl, bl), line(tr, br))
    }

    private fun intersectsVertical(
        linearLine: MutableList<Double>,
        verticalLines: MutableList<MutableList<Double>>
    ): Int {
        val m = linearLine[0]
        val b = linearLine[1]

        fun intersect(v: List<Double>, xRef: Double): Point? {
            val mv = v[0]
            val bv = v[1]
            val yTop = v[2]
            val yBot = v[3]

            val (x, y) = when {
                mv == Double.POSITIVE_INFINITY || bv == -1.0 -> {
                    val x0 = xRef
                    if (m == Double.POSITIVE_INFINITY) return null
                    x0 to (m * x0 + b)
                }
                m == Double.POSITIVE_INFINITY -> {
                    val x0 = b
                    x0 to (mv * x0 + bv)
                }
                abs(m - mv) < 1e-6 -> return null
                else -> {
                    val x0 = (bv - b) / (m - mv)
                    x0 to (m * x0 + b)
                }
            }

            val yMin = min(yTop, yBot)
            val yMax = max(yTop, yBot)
            if (y !in yMin..yMax) return null
            return Point(x, y)
        }

        val xLeft = stringPoints!![0].x
        val xRight = stringPoints!![1].x
        var p1 = intersect(verticalLines[0], xLeft)
        var p2 = intersect(verticalLines[1], xRight)

        if (p1 == null && p2 == null) return 1
        if (p1 == null) p1 = p2
        if (p2 == null) p2 = p1
        return bowHeightIntersection(mutableListOf(p1!!, p2!!), mutableListOf(verticalLines[0], verticalLines[1]))
    }

    private fun bowHeightIntersection(
        intersectionPoints: MutableList<Point>,
        verticalLines: List<List<Double>>
    ): Int {
        val topPct = 0.10
        val botPct = 0.15

        val v1 = verticalLines[0]
        val v2 = verticalLines[1]

        val topY1 = v1[2]
        val topY2 = v2[2]
        val botY1 = v1[3]
        val botY2 = v2[3]

        val height = abs(((botY1 - topY1) + (botY2 - topY2)) / 2.0)
        if (height == 0.0) return 0

        val avgTop = (topY1 + topY2) / 2.0
        val avgBot = (botY1 + botY2) / 2.0

        val tooHigh = avgTop + height * topPct
        val tooLow  = avgBot - height * botPct

        val yAvg = intersectionPoints.map { it.y }.average()

        return when {
            yAvg <= tooHigh -> 2
            yAvg >= tooLow  -> 3
            else            -> 0
        }
    }

    private fun degrees(radians: Double) = radians * (180.0 / Math.PI)
    private fun logFinal(out: returnBow) {
        Log.d(
            "CheckDel2",
            buildString {
                append("FINAL RESULTS → class=${out.classification}, angle=${out.angle}\n")
                append("BOW: ")
                out.bow?.forEach { append("(${String.format("%.1f", it.x)}, ${String.format("%.1f", it.y)}) ") }
                append("\nSTRING: ")
                out.string?.forEach { append("(${String.format("%.1f", it.x)}, ${String.format("%.1f", it.y)}) ") }
            }
        )
    }
    private fun bowAngle(
        bowLine: MutableList<Double>,
        verticalLines: MutableList<MutableList<Double>>
    ): Int {
        val maxAngle = 15
        val mBow = bowLine[0]
        val m1 = verticalLines[0][0]
        val m2 = verticalLines[1][0]
        val a1 = abs(degrees(atan(abs(mBow - m2) / (1 + mBow * m2))))
        val a2 = abs(degrees(atan(abs(m1 - mBow) / (1 + m1 * mBow))))
        val minAngle = abs(90 - min(a1, a2))
        return if (minAngle > maxAngle) 1 else 0
    }

    fun classify(results: YoloResults): returnBow {
        val out = returnBow(null, null, null, null)

        if (results.stringResults != null) {
            stringRepeat = 0
            stringPoints = results.stringResults
        } else if (stringRepeat < 5 && stringPoints != null) {
            out.classification = -1
            stringRepeat++
            results.stringResults = stringPoints!!.toMutableList()
        } else stringPoints = null




        if (results.bowResults != null) {
            bowRepeat = 0
            bowPoints = results.bowResults
        } else if (bowRepeat < 5 && bowPoints != null) {
            out.classification = -1
            bowRepeat++
            results.bowResults = bowPoints!!.toMutableList()
        } else bowPoints = null

        if (stringPoints == null && bowPoints == null) {
            out.classification = -2
            logFinal(out)             // <— log before returning

            return out
        }
        if (results.stringResults == null) {
            out.classification = -1
            out.bow = results.bowResults
            logFinal(out)             // <— log before returning

            return out
        } else if (results.bowResults == null) {
            out.classification = -1
            out.string = results.stringResults
            logFinal(out)             // <— log before returning

            return out
        } else {
            out.string = results.stringResults
            out.bow = results.bowResults
            updatePoints(results.stringResults!!, results.bowResults!!)
            val mid = getMidline()
            val verts = getVerticalLines()
            val intersectClass = intersectsVertical(mid, verts)
            out.angle = bowAngle(mid, verts)
            out.classification = intersectClass
            Log.d("BOW", out.classification.toString())
            logFinal(out)             // <— log before returning

            return out
        }
    }

    // ===== Drawing =====
    fun drawPointsOnBitmap(
        bitmap: Bitmap,
        points: YoloResults,
        classification: Int?,
        angle: Int?
    ): Bitmap {
        val canvas = Canvas(bitmap)
        val hasIssue = (classification != null && classification != 0) || (angle == 1)
        val boxColor = if (hasIssue) Color.rgb(255, 140, 0) else Color.BLUE

        val paint = Paint().apply {
            color = boxColor
            style = Paint.Style.STROKE
            strokeWidth = 8f
            isAntiAlias = true
        }

        fun drawQuad(ps: List<Point>?) {
            if (ps == null || ps.size < 4) return
            canvas.drawLine(ps[0].x.toFloat(), ps[0].y.toFloat(), ps[1].x.toFloat(), ps[1].y.toFloat(), paint)
            canvas.drawLine(ps[1].x.toFloat(), ps[1].y.toFloat(), ps[2].x.toFloat(), ps[2].y.toFloat(), paint)
            canvas.drawLine(ps[2].x.toFloat(), ps[2].y.toFloat(), ps[3].x.toFloat(), ps[3].y.toFloat(), paint)
            canvas.drawLine(ps[3].x.toFloat(), ps[3].y.toFloat(), ps[0].x.toFloat(), ps[0].y.toFloat(), paint)
        }

        drawQuad(points.stringResults)
        drawQuad(points.bowResults)

        val classificationLabels = mapOf(
            0 to "",
            1 to "Bow outside zone",
            2 to "Bow too high",
            3 to "Bow too low"
        )
        val angleLabels = mapOf(
            0 to "",
            1 to "Incorrect bow angle"
        )

        val textPaint = Paint().apply {
            color = Color.rgb(255, 140, 0)
            style = Paint.Style.FILL
            textSize = 56f
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
        }
        val strokePaint = Paint().apply {
            color = Color.rgb(204, 85, 0)
            style = Paint.Style.STROKE
            strokeWidth = 6f
            textSize = 56f
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
        }

        val topMargin = 300f
        val lineSpacing = 70f
        val cx = bitmap.width / 2f
        var y = topMargin

        if (classification != null && classification != 0) {
            val msg = classificationLabels[classification] ?: ""
            if (msg.isNotEmpty()) {
                canvas.drawText(msg, cx, y, strokePaint)
                canvas.drawText(msg, cx, y, textPaint)
                y += lineSpacing
            }
        }
        if (angle == 1) {
            val msg = angleLabels[1]!!
            canvas.drawText(msg, cx, y, strokePaint)
            canvas.drawText(msg, cx, y, textPaint)
        }
        return bitmap
    }

    // Keep this if your module expects it to return an annotated frame
    fun process_frame(bitmap: Bitmap): Bitmap {
        val cls = classify(detect(bitmap))
        return drawPointsOnBitmap(
            bitmap,
            YoloResults(
                bowResults = cls.bow?.toMutableList(),
                stringResults = cls.string?.toMutableList()
            ),
            cls.classification,
            cls.angle
        )
    }

    interface DetectorListener {
        fun noDetect()
        fun detected(results: YoloResults, sourceWidth: Int, sourceHeight: Int)
    }
}
