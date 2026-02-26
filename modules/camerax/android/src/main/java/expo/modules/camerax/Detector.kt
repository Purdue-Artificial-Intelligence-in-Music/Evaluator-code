package expo.modules.camerax

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
// import android.graphics.PointF
// import android.gesture.OrientedBoundingBox
// import com.google.android.gms.tasks.Task
// import com.google.android.gms.tflite.java.TfLite
// import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.util.concurrent.CountDownLatch
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import com.qualcomm.qti.QnnDelegate
import com.qualcomm.qti.QnnDelegate.Options.BackendType
import java.io.File
import kotlin.math.*
import android.graphics.Typeface
import java.nio.ByteBuffer


class Detector (
    private val context: Context,
    private val listener: DetectorListener? = null
){

    private var interpreter: Interpreter
    private var qnnDelegate: QnnDelegate? = null
    private var tfliteGpu: GpuDelegate? = null

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    val modelReadyLatch = CountDownLatch(1)
    private var bowRepeat = 0
    private var stringRepeat = 0
    private var bowPoints: List<Point>? = null
    private var stringPoints: List<Point>? = null
    private val MAX_QUEUE_SIZE = 60
    private val MAX_Y_DELTA_THRESHOLD = 3
    private val MAX_BOW_DIST_THRESHOLD = 5
    private var ogWidth: Int = 0
    private var ogHeight: Int = 0

    // flexibility of angle relative to 90 degrees
    // use 15 as default, and receive input (range 0-90) from the frontend
    private var maxAngle: Int = 20

    //Heaps for yDelta calculation
    private var lowerHeap = java.util.PriorityQueue<Double>(compareByDescending { it })
    private var upperHeap = java.util.PriorityQueue<Double>()

    private val deltaQueue = ArrayDeque<Double>() // MAX_QUEUE_SIZE

    public fun resetHeaps() {
        lowerHeap = java.util.PriorityQueue<Double>(compareByDescending { it })
        upperHeap = java.util.PriorityQueue<Double>()
    }
    private fun addDelta(value: Double) {
        if (lowerHeap.isEmpty() || value <= lowerHeap.peek()) {
            lowerHeap.add(value)
        } else {
            upperHeap.add(value)
        }
        rebalanceHeaps()
    }

    private fun removeDelta(value: Double) {
        if (!lowerHeap.remove(value)) {
            upperHeap.remove(value)
        }
        rebalanceHeaps()
    }

    private fun rebalanceHeaps() {
        when {
            lowerHeap.size > upperHeap.size + 1 ->
                upperHeap.add(lowerHeap.poll())

            upperHeap.size > lowerHeap.size ->
                lowerHeap.add(upperHeap.poll())
        }
    }

    private fun currentMedian(): Double {
        return if (lowerHeap.size == upperHeap.size) {
            (lowerHeap.peek() + upperHeap.peek()) / 2.0
        } else {
            lowerHeap.peek()
        }
    }

    // add setter for MaxAngle
    fun setMaxAngle(angle: Int) {
        maxAngle = angle.coerceIn(0, 90)
        Log.d("MaxAngle", "Max angle set to: $maxAngle")
    }


    private fun Double.f1() = String.format("%.1f", this)
    private fun Double.f3() = String.format("%.3f", this)
    private fun Float.f2() = String.format("%.2f", this)

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()
    private fun nativeLibDir(): String = context.applicationInfo.nativeLibraryDir
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
        private const val MODEL_ASSET = "try.tflite" // <-- set your model file name here
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.6F
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
    private fun createInterpreterWithFallbacks(context: Context): org.tensorflow.lite.Interpreter {
        // use the correct TFLite FileUtil and cast to ByteBuffer to solve ambiguity
        val modelBuffer: java.nio.ByteBuffer = org.tensorflow.lite.support.common.FileUtil.loadMappedFile(context, MODEL_ASSET)

        // 1) Qualcomm NPU (QNN/HTP) attempt
        val skelDir = tryLoadQnnAndPickSkelDir()
        if (skelDir != null) {
            try {
                val qnnOptions = org.tensorflow.lite.Interpreter.Options()
                val qOpts = QnnDelegate.Options().apply {
                    setBackendType(BackendType.HTP_BACKEND)
                    setSkelLibraryDir(skelDir)
                }
                qnnDelegate = QnnDelegate(qOpts)
                qnnOptions.addDelegate(qnnDelegate)

                Log.i(TAG, "Trying Qualcomm QNN delegate...")
                // Fix 2: Explicitly pass the casted buffer
                val interp = org.tensorflow.lite.Interpreter(modelBuffer as java.nio.ByteBuffer, qnnOptions)
                Log.i(TAG, "Successfully using QNN delegate (HTP/NPU)")
                return interp
            } catch (t: Throwable) {
                Log.w(TAG, "QNN delegate failed, cleaning up: ${t.message}")
                qnnDelegate?.close()
                qnnDelegate = null
                // Fall through to next attempt
            }
        }

        // 2) GPU attempt
       try {
            val gpuOptions = org.tensorflow.lite.Interpreter.Options()
            val cl = org.tensorflow.lite.gpu.CompatibilityList()

            if (cl.isDelegateSupportedOnThisDevice) {
                tfliteGpu = org.tensorflow.lite.gpu.GpuDelegate(cl.bestOptionsForThisDevice)
                gpuOptions.addDelegate(tfliteGpu)

                Log.i(TAG, "Trying TFLite GPU delegate...")
                val interp = org.tensorflow.lite.Interpreter(modelBuffer as java.nio.ByteBuffer, gpuOptions)
                Log.i(TAG, "Successfully using GPU delegate")
                return interp
            }
        } catch (t: Throwable) {
            Log.w(TAG, "GPU delegate failed, cleaning up: ${t.message}")
            tfliteGpu?.close()
            tfliteGpu = null
        }

        // 3) CPU Fallback
        val cpuOptions = org.tensorflow.lite.Interpreter.Options()
        Log.i(TAG, "Falling back to CPU/XNNPACK")
        try {
            cpuOptions.setUseXNNPACK(true)
        } catch (_: Throwable) {}
        cpuOptions.setNumThreads(4)

        return org.tensorflow.lite.Interpreter(modelBuffer as java.nio.ByteBuffer, cpuOptions)
    }

    fun close() {
        try { interpreter.close() } catch (_: Throwable) {}
        try { tfliteGpu?.close() } catch (_: Throwable) {}
        try { qnnDelegate?.close() } catch (_: Throwable) {}
    }
    data class YoloResults(
        var bowResults: MutableList<Point>?,
        var stringResults: MutableList<Point>?
    )

    data class Point(
        var x: Double,
        var y: Double
    )


    fun detect(frame: Bitmap, sourceWidth: Int = 1, sourceHeight: Int = 1): YoloResults {
        val results = YoloResults(null, null)

        if (tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0) {
            Log.e(TAG, "MODEL ERROR: invalid tensor shapes")
            return results
        }

        // Resize → normalize → cast
        //val resized = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)
        val resizedNew = Bitmap.createScaledBitmap(frame, 360, 640, false)
        val padded = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(padded)
        canvas.drawBitmap(resizedNew, 0f, 0f, null)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE).also { it.load(padded) }
        val processed = imageProcessor.process(tensorImage)
        val imageBuffer = processed.buffer

        // Model output buffer
        val output = TensorBuffer.createFixedSize(
            intArrayOf(1, numChannel, numElements),
            OUTPUT_IMAGE_TYPE
        )

        // Inference
        val t0 = SystemClock.uptimeMillis()
        interpreter.run(imageBuffer, output.buffer)
        val inferMs = SystemClock.uptimeMillis() - t0
        Log.d(TAG, "inference ${inferMs}ms")
        Log.i("Time Comparing", "Bow YOLO: inference ${inferMs}ms")

        // Handle output layout: [1, 7, N] vs [1, N, 7] (e.g., 8400)
        val outShape = interpreter.getOutputTensor(0).shape()
        val raw = output.floatArray

        val (C, N, parsed) = if (outShape.size == 3 && outShape[1] == 8400 && outShape[2] == 7) {
            // Flattened [1, 8400, 7] → [1, 7, 8400]
            Triple(7, 8400, transposeN7To7N(raw, N = outShape[1], C = outShape[2]))
        } else {
            // Assume [1, 7, N]
            Triple(outShape[1], outShape[2], raw)
        }

        // Parse best boxes from [1, 7, N]
        val bestBoxes = newBestBox(parsed, N)

        // Log a peek
        Log.d(
            "CheckDelBox",
            buildString {
                append("POSTPROCESS BOXES (${bestBoxes.size}) →\n")
                bestBoxes.take(5).forEachIndexed { i, b ->
                    append(
                        "[$i] cls=${b.cls}, conf=${b.conf.f2()}, " +
                                "x=${b.x.f2()}, y=${b.y.f2()}, w=${b.width.f2()}, h=${b.height.f2()}, ang=${b.angle.f2()}\n"
                    )
                }
                if (bestBoxes.size > 5) append("... (${bestBoxes.size - 5} more)\n")
            }
        )


        var bowConf = 0f
        var stringConf = 0f

        val ogWidth = frame.width.toFloat()
        val ogHeight = frame.height.toFloat()
        //val ogWidth = 1
        //val ogHeight = 1

        //Log.d("BOXES123", bestBoxes.size.toString())
        for (box in bestBoxes) {

            if (box.cls == 0 && box.conf > bowConf) {
                Log.d("BOX INITAL BOW", box.toString())
                results.bowResults = rotatedRectToPoints(box.x, box.y, box.width, box.height, box.angle, ogWidth, ogHeight).toMutableList()
                bowConf = box.conf
                Log.d("BOX BOW", results.bowResults.toString())

            } else if (box.cls == 1 && box.conf > stringConf) {
                Log.d("BOX INITAL STRING", box.toString())
                results.stringResults = sortStringPoints(rotatedRectToPoints(box.x, box.y, box.width, box.height, box.angle, ogWidth, ogHeight).toMutableList())
                stringConf = box.conf
                Log.d("BOX STRING", results.stringResults.toString())
            }
        }
        //println("TRUE INFERENCE TIME: $inferenceTime")
        //println("NUMBER OF BOXES: ${bestBoxes.size}")
        //println("bow conf, string conf: $bowConf, $stringConf")

        if (results.bowResults == null && results.stringResults == null) {
            // No Detections Found
            listener?.noDetect()
            Log.d("BOW RESULTS", "NO DETECTIONS")
        } else {
            // Detected Correctly!
            Log.d("BOXES123", "SOMETHING DETECTED")
            //println(results)
            // Update y-box averages
            listener?.detected(results, frame.width, frame.height)
            //print(results)
        }
        return results
    }


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

    fun drawPointsOnBitmap(
        bitmap: Bitmap,
        points: YoloResults,
        classification: Int?,
        angle: Int?
    ): Bitmap {
        val canvas = Canvas(bitmap)

        // Determine if there's an issue with classification or angle
        // 0 = correct, anything else is an issue
        val hasIssue = (classification != null && classification != 0) ||
                (angle != null && angle == 1)

        // Choose colors based on classification
        val boxColor = if (hasIssue) Color.rgb(255, 140, 0) else Color.BLUE // Orange or Blue

        val paint = Paint().apply {
            color = boxColor
            style = Paint.Style.STROKE
            strokeWidth = 8f
            isAntiAlias = true
        }
        val scaleX = 720f * 1.5f
        val scaleY = 1280f * 1.5f

        // Draw string box (rectangle)
        if (points.stringResults != null && points.stringResults!!.size >= 4) {
            val stringBox = points.stringResults!!
            // Draw four lines connecting the corners
            canvas.drawLine(
                stringBox[0].x.toFloat()/640f * scaleX, stringBox[0].y.toFloat()/640f * scaleY,
                stringBox[1].x.toFloat()/640f * scaleX, stringBox[1].y.toFloat()/640f * scaleY,
                paint
            )
            canvas.drawLine(
                stringBox[1].x.toFloat()/640f * scaleX, stringBox[1].y.toFloat()/640f * scaleY,
                stringBox[2].x.toFloat()/640f * scaleX, stringBox[2].y.toFloat()/640f * scaleY,
                paint
            )
            canvas.drawLine(
                stringBox[2].x.toFloat()/640f * scaleX, stringBox[2].y.toFloat()/640f * scaleY,
                stringBox[3].x.toFloat()/640f * scaleX, stringBox[3].y.toFloat()/640f * scaleY,
                paint
            )
            canvas.drawLine(
                stringBox[3].x.toFloat()/640f * scaleX, stringBox[3].y.toFloat()/640f * scaleY,
                stringBox[0].x.toFloat()/640f * scaleX, stringBox[0].y.toFloat()/640f * scaleY,
                paint
            )
        }

        // Draw bow box (rectangle)
        if (points.bowResults != null && points.bowResults!!.size >= 4) {
            val bowBox = points.bowResults!!
            // Draw four lines connecting the corners
            canvas.drawLine(
                bowBox[0].x.toFloat()/640f * scaleX, bowBox[0].y.toFloat()/640f * scaleY,
                bowBox[1].x.toFloat()/640f * scaleX, bowBox[1].y.toFloat()/640f * scaleY,
                paint
            )
            canvas.drawLine(
                bowBox[1].x.toFloat()/640f * scaleX, bowBox[1].y.toFloat()/640f * scaleY,
                bowBox[2].x.toFloat()/640f * scaleX, bowBox[2].y.toFloat()/640f * scaleY,
                paint
            )
            canvas.drawLine(
                bowBox[2].x.toFloat()/640f * scaleX, bowBox[2].y.toFloat()/640f * scaleY,
                bowBox[3].x.toFloat()/640f * scaleX, bowBox[3].y.toFloat()/640f * scaleY,
                paint
            )
            canvas.drawLine(
                bowBox[3].x.toFloat()/640f * scaleX, bowBox[3].y.toFloat()/640f * scaleY,
                bowBox[0].x.toFloat()/640f * scaleX, bowBox[0].y.toFloat()/640f * scaleY,
                paint
            )
        }

        // Classification labels mapping
        // -2: No detection, -1: Partial, 0: Correct, 1: Outside, 2: Too high, 3: Too low
        val classificationLabels = mapOf(
            0 to "",  // Correct - don't display
            1 to "Keep the bow in zone",    // Bow outside zone
            2 to "Lower the bow",    // Bow too high
            3 to "Lift the bow"    // Bow too low
        )

        // Angle labels: 0 = correct, 1 = wrong
        val angleLabels = mapOf(
            0 to "",  // Correct - don't display
            1 to "Adjust your bow angle"    // Incorrect bow angle
        )

        // Prepare text paint styles
        val textPaint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.FILL
            textSize = 56f
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
        }

        // Semi-transparent white rectangle paint
        val labelBackgroundPaint = Paint().apply {
            color = Color.argb(180, 255, 255, 255) // semi-transparent white
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        // Fixed positions from top - below the hand/pose classifications
        val topMargin = 375f  // Below hand (160f) and pose (230f) messages
        val lineSpacing = 60f
        val centerX = bitmap.width / 2f
        val padding = 16f

        var currentY = topMargin

        // Draw classification message if there's an issue
        if (classification != null && classification != 0) {
            val message = classificationLabels[classification] ?: ""
            if (message.isNotEmpty()) {
                val textWidth = textPaint.measureText(message)
                val fm = textPaint.fontMetrics
                val textHeight = fm.bottom - fm.top

                // Rectangle coordinates
                val left = centerX - textWidth / 2 - padding
                val top = currentY + fm.top - padding
                val right = centerX + textWidth / 2 + padding
                val bottom = currentY + fm.bottom + padding

                canvas.drawRect(left, top, right, bottom, labelBackgroundPaint)

                canvas.drawText(message, centerX, currentY, textPaint)
                currentY += (fm.bottom - fm.top) + lineSpacing
            }
        }

        // Draw angle message if there's an issue
        if (angle != null && angle == 1) {
            val message = angleLabels[angle] ?: ""
            if (message.isNotEmpty()) {
                val textWidth = textPaint.measureText(message)
                val fm = textPaint.fontMetrics
                val textHeight = fm.bottom - fm.top

                // Rectangle coordinates
                val left = centerX - textWidth / 2 - padding
                val top = currentY + fm.top - padding
                val right = centerX + textWidth / 2 + padding
                val bottom = currentY + fm.bottom + padding

                canvas.drawRect(left, top, right, bottom, labelBackgroundPaint)

                canvas.drawText(message, centerX, currentY, textPaint)
            }
        }

        return bitmap
    }

    fun process_frame(bitmap: Bitmap): Bitmap {
        val classificationResult = classify(detect(bitmap))
        val annotatedBitmap = drawPointsOnBitmap(
            bitmap,
            YoloResults(
                bowResults = classificationResult.bow?.toMutableList(),
                stringResults = classificationResult.string?.toMutableList()
            ),
            classificationResult.classification,
            classificationResult.angle
        )
        return annotatedBitmap
        //return new bitmapAndClassifications(annotatedBitmap, classificationResult.classification, classificationResult.angle)
    }

    fun analyzeFrame(bitmap:Bitmap): bitmapAndClassifications {
        val classificationResult = classify(detect(bitmap))
        val annotatedBitmap = drawPointsOnBitmap(
            bitmap,
            YoloResults(
                bowResults = classificationResult.bow?.toMutableList(),
                stringResults = classificationResult.string?.toMutableList()
            ),
            classificationResult.classification,
            classificationResult.angle
        )
        return bitmapAndClassifications(annotatedBitmap, classificationResult.classification, classificationResult.angle)
    }



    private fun rotatedRectToPoints(cx: Float, cy: Float, w: Float, h: Float, angleRad: Float, frameWidth: Float, frameHeight: Float): List<Point> {
//        val normalizedAngle = angleRad % Math.PI
//        val swap = normalizedAngle >= Math.PI /2
//        var newW = 0f
//        var newH = 0f
//        if (swap) {
//            newW = h
//            newH = w
//        } else {
//            newW = w
//            newH = h
//        }
//        val newAngle = (angleRad % (Math.PI/2)).toFloat()


        val halfW = (w) / 2
        val halfH = (h) / 2
        println("ANGLE $angleRad")
        val cosA = cos(angleRad)
        val sinA = sin(angleRad)

        //val xMult =
        //val topRightX = cx * frameWidth +
        val corners = listOf(
            Pair(-halfW, -halfH),
            Pair(halfW, -halfH),
            Pair(halfW, halfH),
            Pair(-halfW, halfH)
        )
        return corners.map { (x, y) ->

            val xRot = x * cosA - y * sinA + cx
            val yRot = x * sinA + y * cosA + cy

            //Point(xRot.toDouble() * frameWidth, yRot.toDouble() * frameHeight)
            Point((xRot).toDouble() * 640f/360f, (yRot).toDouble())
        }



        //val topRightX = cx * frameWidth + halfH * sinA + halfW * cosA
    }

    private fun newBestBox(array: FloatArray, N: Int): List<OrientedBoundingBox> {
        val out = mutableListOf<OrientedBoundingBox>()

        for (r in 0 until N) {
            val stringCnf = array[5 * N + r]
            val bowCnf = array[4 * N + r]
            val cls = if (stringCnf > bowCnf) 1 else 0
            var cnf = max(stringCnf, bowCnf)
            if (cls == 0) {
                cnf += 0.3f
            }

            if (cnf > CONFIDENCE_THRESHOLD) {
                val x = array[0 * N + r]
                val y = array[1 * N + r]
                val w = array[2 * N + r]
                val h = array[3 * N + r]
                val angle = array[6 * N + r]

                out.add(
                    OrientedBoundingBox(
                        x = x, y = y, height = h, width = w,
                        conf = cnf, cls = cls, angle = angle
                    )
                )
            }
        }

        return out
    }


    data class OrientedBoundingBox(
        val x: Float,
        val y: Float,
        val height: Float,
        val width: Float,
        val conf: Float,
        val cls: Int,
        val angle: Float
    )

    fun updatePoints(
        stringBox: MutableList<Point>,
        bowBox: MutableList<Point>
    ) {
        bowPoints = bowBox //change bow points to mutable list
        val stringPoints = updateStringPoints(stringBox)
        Log.d("sorted points", stringPoints.toString())
    }

    /*
     * Handles updating the queue of string points and the mean string box height (delta)
     * Returns the stringBox with locked top values if outside of threshold
     */
    fun updateStringPoints(stringBox: MutableList<Point>): MutableList<Point> {
        Log.d("String Box Lock", "New Detection")
        val sortedString = sortStringPoints(stringBox)

        // Height of string box
        val delta_y = kotlin.math.abs(sortedString[0].y - sortedString[3].y)

        deltaQueue.add(delta_y)
        addDelta(delta_y)
        
        // Enforce sliding window
        if (deltaQueue.size > MAX_QUEUE_SIZE) {
            val old = deltaQueue.removeFirst()
            removeDelta(old)
        }

        // First frame: no locking
        if (deltaQueue.size == 1) {
            stringPoints = sortedString
            return sortedString
        }

        val medianDelta = currentMedian()

        //Log.d("String Box Lock", "Delta_Y: $delta_y")
        //Log.d("String Box Lock", "Median Delta_Y: $medianDelta")

        // Lock if deviation too large
        if (kotlin.math.abs(medianDelta - delta_y) > MAX_Y_DELTA_THRESHOLD) {
            // Lock top points using median height
            sortedString[0].y = sortedString[3].y - medianDelta
            sortedString[1].y = sortedString[2].y - medianDelta
            Log.d("String Box Lock", "Deviation too large. Locked box to median_delta")
        }

        // Lock if bow is covering top of string box
        // Need bow box avg top to be above top of string box and within x bounds

        if (bowPoints != null) {
            val sortedBow = sortBowPoints(bowPoints!!.toMutableList())
            val top_avg_bow_y = (sortedBow[0].y + sortedBow[1].y) / 2
            val top_avg_str_y = (sortedString[0].y + sortedString[1].y) / 2
            val bot_avg_bow_y = (sortedBow[2].y + sortedBow[3].y) / 2
            //Log.d("String Box Lock", "y-vals Bow: (" + top_avg_bow_y.toString() + "," +  bot_avg_bow_y.toString() + ") String: " + top_avg_str_y.toString())
            //Log.d("String Box Lock", "x-vals Bow: (" + sortedBow[0].x.toString() + "," + sortedBow[1].x.toString() + ") String: (" + sortedString[0].x.toString() + "," + sortedString[1].x.toString() + ")")
            // Top of image is lower y value, bottom is higher

            if ((top_avg_bow_y <= top_avg_str_y) &&
                (bot_avg_bow_y >= (top_avg_str_y - MAX_BOW_DIST_THRESHOLD))) { // bow y-level on string box y-level
                if ((sortedBow[0].x < sortedString[0].x) and (sortedBow[1].x > sortedString[1].x)) { // within x range
                    // Lock top points using median height
                    sortedString[0].y = sortedString[3].y - medianDelta
                    sortedString[1].y = sortedString[2].y - medianDelta
                    Log.d("String Box Lock", "Bow Covering. Locked box to median_delta")
                }
            }
        }

        return sortedString
    }
    //
    fun sortStringPoints(pts: MutableList<Point>): MutableList<Point> {
        // Sort points by y
        val sortedPoints = pts.sortedBy {it.y }

        // Find first 2 and last pts
        val topPoints = sortedPoints.take(2).sortedBy { it.x }      // Sort by X ascending
        val bottomPoints = sortedPoints.drop(2).sortedByDescending { it.x } // Sort by X descending

        return (topPoints + bottomPoints).toMutableList()
    }

    fun sortBowPoints(pts: MutableList<Point>): MutableList<Point> {
        // Sort points by x
        val sortedPoints = pts.sortedBy {it.x}

        // Sort points by y
        val leftPoints = sortedPoints.take(2).sortedBy {it.y} // Sort by Y ascending
        val rightPoints = sortedPoints.drop(2).sortedBy {it.y} // Sort by Y ascending

        return mutableListOf(leftPoints[0], rightPoints[0], rightPoints[1], leftPoints[1])
    }

    fun getMidline(): MutableList<Double> {
        fun distance(pt1: Point, pt2: Point): Double {
            //just distance formula
            return (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y)
        }

        //find length of the sides of the bow rectangle
        val d1 = distance(bowPoints!![0], bowPoints!![1])
        val d2 = distance(bowPoints!![1], bowPoints!![2])
        val d3 = distance(bowPoints!![2], bowPoints!![3])
        val d4 = distance(bowPoints!![3], bowPoints!![0])
        val distances = listOf(d1, d2, d3, d4)

        val minIndex = distances.indexOf(distances.minOrNull()) //find the smallest distance

        //find the two shortest distances to fin dthe end of teh bow and set those points as pair1 and pair2
        val (pair1, pair2) = when (minIndex) {
            0 -> Pair(bowPoints!![0] to bowPoints!![1], bowPoints!![2] to bowPoints!![3])
            1 -> Pair(bowPoints!![1] to bowPoints!![2], bowPoints!![3] to bowPoints!![0])
            2 -> Pair(bowPoints!![2] to bowPoints!![3], bowPoints!![0] to bowPoints!![1])
            else -> Pair(bowPoints!![3] to bowPoints!![0], bowPoints!![1] to bowPoints!![2])
        }

        val mid1 = listOf((pair1.first.x + pair1.second.x) / 2, (pair1.first.y + pair1.second.y) / 2)
        val mid2 = listOf((pair2.first.x + pair2.second.x) / 2, (pair2.first.y + pair2.second.y) / 2)

        val dy = mid1[1] - mid2[1]
        val dx = mid1[0] - mid2[0]

        return if (dx == 0.0) {
            mutableListOf(Double.POSITIVE_INFINITY, mid1[0])
        } else {
            val slope = dy / dx
            val intercept = mid1[1] - slope * mid1[0]
            mutableListOf(slope, intercept)
        }
    }

    private fun getVerticalLines(): MutableList<MutableList<Double>> {
        System.out.println(stringPoints)
        // Extracting corner points
        val topLeft = stringPoints!![0]
        val topRight = stringPoints!![1]
        val botRight = stringPoints!![2]
        val botLeft = stringPoints!![3]
        // Left vertical line (from topLeft to botLeft)

        // Calculate the horizontal distance between the left points
        val dxLeft = topLeft.x - botLeft.x
        // Initialize slope and y-intercept for the left side
        val leftSlope: Double
        val leftYint: Double

        if (dxLeft == 0.0) {
            leftSlope = Double.POSITIVE_INFINITY
            leftYint = -1.0  // Use -1.0 as a flag for undefined intercept
        } else {
            // Calculate slope
            leftSlope = (topLeft.y - botLeft.y) / dxLeft
            // Calculate y-intercept
            leftYint = topLeft.y - leftSlope * topLeft.x
        }

        // Right vertical line (from topRight to botRight)

        val dxRight = topRight.x - botRight.x
        val rightSlope: Double
        val rightYint: Double

        if (dxRight == 0.0) {
            rightSlope = Double.POSITIVE_INFINITY
            rightYint = -1.0
        } else {
            rightSlope = (topRight.y - botRight.y) / dxRight
            rightYint = topRight.y - rightSlope * topRight.x
        }

        // Heights of each side (just the y-coordinates of top and bottom points)
        val leftTopY = topLeft.y
        val leftBotY = botLeft.y
        val rightTopY = topRight.y
        val rightBotY = botRight.y

        // Each line is a MutableList: [slope, intercept, topY, bottomY]
        val leftLine = mutableListOf(leftSlope, leftYint, leftTopY, leftBotY)
        val rightLine = mutableListOf(rightSlope, rightYint, rightTopY, rightBotY)

        // return a list of both lines
        return mutableListOf(leftLine, rightLine)
    }



    private fun intersectsVertical(
        linearLine: MutableList<Double>,
        verticalLines: MutableList<MutableList<Double>>
    ): Int {
        //println("linear: $linearLine\nvertical: $verticalLines")

        // Midline parameters
        val m = linearLine[0] // slope of the midline
        val b = linearLine[1] // y-intercept of the midline

        // extracts the first vertical line (left side): [slope, yInt, topY, botY]
        val verticalOne = verticalLines[0]
        val verticalTwo = verticalLines[1]

        // Calculates the intersection of the midline with a vertical line
        fun getIntersection(vLine: List<Double>, xRef: Double): Point? {
            val slopeV = vLine[0]
            val interceptV = vLine[1]
            val topY = vLine[2]
            val botY = vLine[3]

            val x: Double
            val y: Double

            if (slopeV == Double.POSITIVE_INFINITY || interceptV == -1.0) {
                x = xRef
                if (m == Double.POSITIVE_INFINITY) return null // both lines vertical
                y = m * x + b
            } else if (m == Double.POSITIVE_INFINITY) {
                // Case: vertical midline
                x = b
                y = slopeV * x + interceptV
            } else if (kotlin.math.abs(m - slopeV) < 1e-6) {
                // parallel lines means no intersection
                return null
            } else {
                x = (interceptV - b) / (m - slopeV)
                y = m * x + b
            }
            // Makes sure intersection y-value is within the vertical segment's range
            val yMin = minOf(topY, botY)
            val yMax = maxOf(topY, botY)

            if (yMin > y || y > yMax) {
                //println("Intersection y=$y is outside vertical range ($yMin, $yMax)")
                return null
            }


            return Point(x,y)
        }

        // Determine x positions from the bounding box
        val xLeft = stringPoints!![0].x
        val xRight = stringPoints!![1].x

        // Calculate intersections of midline with both vertical string lines
        var pt1 = getIntersection(verticalOne, xLeft)
        var pt2 = getIntersection(verticalTwo, xRight)
        Log.d("INTERSECTION", pt1.toString() + " " + pt2.toString())

        if (pt1 == null || pt2 == null) {
            //println("One or both intersections invalid")
            Log.d("BOW", "INVALID INTERSECTION")
            return 1
        }
//        if (pt1 == null) {
//            pt1 = pt2
//        }
//        if (pt2 == null){
//            pt2 = pt1
//        }
        return bowHeightIntersection(mutableListOf(pt1!!, pt2!!), mutableListOf(verticalOne, verticalTwo))
    }


    /*
     * Determines the height level at which the linear line intersects the vertical lines.

        Returns:
        - 3: Intersection is near top of the box (ht1 or ht2)
        - 2: Intersection is near bottom (hb1 or hb2)
        - 0: Intersection is in middle
     */


    private fun bowHeightIntersection(
        intersectionPoints: MutableList<Point>,
        verticalLines: List<List<Double>>
    ): Int {
        val top_zone_percentage = 0.1
        val bottom_zone_percentage = 0.1

        val vertical_one = verticalLines[0]
        val vertical_two = verticalLines[1]

        val top_y1 = vertical_one[2]
        val top_y2 = vertical_two[2]
        val bot_y1 = vertical_one[3]
        val bot_y2 = vertical_two[3]

        val height = abs(((bot_y1 - top_y1) + (bot_y2 - top_y2)) / 2.0)
        if (height == 0.0) return 0

        val avg_top_y = (top_y1 + top_y2) / 2.0
        val avg_bot_y = (bot_y1 + bot_y2) / 2.0

        val too_high_threshold = avg_top_y + height * top_zone_percentage
        val too_low_threshold = avg_bot_y - height * bottom_zone_percentage

        val intersection_y = intersectionPoints.map { it.y }.average()

        if (intersection_y <= too_high_threshold) {
            return 2
        }

        if (intersection_y >= too_low_threshold) {
            return 3
        }

        return 0
    }

    /*
        Returns median item of a list
     */
    private fun median(values: List<Int>): Double {
        if (values.isEmpty()) throw IllegalArgumentException("Empty list has no median.")

        val sorted = values.sorted()
        val middle = sorted.size / 2

        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle].toDouble()
        }
    }

    /*
    converts radians to degrees
     */
    private fun degrees(radians: Double): Double {
        return radians * (180.0 / PI)
    }

    /*
    classifies bow angle relative to two vertical lines of string box
     */
    private fun bowAngle(bowLine: MutableList<Double>, verticalLines: MutableList<MutableList<Double>>): Int {
        // grab bow line and vertical lines
        val m_bow: Double = bowLine[0]
        val m1 = verticalLines[0][0]
        val m2 = verticalLines[1][0]  // assuming format: [m1, b1, m2, b2]

        // calculate angles formed for each vertical line's intersection with bow line
        val angle_one: Double = abs(degrees(atan(abs(m_bow - m2) / (1 + m_bow * m2))))
        val angle_two: Double = abs(degrees(atan(abs(m1 - m_bow) / (1 + m1 * m_bow))))

        val min_angle: Double = min(abs(90 - min(angle_one, angle_two)), min(angle_one, angle_two))
        Log.d("angle1", min_angle.toString())
        //println("ANGLE: $min_angle")
        return if (min_angle > maxAngle) 1 else 0  // 1 = Wrong Angle, 0 = Correct Angle
    }

    data class returnBow(

        var classification: Int?,
        var bow: List<Point>?,
        var string: List<Point>?,
        var angle: Int?
    )

    data class bitmapAndClassifications(
        var bitmap: Bitmap,
        var height: Int?,
        var angle: Int?
    )


    fun classify(results: YoloResults): returnBow {
        val classResults = returnBow(
            classification = null,
            bow = null,
            string = null,
            angle = null
        )
        if (results.stringResults != null) {
            stringRepeat = 0
            stringPoints = results.stringResults
        } else if (stringRepeat < 5 && stringPoints != null) {
            classResults.classification = -1
            stringRepeat++
            results.stringResults = stringPoints!!.toMutableList()
        } else {
            stringPoints = null
        }
        if (results.bowResults != null) {
            bowRepeat = 0
            bowPoints = results.bowResults
        } else if (bowRepeat < 5 && bowPoints != null) {
            classResults.classification = -1
            bowRepeat++
            results.bowResults = bowPoints!!.toMutableList()
        } else {
            bowPoints = null
        }
        if (stringPoints == null && bowPoints == null) {
            classResults.classification = -2
            return classResults
        }
        if (results.stringResults == null) {
            classResults.classification = -1
            classResults.bow = results.bowResults
            return classResults
        } else if (results.bowResults == null) {
            classResults.classification = -1
            classResults.string = results.stringResults
            return classResults
        } else {
            classResults.string = results.stringResults
            classResults.bow = results.bowResults
            updatePoints(results.stringResults!!, results.bowResults!!)
            val midlines = getMidline()
            val vert_lines = getVerticalLines()
            val intersect_points = intersectsVertical(midlines, vert_lines)
            if (intersect_points != -1 && intersect_points != 1) {
                classResults.angle = bowAngle(midlines, vert_lines)
            } else {
                classResults.angle = -1
            }
            classResults.classification = intersect_points
            Log.d("BOW", classResults.classification.toString())
            return classResults
        }
    }

    interface DetectorListener {
        fun noDetect()
        fun detected(results: YoloResults, sourceWidth: Int, sourceHeight: Int)
    }

}
