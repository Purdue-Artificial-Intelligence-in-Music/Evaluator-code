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
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.util.concurrent.CountDownLatch
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import kotlin.math.*
import android.graphics.Typeface


class Detector (
    private val context: Context,
    private val listener: DetectorListener? = null
){

    private var interpreter: Interpreter
    private var nnApiDelegate: NnApiDelegate? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    val modelReadyLatch = CountDownLatch(1)
    private var bowRepeat = 0
    private var stringRepeat = 0
    private var bowPoints: List<Point>? = null
    private var stringPoints: List<Point>? = null
    private var yLocked = false
    private var yAvg: MutableList<Double>? = null
    private var frameCounter = 0
    private var stringYCoordHeights: MutableList<List<Int>> = mutableListOf()
    private val numWaitFrames = 5
    private var ogWidth: Int = 0
    private var ogHeight: Int = 0
    //private var ogWidth: Int = 1
    //private var ogHeight: Int = 1




    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.1F
    }

    init {
        val options = Interpreter.Options().apply{
            //this.setNumThreads(4)
            //this.setUseXNNPACK(true)
            //Log.i("Detector", "isDelegateSupportedOnThisDevice: ${CompatibilityList().isDelegateSupportedOnThisDevice}")
            //this.addDelegate(GpuDelegate(CompatibilityList().bestOptionsForThisDevice))
            /*
            try {
                this.addDelegate(GpuDelegate(CompatibilityList().bestOptionsForThisDevice))
            } catch (e: Exception) {
                println("Gpu delegate failed")
            }
            */

            //this.addDelegate(GpuDelegate(CompatibilityList().bestOptionsForThisDevice))

            if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                this.addDelegate(GpuDelegate(CompatibilityList().bestOptionsForThisDevice))
            } else {
                this.setNumThreads(4)
                this.setUseXNNPACK(true)
            }






            //this.setNumThreads(4)
        }

        /*
        interpreter = Interpreter.create(
            FileUtil.loadMappedFile(
                MainActivity.applicationContext(),
                "nano_best_float32.tflite"
            ),
            options
        )
         */
        val model = FileUtil.loadMappedFile(context, "best_nano_float16.tflite")
        interpreter = Interpreter(model, options)

        modelReadyLatch.countDown()

        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape = interpreter.getOutputTensor(0)?.shape()
        /*
        println("output shape")
        for (x in outputShape!!) {
            println(x)
        }

         */



        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]

            // If in case input shape is in format of [1, 3, ..., ...]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        if (outputShape != null) {
            numElements = outputShape[2]
            numChannel = outputShape[1]
        }

        //println("Numelements, numchannel: $numElements, $numChannel")

    }
    fun close() {
        interpreter.close()
    }

    data class YoloResults(
        var bowResults: MutableList<Point>?,
        var stringResults: MutableList<Point>?
    )

    data class Point(
        var x: Double,
        var y: Double
    )
    /*
    fun setDimensions(dims: Pair<Int, Int>) {
        ogWidth = dims.first
        ogHeight = dims.second
        println("dims: $dims")
    }

     */


    fun detect(frame: Bitmap, sourceWidth: Int = 1, sourceHeight: Int = 1): YoloResults{
        Log.d("DIMENSIONS WIDTH", frame.width.toString())
        Log.d("DIMENSIONS HEIGHT", frame.height.toString())
        //ogWdith = frame.width
        //ogHeight = frame.height
        var inferenceTime = SystemClock.uptimeMillis()
        var results = YoloResults(null, null)
        if (tensorWidth == 0
            || tensorHeight == 0
            || numChannel == 0
            || numElements == 0) println("MODEL ERROR")

        //println(tensorWidth)
        //println(tensorHeight)
        Log.d("TENSOR DIMS", tensorWidth.toString() + " " + tensorHeight.toString())
        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer
        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)

        interpreter.run(imageBuffer, output.buffer)



        val bestBoxes = newBestBox(output.floatArray)
        /*
        val newBoxes = mutableListOf<PointF>()
        for (box in bestBoxes) {
            val points = rotatedRectToPoints(box.x, box.y, box.width, box.height, box.angle)
            newBoxes.addAll(points)
        }
        */

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
//                var newBowResults = results.bowResults
//                for (i in 1..4) {
//                    //newBowResults!![i].x = newBowResults!![i].x * ogWidth
//                    newBowResults!![i].y = newBowResults!![i].y * ogHeight/ogWidth
//                }
//                results.bowResults = newBowResults
                Log.d("BOX BOW", results.bowResults.toString())

            } else if (box.cls == 1 && box.conf > stringConf) {
                Log.d("BOX INITAL STRING", box.toString())
                results.stringResults = sortStringPoints(rotatedRectToPoints(box.x, box.y, box.width, box.height, box.angle, ogWidth, ogHeight).toMutableList())
                /*
                if (box.width > box.height) {
                    results.stringResults = sortStringPoints(rotatedRectToPoints(box.x * ogWidth, box.y * ogHeight, box.width * ogWidth, box.height * ogHeight, box.angle + Math.PI.toFloat() / 2).toMutableList())
                } else {
                    results.stringResults = sortStringPoints(rotatedRectToPoints(box.x * ogWidth, box.y * ogHeight, box.width * ogWidth, box.height * ogHeight, box.angle).toMutableList())
                }
                 */
//                var newBowResults = results.stringResults
//                for (i in 1..4) {
//                    //newBowResults!![i].x = newBowResults!![i].x * ogWidth
//                    newBowResults!![i].y = newBowResults!![i].y * ogHeight/ogWidth
//                }
//                results.stringResults = newBowResults
                stringConf = box.conf
                Log.d("BOX STRING", results.stringResults.toString())
            }
        }
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        //println("TRUE INFERENCE TIME: $inferenceTime")
        //println("NUMBER OF BOXES: ${bestBoxes.size}")
        //println("bow conf, string conf: $bowConf, $stringConf")

        if (results.bowResults == null && results.stringResults == null) {
            listener?.noDetect()
            Log.d("BOW RESULTS", "NO DETECTIONS")
        } else {
            Log.d("BOXES123", "SOMETHING DETECTED")
            //println(results)
            listener?.detected(results, frame.width, frame.height)
            //print(results)
        }
        return results
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

        // Draw string box (rectangle)
        if (points.stringResults != null && points.stringResults!!.size >= 4) {
            val stringBox = points.stringResults!!
            // Draw four lines connecting the corners
            canvas.drawLine(
                stringBox[0].x.toFloat(), stringBox[0].y.toFloat(),
                stringBox[1].x.toFloat(), stringBox[1].y.toFloat(),
                paint
            )
            canvas.drawLine(
                stringBox[1].x.toFloat(), stringBox[1].y.toFloat(),
                stringBox[2].x.toFloat(), stringBox[2].y.toFloat(),
                paint
            )
            canvas.drawLine(
                stringBox[2].x.toFloat(), stringBox[2].y.toFloat(),
                stringBox[3].x.toFloat(), stringBox[3].y.toFloat(),
                paint
            )
            canvas.drawLine(
                stringBox[3].x.toFloat(), stringBox[3].y.toFloat(),
                stringBox[0].x.toFloat(), stringBox[0].y.toFloat(),
                paint
            )
        }

        // Draw bow box (rectangle)
        if (points.bowResults != null && points.bowResults!!.size >= 4) {
            val bowBox = points.bowResults!!
            // Draw four lines connecting the corners
            canvas.drawLine(
                bowBox[0].x.toFloat(), bowBox[0].y.toFloat(),
                bowBox[1].x.toFloat(), bowBox[1].y.toFloat(),
                paint
            )
            canvas.drawLine(
                bowBox[1].x.toFloat(), bowBox[1].y.toFloat(),
                bowBox[2].x.toFloat(), bowBox[2].y.toFloat(),
                paint
            )
            canvas.drawLine(
                bowBox[2].x.toFloat(), bowBox[2].y.toFloat(),
                bowBox[3].x.toFloat(), bowBox[3].y.toFloat(),
                paint
            )
            canvas.drawLine(
                bowBox[3].x.toFloat(), bowBox[3].y.toFloat(),
                bowBox[0].x.toFloat(), bowBox[0].y.toFloat(),
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
    }



    private fun rotatedRectToPoints(cx: Float, cy: Float, w: Float, h: Float, angleRad: Float, frameWidth: Float, frameHeight: Float): List<Point> {
        val halfW = w / 2
        val halfH = h / 2
        println("ANGLE $angleRad")
        val cosA = cos(angleRad - Math.PI.toFloat() / 2)
        val sinA = sin(angleRad - Math.PI.toFloat() / 2)
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
            Point(xRot.toDouble(), yRot.toDouble())
        }
    }

    private fun newBestBox(array : FloatArray) : List<OrientedBoundingBox> {
        val boundingBoxes = mutableListOf<OrientedBoundingBox>()

        for (r in 0 until numElements) {
            val stringCnf = array[5 * numElements + r]
            val bowCnf = array[4 * numElements + r]
            val cls = if (stringCnf > bowCnf) 1 else 0
            val cnf = if (stringCnf > bowCnf) stringCnf else bowCnf
            if (cnf > CONFIDENCE_THRESHOLD) {
                val x = array[r]
                val y = array[1 * numElements + r]
                var h = array[2 * numElements + r]
                var w = array[3 * numElements + r]

                val angle = array[6 * numElements + r]
                boundingBoxes.add(
                    OrientedBoundingBox(
                        x = x, y = y, height = h, width = w,
                        conf = cnf, cls = cls, angle = angle
                    )
                )
            }
        }


        return boundingBoxes
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
        stringPoints = sortStringPoints(stringBox)
        Log.d("sorted points", stringPoints.toString())
        /*
        if (!yLocked) {
            //just assign class variable string points to this
            stringPoints = stringBox
        } else if (yAvg != null) {
            //first sort strings if y is locked
            val sortedString = sortStringPoints(stringBox)
            stringPoints = sortedString
            Log.d("sorted points", stringPoints.toString())

            //stringPoints!![0].y = yAvg!![0]
            //stringPoints!![1].y = yAvg!![1]

            //println("y_avg: $yAvg")
            //println("string_points: $stringPoints")
        }

         */

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

        if (pt1 == null && pt2 == null) {
            //println("One or both intersections invalid")
            Log.d("BOW", "INVALID INTERSECTION")
            return 1
        }
        if (pt1 == null) {
            pt1 = pt2
        }
        if (pt2 == null){
            pt2 = pt1
        }
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
        val top_zone_percentage = 0.3
        val bottom_zone_percentage = 0.15

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

    private fun averageYCoordinate(stringBoxCoords: MutableList<Point>) {
        /*
        Recalculate and updates new string Y coordinate heights for every
        n frames.
         */

        // sort coordinate points so points in consistent order
        val sortedCoords: MutableList<Point> = stringBoxCoords

        // increase frame counter. create and add y coord list of each frame to one list
        frameCounter += 1
        val yCoords = sortedCoords.map { it.y.toInt() } // get Y values from each point
        stringYCoordHeights.add(yCoords)


        // recalculate new y coordinate heights when certain number of frames past
        if (frameCounter % numWaitFrames == 0) {

            // get median of coordinates
            val topLeftAvg: Double = median(stringYCoordHeights.map { it[0] })
            val topRightAvg: Double = median(stringYCoordHeights.map { it[1] })
            val botRightAvg: Double = median(stringYCoordHeights.map { it[2] })
            val botLeftAvg: Double = median(stringYCoordHeights.map { it[3] })

            // map them to string points
            stringPoints!![0].y = topLeftAvg
            stringPoints!![1].y = topRightAvg
            stringPoints!![2].y = botRightAvg
            stringPoints!![3].y = botLeftAvg
            yAvg = mutableListOf(topLeftAvg, topRightAvg)
            yLocked = true
            stringYCoordHeights = mutableListOf()
        }
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
        // flexibility of angle relative to 90 degrees
        val max_angle = 15

        // grab bow line and vertical lines
        val m_bow: Double = bowLine[0]
        val m1 = verticalLines[0][0]
        val m2 = verticalLines[1][0]  // assuming format: [m1, b1, m2, b2]

        // calculate angles formed for each vertical line's intersection with bow line
        val angle_one: Double = abs(degrees(atan(abs(m_bow - m2) / (1 + m_bow * m2))))
        val angle_two: Double = abs(degrees(atan(abs(m1 - m_bow) / (1 + m1 * m_bow))))

        val min_angle: Double = abs(90 - min(angle_one, angle_two))
        //println("ANGLE: $min_angle")

        return if (min_angle > max_angle) 1 else 0  // 1 = Wrong Angle, 0 = Correct Angle
    }

    data class returnBow(
        var classification: Int?,
        var bow: List<Point>?,
        var string: List<Point>?,
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
            classResults.angle = bowAngle(midlines, vert_lines)
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