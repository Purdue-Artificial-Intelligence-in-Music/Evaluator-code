package expo.modules.videoanalyzer

import android.content.Context
import android.gesture.OrientedBoundingBox
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tflite.java.TfLite
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.FileUtil
import java.util.concurrent.CountDownLatch
import kotlin.math.*
// import org.opencv.core.Point
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import android.os.SystemClock
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF

import org.tensorflow.lite.Interpreter

class Detector(private val context: Context) {

    private var interpreter: Interpreter
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
    private val numWaitFrames = 10
    private var ogWidth: Int = 0
    private var ogHeight: Int = 0



    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()
    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.5F
    }

    init {
        val options = Interpreter.Options().apply{
            //this.setNumThreads(8)
            if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                this.addDelegate(GpuDelegate(CompatibilityList().bestOptionsForThisDevice))
            } else {
                this.setNumThreads(4)
                this.setUseXNNPACK(true)
            }
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
        println("output shape")
        for (x in outputShape!!) {
            println(x)
        }

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

        println("Numelements, numchannel: $numElements, $numChannel")

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


    fun detect(frame: Bitmap): YoloResults{

        ogWidth = frame.width
        ogHeight = frame.height
        var inferenceTime = SystemClock.uptimeMillis()
        var results = YoloResults(null, null)
        if (tensorWidth == 0
            || tensorHeight == 0
            || numChannel == 0
            || numElements == 0) return results

        println(tensorWidth)
        println(tensorHeight)
        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer
        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)

        interpreter.run(imageBuffer, output.buffer)
        for (x in output.shape) {
            println(x)
        }

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

        for (box in bestBoxes) {
            if (box.cls == 0 && box.conf > bowConf) {
                results.bowResults = rotatedRectToPoints(box.x, box.y, box.width, box.height, box.angle).toMutableList()
                bowConf = box.conf
            } else if (box.cls == 1 && box.conf > stringConf) {
                results.stringResults = rotatedRectToPoints(box.x, box.y, box.width, box.height, box.angle).toMutableList()
                stringConf = box.conf
            }
        }
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        println("TRUE INFERENCE TIME: $inferenceTime")
        println("NUMBER OF BOXES: ${bestBoxes.size}")
        println("bow conf, string conf: $bowConf, $stringConf")
        return results
    }



    fun drawPointsOnBitmap(
        sourceBitmap: Bitmap,
        points: YoloResults,
        color: Int = Color.RED,
        radius: Float = 10f
    ): Bitmap {
        val resultBitmap = sourceBitmap.copy(Bitmap.Config.ARGB_8888, true)

        val canvas = Canvas(resultBitmap)

        val paint = Paint().apply {
            this.color = color
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        if (points.bowResults != null) {
            for (point in points.bowResults!!) {
                canvas.drawCircle(point.x.toFloat(), point.y.toFloat(), radius, paint)
            }
        }
        if (points.stringResults != null) {
            for (point in points.stringResults!!) {
                canvas.drawCircle(point.x.toFloat(), point.y.toFloat(), radius, paint)
            }
        }

        return resultBitmap
    }



    private fun rotatedRectToPoints(cx: Float, cy: Float, w: Float, h: Float, angleRad: Float): List<Point> {
        val halfW = w / 2
        val halfH = h / 2
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
            Point(xRot.toDouble() * ogWidth, yRot.toDouble() * ogHeight)
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
                val h = array[2 * numElements + r]
                val w = array[3 * numElements + r]

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

        if (!yLocked) {
            //just assign class variable string points to this
            stringPoints = stringBox
        } else if (yAvg != null) {
            //first sort strings if y is locked
            val sortedString = sortStringPoints(stringBox)
            stringPoints = sortedString

            stringPoints!![0].y = yAvg!![0]
            stringPoints!![1].y = yAvg!![1]

            println("y_avg: $yAvg")
            println("string_points: $stringPoints")
        }

    }

    fun sortStringPoints(pts: MutableList<Point>): MutableList<Point> {
        // Sort points by y
        val sortedPoints = pts.sortedBy {it.y }

        // Find first 2 and last pts
        val topPoints = sortedPoints.take(2).sortedBy { it.x }      // Sort by X ascending
        val bottomPoints = sortedPoints.drop(2).sortedByDescending { it.x } // Sort by X descending

        return (topPoints + bottomPoints).toMutableList()
    }

    fun getMidline(): MutableList<Int> {
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
            mutableListOf(Float.POSITIVE_INFINITY.toInt(), mid1[0].toInt())
        } else {
            val slope = dy / dx
            val intercept = mid1[1] - slope * mid1[0]
            mutableListOf(slope.toInt(), intercept.toInt())
        }
    }

    private fun getVerticalLines(): MutableList<MutableList<Double>> {
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
            leftYint = topLeft.y - leftSlope * topLeft.y
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
            rightYint = topRight.y - rightSlope * topRight.y
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
        linearLine: MutableList<Int>,
        verticalLines: MutableList<MutableList<Double>>
    ): Int {
        println("linear: $linearLine\nvertical: $verticalLines")

        // Midline parameters
        val m = linearLine[0].toDouble() // slope of the midline
        val b = linearLine[1].toDouble() // y-intercept of the midline

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

            if (y !in yMin..yMax) {
                println("Intersection y=$y is outside vertical range ($yMin, $yMax)")
                return null
            }

            return Point(x,y)
        }

        // Determine x positions from the bounding box
        val xLeft = stringPoints!![0].x
        val xRight = stringPoints!![1].x

        // Calculate intersections of midline with both vertical string lines
        val pt1 = getIntersection(verticalOne, xLeft)
        val pt2 = getIntersection(verticalTwo, xRight)

        if (pt1 == null || pt2 == null) {
            println("One or both intersections invalid")
            return 1
        }
        return bowHeightIntersection(mutableListOf(pt1, pt2), mutableListOf(verticalOne, verticalTwo))
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
        val bot_scaling_factor = .15
        val top_scaling_factor = .15

        // Extracts the vertical lines from the list
        val vertical_one = verticalLines[0]
        val vertical_two = verticalLines[1]

        // Extracts the bottom y-coordinates of the vertical lines
        val bot_y1 = vertical_one[3]
        val bot_y2 = vertical_two[3]

        // Extracts the top y-coordinates of the vertical lines
        val top_y1 = vertical_one[2]
        val top_y2 = vertical_two[2]

        // Calculates the height of the bow based on the vertical lines
        val height = ((bot_y1 - top_y1) + (bot_y2 - top_y2)) / 2.0

        // Calculates the minimum and maximum y-coordinates for the intersections based on the scaling factors
        val min_y = ((top_y1 + top_y2) / 2) + height * top_scaling_factor
        if (intersectionPoints[0].y <= min_y || intersectionPoints[1].y <= min_y) {
            return 2
        }

        val max_y = ((bot_y1 + bot_y2) / 2) - height * bot_scaling_factor
        if (intersectionPoints[0].y >= max_y || intersectionPoints[1].y >= max_y) {
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
        val sortedCoords: MutableList<Point> = sortStringPoints(stringBoxCoords)

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
    private fun bowAngle(bowLine: MutableList<Int>, verticalLines: MutableList<MutableList<Double>>): Int {
        // flexibility of angle relative to 90 degrees
        val max_angle = 30

        // grab bow line and vertical lines
        val m_bow: Double = bowLine[0].toDouble()
        val m1 = verticalLines[0][0]
        val m2 = verticalLines[1][0]  // assuming format: [m1, b1, m2, b2]

        // calculate angles formed for each vertical line's intersection with bow line
        val angle_one: Double = abs(degrees(atan(abs(m_bow - m2) / (1 + m_bow * m2))))
        val angle_two: Double = abs(degrees(atan(abs(m1 - m_bow) / (1 + m1 * m_bow))))

        val min_angle: Double = abs(90 - min(angle_one, angle_two))
        println("ANGLE: $min_angle")

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
        stringPoints = results.stringResults
        bowPoints = results.bowResults
        if (stringPoints == null && bowPoints == null) {
            classResults.classification = -2
            println("class results: $classResults")
            return classResults
        }
        if (results.stringResults != null) {
            results.stringResults = sortStringPoints(results.stringResults!!)
            //need to do averaging of top two y coords
            classResults.string = results.stringResults
            averageYCoordinate(results.stringResults!!)
            if (results.bowResults == null && bowRepeat < 5 && bowPoints != null) {
                classResults.classification = -1
                bowRepeat++
                classResults.bow = bowPoints
            }
            if (results.bowResults != null) {
                classResults.bow = results.bowResults
                if (results.stringResults == null && stringRepeat < 5 && stringPoints != null) {
                    classResults.classification = -1
                    stringRepeat++
                    classResults.string = stringPoints
                }
            }
            if (results.bowResults != null && results.stringResults != null) {
                updatePoints(results.stringResults!!, results.bowResults!!)
                var midlines = getMidline()
                var vert_lines = getVerticalLines()
                var intersect_points = intersectsVertical(midlines, vert_lines)
                classResults.angle = bowAngle(midlines, vert_lines)
                classResults.classification = intersect_points
                println("class results: $classResults")
                return classResults

            } else {
                classResults.classification = -1
                println("class results: $classResults")
                return classResults
            }
        } else if (results.bowResults != null) {
            classResults.bow = results.bowResults
            classResults.classification = -1
            return classResults
        } else {
            classResults.classification = -2
            return classResults
        }
    }

    fun process_frame(bitmap: Bitmap): returnBow {
        return classify(detect(bitmap))
    }

    fun process_bitmap(bitmap: Bitmap): Bitmap {
        return drawPointsOnBitmap(bitmap, detect(bitmap))
    }
}