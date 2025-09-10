package com.example.evaluator_kotlin

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tflite.java.TfLite
//import org.opencv.core.*
//import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.FileUtil
import java.util.concurrent.CountDownLatch
import kotlin.math.*
//import org.opencv.core.Point
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class Evaluator {
    /*
    val initializeTask: Task<Void> by lazy {
        Log.d("InitDebug", "TfLite.initialize() is being called")
        TfLite.initialize(MainActivity.applicationContext())
    }
    val modelReadyLatch = CountDownLatch(1)
    private lateinit var model: InterpreterApi
    private var bowRepeat = 0
    private var stringRepeat = 0
    private var bowPoints: List<Point>? = null
    private var stringPoints: List<Point>? = null
    private var yLocked = false
    private var yAvg: MutableList<Double>? = null
    private var frameCounter = 0
    private var stringYCoordHeights: MutableList<List<Int>> = mutableListOf()
    private val numWaitFrames = 10

    fun createInterpreter(context: Context) {
        initializeTask.addOnSuccessListener {
            Log.d("InitDebug", "SuccessListener is being called")

            val interpreterOption =
                InterpreterApi.Options()
                    .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            model = InterpreterApi.create(
                FileUtil.loadMappedFile(
                    MainActivity.applicationContext(),
                    "nano_best_float32.tflite"
                ),
                interpreterOption
            )
            modelReadyLatch.countDown()
        }
            .addOnFailureListener { e ->
                Log.e("Interpreter", "Cannot initialize interpreter", e)
            }
    }

    /* takes in original image, a set size (default 640x640 p)
     * Rescales the image while maintaining aspect ratio, then pads to be square
     * Returns the padding and new image
     */
    fun letterbox(img: Mat, newShape: Size = Size(640.0, 640.0)): Pair<Mat, Pair<Double, Double>> {
        val shape = Size(img.width().toDouble(), img.height().toDouble()) // get original image
        // calculates scaling factor (r) to preserve aspect ratio                      
        val r = min(newShape.width / shape.width, newShape.height / shape.height)

        // calculates new unpadded shape
        val newUnpad = Size(round(shape.width * r), round(shape.height * r))

        // computes padding on width and height to center the image
        val dw = (newShape.width - newUnpad.width) / 2
        val dh = (newShape.height - newUnpad.height) / 2

        // resizes image to new unpadded dimensions
        val resized = Mat()
        Imgproc.resize(img, resized, newUnpad)

        // computes border sizes 
        val top = round(dh - 0.1).toInt()
        val bottom = round(dh + 0.1).toInt()
        val left = round(dw - 0.1).toInt()
        val right = round(dw + 0.1).toInt()

        // pads the resized image with constant, gray border 
        val padded = Mat()
        Core.copyMakeBorder(
            resized,
            padded,
            top,
            bottom,
            left,
            right,
            Core.BORDER_CONSTANT,
            Scalar(114.0, 114.0, 114.0)
        )

        // padded image is returned + padding ratio (used to reverse transformation)
        return Pair(
            padded,
            Pair(top / padded.height().toDouble(), left / padded.width().toDouble())
        )
    }

    /*
     * preprocesses the given image. Runs letterbox (rescales/packs).
     * Converts the image to RGB
     * Converts the image to float32 and normalizes it
     * Returns the preprocessed image to be ran on the model
     */
    fun preprocess(img: Mat, newShape: Size = Size(640.0, 640.0)):
            Pair<Mat, Pair<Double, Double>> {
        val (letterboxed, pad) = letterbox(img, newShape)
        val resizedImg = letterboxed
        val rgb = Mat()
        Imgproc.cvtColor(letterboxed, rgb, Imgproc.COLOR_BGR2RGB)
        val floatImg = Mat()
        rgb.convertTo(floatImg, CvType.CV_32FC3, 1.0 / 255.0)
        return Pair(floatImg, pad)
    }

    /*
     * postprocessing of the output
     * requires the original image, the results, resized shape, and padding (from letterbox)
     *
     */
    fun postprocess(
        origImg: Mat,
        outputs: Array<FloatArray>, // shape: [N, 7] (cx, cy, h, w, conf, cls, angle)
        resizedShape: Size,
        pad: Pair<Double, Double>
    ): List<List<Float>> {
        val results = mutableListOf<List<Float>>()
        val targetH = origImg.height()
        val targetW = origImg.width()
        val r = min(resizedShape.height / targetH, resizedShape.width / targetW)
        val targetScale = max(targetH, targetW).toFloat()

        for (out in outputs) {
            // Adjust coordinates
            val cx = targetScale * (out[0] - pad.second)
            val cy = targetScale * (out[1] - pad.first)
            val w = targetScale * out[3]
            val h = targetScale * out[2]
            val conf = out[4]
            val cls = out[5]
            val angleRad = out[6]

            // Convert to 4 corner points (rotated rectangle)
            val points = rotatedRectToPoints(cx.toFloat(), cy.toFloat(), w, h, angleRad)
            // Add confidence to result
            val result = points.map { listOf(it.first, it.second) }.flatten().toMutableList()
            result.add(conf)
            result.add(cls)
            results.add(result)
        }
        return results
    }

    // Helper for rotated rectangle to points
    fun rotatedRectToPoints(
        cx: Float,
        cy: Float,
        w: Float,
        h: Float,
        angleRad: Float
    ): List<Pair<Float, Float>> {
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
            Pair(xRot, yRot)
        }
    }

    fun runModel(frame: Mat): List<List<Float>> {
        // Create input and output buffers
        model.allocateTensors()

        // Prepare input buffer
        val inputBuffers = model.getInputTensor(0)
        val inputShape = inputBuffers.shape()
        val inputDataType = inputBuffers.dataType()
        val targetHeight = inputShape[1]
        val targetWidth = inputShape[2]

        val preprocessed = preprocess(
            frame, Size(
                targetHeight.toDouble(),
                targetWidth.toDouble()
            )
        )

        val preprocessedImg = preprocessed.first

        val input =
            Array(targetHeight) { FloatArray(targetWidth) } // Example for classification output

        val inputArray = Array(1) { Array(targetHeight) { Array(targetWidth) { FloatArray(3) } } }

        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val pixel = preprocessedImg.get(y, x) // [R, G, B]
                inputArray[0][y][x][0] = (pixel[0]).toFloat() // R
                inputArray[0][y][x][1] = (pixel[1]).toFloat() // G
                inputArray[0][y][x][2] = (pixel[2]).toFloat() // B
            }
        }

        // Run inference
        val outputBuffers = model.getOutputTensor(0)
        val outputShape = outputBuffers.shape()
        val outputDataType = outputBuffers.dataType()

        val outputArray =
            Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } } // (1, 300, 7)


        model.run(inputArray, outputArray)
        val outputs = outputArray[0] // shape: [N, 7]
        // Convert output to Array<FloatArray>
        val numDetections = outputs.size / 7
        val output = Array(numDetections) { i ->
            outputs.sliceArray(i * 7 until (i + 1) * 7)
        }

        // Postprocess
        //        origImg: Mat,
        //        outputs: Array<FloatArray>, // shape: [N, 7] (cx, cy, h, w, conf, cls, angle)
        //        resizedShape: Size,
        //        pad: Pair<Double, Double>
        val results = postprocess(
            origImg = frame,
            outputs = outputs,
            resizedShape = Size(targetHeight.toDouble(), targetWidth.toDouble()),
            pad = preprocessed.second
        )
        //println("Detections: $results")
        println("Detections: ${convertYolo(results)}")

        return results
    }

    // Draws a detection polygon (usually quadrilateral) on an img with class based coloring
    fun drawDetections(img: Mat, box: List<Point>, score: Float, classId: Int) {
        // Define color palette and class labels
        val colorPalette = mapOf(
            1 to Scalar(255.0, 255.0, 100.0), // BGR format
            0 to Scalar(100.0, 255.0, 255.0)
        )
        // Define class ID to label mapping 
        val classes = mapOf(
            1 to "bow",
            0 to "string"
        )

        val color = colorPalette[classId] ?: Scalar(255.0, 255.0, 255.0) // fallback color (white)

        // Draw bounding box if the box contains 4 points
        if (box.size == 4) {
            // converts list of Points to OpenCV MatOfPoint
            val pointsArray = MatOfPoint(*box.map { Point(it.x, it.y) }.toTypedArray())
            Imgproc.polylines(img, listOf(pointsArray), true, color, 2)
        }
    }

    data class BoxResults(
        var classification: Int?,
        var box: MutableList<MutableList<Int>>?,
        var angle: Int?
    )

    data class YoloResults(
        var bowResults: MutableList<Point>?,
        var stringResults: MutableList<Point>?
    )


    fun convertYolo(results: List<List<Float>>): YoloResults {
        //Results are always this format: [[x1, y1, x2, y2, x3, y3, x4, y4, conf, cls], ...]
        //First two will be highest conf bow and string boxes, will be two of same class if other is not detected
        val yoloResults = YoloResults(
            bowResults = null,
            stringResults = null
        )
        var coordList = mutableListOf<Point>()
        for (i in 0 until 4) {
            coordList.add(
                Point(
                    results[0][2 * i].toDouble(),
                    results[0][2 * i + 1].toDouble()
                )
            )
        }
        if (results[0][9] == 1.0f) {
            yoloResults.stringResults = coordList
        } else {
            yoloResults.bowResults = coordList
        }
        if (results.size > 1 && results[1][9] != results[0][9]) {
            coordList = mutableListOf<Point>()
            for (i in 0 until 4) {
                coordList.add(
                    Point(
                        results[1][2 * i].toDouble(),
                        results[1][2 * i + 1].toDouble()
                    )
                )
            }
            if (results[1][9] == 1.0f) {
                yoloResults.stringResults = coordList
            } else {
                yoloResults.bowResults = coordList
            }

        }
        return yoloResults
    }

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

            stringPoints!![0].y = yAvg!![0].toDouble()
            stringPoints!![1].y = yAvg!![1].toDouble()

            println("y_avg: $yAvg")
            println("string_points: $stringPoints")
        }

    }

    fun sortStringPoints(pts: MutableList<Point>): MutableList<Point> {
        // Sort points by y
        val sortedPoints = pts.sortedBy { it.y }

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
        //val bot_scaling_factor = .25
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
        if (intersectionPoints[0].y >= min_y || intersectionPoints[1].y >= min_y) {
            return 2
        }

        //val max_y = ((bot_y1 + bot_y2) / 2) - height * bot_scaling_factor
        //if (intersectionPoints[0].y <= max_y or intersectionPoints[1][1].toDouble() <= max_y):
        //return 3

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
        val max_angle = 15

        // grab bow line and vertical lines
        val m_bow: Double = bowLine[0].toDouble()
        val m1 = verticalLines[0][0]
        val m2 = verticalLines[1][0]  // assuming format: [m1, b1, m2, b2]

        // calculate angles formed for each vertical line's intersection with bow line
        val angle_one: Double = abs(degrees(atan(abs(m_bow - m2) / (1 + m_bow * m2))))
        val angle_two: Double = abs(degrees(atan(abs(m1 - m_bow) / (1 + m1 * m_bow))))

        val min_angle: Double = min(angle_one, angle_two)

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
            return classResults
        }
        if (results.stringResults != null) {
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
                return classResults

            } else {
                classResults.classification = -1
                return classResults
            }
        }
        return classResults
    }

     */
}
