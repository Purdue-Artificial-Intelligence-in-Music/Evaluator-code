import org.opencv.core.Mat
import kotlin.math.*

class kotlin_bow {

    private var frameNum: Int = 0
    private val stringYCoordHeights: MutableList<Int> = mutableListOf()
    private var bowPoints: MutableList<MutableList<Int>> = mutableListOf(
        mutableListOf(0, 0),
        mutableListOf(0, 0),
        mutableListOf(0, 0),
        mutableListOf(0, 0)
    )    
    private var stringPoints: MutableList<MutableList<Int>> = mutableListOf(
        mutableListOf(0, 0),
        mutableListOf(0, 0),
        mutableListOf(0, 0),
        mutableListOf(0, 0)
    )    
    private var yLocked: Boolean = false
    private var frameCounter: Int = 0
    private val numWaitFrames: Int = 11
    private var yAvg: MutableList<Int> = mutableListOf(0, 0)
    private var bowRepeat: Int = 0
    private var stringRepeat: Int = 0
    // TODO: Add TFLite model once available
    

    private fun update_points(stringBox: MutableList<Int>, bowBox: MutableList<Int>) {
    }

    private fun get_midline(): MutableList<Int> {
    }

   private fun get_vertical_lines(): MutableList<MutableList<Double>> {
    // Extracting corner points
    val topLeft = stringPoints[0]
    val topRight = stringPoints[1]
    val botRight = stringPoints[2]
    val botLeft = stringPoints[3]

    // Left vertical line (from topLeft to botLeft)

    // Calculate the horizontal distance between the left points
    val dxLeft = topLeft[0] - botLeft[0]

    // Initialize slope and y-intercept for the left side
    val leftSlope: Double
    val leftYint: Double

    if (dxLeft == 0) {
        leftSlope = Double.POSITIVE_INFINITY
        leftYint = -1.0  // Use -1.0 as a flag for undefined intercept
    } else {
        // Calculate slope 
        leftSlope = (topLeft[1] - botLeft[1]).toDouble() / dxLeft
        // Calculate y-intercept 
        leftYint = topLeft[1] - leftSlope * topLeft[0]
    }

    // Right vertical line (from topRight to botRight) 

    val dxRight = topRight[0] - botRight[0]
    val rightSlope: Double
    val rightYint: Double

    if (dxRight == 0) {
        rightSlope = Double.POSITIVE_INFINITY
        rightYint = -1.0
    } else {
        rightSlope = (topRight[1] - botRight[1]).toDouble() / dxRight
        rightYint = topRight[1] - rightSlope * topRight[0]
    }

    // Heights of each side (just the y-coordinates of top and bottom points)
    val leftTopY = topLeft[1].toDouble()
    val leftBotY = botLeft[1].toDouble()
    val rightTopY = topRight[1].toDouble()
    val rightBotY = botRight[1].toDouble()

    // Each line is a MutableList: [slope, intercept, topY, bottomY]
    val leftLine = mutableListOf(leftSlope, leftYint, leftTopY, leftBotY)
    val rightLine = mutableListOf(rightSlope, rightYint, rightTopY, rightBotY)

    // return a list of both lines
    return mutableListOf(leftLine, rightLine)
}

private fun intersects_vertical(
    linearLine: MutableList<Int>,
    verticalLines: MutableList<Int>
): MutableList<Int> {
    println("linear: $linearLine\nvertical: $verticalLines")

    // Midline parameters
    val m = linearLine[0].toDouble() // slope of the midline
    val b = linearLine[1].toDouble() // y-intercept of the midline

    // extracts the first vertical line (left side): [slope, yInt, topY, botY]
    val verticalOne = verticalLines.subList(0, 4).map { it.toDouble() }

    // extracts the second vertical line (right side)
    val verticalTwo = verticalLines.subList(4, 8).map { it.toDouble() }

    // Calculates the intersection of the midline with a vertical line
    fun getIntersection(vLine: List<Double>, xRef: Double): Pair<Double, Double>? {
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
        val ymin = minOf(topY, botY)
        val ymax = maxOf(topY, botY)

        if (y !in ymin..ymax) {
            println("Intersection y=$y is outside vertical range ($ymin, $ymax)")
            return null
        }

        return Pair(x, y)
    }

    // Determine x positions from the bounding box 
    val xLeft = this.stringPoints[0][0].toDouble()  
    val xRight = this.stringPoints[1][0].toDouble() 

    // Calculate intersections of midline with both vertical string lines
    val pt1 = getIntersection(verticalOne, xLeft)
    val pt2 = getIntersection(verticalTwo, xRight)

    if (pt1 == null || pt2 == null) {
        println("One or both intersections invalid")
        return mutableListOf(1) 
    }

    // calls another function to evaluate the height between the intersections
    return bow_height_intersection(mutableListOf(pt1, pt2), verticalLines)
}

    private fun sortStringPoints(pts: List<Pair<Float, Float>>): List<Pair<Float, Float>> {
        // Sort points by y
        val sortedPoints = pts.sortedBy { it.second }
    
        // Find first 2 and last pts
        val topPoints = sortedPoints.take(2).sortedBy { it.first }  // Sort by X ascending
        val bottomPoints = sortedPoints.drop(2).sortedByDescending { it.first }  // Sort by X descending
    
        return topPoints + bottomPoints
}

    /*
     * Determines the height level at which the linear line intersects the vertical lines.

        Returns:
        - 3: Intersection is near top of the box (ht1 or ht2)
        - 2: Intersection is near bottom (hb1 or hb2)
        - 0: Intersection is in middle
     */

    private fun bow_height_intersection(intersectionPoints: MutableList<Pair<Double, Double>>, verticalLines: MutableList<Int>): Int {
        val bot_scaling_factor = .25
        val top_scaling_factor = .20

        // Extracts the vertical lines from the list
        val vertical_one = verticalLines[0].toDouble()
        val vertical_two = verticalLines[1].toDouble()

        // Extracts the bottom y-coordinates of the vertical lines
        val bot_y1 = vertical_one[3].toDouble()
        val bot_y2 = vertical_two[3].toDouble()
        
        // Extracts the top y-coordinates of the vertical lines
        val top_y1 = vertical_one[2].toDouble()
        val top_y2 = vertical_two[2].toDouble()

        // Calculates the height of the bow based on the vertical lines
        val height = ((bot_y1 - top_y1) + (bot_y2 - top_y2)) / 2.0

        // Calculates the minimum and maximum y-coordinates for the intersections based on the scaling factors
        val min_y = ((top_y1 + top_y2) / 2) + height * top_scaling_factor
        if (intersection_points[0][1].toDouble() >= min_y || intersection_points[1][1].toDouble() >= min_y):
            return 2

        val max_y = ((bot_y1 + bot_y2) / 2) - height * bot_scaling_factor
        if (intersection_points[0][1].toDouble() <= max_y or intersection_points[1][1].toDouble() <= max_y):
            return 3

        return 0
    }

    private fun average_y_coordinates(stringBoxCoords: MutableList<Int>) {
        /*
        Recalculate and updates new string Y coordinate heights for every
        n frames.
         */


        // sort coordinate points so points in consistent order
        val sortedCoords : MutableList<Int> = sort_string_points(stringBoxCoords)
        
        // increase frame counter. create and add y coord list of each frame to one list
        frameCounter += 1
        y_coords :  MutableList<Int> = null
        val yCoords = sortedCoords.map { it[1] } // get Y values from each point
        stringYCoordHeights.add(yCoords)

        // recalculate new y coordinate heights when certain number of frames past
        if (frameCounter % numWaitFrames == 0) {

            // get median of coordinates
            val topLeftAvg: Double = median(stringYCoordHeights.map { it[0] })
            val topRightAvg: Double = median(stringYCoordHeights.map { it[1] })
            val botRightAvg: Double = median(stringYCoordHeights.map { it[2] })
            val botLeftAvg: Double = median(stringYCoordHeights.map { it[3] })

            // map them to string points
            stringPoints[0][1] = topLeftAvg.toInt()
            stringPoints[1][1] = topRightAvg.toInt()
            stringPoints[2][1] = botRightAvg.toInt()
            stringPoints[3][1] = botLeftAvg.toInt()
            yAvg = mutableListOf(topLeftAvg, topRightAvg)
            y_locked = True
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
    private fun bow_angle(bowLine: MutableList<Int>, verticalLines: MutableList<Int>): Int {
        // flexibility of angle relative to 90 degrees
        val max_angle = 15

        // grab bow line and vertical lines
        val m_bow : Double = bowLine[0].toDouble()
        val m1 : Double = verticalLines[0].toDouble()
        val m2 : Double = verticalLines[1].toDouble()  // assuming format: [m1, b1, m2, b2]

        // calculate angles formed for each vertical line's intersection with bow line
        val angle_one : Double = abs(degrees(atan(abs(m_bow - m2) / (1 + m_bow * m2))))
        val angle_two : Double = abs(degrees(atan(abs(m1 - m_bow) / (1 + m1 * m_bow))))

        val min_angle : Double = min(angle_one, angle_two)

        return if (min_angle > max_angle) 1 else 0  // 1 = Wrong Angle, 0 = Correct Angle
    }

    fun display_classification(result: Int, opencvFrame: Mat) {
    // Maps for height and angle classification results with associated labels and colors (BGR format)
    val heightLabelMap = mapOf(
        0 to Pair("Correct Bow Height", Scalar(0.0, 255.0, 0.0)),    // Green
        1 to Pair("Outside Bow Zone", Scalar(0.0, 0.0, 255.0)),     // Red
        2 to Pair("Too Low", Scalar(0.0, 165.0, 255.0)),            // Orange
        3 to Pair("Too High", Scalar(255.0, 0.0, 0.0))              // Blue
    )

    val angleLabelMap = mapOf(
        0 to Pair("Correct Bow Angle", Scalar(0.0, 255.0, 0.0)),    // Green
        1 to Pair("Improper Bow Angle", Scalar(0.0, 0.0, 255.0))    // Red
    )

    // Extract classification results from the combined input value
    val heightResult = result / 10   // Tens place = height result
    val angleResult = result % 10    // Units place = angle result

    // Get label and color for height result; fallback to "Unknown" if not in map
    val (heightLabel, heightColor) = heightLabelMap[heightResult]
        ?: Pair("Unknown", Scalar(255.0, 255.0, 255.0))  // White

    // Get label and color for angle result; fallback to "Unknown" if not in map
    val (angleLabel, angleColor) = angleLabelMap[angleResult]
        ?: Pair("Unknown", Scalar(255.0, 255.0, 255.0))  // White

    // Draw height classification label at top-left of the image
    Imgproc.putText(
        opencvFrame, heightLabel, Point(50.0, 50.0),
        Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, heightColor, 3, Imgproc.LINE_AA
    )

    // Draw string polygon outline if 4 valid points exist
    if (stringPoints.isNotEmpty() && stringPoints.size == 4) {
        val points = stringPoints.map { Point(it[0].toDouble(), it[1].toDouble()) }
        val matOfPoints = MatOfPoint(*points.toTypedArray())
        Imgproc.polylines(opencvFrame, listOf(matOfPoints), true, heightColor, 2)
    }

    // Draw bow polygon outline if 4 valid points exist
    if (bowPoints.isNotEmpty() && bowPoints.size == 4) {
        val points = bowPoints.map { Point(it[0].toDouble(), it[1].toDouble()) }
        val matOfPoints = MatOfPoint(*points.toTypedArray())
        Imgproc.polylines(opencvFrame, listOf(matOfPoints), true, angleColor, 2)
    }

    // Draw angle classification label below the first label
    Imgproc.putText(
        opencvFrame, angleLabel, Point(50.0, 250.0),
        Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, angleColor, 3, Imgproc.LINE_AA
    )

    // Draw a curved parabola connecting the top edge of the string box
    if (stringPoints.isNotEmpty() && stringPoints.size == 4) {
        // Sort by y-value (ascending) to get top-most two points
        val sortedByY = stringPoints.sortedBy { it[1] }
        val topCandidates = sortedByY.take(2).sortedBy { it[0] }
        val topLeft = topCandidates[0]
        val topRight = topCandidates[1]

        // Calculate midpoint of the top edge
        val midX = (topLeft[0] + topRight[0]) / 2.0
        val midY = (topLeft[1] + topRight[1]) / 2.0

        // Define curvature for control point (higher curvature → deeper arc)
        val width = kotlin.math.abs(topRight[0] - topLeft[0])
        val curvature = width * 0.25
        val controlPoint = listOf(midX, midY - curvature)

        // Create points along the quadratic Bézier curve
        val curvePts = mutableListOf<Point>()
        for (i in 0..49) {
            val t = i / 49.0
            val x = ((1 - t).pow(2) * topLeft[0] + 2 * (1 - t) * t * controlPoint[0] + t.pow(2) * topRight[0]).toInt()
            val y = ((1 - t).pow(2) * topLeft[1] + 2 * (1 - t) * t * controlPoint[1] + t.pow(2) * topRight[1]).toInt()
            curvePts.add(Point(x.toDouble(), y.toDouble()))
        }

        // Draw the curve in magenta (BGR: 255, 0, 255)
        val curveMat = MatOfPoint(*curvePts.toTypedArray())
        Imgproc.polylines(opencvFrame, listOf(curveMat), false, Scalar(255.0, 0.0, 255.0), 2)
    }
}


    data class bow_results(
        var classification: Int?,
        var bow: MutableList<MutableList<Int>>? ,
        var string: MutableList<MutableList<Int>>?,
        var angle: Int?
    )
    data class yolo_results(
        var bow: MutableList<MutableList<Int>>? ,
        var string: MutableList<MutableList<Int>>?
    )

    fun process_frame(opencvFrame: Mat): bow_results {
        reuturn_bow = bow_results(
            classification = null,
            bow = null,
            string = null,
            angle = null
        )
        results = convert_to_yolo_results(model(opencvFrame)) //NOT WORKING YET
        var string_coords: MutableList<MutableList<Int>>? = results.string
        var bow_coords: MutableList<MutableList<Int>>? = results.bow
        // ASSUMING RESULTS IS AN OBJECT WITH TWO NESTED ARRAYLISTS OF INTS
        // results.bow = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        // results.string = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        // both could be null
        if (string_coords == null && bow_coords == null) {
            return_bow.classification = -2
            return return_bow
        }
        if (results.string != null) {
            //need to do averaging of top two y coords
            return_bow.string = results.string
            average_y_coordinates(results.string)
            if (results.bow == null && bowRepeat < 5 && bowPoints != null) {
                return_bow.classification = -1
                bowRepeat++
                return_bow.bow = bowPoints
            }
            if (results.bow != null) {
                return_bow.bow = results.bow
                if (results.string == null && stringRepeat < 5 && stringPoints != null) {
                    return_bow.classification = -1
                    stringRepeat++
                    return_bow.string = stringPoints
                }
            }
            if (results.bow != null && results.string != null) {
                update_points(results.string, results.bow)
                var midlines = get_midline()
                var vert_lines = get_vertical_lines()
                var intersect_points = intersects_vertical(midlines, vert_lines)
                return_bow.angle = bow_angle(midlines, vert_lines)
                return_bow.classification = intersect_points
                return return_bow

            } else {
                return_bow.classification = -1
                return return_bow
            }

        }
    }

    val model =
        CompiledModel.create(
            context.assets,
            "nano_best_float32.tflite",
            CompiledModel.Options(Accelerator.CPU),
            env,
        )

    /* takes in original image, a set size (default 640x640 p)
     * Rescales the image while maintaining aspect ratio, then pads to be square
     * Returns the padding and new image
     */
    fun letterbox(img: Mat, newShape: Size = Size(640.0, 640.0)): Pair<Mat, Pair<Double, Double>> {
        val shape = Size(img.width().toDouble(), img.height().toDouble())
        val r = min(newShape.width / shape.width, newShape.height / shape.height)
        val newUnpad = Size(round(shape.width * r), round(shape.height * r))
        val dw = (newShape.width - newUnpad.width) / 2
        val dh = (newShape.height - newUnpad.height) / 2

        val resized = Mat()
        Imgproc.resize(img, resized, newUnpad)
        val top = round(dh - 0.1).toInt()
        val bottom = round(dh + 0.1).toInt()
        val left = round(dw - 0.1).toInt()
        val right = round(dw + 0.1).toInt()
        val padded = Mat()
        Core.copyMakeBorder(resized, padded, top, bottom, left, right, Core.BORDER_CONSTANT, Scalar(114.0, 114.0, 114.0))
        return Pair(padded, Pair(top / padded.height().toDouble(), left / padded.width().toDouble()))
    }

    /*
     * preprocesses the given image. Runs letterbox (rescales/packs).
     * Converts the image to RGB
     * Converts the image to float32 and normalizes it
     * Returns the preprocessed image to be ran on the model
     */
    fun preprocess(img: Mat, newShape: Size = Size(640.0, 640.0)): Mat {
        val (letterboxed, pad) = letterbox(img, newShape)
        val rgb = Mat()
        Imgproc.cvtColor(img, rgb, Imgproc.COLOR_BGR2RGB)
        val floatImg = Mat()
        rgb.convertTo(floatImg, CvType.CV_32FC3, 1.0 / 255.0)
        return floatImg
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
            val points = rotatedRectToPoints(cx, cy, w, h, angleRad)
            // Add confidence to result
            val result = points.map { listOf(it.first, it.second) }.flatten().toMutableList()
            result.add(conf)
            result.add(cls)
            results.add(result)
        }
        return results
    }

    // Helper for rotated rectangle to points
    fun rotatedRectToPoints(cx: Float, cy: Float, w: Float, h: Float, angleRad: Float): List<Pair<Float, Float>> {
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

    fun model(frame: Mat, targetShape: Size = Size(640, 640)):  List<List<Float>> {
        val preprocessed = preprocess(frame, targetShape)

        // Prepare input buffer
        val inputBuffers = model.createInputBuffers()
        // Flatten preprocessed Mat to float array
        val inputArray = FloatArray((preprocessed.total() * preprocessed.channels()).toInt())
        preprocessed.get(0, 0, inputArray)
        inputBuffers[0].writeFloat(inputArray)

        // Run inference
        val outputBuffers = model.createOutputBuffers()
        model.run(inputBuffers, outputBuffers)
        val outputs = outputBuffers[0].readFloatArray() // shape: [N, 7]
        // Convert output to Array<FloatArray>
        val numDetections = outputs.size / 7
        val outputArray = Array(numDetections) { i ->
            outputs.sliceArray(i * 7 until (i + 1) * 7)
        }

        // Postprocess
        val results = postprocess(frame, outputArray, letterboxed.size(), pad)
        println("Detections: $results")

        return results
    }

    fun convert_to_yolo_results(results: List<List<Float>>): yolo_results {
        val bowResults = BowResults(
            bow = null,
            string = null,
        )
        var largest_bow_conf = 0.0f
        var largest_string_conf = 0.0f
        var bow_index = -1
        var string_index = -1
        //0 is bow, 1 is string for cls
        for (int i = 0; i < results.size; i++) {
            if (results[i][8] == 1.0f) {
                if (results[i][7] > largest_string_conf) {
                    largest_string_conf = results[i][9]
                    string_index = i
                }
            }
            if (results[i][8] == 0.0f) {
                if (results[i][7] > largest_bow_conf) {
                    largest_bow_conf = results[i][9]
                    bow_index = i
                }
            }
        }
        if (bow_index != -1) {
            var bow_return = mutableListOf<MutableList<Int>>()
            for (int i = 0; i < 4; i++) {
                bow_return.add(mutableListOf(results[bow_index][2*i], results[bow_index][2*i +1]))
            }
            bowResults.bow = bow_return
        }
        if (string_index != -1) {
            var string_return = mutableListOf<MutableList<Int>>()
            for (int i = 0; i < 4; i++) {
                string_return.add(mutableListOf(results[stirng_index][2*i], results[string_index][2*i +1]))
            }
            bowResults.string =  string_return
        }
        return bowResults
    }
} 
