import kotlin.math.*

data class Point(
    val x: Double,
    val y: Double
)

var bowPoints = mutableListOf(
    Point(40.0, 6.0),
    Point(60.0, 50.0),
    Point(48.0, 0.0),
    Point(66.0, 42.0)
)

var stringPoints = mutableListOf(
    Point(5.0, 15.0),
    Point(50.0, 5.0),
    Point(55.0, 22.0),
    Point(10.0, 33.0)
)

val maxAngle = 20
/**
 * You can edit, run, and share this code.
 * play.kotlinlang.org
 */
fun main() {

    println("String points:")
    stringPoints.forEach { println(it) }

    // Test sorting
    stringPoints = sortStringPoints(stringPoints)

    println("\nSorted:")
    stringPoints.forEach { println(it) }

    println("\nSortedBow:")
    bowPoints = sortBowPoints(bowPoints)
    bowPoints.forEach { println(it) }

    // Test horizontal lines
    val lines = getHorizontalLines()

    println("\nHorizontal lines:")
    lines.forEach { println(it) }

    // Test midline
    val midline = getMidline()

    println("\nMidline:")
    println(midline)

    // Test intersection
    val result = intersectsHorizontal(
        midline,
        lines
    )

    println("\nResult:")
    println(result)

    println("\nAngle:")
    println(bowAngle(midline, lines))
}


/* Plan:
 * String box & bow box
 * Get midlines for each
 * Angle:
 * Get angle of intersection of the midline of the string box and midline of the bow box
 * Height:
 * Get 'vertical' lines of string box
 * Check distance of midline to each string box (or simply if outside)
 */

fun sortStringPoints(pts: MutableList<Point>): MutableList<Point> {
    // Sort points by y
    val sortedPoints = pts.sortedBy {it.y }

    // Find first 2 and last pts
    val topPoints = sortedPoints.take(2).sortedBy { it.x }      // Sort by X ascending
    val bottomPoints = sortedPoints.drop(2).sortedBy { it.x } // Sort by X ascending

    return (topPoints + bottomPoints).toMutableList()
}

/*
 * Returns bow points sorted as:
 * Top left, top right, bottom left, bottom right.
 */
fun sortBowPoints(pts: MutableList<Point>): MutableList<Point> {
    // Sort points by y
    val sortedPoints = pts.sortedBy {it.y}

    // Sort points by x
    val topPoints = sortedPoints.take(2).sortedBy {it.x} // Sort by x ascending
    val botPoints = sortedPoints.drop(2).sortedBy {it.x} // Sort by x ascending

    return mutableListOf(topPoints[0], topPoints[1], botPoints[0], botPoints[1])
}

/*
 * Gets the midline for the bow
 * The shortest lines are those intersected by the midline
 */
fun getMidline(): MutableList<Double> {
    fun distance(pt1: Point, pt2: Point): Double {
        //just distance formula
        return (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y)
    }

    //find length of the sides of the bow rectangle
    val dTop = distance(bowPoints[0], bowPoints[1])
    val dRight = distance(bowPoints[1], bowPoints[3])
    val dBottom = distance(bowPoints[3], bowPoints[2])
    val dLeft = distance(bowPoints[2], bowPoints[0])

    println("top = $dTop")
    println("right = $dRight")
    println("bottom = $dBottom")
    println("left = $dLeft")

    val distances = listOf(dTop, dRight, dBottom, dLeft)

    val minIndex = distances.indexOf(	distances.minOrNull()) //find the smallest distance

    //find the two shortest distances to find the end of the bow and set those points as pair1 and pair2
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

/*
 * Extracts vertical lines for the string box
 */

private fun getHorizontalLines(): MutableList<MutableList<Double>> {
    // Extracting corner points
    val topLeft = stringPoints!![0]
    val topRight = stringPoints!![1]
    val botLeft = stringPoints!![2]
    val botRight = stringPoints!![3]

    // Get top line
    // Calculate the vertical and horiztonal distance between the top points
    val dyTop = topLeft.y - topRight.y
    val dxTop = topLeft.x - topRight.x
    // Initialize slope and y-intercept for the left side
    val topSlope: Double
    val topYInt: Double

    if (dxTop == 0.0) {
        topSlope = Double.POSITIVE_INFINITY
        topYInt = -1.0  // Use -1.0 as a flag for undefined intercept
    } else {
        // Calculate slope
        topSlope = dyTop / dxTop
        // Calculate y-intercept
        topYInt = topLeft.y - topSlope * topLeft.x
    }

    // Right vertical line (from topRight to botRight)

    val dxBot = botLeft.x - botRight.x
    val dyBot = botLeft.y - botRight.y
    val botSlope: Double
    val botYInt: Double

    if (dxBot == 0.0) {
        botSlope = Double.POSITIVE_INFINITY
        botYInt = -1.0
    } else {
        botSlope = dyBot / dxBot
        botYInt = botRight.y - botSlope * botRight.x
    }

    // Left & right of each side
    val topLeftX = topLeft.x
    val botLeftX = botLeft.x
    val topRightX = topRight.x
    val botRightX = botRight.x

    // Each line is a MutableList: [slope, intercept, leftx, rightx]
    val topLine = mutableListOf(topSlope, topYInt, topLeftX, topRightX)
    val botLine = mutableListOf(botSlope, botYInt, botLeftX, botRightX)

    // return a list of both lines
    return mutableListOf(topLine, botLine)
}


private fun intersectsHorizontal(
    linearLine: MutableList<Double>,
    horizontalLines: MutableList<MutableList<Double>>
): Int {
    // Midline parameters
    val m = linearLine[0] // slope of the midline
    val b = linearLine[1] // y-intercept of the midline

    // extracts the first horizontal line (left side): [slope, yInt, topY, botY]
    val horizontalOne = horizontalLines[0]
    val horizontalTwo = horizontalLines[1]

    // Calculates the intersection of the midline with a horizontal line
    fun getIntersection(hLine: List<Double>, xRef: Double): Point? {
        val slopeH = hLine[0]
        val interceptH = hLine[1]
        val leftX = hLine[2]
        val rightX = hLine[3]

        val x: Double
        val y: Double

        if (slopeH == Double.POSITIVE_INFINITY || interceptH == -1.0) {
            x = xRef
            if (m == Double.POSITIVE_INFINITY) return null // both lines vertical
            y = m * x + b
        } else if (m == Double.POSITIVE_INFINITY) {
            // Case: vertical midline
            x = b
            y = slopeH * x + interceptH
        } else if (kotlin.math.abs(m - slopeH) < 1e-6) {
            // parallel lines means no intersection
            return null
        } else {
            x = (interceptH - b) / (m - slopeH)
            y = m * x + b
        }
        // Makes sure intersection y-value is within the horizontal segment's range
        val xMin = minOf(leftX, rightX)
        val xMax = maxOf(leftX, rightX)

        if (xMin > x || x > xMax) {
            return null
        }

        return Point(x,y)
    }

    // Determine x positions from the bounding box
    val xLeft = stringPoints!![0].x
    val xRight = stringPoints!![1].x

    // Calculate intersections of midline with both horizontal string lines
    var pt1 = getIntersection(horizontalOne, xLeft)
    var pt2 = getIntersection(horizontalTwo, xRight)

    if (pt1 == null || pt2 == null) {
        //println("One or both intersections invalid")
        return 1
    }
//        if (pt1 == null) {
//            pt1 = pt2
//        }
//        if (pt2 == null){
//            pt2 = pt1
//        }
    return bowWidthIntersection(mutableListOf(pt1!!, pt2!!), mutableListOf(horizontalOne, horizontalTwo))
}


/*
 * Determines the height level at which the linear line intersects the vertical lines.

    Returns:
    - 3: Intersection is near right of the box (ht1 or ht2)
    - 2: Intersection is near left (hb1 or hb2)
    - 0: Intersection is in middle
 */


private fun bowWidthIntersection(
    intersectionPoints: MutableList<Point>,
    horizontalLines: List<List<Double>>
): Int {
    val left_zone_percentage = 0.1
    val right_zone_percentage = 0.1

    val horizontal_one = horizontalLines[0]
    val horizontal_two = horizontalLines[1]

    val top_x1 = horizontal_one[2]
    val top_x2 = horizontal_two[2]
    val bot_x1 = horizontal_one[3]
    val bot_x2 = horizontal_two[3]

    val width = abs(((top_x1 - top_x2) + (bot_x1 - bot_x2)) / 2.0)
    if (width == 0.0) return 0

    val avg_left_x = (top_x1 + bot_x1) / 2.0
    val avg_right_x = (top_x2 + bot_x2) / 2.0

    val too_left_threshold = avg_left_x + width * left_zone_percentage
    val too_right_threshold = avg_right_x - width * right_zone_percentage

    val intersection_x = intersectionPoints.map { it.x }.average()

    if (intersection_x <= too_left_threshold) {
        return 2
    }

    if (intersection_x >= too_right_threshold) {
        return 3
    }

    return 0
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
private fun bowAngle(bowLine: MutableList<Double>, horizontalLines: MutableList<MutableList<Double>>): Int {
    // grab bow line and vertical lines
    val m_bow: Double = bowLine[0]
    val m1 = horizontalLines[0][0]
    val m2 = horizontalLines[1][0]  // assuming format: [m1, b1, m2, b2]

    // calculate angles formed for each vertical line's intersection with bow line
    val angle_one: Double = abs(degrees(atan(abs(m_bow - m2) / (1 + m_bow * m2))))
    val angle_two: Double = abs(degrees(atan(abs(m1 - m_bow) / (1 + m1 * m_bow))))

    print("angle_one: ")
    println(angle_one)
    print("angle_two: ")
    println(angle_two)

    val min_angle: Double = min(abs(90 - min(angle_one, angle_two)), min(angle_one, angle_two))
    //println("ANGLE: $min_angle")
    return if (min_angle > maxAngle) 1 else 0  // 1 = Wrong Angle, 0 = Correct Angle
}