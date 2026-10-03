package expo.modules.camerax

import kotlin.math.*

object ViolinBowClassifier {

    data class Point(
        val x: Double,
        val y: Double
    )

    data class ClassificationResult(
        val position: Int,
        val angle: Int
    )

    fun classify(
        bowPoints: List<Point>,
        stringPoints: List<Point>,
        maxAngle: Double = 20.0
    ): ClassificationResult {

        require(bowPoints.size == 4) {
            "bowPoints must contain exactly 4 points"
        }

        require(stringPoints.size == 4) {
            "stringPoints must contain exactly 4 points"
        }

        val sortedBowPoints = sortBowPoints(bowPoints)
        val sortedStringPoints = sortStringPoints(stringPoints)

        val midline = getMidline(sortedBowPoints)

        val horizontalLines = getHorizontalLines(sortedStringPoints)

        val position = intersectsHorizontal(
            midline,
            horizontalLines,
            sortedStringPoints
        )

        val angle = bowAngle(
            midline,
            horizontalLines,
            maxAngle
        )

        return ClassificationResult(
            position = position,
            angle = angle
        )
    }

    private fun sortStringPoints(
        pts: List<Point>
    ): List<Point> {

        val sortedPoints = pts.sortedBy { it.y }

        val topPoints = sortedPoints.take(2).sortedBy { it.x }
        val bottomPoints = sortedPoints.drop(2).sortedBy { it.x }

        return topPoints + bottomPoints
    }

    private fun sortBowPoints(
        pts: List<Point>
    ): List<Point> {

        val sortedPoints = pts.sortedBy { it.y }

        val topPoints = sortedPoints.take(2).sortedBy { it.x }
        val bottomPoints = sortedPoints.drop(2).sortedBy { it.x }

        return listOf(
            topPoints[0],
            topPoints[1],
            bottomPoints[0],
            bottomPoints[1]
        )
    }

    private fun getMidline(
        bowPoints: List<Point>
    ): List<Double> {

        // Calculate squared distance between two points
        fun distance(pt1: Point, pt2: Point): Double {
            return (pt1.x - pt2.x) * (pt1.x - pt2.x) +
                    (pt1.y - pt2.y) * (pt1.y - pt2.y)
        }

        // Find the length of all four sides of the bow rectangle
        val dTop = distance(bowPoints[0], bowPoints[1])
        val dRight = distance(bowPoints[1], bowPoints[3])
        val dBottom = distance(bowPoints[3], bowPoints[2])
        val dLeft = distance(bowPoints[2], bowPoints[0])

        val distances = listOf(
            dTop,
            dRight,
            dBottom,
            dLeft
        )

        // Find the shortest side of the bow rectangle
        val minIndex = distances.indexOf(distances.minOrNull())

        // The two short sides are the ends of the bow.
        // Get the two pairs of points that form those ends.
        val (pair1, pair2) = when (minIndex) {
            0 -> Pair(
                bowPoints[0] to bowPoints[1],
                bowPoints[2] to bowPoints[3]
            )

            1 -> Pair(
                bowPoints[1] to bowPoints[3],
                bowPoints[0] to bowPoints[2]
            )

            2 -> Pair(
                bowPoints[3] to bowPoints[2],
                bowPoints[1] to bowPoints[0]
            )

            else -> Pair(
                bowPoints[2] to bowPoints[0],
                bowPoints[3] to bowPoints[1]
            )
        }

        // Find the midpoint of each end
        val mid1 = listOf(
            (pair1.first.x + pair1.second.x) / 2,
            (pair1.first.y + pair1.second.y) / 2
        )

        val mid2 = listOf(
            (pair2.first.x + pair2.second.x) / 2,
            (pair2.first.y + pair2.second.y) / 2
        )

        // Calculate the line through the two midpoints
        val dy = mid1[1] - mid2[1]
        val dx = mid1[0] - mid2[0]

        return if (dx == 0.0) {
            // Vertical line:
            // first value = infinite slope
            // second value = x coordinate
            listOf(Double.POSITIVE_INFINITY, mid1[0])
        } else {
            val slope = dy / dx
            val intercept = mid1[1] - slope * mid1[0]

            listOf(slope, intercept)
        }
    }

    private fun getHorizontalLines(
        stringPoints: List<Point>
    ): List<List<Double>> {

        val topLeft = stringPoints[0]
        val topRight = stringPoints[1]
        val bottomLeft = stringPoints[2]
        val bottomRight = stringPoints[3]

        // Top edge of the string box
        val dyTop = topLeft.y - topRight.y
        val dxTop = topLeft.x - topRight.x

        val topSlope: Double
        val topIntercept: Double

        if (dxTop == 0.0) {
            topSlope = Double.POSITIVE_INFINITY
            topIntercept = -1.0
        } else {
            topSlope = dyTop / dxTop
            topIntercept = topLeft.y - topSlope * topLeft.x
        }

        // Bottom edge of the string box
        val dxBottom = bottomLeft.x - bottomRight.x
        val dyBottom = bottomLeft.y - bottomRight.y

        val bottomSlope: Double
        val bottomIntercept: Double

        if (dxBottom == 0.0) {
            bottomSlope = Double.POSITIVE_INFINITY
            bottomIntercept = -1.0
        } else {
            bottomSlope = dyBottom / dxBottom
            bottomIntercept = bottomRight.y - bottomSlope * bottomRight.x
        }

        // Each line:
        // [slope, intercept, leftX, rightX]
        val topLine = listOf(
            topSlope,
            topIntercept,
            topLeft.x,
            topRight.x
        )

        val bottomLine = listOf(
            bottomSlope,
            bottomIntercept,
            bottomLeft.x,
            bottomRight.x
        )

        return listOf(topLine, bottomLine)
    }

    private fun intersectsHorizontal(
        bowLine: List<Double>,
        horizontalLines: List<List<Double>>,
        stringPoints: List<Point>
    ): Int {

        val bowSlope = bowLine[0]
        val bowIntercept = bowLine[1]

        val horizontalOne = horizontalLines[0]
        val horizontalTwo = horizontalLines[1]

        // Find the intersection between the bow midline
        // and one edge of the string box.
        fun getIntersection(
            horizontalLine: List<Double>,
            xReference: Double
        ): Point? {

            val lineSlope = horizontalLine[0]
            val lineIntercept = horizontalLine[1]
            val leftX = horizontalLine[2]
            val rightX = horizontalLine[3]

            val x: Double
            val y: Double

            if (
                lineSlope == Double.POSITIVE_INFINITY ||
                lineIntercept == -1.0
            ) {
                x = xReference

                if (bowSlope == Double.POSITIVE_INFINITY) {
                    return null
                }

                y = bowSlope * x + bowIntercept

            } else if (bowSlope == Double.POSITIVE_INFINITY) {

                x = bowIntercept
                y = lineSlope * x + lineIntercept

            } else if (abs(bowSlope - lineSlope) < 1e-6) {

                // Parallel lines do not intersect
                return null

            } else {

                x = (lineIntercept - bowIntercept) /
                        (bowSlope - lineSlope)

                y = bowSlope * x + bowIntercept
            }

            // Check whether the intersection is actually
            // inside this string-box edge.
            val xMin = minOf(leftX, rightX)
            val xMax = maxOf(leftX, rightX)

            if (x < xMin || x > xMax) {
                return null
            }

            return Point(x, y)
        }

        val xLeft = stringPoints[0].x
        val xRight = stringPoints[1].x

        val pointOne = getIntersection(
            horizontalOne,
            xLeft
        )

        val pointTwo = getIntersection(
            horizontalTwo,
            xRight
        )

        // No valid intersection means the bow is fully outside.
        if (pointOne == null || pointTwo == null) {
            return 1
        }

        return bowWidthIntersection(
            listOf(pointOne, pointTwo),
            horizontalLines
        )
    }

    private fun bowWidthIntersection(
        intersectionPoints: List<Point>,
        horizontalLines: List<List<Double>>
    ): Int {

        val leftZonePercentage = 0.1
        val rightZonePercentage = 0.1

        val horizontalOne = horizontalLines[0]
        val horizontalTwo = horizontalLines[1]

        val topWidth = abs(horizontalOne[3] - horizontalOne[2])

        val bottomWidth = abs(horizontalTwo[3] - horizontalTwo[2])

        val width = (topWidth + bottomWidth) / 2.0

        val averageLeftX = (horizontalOne[2] + horizontalTwo[2]) / 2.0

        val averageRightX = (horizontalOne[3] + horizontalTwo[3]) / 2.0

        val tooLeftThreshold =
            averageLeftX + width * leftZonePercentage

        val tooRightThreshold =
            averageRightX - width * rightZonePercentage

        val intersectionX =
            intersectionPoints.map { it.x }.average()

        if (intersectionX <= tooLeftThreshold) {
            return 2
        }

        if (intersectionX >= tooRightThreshold) {
            return 3
        }

        return 0
    }

    private fun degrees(
        radians: Double
    ): Double {
        return radians * (180.0 / PI)
    }

    private fun bowAngle(
        bowLine: List<Double>,
        horizontalLines: List<List<Double>>,
        maxAngle: Double
    ): Int {

        val bowSlope = bowLine[0]

        val slopeOne = horizontalLines[0][0]
        val slopeTwo = horizontalLines[1][0]

        fun directionAngle(slope: Double): Double {
            return if (slope.isInfinite()) {
                90.0
            } else {
                degrees(atan(slope))
            }
        }

        fun normalizedAngleDifference(
            angleOne: Double,
            angleTwo: Double
        ): Double {
            val difference = abs(angleOne - angleTwo) % 180.0
            return min(difference, 180.0 - difference)
        }

        val bowDirection = directionAngle(bowSlope)
        val stringDirectionOne = directionAngle(slopeOne)
        val stringDirectionTwo = directionAngle(slopeTwo)

        if (
            !bowDirection.isFinite() ||
            !stringDirectionOne.isFinite() ||
            !stringDirectionTwo.isFinite()
        ) {
            return 1
        }

        val angleOne = normalizedAngleDifference(
            bowDirection,
            stringDirectionOne
        )

        val angleTwo = normalizedAngleDifference(
            bowDirection,
            stringDirectionTwo
        )

        val minimumAngle = min(angleOne, angleTwo)

        return if (minimumAngle > maxAngle) {
            1
        } else {
            0
        }
    }
}