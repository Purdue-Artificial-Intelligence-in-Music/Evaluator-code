package expo.modules.camerax

import org.junit.Assert.assertEquals
import org.junit.Test

// Edge case:
// When the bow intersects exactly at the right boundary,
// the classifier may return Fully Outside instead of Too Far Right.

// Geometry ambiguity:
// A perfectly axis-aligned rectangular string box may produce
// a zero width in the current left/right threshold calculation.

class ViolinBowClassifierTest {

    @Test
    fun bowCrossingStringAreaCenter_returnsGoodPosition() {
        val bowPoints = listOf(
            ViolinBowClassifier.Point(45.0, 0.0),
            ViolinBowClassifier.Point(55.0, 0.0),
            ViolinBowClassifier.Point(45.0, 100.0),
            ViolinBowClassifier.Point(55.0, 100.0)
        )

        val stringPoints = listOf(
            ViolinBowClassifier.Point(0.0, 30.0),
            ViolinBowClassifier.Point(100.0, 30.0),
            ViolinBowClassifier.Point(0.0, 70.0),
            ViolinBowClassifier.Point(100.0, 70.0)
        )

        val result = ViolinBowClassifier.classify(bowPoints, stringPoints)

        assertEquals(
            "A bow crossing the center of the string area should have a good position",
            0,
            result.position
        )
    }

    @Test
    fun bowOutsideStringArea_returnsFullyOutside() {
        val bowPoints = listOf(
            ViolinBowClassifier.Point(145.0, 0.0),
            ViolinBowClassifier.Point(155.0, 0.0),
            ViolinBowClassifier.Point(145.0, 100.0),
            ViolinBowClassifier.Point(155.0, 100.0)
        )

        val stringPoints = listOf(
            ViolinBowClassifier.Point(0.0, 30.0),
            ViolinBowClassifier.Point(100.0, 30.0),
            ViolinBowClassifier.Point(0.0, 70.0),
            ViolinBowClassifier.Point(100.0, 70.0)
        )

        val result = ViolinBowClassifier.classify(
            bowPoints,
            stringPoints
        )

        assertEquals(
            "A bow completely outside the string area should be classified as fully outside",
            1,
            result.position
        )
    }
    @Test
    fun bowNearLeftEdge_returnsTooFarLeft() {
        val bowPoints = listOf(
            ViolinBowClassifier.Point(1.0, 0.0),
            ViolinBowClassifier.Point(3.0, 0.0),
            ViolinBowClassifier.Point(1.0, 100.0),
            ViolinBowClassifier.Point(3.0, 100.0)
        )

        val stringPoints = listOf(
            ViolinBowClassifier.Point(0.0, 30.0),
            ViolinBowClassifier.Point(40.0, 30.0),
            ViolinBowClassifier.Point(0.0, 70.0),
            ViolinBowClassifier.Point(100.0, 70.0)
        )

        val result = ViolinBowClassifier.classify(
            bowPoints,
            stringPoints
        )

        assertEquals(
            "A bow near the left edge of the string area should be classified as too far left",
            2,
            result.position
        )
    }
    @Test
    fun bowNearRightEdge_returnsTooFarRight() {
        val bowPoints = listOf(
            ViolinBowClassifier.Point(-7.0, 0.0),
            ViolinBowClassifier.Point(-5.0, 0.0),
            ViolinBowClassifier.Point(143.0, 100.0),
            ViolinBowClassifier.Point(145.0, 100.0)
        )

        val stringPoints = listOf(
            ViolinBowClassifier.Point(0.0, 30.0),
            ViolinBowClassifier.Point(40.0, 30.0),
            ViolinBowClassifier.Point(0.0, 70.0),
            ViolinBowClassifier.Point(100.0, 70.0)
        )

        val result = ViolinBowClassifier.classify(
            bowPoints,
            stringPoints
        )

        assertEquals(
            "A bow near the right edge of the string area should be classified as too far right",
            3,
            result.position
        )
    }
    //18.3 degrees
    @Test
    fun bowAngleNear20Degrees_returnsGoodAngle() {
        val bowPoints = listOf(
            ViolinBowClassifier.Point(0.0, -1.0),
            ViolinBowClassifier.Point(0.0, 1.0),
            ViolinBowClassifier.Point(300.0, 99.0),
            ViolinBowClassifier.Point(300.0, 101.0)
        )

        val stringPoints = listOf(
            ViolinBowClassifier.Point(0.0, 30.0),
            ViolinBowClassifier.Point(400.0, 30.0),
            ViolinBowClassifier.Point(0.0, 70.0),
            ViolinBowClassifier.Point(400.0, 70.0)
        )

        val result = ViolinBowClassifier.classify(
            bowPoints,
            stringPoints
        )

        assertEquals(
            "A bow angle close to but below 20 degrees should be classified as a good angle",
            0,
            result.angle
        )
    }
    //20.3 degrees
    @Test
    fun bowAngleJustAbove20Degrees_returnsTooAngled() {
        val bowPoints = listOf(
            ViolinBowClassifier.Point(100.0, -1.0),
            ViolinBowClassifier.Point(100.0, 1.0),
            ViolinBowClassifier.Point(370.0, 99.0),
            ViolinBowClassifier.Point(370.0, 101.0)
        )

        val stringPoints = listOf(
            ViolinBowClassifier.Point(0.0, 30.0),
            ViolinBowClassifier.Point(500.0, 30.0),
            ViolinBowClassifier.Point(0.0, 70.0),
            ViolinBowClassifier.Point(500.0, 70.0)
        )
        val result = ViolinBowClassifier.classify(
            bowPoints,
            stringPoints
        )
        assertEquals(
            "A bow angle just above 20 degrees should be classified as too angled",
            1,
            result.angle
        )
    }
 //exact 20 degrees
    @Test
    fun bowAngleExactly20Degrees_returnsGoodAngle() {
        val bowPoints = listOf(
            // Bow center line is approximately 20 degrees
            ViolinBowClassifier.Point(100.0, -1.0),
            ViolinBowClassifier.Point(100.0, 1.0),
            ViolinBowClassifier.Point(374.75, 99.0),
            ViolinBowClassifier.Point(374.75, 101.0)
        )
        val stringPoints = listOf(
            ViolinBowClassifier.Point(0.0, 30.0),
            ViolinBowClassifier.Point(500.0, 30.0),
            ViolinBowClassifier.Point(0.0, 70.0),
            ViolinBowClassifier.Point(500.0, 70.0)
        )

        val result = ViolinBowClassifier.classify(
            bowPoints,
            stringPoints
        )

        assertEquals(
            "A bow angle at 20 degrees should be classified as a good angle",
            0,
            result.angle
        )
    }
// Bow center line: x = 50
    @Test
    fun rotatedStringBox_returnsGoodPosition() {
        val bowPoints = listOf(
            ViolinBowClassifier.Point(45.0, 0.0),
            ViolinBowClassifier.Point(55.0, 0.0),
            ViolinBowClassifier.Point(45.0, 100.0),
            ViolinBowClassifier.Point(55.0, 100.0)
        )

        val stringPoints = listOf(
            ViolinBowClassifier.Point(0.0, 25.0),
            ViolinBowClassifier.Point(100.0, 35.0),
            ViolinBowClassifier.Point(0.0, 65.0),
            ViolinBowClassifier.Point(100.0, 75.0)
        )

        val result = ViolinBowClassifier.classify(
            bowPoints,
            stringPoints
        )

        assertEquals(
            "A bow crossing the center of a rotated string box should have a good position",
            0,
            result.position
        )
    }
    // Vertical bow centered at x = 50
    @Test
    fun verticalBow_returnsGoodPosition() {
        val bowPoints = listOf(
            ViolinBowClassifier.Point(49.0, 0.0),
            ViolinBowClassifier.Point(51.0, 0.0),
            ViolinBowClassifier.Point(49.0, 100.0),
            ViolinBowClassifier.Point(51.0, 100.0)
        )
        val stringPoints = listOf(
            ViolinBowClassifier.Point(0.0, 30.0),
            ViolinBowClassifier.Point(100.0, 30.0),
            ViolinBowClassifier.Point(0.0, 70.0),
            ViolinBowClassifier.Point(100.0, 70.0)
        )

        val result = ViolinBowClassifier.classify(
            bowPoints,
            stringPoints
        )
        assertEquals(
            "A vertical bow crossing the center of the string area should have a good position",
            0,
            result.position
        )
    }
    //For invalid inputs, I mainly tested cases where the classifier
    // receives fewer than the expected four bow or string points. two cases
    //first one is 3 points in bow
    @Test(expected = IndexOutOfBoundsException::class)
    fun missingBowPoint_throwsException() {
        val bowPoints = listOf(
            ViolinBowClassifier.Point(45.0, 0.0),
            ViolinBowClassifier.Point(55.0, 0.0),
            ViolinBowClassifier.Point(45.0, 100.0)
        )
        val stringPoints = listOf(
            ViolinBowClassifier.Point(0.0, 30.0),
            ViolinBowClassifier.Point(100.0, 30.0),
            ViolinBowClassifier.Point(0.0, 70.0),
            ViolinBowClassifier.Point(100.0, 70.0)
        )
        ViolinBowClassifier.classify(
            bowPoints,
            stringPoints
        )
    }
    //second one is 3 points in string
    @Test(expected = IndexOutOfBoundsException::class)
    fun missingStringPoint_throwsException() {
        val bowPoints = listOf(
            ViolinBowClassifier.Point(45.0, 0.0),
            ViolinBowClassifier.Point(55.0, 0.0),
            ViolinBowClassifier.Point(45.0, 100.0),
            ViolinBowClassifier.Point(55.0, 100.0)
        )
        val stringPoints = listOf(
            ViolinBowClassifier.Point(0.0, 30.0),
            ViolinBowClassifier.Point(100.0, 30.0),
            ViolinBowClassifier.Point(0.0, 70.0)
        )
        ViolinBowClassifier.classify(
            bowPoints,
            stringPoints
        )
    }
}