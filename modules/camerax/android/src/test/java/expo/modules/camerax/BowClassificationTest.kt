package expo.modules.camerax

import org.junit.Assert.assertEquals
import org.junit.Test

class BowClassificationTest {

    @Test
    fun testBowPositionFromCode() {
        assertEquals(BowPosition.NONE, BowPosition.fromCode(-2))
        assertEquals(BowPosition.PARTIAL, BowPosition.fromCode(-1))
        assertEquals(BowPosition.GOOD, BowPosition.fromCode(0))
        assertEquals(BowPosition.OUTSIDE, BowPosition.fromCode(1))
        assertEquals(BowPosition.LEFT, BowPosition.fromCode(2))
        assertEquals(BowPosition.RIGHT, BowPosition.fromCode(3))

        // Invalid or missing codes
        assertEquals(BowPosition.UNKNOWN, BowPosition.fromCode(-3))
        assertEquals(BowPosition.UNKNOWN, BowPosition.fromCode(99))
        assertEquals(BowPosition.UNKNOWN, BowPosition.fromCode(null))
    }

    @Test
    fun testBowAngleFromCode() {
        assertEquals(BowAngle.UNKNOWN, BowAngle.fromCode(-1))
        assertEquals(BowAngle.GOOD, BowAngle.fromCode(0))
        assertEquals(BowAngle.TOO_ANGLED, BowAngle.fromCode(1))

        // Invalid or missing codes
        assertEquals(BowAngle.UNKNOWN, BowAngle.fromCode(99))
        assertEquals(BowAngle.UNKNOWN, BowAngle.fromCode(null))
    }

    @Test
    fun testPositionLabels() {
        assertEquals("No detection", BowFeedback.positionLabel(BowPosition.NONE))
        assertEquals("Partial detection", BowFeedback.positionLabel(BowPosition.PARTIAL))
        assertEquals("Good Bow Position", BowFeedback.positionLabel(BowPosition.GOOD))
        assertEquals("Bow Fully Outside", BowFeedback.positionLabel(BowPosition.OUTSIDE))
        assertEquals("Bow Too Far Left", BowFeedback.positionLabel(BowPosition.LEFT))
        assertEquals("Bow Too Far Right", BowFeedback.positionLabel(BowPosition.RIGHT))
        assertEquals("Unknown", BowFeedback.positionLabel(BowPosition.UNKNOWN))
    }

    @Test
    fun testAngleLabels() {
        assertEquals("Good Bow Angle", BowFeedback.angleLabel(BowAngle.GOOD))
        assertEquals("Bow Too Angled", BowFeedback.angleLabel(BowAngle.TOO_ANGLED))
        assertEquals("Unknown", BowFeedback.angleLabel(BowAngle.UNKNOWN))
    }

    @Test
    fun testPositionInstructions() {
        assertEquals("", BowFeedback.positionInstruction(BowPosition.NONE))
        assertEquals("", BowFeedback.positionInstruction(BowPosition.PARTIAL))
        assertEquals("", BowFeedback.positionInstruction(BowPosition.GOOD))
        assertEquals("Keep the bow in zone", BowFeedback.positionInstruction(BowPosition.OUTSIDE))
        assertEquals("Move the bow to the right", BowFeedback.positionInstruction(BowPosition.LEFT))
        assertEquals("Move the bow to the left", BowFeedback.positionInstruction(BowPosition.RIGHT))
        assertEquals("", BowFeedback.positionInstruction(BowPosition.UNKNOWN))
    }

    @Test
    fun testAngleInstructions() {
        assertEquals("", BowFeedback.angleInstruction(BowAngle.GOOD))
        assertEquals("Adjust your bow angle", BowFeedback.angleInstruction(BowAngle.TOO_ANGLED))
        assertEquals("", BowFeedback.angleInstruction(BowAngle.UNKNOWN))
    }

    @Test
    fun testBowHeightFromCode() {
        assertEquals(BowHeight.NONE, BowHeight.fromCode(-2))
        assertEquals(BowHeight.PARTIAL, BowHeight.fromCode(-1))
        assertEquals(BowHeight.GOOD, BowHeight.fromCode(0))
        assertEquals(BowHeight.OUTSIDE, BowHeight.fromCode(1))
        assertEquals(BowHeight.TOO_HIGH, BowHeight.fromCode(2))
        assertEquals(BowHeight.TOO_LOW, BowHeight.fromCode(3))
        assertEquals(BowHeight.UNKNOWN, BowHeight.fromCode(99))
        assertEquals(BowHeight.UNKNOWN, BowHeight.fromCode(null))
    }

    @Test
    fun testBowHeightFeedback() {
        assertEquals(
            "Bow Too High",
            BowFeedback.heightLabel(BowHeight.TOO_HIGH)
        )
        assertEquals(
            "Bow Too Low",
            BowFeedback.heightLabel(BowHeight.TOO_LOW)
        )
        assertEquals(
            "Lower the bow",
            BowFeedback.heightInstruction(BowHeight.TOO_HIGH)
        )
        assertEquals(
            "Lift the bow",
            BowFeedback.heightInstruction(BowHeight.TOO_LOW)
        )
    }
}
