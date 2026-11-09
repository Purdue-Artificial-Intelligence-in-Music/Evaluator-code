package expo.modules.camerax

enum class BowPosition(val code: Int) {
    NONE(-2),
    PARTIAL(-1),
    GOOD(0),
    OUTSIDE(1),
    LEFT(2),
    RIGHT(3),
    UNKNOWN(-3);

    companion object {
        fun fromCode(code: Int?): BowPosition =
            entries.find { it.code == code } ?: UNKNOWN
    }
}

enum class BowHeight(val code: Int) {
    NONE(-2),
    PARTIAL(-1),
    GOOD(0),
    OUTSIDE(1),
    TOO_HIGH(2),
    TOO_LOW(3),
    UNKNOWN(-3);

    companion object {
        fun fromCode(code: Int?): BowHeight =
            entries.find { it.code == code } ?: UNKNOWN
    }
}
enum class BowAngle(val code: Int) {
    UNKNOWN(-1),
    GOOD(0),
    TOO_ANGLED(1);

    companion object {
        fun fromCode(code: Int?): BowAngle =
            entries.find { it.code == code } ?: UNKNOWN
    }
}

object BowFeedback {

    fun heightLabel(height: BowHeight): String = when (height) {
        BowHeight.NONE -> "No detection"
        BowHeight.PARTIAL -> "Partial detection"
        BowHeight.GOOD -> "Good Bow Contact Point"
        BowHeight.OUTSIDE -> "Bow Fully Outside"
        BowHeight.TOO_HIGH -> "Bow Too High"
        BowHeight.TOO_LOW -> "Bow Too Low"
        BowHeight.UNKNOWN -> "Unknown"
    }

    fun heightInstruction(height: BowHeight): String = when (height) {
        BowHeight.OUTSIDE -> "Keep the bow in zone"
        BowHeight.TOO_HIGH -> "Lower the bow"
        BowHeight.TOO_LOW -> "Lift the bow"
        else -> ""
    }

    fun positionLabel(position: BowPosition): String =
        when (position) {
            BowPosition.NONE -> "No detection"
            BowPosition.PARTIAL -> "Partial detection"
            BowPosition.GOOD -> "Good Bow Position"
            BowPosition.OUTSIDE -> "Bow Fully Outside"
            BowPosition.LEFT -> "Bow Too Far Left"
            BowPosition.RIGHT -> "Bow Too Far Right"
            BowPosition.UNKNOWN -> "Unknown"
        }

    fun angleLabel(angle: BowAngle): String =
        when (angle) {
            BowAngle.GOOD -> "Good Bow Angle"
            BowAngle.TOO_ANGLED -> "Bow Too Angled"
            BowAngle.UNKNOWN -> "Unknown"
        }

    fun positionInstruction(position: BowPosition): String =
        when (position) {
            BowPosition.OUTSIDE -> "Keep the bow in zone"
            BowPosition.LEFT -> "Move the bow to the right"
            BowPosition.RIGHT -> "Move the bow to the left"
            else -> ""
        }

    fun angleInstruction(angle: BowAngle): String =
        when (angle) {
            BowAngle.TOO_ANGLED -> "Adjust your bow angle"
            else -> ""
        }
}
