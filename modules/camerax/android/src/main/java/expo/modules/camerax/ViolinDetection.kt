package expo.modules.camerax

/**
 * Model-independent detection result. Every [ModelOutputParser] produces this,
 * so the rest of the pipeline never needs to know the model's tensor layout.
 *
 * Coordinate space: all [Quad] corners are in NORMALIZED SOURCE-BITMAP coordinates,
 * i.e. (0,0) is the top-left of the bitmap passed to [Detector.detect] and (1,1)
 * is its bottom-right. Values may fall slightly outside [0,1] when a box touches
 * the frame edge.
 */

data class ViolinDetection(
    val bow: Quad?,
    val strings: Quad?,
    val bowConfidence: Float = 0f,
    val stringConfidence: Float = 0f
) {
    val isEmpty: Boolean get() = bow == null && strings == null

    companion object {
        val EMPTY = ViolinDetection(bow = null, strings = null)
    }
}

/**
 * Four corner points in normalized source-bitmap coordinates (see [ViolinDetection]).
 * Corner order is whatever the parser emits; consumers that need a specific order
 * (e.g. string-box sorting) sort them themselves.
 */
data class Quad(val corners: List<Corner>) {
    data class Corner(val x: Double, val y: Double)

    init {
        require(corners.size == 4) { "Quad must have exactly 4 corners, got ${corners.size}" }
    }
}
