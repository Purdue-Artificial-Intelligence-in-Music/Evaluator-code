package expo.modules.camerax

/**
 * Turns raw model outputs into a [ViolinDetection].
 *
 * This is the only piece that should need to change when a new model is plugged in:
 * write a new implementation and pass it to [Detector]. Implementations must stay
 * free of Android / TFLite types so they can be unit-tested on the JVM.
 */
interface ModelOutputParser {
    /**
     * @param outputs raw output tensors keyed by the interpreter's output index.
     * @param letterbox how the source bitmap was placed into the model input tensor,
     *   used to map tensor-space coordinates back to normalized source coordinates.
     * @return the detection; [ViolinDetection.EMPTY] when nothing passes the parser's filters.
     * @throws IllegalArgumentException if [outputs] are not in a layout this parser understands.
     */
    fun parse(outputs: Map<Int, OutputTensor>, letterbox: LetterboxInfo): ViolinDetection
}

/** One raw output tensor: its shape as reported by the interpreter and its values, flattened row-major. */
class OutputTensor(val shape: IntArray, val data: FloatArray) {
    init {
        require(shape.fold(1) { acc, d -> acc * d } == data.size) {
            "data size ${data.size} does not match shape ${shape.contentToString()}"
        }
    }
}

/**
 * Where the source bitmap sits inside the model input tensor, in tensor pixels.
 *
 * The source content occupies the rectangle [padX, padX + contentWidth) x [padY, padY + contentHeight)
 * of a tensorWidth x tensorHeight input. Separate content width/height allow both an
 * aspect-preserving letterbox and a non-uniform stretch (the current preprocessing).
 */
data class LetterboxInfo(
    val tensorWidth: Int,
    val tensorHeight: Int,
    val contentWidth: Float,
    val contentHeight: Float,
    val padX: Float = 0f,
    val padY: Float = 0f
) {
    fun toNormalizedX(tensorX: Double): Double = (tensorX - padX) / contentWidth
    fun toNormalizedY(tensorY: Double): Double = (tensorY - padY) / contentHeight
}
