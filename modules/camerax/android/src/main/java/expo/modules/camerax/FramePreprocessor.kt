package expo.modules.camerax

import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Decides where a camera frame goes inside the model input tensor, and maps
 * coordinates between source-frame pixels and tensor pixels.
 *
 * Pure geometry: no Android or TFLite types, so it can be unit-tested on the JVM.
 * Drawing the bitmap is left to the caller (Detector), which scales the frame to
 * [PreprocessPlan.contentWidth] x [PreprocessPlan.contentHeight] and draws it at
 * ([PreprocessPlan.padX], [PreprocessPlan.padY]).
 *
 * The tensor size is passed per call, so any model input size works (no fixed 640x640).
 */
class FramePreprocessor(val mode: ResizeMode = ResizeMode.LEGACY_STRETCH) {

    enum class ResizeMode {
        /**
         * Current try.tflite behavior: the frame is stretched, ignoring its aspect ratio, into a
         * 9:16 portrait region at the top-left of the tensor, using the full tensor height.
         * For a 640x640 input this is exactly the 360x640 region Detector uses today.
         * Landscape frames are distorted in this mode.
         */
        LEGACY_STRETCH,

        /** Uniform scale to fit the tensor, centered, with padding on the short side. */
        LETTERBOX
    }

    companion object {
        // Legacy content region aspect (width:height = 9:16), taken from the original 360x640.
        private const val LEGACY_ASPECT_W = 9
        private const val LEGACY_ASPECT_H = 16
    }

    fun plan(sourceWidth: Int, sourceHeight: Int, tensorWidth: Int, tensorHeight: Int): PreprocessPlan {
        require(sourceWidth > 0 && sourceHeight > 0) { "invalid source size ${sourceWidth}x$sourceHeight" }
        require(tensorWidth > 0 && tensorHeight > 0) { "invalid tensor size ${tensorWidth}x$tensorHeight" }

        val contentWidth: Int
        val contentHeight: Int
        val padX: Int
        val padY: Int
        when (mode) {
            ResizeMode.LEGACY_STRETCH -> {
                contentHeight = tensorHeight
                contentWidth = (tensorHeight * LEGACY_ASPECT_W / LEGACY_ASPECT_H.toDouble())
                    .roundToInt().coerceIn(1, tensorWidth)
                padX = 0
                padY = 0
            }
            ResizeMode.LETTERBOX -> {
                val scale = min(tensorWidth.toDouble() / sourceWidth, tensorHeight.toDouble() / sourceHeight)
                // Integer content size so the plan matches the scaled bitmap exactly.
                contentWidth = (sourceWidth * scale).roundToInt().coerceIn(1, tensorWidth)
                contentHeight = (sourceHeight * scale).roundToInt().coerceIn(1, tensorHeight)
                padX = (tensorWidth - contentWidth) / 2
                padY = (tensorHeight - contentHeight) / 2
            }
        }

        return PreprocessPlan(
            sourceWidth = sourceWidth,
            sourceHeight = sourceHeight,
            letterbox = LetterboxInfo(
                tensorWidth = tensorWidth,
                tensorHeight = tensorHeight,
                contentWidth = contentWidth.toFloat(),
                contentHeight = contentHeight.toFloat(),
                padX = padX.toFloat(),
                padY = padY.toFloat()
            )
        )
    }
}

/**
 * Result of [FramePreprocessor.plan]: the [LetterboxInfo] handed to parsers, plus the source
 * frame size needed to map back to source pixels.
 *
 * Forward: source px → tensor px. Inverse: tensor px → source px.
 * Parsers keep producing normalized source coordinates through [LetterboxInfo.toNormalizedX]/[LetterboxInfo.toNormalizedY];
 * multiplying those by the source size gives the same result as [tensorToSourceX]/[tensorToSourceY].
 */
data class PreprocessPlan(
    val sourceWidth: Int,
    val sourceHeight: Int,
    val letterbox: LetterboxInfo
) {
    val contentWidth: Int get() = letterbox.contentWidth.toInt()
    val contentHeight: Int get() = letterbox.contentHeight.toInt()
    val padX: Int get() = letterbox.padX.toInt()
    val padY: Int get() = letterbox.padY.toInt()

    fun sourceToTensorX(sourceX: Double): Double =
        letterbox.padX + sourceX * letterbox.contentWidth / sourceWidth

    fun sourceToTensorY(sourceY: Double): Double =
        letterbox.padY + sourceY * letterbox.contentHeight / sourceHeight

    fun tensorToSourceX(tensorX: Double): Double = letterbox.toNormalizedX(tensorX) * sourceWidth

    fun tensorToSourceY(tensorY: Double): Double = letterbox.toNormalizedY(tensorY) * sourceHeight
}
