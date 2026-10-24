package expo.modules.camerax

import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * Parser for the current `try.tflite` oriented-bounding-box output (moved out of Detector).
 *
 * Expected output 0: `[1, 7, N]`, or `[1, 8400, 7]` which is transposed first. Channels are
 * `cx, cy, w, h, bowScore, stringScore, angleRad`, with geometry in tensor pixels.
 *
 * Per anchor: class = the higher of the two scores; bow scores get +0.3 before the
 * threshold. For each class, the highest-scoring box above [CONFIDENCE_THRESHOLD]
 * is kept (first one wins on ties).
 */
class LegacyObbParser : ModelOutputParser {

    companion object {
        const val CONFIDENCE_THRESHOLD = 0.6F
        const val BOW_CONFIDENCE_BOOST = 0.3f
        private const val NUM_CHANNELS = 7
        private const val CLS_BOW = 0
        private const val CLS_STRING = 1
    }

    internal data class OrientedBoundingBox(
        val x: Float,
        val y: Float,
        val height: Float,
        val width: Float,
        val conf: Float,
        val cls: Int,
        val angle: Float
    )

    override fun parse(outputs: Map<Int, OutputTensor>, letterbox: LetterboxInfo): ViolinDetection {
        val output = requireNotNull(outputs[0]) { "LegacyObbParser needs output tensor 0" }
        val outShape = output.shape
        require(outShape.size == 3 && outShape[1] > 0 && outShape[2] > 0) {
            "LegacyObbParser expects a rank-3 output, got ${outShape.contentToString()}"
        }

        // Handle output layout: [1, 7, N] vs [1, N, 7] (e.g., 8400)
        val (C, N, parsed) = if (outShape[1] == 8400 && outShape[2] == NUM_CHANNELS) {
            // Flattened [1, 8400, 7] → [1, 7, 8400]
            Triple(NUM_CHANNELS, 8400, transposeN7To7N(output.data, N = outShape[1], C = outShape[2]))
        } else {
            // Assume [1, 7, N]
            Triple(outShape[1], outShape[2], output.data)
        }
        require(C >= NUM_CHANNELS) { "LegacyObbParser needs $NUM_CHANNELS channels, got $C" }

        val bestBoxes = newBestBox(parsed, N)

        var bow: Quad? = null
        var strings: Quad? = null
        var bowConf = 0f
        var stringConf = 0f
        for (box in bestBoxes) {
            if (box.cls == CLS_BOW && box.conf > bowConf) {
                bow = rotatedRectToQuad(box, letterbox)
                bowConf = box.conf
            } else if (box.cls == CLS_STRING && box.conf > stringConf) {
                strings = rotatedRectToQuad(box, letterbox)
                stringConf = box.conf
            }
        }
        return ViolinDetection(bow, strings, bowConf, stringConf)
    }

    private fun transposeN7To7N(src: FloatArray, N: Int = 8400, C: Int = 7): FloatArray {
        val dst = FloatArray(C * N)
        var n = 0
        while (n < N) {
            val base = n * C
            var c = 0
            while (c < C) {
                dst[c * N + n] = src[base + c]
                c++
            }
            n++
        }
        return dst
    }

    internal fun newBestBox(array: FloatArray, N: Int): List<OrientedBoundingBox> {
        val out = mutableListOf<OrientedBoundingBox>()

        for (r in 0 until N) {
            val stringCnf = array[5 * N + r]
            val bowCnf = array[4 * N + r]
            val cls = if (stringCnf > bowCnf) CLS_STRING else CLS_BOW
            var cnf = max(stringCnf, bowCnf)
            if (cls == CLS_BOW) {
                cnf += BOW_CONFIDENCE_BOOST
            }

            if (cnf > CONFIDENCE_THRESHOLD) {
                val x = array[0 * N + r]
                val y = array[1 * N + r]
                val w = array[2 * N + r]
                val h = array[3 * N + r]
                val angle = array[6 * N + r]

                out.add(
                    OrientedBoundingBox(
                        x = x, y = y, height = h, width = w,
                        conf = cnf, cls = cls, angle = angle
                    )
                )
            }
        }

        return out
    }

    /** Corners of the rotated rect (TL, TR, BR, BL before rotation), mapped to normalized source coords. */
    private fun rotatedRectToQuad(box: OrientedBoundingBox, letterbox: LetterboxInfo): Quad {
        val halfW = box.width / 2
        val halfH = box.height / 2
        val cosA = cos(box.angle)
        val sinA = sin(box.angle)

        val corners = listOf(
            Pair(-halfW, -halfH),
            Pair(halfW, -halfH),
            Pair(halfW, halfH),
            Pair(-halfW, halfH)
        )
        return Quad(corners.map { (x, y) ->
            val xRot = x * cosA - y * sinA + box.x
            val yRot = x * sinA + y * cosA + box.y
            Quad.Corner(
                letterbox.toNormalizedX(xRot.toDouble()),
                letterbox.toNormalizedY(yRot.toDouble())
            )
        })
    }
}
