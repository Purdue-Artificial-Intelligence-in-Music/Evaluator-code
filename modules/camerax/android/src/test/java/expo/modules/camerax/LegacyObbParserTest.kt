package expo.modules.camerax

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

// Synthetic-tensor tests for LegacyObbParser.
// Anchors are written as (cx, cy, w, h, bowScore, stringScore, angleRad) in tensor pixels.

class LegacyObbParserTest {

    private val parser = LegacyObbParser()

    // Same placement Detector uses today: 360x640 content at the top-left of a 640x640 input.
    private val legacyLetterbox = LetterboxInfo(
        tensorWidth = 640,
        tensorHeight = 640,
        contentWidth = 360f,
        contentHeight = 640f
    )

    private data class Anchor(
        val cx: Float, val cy: Float, val w: Float, val h: Float,
        val bow: Float, val string: Float, val angle: Float = 0f
    )

    private fun Anchor.channels() = floatArrayOf(cx, cy, w, h, bow, string, angle)

    /** Builds a [1, 7, N] tensor (channel-major). */
    private fun channelMajor(anchors: List<Anchor>): OutputTensor {
        val n = anchors.size
        val data = FloatArray(7 * n)
        anchors.forEachIndexed { r, a ->
            a.channels().forEachIndexed { c, v -> data[c * n + r] = v }
        }
        return OutputTensor(intArrayOf(1, 7, n), data)
    }

    /** Builds a [1, 8400, 7] tensor (anchor-major); unused anchors are zero. */
    private fun anchorMajor8400(anchors: List<Anchor>): OutputTensor {
        val data = FloatArray(8400 * 7)
        anchors.forEachIndexed { r, a ->
            a.channels().forEachIndexed { c, v -> data[r * 7 + c] = v }
        }
        return OutputTensor(intArrayOf(1, 8400, 7), data)
    }

    private fun parse(tensor: OutputTensor) = parser.parse(mapOf(0 to tensor), legacyLetterbox)

    /** The pre-refactor Detector.rotatedRectToPoints formula, in the legacy 0..640 point space. */
    private fun legacyPoints(a: Anchor): List<Pair<Double, Double>> {
        val halfW = a.w / 2
        val halfH = a.h / 2
        val cosA = cos(a.angle)
        val sinA = sin(a.angle)
        return listOf(
            Pair(-halfW, -halfH), Pair(halfW, -halfH), Pair(halfW, halfH), Pair(-halfW, halfH)
        ).map { (x, y) ->
            val xRot = x * cosA - y * sinA + a.cx
            val yRot = x * sinA + y * cosA + a.cy
            Pair(xRot.toDouble() * 640f / 360f, yRot.toDouble())
        }
    }

    private fun assertMatchesLegacy(quad: Quad?, anchor: Anchor) {
        assertNotNull(quad)
        val expected = legacyPoints(anchor)
        quad!!.corners.forEachIndexed { i, c ->
            assertEquals("x of corner $i", expected[i].first, c.x * 640.0, 1e-6)
            assertEquals("y of corner $i", expected[i].second, c.y * 640.0, 1e-6)
        }
    }

    @Test
    fun noAnchorAboveThreshold_returnsEmptyDetection() {
        val result = parse(channelMajor(listOf(
            Anchor(100f, 100f, 50f, 20f, bow = 0.1f, string = 0.2f)
        )))

        assertTrue("Low-score anchors should produce no detection", result.isEmpty)
    }

    @Test
    fun bowScoreIsBoostedBeforeThreshold() {
        // 0.35 + 0.3 boost = 0.65 > 0.6, so this bow survives
        val result = parse(channelMajor(listOf(
            Anchor(100f, 100f, 50f, 20f, bow = 0.35f, string = 0.1f)
        )))

        assertNotNull("Bow at 0.35 should pass after the +0.3 boost", result.bow)
        assertNull(result.strings)
        assertEquals(0.65f, result.bowConfidence, 1e-6f)
    }

    @Test
    fun stringScoreIsNotBoosted() {
        // 0.55 is below 0.6 and strings get no boost
        val result = parse(channelMajor(listOf(
            Anchor(100f, 100f, 50f, 20f, bow = 0.1f, string = 0.55f)
        )))

        assertTrue("String at 0.55 should be rejected", result.isEmpty)
    }

    @Test
    fun highestScoringBoxPerClassIsKept() {
        val weakBow = Anchor(50f, 50f, 10f, 10f, bow = 0.5f, string = 0f)
        val strongBow = Anchor(150f, 150f, 10f, 10f, bow = 0.6f, string = 0f)
        val weakString = Anchor(60f, 300f, 200f, 80f, bow = 0f, string = 0.7f)
        val strongString = Anchor(180f, 320f, 200f, 80f, bow = 0f, string = 0.9f)

        val result = parse(channelMajor(listOf(weakBow, strongString, strongBow, weakString)))

        assertMatchesLegacy(result.bow, strongBow)
        assertMatchesLegacy(result.strings, strongString)
        assertEquals(0.9f, result.bowConfidence, 1e-6f)
        assertEquals(0.9f, result.stringConfidence, 1e-6f)
    }

    @Test
    fun equalScores_firstAnchorWins() {
        val first = Anchor(50f, 50f, 10f, 10f, bow = 0.5f, string = 0f)
        val second = Anchor(250f, 250f, 10f, 10f, bow = 0.5f, string = 0f)

        val result = parse(channelMajor(listOf(first, second)))

        assertMatchesLegacy(result.bow, first)
    }

    @Test
    fun rotatedBox_matchesLegacyCornerFormula() {
        val bow = Anchor(180f, 200f, 300f, 12f, bow = 0.8f, string = 0f, angle = 0.4f)
        val string = Anchor(170f, 330f, 140f, 90f, bow = 0f, string = 0.9f, angle = -0.2f)

        val result = parse(channelMajor(listOf(bow, string)))

        assertMatchesLegacy(result.bow, bow)
        assertMatchesLegacy(result.strings, string)
    }

    @Test
    fun cornersAreNormalizedToSourceBitmap() {
        // Axis-aligned box covering the whole 360x640 content area
        val result = parse(channelMajor(listOf(
            Anchor(180f, 320f, 360f, 640f, bow = 0f, string = 0.9f)
        )))

        val corners = result.strings!!.corners
        assertEquals(Quad.Corner(0.0, 0.0), corners[0])
        assertEquals(Quad.Corner(1.0, 0.0), corners[1])
        assertEquals(Quad.Corner(1.0, 1.0), corners[2])
        assertEquals(Quad.Corner(0.0, 1.0), corners[3])
    }

    @Test
    fun letterboxPaddingIsRemoved() {
        val padded = LetterboxInfo(
            tensorWidth = 640, tensorHeight = 640,
            contentWidth = 640f, contentHeight = 360f,
            padX = 0f, padY = 140f
        )
        val result = parser.parse(
            mapOf(0 to channelMajor(listOf(Anchor(320f, 320f, 640f, 360f, bow = 0f, string = 0.9f)))),
            padded
        )

        val corners = result.strings!!.corners
        assertEquals(0.0, corners[0].y, 1e-9)
        assertEquals(1.0, corners[2].y, 1e-9)
    }

    @Test
    fun anchorMajorLayout_givesSameResultAsChannelMajor() {
        val anchors = listOf(
            Anchor(180f, 200f, 300f, 12f, bow = 0.8f, string = 0f, angle = 0.4f),
            Anchor(170f, 330f, 140f, 90f, bow = 0f, string = 0.9f, angle = -0.2f)
        )

        val fromChannelMajor = parse(channelMajor(anchors))
        val fromAnchorMajor = parse(anchorMajor8400(anchors))

        assertEquals(fromChannelMajor, fromAnchorMajor)
    }

    @Test(expected = IllegalArgumentException::class)
    fun missingOutputZero_throws() {
        parser.parse(emptyMap(), legacyLetterbox)
    }

    @Test(expected = IllegalArgumentException::class)
    fun nonRank3Output_throws() {
        parse(OutputTensor(intArrayOf(1, 7), FloatArray(7)))
    }

    @Test(expected = IllegalArgumentException::class)
    fun tooFewChannels_throws() {
        parse(OutputTensor(intArrayOf(1, 5, 2), FloatArray(10)))
    }
}
