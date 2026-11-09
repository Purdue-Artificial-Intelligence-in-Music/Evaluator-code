package expo.modules.camerax

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

// Geometry tests for FramePreprocessor: legacy stretch, letterbox, and the
// source px ↔ tensor px round trip (acceptance: within 1 px) for portrait, landscape and square.

class FramePreprocessorTest {

    private val legacy = FramePreprocessor(FramePreprocessor.ResizeMode.LEGACY_STRETCH)
    private val letterbox = FramePreprocessor(FramePreprocessor.ResizeMode.LETTERBOX)

    private data class Size(val w: Int, val h: Int)

    private val portrait = Size(1080, 1920)   // 9:16
    private val landscape = Size(960, 540)    // 16:9, the camera-only landscape frame seen on device
    private val square = Size(720, 720)
    private val frames = listOf(portrait, landscape, square, Size(1920, 1080), Size(540, 960), Size(1001, 563))

    // Non-square and non-640 inputs to show nothing depends on a fixed 640x640 tensor.
    private val tensors = listOf(Size(640, 640), Size(320, 320), Size(640, 384), Size(384, 640), Size(1024, 1024))

    private val roundTripTolerancePx = 1.0

    // ---- legacy stretch ----

    @Test
    fun legacy_defaultMode_isLegacyStretch() {
        assertEquals(FramePreprocessor.ResizeMode.LEGACY_STRETCH, FramePreprocessor().mode)
    }

    @Test
    fun legacy_640Tensor_matchesCurrentDetectorPlacement_forEveryFrameShape() {
        // Detector.PREPROCESS_LETTERBOX today: 360x640 content at (0,0) of a 640x640 input.
        val expected = LetterboxInfo(tensorWidth = 640, tensorHeight = 640, contentWidth = 360f, contentHeight = 640f)
        for (f in listOf(portrait, landscape, square)) {
            assertEquals("frame $f", expected, legacy.plan(f.w, f.h, 640, 640).letterbox)
        }
    }

    @Test
    fun legacy_scalesWithTensorSize_keepingPortraitRegionAtTopLeft() {
        val lb = legacy.plan(960, 540, 640, 384).letterbox
        assertEquals(216f, lb.contentWidth, 0f)  // 384 * 9 / 16
        assertEquals(384f, lb.contentHeight, 0f)
        assertEquals(0f, lb.padX, 0f)
        assertEquals(0f, lb.padY, 0f)
    }

    @Test
    fun legacy_landscapeFrame_isDistorted() {
        // Documents the known limitation: 16:9 is squeezed into 9:16.
        val plan = legacy.plan(landscape.w, landscape.h, 640, 640)
        val srcAspect = landscape.w.toDouble() / landscape.h
        val contentAspect = plan.contentWidth.toDouble() / plan.contentHeight
        assertTrue(srcAspect / contentAspect > 3.0)
    }

    // ---- letterbox ----

    @Test
    fun letterbox_landscape960x540_into640_isCenteredVertically() {
        val plan = letterbox.plan(960, 540, 640, 640)
        assertEquals(640, plan.contentWidth)
        assertEquals(360, plan.contentHeight)
        assertEquals(0, plan.padX)
        assertEquals(140, plan.padY)
    }

    @Test
    fun letterbox_portrait1080x1920_into640_isCenteredHorizontally() {
        val plan = letterbox.plan(1080, 1920, 640, 640)
        assertEquals(360, plan.contentWidth)
        assertEquals(640, plan.contentHeight)
        assertEquals(140, plan.padX)
        assertEquals(0, plan.padY)
    }

    @Test
    fun letterbox_square_fillsSquareTensor() {
        val plan = letterbox.plan(720, 720, 640, 640)
        assertEquals(640, plan.contentWidth)
        assertEquals(640, plan.contentHeight)
        assertEquals(0, plan.padX)
        assertEquals(0, plan.padY)
    }

    @Test
    fun letterbox_preservesAspect_fitsInside_andIsCentered_forAllFramesAndTensors() {
        for (f in frames) for (t in tensors) {
            val plan = letterbox.plan(f.w, f.h, t.w, t.h)
            val ctx = "frame $f tensor $t"
            assertTrue(ctx, plan.padX >= 0 && plan.padY >= 0)
            assertTrue(ctx, plan.padX + plan.contentWidth <= t.w)
            assertTrue(ctx, plan.padY + plan.contentHeight <= t.h)
            // One side fills the tensor exactly.
            assertTrue(ctx, plan.contentWidth == t.w || plan.contentHeight == t.h)
            // Centered to within one pixel (odd leftovers go to the right/bottom).
            assertTrue(ctx, abs((t.w - plan.contentWidth - plan.padX) - plan.padX) <= 1)
            assertTrue(ctx, abs((t.h - plan.contentHeight - plan.padY) - plan.padY) <= 1)
            // Aspect preserved up to integer rounding of the content size.
            val ideal = min(t.w.toDouble() / f.w, t.h.toDouble() / f.h)
            assertEquals(ctx, f.w * ideal, plan.contentWidth.toDouble(), 0.5 + 1e-9)
            assertEquals(ctx, f.h * ideal, plan.contentHeight.toDouble(), 0.5 + 1e-9)
        }
    }

    // ---- round trip (acceptance criterion) ----

    @Test
    fun roundTrip_sourceToTensorToSource_within1px_bothModes_allShapes() {
        for (pre in listOf(legacy, letterbox)) for (f in frames) for (t in tensors) {
            val plan = pre.plan(f.w, f.h, t.w, t.h)
            for ((sx, sy) in samplePoints(f)) {
                val bx = plan.tensorToSourceX(plan.sourceToTensorX(sx))
                val by = plan.tensorToSourceY(plan.sourceToTensorY(sy))
                val ctx = "${pre.mode} frame $f tensor $t point ($sx,$sy)"
                assertEquals(ctx, sx, bx, roundTripTolerancePx)
                assertEquals(ctx, sy, by, roundTripTolerancePx)
            }
        }
    }

    @Test
    fun roundTrip_normalizedParserPath_agreesWithInverse() {
        // Parsers use LetterboxInfo.toNormalized*; scaled by the source size it must equal tensorToSource*.
        for (pre in listOf(legacy, letterbox)) for (f in frames) {
            val plan = pre.plan(f.w, f.h, 640, 640)
            for ((sx, sy) in samplePoints(f)) {
                val tx = plan.sourceToTensorX(sx)
                val ty = plan.sourceToTensorY(sy)
                assertEquals(plan.tensorToSourceX(tx), plan.letterbox.toNormalizedX(tx) * f.w, 1e-9)
                assertEquals(plan.tensorToSourceY(ty), plan.letterbox.toNormalizedY(ty) * f.h, 1e-9)
            }
        }
    }

    @Test
    fun letterbox_integerContentRounding_staysWithin1pxOfIdealScale() {
        // The content size is rounded to whole pixels; the inverse uses that rounded size.
        // Compare against an ideal (unrounded) uniform-scale letterbox: worst case is the far frame edge.
        for (f in frames) for (t in tensors) {
            val plan = letterbox.plan(f.w, f.h, t.w, t.h)
            val ideal = min(t.w.toDouble() / f.w, t.h.toDouble() / f.h)
            val idealPadX = (t.w - f.w * ideal) / 2
            val idealPadY = (t.h - f.h * ideal) / 2
            for ((sx, sy) in samplePoints(f)) {
                // Tensor position under the ideal transform, mapped back with the real plan.
                val bx = plan.tensorToSourceX(idealPadX + sx * ideal)
                val by = plan.tensorToSourceY(idealPadY + sy * ideal)
                val ctx = "frame $f tensor $t point ($sx,$sy)"
                assertEquals(ctx, sx, bx, roundTripTolerancePx)
                assertEquals(ctx, sy, by, roundTripTolerancePx)
            }
        }
    }

    @Test
    fun padding_mapsOutsideSourceFrame() {
        val plan = letterbox.plan(960, 540, 640, 640)
        assertTrue(plan.tensorToSourceY(0.0) < 0.0)         // top pad
        assertTrue(plan.tensorToSourceY(639.0) > 540.0)     // bottom pad
        assertEquals(0.0, plan.tensorToSourceY(140.0), 1e-9)
        assertEquals(540.0, plan.tensorToSourceY(500.0), 1e-9)
    }

    // ---- end to end with the existing parser ----

    @Test
    fun letterbox_rotatedBoxThroughLegacyObbParser_returnsToSourceWithin1px() {
        // Uniform scale keeps angles, so a rotated source box maps to a rotated tensor box with the same angle.
        val angle = 0.4f
        for (f in listOf(portrait, landscape, square)) for (t in tensors) {
            val plan = letterbox.plan(f.w, f.h, t.w, t.h)
            val scale = plan.letterbox.contentWidth.toDouble() / f.w
            val cx = f.w * 0.4; val cy = f.h * 0.6
            val w = f.w * 0.2;  val h = f.h * 0.1

            val anchor = floatArrayOf(
                plan.sourceToTensorX(cx).toFloat(), plan.sourceToTensorY(cy).toFloat(),
                (w * scale).toFloat(), (h * scale).toFloat(),
                0.9f, 0.1f, angle
            )
            val quad = LegacyObbParser().parse(mapOf(0 to OutputTensor(intArrayOf(1, 7, 1), anchor)), plan.letterbox).bow
            assertNotNull(quad)

            val expected = rotatedCorners(cx, cy, w, h, angle.toDouble())
            quad!!.corners.zip(expected).forEach { (c, e) ->
                val ctx = "frame $f tensor $t"
                // Float32 tensor values plus the rounded content height limit precision; 1 px is the acceptance bound.
                assertEquals(ctx, e.first, c.x * f.w, roundTripTolerancePx)
                assertEquals(ctx, e.second, c.y * f.h, roundTripTolerancePx)
            }
        }
    }

    @Test
    fun legacy_axisAlignedBoxThroughLegacyObbParser_returnsToSourceWithin1px() {
        // Non-uniform stretch does not preserve angles, so only angle 0 round-trips as a rectangle.
        for (f in listOf(portrait, landscape, square)) {
            val plan = legacy.plan(f.w, f.h, 640, 640)
            val sxScale = plan.letterbox.contentWidth.toDouble() / f.w
            val syScale = plan.letterbox.contentHeight.toDouble() / f.h
            val cx = f.w * 0.5; val cy = f.h * 0.3
            val w = f.w * 0.3;  val h = f.h * 0.2

            val anchor = floatArrayOf(
                plan.sourceToTensorX(cx).toFloat(), plan.sourceToTensorY(cy).toFloat(),
                (w * sxScale).toFloat(), (h * syScale).toFloat(),
                0.1f, 0.9f, 0f
            )
            val quad = LegacyObbParser().parse(mapOf(0 to OutputTensor(intArrayOf(1, 7, 1), anchor)), plan.letterbox).strings
            assertNotNull(quad)
            quad!!.corners.zip(rotatedCorners(cx, cy, w, h, 0.0)).forEach { (c, e) ->
                assertEquals("frame $f", e.first, c.x * f.w, roundTripTolerancePx)
                assertEquals("frame $f", e.second, c.y * f.h, roundTripTolerancePx)
            }
        }
    }

    // ---- validation ----

    @Test(expected = IllegalArgumentException::class)
    fun zeroSourceSize_throws() {
        letterbox.plan(0, 540, 640, 640)
    }

    @Test(expected = IllegalArgumentException::class)
    fun zeroTensorSize_throws() {
        legacy.plan(960, 540, 640, 0)
    }

    // ---- helpers ----

    private fun samplePoints(f: Size): List<Pair<Double, Double>> = listOf(
        0.0 to 0.0,
        f.w.toDouble() to f.h.toDouble(),
        f.w.toDouble() to 0.0,
        0.0 to f.h.toDouble(),
        f.w / 2.0 to f.h / 2.0,
        f.w * 0.137 to f.h * 0.861,
        f.w - 0.5 to 0.5
    )

    /** Same corner order and rotation as LegacyObbParser (TL, TR, BR, BL before rotation). */
    private fun rotatedCorners(cx: Double, cy: Double, w: Double, h: Double, a: Double): List<Pair<Double, Double>> {
        val hw = w / 2; val hh = h / 2
        return listOf(-hw to -hh, hw to -hh, hw to hh, -hw to hh).map { (x, y) ->
            (x * cos(a) - y * sin(a) + cx) to (x * sin(a) + y * cos(a) + cy)
        }
    }
}
