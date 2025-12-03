package expo.modules.camerax

/**
 * Shared utilities for the MediaPipe-style LiteRT pipelines (hand/pose).
 * Functions will be added iteratively (resizePad, anchor decode, NMS, affine transforms).
 */
object MediapipeLiteRtUtil {
    /**
     * Minimal NPY loader for float arrays (supports little-endian f4, fortran_order=False).
     */
    fun loadNpyFloatArray(assetManager: android.content.res.AssetManager, assetName: String): FloatArray? {
        assetManager.open(assetName).use { input ->
            val magic = ByteArray(6)
            if (input.read(magic) != 6 || !magic.contentEquals(byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte()))) {
                return null
            }
            val ver = ByteArray(2)
            if (input.read(ver) != 2) return null
            val headerLenBytes = ByteArray(2)
            if (input.read(headerLenBytes) != 2) return null
            val headerLen = java.nio.ByteBuffer.wrap(headerLenBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).short.toInt()
            val headerBytes = ByteArray(headerLen)
            if (input.read(headerBytes) != headerLen) return null
            val header = String(headerBytes)
            if (!header.contains("'descr':") || !header.contains("f4") || header.contains("True")) return null

            val shape = Regex("shape': \\(([^)]*)\\)").find(header)?.groupValues?.get(1)?.trim()
            val dims = if (shape.isNullOrEmpty()) {
                intArrayOf()
            } else {
                shape.split(",").mapNotNull { it.trim().toIntOrNull() }.toIntArray()
            }
            val count = if (dims.isEmpty()) 0 else dims.fold(1) { acc, v -> acc * v }
            if (count <= 0) return null

            val dataBytes = count * 4
            val buf = ByteArray(dataBytes)
            var off = 0
            while (off < dataBytes) {
                val r = input.read(buf, off, dataBytes - off)
                if (r <= 0) break
                off += r
            }
            if (off != dataBytes) return null
            val bb = java.nio.ByteBuffer.wrap(buf).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            val out = FloatArray(count)
            var i = 0
            while (i < count) {
                out[i] = bb.float
                i++
            }
            return out
        }
    }

    data class ResizeResult(
        val bitmap: android.graphics.Bitmap,
        val scale: Float,
        val padX: Int,
        val padY: Int,
        val targetWidth: Int,
        val targetHeight: Int,
        val originalWidth: Int,
        val originalHeight: Int
    )

    /**
        * Letterbox resize to target size while preserving aspect ratio.
        * Returns the resized/padded bitmap plus scale/pad used for back-projection.
        */
    fun resizePadTo(
        src: android.graphics.Bitmap,
        dstWidth: Int,
        dstHeight: Int
    ): ResizeResult {
        val srcW = src.width
        val srcH = src.height
        val scale = kotlin.math.min(dstWidth.toFloat() / srcW, dstHeight.toFloat() / srcH)
        val scaledW = (srcW * scale).toInt()
        val scaledH = (srcH * scale).toInt()

        val scaled = android.graphics.Bitmap.createScaledBitmap(src, scaledW, scaledH, true)
        val padded = android.graphics.Bitmap.createBitmap(dstWidth, dstHeight, android.graphics.Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(padded)
        val padX = (dstWidth - scaledW) / 2
        val padY = (dstHeight - scaledH) / 2
        canvas.drawBitmap(scaled, padX.toFloat(), padY.toFloat(), null)

        return ResizeResult(
            bitmap = padded,
            scale = scale,
            padX = padX,
            padY = padY,
            targetWidth = dstWidth,
            targetHeight = dstHeight,
            originalWidth = srcW,
            originalHeight = srcH
        )
    }

    data class Detection(
        val score: Float,
        val x0: Float,
        val y0: Float,
        val x1: Float,
        val y1: Float,
        val keypoints: FloatArray? = null
    )

    /**
        * Decode box predictions using anchors.
        * anchors is flatten [numAnchors * 4] (x_center, y_center, w, h) in normalized coords.
        * coords is flatten [numAnchors * coordsPerAnchor] with layout [cx, cy, w, h, k0x, k0y, ...].
        */
    fun decodeWithAnchors(
        coords: FloatArray,
        scores: FloatArray,
        anchors: FloatArray,
        numAnchors: Int,
        coordsPerAnchor: Int,
        scoreThreshold: Float,
        inputW: Int,
        inputH: Int
    ): List<Detection> {
        if (anchors.size < numAnchors * 4) return emptyList()
        val out = ArrayList<Detection>()
        var idx = 0
        while (idx < numAnchors) {
            val score = scores.getOrNull(idx) ?: 0f
            if (score >= scoreThreshold) {
                val base = idx * coordsPerAnchor
                val ax = anchors[idx * 4]
                val ay = anchors[idx * 4 + 1]
                val aw = anchors[idx * 4 + 2]
                val ah = anchors[idx * 4 + 3]

                val cx = coords[base + 0] / inputW * aw + ax
                val cy = coords[base + 1] / inputH * ah + ay
                val w = coords[base + 2] / inputW * aw
                val h = coords[base + 3] / inputH * ah

                val x0 = cx - w / 2f
                val y0 = cy - h / 2f
                val x1 = cx + w / 2f
                val y1 = cy + h / 2f

                // Copy any extra keypoints if present.
                val kpCount = (coordsPerAnchor - 4) / 2
                val keypoints =
                    if (kpCount > 0) FloatArray(kpCount * 2) else null
                if (keypoints != null) {
                    for (k in 0 until kpCount) {
                        val kx = coords[base + 4 + k * 2] / inputW * aw + ax
                        val ky = coords[base + 4 + k * 2 + 1] / inputH * ah + ay
                        keypoints[k * 2] = kx
                        keypoints[k * 2 + 1] = ky
                    }
                }
                out.add(Detection(score, x0, y0, x1, y1, keypoints))
            }
            idx++
        }
        return out
    }

    /**
        * Simple NMS on normalized boxes.
        */
    fun nms(
        detections: List<Detection>,
        iouThreshold: Float,
        maxDetections: Int = 4
    ): List<Detection> {
        val sorted = detections.sortedByDescending { it.score }.toMutableList()
        val kept = mutableListOf<Detection>()
        while (sorted.isNotEmpty() && kept.size < maxDetections) {
            val first = sorted.removeAt(0)
            kept.add(first)
            val it = sorted.iterator()
            while (it.hasNext()) {
                val det = it.next()
                val iou = iou(first, det)
                if (iou > iouThreshold) {
                    it.remove()
                }
            }
        }
        return kept
    }

    private fun iou(a: Detection, b: Detection): Float {
        val x0 = kotlin.math.max(a.x0, b.x0)
        val y0 = kotlin.math.max(a.y0, b.y0)
        val x1 = kotlin.math.min(a.x1, b.x1)
        val y1 = kotlin.math.min(a.y1, b.y1)
        val interW = kotlin.math.max(0f, x1 - x0)
        val interH = kotlin.math.max(0f, y1 - y0)
        val inter = interW * interH
        val areaA = (a.x1 - a.x0) * (a.y1 - a.y0)
        val areaB = (b.x1 - b.x0) * (b.y1 - b.y0)
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    /**
     * Compute square ROI corners for pose from keypoints 2->3 (bottom->top) with rotation and scale.
     * Returns float array of size 8: tl(x,y), bl(x,y), tr(x,y), br(x,y).
     */
    fun computePoseRoiCorners(
        detection: Detection,
        imageWidth: Int,
        imageHeight: Int,
        keypointStartIdx: Int = 2,
        keypointEndIdx: Int = 3,
        boxScale: Float = 1.5f,
        rotationOffsetRad: Float = (Math.PI.toFloat() / 2f)
    ): FloatArray? {
        val kps = detection.keypoints ?: return null
        val kpCount = kps.size / 2
        if (kpCount <= keypointEndIdx) return null
        val sx = kps[keypointStartIdx * 2] * imageWidth
        val sy = kps[keypointStartIdx * 2 + 1] * imageHeight
        val ex = kps[keypointEndIdx * 2] * imageWidth
        val ey = kps[keypointEndIdx * 2 + 1] * imageHeight

        val dx = sx - ex
        val dy = sy - ey
        val dist = kotlin.math.sqrt(dx * dx + dy * dy)
        if (dist == 0f) return null

        val angle = kotlin.math.atan2(dy, dx) - rotationOffsetRad
        val xc = sx
        val yc = sy
        val w = dist * 2f * boxScale
        val h = w

        val cosA = kotlin.math.cos(angle)
        val sinA = kotlin.math.sin(angle)
        val hw = w / 2f
        val hh = h / 2f

        // Order: TL, BL, TR, BR
        val pts = floatArrayOf(
            -hw, -hh,
            -hw, hh,
            hw, -hh,
            hw, hh
        )
        for (i in 0 until 4) {
            val x = pts[i * 2]
            val y = pts[i * 2 + 1]
            pts[i * 2] = x * cosA - y * sinA + xc
            pts[i * 2 + 1] = x * sinA + y * cosA + yc
        }
        return pts
    }

    /**
     * Build an affine matrix mapping source ROI corners to a destination rect (0,0)-(dstW,dstH).
     * Returns pair of (forward, inverse) matrices.
     */
    fun buildAffineFromRoi(
        roiCorners: FloatArray,
        dstW: Int,
        dstH: Int
    ): Pair<android.graphics.Matrix, android.graphics.Matrix?> {
        val src = floatArrayOf(
            roiCorners[0], roiCorners[1],
            roiCorners[2], roiCorners[3],
            roiCorners[4], roiCorners[5],
            roiCorners[6], roiCorners[7]
        )
        val dst = floatArrayOf(
            0f, 0f,
            0f, dstH.toFloat(),
            dstW.toFloat(), 0f,
            dstW.toFloat(), dstH.toFloat()
        )
        val forward = android.graphics.Matrix()
        forward.setPolyToPoly(src, 0, dst, 0, 4)
        val inverse = android.graphics.Matrix()
        val hasInv = forward.invert(inverse)
        return Pair(forward, if (hasInv) inverse else null)
    }

    /**
     * Warp the source bitmap into destination size using the provided matrix.
     */
    fun warpBitmap(
        src: android.graphics.Bitmap,
        matrix: android.graphics.Matrix,
        dstW: Int,
        dstH: Int
    ): android.graphics.Bitmap {
        val out = android.graphics.Bitmap.createBitmap(dstW, dstH, android.graphics.Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(out)
        canvas.drawBitmap(src, matrix, null)
        return out
    }

    /**
     * Map landmarks from cropped space back to original using inverse matrix.
     */
    fun mapLandmarksBack(
        landmarks: FloatArray,
        inverse: android.graphics.Matrix
    ): FloatArray {
        val pts = landmarks.copyOf()
        inverse.mapPoints(pts)
        return pts
    }
}
