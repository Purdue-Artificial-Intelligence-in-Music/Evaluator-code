package expo.modules.videoanalyzer

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.view.Surface
import java.io.File

class VideoEncoder(
    private val outputFile: File,
    private val width: Int,
    private val height: Int,
    private val fps: Int
) {
    private val codec: MediaCodec
    private val muxer: MediaMuxer
    private var trackIndex = -1
    private var muxerStarted = false
    private var frameIndex = 0L

    private val surface: Surface
    private var surfaceCanvas: Canvas? = null
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    init {
        val format = MediaFormat.createVideoFormat("video/avc", width, height)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        format.setInteger(MediaFormat.KEY_BIT_RATE, width * height * 4)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

        codec = MediaCodec.createEncoderByType("video/avc")
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)

        surface = codec.createInputSurface() // Surface input
        codec.start()

        muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    }

    fun encodeFrame(bitmap: Bitmap) {
        // Scale to target size
        // Ideally encoder is initialized correctly and this step isn't executed
        val scaledBitmap = if (bitmap.width != width || bitmap.height != height) {
            Bitmap.createScaledBitmap(bitmap, width, height, true)
        } else bitmap

        // Draw bitmap to Surface
        surfaceCanvas = surface.lockCanvas(null)
        surfaceCanvas?.drawBitmap(scaledBitmap, 0f, 0f, paint)
        surface.unlockCanvasAndPost(surfaceCanvas)

        // If scaledBitmap was created, recyle the old bitmap
        if (scaledBitmap != bitmap) scaledBitmap.recycle()

        // Drain output
        drainEncoder()
        frameIndex++
    }

    private fun drainEncoder(endOfStream: Boolean = false) {
        if (endOfStream) {
            codec.signalEndOfInputStream()
        }

        val bufferInfo = MediaCodec.BufferInfo()
        while (true) {
            val outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
            when {
                outputBufferIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) break
                }
                outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (muxerStarted) throw RuntimeException("Format changed twice")
                    val newFormat = codec.outputFormat
                    trackIndex = muxer.addTrack(newFormat)
                    muxer.start()
                    muxerStarted = true
                }
                outputBufferIndex >= 0 -> {
                    val encodedData = codec.getOutputBuffer(outputBufferIndex)
                        ?: throw RuntimeException("encoderOutputBuffer $outputBufferIndex was null")

                    if (bufferInfo.size > 0 && muxerStarted) {
                        // set timestamps
                        bufferInfo.presentationTimeUs = computePresentationTime(frameIndex - 1)
                        
                        encodedData.position(bufferInfo.offset)
                        encodedData.limit(bufferInfo.offset + bufferInfo.size)
                        muxer.writeSampleData(trackIndex, encodedData, bufferInfo)
                    }

                    codec.releaseOutputBuffer(outputBufferIndex, false)

                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        break
                    }
                }
            }
        }
    }

    fun finish() {
        // flush results and stop execution, release resources
        drainEncoder(endOfStream = true)

        codec.stop()
        codec.release()

        muxer.stop()
        muxer.release()
    }

    private fun computePresentationTime(frameIndex: Long): Long {
        return frameIndex * 1_000_000L / fps
    }
}