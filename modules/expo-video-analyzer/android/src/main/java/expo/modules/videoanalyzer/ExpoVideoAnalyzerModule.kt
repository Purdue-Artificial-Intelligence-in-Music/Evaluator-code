package expo.modules.videoanalyzer

import android.util.Log
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.Promise
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlinx.coroutines.*
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.InputStream
import android.content.Context
import android.content.res.AssetManager
import java.io.File
import java.io.FileOutputStream
import android.graphics.Bitmap.CompressFormat
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.ReturnCode
import android.os.Environment

class ExpoVideoAnalyzerModule : Module() {
    private var detector: Detector? = null
    private var isOpenCVInitialized = false
    private var initializationAttempted = false
    
    override fun definition() = ModuleDefinition {
        Name("ExpoVideoAnalyzer")
        
        // function for testing, remove later
        Function("getStatus") {
            val status = StringBuilder()
            status.append("Video Analyzer Module Connected | ")
            status.append("OpenCV: ${if (isOpenCVInitialized) "Ready" else "Not Ready"} | ")
            status.append("Detector: ${if (detector != null) "Ready" else "Not Ready"}")
            return@Function status.toString()
        }
        
        AsyncFunction("initialize") { promise: Promise ->
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    if (!initializationAttempted) {
                        initializationAttempted = true
                        
                        // try initializing OpenCV
                        isOpenCVInitialized = try {
                            org.opencv.android.OpenCVLoader.initDebug()
                        } catch (e: Exception) {
                            println("OpenCV initialization failed: ${e.message}")
                            Log.d("init", "OpenCV initialization failed: ${e.message}")
                            false
                        }
                        

                        if (isOpenCVInitialized) {
                            initializeDetector()
                        }
                    }
                    
                    withContext(Dispatchers.Main) {
                        promise.resolve(mapOf(
                            "success" to true,
                            "openCV" to isOpenCVInitialized,
                            "detector" to (detector != null),
                            "initializationMethod" to if (isOpenCVInitialized) "successful" else "failed"
                        ))
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        promise.reject("INIT_ERROR", "Initialization failed: ${e.message}", e)
                    }
                }
            }
        }
        
        AsyncFunction("processFrame") { videoUri: String, promise: Promise ->
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    if (!isOpenCVInitialized) {
                        withContext(Dispatchers.Main) {
                            promise.reject("OPENCV_NOT_READY", "OpenCV not initialized. Call initialize() first.", null)
                        }
                        return@launch
                    }
                    
                    if (detector == null) {
                        withContext(Dispatchers.Main) {
                            promise.reject("DETECTOR_NOT_READY", "Detector not initialized. Call initialize() first.", null)
                        }
                        return@launch
                    }
                    val bitmapImage = getBitmapFromAssets(appContext.reactContext!!, "Sample Input.png")
//                    val bitmapImage = extractFrameFromVideo(videoUri, 10_000_000L) // convert a frame to bitmap
                    if (bitmapImage == null) {
                        withContext(Dispatchers.Main) {
                            promise.reject("ASSET_ERROR", "Failed to load image from assets", null)
                        }
                        Log.d("bitmapImage", "bitmapImage is null")
                        return@launch
                    }
                    val result = detector!!.process_frame(bitmapImage)
                    

                    Log.d("result", "Detector result: $result")
                    val response = mapOf(
                        "success" to true,
                        "classification" to (result.classification ?: -1),
                        "angle" to (result.angle ?: 0),
                        "hasBow" to (result.bow != null),
                        "hasString" to (result.string != null),
                        "bowPoints" to if (result.bow != null) result.bow!!.size else 0,
                        "stringPoints" to if (result.string != null) result.string!!.size else 0,
                        "imageWidth" to bitmapImage.width,
                        "imageHeight" to bitmapImage.height
                    )
                    
                    // clear bitmap storage
                    bitmapImage.recycle()
                    
                    withContext(Dispatchers.Main) {
                        promise.resolve(response)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        promise.reject("PROCESS_ERROR", "Frame processing failed: ${e.message}", e)
                    }
                }
            }
        }
        
        // Function only for testing, remove later
        AsyncFunction("openVideo") { videoUri: String, promise: Promise ->
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val result = openAndValidateVideo(videoUri)
                    withContext(Dispatchers.Main) {
                        promise.resolve(result)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        promise.reject("VIDEO_OPEN_ERROR", "Failed to open video: ${e.message}", e)
                    }
                }
            }
        }

        // TODO not implemented yet
        AsyncFunction("resetDetector") { promise: Promise ->
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    detector?.let {
                        // add logic to reset the detector's internal state
                        // For example, clearing historical data, resetting counters, etc
                    }
                    
                    withContext(Dispatchers.Main) {
                        promise.resolve(mapOf(
                            "success" to true,
                            "message" to "Detector reset successfully"
                        ))
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        promise.reject("RESET_ERROR", "Failed to reset detector: ${e.message}", e)
                    }
                }
            }
        }

        // This is the main function that is being called and used for video processing right now,
        // NOT optimal. It loops through the video, extracts frames into bitmap format, calls
        // detect() and drawPointsOnBitmap(), saves annotated frame, then uses FFmpeg to recollect
        // back into a video.
        AsyncFunction("processVideoComplete") { videoUri: String, promise: Promise ->
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    if (!isOpenCVInitialized || detector == null) {
                        withContext(Dispatchers.Main) {
                            promise.reject("NOT_INITIALIZED", "OpenCV or detector not initialized", null)
                        }
                        return@launch
                    }

                    // Create temp directory for annotated frames
                    val tempDir = File(appContext.reactContext!!.cacheDir, "video_frames_${System.currentTimeMillis()}")
                    tempDir.mkdirs()

                    // Outputpath for final video. Use public directory to be able to access later.
                    val publicMoviesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
                    val outputPath = File(publicMoviesDir, "processed_video_${System.currentTimeMillis()}.mp4").absolutePath

                    val framePaths = mutableListOf<String>()
                    var frameIndex = 0

                    Log.d("ProcessVideo", "Starting video processing for: $videoUri")
                    Log.d("ProcessVideo", "Output path will be: $outputPath")

                    // Use processVideoStream funciton to loop through and annotate frames, then
                    // save processed frame to disk.
                    val success = processVideoStream(videoUri, 15) { annotatedFrame, timeUs ->
                        try {
                            val framePath = File(tempDir, String.format("frame_%06d.jpg", frameIndex))

                            FileOutputStream(framePath).use { out ->
                                annotatedFrame.compress(CompressFormat.JPEG, 85, out)
                            }

                            // Track the path to which the file has been written to
                            framePaths.add(framePath.absolutePath)
                            frameIndex++

                            Log.d("SaveFrame", "Saved frame $frameIndex")

                        } catch (e: Exception) {
                            Log.e("SaveFrame", "Failed to save frame $frameIndex: ${e.message}")
                        } finally {
                            annotatedFrame.recycle()
                        }
                    }

                    if (success && framePaths.isNotEmpty()) {
                        // Use FFmpeg to construct video from saved frames

                        Log.d("ProcessVideo", "Combining ${framePaths.size} frames to video")
                        val videoSuccess = combineFramesToVideo(tempDir, outputPath, 30)

                        withContext(Dispatchers.Main) {
                            val resultMap: Map<String, Any> = if (videoSuccess && File(outputPath).exists()) {
                                mapOf(
                                    "success" to true,
                                    "outputPath" to outputPath,
                                    "frameCount" to framePaths.size
                                )
                            } else {
                                mapOf(
                                    "success" to false,
                                    "error" to "Failed to create video"
                                )
                            }
                            if (resultMap["success"] as Boolean) {
                                promise.resolve(resultMap)
                            } else {
                                promise.reject("VIDEO_CREATION_ERROR", "Failed to create video", null)
                            }
                        }
                    } else {
                        withContext(Dispatchers.Main) {
                            promise.reject("FRAME_SAVE_ERROR", "Failed to save frames", null)
                        }
                    }

                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        promise.reject("PROCESS_ERROR", "Video processing failed: ${e.message}", e)
                    }
                }
            }
        }

        // Helper function
        AsyncFunction("checkFileExists") { filePath: String, promise: Promise ->
            try {
                val file = File(filePath)
                val resultMap: Map<String, Any> = mapOf(
                    "exists" to file.exists(),
                    "size" to file.length(),
                    "path" to file.absolutePath,
                    "canRead" to file.canRead()
                )
                promise.resolve(resultMap)
            } catch (e: Exception) {
                promise.reject("CHECK_ERROR", e.message, e)
            }
        }

        // For testing. Verifies if FFmpeg works.
        AsyncFunction("testFFmpeg") { promise: Promise ->
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    Log.d("FFmpegTest", "Testing FFmpeg availability...")

                    // Verify that FFmpegKit works
                    val session = FFmpegKit.execute("-version")

                    Log.d("FFmpegTest", "FFmpeg version command completed")
                    Log.d("FFmpegTest", "Return code: ${session.returnCode}")
                    Log.d("FFmpegTest", "Output: ${session.output}")

                    withContext(Dispatchers.Main) {
                        promise.resolve(mapOf(
                            "success" to true,
                            "returnCode" to session.returnCode.value,
                            "output" to session.output
                        ))
                    }
                } catch (e: Exception) {
                    Log.e("FFmpegTest", "FFmpeg test failed: ${e.message}")
                    Log.e("FFmpegTest", "Stack trace: ${e.stackTrace.contentToString()}")

                    withContext(Dispatchers.Main) {
                        promise.reject("FFMPEG_TEST_ERROR", "FFmpeg test failed: ${e.message}", e)
                    }
                }
            }
        }
    }
    

    private fun initializeDetector() {
        try {
            detector = Detector(appContext.reactContext!!)
            println("Detector initialized successfully")
            Log.d("init", "Detector initialized successfully")
        } catch (e: Exception) {
            println("Failed to initialize detector: ${e.message}")
            Log.d("init error", "Failed to initialize detector: ${e.message}")
            detector = null
        }
    }
    
    // Function only for testing, remove later
    private fun openAndValidateVideo(videoUri: String): Map<String, Any> {
        val retriever = MediaMetadataRetriever()
        
        return try {
            println("Validating video: $videoUri")
            Log.d("video uri", "Validating video: $videoUri")
            retriever.setDataSource(appContext.reactContext, Uri.parse(videoUri))
            
            val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            val frameRate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toFloatOrNull() ?: 0f
            val bitrate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE)?.toLongOrNull() ?: 0L
            
            mapOf(
                "success" to true,
                "message" to "Video opened and validated successfully",
                "duration" to duration,
                "width" to width,
                "height" to height,
                "frameRate" to frameRate,
                "bitrate" to bitrate,
                "videoUri" to videoUri,
                "durationSeconds" to (duration / 1000.0)
            )
            
        } catch (e: Exception) {
            mapOf(
                "success" to false,
                "error" to (e.message ?: "Unknown error"),
                "videoUri" to videoUri
            )
        } finally {
            retriever.release()
        }
    }

    private fun extractFrameFromVideo(videoUri: String, timeUs: Long = 10_000_000L): Bitmap? {
        val retriever = MediaMetadataRetriever()
        var originalBitmap: Bitmap? = null

        return try {
            println("Extracting frame from video: $videoUri at time: ${timeUs}μs")
            Log.d("VideoFrame", "Extracting frame from video: $videoUri at time: ${timeUs}μs")

            val uri = Uri.parse(videoUri)
            retriever.setDataSource(appContext.reactContext, uri)

            originalBitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            if (originalBitmap == null) {
                println("Failed to extract frame from video at time ${timeUs}μs")
                return null
            }

            println("Extracted frame: ${originalBitmap.width}x${originalBitmap.height}, config: ${originalBitmap.config}")
            Log.d("Extract frame", "Extracted frame: ${originalBitmap.width}x${originalBitmap.height}, config: ${originalBitmap.config}")

            // bitmap has to be in ARGB_8888 format
            val processedBitmap = if (originalBitmap.config != Bitmap.Config.ARGB_8888) {
                println("Converting bitmap format from ${originalBitmap.config} to ARGB_8888")
                Log.d("bitmap8888", "Converting bitmap format from ${originalBitmap.config} to ARGB_8888")
                val convertedBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, false)
                originalBitmap.recycle()
                convertedBitmap
            } else {
                originalBitmap
            }

            println("Frame ready for processing: ${processedBitmap.width}x${processedBitmap.height}")
            Log.d("extract frame", "Frame ready for processing: ${processedBitmap.width}x${processedBitmap.height}")
            processedBitmap

        } catch (e: Exception) {
            originalBitmap?.recycle()
            println("Error extracting frame: ${e.message}")
            Log.d("extract frame", "Error extracting frame: ${e.message}")
            null
        } finally {
            retriever.release()
        }
    }

    // Currently will send bitmaps back to run on MainActivity.
    // Will need adjusted to instead send back images somewhere else or to compose them into an array
    private fun getVideoAnnotations(videoURI: String, targetFPS: Int = 30):
            MutableList<Pair<Detector.YoloResults, Long>> {
        val results = mutableListOf<Pair<Detector.YoloResults, Long>>()
        // Set time delta using fps
        var timeUs: Long = 0
        val timeDelta: Long = (1000 / targetFPS) * 1000L // Convert milliseconds to microseconds
        // Get frame by time (using time delta to loop over)
        var bitmapImage: Bitmap? = extractFrameFromVideo(videoURI, timeUs)
        while (bitmapImage != null) {
            //val bitmapImage = getBitmapFromAssets(MainActivity.applicationContext(), "Sample Input.png")
            //evaluator.createInterpreter(MainActivity.applicationContext())
            try {
                //Tasks.await(detector.initializeTask)
                Log.d("Interpreter", "TfLite.initialize() completed successfully")
                Log.d("Timing", "Current time is ${(timeUs.toDouble() / 1000000.0)}")
                bitmapImage.let {
                    detector!!.modelReadyLatch.await()

                    results.add(Pair(detector!!.detect(it), timeUs))
                }
                timeUs += timeDelta
                bitmapImage = extractFrameFromVideo(videoURI, timeUs)
            } catch (e: Exception) {
                Log.e("Interpreter", "Error during evaluation", e)
            }
        }

        return results
    }


    /*
     * Returns a list of <Bitmap, Long> pairs from a passed in video.
     * Uses given FPS to determine how many frames to take
     * WARNING: will store a lot of frames at once, only use for shorter videos/testing
     *          need another function that updates a synchronized stack of frames
     */
    private fun getVideoBitmaps(videoURI: String, targetFPS: Int = 30, maxTime: Int = 3):
            MutableList<Pair<Bitmap, Long>> {

        val results = mutableListOf<Pair<Bitmap, Long>>()
        // Set time delta using fps
        var timeUs: Long = 0
        var maxTimeLong: Long = 1000000L * maxTime
        val timeDelta: Long = (1000 / targetFPS) * 1000L // Convert milliseconds to microseconds
        // Get frame by time (using time delta to loop over)
        var bitmapImage: Bitmap? = extractFrameFromVideo(videoURI, timeUs)
        while ((timeUs < maxTimeLong) && (bitmapImage != null)) {
            //val bitmapImage = getBitmapFromAssets(MainActivity.applicationContext(), "Sample Input.png")
            //evaluator.createInterpreter(MainActivity.applicationContext())
            try {
                //Tasks.await(detector.initializeTask)
                Log.d("Interpreter", "TfLite.initialize() completed successfully")
                Log.d("Timing", "Current time is ${(timeUs.toDouble() / 1000000.0)}")
                bitmapImage.let {
                    detector!!.modelReadyLatch.await()

                    val result = detector!!.detect(it)
                    results.add(Pair(detector!!.drawPointsOnBitmap(it, result), timeUs))
                }
                timeUs += timeDelta
                bitmapImage = extractFrameFromVideo(videoURI, timeUs)
            } catch (e: Exception) {
                Log.e("Interpreter", "Error during evaluation", e)
            }
        }

        return results
    }

    // Helper function. It loops through the video, extracts frames into bitmap format, calls
    // detect() and drawPointsOnBitmap() from detector to get annotated frames.
    private fun processVideoStream(
        videoURI: String,
        targetFPS: Int = 15, // testing with lower fps rate
        onFrameProcessed: (Bitmap, Long) -> Unit
    ): Boolean {
        val retriever = MediaMetadataRetriever()

        try {
            retriever.setDataSource(appContext.reactContext, Uri.parse(videoURI))
            val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
            val timeDelta = (1000 / targetFPS) * 1000L

            // Loop through frames
            var timeUs = 0L
            while (timeUs < duration * 1000) {
                try {
                    // Extract specific frame
                    val frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    if (frame != null) {
                        // convert format to ARGB_8888
                        val processedFrame = if (frame.config != Bitmap.Config.ARGB_8888) {
                            val convertedBitmap = frame.copy(Bitmap.Config.ARGB_8888, false)
                            frame.recycle()
                            convertedBitmap
                        } else {
                            frame
                        }

                        // Annotate frame using Detector
                        val result = detector!!.detect(processedFrame)
                        val annotatedFrame = detector!!.drawPointsOnBitmap(processedFrame, result)

                        onFrameProcessed(annotatedFrame, timeUs)

                        processedFrame.recycle()
                    }
                } catch (e: Exception) {
                    Log.e("FrameProcess", "Error at time $timeUs: ${e.message}")
                }
                timeUs += timeDelta
            }
            return true
        } catch (e: Exception) {
            Log.e("VideoProcess", "Failed to process video: ${e.message}")
            return false
        } finally {
            retriever.release()
        }
    }

    // Helper function. Compines frames into video using FFmpeg
    private fun combineFramesToVideo(
        frameDir: File,
        outputVideoPath: String,
        fps: Int = 15
    ): Boolean {
        try {
            // Check if file exists
            val frameFiles = frameDir.listFiles { file -> file.name.endsWith(".jpg") }
            Log.d("FFmpeg", "Found ${frameFiles?.size} frame files in ${frameDir.absolutePath}")
            // Check permissions
            val outputFile = File(outputVideoPath)
            val parentDir = outputFile.parentFile
            Log.d("FFmpeg", "Output directory writable: ${parentDir?.canWrite()}")
            Log.d("FFmpeg", "Output path: $outputVideoPath")
            // FFmpeg command：Combine the frame sequence into a video (order depends on filenames)
            val command = "-y -r $fps -i ${frameDir.absolutePath}/frame_%06d.jpg -vcodec libx264 $outputVideoPath"
//            val command = "-y -framerate $fps -i ${frameDir.absolutePath}/frame_%06d.jpg " +
//                    "-c:v libx264 -pix_fmt yuv420p -crf 23 $outputVideoPath"

            Log.d("FFmpeg", "Executing command: $command")

            val session = FFmpegKit.execute(command)

            Log.d("FFmpeg", "Return code: ${session.returnCode}")
            Log.d("FFmpeg", "State: ${session.state}")
            Log.d("FFmpeg", "Output: ${session.output}")
            if (session.failStackTrace != null) {
                Log.e("FFmpeg", "Fail stack trace: ${session.failStackTrace}")
            }

            return if (ReturnCode.isSuccess(session.returnCode)) {
                Log.d("FFmpeg", "Video creation successful")
                val outputExists = File(outputVideoPath).exists()
                Log.d("FFmpeg", "Output file exists: $outputExists")
                // !! Cleanup frames
                cleanupFrames(frameDir)
                outputExists
            } else {
                Log.e("FFmpeg", "Video creation failed with code: ${session.returnCode}")
                Log.e("FFmpeg", "Video creation failed: ${session.failStackTrace}")
                false
            }

        } catch (e: Exception) {
            Log.e("FFmpeg", "Exception in combineFramesToVideo: ${e.message}")
            Log.e("FFmpeg", "Stack trace: ${e.stackTrace.contentToString()}")
            return false
        }
    }

    private fun cleanupFrames(frameDir: File) {
        try {
            frameDir.listFiles()?.forEach { it.delete() }
            frameDir.delete()
            Log.d("Cleanup", "Cleaned up temporary frames")
        } catch (e: Exception) {
            Log.e("Cleanup", "Failed to cleanup frames: ${e.message}")
        }
    }

    // This is a function from bow team, used for testing on an image
    fun getBitmapFromAssets(context: Context, fileName: String): Bitmap? {
        val assetManager = context.assets
        val inputStream: InputStream?
        try {
            inputStream = assetManager.open(fileName)
            return BitmapFactory.decodeStream(inputStream)
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return null
    }

}