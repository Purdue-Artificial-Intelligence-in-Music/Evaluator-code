package expo.modules.videoanalyzer

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.collections.ArrayDeque
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

import kotlin.time.measureTime

class ExpoVideoAnalyzerModule : Module() {
    private var detector: Detector? = null
    private var isOpenCVInitialized = false
    private var initializationAttempted = false

    // Input for one detector (likely going to be an array later for abstraction)
    var inputBitmaps: Array<ArrayDeque<Bitmap>>? = null

    // Results from async processing videos
    var resultsAsync = HashMap<Long, Bitmap?>() // Holds bitmap with time as key
    // Boolean for when done reading more bitmaps
    var readingBitmaps = false

    // Mutexes for accessing async aspects
    var inputMutexes: Array<Mutex>? = null
    val outputMutex = Mutex()
    val readingBitmapsMutex = Mutex()
    val processingMutex = Mutex()
    var processing = false
    var outputVideoPath: String? = null


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
                        promise.resolve(
                            mapOf(
                                "success" to true,
                                "openCV" to isOpenCVInitialized,
                                "detector" to (detector != null),
                                "initializationMethod" to if (isOpenCVInitialized) "successful" else "failed"
                            )
                        )
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
                            promise.reject(
                                "OPENCV_NOT_READY",
                                "OpenCV not initialized. Call initialize() first.",
                                null
                            )
                        }
                        return@launch
                    }

                    if (detector == null) {
                        withContext(Dispatchers.Main) {
                            promise.reject(
                                "DETECTOR_NOT_READY",
                                "Detector not initialized. Call initialize() first.",
                                null
                            )
                        }
                        return@launch
                    }
                    val bitmapImage =
                        getBitmapFromAssets(appContext.reactContext!!, "Sample Input.png")
//                    val bitmapImage = extractFrameFromVideo(videoUri, 10_000_000L) // convert a frame to bitmap
                    if (bitmapImage == null) {
                        withContext(Dispatchers.Main) {
                            promise.reject("ASSET_ERROR", "Failed to load image from assets", null)
                        }
                        Log.d("bitmapImage", "bitmapImage is null")
                        return@launch
                    }

                    //val result = detector!!.process_frame(bitmapImage)

                    //val bitmap = extractFrameFromVideo(videoUri, 5_000_000L) // convert a frame to bitmap
                    //val result = detector!!.process_frame(bitmap)

                    val bitmap = extractFrameFromVideo(videoUri, 5_000_000L)
                    if (bitmap == null) {
                        withContext(Dispatchers.Main) {
                            promise.reject("FRAME_ERROR", "Failed to extract frame", null)
                        }
                        return@launch
                    }

                    val result = detector!!.process_frame(bitmap)

                    println(result)
                    Log.d("result", "$result")
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
                        promise.resolve(
                            mapOf(
                                "success" to true,
                                "message" to "Detector reset successfully"
                            )
                        )
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
                            promise.reject(
                                "NOT_INITIALIZED",
                                "OpenCV or detector not initialized",
                                null
                            )
                        }
                        return@launch
                    }

                    Log.d("ProcessVideo", "Starting video processing for: $videoUri")

                    var processedFrameCount = 0

                    // Use processVideoStream function to loop through and annotate frames
                    processVideoStreamAsync(videoUri, targetFPS=15) { annotatedFrame, index ->
                        try {
                            processedFrameCount++
                            Log.d(
                                "ProcessFrame",
                                "Processed frame $processedFrameCount at index ${index}μs"
                            )

                            // Frame processing is complete, just count and log

                        } catch (e: Exception) {
                            Log.e(
                                "ProcessFrame",
                                "Failed to process frame $processedFrameCount: ${e.message}"
                            )
                        } finally {
                            // Always recycle the bitmap to prevent memory leaks
                            //annotatedFrame.recycle()
                        }
                    }

                    withContext(Dispatchers.Main) {
                        while (processingMutex.withLock {processing}) {
                            delay(100)
                            Log.d("waiting", "waiting for processing to finish")
                        }
                        if ((outputVideoPath != null) && processedFrameCount > 0) {
                            Log.d(
                                "ProcessVideo",
                                "Successfully processed $processedFrameCount frames"
                            )

                            val resultMap: Map<String, Any> = mapOf(
                                "success" to true,
                                "frameCount" to processedFrameCount,
                                "message" to "Video processing completed successfully",
                                "outputPath" to "file://$outputVideoPath"
                            )
                            promise.resolve(resultMap)
                        } else {
                            Log.e("ProcessVideo", "Failed to process video or no frames processed")
                            promise.reject(
                                "PROCESSING_ERROR",
                                "Failed to process video frames",
                                null
                            )
                        }
                    }

                } catch (e: Exception) {
                    Log.e("ProcessVideo", "Video processing failed: ${e.message}", e)
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
                        promise.resolve(
                            mapOf(
                                "success" to true,
                                "returnCode" to session.returnCode.value,
                                "output" to session.output
                            )
                        )
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

            val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
                ?.toIntOrNull() ?: 0
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
                ?.toIntOrNull() ?: 0
            val frameRate =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
                    ?.toFloatOrNull() ?: 0f
            val bitrate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE)
                ?.toLongOrNull() ?: 0L

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

            originalBitmap =
                retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            if (originalBitmap == null) {
                println("Failed to extract frame from video at time ${timeUs}μs")
                return null
            }

            println("Extracted frame: ${originalBitmap.width}x${originalBitmap.height}, config: ${originalBitmap.config}")
            Log.d(
                "Extract frame",
                "Extracted frame: ${originalBitmap.width}x${originalBitmap.height}, config: ${originalBitmap.config}"
            )

            // bitmap has to be in ARGB_8888 format
            val processedBitmap = if (originalBitmap.config != Bitmap.Config.ARGB_8888) {
                println("Converting bitmap format from ${originalBitmap.config} to ARGB_8888")
                Log.d(
                    "bitmap8888",
                    "Converting bitmap format from ${originalBitmap.config} to ARGB_8888"
                )
                val convertedBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, false)
                originalBitmap.recycle()
                originalBitmap = null
                convertedBitmap
            } else {
                originalBitmap
            }

            println("Frame ready for processing: ${processedBitmap.width}x${processedBitmap.height}")
            Log.d(
                "extract frame",
                "Frame ready for processing: ${processedBitmap.width}x${processedBitmap.height}"
            )
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
    // maxTime is in seconds
    private fun getVideoAnnotations(videoURI: String, targetFPS: Int = 30, maxTime: Float = -1.0f):
            MutableList<Pair<Detector.returnBow, Long>> {
        var maxTimeLong: Long = 1000000L * maxTime.toLong()
        Log.d("maxTimeLong:", "$maxTimeLong")
        var overMaxTime: Boolean = false
        val results = mutableListOf<Pair<Detector.returnBow, Long>>()
        // Set time delta using fps
        var timeUs: Long = 0
        val timeDelta: Long = (1000 / targetFPS) * 1000L // Convert milliseconds to microseconds
        // Get frame by time (using time delta to loop over)
        var bitmapImage: Bitmap? = extractFrameFromVideo(videoURI, timeUs)
        while ((!overMaxTime) && (bitmapImage != null)) {
            //val bitmapImage = getBitmapFromAssets(MainActivity.applicationContext(), "Sample Input.png")
            //evaluator.createInterpreter(MainActivity.applicationContext())
            try {
                //Tasks.await(detector.initializeTask)
                Log.d("Interpreter", "TfLite.initialize() completed successfully")
                Log.d("Timing", "Current time is ${(timeUs.toDouble() / 1000000.0)}")
                bitmapImage.let {
                    detector!!.modelReadyLatch.await()
                    val timeTaken = measureTime {
                        results.add(Pair(detector!!.classify(detector!!.detect(it)), timeUs))
                    }
                    Log.d("InferenceTime", "Time Taken: $timeTaken")
                }
                timeUs += timeDelta
                bitmapImage = extractFrameFromVideo(videoURI, timeUs)
            } catch (e: Exception) {
                Log.e("Interpreter", "Error during evaluation", e)
            }
            // Check over max time (including -1 as whole video)
            if ((maxTime > 0) && (timeUs > maxTimeLong)) {
                Log.d("OverMaxTime", "timeUs: $timeUs maxTimeLong: $maxTimeLong")
                overMaxTime = true
            }
        }

        return results
    }

    private suspend fun getVideoAnnotationsAsync(
        videoURI: String, targetFPS: Int = 30,
        maxTime: Float = -1.0f, numDetectors: Int = 2
    ): Boolean {
        val maxTimeLong: Long = 1000000L * maxTime.toLong()
        Log.d("maxTimeLong:", "$maxTimeLong")

        // Initialize results
        val results = mutableListOf<Pair<Detector.returnBow, Long>>()

        // Set time delta using fps
        val timeDelta: Long = (1000 / targetFPS) * 1000L // Convert milliseconds to microseconds

        val jobs = mutableListOf<Job>()

        // Set awaiting for bitmaps to be done reading
        readingBitmapsMutex.withLock {
            readingBitmaps = true
        }

        // Continuously add bitmaps to the input for the detectors to run inference on
        jobs.add(CoroutineScope(Dispatchers.Default).launch {
            val timeTaken = measureTime {
                var bitmapOverTime: Boolean = false
                var timeUsBitmap: Long = 0
                var bitmapImage: Bitmap? = extractFrameFromVideo(videoURI, timeUsBitmap)
                var bitmapIndex = 0
                while ((!bitmapOverTime) && (bitmapImage != null)) {
                    // Add bitmap image to inputs
                    val arrayMember = inputBitmaps!![bitmapIndex]
                    val arrayMutex = inputMutexes!![bitmapIndex]
                    arrayMutex.withLock {
                        arrayMember.addFirst(bitmapImage!!)
                    }

                    timeUsBitmap += timeDelta
                    bitmapIndex++
                    if (bitmapIndex >= numDetectors) {
                        bitmapIndex = 0
                    }

                    // might add a sleep or something here to prevent over blocking access to bitmaps

                    // Check over max time (including -1 as whole video)
                    if (timeUsBitmap > maxTimeLong) {
                        Log.d(
                            "OverMaxTime",
                            "timeUsBitmap: $timeUsBitmap maxTimeLong: $maxTimeLong"
                        )
                        bitmapOverTime = true
                    }

                    bitmapImage = extractFrameFromVideo(videoURI, timeUsBitmap)
                }

                // Mark reading bitmaps as done
                readingBitmapsMutex.withLock {
                    readingBitmaps = false
                }
            }
            Log.d("OverMaxTime", "Bitmap total time: $timeTaken")
        })

        // Launch both detectors to do inference on their sets of images
        repeat(numDetectors) { index ->
            jobs.add(CoroutineScope(Dispatchers.Default).launch {
                val timeTaken = measureTime {
                    var overTime: Boolean = false
                    var timeUs: Long = (index) * timeDelta
                    var isBitmapEmpty = true
                    // get this detector's mutex
                    var inputMutex = inputMutexes!![index]
                    // get this detector's input bitmap array
                    var inputArray = inputBitmaps!![index]
                    // Await first bitmap
                    while (isBitmapEmpty) {
                        delay(10)
                        inputMutex.withLock {
                            isBitmapEmpty = inputArray.isEmpty()
                        }
                    }
                    // Get first bitmap image
                    var bitmapImage: Bitmap? = null
                    inputMutex.withLock {
                        bitmapImage = inputArray.removeFirst()
                    }
                    var isReadingBitmaps = true
                    while ((!overTime) || ((isReadingBitmaps) && (bitmapImage != null))) {
                        try {
                            Log.d("Interpreter", "TfLite.initialize() completed successfully")
                            Log.d(
                                "Timing",
                                "Index: $index time is ${(timeUs.toDouble() / 1000000.0)}"
                            )
                            bitmapImage.let {
                                val index = (timeUs / timeDelta).toInt()
                                detector!!.modelReadyLatch.await()
                                val result: Pair<Detector.returnBow, Long>
                                val timeTaken = measureTime {
                                    result =
                                        Pair(detector!!.classify(detector!!.detect(it!!)), timeUs)
                                }
                                /*outputMutex.withLock {
                                    resultsAsync!![index] = result
                                }*/
                                Log.d("InferenceTime", "Time Taken: $timeTaken")
                            }
                        } catch (e: Exception) {
                            Log.e("Interpreter", "Error during evaluation", e)
                        }
                        timeUs += numDetectors.toLong() * timeDelta

                        if (timeUs > (maxTimeLong - numDetectors.toLong() * timeDelta)) {
                            Log.d(
                                "OverMaxTime",
                                "index: $index timeUs: $timeUs maxTimeLong: $maxTimeLong"
                            )
                            overTime = true
                        }

                        // Check if still waiting for more bitmaps
                        readingBitmapsMutex.withLock {
                            isReadingBitmaps = readingBitmaps
                        }
                        // If bitmaps are still being read in, wait
                        while (isBitmapEmpty && isReadingBitmaps) {
                            delay(10)
                            inputMutex.withLock {
                                isBitmapEmpty = inputArray.isEmpty()
                            }
                        }
                        // Get next bitmap
                        if (isReadingBitmaps) {
                            inputMutex.withLock {
                                bitmapImage = inputArray.removeFirst()
                            }
                        }
                    }
                }
                Log.d("OverMaxTime", "Index: $index Total Time: $timeTaken")
            })
        }

        return true
    }


    /*
     * Returns a list of <Bitmap, Long> pairs from a passed in video.
     * Uses given FPS to determine how many frames to take
     * WARNING: will store a lot of frames at once, only use for shorter videos/testing
     *          need another function that updates a synchronized stack of frames
     */
    private fun getVideoBitmaps(videoURI: String, targetFPS: Int = 30, maxTime: Float = 3.0f):
            MutableList<Pair<Bitmap, Long>> {

        val results = mutableListOf<Pair<Bitmap, Long>>()
        // Set time delta using fps
        var timeUs: Long = 0
        var maxTimeLong: Long = 1000000L * maxTime.toLong()
        var overMaxTime: Boolean = false
        val timeDelta: Long = (1000 / targetFPS) * 1000L // Convert milliseconds to microseconds
        // Get frame by time (using time delta to loop over)
        var bitmapImage: Bitmap? = extractFrameFromVideo(videoURI, timeUs)
        while ((!overMaxTime) && (bitmapImage != null)) {
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
            // Check over max time (including -1 as whole video)
            if ((maxTime > 0) && (timeUs > maxTimeLong)) {
                overMaxTime = true
            }
        }

        return results
    }

    // Helper function. It loops through the video, extracts frames into bitmap format, calls
    // detect() and drawPointsOnBitmap() from detector to get annotated frames.
    private fun processVideoStream(
        videoURI: String,
        targetFPS: Int = 15,
        onFrameProcessed: (Bitmap, Long) -> Unit
    ): String? {
        val retriever = MediaMetadataRetriever()

        try {
            retriever.setDataSource(appContext.reactContext, Uri.parse(videoURI))
            val duration =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong()
                    ?: 0L

            val videoWidth =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt()
                    ?: 1920
            val videoHeight =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt()
                    ?: 1080

            val rotation =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                    ?.toInt() ?: 0

            // Adjust width and height by rotating
            val (outputWidth, outputHeight) = if (rotation == 90 || rotation == 270) {
                Pair(videoHeight, videoWidth)
            } else {
                Pair(videoWidth, videoHeight)
            }

            Log.d("Encode", "Original video: ${videoWidth}x${videoHeight}, rotation: $rotation")
            Log.d("Encode", "Output video: ${outputWidth}x${outputHeight}")

            val timeDelta = 1_000_000L / targetFPS // microsecond，30fps = 33,333 microseconds

            //val publicMoviesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
            //val outputPath = File(publicMoviesDir, "processed_video_${System.currentTimeMillis()}.mp4").absolutePath
            val cacheDir = appContext.reactContext!!.cacheDir
            val outputPath =
                File(cacheDir, "processed_video_${System.currentTimeMillis()}.mp4").absolutePath

            Log.d("Encode", "Encoded video path: $outputPath")

            val totalFrames = (duration * 1000L / timeDelta).toInt()
            Log.d("Encode", "Target frame interval: ${timeDelta}μs, Total frames: $totalFrames")

            // Use the original video's resolution and the same FPS
            val encoder = VideoEncoder(File(outputPath), outputWidth, outputHeight, fps = targetFPS)

            var timeUs = 0L
            var frameIndex = 0

            while (timeUs < duration * 1000) {
                var frame: Bitmap? = null
                var processedFrame: Bitmap? = null
                var annotatedFrame: Bitmap? = null

                try {
                    // Extract specific frame
                    frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                    frame?.let { originalFrame ->
                        // convert format to ARGB_8888
                        processedFrame = if (originalFrame.config != Bitmap.Config.ARGB_8888) {
                            val convertedBitmap = originalFrame.copy(Bitmap.Config.ARGB_8888, false)
                            originalFrame.recycle()
                            frame = null // Avoid repeated recycling
                            convertedBitmap
                        } else {
                            originalFrame
                        }

                        processedFrame?.let { pFrame ->
                            // Annotate frame using Detector
                            val result = detector!!.detect(pFrame)
                            // TODO: important! Potential memory leak in Detector
                            annotatedFrame = detector!!.drawPointsOnBitmap(pFrame, result)

                            annotatedFrame?.let { aFrame ->
                                if (frameIndex == 0) {
                                    Log.d(
                                        "Encode",
                                        "bitmap size = ${aFrame.height}x${aFrame.width}"
                                    )
                                }

                                // TODO: Logs. Remove later
                                val currentSeconds = timeUs / 1_000_000.0
                                Log.d(
                                    "Encode",
                                    "Encode frame $frameIndex at ${
                                        String.format(
                                            "%.3f",
                                            currentSeconds
                                        )
                                    }s (${timeUs}μs)"
                                )

                                encoder.encodeFrame(aFrame)
                                Log.d("Encode", "Frame $frameIndex encoded")

                                onFrameProcessed(aFrame, timeUs)
                                frameIndex++

                                // Ensure timely memory cleanup
                                if (pFrame != aFrame) {
                                    pFrame.recycle()
                                }
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e("FrameProcess", "Error at frame $frameIndex, time $timeUs: ${e.message}")
                    // Ensure memory is reclaimed even if error occurs
                    frame?.recycle()
                    processedFrame?.let { pf ->
                        if (pf != frame) {
                            pf.recycle()
                        }
                    }
                    annotatedFrame?.let { af ->
                        if (af != processedFrame) {
                            af.recycle()
                        }
                    }
                }

                timeUs += timeDelta
            }

            encoder.finish()
            Log.d("Encode", "Video encoding completed. Total frames processed: $frameIndex")

            // Check output file was generated
            val fileExists = File(outputPath).exists() && File(outputPath).length() > 0
            if (fileExists) {
                return outputPath
            } else {
                return null
            }

        } catch (e: Exception) {
            Log.e("VideoProcess", "Failed to process video: ${e.message}")
            return null
        } finally {
            retriever.release()
        }
    }

    // Helper function. It loops through the video, extracts frames into bitmap format, calls
    // detect() and drawPointsOnBitmap() from detector to get annotated frames.
    private fun processVideoStreamAsync(
        videoURI: String,
        targetFPS: Int = 15,
        onFrameProcessed: (Bitmap, Long) -> Unit
    ) {
        processing = true
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(appContext.reactContext, Uri.parse(videoURI))
        val numDetectors = 4
        inputBitmaps = Array(numDetectors) {ArrayDeque<Bitmap>()}
        inputMutexes = Array(numDetectors) {Mutex()}
        val timeDelta = 1_000_000L / targetFPS // microsecond，30fps = 33,333 microseconds
        val duration = (retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong()
                ?: 0L) / 2L // in milliseconds
        val maxTimeLong: Long = 1000L * duration // milliseconds to microseconds
        Log.d("duration", "duration: $duration maxTimeLong: $maxTimeLong")

        val videoWidth =
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt()
                ?: 1920
        val videoHeight =
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt()
                ?: 1080

        val rotation =
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                ?.toInt() ?: 0

        // Adjust width and height by rotating
        val (outputWidth, outputHeight) = if (rotation == 90 || rotation == 270) {
            Pair(videoHeight, videoWidth)
        } else {
            Pair(videoWidth, videoHeight)
        }

        retriever.release()

        Log.d("Encode", "Original video: ${videoWidth}x${videoHeight}, rotation: $rotation")
        Log.d("Encode", "Output video: ${outputWidth}x${outputHeight}")


        //val publicMoviesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
        //val outputPath = File(publicMoviesDir, "processed_video_${System.currentTimeMillis()}.mp4").absolutePath
        val cacheDir = appContext.reactContext!!.cacheDir
        val outputPath =
            File(cacheDir, "processed_video_${System.currentTimeMillis()}.mp4").absolutePath

        Log.d("Encode", "Encoded video path: $outputPath")

        val totalFrames = (duration * 1000L / timeDelta).toInt()

        Log.d("Encode", "Target frame interval: ${timeDelta}μs, Total frames: $totalFrames")


        try {
            Log.d("maxTimeLong:", "$maxTimeLong")

            // Set for awaiting for bitmaps to be done
            readingBitmaps = true

            // Continuously add bitmaps to the input for the detectors to run inference on
            CoroutineScope(Dispatchers.Default).launch {
                var frame: Bitmap? = null
                var processedFrame: Bitmap? = null
                var bitmapOverTime: Boolean = false
                var timeUsBitmap: Long = 0
                var frameIndex = 0
                var bitmapIndex = 0
                val retriever = MediaMetadataRetriever()
                retriever.setDataSource(appContext.reactContext, Uri.parse(videoURI))
                // Extract specific frame
                frame = retriever.getFrameAtTime(timeUsBitmap, MediaMetadataRetriever.OPTION_CLOSEST)
                frame?.let { originalFrame ->
                    // convert format to ARGB_8888
                    processedFrame = if (originalFrame.config != Bitmap.Config.ARGB_8888) {
                        val convertedBitmap = originalFrame.copy(Bitmap.Config.ARGB_8888, false)
                        Log.d("bitmap", "recycling originalFrame")
                        originalFrame.recycle()
                        frame = null // Avoid repeated recycling
                        convertedBitmap
                    } else {
                        originalFrame
                    }
                }
                while ((!bitmapOverTime) && (processedFrame != null)) {
                    // Add bitmap image to inputs
                    val arrayMember = inputBitmaps!![bitmapIndex]
                    val arrayMutex = inputMutexes!![bitmapIndex]
                    Log.d("Detecting", "adding bitmap $frameIndex to input $bitmapIndex")
                    if (arrayMutex.withLock{arrayMember.size > 20}) {
                        delay(100)
                        Log.d("bitmap", "awaiting detector processing")
                        bitmapIndex++
                        if (bitmapIndex >= numDetectors) {
                            bitmapIndex = 0
                        }
                    } else {
                        arrayMutex.withLock {
                            arrayMember.addFirst(processedFrame!!)
                        }

                        timeUsBitmap += timeDelta
                        bitmapIndex++
                        frameIndex++
                        if (bitmapIndex >= numDetectors) {
                            bitmapIndex = 0
                        }

                        // might add a sleep or something here to prevent over blocking access to bitmaps

                        // Check over max time (including -1 as whole video)
                        if (timeUsBitmap > maxTimeLong) {
                            Log.d(
                                "OverMaxTime",
                                "timeUsBitmap: $timeUsBitmap maxTimeLong: $maxTimeLong"
                            )
                            bitmapOverTime = true
                        }

                        // Extract specific frame
                        frame =
                            retriever.getFrameAtTime(timeUsBitmap, MediaMetadataRetriever.OPTION_CLOSEST)
                        frame?.let { originalFrame ->
                            // convert format to ARGB_8888
                            processedFrame = if (originalFrame.config != Bitmap.Config.ARGB_8888) {
                                val convertedBitmap =
                                    originalFrame.copy(Bitmap.Config.ARGB_8888, false)
                                Log.d("bitmap", "recycling originalFrame")
                                originalFrame.recycle()
                                frame = null // Avoid repeated recycling
                                convertedBitmap
                            } else {
                                originalFrame
                            }
                        }
                    }
                }

                // Mark reading bitmaps as done
                readingBitmapsMutex.withLock {
                    readingBitmaps = false
                }
                Log.d("OverMaxTime", "Bitmap done")
            }

        } catch (e: Exception) {
            Log.e("VideoProcess", "Failed to process video: ${e.message}")
        } finally {
            retriever.release()
        }

        // Launch all detectors to do inference on their arrays of images
        var detecting = true
        val detectingMutex = Mutex();
        var numDetecting = numDetectors
        /* Inputs:
        * detecting/detectingMutex
        * inputArray/inputMutex/resultsAsync
        * timeDelta/maxTimeLong
        * outputPath, outputWidth, outputHeight, fps
        * numDetecting
        */
        repeat(numDetectors) { index ->
            CoroutineScope(Dispatchers.Default).launch {
                var overTime: Boolean = false
                var timeUs: Long = (index) * timeDelta
                var isBitmapEmpty = true
                // get this detector's mutex
                var inputMutex = inputMutexes!![index]
                // get this detector's input bitmap array
                var inputArray = inputBitmaps!![index]
                // Await first bitmap
                while (isBitmapEmpty) {
                    delay(10)
                    inputMutex.withLock {
                        isBitmapEmpty = inputArray.isEmpty()
                    }
                }
                // Get first bitmap image
                var bitmapImage: Bitmap? = null
                var annotatedImage: Bitmap? = null
                inputMutex.withLock {
                    bitmapImage = inputArray.removeFirst()
                }
                var isReadingBitmaps = true
                while ((!overTime) && ((isReadingBitmaps) ||
                            !(inputMutex.withLock {inputArray.isEmpty()}))
                            && (bitmapImage != null)) {
                    if (outputMutex.withLock {resultsAsync!!.size > 10}) {
                        delay(100)
                        Log.d("detector", "Awaiting results shrinking")
                    }
                    val frameIndex = (timeUs / timeDelta).toInt()
                    try {
                        annotatedImage = detector!!.process_bitmap(bitmapImage!!)
                        bitmapImage!!.recycle()
                        if (frameIndex == 0) {
                            Log.d(
                                "Encode",
                                "bitmap size = ${bitmapImage!!.height}x${bitmapImage!!.width}"
                            )
                        }

                        // TODO: Logs. Remove later
                        val currentSeconds = timeUs / 1_000_000.0
                        Log.d(
                            "Detecting",
                            "Detect frame $frameIndex at ${
                                String.format(
                                    "%.3f",
                                    currentSeconds
                                )
                            }s (${timeUs}μs)"
                        )

                        Log.d("detector", "adding result to results")
                        outputMutex.withLock {
                            resultsAsync.put(timeUs, annotatedImage!!)
                        }
                        onFrameProcessed(annotatedImage!!, timeUs)
                    } catch (e: Exception) {
                        Log.e(
                            "FrameProcess",
                            "Error at frame $frameIndex, time $timeUs: ${e.message}"
                        )
                        // Ensure memory is reclaimed even if error occurs
                        bitmapImage?.recycle()
                    }
                    timeUs += numDetectors.toLong() * timeDelta

                    if (timeUs > (maxTimeLong - numDetectors.toLong() * timeDelta)) {
                        Log.d(
                            "OverMaxTime",
                            "index: $index timeUs: $timeUs maxTimeLong: $maxTimeLong"
                        )
                        overTime = true
                    }

                    // set to await next bitmap
                    isBitmapEmpty = true

                    // Check if still waiting for more bitmaps
                    readingBitmapsMutex.withLock {
                        isReadingBitmaps = readingBitmaps
                    }
                    // If bitmaps are still being read in, wait
                    while (isBitmapEmpty && isReadingBitmaps) {
                        delay(10)
                        inputMutex.withLock {
                            isBitmapEmpty = inputArray.isEmpty()
                        }
                    }
                    // Get next bitmap
                    if (isReadingBitmaps) {
                        inputMutex.withLock {
                            bitmapImage = inputArray.removeFirst()
                        }
                    }
                }
                detectingMutex.withLock {
                    numDetecting--;
                    if (numDetecting == 0) {
                        detecting = false;
                    }
                }
                Log.d("OverMaxTime", "Index: $index done")
            }
        }


        // Encoder can wait for and encode output
        /* Inputs:
         * detecting/detectingMutex
         * resultsAsync
         * timeDelta
         * outputPath, outputWidth, outputHeight, fps
         */
        CoroutineScope(Dispatchers.Default).launch {
            // Use the original video's resolution and the same FPS
            val encoder = VideoEncoder(File(outputPath), outputWidth, outputHeight, fps = targetFPS)
            var index = 0
            var timeUs = 0L
            var result: Pair<Bitmap, Long>? = null
            var frame: Bitmap? = null
            var frameTime: Long = 0L
            // encode images while detecting is happening
            while ((detectingMutex.withLock { detecting })) {
                while (!outputMutex.withLock { resultsAsync.containsKey(timeUs) } &&
                    (detectingMutex.withLock { detecting })) {
                    delay(100);
                    Log.d("Encode", "Awaiting next image")
                }

                while (outputMutex.withLock {resultsAsync.containsKey(timeUs)}) {
                    outputMutex.withLock {
                        frame = resultsAsync.remove(timeUs)
                    }
                    if (frame!!.isRecycled()) {
                        Log.d("Encode", "FRAME IS RECYCLED")
                    } else {
                        index = (timeUs / timeDelta).toInt()
                        encoder.encodeFrame(frame!!)
                        frame!!.recycle()
                        Log.d("Detecting", "Frame $index encoded")
                    }
                    timeUs += timeDelta
                }
            }
            // Ensures that all images were processed before exiting
            while (outputMutex.withLock {resultsAsync.containsKey(timeUs)}) {
                outputMutex.withLock {
                    frame = resultsAsync.remove(timeUs)
                }
                if (frame!!.isRecycled()) {
                    Log.d("Encode", "FRAME IS RECYCLED")
                } else {
                    index = (timeUs / timeDelta).toInt()
                    encoder.encodeFrame(frame!!)
                    frame!!.recycle()
                    Log.d("Detecting", "Frame $index encoded")
                }
                timeUs += timeDelta
            }

            encoder.finish()
            Log.d("Encode", "Video encoding completed. Total frames processed: $index")
            // Check output file was generated
            val fileExists = File(outputPath).exists() && File(outputPath).length() > 0
            if (fileExists) {
                outputVideoPath = outputPath
            } else {
                outputVideoPath = null
            }

            processingMutex.withLock {
                processing = false
            }
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