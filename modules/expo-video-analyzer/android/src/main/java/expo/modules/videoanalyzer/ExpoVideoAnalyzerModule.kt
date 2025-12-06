package expo.modules.videoanalyzer
import expo.modules.camerax.Detector

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
import android.media.MediaExtractor
import android.media.MediaCodec
import android.media.Image
import android.media.MediaFormat
import android.graphics.ImageFormat
import android.graphics.YuvImage
import android.graphics.Rect
import java.io.ByteArrayOutputStream

import com.google.mediapipe.tasks.vision.core.RunningMode
import expo.modules.camerax.HandLandmarkerHelper

import kotlin.time.measureTime

class ExpoVideoAnalyzerModule : Module() {
    private var detector: Detector? = null
    private var initializationAttempted = false
    @Volatile private var isCancelled = false

    // Input for one detector (likely going to be an array later for abstraction)
    var inputBitmaps: Array<ArrayDeque<Bitmap>>? = null
    // Results from async processing videos
    //var resultsAsync: Array<Pair<Detector.returnBow, Long>?>? = null
    // Boolean for when done reading more bitmaps
    var readingBitmaps = false
    // Mutexes for accessing async aspects
    var inputMutexes: Array<Mutex>? = null
    val outputMutex = Mutex()
    val readingBitmapsMutex = Mutex()

    override fun definition() = ModuleDefinition {
        Name("ExpoVideoAnalyzer")

        Events("onVideoProgress")

        AsyncFunction("initialize") { promise: Promise ->
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    if (!initializationAttempted) {
                        initializationAttempted = true

                        initializeDetector()
                    }
                    
                    withContext(Dispatchers.Main) {
                        promise.resolve(mapOf(
                            "success" to true,
                            "detector" to (detector != null),
                        ))
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        promise.reject("INIT_ERROR", "Initialization failed: ${e.message}", e)
                    }
                }
            }
        }
        
//        AsyncFunction("processFrame") { videoUri: String, promise: Promise ->
//            CoroutineScope(Dispatchers.IO).launch {
//                try {
//                    if (!isOpenCVInitialized) {
//                        withContext(Dispatchers.Main) {
//                            promise.reject("OPENCV_NOT_READY", "OpenCV not initialized. Call initialize() first.", null)
//                        }
//                        return@launch
//                    }
//
//                    if (detector == null) {
//                        withContext(Dispatchers.Main) {
//                            promise.reject("DETECTOR_NOT_READY", "Detector not initialized. Call initialize() first.", null)
//                        }
//                        return@launch
//                    }
//                    val bitmapImage = getBitmapFromAssets(appContext.reactContext!!, "Sample Input.png")
////                    val bitmapImage = extractFrameFromVideo(videoUri, 10_000_000L) // convert a frame to bitmap
//                    if (bitmapImage == null) {
//                        withContext(Dispatchers.Main) {
//                            promise.reject("ASSET_ERROR", "Failed to load image from assets", null)
//                        }
//                        Log.d("bitmapImage", "bitmapImage is null")
//                        return@launch
//                    }
//
//                    //val result = detector!!.process_frame(bitmapImage)
//
//                    //val bitmap = extractFrameFromVideo(videoUri, 5_000_000L) // convert a frame to bitmap
//                    //val result = detector!!.process_frame(bitmap)
//
//                    val bitmap = extractFrameFromVideo(videoUri, 5_000_000L)
//                    if (bitmap == null) {
//                        withContext(Dispatchers.Main) {
//                            promise.reject("FRAME_ERROR", "Failed to extract frame", null)
//                        }
//                        return@launch
//                    }
//
//                    val result = detector!!.process_frame(bitmap)
//
//                    println(result)
//                    Log.d("result", "$result")
//                    val response = mapOf(
//                        "success" to true,
//                        "classification" to (result.classification ?: -1),
//                        "angle" to (result.angle ?: 0),
//                        "hasBow" to (result.bow != null),
//                        "hasString" to (result.string != null),
//                        "bowPoints" to if (result.bow != null) result.bow!!.size else 0,
//                        "stringPoints" to if (result.string != null) result.string!!.size else 0,
//                        "imageWidth" to bitmapImage.width,
//                        "imageHeight" to bitmapImage.height
//                    )
//
//                    // clear bitmap storage
//                    bitmapImage.recycle()
//
//                    withContext(Dispatchers.Main) {
//                        promise.resolve(response)
//                    }
//                } catch (e: Exception) {
//                    withContext(Dispatchers.Main) {
//                        promise.reject("PROCESS_ERROR", "Frame processing failed: ${e.message}", e)
//                    }
//                }
//            }
//        }

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
            var outputpath: String? = null
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    if (detector == null) {
                        withContext(Dispatchers.Main) {
                            promise.reject("NOT_INITIALIZED", "Detector not initialized", null)
                        }
                        return@launch
                    }

                    Log.d("ProcessVideo", "Starting video processing for: $videoUri")
                    var processedFrameCount = 0
                    isCancelled = false // reset cancel flag at start

                    outputpath = processVideoStreamWithExtractor(videoUri) { annotatedFrame, timeUs ->
                        if (isCancelled) {
                            Log.d("ProcessVideo", "Processing cancelled at frame $processedFrameCount")
                            annotatedFrame.recycle()
                            throw CancellationException("Processing was cancelled")
                        }

                        try {
                            processedFrameCount++
                            Log.d("ProcessFrame", "Processed frame $processedFrameCount at time ${timeUs}μs")
                        } catch (e: Exception) {
                            Log.e("ProcessFrame", "Failed to process frame $processedFrameCount: ${e.message}")
                        } finally {
                            annotatedFrame.recycle()
                        }
                    }

                    if (isCancelled) {
                        outputpath?.let { File(it).delete() }
                        withContext(Dispatchers.Main) {
                            promise.reject("PROCESSING_CANCELLED", "User cancelled video processing", null)
                        }
                        return@launch
                    }

                    withContext(Dispatchers.Main) {
                        if ((outputpath != null) && processedFrameCount > 0) {
                            Log.d("ProcessVideo", "Successfully processed $processedFrameCount frames")

                            val resultMap: Map<String, Any> = mapOf(
                                "success" to true,
                                "frameCount" to processedFrameCount,
                                "message" to "Video processing completed successfully",
                                "outputPath" to "file://$outputpath"
                            )
                            promise.resolve(resultMap)
                        } else {
                            Log.e("ProcessVideo", "Failed to process video or no frames processed")
                            promise.reject("PROCESSING_ERROR", "Failed to process video frames", null)
                            outputpath?.let { File(it).delete() }
                        }
                    }
                } catch (e: CancellationException) {
                    outputpath?.let { File(it).delete() }
                    withContext(Dispatchers.Main) {
                        promise.reject("CANCELLED", e.message, e)
                    }
                } catch (e: Exception) {
                    outputpath?.let { File(it).delete() }
                    Log.e("ProcessVideo", "Video processing failed: ${e.message}", e)
                    withContext(Dispatchers.Main) {
                        promise.reject("PROCESS_ERROR", "Video processing failed: ${e.message}", e)
                    }
                } finally {
                    isCancelled = false // reset flag
                }
            }
        }

        AsyncFunction("cancelProcessing") {
            isCancelled = true
            Log.d("ProcessVideo", "Cancel flag set to true")
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
                originalBitmap = null
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

    private suspend fun getVideoAnnotationsAsync(videoURI: String, targetFPS: Int = 30,
                                         maxTime: Float = -1.0f, numDetectors: Int = 2): Boolean {
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
                        inputMutex.withLock {
                            bitmapImage = inputArray.removeFirst()
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
//    private fun getVideoBitmaps(videoURI: String, targetFPS: Int = 30, maxTime: Float = 3.0f):
//            MutableList<Pair<Bitmap, Long>> {
//
//        val results = mutableListOf<Pair<Bitmap, Long>>()
//        // Set time delta using fps
//        var timeUs: Long = 0
//        var maxTimeLong: Long = 1000000L * maxTime.toLong()
//        var overMaxTime: Boolean = false
//        val timeDelta: Long = (1000 / targetFPS) * 1000L // Convert milliseconds to microseconds
//        // Get frame by time (using time delta to loop over)
//        var bitmapImage: Bitmap? = extractFrameFromVideo(videoURI, timeUs)
//        while ((!overMaxTime) && (bitmapImage != null)) {
//            //val bitmapImage = getBitmapFromAssets(MainActivity.applicationContext(), "Sample Input.png")
//            //evaluator.createInterpreter(MainActivity.applicationContext())
//            try {
//                //Tasks.await(detector.initializeTask)
//                Log.d("Interpreter", "TfLite.initialize() completed successfully")
//                Log.d("Timing", "Current time is ${(timeUs.toDouble() / 1000000.0)}")
//                bitmapImage.let {
//                    detector!!.modelReadyLatch.await()
//
//                    val result = detector!!.detect(it)
//                    results.add(Pair(detector!!.drawPointsOnBitmap(it, result), timeUs))
//                }
//                timeUs += timeDelta
//                bitmapImage = extractFrameFromVideo(videoURI, timeUs)
//            } catch (e: Exception) {
//                Log.e("Interpreter", "Error during evaluation", e)
//            }
//            // Check over max time (including -1 as whole video)
//            if ((maxTime > 0) && (timeUs > maxTimeLong)) {
//                overMaxTime = true
//            }
//        }
//
//        return results
//    }

    // Helper function. It loops through the video, extracts frames into bitmap format, calls
    // detect() and drawPointsOnBitmap() from detector to get annotated frames.
    private fun processVideoStreamWithExtractor(
        videoURI: String,
        onFrameProcessed: (Bitmap, Long) -> Unit
    ): String? {
        val extractor = MediaExtractor()
        var landmarkerHelper: HandLandmarkerHelper? = null
        var codec: MediaCodec? = null
        var outputPath: String? = null

        try {
            extractor.setDataSource(appContext.reactContext!!, Uri.parse(videoURI), null)
            val trackIndex = (0 until extractor.trackCount).firstOrNull { i ->
                extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
            } ?: run {
                Log.e("VideoProcess", "No video track found")
                return null
            }

            extractor.selectTrack(trackIndex)
            val format = extractor.getTrackFormat(trackIndex)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: return null
            val videoWidth = format.getInteger(MediaFormat.KEY_WIDTH)
            val videoHeight = format.getInteger(MediaFormat.KEY_HEIGHT)
            val frameRate = if (format.containsKey(MediaFormat.KEY_FRAME_RATE)) {
                format.getInteger(MediaFormat.KEY_FRAME_RATE)
            } else 30
            val safeFps = frameRate.coerceIn(1, 60)
            val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else 0L
            val estimatedFrames = if (durationUs > 0) {
                (durationUs / (1_000_000L / safeFps)).coerceAtLeast(1)
            } else 0L

            val cacheDir = appContext.reactContext!!.cacheDir
            outputPath = File(cacheDir, "processed_video_${System.currentTimeMillis()}.mp4").absolutePath

            Log.d("Encode", "Output video: ${videoWidth}x${videoHeight} @ $safeFps fps")

            landmarkerHelper = HandLandmarkerHelper(
                context = appContext.reactContext!!,
                runningMode = RunningMode.VIDEO,
                combinedLandmarkerHelperListener = null,
                maxNumHands = 2
            )

            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()

            val encoder = VideoEncoder(File(outputPath), videoWidth, videoHeight, fps = safeFps)

            val bufferInfo = MediaCodec.BufferInfo()
            var sawInputEOS = false
            var sawOutputEOS = false
            var frameIndex = 0
            var lastProgressSent = -1

            while (!sawOutputEOS && !isCancelled) {
                if (!sawInputEOS) {
                    val inputIndex = codec.dequeueInputBuffer(10_000)
                    if (inputIndex >= 0) {
                        val inputBuffer = codec.getInputBuffer(inputIndex)
                        val sampleSize = extractor.readSampleData(inputBuffer!!, 0)
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(
                                inputIndex, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            sawInputEOS = true
                        } else {
                            val presentationTimeUs = extractor.sampleTime
                            codec.queueInputBuffer(inputIndex, 0, sampleSize, presentationTimeUs, 0)
                            extractor.advance()
                        }
                    }
                }

                val outputIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                when {
                    outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        // ignore
                    }
                    outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                        // no-op
                    }
                    outputIndex >= 0 -> {
                        val outputImage: Image? = codec.getOutputImage(outputIndex)
                        if (outputImage != null && bufferInfo.size > 0) {
                            val bitmap = yuv420ToBitmap(outputImage)
                            outputImage.close()

                            bitmap?.let { pFrame ->
                                var annotatedFrame: Bitmap? = null
                                try {
                                    annotatedFrame = detector!!.process_frame(pFrame)
                                    val (landmarkerResult, mpAnnotatedFrame) = landmarkerHelper?.detectAndDrawVideoFrame(
                                        annotatedFrame,
                                        bufferInfo.presentationTimeUs / 1000
                                    ) ?: Pair(null, null)
                                    if (mpAnnotatedFrame != null) {
                                        annotatedFrame = mpAnnotatedFrame
                                        if (landmarkerResult != null) {
                                            Log.d("Encode", "Frame $frameIndex - Hand: ${landmarkerResult.handDetection}, Pose: ${landmarkerResult.poseDetection}")
                                        }
                                    }

                                    annotatedFrame?.let { aFrame ->
                                        encoder.encodeFrame(aFrame)
                                        onFrameProcessed(aFrame, bufferInfo.presentationTimeUs)
                                        frameIndex++
                                        if (estimatedFrames > 0) {
                                            val progressPercent = ((frameIndex.toLong() * 100L) / estimatedFrames).toInt().coerceIn(0, 100)
                                            if (progressPercent != lastProgressSent) {
                                                lastProgressSent = progressPercent
                                                CoroutineScope(Dispatchers.Main).launch {
                                                    sendEvent(
                                                        "onVideoProgress",
                                                        mapOf(
                                                            "progress" to progressPercent / 100f,
                                                            "frame" to frameIndex,
                                                            "estimatedFrames" to estimatedFrames
                                                        )
                                                    )
                                                }
                                            }
                                        }
                                        if (aFrame != annotatedFrame) {
                                            aFrame.recycle()
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e("FrameProcess", "Error at frame $frameIndex: ${e.message}")
                                } finally {
                                    if (annotatedFrame != null && annotatedFrame != pFrame) {
                                        annotatedFrame.recycle()
                                    }
                                    pFrame.recycle()
                                }
                            }
                        }

                        if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                            sawOutputEOS = true
                        }
                        codec.releaseOutputBuffer(outputIndex, false)
                    }
                }
            }

            if (isCancelled) {
                Log.d("ProcessVideo", "Cancelling encoder and cleaning up")
                encoder.finish()
                File(outputPath).delete()
                return null
            }

            encoder.finish()
            if (estimatedFrames > 0) {
                CoroutineScope(Dispatchers.Main).launch {
                    sendEvent(
                        "onVideoProgress",
                        mapOf(
                            "progress" to 1.0f,
                            "frame" to frameIndex,
                            "estimatedFrames" to estimatedFrames
                        )
                    )
                }
            }
            Log.d("Encode", "Video encoding completed. Total frames processed: $frameIndex")
            val fileExists = File(outputPath).exists() && File(outputPath).length() > 0
            return if (fileExists) outputPath else null
        } catch (e: Exception) {
            Log.e("VideoProcess", "Failed to process video: ${e.message}")
            return null
        } finally {
            try { extractor.release() } catch (_: Throwable) {}
            try { codec?.stop(); codec?.release() } catch (_: Throwable) {}
            landmarkerHelper?.clearLandmarkers()
        }
    }

    private fun yuv420ToBitmap(image: Image): Bitmap? {
        if (image.format != ImageFormat.YUV_420_888) return null
        val width = image.width
        val height = image.height
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)

        // VU ordering
        val chromaStart = ySize
        val vRowStride = image.planes[2].rowStride
        val uRowStride = image.planes[1].rowStride
        val vPixelStride = image.planes[2].pixelStride
        val uPixelStride = image.planes[1].pixelStride

        var offset = chromaStart
        for (row in 0 until height / 2) {
            var vIdx = row * vRowStride
            var uIdx = row * uRowStride
            for (col in 0 until width / 2) {
                nv21[offset++] = vBuffer.get(vIdx)
                nv21[offset++] = uBuffer.get(uIdx)
                vIdx += vPixelStride
                uIdx += uPixelStride
            }
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        val jpegBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)?.copy(Bitmap.Config.ARGB_8888, true)
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
