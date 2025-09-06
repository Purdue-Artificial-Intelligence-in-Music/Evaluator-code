package expo.modules.videoanalyzer

import android.util.Log
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.Promise
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlinx.coroutines.*
import android.graphics.Bitmap

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
                    
                    val bitmap = extractFrameFromVideo(videoUri, 5_000_000L) // convert a frame to bitmap
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
                        "imageWidth" to bitmap.width,
                        "imageHeight" to bitmap.height
                    )
                    
                    // clear bitmap storage
                    bitmap.recycle()
                    
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
    
    private fun extractFrameFromVideo(videoUri: String, timeUs: Long = 5): Bitmap {
        val retriever = MediaMetadataRetriever()
        var originalBitmap: Bitmap? = null
        
        return try {
            println("Extracting frame from video: $videoUri at time: ${timeUs}μs")
            Log.d("VideoFrame", "Extracting frame from video: $videoUri at time: ${timeUs}μs")

            val uri = Uri.parse(videoUri) ?: throw IllegalArgumentException("Invalid video URI")
            retriever.setDataSource(appContext.reactContext, uri)
            
            originalBitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                ?: throw Exception("Failed to extract frame from video at time ${timeUs}μs")
            
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
                val result = originalBitmap
                originalBitmap = null
                result
            }
            
            println("Frame ready for processing: ${processedBitmap.width}x${processedBitmap.height}")
            Log.d("extract frame", "Frame ready for processing: ${processedBitmap.width}x${processedBitmap.height}")
            processedBitmap
            
        } catch (e: Exception) {
            originalBitmap?.recycle()
            println("Error extracting frame: ${e.message}")
            Log.d("extract frame", "Error extracting frame: ${e.message}")
            throw Exception("Failed to extract frame from video: ${e.message}")
        } finally {
            retriever.release()
        }
    }
}