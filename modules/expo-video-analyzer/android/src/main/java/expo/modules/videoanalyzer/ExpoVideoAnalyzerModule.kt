package expo.modules.videoanalyzer

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.Promise
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlinx.coroutines.*

class ExpoVideoAnalyzerModule : Module() {
    override fun definition() = ModuleDefinition {
        Name("ExpoVideoAnalyzer")
        
        Function("getStatus") {
            return@Function "Video Analyzer Module Connected"
        }
        
        AsyncFunction("analyzeVideo") { videoUri: String, promise: Promise ->
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
    }
    
    private fun openAndValidateVideo(videoUri: String): Map<String, Any> {
        val retriever = MediaMetadataRetriever()
        
        return try {
            retriever.setDataSource(appContext.reactContext, Uri.parse(videoUri))
            
            val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            
            mapOf(
                "success" to true,
                "message" to "Video opened successfully",
                "duration" to duration,
                "width" to width,
                "height" to height,
                "videoUri" to videoUri
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
}