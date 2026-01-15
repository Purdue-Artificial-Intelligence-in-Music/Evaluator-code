package expo.modules.camerax

import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * Shared result bundle for both MediaPipe and NPU pipelines.
 *
 * For NPU pipeline: handResults/poseResults are empty, use handDetection/poseDetection strings.
 * For MediaPipe pipeline: All fields populated.
 */
data class CombinedResultBundle(
    val handResults: List<HandLandmarkerResult>,
    val poseResults: List<PoseLandmarkerResult>,
    val inferenceTime: Long,
    val inputImageHeight: Int,
    val inputImageWidth: Int,
    val handCoordinates: FloatArray?,
    val poseCoordinates: FloatArray?,
    val handDetection: String,
    val poseDetection: String,
    val targetHandIndex: Int = -1
)

/**
 * Listener for combined hand+pose results.
 */
interface CombinedLandmarkerListener {
    fun onError(error: String, errorCode: Int = 0)
    fun onResults(resultBundle: CombinedResultBundle)
}
