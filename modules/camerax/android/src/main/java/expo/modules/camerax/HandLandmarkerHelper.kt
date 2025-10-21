/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package expo.modules.camerax

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import androidx.core.view.indices
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.String
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import kotlin.plus

class HandLandmarkerHelper(
    var minHandDetectionConfidence: Float = DEFAULT_HAND_DETECTION_CONFIDENCE,
    var minHandTrackingConfidence: Float = DEFAULT_HAND_TRACKING_CONFIDENCE,
    var minHandPresenceConfidence: Float = DEFAULT_HAND_PRESENCE_CONFIDENCE,
    var maxNumHands: Int = DEFAULT_NUM_HANDS,
    var currentDelegate: Int = DELEGATE_CPU,
    var runningMode: RunningMode = RunningMode.IMAGE,
    val context: Context,
    val combinedLandmarkerHelperListener: CombinedLandmarkerListener? = null

) {

    var handLandmarker: HandLandmarker? = null
    var poseLandmarker: PoseLandmarker? = null
    private var handTFLite: Interpreter? = null
    private var poseTFLite: Interpreter? = null

    private val tfliteLock = Any()
    private var isClosed = false

    // New state variables to store the most recent results
    private var latestHandResult: HandLandmarkerResult? = null
    private var latestPoseResult: PoseLandmarkerResult? = null
    private var latestImage: MPImage? = null
    private var latestFrameTime: Long = 0

    private var isFrontCameraActive: Boolean = false

    init {
        setupHandLandmarker()
        setupPoseLandmarker()
    }

    fun clearLandmarkers() {
        synchronized(tfliteLock) {
            isClosed = true // Signal that the helper is shutting down
            handLandmarker?.close()
            handLandmarker = null
            poseLandmarker?.close()
            poseLandmarker = null
            handTFLite?.close()
            handTFLite = null
            poseLandmarker?.close()
            poseLandmarker = null
        }
    }

    fun isClose(): Boolean {
        return handLandmarker == null && poseLandmarker == null
    }

    fun setupHandLandmarker() {
        val baseOptionBuilder = BaseOptions.builder()

        when (currentDelegate) {
            DELEGATE_CPU -> baseOptionBuilder.setDelegate(Delegate.CPU)
            DELEGATE_GPU -> baseOptionBuilder.setDelegate(Delegate.GPU)
        }

        baseOptionBuilder.setModelAssetPath(MP_HAND_LANDMARKER_TASK)

        if (runningMode == RunningMode.LIVE_STREAM && combinedLandmarkerHelperListener == null) {
            throw IllegalStateException("Listener must be set when runningMode is LIVE_STREAM.")
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            val optionsBuilder =
                HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMinHandDetectionConfidence(minHandDetectionConfidence)
                    .setMinTrackingConfidence(minHandTrackingConfidence)
                    .setMinHandPresenceConfidence(minHandPresenceConfidence)
                    .setNumHands(maxNumHands)
                    .setRunningMode(runningMode)

            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                    .setResultListener(this::receiveHandLivestreamResult)
                    .setErrorListener(this::returnLivestreamError)
            }

            val options = optionsBuilder.build()
            handLandmarker = HandLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            combinedLandmarkerHelperListener?.onError("Hand Landmarker failed to initialize.", if (e is RuntimeException) GPU_ERROR else OTHER_ERROR)
            Log.e(TAG, "MediaPipe failed to load hand task: " + e.message)
        }
    }

    fun setupPoseLandmarker() {
        val baseOptionBuilder = BaseOptions.builder()

        when (currentDelegate) {
            DELEGATE_CPU -> baseOptionBuilder.setDelegate(Delegate.CPU)
            DELEGATE_GPU -> baseOptionBuilder.setDelegate(Delegate.GPU)
        }

        baseOptionBuilder.setModelAssetPath("pose_landmarker_full.task")

        if (runningMode == RunningMode.LIVE_STREAM && combinedLandmarkerHelperListener == null) {
            throw IllegalStateException("Listener must be set when runningMode is LIVE_STREAM.")
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            val optionsBuilder =
                PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(runningMode)

            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                    .setResultListener(this::receivePoseLivestreamResult)
                    .setErrorListener(this::returnLivestreamError)
            }

            val options = optionsBuilder.build()
            poseLandmarker = PoseLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            combinedLandmarkerHelperListener?.onError("Pose Landmarker failed to initialize.", if (e is RuntimeException) GPU_ERROR else OTHER_ERROR)
            Log.e(TAG, "MediaPipe failed to load pose task: " + e.message)
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun detectLiveStream(imageProxy: ImageProxy, isFrontCamera: Boolean) {
        synchronized(tfliteLock) {
            if (isClosed) {
                imageProxy.close()
                return
            }

            // Store which camera is active
            isFrontCameraActive = isFrontCamera

            if (runningMode != RunningMode.LIVE_STREAM) {
                imageProxy.close()
                throw IllegalArgumentException("Attempting to call detectLiveStream while not using RunningMode.LIVE_STREAM")
            }

            // 2. PREPARE IMAGE (Your original logic is preserved here)
            latestFrameTime = SystemClock.uptimeMillis()

            val bitmapBuffer =
                Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)

            // Using .use { ... } is a great pattern as it automatically calls imageProxy.close()
            // when the block is finished, even if an exception occurs.
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(it.planes[0].buffer) }

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                if (isFrontCamera) {
                    postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true
            )
            //Log.d("HANDS DIMENSION WIDTH", rotatedBitmap.width.toString() + ", " + rotatedBitmap.height.toString())
            latestImage = BitmapImageBuilder(rotatedBitmap).build()

            // 3. RUN INFERENCE
            // Because of the synchronized block, we know that the `handLandmarker` and
            // `poseLandmarker` objects cannot be nullified by `clearLandmarkers()`
            // during this detection call.
            handLandmarker?.detectAsync(latestImage!!, latestFrameTime)
            poseLandmarker?.detectAsync(latestImage!!, latestFrameTime)
        }
    }

    // Updated detectVideoFile to return a CombinedResultBundle
//    fun detectVideoFile(videoUri: Uri, inferenceIntervalMs: Long): CombinedResultBundle? {
//        if (runningMode != RunningMode.VIDEO) {
//            throw IllegalArgumentException("Attempting to call detectVideoFile while not using RunningMode.VIDEO")
//        }
//
//        // might not account for complex threading
//        val retriever = MediaMetadataRetriever()
//        retriever.setDataSource(context, videoUri)
//        val videoLengthMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: return null
//        val firstFrame = retriever.getFrameAtTime(0) ?: return null
//
//        val handResultList = mutableListOf<HandLandmarkerResult>()
//        val poseResultList = mutableListOf<PoseLandmarkerResult>()
//
//        val numberOfFrames = videoLengthMs.div(inferenceIntervalMs)
//
//        for (i in 0..numberOfFrames) {
//            val timestampMs = i * inferenceIntervalMs
//            val frame = retriever.getFrameAtTime(timestampMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST)
//            frame?.let {
//                val mpImage = BitmapImageBuilder(it).build()
//                handLandmarker?.detectForVideo(mpImage, timestampMs)?.let { handResult -> handResultList.add(handResult) }
//                poseLandmarker?.detectForVideo(mpImage, timestampMs)?.let { poseResult -> poseResultList.add(poseResult) }
//            }
//        }
//        retriever.release()
//
//        return CombinedResultBundle(
//            handResults = handResultList,
//            poseResults = poseResultList,
//            inferenceTime = 0, // Placeholder
//            inputImageHeight = firstFrame.height,
//            inputImageWidth = firstFrame.width,
//            handCoordinates = null,
//            poseCoordinates = null,
//            handDetection = "",
//            poseDetection = ""
//        )
//    }
    fun detectAndDrawVideoFrame(frame: Bitmap?, timestampMs: Long): Pair<CombinedResultBundle?, Bitmap?> {
        if (runningMode != RunningMode.VIDEO) {
            throw IllegalArgumentException("Attempting to call detectAndDrawVideoFrame while not using RunningMode.VIDEO")
        }
        if (frame == null) return Pair(null, null)
        val startTime = SystemClock.uptimeMillis()

        val mpImage = BitmapImageBuilder(frame).build()
        val handResult = handLandmarker?.detectForVideo(mpImage, timestampMs)
        val poseResult = poseLandmarker?.detectForVideo(mpImage, timestampMs)

        var handCoords: FloatArray? = null
        var handPrediction: String = "No hand detected"
        var targetHandIndex = -1

        if (handResult != null && handResult.landmarks().isNotEmpty()) {
            val handednessList = handResult.handedness()
            val landmarksList = handResult.landmarks()
            var maxY: Float = -1.0f

            // traverse detected hands
            handednessList.forEachIndexed { index, handedness ->
                landmarksList.getOrNull(index)?.takeIf { it.isNotEmpty() }?.let { landmarks ->
                    val wristY = landmarks[0].y()

                    // Find lowest hand (lowest hand has max wristY value)
                    if (wristY > maxY) {
                        maxY = wristY
                        targetHandIndex = index
                    }
                }
            }

            // Only process lowest hand
            if (targetHandIndex != -1) {
                handCoords = extractHandCoordinates(handResult, targetHandIndex)
                handCoords?.let {
                    handPrediction = runTFLiteInference(it)
                }
            }
        }

        var poseCoords: FloatArray? = null
        var posePrediction: String = "No pose detected"
        if (poseResult != null && poseResult.landmarks().isNotEmpty()) {
            poseCoords = extractPoseCoordinates(poseResult)
            posePrediction = runTFLitePoseInference(poseCoords)
        }
        val inferenceTime = SystemClock.uptimeMillis() - startTime

        val resultBundle = CombinedResultBundle(
            handResults = if (handResult != null) listOf(handResult) else emptyList(),
            poseResults = if (poseResult != null) listOf(poseResult) else emptyList(),
            inferenceTime = inferenceTime,
            inputImageHeight = frame.height,
            inputImageWidth = frame.width,
            handCoordinates = handCoords,
            poseCoordinates = poseCoords,
            handDetection = handPrediction,
            poseDetection = posePrediction,
            targetHandIndex = targetHandIndex
        )
        val annotatedFrame = drawMediaPipeAnnotations(frame, resultBundle)

        return Pair(resultBundle, annotatedFrame)
    }

    // Corrected detectImage to return a CombinedResultBundle
    fun detectImage(image: Bitmap): CombinedResultBundle? {
        if (runningMode != RunningMode.IMAGE) {
            throw IllegalArgumentException("Attempting to call detectImage while not using RunningMode.IMAGE")
        }

        val startTime = SystemClock.uptimeMillis()
        val mpImage = BitmapImageBuilder(image).build()

        val handResult = handLandmarker?.detect(mpImage)
        val poseResult = poseLandmarker?.detect(mpImage)

        val inferenceTimeMs = SystemClock.uptimeMillis() - startTime

        var leftHandPrediction = ""
        var leftHandCoordinates: FloatArray? = null
        val poseCoordinates = poseResult?.let { extractPoseCoordinates(it) }

        // Process hand result for left hand if available
        handResult?.let { result ->
            val handednessList = result.handedness()
            val leftHandIndex = handednessList.indexOfFirst { it.firstOrNull()?.displayName() == "Left" }

            if (leftHandIndex != -1) {
                leftHandCoordinates = extractHandCoordinates(result, leftHandIndex)
                leftHandCoordinates?.let { coords ->
                    leftHandPrediction = runTFLiteInference(coords)
                }
            }
        }

        if (handResult != null || poseResult != null) {
            return CombinedResultBundle(
                handResults = if (handResult != null) listOf(handResult) else emptyList(),
                poseResults = if (poseResult != null) listOf(poseResult) else emptyList(),
                inferenceTime = inferenceTimeMs,
                inputImageHeight = image.height,
                inputImageWidth = image.width,
                handCoordinates = leftHandCoordinates,
                poseCoordinates = poseCoordinates,
                handDetection = leftHandPrediction,
                poseDetection = ""
            )
        }
        return null
    }

    // --- New callback functions for combining results ---
    private fun receiveHandLivestreamResult(result: HandLandmarkerResult, input: MPImage) {
        latestHandResult = result
        maybeSendCombinedResult()
    }

    private fun receivePoseLivestreamResult(result: PoseLandmarkerResult, input: MPImage) {
        latestPoseResult = result
        maybeSendCombinedResult()
    }

    private fun maybeSendCombinedResult() {
        synchronized(tfliteLock) {
            if (isClosed) {
                return
            }

            if (latestHandResult != null && latestPoseResult != null) {
                val inferenceTime = SystemClock.uptimeMillis() - latestFrameTime

                var handPrediction = ""
                var posePrediction = ""
                var targetHandCoordinates: FloatArray? = null
                val poseCoordinates = extractPoseCoordinates(latestPoseResult!!)

                if (latestPoseResult!!.landmarks().isNotEmpty()) {
                    posePrediction = runTFLitePoseInference(poseCoordinates)
                }

                // run hand tflite inference only on left hand that is the lowest
                if (latestHandResult!!.landmarks().isNotEmpty()) {
                    val handednessList = latestHandResult!!.handedness()
                    val landmarksList = latestHandResult!!.landmarks()
                    var targetHandIndex = -1
                    var maxY: Float = -1.0f
                    var finalHandResult: HandLandmarkerResult? = null

                    // Iterate through all detected hands to find the target
                    handednessList.forEachIndexed { index, handedness ->
                        // checks for left hand
                        //if (handedness.firstOrNull()?.displayName() == "Left") {
                        // gets landmark for left hand
                        landmarksList.getOrNull(index)?.takeIf { it.isNotEmpty() }?.let { landmarks ->
                            val wristY = landmarks[0].y()
                            // checks for lowest left hand
                            if (wristY > maxY) {
                                maxY = wristY
                                targetHandIndex = index
                            }
                        }
                        //}
                    }

                    // process the lowest left hand
                    if (targetHandIndex != -1) {
                        targetHandCoordinates = extractHandCoordinates(latestHandResult!!, targetHandIndex)
                        targetHandCoordinates?.let {
                            handPrediction = runTFLiteInference(it)
                        }
                    }
                }

                if (!isClosed && combinedLandmarkerHelperListener != null) {
                    combinedLandmarkerHelperListener.onResults(
                        CombinedResultBundle(
                            handResults = listOf(latestHandResult!!),
                            poseResults = listOf(latestPoseResult!!),
                            inferenceTime = inferenceTime,
                            inputImageHeight = latestImage!!.height,
                            inputImageWidth = latestImage!!.width,
                            handCoordinates = targetHandCoordinates,
                            poseCoordinates = poseCoordinates,
                            handDetection = handPrediction,
                            poseDetection = posePrediction
                        )
                    )
                }

                latestHandResult = null
                latestPoseResult = null
            }
        }
    }

    /**
     * Extracts a 42-length array of [x0, y0, x1, y1, ..., x20, y20] from the first detected hand.
     * The coordinates are normalized relative to the wrist and scaled to the [-1, 1] range.
     * If no hand is present, returns a zero-filled array.
     */
    private fun extractHandCoordinates(result: HandLandmarkerResult, handIndex: Int): FloatArray? {
        val landmarksList = result.landmarks()
        if (handIndex < 0 || handIndex >= landmarksList.size) return null

        val handLandmarks = landmarksList[handIndex]
        if (handLandmarks.isEmpty()) return null

        val coords = FloatArray(42)
        val originX = handLandmarks[0].x()
        val originY = handLandmarks[0].y()
        var maxAbsValue = 0f

        val relativeCoords = FloatArray(42)
        for ((j, landmark) in handLandmarks.withIndex()) {
            var relativeX = landmark.x() - originX
            val relativeY = landmark.y() - originY

            // Only flip x for back camera
            if (!isFrontCameraActive) {
                relativeX *= -1
            }

            val base = j * 2
            relativeCoords[base] = relativeX
            relativeCoords[base + 1] = relativeY

            maxAbsValue = max(maxAbsValue, abs(relativeX))
            maxAbsValue = max(maxAbsValue, abs(relativeY))
        }

        if (maxAbsValue > 0) {
            for (j in relativeCoords.indices) {
                coords[j] = relativeCoords[j] / maxAbsValue
            }
        }

        return coords
    }


    private fun extractPoseCoordinates(result: PoseLandmarkerResult): FloatArray {
        val coords = FloatArray(9) { 0f }
        val pose = result.landmarks()

        if (pose.isNotEmpty() && pose[0].size > 19) {
            val landmarks = pose[0]

            // Select landmarks depending on camera orientation
            val (shoulderIdx, elbowIdx, handIdx) = if (isFrontCameraActive) {
                Triple(11, 13, 15)  // left shoulder, elbow, hand
            } else {
                Triple(12, 14, 16)  // right shoulder, elbow, hand
            }

            // Flip x only for back camera
            val shoulder_x = if (isFrontCameraActive) landmarks[shoulderIdx].x() else 1f - landmarks[shoulderIdx].x()
            val shoulder_y = landmarks[shoulderIdx].y()
            val shoulder_z = landmarks[shoulderIdx].z()

            val elbow_x = if (isFrontCameraActive) landmarks[elbowIdx].x() else 1f - landmarks[elbowIdx].x()
            val elbow_y = landmarks[elbowIdx].y()
            val elbow_z = landmarks[elbowIdx].z()

            val hand_x = if (isFrontCameraActive) landmarks[handIdx].x() else 1f - landmarks[handIdx].x()
            val hand_y = landmarks[handIdx].y()
            val hand_z = landmarks[handIdx].z()

            val shoulderElbowVec = floatArrayOf(
                shoulder_x - elbow_x,
                shoulder_y - elbow_y,
                shoulder_z - elbow_z
            )
            val handElbowVec = floatArrayOf(
                hand_x - elbow_x,
                hand_y - elbow_y,
                hand_z - elbow_z
            )

            val shoulderElbowDist = sqrt(
                shoulderElbowVec[0] * shoulderElbowVec[0] +
                        shoulderElbowVec[1] * shoulderElbowVec[1] +
                        shoulderElbowVec[2] * shoulderElbowVec[2]
            )
            val handElbowDist = sqrt(
                handElbowVec[0] * handElbowVec[0] +
                        handElbowVec[1] * handElbowVec[1] +
                        handElbowVec[2] * handElbowVec[2]
            )

            if (shoulderElbowDist > 0 && handElbowDist > 0) {
                val cosTheta =
                    (shoulderElbowVec[0] * handElbowVec[0] +
                            shoulderElbowVec[1] * handElbowVec[1] +
                            shoulderElbowVec[2] * handElbowVec[2]) / (shoulderElbowDist * handElbowDist)

                val theta = acos(cosTheta.coerceIn(-1f, 1f))

                coords[0] = shoulderElbowVec[0] / shoulderElbowDist
                coords[1] = shoulderElbowVec[1] / shoulderElbowDist
                coords[2] = shoulderElbowVec[2] / shoulderElbowDist
                coords[3] = handElbowVec[0] / handElbowDist
                coords[4] = handElbowVec[1] / handElbowDist
                coords[5] = handElbowVec[2] / handElbowDist
                coords[6] = theta
                coords[7] = shoulderElbowDist
                coords[8] = handElbowDist
            }
        }
        return coords
    }




    private fun runTFLitePoseInference(inputData: FloatArray): String{

        try {
            // 1. INITIALIZE ON FIRST USE (Lazy Initialization)
            if (poseTFLite == null) {
                // Add a guard clause here. If the helper was closed before the interpreter
                // had a chance to initialize, we must abort.
                if (isClosed) {
                    return "Error: Helper is closed."
                }
                Log.d(TAG, "Initializing TFLite interpreter.")
                val model = loadModelFile("keypoint_classifier (1).tflite")
                poseTFLite = Interpreter(model)
            }

            // 2. RUN INFERENCE
            // The `handTFLite` object is guaranteed to be valid here because the surrounding
            // synchronized block in the calling function prevents it from being closed.
            val output = Array(1) { FloatArray(3) }
            val inputArray = arrayOf(inputData)
            poseTFLite!!.run(inputArray, output) // We can use non-null assertion `!!` here

            val results = output[0]
            val maxIndex = results.indices.maxByOrNull { results[it] } ?: -1
            val confidence = results.getOrNull(maxIndex) ?: 0f

            Log.d("TFLITE", "Predicted pose class: $maxIndex with confidence: $confidence")
            return "Prediction: $maxIndex (Confidence: %.2f)".format(confidence)

        } catch (e: Exception) {
            // Catch any errors during model loading or inference.
            Log.e("TFLITE", "Error during inference", e)
            return "Error: ${e.message}"
        }

    }

    private fun runTFLiteInference(inputData: FloatArray): String {
        try {

            // Lazy initialization
            if (handTFLite == null) {
                if (isClosed) {
                    return "Error: Helper is closed."
                }
                Log.d(TAG, "Initializing TFLite interpreter.")
                val model = loadModelFile("keypoint_classifier_FINAL.tflite")
                handTFLite = Interpreter(model)
            }

            // Prepare model input/output
            val output = Array(1) { FloatArray(4) }
            val inputArray = arrayOf(inputData)
            handTFLite!!.run(inputArray, output)

            val results = output[0]
            val supinationIndex = 1 // supination class index
            results[supinationIndex] *= 0.7f

            val maxIndex = results.indices.maxByOrNull { results[it] } ?: -1
            val confidence = results[maxIndex]

            // if detected as supination but weak confidence, treat as neutral
            if (maxIndex == supinationIndex && confidence < 0.60f) {
                return "Prediction: 0 (Confidence: %.2f)".format(confidence)
            }

            Log.d("TFLITE", "Predicted hand class: $maxIndex with confidence: $confidence")
            return "Prediction: $maxIndex (Confidence: %.2f)".format(confidence)

        } catch (e: Exception) {
            Log.e("TFLITE", "Error during inference", e)
            return "Error: ${e.message}"
        }
    }


    private fun returnLivestreamError(error: RuntimeException) {
        combinedLandmarkerHelperListener?.onError(error.message ?: "An unknown error has occurred")
    }

    companion object {
        const val TAG = "hands"
        private const val MP_HAND_LANDMARKER_TASK = "hand_landmarker.task"

        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DEFAULT_HAND_DETECTION_CONFIDENCE = 0.7F
        const val DEFAULT_HAND_TRACKING_CONFIDENCE = 0.7F
        const val DEFAULT_HAND_PRESENCE_CONFIDENCE = 0.5F
        const val DEFAULT_NUM_HANDS = 1
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1

        const val DEFAULT_POSE_DETECTION_CONFIDENCE = 0.7F
        const val DEFAULT_POSE_TRACKING_CONFIDENCE = 0.7F
        const val DEFAULT_POSE_PRESENCE_CONFIDENCE = 0.5F
    }

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

    interface CombinedLandmarkerListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: CombinedResultBundle)
    }

    private fun drawMediaPipeAnnotations(
        bitmap: Bitmap,
        result: CombinedResultBundle
    ): Bitmap {
        val canvas = Canvas(bitmap) // do not copy bitmap again, use mutable bitmap passed in

        val imageWidth = bitmap.width
        val imageHeight = bitmap.height

        // Parse hand classification
        val classRegex = """Prediction: (\d+) \(Confidence: ([\d.]+)\)""".toRegex()
        val handMatch = classRegex.find(result.handDetection)
        val handClass = handMatch?.groupValues?.get(1)?.toIntOrNull() ?: -1

        // Parse pose classification
        val poseMatch = classRegex.find(result.poseDetection)
        val poseClass = poseMatch?.groupValues?.get(1)?.toIntOrNull() ?: -1

        // Determine colors based on classification
        val handHasIssue = handClass in 1..2
        val poseHasIssue = poseClass in 1..2

        if (result.handResults.isNotEmpty() && result.targetHandIndex != -1) {
            val handResult = result.handResults[0]
            val landmarksList = handResult.landmarks()
            val handednessList = handResult.handedness()

            if (result.targetHandIndex < landmarksList.size) {
                val targetLandmarks = landmarksList[result.targetHandIndex]

                if (targetLandmarks.isNotEmpty()) {
                    // Choose color based on hand classification
                    // Using a brighter, more saturated orange (Deep Orange/Amber)
                    val handColor = if (handHasIssue) Color.rgb(255, 140, 0) else Color.BLUE // Vivid orange or Blue

                    val linePaint = Paint().apply {
                        color = handColor
                        strokeWidth = 10f
                        style = Paint.Style.STROKE
                    }

                    val pointPaint = Paint().apply {
                        color = if (handHasIssue) Color.rgb(255, 180, 50) else Color.CYAN // Bright amber or Cyan
                        strokeWidth = 10f
                        style = Paint.Style.FILL
                    }

                    // Draw hand connections
                    HandLandmarker.HAND_CONNECTIONS.forEach { connection ->
                        val startLandmark = targetLandmarks[connection!!.start()]
                        val endLandmark = targetLandmarks[connection.end()]

                        canvas.drawLine(
                            startLandmark.x() * imageWidth,
                            startLandmark.y() * imageHeight,
                            endLandmark.x() * imageWidth,
                            endLandmark.y() * imageHeight,
                            linePaint
                        )
                    }

                    // Draw keypoints
                    for (landmark in targetLandmarks) {
                        canvas.drawPoint(
                            landmark.x() * imageWidth,
                            landmark.y() * imageHeight,
                            pointPaint
                        )
                    }

                    // Hand label
                    val wrist = targetLandmarks[0]
                    val handName = handednessList.getOrNull(result.targetHandIndex)?.firstOrNull()?.displayName() ?: "Unknown"

                    val labelPaint = Paint().apply {
                        color = Color.WHITE
                        textSize = 40f
                        style = Paint.Style.FILL
                        isAntiAlias = true
                        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
                        setShadowLayer(4f, 2f, 2f, Color.BLACK)
                    }

                    canvas.drawText(
                        "Bow Hand ($handName)",
                        wrist.x() * imageWidth + 20f,
                        wrist.y() * imageHeight - 20f,
                        labelPaint
                    )
                }
            }
        }

        // Prepare text paint styles
        val textPaint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.FILL
            textSize = 56f
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
        }

        // Semi-transparent white rectangle paint
        val labelBackgroundPaint = Paint().apply {
            color = Color.argb(180, 255, 255, 255) // semi-transparent white
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        // Fixed positions from top with margin - centered horizontally
        val topMargin = 120f
        var currentY = topMargin
        val lineSpacing = 60f
        val centerX = imageWidth / 2f  // Horizontal center of the image
        val padding = 16f

        // Draw hand message if there's an issue
        if (handHasIssue) {
            val handMessage = when (handClass) {
                1 -> "Pronate your wrist more"    // Supination
                2 -> "Supinate your wrist more"    // Too much pronation
                else -> ""
            }
            if (handMessage.isNotEmpty()) {
                val textWidth = textPaint.measureText(handMessage)
                val fm = textPaint.fontMetrics
                val textHeight = fm.bottom - fm.top

                // Rectangle coordinates
                val left = centerX - textWidth / 2 - padding
                val top = currentY + fm.top - padding
                val right = centerX + textWidth / 2 + padding
                val bottom = currentY + fm.bottom + padding

                canvas.drawRect(left, top, right, bottom, labelBackgroundPaint)

                canvas.drawText(handMessage, centerX, currentY, textPaint)
                currentY += (fm.bottom - fm.top) + lineSpacing
            }
        }

        // Draw pose message if there's an issue
        if (poseHasIssue) {
            val poseMessage = when (poseClass) {
                1 -> "Raise your elbow a bit"    // Low elbow
                2 -> "Lower your elbow a bit"    // Elbow too high
                else -> ""
            }
            if (poseMessage.isNotEmpty()) {
                val textWidth = textPaint.measureText(poseMessage)
                val fm = textPaint.fontMetrics
                val textHeight = fm.bottom - fm.top

                // Rectangle coordinates
                val left = centerX - textWidth / 2 - padding
                val top = currentY + fm.top - padding
                val right = centerX + textWidth / 2 + padding
                val bottom = currentY + fm.bottom + padding

                canvas.drawRect(left, top, right, bottom, labelBackgroundPaint)

                canvas.drawText(poseMessage, centerX, currentY, textPaint)
            }
        }

        return bitmap
    }
}
