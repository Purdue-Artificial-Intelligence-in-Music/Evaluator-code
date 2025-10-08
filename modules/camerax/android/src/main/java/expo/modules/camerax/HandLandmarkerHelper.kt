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

    fun detectLiveStream(
        imageProxy: ImageProxy,
        isFrontCamera: Boolean
    ) {
        // Synchronize this entire block to prevent it from running at the same time as
        // the clearLandmarkers() function.
        synchronized(tfliteLock) {
            // 1. GUARD CLAUSE: If the helper is already closed, ignore all new frames.
            if (isClosed) {
                // It is CRITICAL to close the imageProxy before returning, otherwise,
                // the camera analyzer will stall.
                imageProxy.close()
                return
            }

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
    fun detectVideoFile(videoUri: Uri, inferenceIntervalMs: Long): CombinedResultBundle? {
        if (runningMode != RunningMode.VIDEO) {
            throw IllegalArgumentException("Attempting to call detectVideoFile while not using RunningMode.VIDEO")
        }

        // might not account for complex threading
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, videoUri)
        val videoLengthMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: return null
        val firstFrame = retriever.getFrameAtTime(0) ?: return null

        val handResultList = mutableListOf<HandLandmarkerResult>()
        val poseResultList = mutableListOf<PoseLandmarkerResult>()

        val numberOfFrames = videoLengthMs.div(inferenceIntervalMs)

        for (i in 0..numberOfFrames) {
            val timestampMs = i * inferenceIntervalMs
            val frame = retriever.getFrameAtTime(timestampMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST)
            frame?.let {
                val mpImage = BitmapImageBuilder(it).build()
                handLandmarker?.detectForVideo(mpImage, timestampMs)?.let { handResult -> handResultList.add(handResult) }
                poseLandmarker?.detectForVideo(mpImage, timestampMs)?.let { poseResult -> poseResultList.add(poseResult) }
            }
        }
        retriever.release()

        return CombinedResultBundle(
            handResults = handResultList,
            poseResults = poseResultList,
            inferenceTime = 0, // Placeholder
            inputImageHeight = firstFrame.height,
            inputImageWidth = firstFrame.width,
            handCoordinates = null,
            poseCoordinates = null,
            handDetection = "",
            poseDetection = ""
        )
    }

    fun detectAndDrawVideoFrame(frame: Bitmap?, timestampMs: Long): Pair<CombinedResultBundle?, Bitmap?> {
        if (runningMode != RunningMode.VIDEO) {
            throw IllegalArgumentException("Attempting to call detectVideoFile while not using RunningMode.VIDEO")
        }
        if (frame == null) return Pair(null, null)
        val startTime = SystemClock.uptimeMillis()

        val mpImage = BitmapImageBuilder(frame).build()
        val handResult = handLandmarker?.detectForVideo(mpImage, timestampMs)
        val poseResult = poseLandmarker?.detectForVideo(mpImage, timestampMs)

        var handCoords: FloatArray? = null
        var handPrediction: String? = null
        if (handResult != null && handResult.landmarks().isNotEmpty()) {
            handCoords = extractHandCoordinates(handResult)
            handPrediction = runTFLiteInference(handCoords)
        } else {
            Log.d("Encode", "handResult is null")
        }

        var poseCoords: FloatArray? = null
        var posePrediction: String? = null
        if (poseResult != null && poseResult.landmarks().isNotEmpty()) {
            poseCoords = extractPoseCoordinates(poseResult)
            posePrediction = runTFLitePoseInference(poseCoords)
        } else {
            Log.d("Encode", "poseResult is null")
        }
        val inferenceTime = SystemClock.uptimeMillis() - startTime
        Log.d("Encode", "inferenceTime: $inferenceTime")

        val resultBundle = CombinedResultBundle(
            handResults = if (handResult != null) listOf(handResult) else emptyList(),
            poseResults = if (poseResult != null) listOf(poseResult) else emptyList(),
            inferenceTime = inferenceTime,
            inputImageHeight = frame.height,
            inputImageWidth = frame.width,
            handCoordinates = handCoords,
            poseCoordinates = poseCoords,
            handDetection = handPrediction ?: "No hand detected",
            poseDetection = posePrediction ?: "No pose detected"
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
        if (handIndex < 0 || handIndex >= landmarksList.size) {
            return null
        }

        val handLandmarks = landmarksList[handIndex]
        if (handLandmarks.isEmpty()) {
            return null
        }

        val coords = FloatArray(42)
        val originX = handLandmarks[0].x()
        val originY = handLandmarks[0].y()
        var maxAbsValue = 0f

        val relativeCoords = FloatArray(42)
        for ((j, landmark) in handLandmarks.withIndex()) {
            val relativeX = landmark.x() - originX
            val relativeY = landmark.y() - originY
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
            val right_shoulder = landmarks[12]
            val elbow = landmarks[14]
            val right_hand = landmarks[16]
            val shoulder_elbow_dist_vec = floatArrayOf(
                right_shoulder.x() - elbow.x(),
                right_shoulder.y() - elbow.y(),
                right_shoulder.z() - elbow.z()
            )
            val hand_elbow_dist_vec = floatArrayOf(
                right_hand.x() - elbow.x(),
                right_hand.y() - elbow.y(),
                right_hand.z() - elbow.z()
            )
            val shoulder_elbow_dist = sqrt(shoulder_elbow_dist_vec[0].pow(2) + shoulder_elbow_dist_vec[1].pow(2) + shoulder_elbow_dist_vec[2].pow(2))
            val hand_elbow_dist = sqrt(hand_elbow_dist_vec[0].pow(2) + hand_elbow_dist_vec[1].pow(2) + hand_elbow_dist_vec[2].pow(2))
            if (shoulder_elbow_dist > 0 && hand_elbow_dist > 0) {
                val shoulder_elbow_dist_norm_x = shoulder_elbow_dist_vec[0] / shoulder_elbow_dist
                val shoulder_elbow_dist_norm_y = shoulder_elbow_dist_vec[1] / shoulder_elbow_dist
                val shoulder_elbow_dist_norm_z = shoulder_elbow_dist_vec[2] / shoulder_elbow_dist
                val hand_elbow_dist_norm_x = hand_elbow_dist_vec[0] / hand_elbow_dist
                val hand_elbow_dist_norm_y = hand_elbow_dist_vec[1] / hand_elbow_dist
                val hand_elbow_dist_norm_z = hand_elbow_dist_vec[2] / hand_elbow_dist
                val cos_theta_3d = ((shoulder_elbow_dist_vec[0] * hand_elbow_dist_vec[0]) + (shoulder_elbow_dist_vec[1] * hand_elbow_dist_vec[1]) + (shoulder_elbow_dist_vec[2] * hand_elbow_dist_vec[2])) / (shoulder_elbow_dist * hand_elbow_dist)
                val clipped_cos_theta_3d = cos_theta_3d.coerceIn(-1.0f, 1.0f)
                val theta_rad_3d = acos(clipped_cos_theta_3d)
                coords[0] = shoulder_elbow_dist_norm_x
                coords[1] = shoulder_elbow_dist_norm_y
                coords[2] = shoulder_elbow_dist_norm_z
                coords[3] = hand_elbow_dist_norm_x
                coords[4] = hand_elbow_dist_norm_y
                coords[5] = hand_elbow_dist_norm_z
                coords[6] = theta_rad_3d
                coords[7] = shoulder_elbow_dist
                coords[8] = hand_elbow_dist
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
            //var newArray = mutableListOf(inputData)

            for (i in inputData.indices step 2) {
                inputData[i] = inputData[i] * -1
            }


            //val newOldArray = newArray.toTypedArray()

            // Lazy intitialization
            if (handTFLite == null) {
                // Add a guard clause here. If the helper was closed before the interpreter
                // had a chance to initialize, we must abort.
                if (isClosed) {
                    return "Error: Helper is closed."
                }
                Log.d(TAG, "Initializing TFLite interpreter.")
                val model = loadModelFile("keypoint_classifier_FINAL.tflite")
                handTFLite = Interpreter(model)
            }

            val output = Array(1) { FloatArray(4) }
            val inputArray = arrayOf(inputData)
            handTFLite!!.run(inputData, output) // !!: non-null assertion

            val results = output[0]
            val maxIndex = results.indices.maxByOrNull { results[it] } ?: -1
            val confidence = results.getOrNull(maxIndex) ?: 0f

            Log.d("TFLITE", "Predicted hand class: $maxIndex with confidence: $confidence")
            return "Prediction: $maxIndex (Confidence: %.2f)".format(confidence)

        } catch (e: Exception) {
            // Catch any errors during model loading or inference.
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
        val poseDetection: String
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

        if (result.handResults.isNotEmpty()) {
            val linePaint = Paint().apply {
                color = Color.RED
                strokeWidth = 8f
                style = Paint.Style.STROKE
            }

            val pointPaint = Paint().apply {
                color = Color.YELLOW
                strokeWidth = 8f
                style = Paint.Style.FILL
            }

            val handResult = result.handResults[0]
            for (landmarks in handResult.landmarks()) {
                // draw hand connections
                HandLandmarker.HAND_CONNECTIONS.forEach { connection ->
                    val startLandmark = landmarks.get(connection!!.start())
                    val endLandmark = landmarks.get(connection.end())

                    canvas.drawLine(
                        startLandmark.x() * imageWidth,
                        startLandmark.y() * imageHeight,
                        endLandmark.x() * imageWidth,
                        endLandmark.y() * imageHeight,
                        linePaint
                    )
                }

                for (landmark in landmarks) {
                    canvas.drawPoint(
                        landmark.x() * imageWidth,
                        landmark.y() * imageHeight,
                        pointPaint
                    )
                }
            }
        }

        // write down classificartion
        val textPaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.FILL
            textSize = 56f
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }

        val strokePaint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.STROKE
            strokeWidth = 6f
            textSize = 56f
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }

        val handText = "hands: ${result.handDetection}"
        val poseText = "pose: ${result.poseDetection}"

        canvas.drawText(handText, 50f, 240f, strokePaint)
        canvas.drawText(poseText, 50f, 310f, strokePaint)

        canvas.drawText(handText, 50f, 240f, textPaint)
        canvas.drawText(poseText, 50f, 310f, textPaint)

        return bitmap
    }
}
