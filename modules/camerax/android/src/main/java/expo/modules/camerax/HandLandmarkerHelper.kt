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
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Typeface
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
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
import kotlin.math.max
import kotlin.math.sqrt
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.String

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

    // Latest state for live stream
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
            isClosed = true
            handLandmarker?.close()
            handLandmarker = null
            poseLandmarker?.close()
            poseLandmarker = null
            handTFLite?.close()
            handTFLite = null
            poseTFLite?.close()
            poseTFLite = null
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
            combinedLandmarkerHelperListener?.onError(
                "Hand Landmarker failed to initialize.",
                if (e is RuntimeException) GPU_ERROR else OTHER_ERROR
            )
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
            combinedLandmarkerHelperListener?.onError(
                "Pose Landmarker failed to initialize.",
                if (e is RuntimeException) GPU_ERROR else OTHER_ERROR
            )
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

            isFrontCameraActive = isFrontCamera

            if (runningMode != RunningMode.LIVE_STREAM) {
                imageProxy.close()
                throw IllegalArgumentException("Attempting to call detectLiveStream while not using RunningMode.LIVE_STREAM")
            }

            latestFrameTime = SystemClock.uptimeMillis()

            val bitmapBuffer =
                Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)

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

            handLandmarker?.detectAsync(latestImage!!, latestFrameTime)
            poseLandmarker?.detectAsync(latestImage!!, latestFrameTime)
        }
    }

    // VIDEO MODE: detect + draw only the selected hand
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

        // Pick closest hand, using wrist cord, to pose landmark 16 using 2D screen-normalized coords
        if (
            handResult != null && handResult.landmarks().isNotEmpty() &&
            poseResult != null && poseResult.landmarks().isNotEmpty() &&
            poseResult.landmarks()[0].size > 16
        ) {
            val poseLandmarks = poseResult.landmarks()[0]
            val poseHandX = poseLandmarks[16].x()
            val poseHandY = poseLandmarks[16].y()

            val landmarksList = handResult.landmarks()
            var bestDist = Float.MAX_VALUE
            val threshold = 0.2f

            landmarksList.forEachIndexed { index, handLandmarks ->
                if (handLandmarks.isNotEmpty()) {
                    val wrist = handLandmarks[0]
                    val dx = wrist.x() - poseHandX
                    val dy = wrist.y() - poseHandY
                    val dist = sqrt(dx * dx + dy * dy)

                    if (dist < threshold && dist < bestDist) {
                        bestDist = dist
                        targetHandIndex = index
                    }
                }
            }

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

        val cleanBitmap = frame.copy(Bitmap.Config.ARGB_8888, true)
        val annotatedFrame = drawMediaPipeAnnotations(cleanBitmap, resultBundle)

        return Pair(resultBundle, annotatedFrame)
    }

    // IMAGE MODE (still image)
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

        return if (handResult != null || poseResult != null) {
            CombinedResultBundle(
                handResults = if (handResult != null) listOf(handResult) else emptyList(),
                poseResults = if (poseResult != null) listOf(poseResult) else emptyList(),
                inferenceTime = inferenceTimeMs,
                inputImageHeight = image.height,
                inputImageWidth = image.width,
                handCoordinates = leftHandCoordinates,
                poseCoordinates = poseCoordinates,
                handDetection = leftHandPrediction,
                poseDetection = "",
                targetHandIndex = -1
            )
        } else {
            null
        }
    }

    // ==== LIVE STREAM CALLBACKS ====

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
            if (isClosed) return

            if (latestHandResult != null && latestPoseResult != null && latestImage != null) {
                val inferenceTime = SystemClock.uptimeMillis() - latestFrameTime

                var handPrediction = ""
                var posePrediction = ""
                var targetHandCoordinates: FloatArray? = null
                var selectedHandIndex = -1

                val poseCoordinates = extractPoseCoordinates(latestPoseResult!!)

                if (latestPoseResult!!.landmarks().isNotEmpty()) {
                    posePrediction = runTFLitePoseInference(poseCoordinates)
                }

                // Pick closest hand to pose landmark 16 using 2D distance
                if (
                    latestHandResult!!.landmarks().isNotEmpty() &&
                    latestPoseResult!!.landmarks().isNotEmpty() &&
                    latestPoseResult!!.landmarks()[0].size > 16
                ) {
                    val pose16 = latestPoseResult!!.landmarks()[0][16]
                    val poseX = pose16.x()
                    val poseY = pose16.y()

                    var bestDist = Float.MAX_VALUE
                    val threshold = 0.2f

                    latestHandResult!!.landmarks().forEachIndexed { index, handLandmarks ->
                        if (handLandmarks.isNotEmpty()) {
                            val wrist = handLandmarks[0]
                            val dx = wrist.x() - poseX
                            val dy = wrist.y() - poseY
                            val dist = sqrt(dx * dx + dy * dy)

                            if (dist < threshold && dist < bestDist) {
                                bestDist = dist
                                selectedHandIndex = index
                            }
                        }
                    }

                    if (selectedHandIndex != -1) {
                        targetHandCoordinates = extractHandCoordinates(latestHandResult!!, selectedHandIndex)
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
                            poseDetection = posePrediction,
                            targetHandIndex = selectedHandIndex
                        )
                    )
                }

                latestHandResult = null
                latestPoseResult = null
            }
        }
    }

    /**
     * Extracts a 42-length array of [x0, y0, x1, y1, ..., x20, y20] from the selected hand.
     * Coordinates are normalized relative to the wrist and scaled to [-1, 1].
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

            // Flip x only for back camera
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

            val (shoulderIdx, elbowIdx, handIdx) = if (isFrontCameraActive) {
                Triple(11, 13, 15) // left side
            } else {
                Triple(12, 14, 16) // right side
            }

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

    private fun runTFLitePoseInference(inputData: FloatArray): String {
        try {
            if (poseTFLite == null) {
                if (isClosed) {
                    return "Error: Helper is closed."
                }
                Log.d(TAG, "Initializing pose TFLite interpreter.")
                val model = loadModelFile("keypoint_classifier (1).tflite")
                poseTFLite = Interpreter(model)
            }

            val output = Array(1) { FloatArray(3) }
            val inputArray = arrayOf(inputData)
            poseTFLite!!.run(inputArray, output)

            val results = output[0]
            val maxIndex = results.indices.maxByOrNull { results[it] } ?: -1
            val confidence = results.getOrNull(maxIndex) ?: 0f

            Log.d("TFLITE", "Predicted pose class: $maxIndex with confidence: $confidence")
            return "Prediction: $maxIndex (Confidence: %.2f)".format(confidence)
        } catch (e: Exception) {
            Log.e("TFLITE", "Error during pose inference", e)
            return "Error: ${e.message}"
        }
    }

    private fun runTFLiteInference(inputData: FloatArray): String {
        try {
            if (handTFLite == null) {
                if (isClosed) {
                    return "Error: Helper is closed."
                }
                Log.d(TAG, "Initializing hand TFLite interpreter.")
                val model = loadModelFile("keypoint_classifier_FINAL.tflite")
                handTFLite = Interpreter(model)
            }

            val output = Array(1) { FloatArray(4) }
            val inputArray = arrayOf(inputData)
            handTFLite!!.run(inputArray, output)

            val results = output[0]
            val supinationIndex = 1
            results[supinationIndex] *= 0.7f

            val maxIndex = results.indices.maxByOrNull { results[it] } ?: -1
            val confidence = results[maxIndex]

            if (maxIndex == supinationIndex && confidence < 0.60f) {
                return "Prediction: 0 (Confidence: %.2f)".format(confidence)
            }

            Log.d("TFLITE", "Predicted hand class: $maxIndex with confidence: $confidence")
            return "Prediction: $maxIndex (Confidence: %.2f)".format(confidence)
        } catch (e: Exception) {
            Log.e("TFLITE", "Error during hand inference", e)
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
        const val DEFAULT_NUM_HANDS = 2
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
        val canvas = Canvas(bitmap)
        val imageWidth = bitmap.width
        val imageHeight = bitmap.height

        val classRegex = """Prediction: (\d+) \(Confidence: ([\d.]+)\)""".toRegex()
        val handMatch = classRegex.find(result.handDetection)
        val handClass = handMatch?.groupValues?.get(1)?.toIntOrNull() ?: -1

        val poseMatch = classRegex.find(result.poseDetection)
        val poseClass = poseMatch?.groupValues?.get(1)?.toIntOrNull() ?: -1

        val handHasIssue = handClass in 1..2
        val poseHasIssue = poseClass in 1..2

        // Draw only the selected hand (targetHandIndex) from the first HandLandmarkerResult
        if (result.handResults.isNotEmpty() && result.targetHandIndex != -1) {
            val handResult = result.handResults[0]
            val landmarksList = handResult.landmarks()
            val handednessList = handResult.handedness()

            if (result.targetHandIndex < landmarksList.size) {
                val targetLandmarks = landmarksList[result.targetHandIndex]

                if (targetLandmarks.isNotEmpty()) {
                    val handColor =
                        if (handHasIssue) Color.rgb(255, 140, 0) else Color.BLUE

                    val linePaint = Paint().apply {
                        color = handColor
                        strokeWidth = 10f
                        style = Paint.Style.STROKE
                    }

                    val pointPaint = Paint().apply {
                        color = if (handHasIssue) Color.rgb(255, 180, 50) else Color.CYAN
                        strokeWidth = 10f
                        style = Paint.Style.FILL
                    }

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

                    for (landmark in targetLandmarks) {
                        canvas.drawPoint(
                            landmark.x() * imageWidth,
                            landmark.y() * imageHeight,
                            pointPaint
                        )
                    }

                    val wrist = targetLandmarks[0]
                    val handName =
                        handednessList.getOrNull(result.targetHandIndex)?.firstOrNull()?.displayName()
                            ?: "Unknown"

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

        val textPaint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.FILL
            textSize = 56f
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
        }

        val labelBackgroundPaint = Paint().apply {
            color = Color.argb(180, 255, 255, 255)
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        val topMargin = 120f
        var currentY = topMargin
        val lineSpacing = 60f
        val centerX = imageWidth / 2f
        val padding = 16f

        if (handHasIssue) {
            val handMessage = when (handClass) {
                1 -> "Pronate your wrist more"
                2 -> "Supinate your wrist more"
                else -> ""
            }
            if (handMessage.isNotEmpty()) {
                val textWidth = textPaint.measureText(handMessage)
                val fm = textPaint.fontMetrics

                val left = centerX - textWidth / 2 - padding
                val top = currentY + fm.top - padding
                val right = centerX + textWidth / 2 + padding
                val bottom = currentY + fm.bottom + padding

                canvas.drawRect(left, top, right, bottom, labelBackgroundPaint)
                canvas.drawText(handMessage, centerX, currentY, textPaint)
                currentY += (fm.bottom - fm.top) + lineSpacing
            }
        }

        if (poseHasIssue) {
            val poseMessage = when (poseClass) {
                1 -> "Raise your elbow a bit"
                2 -> "Lower your elbow a bit"
                else -> ""
            }
            if (poseMessage.isNotEmpty()) {
                val textWidth = textPaint.measureText(poseMessage)
                val fm = textPaint.fontMetrics

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
