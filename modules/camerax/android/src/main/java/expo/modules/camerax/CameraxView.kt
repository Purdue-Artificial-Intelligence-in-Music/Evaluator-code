package expo.modules.camerax

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Environment
import android.util.Log
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import expo.modules.kotlin.AppContext
import expo.modules.kotlin.viewevent.EventDispatcher
import expo.modules.kotlin.views.ExpoView
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector


// MediaPipe
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * CameraxView:
 * - Handles camera input and MediaPipe analysis
 * - Uses Profile.kt for periodic session summaries (10-second intervals)
 */
class CameraxView(
    context: Context,
    appContext: AppContext
) : ExpoView(context, appContext),
    Detector.DetectorListener,
    CombinedLandmarkerListener {

    private val onDetectionResult by EventDispatcher()
    private val onNoDetection by EventDispatcher()
    private val onSessionEnd by EventDispatcher()

    private val activity get() = requireNotNull(appContext.activityProvider?.currentActivity)

    private val frameLayout = FrameLayout(context)
    private var camera: Camera? = null
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private val viewFinder: PreviewView = PreviewView(context)
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    private var detector: Detector? = null
    private lateinit var handLandmarkerHelper: HandLandmarkerHelper

    private val profile = Profile()
    private var userId: String = "default_user"
    //private val userId = "session" // You can dynamically change this per user

    private var isDetectionEnabled = false
    private var isCameraActive = false
    private var lensType = CameraSelector.LENS_FACING_BACK

    // NPU Pipeline Toggle
    private var useNpuPipeline = true  // Default to NPU when available

    // LiteRT instances (lazy init)
    private var handsLiteRt: Hands? = null
    private var poseLiteRt: Pose? = null

    private lateinit var overlayView: OverlayView

    private var latestBowResults: Detector.returnBow? = null
    private var latestHandPoints: List<HandLandmarkerResult> = emptyList()
    private var latestPosePoints: List<PoseLandmarkerResult> = emptyList()
    private var latestHandDetection: String = ""
    private var latestPoseDetection: String = ""

    // LiteRT results (for NPU pipeline)
    private var latestLiteRtHandResults: List<Hands.HandResult> = emptyList()
    private var latestLiteRtPoseResults: List<Pose.PoseLandmarks> = emptyList()
    private var latestImageWidth: Int = 0
    private var latestImageHeight: Int = 0

    init {
        // Root layout
        frameLayout.layoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        )
        addView(frameLayout)

        // Camera preview
        viewFinder.layoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        )
        viewFinder.isFocusableInTouchMode = true
        viewFinder.requestFocusFromTouch()
        installHierarchyFitter(viewFinder)
        frameLayout.addView(viewFinder)

        // Overlay
        overlayView = OverlayView(context).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }
        frameLayout.addView(overlayView)

        // Initialize detection tools
        // OLD APPROACH (eager init - didn't respect forcedDelegate prop):
        // detector = Detector(context, this)
        //
        // NEW APPROACH: All models initialized in initializePipeline(), called from setupCamera()
        // This way models load when camera preview starts (before "Start Detection" button)
        // and forcedDelegate prop is respected since props are applied before setupCamera()
        handLandmarkerHelper = HandLandmarkerHelper(
            context = context,
            runningMode = RunningMode.LIVE_STREAM,
            minHandDetectionConfidence = HandLandmarkerHelper.DEFAULT_HAND_DETECTION_CONFIDENCE,
            minHandTrackingConfidence = HandLandmarkerHelper.DEFAULT_HAND_TRACKING_CONFIDENCE,
            minHandPresenceConfidence = HandLandmarkerHelper.DEFAULT_HAND_PRESENCE_CONFIDENCE,
            maxNumHands = HandLandmarkerHelper.DEFAULT_NUM_HANDS,
            currentDelegate = HandLandmarkerHelper.DELEGATE_CPU,
            combinedLandmarkerHelperListener = this
        )
    }

    // ===== React Native Props =====
    fun setUserId(newUserId: String) {
        userId = newUserId
        Log.d("Data Collection", "User ID set to: $userId")
    }

    fun setMaxBowAngle(angle: Int) {
        detector?.setMaxAngle(angle)
    }

    fun setDetectionEnabled(enabled: Boolean) {
        if (isDetectionEnabled == enabled) return
        isDetectionEnabled = enabled

        if (enabled) {
            // Start a new session
            profile.createNewID(userId)
        } else {
            // Stop and finalize JSON files
            val detailedSummary = profile.endSessionAndGetSummary(userId)

            // send summary for display
            if (detailedSummary != null) {
                sendDetailedSummary(detailedSummary)
            } else {
                onSessionEnd(mapOf("error" to "No data available"))
            }

            // Reset overlay
            latestBowResults = null
            latestHandPoints = emptyList()
            latestPosePoints = emptyList()
            latestHandDetection = ""
            latestPoseDetection = ""
            activity.runOnUiThread { overlayView.clear() }
        }

        if (cameraProvider != null && isCameraActive) {
            bindCameraUseCases()
        }
    }

    private fun sendDetailedSummary(summary: Profile.SessionSummary) {
        try {
            val summaryMap = mapOf(
                "heightBreakdown" to summary.heightBreakdown,
                "angleBreakdown" to summary.angleBreakdown,
                "handPresenceBreakdown" to summary.handPresenceBreakdown,
                "handPostureBreakdown" to summary.handPostureBreakdown,
                "posePresenceBreakdown" to summary.posePresenceBreakdown,
                "elbowPostureBreakdown" to summary.elbowPostureBreakdown,
                "userId" to userId,
                "timestamp" to summary.timestamp
            )
            onSessionEnd(summaryMap)
        } catch (e: Exception) {
            Log.e(TAG, "Error sending session summary", e)
        }
    }

    fun setLensType(type: String) {
        val newLensType = when (type.lowercase()) {
            "front" -> CameraSelector.LENS_FACING_FRONT
            else -> CameraSelector.LENS_FACING_BACK
        }
        if (lensType != newLensType) {
            lensType = newLensType
            if (cameraProvider != null && isCameraActive) {
                bindCameraUseCases()
            }
        }
    }

    fun setUseNpuPipeline(enabled: Boolean) {
        if (useNpuPipeline == enabled) return
        useNpuPipeline = enabled
        Log.d(TAG, "NPU pipeline: $enabled")
        if (cameraProvider != null && isCameraActive) {
            initializePipeline()
        }
    }

    /**
     * Force a specific delegate for testing (HTP/GPU/CPU).
     * Pass null or empty string to return to automatic fallback.
     * Must be called BEFORE camera is active for effect.
     */
    fun setForcedDelegate(delegate: String?) {
        val type = when (delegate?.lowercase()) {
            "htp", "npu" -> DelegateManager.DelegateType.HTP
            "gpu" -> DelegateManager.DelegateType.GPU
            "cpu" -> DelegateManager.DelegateType.CPU
            else -> null
        }
        DelegateManager.forceDelegate(type)
        Log.d(TAG, "Delegate forced to: ${type ?: "auto"}")
        // If camera is already active, need to reinitialize all pipelines
        if (cameraProvider != null && isCameraActive) {
            // Close existing instances to force re-creation with new delegate
            detector?.close()
            detector = null
            handsLiteRt?.close()
            poseLiteRt?.close()
            handsLiteRt = null
            poseLiteRt = null
            DelegateManager.resetDelegateState()
            initializePipeline()
        }
    }

    fun setCameraActive(active: Boolean) {
        if (isCameraActive == active) return
        isCameraActive = active
        if (active) {
            if (hasPermissions()) {
                viewFinder.post { setupCamera() }
            }
        } else {
            stopCamera()
        }
    }

    // ===== Lifecycle =====
    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        cameraExecutor.shutdown()
        cameraProvider?.unbindAll()
        cameraProvider = null
        handsLiteRt?.close()
        poseLiteRt?.close()
        handsLiteRt = null
        poseLiteRt = null
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        camera = null
        preview = null
        imageAnalyzer = null
    }

    private fun initializePipeline() {
        // Initialize Detector (YOLO bow/string) - uses DelegateManager
        if (detector == null) {
            Log.d("NPUPipeline", "Creating Detector instance...")
            detector = Detector(context, this)
            Log.d("NPUPipeline", "Detector instance created")
        }

        if (useNpuPipeline) {
            Log.d("NPUPipeline", "initializePipeline starting...")
            try {
                if (handsLiteRt == null) {
                    Log.d("NPUPipeline", "Creating Hands instance...")
                    handsLiteRt = Hands(context)
                    Log.d("NPUPipeline", "Hands instance created: ${handsLiteRt != null}")
                }
                if (poseLiteRt == null) {
                    Log.d("NPUPipeline", "Creating Pose instance...")
                    poseLiteRt = Pose(context)
                    Log.d("NPUPipeline", "Pose instance created: ${poseLiteRt != null}")
                }
                Log.d("NPUPipeline", "initializePipeline complete - hands=${handsLiteRt != null}, pose=${poseLiteRt != null}")
            } catch (e: Exception) {
                Log.e("NPUPipeline", "initializePipeline FAILED: ${e.message}", e)
            }
        }
    }

    private fun installHierarchyFitter(view: ViewGroup) {
        view.setOnHierarchyChangeListener(object : ViewGroup.OnHierarchyChangeListener {
            override fun onChildViewRemoved(parent: View?, child: View?) = Unit
            override fun onChildViewAdded(parent: View?, child: View?) {
                parent?.measure(
                    View.MeasureSpec.makeMeasureSpec(measuredWidth, View.MeasureSpec.EXACTLY),
                    View.MeasureSpec.makeMeasureSpec(measuredHeight, View.MeasureSpec.EXACTLY)
                )
                parent?.layout(0, 0, parent.measuredWidth, parent.measuredHeight)
            }
        })
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupCamera() {
        // Preemptively initialize all models when camera starts
        // Models load while user sees preview, before "Start Detection" is pressed
        initializePipeline()

        val cameraProviderFuture = ProcessCameraProvider.getInstance(activity)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(activity))
    }

    private fun bindCameraUseCases() {
        if (viewFinder.display == null) return

        val rotation = viewFinder.display.rotation

        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensType)
            .build()

        val aspectRatioStrategy = AspectRatioStrategy(
            AspectRatio.RATIO_16_9,
            AspectRatioStrategy.FALLBACK_RULE_NONE
        )
        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(aspectRatioStrategy)
            .build()

        // --- Create and safely unwrap Preview ---
        val safePreview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setTargetRotation(rotation)
            .build()

        preview = safePreview

        // --- Build use case group ---
        val useCaseGroupBuilder = UseCaseGroup.Builder()
            .addUseCase(safePreview)

        // --- Add analyzer if enabled ---
        if (isDetectionEnabled) {
            imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setResolutionSelector(resolutionSelector)
                .setTargetRotation(rotation)
                .build()

            imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
                processImage(imageProxy)
            }

            useCaseGroupBuilder.addUseCase(imageAnalyzer!!)
        }

        val useCaseGroup = useCaseGroupBuilder.build()

        // --- Bind everything safely ---
        cameraProvider.unbindAll()
        try {
            safePreview.surfaceProvider = viewFinder.surfaceProvider
            camera = cameraProvider.bindToLifecycle(
                activity as AppCompatActivity,
                cameraSelector,
                useCaseGroup
            )
        } catch (exc: Exception) {
            Log.e(TAG, "Failed to bind camera use cases", exc)
        }
    }



    private fun processImage(imageProxy: ImageProxy) {
        try {
            val bitmapBuffer = createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )

            imageProxy.use {
                bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
                imageProxy.planes[0].buffer.rewind()

                // Run MediaPipe live stream hand detection
                /*
                handLandmarkerHelper.detectLiveStream(
                    imageProxy,
                    lensType != CameraSelector.LENS_FACING_BACK
                )

                 */



                // Rotate + mirror if needed
                val matrix = Matrix().apply {
                    postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                    if (lensType == CameraSelector.LENS_FACING_FRONT) {
                        postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
                    }
                }

                val rotatedBitmap = Bitmap.createBitmap(
                    bitmapBuffer, 0, 0,
                    bitmapBuffer.width, bitmapBuffer.height,
                    matrix, true
                )

                // set the bitmap in the overlayView so we can capture
                // frames for summary session
                overlayView.setBitmapFrame(rotatedBitmap)

                // OLD: Lazy init on first frame (caused 7s delay after "Start Detection")
                // initializePipeline()
                // NEW: Now called in setupCamera() - models ready before user hits button

                // Perform YOLO detection
                performDetection(rotatedBitmap)

                // Hand/Pose pipeline - NPU or MediaPipe
                if (useNpuPipeline) {
                    if (handsLiteRt != null && poseLiteRt != null) {
                        processWithNpuPipeline(rotatedBitmap)
                    } else {
                        // Fallback if initialization failed
                        handLandmarkerHelper.detectBitmap(rotatedBitmap, lensType == CameraSelector.LENS_FACING_FRONT)
                    }
                } else {
                    handLandmarkerHelper.detectBitmap(rotatedBitmap, lensType == CameraSelector.LENS_FACING_FRONT)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Analyzer failure", e)
        } finally {
            imageProxy.close()
        }
    }


    private fun performDetection(bitmap: Bitmap) {
        detector?.detect(bitmap)
    }

    private fun processWithNpuPipeline(bitmap: Bitmap) {
        if (!isDetectionEnabled) return
        Log.d("NPUPipeline", "processWithNpuPipeline frame=${bitmap.width}x${bitmap.height}")

        try {
            val isFront = lensType == CameraSelector.LENS_FACING_FRONT

            // Run pose detection first (provides hints for hand detection)
            // isFrontCamera passed for correct arm selection in classification
            val poseResults = poseLiteRt?.detectAndLandmark(bitmap, isFront) ?: emptyList()
            Log.d("NPUPipeline", "Pose results: ${poseResults.size} detections")

            // Convert pose results to hints for hand detection
            val poseHints = poseResults.map { it.landmarks }

            // Run hand detection with pose hints
            // isFrontCamera passed for correct X-flip in classification
            val handResults = handsLiteRt?.detectAndLandmark(bitmap, poseHints, isFront) ?: emptyList()
            Log.d("NPUPipeline", "Hand results: ${handResults.size} detections")

            // Store results
            latestLiteRtHandResults = handResults
            latestLiteRtPoseResults = poseResults
            latestImageWidth = bitmap.width
            latestImageHeight = bitmap.height

            // Get classification strings from results (classification now done in Hands.kt/Pose.kt)
            val handPrediction = if (handResults.isNotEmpty()) {
                handResults.first().prediction.ifEmpty { "No hand detected" }
            } else {
                "No hand detected"
            }

            val posePrediction = if (poseResults.isNotEmpty()) {
                poseResults.first().prediction.ifEmpty { "No pose detected" }
            } else {
                "No pose detected"
            }

            latestHandDetection = handPrediction
            latestPoseDetection = posePrediction

            // Feed Profile.kt for session summaries
            // Create CombinedResultBundle with empty MediaPipe results but valid detection strings
            // Profile.kt was updated to check detection strings for presence (not just handResults.isNotEmpty())
            val bundle = CombinedResultBundle(
                handResults = emptyList(),  // MediaPipe type - empty for NPU path
                poseResults = emptyList(),  // MediaPipe type - empty for NPU path
                inferenceTime = 0L,
                inputImageHeight = bitmap.height,
                inputImageWidth = bitmap.width,
                handCoordinates = null,
                poseCoordinates = null,
                handDetection = handPrediction,
                poseDetection = posePrediction
            )
            profile.addSessionData(userId, bundle)

            // Clear MediaPipe results when using NPU pipeline
            latestHandPoints = emptyList()
            latestPosePoints = emptyList()

            // Update overlay
            overlayView.setFrontCameraState(isFront)
            overlayView.setImageDimensions(bitmap.width, bitmap.height)
            updateOverlayLiteRt()
            Log.d("NPUPipeline", "Frame processed successfully")
        } catch (e: Exception) {
            Log.e("NPUPipeline", "processWithNpuPipeline FAILED: ${e.message}", e)
        }
    }

    // NOTE: Classification now handled directly in Hands.kt and Pose.kt
    // The old runLiteRtHandClassifier and runLiteRtPoseClassifier functions have been removed.
    // Results are read from handResult.prediction and poseResult.prediction instead.

    private fun updateOverlayLiteRt() {
        activity.runOnUiThread {
            overlayView.updateResultsLiteRt(
                results = latestBowResults,
                handResults = latestLiteRtHandResults,
                poseResults = latestLiteRtPoseResults,
                handDetection = latestHandDetection,
                poseDetection = latestPoseDetection,
                imageWidth = latestImageWidth,
                imageHeight = latestImageHeight
            )
        }
    }

    // ===== Detector Callbacks =====
    override fun detected(results: Detector.YoloResults, sourceWidth: Int, sourceHeight: Int) {
        if (!isDetectionEnabled) return
        val bowPoints = detector?.classify(results)
        if (bowPoints != null) {
            profile.addSessionData(userId, bowPoints)
        }
        latestBowResults = bowPoints
        updateOverlay()
        sendDetectionResults()
    }

    override fun noDetect() {
        if (!isDetectionEnabled) return
        latestBowResults = Detector.returnBow(-2, null, null, 0)
        updateOverlay()
        onNoDetection(mapOf("message" to "No objects detected"))
    }

    // ===== Hand + Pose =====
    override fun onError(error: String, errorCode: Int) {
        Log.e(TAG, "Hand/Pose error: $error ($errorCode)")
    }

    override fun onResults(resultBundle: CombinedResultBundle) {
        if (!isDetectionEnabled) return
        overlayView.setFrontCameraState(lensType == CameraSelector.LENS_FACING_FRONT)
        profile.addSessionData(userId, resultBundle)
        latestHandPoints = resultBundle.handResults
        latestPosePoints = resultBundle.poseResults
        latestHandDetection = resultBundle.handDetection
        latestPoseDetection = resultBundle.poseDetection
        overlayView.setImageDimensions(resultBundle.inputImageWidth, resultBundle.inputImageHeight)
        updateOverlay()
        sendDetectionResults()
    }

    // ===== File Save =====
    private fun hasPermissions(): Boolean {
        val requiredPermissions = arrayOf(android.Manifest.permission.CAMERA)
        val granted = requiredPermissions.all {
            ContextCompat.checkSelfPermission(context, it) ==
                    android.content.pm.PackageManager.PERMISSION_GRANTED
        }
        if (!granted) {
            ActivityCompat.requestPermissions(activity, requiredPermissions, 42)
        }
        return granted
    }

    private fun updateOverlay() {
        activity.runOnUiThread {
            overlayView.updateResults(
                results = latestBowResults,
                hands = latestHandPoints.firstOrNull(),
                pose = latestPosePoints.firstOrNull(),
                handDetection = latestHandDetection,
                poseDetection = latestPoseDetection
            )
        }
    }

    private fun sendDetectionResults() {
        // Optionally send to JS via event dispatcher
    }

    companion object {
        private const val TAG = "CameraxView"
    }
}
