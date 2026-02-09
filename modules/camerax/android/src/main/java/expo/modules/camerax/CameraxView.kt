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
    HandLandmarkerHelper.CombinedLandmarkerListener {

    private val onDetectionResult by EventDispatcher()
    private val onNoDetection by EventDispatcher()
    private val onSessionEnd by EventDispatcher()
    private val onCalibrated by EventDispatcher()

    private val activity get() = requireNotNull(appContext.activityProvider?.currentActivity)

    private val frameLayout = FrameLayout(context)
    private var camera: Camera? = null
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private val viewFinder: PreviewView = PreviewView(context)
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    private var detector: Detector? = null

    private var calibrationCount: Int = 0
    private var calibrationCorrect: Int = 0
    private var isCalibrated: Boolean = false

    private lateinit var handLandmarkerHelper: HandLandmarkerHelper

    private val profile = Profile()
    private var userId: String = "default_user"
    //private val userId = "session" // You can dynamically change this per user

    private var isDetectionEnabled = false
    private var isCameraActive = false
    private var lensType = CameraSelector.LENS_FACING_BACK
    private var flip: Boolean = false

    private lateinit var overlayView: OverlayView

    private var latestBowResults: Detector.returnBow? = null
    private var latestHandPoints: List<HandLandmarkerResult> = emptyList()
    private var latestPosePoints: List<PoseLandmarkerResult> = emptyList()
    private var latestHandDetection: String = ""
    private var latestPoseDetection: String = ""

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
        detector = Detector(context, this)
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
        Log.d("DetectionEnabled: ", enabled.toString())
        if (isDetectionEnabled == enabled) return
        isDetectionEnabled = enabled
        isCalibrated = false
        calibrationCount = 0
        calibrationCorrect = 0

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
            detector?.resetHeaps()
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

    fun skipCalibration(skip: Boolean) {
        if (skip) {
            isCalibrated = true
            calibrationCorrect = 0
            calibrationCount = 0
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
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        camera = null
        preview = null
        imageAnalyzer = null
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

        if (lensType == CameraSelector.LENS_FACING_FRONT && flip){
            viewFinder.scaleX = -1f
        } else {
            viewFinder.scaleX = 1f
        }

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
                    if (lensType == CameraSelector.LENS_FACING_FRONT && !flip) {
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

                // Perform YOLO detection
                performDetection(rotatedBitmap)
                handLandmarkerHelper.detectBitmap(rotatedBitmap, (lensType == CameraSelector.LENS_FACING_FRONT && !flip))
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

    // ===== Detector Callbacks =====
    override fun detected(results: Detector.YoloResults, sourceWidth: Int, sourceHeight: Int) {
        if (!isDetectionEnabled) return
        val bowPoints = detector?.classify(results)
        if (isCalibrated) {
            if (bowPoints != null) {
                profile.addSessionData(userId, bowPoints)
            }
            latestBowResults = bowPoints
            updateOverlay()
            sendDetectionResults()
        } else {
            calibrationCount++
            calibrationCorrect++
            if (calibrationCount == 30) {
                if ((calibrationCorrect.toDouble() / calibrationCount.toDouble()) >= .6) {
                    isCalibrated = true
                    Log.d("Calibration", "Model Calibrated, emitting Event")
                    onCalibrated(mapOf("calibration" to isCalibrated))
                } else {
                    calibrationCount = 0
                    Log.d("Calibration", "Calibration Failed")
                }
            }
            latestBowResults = Detector.returnBow(-2, null, null, 0)
            updateOverlay()
            sendDetectionResults()
            onNoDetection(mapOf("message" to "No objects detected"))
        }
    }

    override fun noDetect() {
        if (!isDetectionEnabled) return
        if (!isCalibrated) {
            calibrationCount++
            if (calibrationCount == 30) {
                if ((calibrationCorrect.toDouble() / calibrationCount.toDouble()) >= .6) {
                    isCalibrated = true
                    Log.d("Calibration", "Model Calibrated, emitting Event")
                    onCalibrated(mapOf("calibration" to isCalibrated))
                } else {
                    calibrationCount = 0
                    Log.d("Calibration", "Calibration Failed")
                }
            }
        }
        latestBowResults = Detector.returnBow(-2, null, null, 0)
        updateOverlay()
        onNoDetection(mapOf("message" to "No objects detected"))
    }

    // ===== Hand + Pose =====
    override fun onError(error: String, errorCode: Int) {
        Log.e(TAG, "Hand/Pose error: $error ($errorCode)")
    }

    override fun onResults(resultBundle: HandLandmarkerHelper.CombinedResultBundle) {
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
