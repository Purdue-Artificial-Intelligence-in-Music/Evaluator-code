package expo.modules.camerax

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import expo.modules.kotlin.AppContext
import expo.modules.kotlin.viewevent.EventDispatcher
import expo.modules.kotlin.views.ExpoView
import android.content.pm.PackageManager
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraEffect.PREVIEW
import androidx.camera.core.CameraEffect.VIDEO_CAPTURE
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import com.google.mediapipe.proto.MediaPipeLoggingEnumsProto.ErrorCode
//import com.google.mediapipe.examples.handlandmarker.HandLandmarkerHelper
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

import android.util.Log

class CameraxView(context: Context, appContext: AppContext) : ExpoView(context, appContext), Detector.DetectorListener, HandLandmarkerHelper.CombinedLandmarkerListener {
    // Event dispatchers for different events
    private val onDetectionResult by EventDispatcher()
    private val onNoDetection by EventDispatcher()

    private val activity
        get() = requireNotNull(appContext.activityProvider?.currentActivity)

    private var frameLayout = FrameLayout(context)

    private var camera: Camera? = null
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null

    private var viewFinder: PreviewView = PreviewView(context)
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    // Detection related properties
    private var detector: Detector? = null
    private lateinit var handLandmarkerHelper: HandLandmarkerHelper
    private var isDetectionEnabled = false
    private var isCameraActive = false

    // Camera Props
    private var lensType = CameraSelector.LENS_FACING_BACK

    private lateinit var overlayView: OverlayView

    private var latestBowResults: Detector.returnBow? = null
    private var latestHandPoints: List<HandLandmarkerResult?> = emptyList()
    private var latestPosePoints: List<PoseLandmarkerResult?> = emptyList()
    private var latestHandDetection: String = ""
    private var latestPoseDetection: String = ""

    init {
        frameLayout.layoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        )
        addView(frameLayout)

        viewFinder.layoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        )
        viewFinder.isFocusableInTouchMode = true
        viewFinder.requestFocusFromTouch()
        installHierarchyFitter(viewFinder)
        frameLayout.addView(viewFinder)

        // OverlayView to overlay results on camera preview
        overlayView = OverlayView(context)
        overlayView.layoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        )
        frameLayout.addView(overlayView)

        // Initialize detector and hands stuff
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

    // Props that can be set from React Native
    fun setDetectionEnabled(enabled: Boolean) {
        if (isDetectionEnabled != enabled) {
            // set isDetectionEnabled to the value that was passed in
            isDetectionEnabled = enabled

            // stop detection
            if (!enabled) {
                // clear all results
                latestBowResults = null
                latestHandPoints = emptyList()
                latestPosePoints = emptyList()
                latestHandDetection = ""
                latestPoseDetection = ""

                // clear overlay drawings
                activity.runOnUiThread {
                    overlayView.clear()
                }
            }

            if (cameraProvider != null && isCameraActive) {
                bindCameraUseCases() // Rebind to add/remove image analyzer
            }
        }
    }

    fun setLensType(type: String) {
        val newLensType = when (type.lowercase()) {
            "front" -> CameraSelector.LENS_FACING_FRONT
            "back" -> CameraSelector.LENS_FACING_BACK
            else -> CameraSelector.LENS_FACING_BACK
        }
        
        if (lensType != newLensType) {
            lensType = newLensType
            if (cameraProvider != null && isCameraActive) {
                bindCameraUseCases() // Rebind with new camera
            }
        }
    }

    fun setCameraActive(active: Boolean) {
        if (isCameraActive != active) {
            isCameraActive = active
            if (active) {
                if (hasPermissions()) {
                    viewFinder.post { setupCamera() }
                }
            } else {
                stopCamera()
            }
        }
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        camera = null
        preview = null
        imageAnalyzer = null
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        // Camera will be started only when setCameraActive(true) is called
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        cameraExecutor.shutdown()
        cameraProvider?.unbindAll()
    }

    // If this is not called correctly, view finder will be black/blank
    // https://github.com/facebook/react-native/issues/17968#issuecomment-633308615
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

    /** Initialize CameraX, and prepare to bind the camera use cases */
    @SuppressLint("ClickableViewAccessibility")
    private fun setupCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(activity)
        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            cameraProvider = cameraProviderFuture.get()

            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(activity))
    }

    private fun bindCameraUseCases() {
        if (viewFinder.display == null) return

        val rotation = viewFinder.display.rotation

        // CameraProvider
        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        // CameraSelector
        val cameraSelector = CameraSelector.Builder().requireLensFacing(lensType).build()

        val aspectRatioStrategy = AspectRatioStrategy(
            AspectRatio.RATIO_16_9, AspectRatioStrategy.FALLBACK_RULE_NONE
        )
        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(aspectRatioStrategy)
            .build()

        // Preview
        preview = Preview.Builder()
            .setResolutionSelector(resolutionSelector)
            .setTargetRotation(rotation)
            .build()

        val useCaseGroupBuilder = UseCaseGroup.Builder()
            .addUseCase(preview!!)

        // Add ImageAnalysis if detection is enabled
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

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            val newCamera = cameraProvider.bindToLifecycle(
                activity as AppCompatActivity,
                cameraSelector,
                useCaseGroupBuilder.build()
            )

            camera = newCamera

            // Attach the viewfinder's surface provider to preview use case
            preview?.surfaceProvider = viewFinder.surfaceProvider
        } catch (exc: Exception) {
            // Handle exception
        }
    }

    private fun processImage(imageProxy: ImageProxy) {
        android.util.Log.d("live", "Processing Image")
        try {
            // Convert ImageProxy to Bitmap
            val bitmapBuffer = createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )

            imageProxy.use {
                bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
                imageProxy.planes[0].buffer.rewind()
                handLandmarkerHelper.detectLiveStream(
                    imageProxy,
                    false //CHANGE TO TRUE ONCE WE SWITCH TO FRONT CAMERA, WILL ALSO HAVE TO MIRROR IMAGE FOR DETECTOR PROCESSING
                )
            }


                // Rotate bitmap based on device orientation
            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                //postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
                //UNCOMMENT THIS WHEN DOING FRONT CAMERA
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, false
            )

            android.util.Log.d("live", "Calling detector.detect() with bitmap: ${rotatedBitmap.width}x${rotatedBitmap.height}")
            // Perform detection
            performDetection(rotatedBitmap, imageProxy.width, imageProxy.height)

        } catch (e: Exception) {
            // Handle error
        } finally {
            imageProxy.close()
        }
    }

    private fun performDetection(bitmap: Bitmap, sourceWidth: Int, sourceHeight: Int) {
        // Use the actual detector
        detector?.detect(bitmap)
    }

    // Implement DetectorListener interface methods
    override fun detected(results: Detector.YoloResults, sourceWidth: Int, sourceHeight: Int) {
        if (!isDetectionEnabled) {
            // if "stop detection" was already pressed
            return
        }

        val bowPoints = detector?.classify(results)

        val overlayWidth = overlayView.width
        val overlayHeight = overlayView.height
        val scaleFactor = max(
            overlayWidth.toFloat() / sourceWidth.toFloat(),
            overlayHeight.toFloat() / sourceHeight.toFloat()
        )
        val scaledImageWidth = sourceWidth.toFloat() * scaleFactor
        val scaledImageHeight = sourceHeight.toFloat() * scaleFactor
        val offsetX = (scaledImageWidth - overlayWidth.toFloat()) / 2f
        val offsetY = (scaledImageHeight - overlayHeight.toFloat()) / 2f

        bowPoints?.bow?.forEach { point ->
            point.x = (point.x.toDouble() * scaleFactor.toDouble()) - offsetX.toDouble()
            point.y = (point.y.toDouble() * scaleFactor.toDouble()) - offsetY.toDouble()
        }
        bowPoints?.string?.forEach { point ->
            point.x = (point.x.toDouble() * scaleFactor.toDouble()) - offsetX.toDouble()
            point.y = (point.y.toDouble() * scaleFactor.toDouble()) - offsetY.toDouble()
        }

        println("DETECTED")
        latestBowResults = bowPoints

        /*activity.runOnUiThread {
            if (bowPoints != null) {
                // 🔹 this makes your OverlayView actually draw the boxes
                overlayView.updateResults(bowPoints)
            }
        }*/

        /*val detectionResults: Map<String, Any> = mapOf(
            "classification" to (bowPoints?.classification ?: -1),
            "angle" to (bowPoints?.angle ?: 0),
            "bow" to (bowPoints?.bow?.map { point ->
                mapOf("x" to point.x, "y" to point.y)
            } ?: emptyList<Map<String, Double>>()),
            "string" to (bowPoints?.string?.map { point ->
                mapOf("x" to point.x, "y" to point.y)
            } ?: emptyList<Map<String, Double>>()),
            "sourceWidth" to sourceWidth,
            "sourceHeight" to sourceHeight,
            "viewWidth" to overlayView.width,
            "viewHeight" to overlayView.height
        )*/

        updateOverlay()
        //onDetectionResult(detectionResults)
        sendDetectionResults()
    }


    override fun noDetect() {
        /*activity.runOnUiThread {
            overlayView.updateResults(Detector.returnBow(-2, null, null, 0))
        }*/
        if (!isDetectionEnabled) {
            // if "stop detection" was already pressed
            return
        }
        latestBowResults = Detector.returnBow(-2, null, null, 0)
        updateOverlay()
        
        onNoDetection(mapOf("message" to "No objects detected"))
    }

    override fun onError(error: String, errorCode: Int) {
        android.util.Log.d("Hands", "Error from HandLandmarkerHelper: $error")
    }

    override fun onResults(resultBundle: HandLandmarkerHelper.CombinedResultBundle) {
        if (!isDetectionEnabled) {
            // if "stop detection" was already pressed
            return
        }
        android.util.Log.d("Hands", "Hand/Pose landmarks detected. Inference time: ${resultBundle.inferenceTime}ms $resultBundle")
        Log.d("cameraxview", "detected hands")

        latestHandPoints = resultBundle.handResults
        latestPosePoints = resultBundle.poseResults

        latestHandDetection = resultBundle.handDetection
        latestPoseDetection = resultBundle.poseDetection

        overlayView.setImageDimensions(
            resultBundle.inputImageWidth,
            resultBundle.inputImageHeight
        )

        //not sending any at the moment, still seems to be updating


        /*val detectionResults: Map<String, Any> = mapOf(
            "hands" to handPoints,
            "pose" to posePoints,
            "inferenceTime" to resultBundle.inferenceTime,
            "sourceWidth" to resultBundle.inputImageWidth,
            "sourceHeight" to resultBundle.inputImageHeight,
            "viewWidth" to overlayView.width,
            "viewHeight" to overlayView.height
        )*/

        updateOverlay()
        //onDetectionResult(detectionResults)
        sendDetectionResults()
    }


    private fun hasPermissions(): Boolean {
        val requiredPermissions = arrayOf(android.Manifest.permission.CAMERA)
        if (requiredPermissions.all {
                ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
            }) {
            return true
        }
        ActivityCompat.requestPermissions(
            activity,
            requiredPermissions,
            42 // random callback identifier
        )
        return false
    }

    companion object {
        private const val TAG = "MyCamera"
    }

    private fun updateOverlay() {
        // Use the latest bow, hand, and pose results
        //var index = 0
        //var x_coord_mid = 10000
        //Log.d("HANDS", latestHandPoints?.get(0)?.landmarks()?.get(0)?.get(0)?.x().toString())

//        if (latestHandPoints?.handedness().get(0).get(0).    == "Left") {
//            index = 0
//        } else {
//            index = 1
//        }





        activity.runOnUiThread {
            overlayView.updateResults(
                results = latestBowResults,
                hands = latestHandPoints?.firstOrNull(),
                pose = latestPosePoints?.firstOrNull(),
                handDetection = latestHandDetection,
                poseDetection = latestPoseDetection
            )
        }
    }

    //not sending any at the moment, still seems to be updating
    private fun sendDetectionResults() {
        val bowPoints = latestBowResults

        /*val detectionResults: Map<String, Any> = mapOf(
            "classification" to (bowPoints?.classification ?: -1),
            "angle" to (bowPoints?.angle ?: 0),
            "bow" to (bowPoints?.bow?.map { point -> mapOf("x" to point.x, "y" to point.y) } ?: emptyList()),
            "string" to (bowPoints?.string?.map { point -> mapOf("x" to point.x, "y" to point.y) } ?: emptyList()),
            "hands" to latestHandPoints.first() ?: emptyList(),
            "pose" to latestPosePoints.first ?: emptyList(),
            "viewWidth" to overlayView.width,
            "viewHeight" to overlayView.height
        )

        onDetectionResult(detectionResults)*/
    }


}