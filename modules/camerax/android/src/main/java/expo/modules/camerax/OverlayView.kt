package expo.modules.camerax

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import com.google.mediapipe.tasks.vision.core.RunningMode
import kotlin.math.min
import kotlin.math.max


import kotlin.text.toFloat

class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var results: Detector.returnBow? = null
    private var imageWidth = 1
    private var imageHeight = 1
    private var boxPaint : Paint = Paint()
    private var textPaint = Paint()
    private var anglePaint = Paint()

    private var handLandmarkerResult: HandLandmarkerResult? = null
    private var poseLandmarkerResult: PoseLandmarkerResult? = null

    private var handDetect: String = ""
    private var poseDetect: String = ""

    private var linePaint = Paint()
    private var pointPaint = Paint()
    private var poseLinePaint = Paint()
    private var posePointPaint = Paint()
    private var testPaint = Paint()
    private var handDetectPaint = Paint()
    private var poseDetectPaint = Paint()

    // New properties to store the offsets
    private var xOffset: Float = 0f
    private var yOffset: Float = 0f
    private var handsScaleFactor = 0f

    companion object {
        // Classification constants
        const val CLASS_NONE = -2
        const val CLASS_PARTIAL = -1
        const val CLASS_CORRECT = 0
        const val CLASS_OUTSIDE = 1
        const val CLASS_TOO_HIGH = 2
        const val CLASS_TOO_LOW = 3

        const val ANGLE_RIGHT = 0
        const val ANGLE_WRONG = 1

        // Map classifications to labels
        val CLASSIFICATION_LABELS = mapOf(
            CLASS_NONE to "No detection",
            CLASS_PARTIAL to "Partial detection",
            CLASS_CORRECT to "Correct Bow Placement",
            CLASS_OUTSIDE to "Bow Outside Zone",
            CLASS_TOO_LOW to "Bow Too Low",
            CLASS_TOO_HIGH to "Bow Too High"
        )

        val ANGLE_LABELS = mapOf(
            ANGLE_RIGHT to "Correct Bow Angle",
            ANGLE_WRONG to "Incorrect Bow Angle"
        )

        const val LANDMARK_STROKE_WIDTH = 8f
    }

    init {
        boxPaint.setColor(Color.GREEN)
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 8f

        textPaint.setColor(Color.GREEN)
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 48f

        anglePaint.setColor(Color.GREEN)
        anglePaint.style = Paint.Style.FILL
        anglePaint.textSize = 48f

        testPaint.setColor(Color.GREEN)
        testPaint.style = Paint.Style.FILL
        testPaint.textSize = 48f

        linePaint.color =
            Color.RED
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL

        poseLinePaint.color = Color.WHITE
        poseLinePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        poseLinePaint.style = Paint.Style.STROKE

        posePointPaint.color = Color.BLUE
        posePointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        posePointPaint.style = Paint.Style.FILL

        handDetectPaint.setColor(Color.GREEN)
        handDetectPaint.style = Paint.Style.FILL
        handDetectPaint.textSize = 48f

        poseDetectPaint.setColor(Color.GREEN)
        poseDetectPaint.style = Paint.Style.FILL
        poseDetectPaint.textSize = 48f
    }

    fun returnDims() : Pair<Int, Int> {
        println("NEW WIDTH AND HEIGHT: $width, $height")
        return Pair(width, height)
    }

    override fun onDraw(canvas: Canvas) {
        if (results == null && handLandmarkerResult == null && handDetect.isEmpty() && poseDetect.isEmpty()) {
            // detection stopped, do not draw anything
            return
        }

        val scaleX = 1f
        val scaleY = 1f
        val scaleFactor = 1f //max(scaleX, scaleY)
        if (results?.classification != -2) {
            val stringBox = results?.string
            val bowBox = results?.bow

            // Determine if there's an issue
            val hasIssue = (results?.classification != null && results?.classification != 0) ||
                    (results?.angle != null && results?.angle == 1)

            // Choose box color based on classification
            val boxColor = if (hasIssue) Color.rgb(255, 140, 0) else Color.BLUE // Orange or Blue

            // Update boxPaint color dynamically
            boxPaint.color = boxColor

            if (stringBox != null) {
                canvas.drawLine(stringBox[0].x.toFloat() * scaleX,
                    stringBox[0].y.toFloat() * scaleY,
                    stringBox[1].x.toFloat() * scaleX,
                    stringBox[1].y.toFloat() * scaleY,
                    boxPaint)
                canvas.drawLine(stringBox[1].x.toFloat() * scaleX,
                    stringBox[1].y.toFloat() * scaleY,
                    stringBox[2].x.toFloat()  * scaleX,
                    stringBox[2].y.toFloat() * scaleY,
                    boxPaint)
                canvas.drawLine(stringBox[2].x.toFloat() * scaleX,
                    stringBox[2].y.toFloat() * scaleY,
                    stringBox[3].x.toFloat()  * scaleX,
                    stringBox[3].y.toFloat() * scaleY,
                    boxPaint)
                canvas.drawLine(stringBox[3].x.toFloat() * scaleX,
                    stringBox[3].y.toFloat() * scaleY,
                    stringBox[0].x.toFloat()  * scaleX,
                    stringBox[0].y.toFloat() * scaleY
                    , boxPaint)
            }
            if (bowBox != null) {
                canvas.drawLine(bowBox[0].x.toFloat() * scaleX,
                    bowBox[0].y.toFloat() * scaleY,
                    bowBox[1].x.toFloat() * scaleX,
                    bowBox[1].y.toFloat() * scaleY,
                    boxPaint)
                canvas.drawLine(bowBox[1].x.toFloat() * scaleX,
                    bowBox[1].y.toFloat() * scaleY,
                    bowBox[2].x.toFloat()  * scaleX,
                    bowBox[2].y.toFloat() * scaleY,
                    boxPaint)
                canvas.drawLine(bowBox[2].x.toFloat() * scaleX,
                    bowBox[2].y.toFloat() * scaleY,
                    bowBox[3].x.toFloat()  * scaleX,
                    bowBox[3].y.toFloat() * scaleY,
                    boxPaint)
                canvas.drawLine(bowBox[3].x.toFloat() * scaleX,
                    bowBox[3].y.toFloat() * scaleY,
                    bowBox[0].x.toFloat()  * scaleX,
                    bowBox[0].y.toFloat() * scaleY
                    , boxPaint)
            }

            // Classification labels - only show if not correct
            val classificationLabels = mapOf(
                0 to "",  // Correct - don't display
                1 to "Keep the bow in zone",    // Bow outside zone
                2 to "Lower the bow",    // Bow too high
                3 to "Lift the bow"    // Bow too low
            )

            val angleLabels = mapOf(
                0 to "",  // Correct - don't display
                1 to "Adjust your bow angle"    // Incorrect bow angle
            )

            // Prepare text paint styles for bow/string classification
            val orangeTextPaint = Paint().apply {
                color = Color.rgb(255, 140, 0) // Orange
                style = Paint.Style.FILL
                textSize = 56f
                isAntiAlias = true
                typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
                textAlign = Paint.Align.CENTER
            }

            val orangeStrokePaint = Paint().apply {
                color = Color.rgb(204, 85, 0) // Dark orange
                style = Paint.Style.STROKE
                strokeWidth = 6f
                textSize = 56f
                isAntiAlias = true
                typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
                textAlign = Paint.Align.CENTER
            }

            // Fixed positions from top - below hand/pose classifications
            val topMargin = 300f
            val lineSpacing = 70f
            val centerX = width / 2f

            var currentY = topMargin

            // Draw classification message if there's an issue
            if (results?.classification != null && results?.classification != 0) {
                val message = classificationLabels[results?.classification] ?: ""
                if (message.isNotEmpty()) {
                    canvas.drawText(message, centerX, currentY, orangeStrokePaint)
                    canvas.drawText(message, centerX, currentY, orangeTextPaint)
                    currentY += lineSpacing
                }
            }

            // Draw angle message if there's an issue
            if (results?.angle != null && results?.angle == 1) {
                val message = angleLabels[results?.angle] ?: ""
                if (message.isNotEmpty()) {
                    canvas.drawText(message, centerX, currentY, orangeStrokePaint)
                    canvas.drawText(message, centerX, currentY, orangeTextPaint)
                }
            }
        }

        // Draw Hand Landmarks
        handLandmarkerResult?.let { handResult ->
            for (landmarks in handResult.landmarks()) {
                // Draw hand connections
                HandLandmarker.HAND_CONNECTIONS.forEach { connection ->
                    canvas.drawLine(
                        landmarks.get(connection!!.start()).x() * imageWidth * handsScaleFactor + xOffset,
                        landmarks.get(connection.start()).y() * imageHeight * handsScaleFactor + yOffset,
                        landmarks.get(connection.end()).x() * imageWidth * handsScaleFactor + xOffset,
                        landmarks.get(connection.end()).y() * imageHeight * handsScaleFactor + yOffset,
                        linePaint
                    )
                }

                // Draw hand points
                for (normalizedLandmark in landmarks) {
                    canvas.drawPoint(
                        normalizedLandmark.x() * imageWidth * handsScaleFactor + xOffset,
                        normalizedLandmark.y() * imageHeight * handsScaleFactor + yOffset,
                        pointPaint
                    )
                }
            }
        }

        // Parse hand classification
        val handClassRegex = """Prediction: (\d+) \(Confidence: ([\d.]+)\)""".toRegex()
        val handMatch = handClassRegex.find(handDetect)
        val handClass = handMatch?.groupValues?.get(1)?.toIntOrNull() ?: -1

        // Parse pose classification
        val poseMatch = handClassRegex.find(poseDetect)
        val poseClass = poseMatch?.groupValues?.get(1)?.toIntOrNull() ?: -1

        // Hand/pose classification text paint styles
        val textPaint = Paint().apply {
            color = Color.rgb(255, 140, 0) // Orange
            style = Paint.Style.FILL
            textSize = 56f
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
        }

        val strokePaint = Paint().apply {
            color = Color.rgb(204, 85, 0) // Dark orange
            style = Paint.Style.STROKE
            strokeWidth = 6f
            textSize = 56f
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
        }

        val handTopMargin = 160f
        val poseTopMargin = 230f
        val centerX = width / 2f

        // Draw hand message if there's an issue
        if (handClass in 1..2) {
            val handMessage = when (handClass) {
                1 -> "Pronate your wrist more"    // Supination
                2 -> "Supinate your wrist more"    // Too much pronation
                else -> ""
            }
            if (handMessage.isNotEmpty()) {
                canvas.drawText(handMessage, centerX, handTopMargin, strokePaint)
                canvas.drawText(handMessage, centerX, handTopMargin, textPaint)
            }
        }

        // Draw pose message if there's an issue
        if (poseClass in 1..2) {
            val poseMessage = when (poseClass) {
                1 -> "Raise your elbow a bit"    // Low elbow
                2 -> "Lower your elbow a bit"    // Elbow too high
                else -> ""
            }
            if (poseMessage.isNotEmpty()) {
                canvas.drawText(poseMessage, centerX, poseTopMargin, strokePaint)
                canvas.drawText(poseMessage, centerX, poseTopMargin, textPaint)
            }
        }
    }

    fun updateResults(results: Detector.returnBow?,
                      hands:  HandLandmarkerResult?,
                      pose:   PoseLandmarkerResult?,
                      handDetection: String,
                      poseDetection: String, ) {
        this.results = results
        this.handLandmarkerResult = hands
        this.poseLandmarkerResult = pose
        this.handDetect = handDetection
        this.poseDetect = poseDetection
        invalidate()
    }

    fun setImageDimensions(imgWidth: Int, imgHeight: Int) {


        imageWidth = imgWidth
        imageHeight = imgHeight

        val scaleX = this.width.toFloat() / imageWidth.toFloat()
        val scaleY = this.height.toFloat() / imageHeight.toFloat()

        //need to be updated for non-live feed?
        handsScaleFactor = min(scaleX, scaleY)
    }

    fun clear() {
        results = null
        handLandmarkerResult = null
        poseLandmarkerResult = null
        handDetect = ""
        poseDetect = ""
        postInvalidate() // remove drawings
    }

}