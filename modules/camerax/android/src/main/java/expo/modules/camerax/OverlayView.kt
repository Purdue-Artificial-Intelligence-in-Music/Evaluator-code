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
        //val scaleX = width.toFloat() / imageWidth
        //val scaleY = height.toFloat() / imageHeight
        val scaleX = 1f
        val scaleY = 1f
        val scaleFactor = 1f //max(scaleX, scaleY)
        if (results?.classification != -2) {
            val stringBox = results?.string
            val bowBox = results?.bow
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
                //println("DRAWING BOW BOX")
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


            val label = CLASSIFICATION_LABELS[results?.classification] ?: "Unknown"
            val angle = ANGLE_LABELS[results?.angle] ?: "Unknown"

            canvas.drawText(
                "Classification: ${label}",
                50f,
                100f,
                textPaint
            )

            canvas.drawText(
                "${angle}",
                50f,
                160f,
                anglePaint
            )
        }




        //comment out pose detector, distracting on top of live feed. maybe use for skeleton later?
        /*poseLandmarkerResult?.let { poseResult ->
            if (poseResult.landmarks().isNotEmpty()) {
                val landmarks = poseResult.landmarks()[0]

                // Draw connections
                PoseLandmarker.POSE_LANDMARKS.forEach { connection ->
                    canvas.drawLine(
                        // Apply the x and y offsets to all coordinates
                        landmarks[connection.start()].x() * imageWidth * handsScaleFactor + xOffset,
                        landmarks[connection.start()].y() * imageHeight * handsScaleFactor + yOffset,
                        landmarks[connection.end()].x() * imageWidth * handsScaleFactor + xOffset,
                        landmarks[connection.end()].y() * imageHeight * handsScaleFactor + yOffset,
                        poseLinePaint
                    )
                }

                // Draw points
                for (normalizedLandmark in landmarks) {
                    canvas.drawPoint(
                        // Apply the x and y offsets
                        normalizedLandmark.x() * imageWidth * handsScaleFactor + xOffset,
                        normalizedLandmark.y() * imageHeight * handsScaleFactor + yOffset,
                        posePointPaint
                    )
                }
            }
        }*/

        // Draw Hand Landmarks
        handLandmarkerResult?.let { handResult ->
            for (landmarks in handResult.landmarks()) {
                // Draw hand connections
                HandLandmarker.HAND_CONNECTIONS.forEach { connection ->
                    canvas.drawLine(
                        // Apply the x and y offsets
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
                        // Apply the x and y offsets
                        normalizedLandmark.x() * imageWidth * handsScaleFactor + xOffset,
                        normalizedLandmark.y() * imageHeight * handsScaleFactor + yOffset,
                        pointPaint
                    )
                }
            }
        }





        canvas.drawText(
            "hands: ${handDetect}",
            50f,
            220f,
            handDetectPaint
        )

        canvas.drawText(
            "pose: ${poseDetect}",
            50f,
            280f,
            poseDetectPaint
        )


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




}