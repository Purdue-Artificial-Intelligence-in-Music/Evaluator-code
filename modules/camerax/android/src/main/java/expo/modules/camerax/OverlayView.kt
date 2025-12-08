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
import android.util.Log
import java.io.File
import java.io.FileOutputStream


import kotlin.text.toFloat

class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var results: Detector.returnBow? = null
    private var imageWidth = 1
    private var imageHeight = 1
    private var boxPaint : Paint = Paint()
    private var anglePaint = Paint()

    private var handLandmarkerResult: HandLandmarkerResult? = null
    private var poseLandmarkerResult: PoseLandmarkerResult? = null

    private var handDetect: String = ""
    private var poseDetect: String = ""

    private var centerX = 0F
    private var now = 0L

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
    private var bowImageWidth = 1
    private var bowImageHeight = 1

    private var bowMessage = ""
    private var angleMessage = ""

    // to save frames, start by saving to bitmap
    private var overlayBitmap: Bitmap? = null
    private var tempCanvas: Canvas? = null

    private var file_list: MutableList<String> = mutableListOf()

    private var lastBowIssue: String? = null
    private var bowIssueStartTime: Long = 0L
    private var bowIssueLastShownTime: Long = 0L
    private var displayBowIssue: String? = null

    private var lastAngleIssue: String? = null
    private var angleIssueStartTime: Long = 0L
    private var angleIssueLastShownTime: Long = 0L
    private var displayAngleIssue: String? = null

    private var lastHandIssue: String? = null
    private var handIssueStartTime: Long = 0L
    private var handIssueLastShownTime: Long = 0L
    private var displayHandIssue: String? = null

    private var lastPoseIssue: String? = null
    private var poseIssueStartTime: Long = 0L
    private var poseIssueLastShownTime: Long = 0L
    private var displayPoseIssue: String? = null

    private val issueHoldDuration = 3000L // 3 seconds in milliseconds
    private var issueMinDisplayTime: Long = 1000 // must remain visible 1s after disappearing

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

    init {
        boxPaint.setColor(Color.GREEN)
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 8f

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
    fun getBowDims(width: Int, height: Int) {
        bowImageHeight = height
        bowImageWidth = width
    }

    override fun onDraw(canvas: Canvas) {
        if (results == null && handLandmarkerResult == null && handDetect.isEmpty() && poseDetect.isEmpty()) {
            // detection stopped, do not draw anything
            return
        }

        //overlayBitmap = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888)
        tempCanvas = Canvas(overlayBitmap!!)
        file_list.clear()

        var currentY = 120f
        val lineSpacing = 60f
        val padding = 16f

        val scaleX = 720f * 1.5f
        val scaleY = 1280f * 1.5f
        val scaleFactor = 1f //max(scaleX, scaleY)

        if (results?.classification != -2) {
            val stringBox = results?.string
            val bowBox = results?.bow
            Log.d("CheckDelCoords", "stringBox value: $stringBox")

            Log.d("CheckDelCoords", "bowBox value: $bowBox")
            if (stringBox != null) {
                stringBox.forEachIndexed { index, point ->
                    val x = point.x.toFloat()
                    val y = point.y.toFloat()
                    if (x > 1 || y > 1) {
                        Log.d("CheckDelStringBoxCoords", "Point[$index] has coords > 1: x=$x, y=$y - NEEDS DIVISION BY 640")
                    } else {
                        Log.d("CheckDelStringBoxCoords", "Point[$index] is normalized: x=$x, y=$y")
                    }
                }
            }

            if (bowBox != null) {
                bowBox.forEachIndexed { index, point ->
                    val x = point.x.toFloat()
                    val y = point.y.toFloat()
                    if (x > 1 || y > 1) {
                        Log.d("CheckDelBowBoxCoords", "Point[$index] has coords > 1: x=$x, y=$y - NEEDS DIVISION BY 640")
                    } else {
                        Log.d("CheckDelBowBoxCoords", "Point[$index] is normalized: x=$x, y=$y")
                    }
                }
            }
            // Determine if there's an issue
            val hasIssue = (results?.classification != null && results?.classification != 0) ||
                    (results?.angle != null && results?.angle == 1)

            // Choose box color based on classification
            val boxColor = if (hasIssue) Color.rgb(255, 140, 0) else Color.BLUE // Orange or Blue

            // Update boxPaint color dynamically
            boxPaint.color = boxColor
            if (stringBox != null) {
                canvas.drawLine(
                    (stringBox[0].x.toFloat() / 640f) * scaleX,
                    (stringBox[0].y.toFloat() / 640f) * scaleY,
                    (stringBox[1].x.toFloat() / 640f) * scaleX,
                    (stringBox[1].y.toFloat() / 640f) * scaleY,
                    boxPaint)
                canvas.drawLine(
                    (stringBox[1].x.toFloat() / 640f) * scaleX,
                    (stringBox[1].y.toFloat() / 640f) * scaleY,
                    (stringBox[2].x.toFloat() / 640f) * scaleX,
                    (stringBox[2].y.toFloat() / 640f) * scaleY,
                    boxPaint)
                canvas.drawLine(
                    (stringBox[2].x.toFloat() / 640f) * scaleX,
                    (stringBox[2].y.toFloat() / 640f) * scaleY,
                    (stringBox[3].x.toFloat() / 640f) * scaleX,
                    (stringBox[3].y.toFloat() / 640f) * scaleY,
                    boxPaint)
                canvas.drawLine(
                    (stringBox[3].x.toFloat() / 640f) * scaleX,
                    (stringBox[3].y.toFloat() / 640f) * scaleY,
                    (stringBox[0].x.toFloat() / 640f) * scaleX,
                    (stringBox[0].y.toFloat() / 640f) * scaleY,
                    boxPaint)
                /*tempCanvas?.let { c ->
                    c.drawLine(
                        (stringBox[0].x.toFloat() / 640f) * scaleX,
                        (stringBox[0].y.toFloat() / 640f) * scaleY,
                        (stringBox[1].x.toFloat() / 640f) * scaleX,
                        (stringBox[1].y.toFloat() / 640f) * scaleY,
                        boxPaint
                    )
                }
                tempCanvas?.let { c ->
                    c.drawLine(
                        (stringBox[1].x.toFloat() / 640f) * scaleX,
                        (stringBox[1].y.toFloat() / 640f) * scaleY,
                        (stringBox[2].x.toFloat() / 640f) * scaleX,
                        (stringBox[2].y.toFloat() / 640f) * scaleY,
                        boxPaint
                    )
                }
                tempCanvas?.let { c ->
                    c.drawLine(
                        (stringBox[2].x.toFloat() / 640f) * scaleX,
                        (stringBox[2].y.toFloat() / 640f) * scaleY,
                        (stringBox[3].x.toFloat() / 640f) * scaleX,
                        (stringBox[3].y.toFloat() / 640f) * scaleY,
                        boxPaint
                    )
                }
                tempCanvas?.let { c ->
                    c.drawLine(
                        (stringBox[3].x.toFloat() / 640f) * scaleX,
                        (stringBox[3].y.toFloat() / 640f) * scaleY,
                        (stringBox[0].x.toFloat() / 640f) * scaleX,
                        (stringBox[0].y.toFloat() / 640f) * scaleY,
                        boxPaint
                    )
                }*/
            }

            if (bowBox != null) {
                canvas.drawLine(
                    (bowBox[0].x.toFloat() / 640f) * scaleX,
                    (bowBox[0].y.toFloat() / 640f) * scaleY,
                    (bowBox[1].x.toFloat() / 640f) * scaleX,
                    (bowBox[1].y.toFloat() / 640f) * scaleY,
                    boxPaint)
                canvas.drawLine(
                    (bowBox[1].x.toFloat() / 640f) * scaleX,
                    (bowBox[1].y.toFloat() / 640f) * scaleY,
                    (bowBox[2].x.toFloat() / 640f) * scaleX,
                    (bowBox[2].y.toFloat() / 640f) * scaleY,
                    boxPaint)
                canvas.drawLine(
                    (bowBox[2].x.toFloat() / 640f) * scaleX,
                    (bowBox[2].y.toFloat() / 640f) * scaleY,
                    (bowBox[3].x.toFloat() / 640f) * scaleX,
                    (bowBox[3].y.toFloat() / 640f) * scaleY,
                    boxPaint)
                canvas.drawLine(
                    (bowBox[3].x.toFloat() / 640f) * scaleX,
                    (bowBox[3].y.toFloat() / 640f) * scaleY,
                    (bowBox[0].x.toFloat() / 640f) * scaleX,
                    (bowBox[0].y.toFloat() / 640f) * scaleY,
                    boxPaint)
                /*tempCanvas?.let { c ->
                    c.drawLine(
                        (bowBox[0].x.toFloat() / 640f) * scaleX,
                        (bowBox[0].y.toFloat() / 640f) * scaleY,
                        (bowBox[1].x.toFloat() / 640f) * scaleX,
                        (bowBox[1].y.toFloat() / 640f) * scaleY,
                        boxPaint
                    )
                }*/
                /*tempCanvas?.let { c ->
                    c.drawLine(
                        (bowBox[1].x.toFloat() / 640f) * scaleX,
                        (bowBox[1].y.toFloat() / 640f) * scaleY,
                        (bowBox[2].x.toFloat() / 640f) * scaleX,
                        (bowBox[2].y.toFloat() / 640f) * scaleY,
                        boxPaint
                    )
                }*/
                /*tempCanvas?.let { c ->
                    c.drawLine(
                        (bowBox[2].x.toFloat() / 640f) * scaleX,
                        (bowBox[2].y.toFloat() / 640f) * scaleY,
                        (bowBox[3].x.toFloat() / 640f) * scaleX,
                        (bowBox[3].y.toFloat() / 640f) * scaleY,
                        boxPaint
                    )
                }*/
                /*tempCanvas?.let { c ->
                    c.drawLine(
                        (bowBox[3].x.toFloat() / 640f) * scaleX,
                        (bowBox[3].y.toFloat() / 640f) * scaleY,
                        (bowBox[0].x.toFloat() / 640f) * scaleX,
                        (bowBox[0].y.toFloat() / 640f) * scaleY,
                        boxPaint
                    )
                }*/
            }
//            if (stringBox != null) {
//                canvas.drawLine(stringBox[0].x.toFloat() * scaleX,
//                    stringBox[0].y.toFloat() * scaleY,
//                    stringBox[1].x.toFloat() * scaleX,
//                    stringBox[1].y.toFloat() * scaleY,
//                    boxPaint)
//                canvas.drawLine(stringBox[1].x.toFloat() * scaleX,
//                    stringBox[1].y.toFloat() * scaleY,
//                    stringBox[2].x.toFloat()  * scaleX,
//                    stringBox[2].y.toFloat() * scaleY,
//                    boxPaint)
//                canvas.drawLine(stringBox[2].x.toFloat() * scaleX,
//                    stringBox[2].y.toFloat() * scaleY,
//                    stringBox[3].x.toFloat()  * scaleX,
//                    stringBox[3].y.toFloat() * scaleY,
//                    boxPaint)
//                canvas.drawLine(stringBox[3].x.toFloat() * scaleX,
//                    stringBox[3].y.toFloat() * scaleY,
//                    stringBox[0].x.toFloat()  * scaleX,
//                    stringBox[0].y.toFloat() * scaleY
//                    , boxPaint)
//            }
//            if (bowBox != null) {
//                canvas.drawLine(bowBox[0].x.toFloat() * scaleX,
//                    bowBox[0].y.toFloat() * scaleY,
//                    bowBox[1].x.toFloat() * scaleX,
//                    bowBox[1].y.toFloat() * scaleY,
//                    boxPaint)
//                canvas.drawLine(bowBox[1].x.toFloat() * scaleX,
//                    bowBox[1].y.toFloat() * scaleY,
//                    bowBox[2].x.toFloat()  * scaleX,
//                    bowBox[2].y.toFloat() * scaleY,
//                    boxPaint)
//                canvas.drawLine(bowBox[2].x.toFloat() * scaleX,
//                    bowBox[2].y.toFloat() * scaleY,
//                    bowBox[3].x.toFloat()  * scaleX,
//                    bowBox[3].y.toFloat() * scaleY,
//                    boxPaint)
//                canvas.drawLine(bowBox[3].x.toFloat() * scaleX,
//                    bowBox[3].y.toFloat() * scaleY,
//                    bowBox[0].x.toFloat()  * scaleX,
//                    bowBox[0].y.toFloat() * scaleY
//                    , boxPaint)
//            }

            // Classification labels - only show if not correct
            val classificationLabels = mapOf(
                0 to "",  // Correct - don't display
                1 to "Keep the bow in zone",    // Bow outside zone
                2 to "Lower the bow",    // Bow too high
                3 to "Lift the bow"    // Bow too low
            )

            val fileLabelsBow = mapOf(
                0 to "correct_bow",  // Correct
                1 to "bow_outside_zone",    // Bow outside zone
                2 to "bow_too_high",    // Bow too high
                3 to "bow_too_low"    // Bow too low
            )

            val angleLabels = mapOf(
                0 to "",  // Correct - don't display
                1 to "Adjust your bow angle"    // Incorrect bow angle
            )

            val fileLabelsAngle = mapOf(
                0 to "correct_angle",  // Correct
                1 to "incorrect_angle"    // Incorrect bow angle
            )

            // Fixed positions from top - below hand/pose classifications
            centerX = width / 2f


            // Draw classification message if there's an issue
            now = System.currentTimeMillis()
            if (results?.classification != null && results?.classification != 0) {
                bowMessage = classificationLabels[results?.classification] ?: ""
                file_list.add(fileLabelsBow[results?.classification] ?: "")
            } else if (results?.classification == 0) {
                file_list.add(fileLabelsBow[results?.classification] ?: "")
            }

            if (bowMessage.isNotEmpty()) {
                if (bowMessage == lastBowIssue) {
                    if (now - bowIssueStartTime >= issueHoldDuration) {
                        displayBowIssue = bowMessage
                        bowIssueLastShownTime = now
                    }
                } else {
                    lastBowIssue = bowMessage
                    bowIssueStartTime = now
                    displayBowIssue = null
                }
            } else {
                if (displayBowIssue != null && now - bowIssueLastShownTime >= issueMinDisplayTime) {
                    lastBowIssue = null
                    bowIssueStartTime = 0L
                    displayBowIssue = null
                    bowIssueLastShownTime = 0L
                }
            }

            // Draw angle message if there's an issue
            if (results?.angle != null && results?.angle == 1) {
                angleMessage = angleLabels[results?.angle] ?: ""
                file_list.add(fileLabelsAngle[results?.angle] ?: "")
            } else if (results?.angle == 0) {
                file_list.add(fileLabelsAngle[results?.angle] ?: "")
            }

            if (angleMessage.isNotEmpty()) {
                if (angleMessage == lastAngleIssue) {
                    if (now - angleIssueStartTime >= issueHoldDuration) {
                        displayAngleIssue = angleMessage
                        angleIssueLastShownTime = now

                    }
                } else {
                    lastAngleIssue = angleMessage
                    angleIssueStartTime = now
                    displayAngleIssue = null
                }
            } else {
                if (displayAngleIssue != null && now - angleIssueLastShownTime >= issueMinDisplayTime) {
                    lastAngleIssue = null
                    angleIssueStartTime = 0L
                    displayAngleIssue = null
                    angleIssueLastShownTime = 0L
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
                    /*tempCanvas?.let { c ->
                        c.drawLine(
                            landmarks.get(connection!!.start())
                                .x() * imageWidth * handsScaleFactor + xOffset,
                            landmarks.get(connection.start())
                                .y() * imageHeight * handsScaleFactor + yOffset,
                            landmarks.get(connection.end())
                                .x() * imageWidth * handsScaleFactor + xOffset,
                            landmarks.get(connection.end())
                                .y() * imageHeight * handsScaleFactor + yOffset,
                            linePaint
                        )
                    }*/
                }

                // Draw hand points
                for (normalizedLandmark in landmarks) {
                    canvas.drawPoint(
                        normalizedLandmark.x() * imageWidth * handsScaleFactor + xOffset,
                        normalizedLandmark.y() * imageHeight * handsScaleFactor + yOffset,
                        pointPaint
                    )
                    /*tempCanvas?.let { c ->
                        c.drawPoint(
                            normalizedLandmark.x() * imageWidth * handsScaleFactor + xOffset,
                            normalizedLandmark.y() * imageHeight * handsScaleFactor + yOffset,
                            pointPaint
                        )
                    }*/
                }
            }
        }

        if (displayBowIssue != null) {
            val textWidth = textPaint.measureText(displayBowIssue)
            val fm = textPaint.fontMetrics
            val textHeight = fm.bottom - fm.top

            // Rectangle coordinates
            val left = centerX - textWidth / 2 - padding
            val top = currentY + fm.top - padding
            val right = centerX + textWidth / 2 + padding
            val bottom = currentY + fm.bottom + padding

            canvas.drawRect(left, top, right, bottom, labelBackgroundPaint)

            canvas.drawText(displayBowIssue!!, centerX, currentY, textPaint)
            currentY += (fm.bottom - fm.top) + lineSpacing
        }

        if (displayAngleIssue != null) {
            val textWidth = textPaint.measureText(displayAngleIssue)
            val fm = textPaint.fontMetrics
            val textHeight = fm.bottom - fm.top

            // Rectangle coordinates
            val left = centerX - textWidth / 2 - padding
            val top = currentY + fm.top - padding
            val right = centerX + textWidth / 2 + padding
            val bottom = currentY + fm.bottom + padding

            canvas.drawRect(left, top, right, bottom, labelBackgroundPaint)

            canvas.drawText(displayAngleIssue!!, centerX, currentY, textPaint)
            currentY += (fm.bottom - fm.top) + lineSpacing
        }

        // Parse hand classification
        val handClassRegex = """Prediction: (\d+) \(Confidence: ([\d.]+)\)""".toRegex()
        val handMatch = handClassRegex.find(handDetect)
        val handClass = handMatch?.groupValues?.get(1)?.toIntOrNull() ?: -1

        // Parse pose classification
        val poseMatch = handClassRegex.find(poseDetect)
        val poseClass = poseMatch?.groupValues?.get(1)?.toIntOrNull() ?: -1

        val centerX = width / 2f

        // Draw hand message if there's an issue
        if (handClass in 1..2) {
            val currentHandIssue = when (handClass) {
                1 -> "Pronate your wrist more"    // Supination
                2 -> "Supinate your wrist more"    // Too much pronation
                else -> ""
            }
            val wristFileLabel = when (handClass) {
                1 -> "supination"    // Supination
                2 -> "too_much_pronation"    // Too much pronation
                else -> "good_pronation"
            }
            if (currentHandIssue.isNotEmpty()) {
                file_list.add(wristFileLabel)
                if (currentHandIssue == lastHandIssue) {
                    if (now - handIssueStartTime >= issueHoldDuration) {
                        displayHandIssue = currentHandIssue
                        handIssueLastShownTime = now

                    }
                } else {
                    lastHandIssue = currentHandIssue
                    handIssueStartTime = now
                    displayHandIssue = null
                }
            } else {
                if (displayHandIssue != null && now - handIssueLastShownTime >= issueMinDisplayTime) {
                    lastHandIssue = null
                    handIssueStartTime = 0L
                    displayHandIssue = null
                    handIssueLastShownTime = 0L
                }
            }

            if (displayHandIssue != null) {
                val textWidth = textPaint.measureText(displayHandIssue)
                val fm = textPaint.fontMetrics
                val textHeight = fm.bottom - fm.top

                // Rectangle coordinates
                val left = centerX - textWidth / 2 - padding
                val top = currentY + fm.top - padding
                val right = centerX + textWidth / 2 + padding
                val bottom = currentY + fm.bottom + padding

                canvas.drawRect(left, top, right, bottom, labelBackgroundPaint)

                canvas.drawText(displayHandIssue!!, centerX, currentY, textPaint)

                currentY += (fm.bottom - fm.top) + lineSpacing
            }
        } else {
            file_list.add("good_pronation")
        }

        // Draw pose message if there's an issue
        if (poseClass in 1..2) {
            val currentPoseIssue = when (poseClass) {
                1 -> "Raise your elbow a bit"    // Low elbow
                2 -> "Lower your elbow a bit"    // Elbow too high
                else -> ""
            }
            val pose_file_name = when (poseClass) {
                1 -> "low_elbow"    // Low elbow
                2 -> "high_elbow"    // Elbow too high
                else -> "good_elbow"
            }
            if (currentPoseIssue.isNotEmpty()) {
                file_list.add(pose_file_name)
                if (currentPoseIssue == lastPoseIssue) {
                    if (now - poseIssueStartTime >= issueHoldDuration) {
                        displayPoseIssue = currentPoseIssue
                        poseIssueLastShownTime = now

                    }
                } else {
                    lastPoseIssue = currentPoseIssue
                    poseIssueStartTime = now
                    displayPoseIssue = null
                }
            } else {
                if (displayPoseIssue != null && now - poseIssueLastShownTime >= issueMinDisplayTime) {
                    lastPoseIssue = null
                    poseIssueStartTime = 0L
                    displayPoseIssue = null
                    poseIssueLastShownTime = 0L
                }
            }

            if (displayPoseIssue != null) {

                val textWidth = textPaint.measureText(displayPoseIssue)
                val fm = textPaint.fontMetrics
                val textHeight = fm.bottom - fm.top

                // Rectangle coordinates
                val left = centerX - textWidth / 2 - padding
                val top = currentY + fm.top - padding
                val right = centerX + textWidth / 2 + padding
                val bottom = currentY + fm.bottom + padding

                canvas.drawRect(left, top, right, bottom, labelBackgroundPaint)

                canvas.drawText(displayPoseIssue!!, centerX, currentY, textPaint)
            }
        } else {
            file_list.add("good_elbow")
        }

        file_list.forEach { fName ->
            saveImageForSession(fName)
        }
    }

    fun updateResults(
        results: Detector.returnBow?,
        hands: HandLandmarkerResult?,
        pose: PoseLandmarkerResult?,
        handDetection: String,
        poseDetection: String,
    ) {
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
        android.util.Log.d("HAND STUFF", imageWidth.toString() + " " + imageHeight.toString() + " " + handsScaleFactor.toString())
    }

    fun clear() {
        results = null
        handLandmarkerResult = null
        poseLandmarkerResult = null
        handDetect = ""
        poseDetect = ""
        postInvalidate() // remove drawings
    }


    fun saveImageForSession(filename: String) {
        val userId = Profile.getUserId()
        val timestamp = Profile.getTimeStamp()

        val baseDir = File(
            android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOCUMENTS),
            "sessions/$userId/$timestamp"
        )

        // Create the directory if it doesn't exist
        if (!baseDir.exists()) {
            val created = baseDir.mkdirs()
            if (!created) {
                Log.e("FileSave", "Failed to create directory: ${baseDir.absolutePath}")
            }
        }

        // Now create a file inside that directory
        val outFile = File(baseDir, filename)

        if (outFile.exists()) {
            Log.d("SaveOverlay", "File already exists, skipping save: ${outFile.absolutePath}")
            return
        }

        // Write to the file
        try {
            FileOutputStream(outFile).use { fos ->
                overlayBitmap?.compress(Bitmap.CompressFormat.PNG, 100, fos)
            }
            Log.d("SaveOverlay", "Saved overlay to: ${outFile.absolutePath}")
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }



    fun setBitmapFrame(frame: Bitmap) {
        overlayBitmap = frame.copy(Bitmap.Config.ARGB_8888, true)
    }



}