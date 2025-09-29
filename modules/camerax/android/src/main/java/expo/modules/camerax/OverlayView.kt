package expo.modules.camerax

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat


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
        anglePaint.textSize =48f
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
                println("DRAWING BOW BOX")
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


    }

    fun updateResults(results: Detector.returnBow) {
        this.results = results
        invalidate()

    }


}