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


    init {
        boxPaint.setColor(Color.GREEN)
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 8f

        textPaint.setColor(Color.GREEN)
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 48f
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

            canvas.drawText(
                "Classification: ${results?.classification}",
                50f,
                100f,
                textPaint
            )

        }


    }

    fun updateResults(results: Detector.returnBow) {
        this.results = results
        invalidate()

    }

}