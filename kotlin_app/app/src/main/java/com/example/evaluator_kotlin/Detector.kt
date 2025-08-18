package com.example.evaluator_kotlin
import android.content.Context
import android.gesture.OrientedBoundingBox
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tflite.java.TfLite
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
//import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.FileUtil
import java.util.concurrent.CountDownLatch
import kotlin.math.*
import org.opencv.core.Point
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import android.os.SystemClock
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import org.tensorflow.lite.Interpreter

class Detector {

    private var interpreter: Interpreter
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    val modelReadyLatch = CountDownLatch(1)

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()
    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.2F
    }

    init {
        val options = Interpreter.Options().apply{
            this.setNumThreads(8)
        }
        /*
        interpreter = Interpreter.create(
            FileUtil.loadMappedFile(
                MainActivity.applicationContext(),
                "nano_best_float32.tflite"
            ),
            options
        )
         */
        val model = FileUtil.loadMappedFile(MainActivity.applicationContext(), "nano_float16.tflite")
        interpreter = Interpreter(model, options)

        modelReadyLatch.countDown()

        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape = interpreter.getOutputTensor(0)?.shape()

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]

            // If in case input shape is in format of [1, 3, ..., ...]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        if (outputShape != null) {
            numElements = outputShape[1]
            numChannel = outputShape[2]
        }

    }
    fun close() {
        interpreter.close()
    }

    data class YoloResults(
        var bowResults: MutableList<Point>?,
        var stringResults: MutableList<Point>?
    )


    fun detect(frame: Bitmap): List<PointF>{
        if (tensorWidth == 0
            || tensorHeight == 0
            || numChannel == 0
            || numElements == 0) return emptyList()

        println(tensorWidth)
        println(tensorHeight)
        var inferenceTime = SystemClock.uptimeMillis()



        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)

        interpreter.run(imageBuffer, output.buffer)

        println(output.floatArray.contentToString())

        val bestBoxes = newBestBox(output.floatArray)

        val newBoxes = mutableListOf<PointF>()
        for (box in bestBoxes) {
            val points = rotatedRectToPoints(box.x, box.y, box.width, box.height, box.angle)
            newBoxes.addAll(points)
        }
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        println("TRUE INFERENCE TIME: $inferenceTime")
        return newBoxes
    }

    fun drawPointsOnBitmap(
        sourceBitmap: Bitmap,
        points: List<PointF>,
        color: Int = Color.RED,
        radius: Float = 10f
    ): Bitmap {
        val resultBitmap = sourceBitmap.copy(Bitmap.Config.ARGB_8888, true)

        val canvas = Canvas(resultBitmap)

        val paint = Paint().apply {
            this.color = color
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        for (point in points) {
            canvas.drawCircle(point.x * sourceBitmap.width, point.y * sourceBitmap.height, radius, paint)
        }

        return resultBitmap
    }



    private fun rotatedRectToPoints(cx: Float, cy: Float, w: Float, h: Float, angleRad: Float): List<PointF> {
        val halfW = w / 2
        val halfH = h / 2
        val cosA = cos(angleRad - Math.PI.toFloat() / 2)
        val sinA = sin(angleRad - Math.PI.toFloat() / 2)
        val corners = listOf(
            Pair(-halfW, -halfH),
            Pair(halfW, -halfH),
            Pair(halfW, halfH),
            Pair(-halfW, halfH)
        )
        return corners.map { (x, y) ->
            val xRot = x * cosA - y * sinA + cx
            val yRot = x * sinA + y * cosA + cy
            PointF(xRot, yRot)
        }
    }

    private fun newBestBox(array : FloatArray) : List<OrientedBoundingBox> {
        val boundingBoxes = mutableListOf<OrientedBoundingBox>()

        for (r in 0 until numElements) {
            val cnf = array[r * numChannel + 4]
            if (cnf > CONFIDENCE_THRESHOLD) {
                val x = array[r * numChannel]
                val y = array[r * numChannel + 1]
                val h = array[r * numChannel + 2]
                val w = array[r * numChannel + 3]
                val cls = array[r * numChannel + 5].toInt()
                val angle = array[r * numChannel + 6]
                boundingBoxes.add(
                    OrientedBoundingBox(
                        x = x, y = y, height = h, width = w,
                        conf = cnf, cls = cls, angle = angle, clsName = "p"
                    )
                )
            }
        }
        return boundingBoxes
    }


    data class OrientedBoundingBox(
        val x: Float,
        val y: Float,
        val height: Float,
        val width: Float,
        val conf: Float,
        val cls: Int,
        val angle: Float,
        val clsName: String
    )

}