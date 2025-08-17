package com.example.evaluator_kotlin
import android.content.Context
import android.gesture.OrientedBoundingBox
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tflite.java.TfLite
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.InterpreterApi
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

class Detector {

    private var interpreter: InterpreterApi
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
        val options = InterpreterApi.Options().apply{
            this.setNumThreads(8)
        }

        interpreter = InterpreterApi.create(
            FileUtil.loadMappedFile(
                MainActivity.applicationContext(),
                "nano_best_float32.tflite"
            ),
            options
        )
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


    fun detect(frame: Bitmap): List<Point>{
        if (tensorWidth == 0
            || tensorHeight == 0
            || numChannel == 0
            || numElements == 0) return emptyList()

        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)
        interpreter.run(imageBuffer, output.buffer)

        val bestBoxes = newBestBox(output.floatArray)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        println("TRUE INFERENCE TIME: $inferenceTime")
        val newBoxes = mutableListOf<Point>()
        for (box in bestBoxes) {
            val points = rotatedRectToPoints(box.x, box.y, box.width, box.height, box.angle)
            newBoxes.addAll(points)
        }
        return newBoxes
    }

    fun drawDetections(img: Mat, box: List<Point>, score: Float, classId: Int) {
        // Define color palette and class labels
        val colorPalette = mapOf(
            1 to Scalar(255.0, 255.0, 100.0), // BGR format
            0 to Scalar(100.0, 255.0, 255.0)
        )
        // Define class ID to label mapping
        val classes = mapOf(
            1 to "bow",
            0 to "string"
        )

        val color = colorPalette[classId] ?: Scalar(255.0, 255.0, 255.0) // fallback color (white)

        // Draw bounding box if the box contains 4 points
        if (box.size == 4) {
            // converts list of Points to OpenCV MatOfPoint
            val pointsArray = MatOfPoint(*box.map { Point(it.x, it.y) }.toTypedArray())
            Imgproc.polylines(img, listOf(pointsArray), true, color, 2)
        }
    }

    private fun rotatedRectToPoints(cx: Float, cy: Float, w: Float, h: Float, angleRad: Float): List<Point> {
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
            Point(xRot.toDouble(), yRot.toDouble())
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
                val clsName = labels[cls]
                boundingBoxes.add(
                    OrientedBoundingBox(
                        x = x, y = y, height = h, width = w,
                        conf = cnf, cls = cls, angle = angle, clsName = clsName
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