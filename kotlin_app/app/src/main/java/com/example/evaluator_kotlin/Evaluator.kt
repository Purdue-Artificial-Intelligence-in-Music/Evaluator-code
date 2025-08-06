package com.example.evaluator_kotlin

import android.content.Context
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


class Evaluator {
    val initializeTask: Task<Void> by lazy {
        Log.d("InitDebug", "TfLite.initialize() is being called")
        TfLite.initialize(MainActivity.applicationContext()) }
    val modelReadyLatch = CountDownLatch(1)
    private lateinit var model: InterpreterApi

    fun createInterpreter(context: Context) {
        initializeTask.addOnSuccessListener {
            Log.d("InitDebug", "SuccessListener is being called")

            val interpreterOption =
                InterpreterApi.Options()
                    .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            model = InterpreterApi.create(
                FileUtil.loadMappedFile(
                    MainActivity.applicationContext(),
                    "nano_best_float32.tflite"
                ),
                interpreterOption
            )
            modelReadyLatch.countDown()
        }
            .addOnFailureListener { e ->
                Log.e("Interpreter", "Cannot initialize interpreter", e)
            }
    }

    /* takes in original image, a set size (default 640x640 p)
     * Rescales the image while maintaining aspect ratio, then pads to be square
     * Returns the padding and new image
     */
    fun letterbox(img: Mat, newShape: Size = Size(640.0, 640.0)): Pair<Mat, Pair<Double, Double>> {
        val shape = Size(img.width().toDouble(), img.height().toDouble())
        val r = min(newShape.width / shape.width, newShape.height / shape.height)
        val newUnpad = Size(round(shape.width * r), round(shape.height * r))
        val dw = (newShape.width - newUnpad.width) / 2
        val dh = (newShape.height - newUnpad.height) / 2

        val resized = Mat()
        Imgproc.resize(img, resized, newUnpad)
        val top = round(dh - 0.1).toInt()
        val bottom = round(dh + 0.1).toInt()
        val left = round(dw - 0.1).toInt()
        val right = round(dw + 0.1).toInt()
        val padded = Mat()
        Core.copyMakeBorder(resized, padded, top, bottom, left, right, Core.BORDER_CONSTANT, Scalar(114.0, 114.0, 114.0))
        return Pair(padded, Pair(top / padded.height().toDouble(), left / padded.width().toDouble()))
    }

    /*
     * preprocesses the given image. Runs letterbox (rescales/packs).
     * Converts the image to RGB
     * Converts the image to float32 and normalizes it
     * Returns the preprocessed image to be ran on the model
     */
    fun preprocess(img: Mat, newShape: Size = Size(640.0, 640.0)):
            Pair<Mat, Pair<Double, Double>> {
        val (letterboxed, pad) = letterbox(img, newShape)
        val resizedImg = letterboxed
        val rgb = Mat()
        Imgproc.cvtColor(letterboxed, rgb, Imgproc.COLOR_BGR2RGB)
        val floatImg = Mat()
        rgb.convertTo(floatImg, CvType.CV_32FC3, 1.0 / 255.0)
        return Pair(floatImg, pad)
    }

    /*
     * postprocessing of the output
     * requires the original image, the results, resized shape, and padding (from letterbox)
     *
     */
    fun postprocess(
        origImg: Mat,
        outputs: Array<FloatArray>, // shape: [N, 7] (cx, cy, h, w, conf, cls, angle)
        resizedShape: Size,
        pad: Pair<Double, Double>
    ): List<List<Float>> {
        val results = mutableListOf<List<Float>>()
        val targetH = origImg.height()
        val targetW = origImg.width()
        val r = min(resizedShape.height / targetH, resizedShape.width / targetW)
        val targetScale = max(targetH, targetW).toFloat()

        for (out in outputs) {
            // Adjust coordinates
            val cx = targetScale * (out[0] - pad.second)
            val cy = targetScale * (out[1] - pad.first)
            val w = targetScale * out[3]
            val h = targetScale * out[2]
            val conf = out[4]
            val cls = out[5]
            val angleRad = out[6]

            // Convert to 4 corner points (rotated rectangle)
            val points = rotatedRectToPoints(cx.toFloat(), cy.toFloat(), w, h, angleRad)
            // Add confidence to result
            val result = points.map { listOf(it.first, it.second) }.flatten().toMutableList()
            result.add(conf)
            result.add(cls)
            results.add(result)
        }
        return results
    }

    // Helper for rotated rectangle to points
    fun rotatedRectToPoints(cx: Float, cy: Float, w: Float, h: Float, angleRad: Float): List<Pair<Float, Float>> {
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
            Pair(xRot, yRot)
        }
    }

    fun runModel(frame: Mat):  List<List<Float>> {
        // Create input and output buffers
        model.allocateTensors()

        // Prepare input buffer
        val inputBuffers = model.getInputTensor(0)
        val inputShape = inputBuffers.shape()
        val inputDataType = inputBuffers.dataType()
        val targetHeight = inputShape[1]
        val targetWidth = inputShape[2]

        val preprocessed = preprocess(frame, Size(targetHeight.toDouble(),
            targetWidth.toDouble()))

        val preprocessedImg = preprocessed.first

        val input = Array(targetHeight) { FloatArray(targetWidth) } // Example for classification output

        val inputArray = Array(1) { Array(targetHeight) { Array(targetWidth) { FloatArray(3) } } }

        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val pixel = preprocessedImg.get(y, x) // [R, G, B]
                inputArray[0][y][x][0] = (pixel[0]).toFloat() // R
                inputArray[0][y][x][1] = (pixel[1]).toFloat() // G
                inputArray[0][y][x][2] = (pixel[2]).toFloat() // B
            }
        }

        // Run inference
        val outputBuffers = model.getOutputTensor(0)
        val outputShape = outputBuffers.shape()
        val outputDataType = outputBuffers.dataType()

        val outputArray = Array(outputShape[0]) { Array(outputShape[1]) {FloatArray(outputShape[2])} } // (1, 300, 7)

        model.run(inputArray, outputArray)
        val outputs = outputArray[0] // shape: [N, 7]
        // Convert output to Array<FloatArray>
        val numDetections = outputs.size / 7
        val output = Array(numDetections) { i ->
            outputs.sliceArray(i * 7 until (i + 1) * 7)
        }

        // Postprocess
        //        origImg: Mat,
        //        outputs: Array<FloatArray>, // shape: [N, 7] (cx, cy, h, w, conf, cls, angle)
        //        resizedShape: Size,
        //        pad: Pair<Double, Double>
        val results = postprocess(origImg = frame, outputs = outputs,
            resizedShape = Size(targetHeight.toDouble(), targetWidth.toDouble()), pad = preprocessed.second)
        //println("Detections: $results")
        println("Detections: ${convertYolo(results)}")

        return results
    }



    fun drawDetections(img: Mat, box: List<Point>, score: Float, classId: Int) {
        // Define color palette and class labels
        val colorPalette = mapOf(
            1 to Scalar(255.0, 255.0, 100.0), // BGR format
            0 to Scalar(100.0, 255.0, 255.0)
        )
        val classes = mapOf(
            1 to "bow",
            0 to "string"
        )

        val color = colorPalette[classId] ?: Scalar(255.0, 255.0, 255.0) // fallback color

        // Draw bounding box if the box contains 4 points
        if (box.size == 4) {
            val pointsArray = MatOfPoint(*box.map { Point(it.x, it.y) }.toTypedArray())
            Imgproc.polylines(img, listOf(pointsArray), true, color, 2)
        }
    }

    data class BoxResults(
        var classification: Int?,
        var box: MutableList<MutableList<Int>>? ,
        var angle: Int?
    )
    data class YoloResults(
        var bowResults: MutableList<Point>? ,
        var stringResults: MutableList<Point>?
    )


    fun convertYolo(results: List<List<Float>>): YoloResults {
        //Results are always this format: [[x1, y1, x2, y2, x3, y3, x4, y4, conf, cls], ...]
        //First two will be highest conf bow and string boxes, will be two of same class if other is not detected
        val yoloResults = YoloResults(
            bowResults = null,
            stringResults = null
        )
        var coordList = mutableListOf<Point>()
        for (i in 0 until 4) {
            coordList.add(
                Point(
                    results[0][2 * i].toDouble(),
                    results[0][2 * i + 1].toDouble()
                )
            )
        }
        if (results[0][9] == 1.0f) {
            yoloResults.stringResults = coordList
        } else {
            yoloResults.bowResults = coordList
        }
        if (results.size > 1 && results[1][9] != results[0][9]) {
            coordList = mutableListOf<Point>()
            for (i in 0 until 4) {
                coordList.add(
                    Point(
                        results[1][2 * i].toDouble(),
                        results[1][2 * i + 1].toDouble()
                    )
                )
            }
            if (results[1][9] == 1.0f) {
                yoloResults.stringResults = coordList
            } else {
                yoloResults.bowResults = coordList
            }

        }
        return yoloResults
    }

    /*
    fun convert_to_yolo_results(results: List<List<Float>>): YoloResults {
        //Data class for storing bow box
        val bowResults = BoxResults(
            classification = 0,
            box = null,
            angle = 0
        )

        //Data class for storing string box
        val strResults = BoxResults(
            classification = 0,
            box = null,
            angle = 0
        )

        //Data class for storing bow and string results
        val yoloResults = YoloResults(
            bowResults = null,
            strResults = null
        )


        var bowConf = 0.0f
        var stringConf = 0.0f
        var bowIndex = -1
        var stringIndex = -1
        //0 is bow, 1 is string for cls
        for (i in results.indices) {
            if (results[i][9] == 1.0f) {
                if (results[i][8] > stringConf) {
                    stringConf = results[i][8]
                    stringIndex = i
                }
            }
            if (results[i][9] == 0.0f) {
                if (results[i][8] > bowConf) {
                    bowConf = results[i][8]
                    bowIndex = i
                }
            }
        }
        if (bowIndex != -1) {
            var bow_return = mutableListOf<MutableList<Int>>()
            for (i in 0 until 4) {
                bow_return.add(mutableListOf(results[bowIndex][2*i].toInt(),
                    results[bowIndex][2*i +1].toInt()))
            }
            yoloResults.bowResults = bow_return
        }
        if (stringIndex != -1) {
            var string_return = mutableListOf<MutableList<Int>>()
            for (i in 0 until 4) {
                string_return.add(mutableListOf(results[stringIndex][2*i].toInt(),
                    results[stringIndex][2*i +1].toInt()))
            }
            yoloResults.strResults =  string_return
        }
        return yoloResults
    }
    */
}
