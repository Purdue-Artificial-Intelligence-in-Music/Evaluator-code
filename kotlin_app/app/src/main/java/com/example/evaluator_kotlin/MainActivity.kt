package com.example.evaluator_kotlin

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import com.example.evaluator_kotlin.ui.theme.EvaluatorKotlinTheme
import com.google.android.gms.tasks.Tasks
import java.io.File
import androidx.compose.ui.graphics.asImageBitmap
import org.opencv.imgproc.Imgproc
import androidx.core.graphics.createBitmap
import java.io.IOException
import java.io.InputStream

private var detector = Detector()


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        System.loadLibrary("opencv_java4")
        setContent {
            EvaluatorKotlinTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    MainScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
        instance = this
    }

    companion object {
        var instance: MainActivity? = null

        fun applicationContext() : Context {
            return instance!!.applicationContext
        }
    }
}
@Composable
fun MainScreen(modifier: Modifier = Modifier) {
    val resultImage = remember { mutableStateOf<Bitmap?>(null) }

    Column(modifier = modifier) {
        Text(text = "Object Detection")
        Button(onClick = {
            runEvaluation { bitmap ->
                resultImage.value = bitmap
            }
        }) {
            Text("Run Evaluation")
        }

        resultImage.value?.let { bitmap ->
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Detection Results"
            )
        }
    }
}

fun runEvaluation(onComplete: (Bitmap) -> Unit) {
    //val evaluator = Evaluator()

    val bitmapImage = getBitmapFromAssets(MainActivity.applicationContext(), "Sample Input.png")

    //evaluator.createInterpreter(MainActivity.applicationContext())

    try {
        //Tasks.await(detector.initializeTask)
        Log.d("Interpreter", "TfLite.initialize() completed successfully")

        bitmapImage?.let {
            detector.modelReadyLatch.await()
            val results = detector.detect(it)
            val newBitmap = detector.drawPointsOnBitmap(it, results)


            println("CLASSIFICATION: " + detector.classify(results))


            // Pass the bitmap back to the UI thread
            MainActivity.instance?.runOnUiThread {
                onComplete(newBitmap)
            }
        }
    } catch (e: Exception) {
        Log.e("Interpreter", "Error during evaluation", e)
    }

}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Column(modifier = modifier) {
        Text(text = "Hello $name!")
        Button(onClick = onClick) {
            Text("Run Evaluation")
        }
    }
}

fun getBitmapFromAssets(context: Context, fileName: String): Bitmap? {
    val assetManager = context.assets
    val inputStream: InputStream?
    try {
        inputStream = assetManager.open(fileName)
        return BitmapFactory.decodeStream(inputStream)
    } catch (e: IOException) {
        e.printStackTrace()
    }
    return null
}
/*
fun readImageFromAssets(context: Context, filename: String): Mat? {
    try {
        // Open the asset file
        val assetManager = context.assets
        val inputStream = assetManager.open(filename)

        // Save to a temporary file (OpenCV needs a file path)
        val tempFile = File.createTempFile("Sample Input", ".png", context.cacheDir)
        tempFile.outputStream().use { output ->
            inputStream.copyTo(output)
        }

        // Load with OpenCV
        return Imgcodecs.imread(tempFile.absolutePath)
    } catch (e: Exception) {
        Log.e("OpenCV", "Failed to read asset: ${e.message}")
        return null
    }
}

fun readImageFromPath(imagePath: String): Mat? {
    // Ensure OpenCV is loaded and initialized before using its functions
    // This typically happens in your application's initialization logic
    // System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

    val image: Mat? = readImageFromAssets(MainActivity.applicationContext(), imagePath)

    // Check if the image was loaded successfully
    if (image == null || image.empty()) {
        println("Error: Could not read image from path: $imagePath")
        return null
    }

    return image
}



fun main() {
    val evaluator = Evaluator()
    val img = readImageFromPath("Sample Input.png")
    var results: List<List<Float>>? = null
    evaluator.createInterpreter(MainActivity.applicationContext())
    Thread {
        try {
            Tasks.await(evaluator.initializeTask)
            Log.d("Interpreter", "TfLite.initialize() completed successfully")

            // Now run inference
            img?.let {
                evaluator.modelReadyLatch.await()
                results = evaluator.runModel(it)
            }
        } catch (e: Exception) {
            Log.e("Interpreter", "TfLite.initialize() threw an exception", e)
        }

    }.start()

    if (results != null) {
        val yoloResults = evaluator.convertYolo(results!!)
        if (yoloResults.stringResults != null) {
            evaluator.drawDetections(img!!, yoloResults.stringResults!!, 1.0f, 1)
        }
        if (yoloResults.bowResults != null) {
            evaluator.drawDetections(img!!, yoloResults.bowResults!!, 1.0f, 0)
        }

    }

    /*
    if (results != null) {
        for (result in results!!) {
            val points: List<Point> = result.slice(0..7).chunked(2).map { (x, y) -> Point(x.toDouble(), y.toDouble()) }
            evaluator.drawDetections(img!!, points, result[8], result[9].toInt())
        }
    }
    */


}

 */
