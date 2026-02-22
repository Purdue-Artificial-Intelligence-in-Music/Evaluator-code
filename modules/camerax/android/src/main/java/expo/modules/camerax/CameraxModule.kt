package expo.modules.camerax

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import android.os.Environment
import android.util.Log
import org.json.JSONObject
import java.io.File

class CameraxModule : Module() {
    override fun definition() = ModuleDefinition {
        Name("Camerax")

        View(CameraxView::class) {
            Prop("userId") { view: CameraxView, userId: String ->
                view.setUserId(userId)
            }

            Prop("detectionEnabled") { view: CameraxView, enabled: Boolean ->
                view.setDetectionEnabled(enabled)
            }

            Prop("lensType") { view: CameraxView, lensType: String ->
                view.setLensType(lensType)
            }

            Prop("flip") { view: CameraxView, flip: Boolean ->
                view.setFlip(flip)
            }

            Prop("cameraActive") { view: CameraxView, active: Boolean ->
                view.setCameraActive(active)
            }

            Prop("maxBowAngle") { view: CameraxView, angle: Int ->
                view.setMaxBowAngle(angle)
            }

            Prop("skipCalibration") { view: CameraxView, skip: Boolean ->
                view.skipCalibration(skip)
            }

            Events("onDetectionResult", "onNoDetection", "onSessionEnd", "onCalibrated")
        }

        AsyncFunction("getRecentSessions") { userId: String, count: Int ->
            getRecentSessionsImpl(userId, count)
        }

        AsyncFunction("getSessionImages") { userId: String, timestamp: String ->
            getSessionImagesImpl(userId, timestamp)
        }

    }

    private fun getRecentSessionsImpl(userId: String, count: Int): List<Map<String, Any>> {
        try {
            val sessionsDir = File(
                Environment.getExternalStoragePublicDirectory(
                    Environment.DIRECTORY_DOCUMENTS
                ), "sessions"
            )

            Log.d("SessionHistory", "Sessions directory path: ${sessionsDir.absolutePath}")

            if (!sessionsDir.exists()) {
                Log.e("SessionHistory", "Sessions directory does NOT exist!")
                return emptyList()
            }

            val summaryFiles = sessionsDir.listFiles { file ->
                val matches = file.name.startsWith("session_${userId}_") &&
                        file.name.endsWith("_summary.json")
                if (matches) {
                    Log.d("SessionHistory", "Found summary file: ${file.name}")
                }
                matches
            }?.toList() ?: emptyList()

            // sort files by time
            val sortedFiles = summaryFiles.sortedByDescending { it.lastModified() }

            // get latest N files
            val recentFiles = sortedFiles.take(count)

            val sessions = mutableListOf<Map<String, Any>>()
            recentFiles.forEachIndexed { index, file ->
                try {
                    val jsonContent = file.readText()
                    val sessionData = parseSummaryFile(jsonContent, file.name)
                    if (sessionData != null) {
                        Log.d("SessionHistory", "Successfully parsed summary file ${index + 1}")
                        sessions.add(sessionData)
                    } else {
                        Log.w("SessionHistory", "Failed to parse summary file ${index + 1}")
                    }
                } catch (e: Exception) {
                    Log.e("SessionHistory", "Exception parsing summary file ${index + 1}: ${e.message}", e)
                    e.printStackTrace()
                }
            }

            return sessions
        } catch (e: Exception) {
            Log.e("SessionHistory", "Fatal error in getRecentSessionsImpl", e)
            e.printStackTrace()
            return emptyList()
        }
    }

    private fun getSessionImagesImpl(userId: String, timestamp: String): List<String> {
        val images = mutableListOf<String>()
        try {
            // convert timestamp
            // eg. from "2025-12-09 01:13:48" to "20251209_011348"
            val normalizedTimestamp = normalizeTimestampToFolderName(timestamp)

            Log.d("SessionImages", "Original timestamp: $timestamp")
            Log.d("SessionImages", "Normalized timestamp: $normalizedTimestamp")

            // Build the session directory path
            val sessionDir = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
                "sessions/$userId/$normalizedTimestamp"
            )

            Log.d("SessionImages", "Looking for images in: ${sessionDir.absolutePath}")

            if (!sessionDir.exists() || !sessionDir.isDirectory) {
                Log.w("SessionImages", "Session directory does not exist: ${sessionDir.absolutePath}")
                return images
            }

            // List all image files (assuming PNG or JPG)
            val imageFiles = sessionDir.listFiles { file ->
                val name = file.name.lowercase()
                name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg")
            } ?: emptyArray()

            imageFiles.sortedBy { it.name }.forEach { file ->
                images.add(file.absolutePath)
                Log.d("SessionImages", "Found image: ${file.absolutePath}")
            }

        } catch (e: Exception) {
            Log.e("SessionImages", "Error fetching images for $userId/$timestamp: ${e.message}", e)
        }

        return images
    }

    private fun normalizeTimestampToFolderName(timestamp: String): String {
        try {
            // input format: "2025-12-09 01:13:48"
            // output format: "20251209_011348"
            val cleaned = timestamp.replace("-", "").replace(" ", "_").replace(":", "")

            Log.d("SessionImages", "Timestamp conversion: $timestamp -> $cleaned")
            return cleaned
        } catch (e: Exception) {
            Log.e("SessionImages", "Error normalizing timestamp: ${e.message}", e)
            return timestamp
        }
    }


    private fun parseSummaryFile(jsonContent: String, fileName: String): Map<String, Any>? {
        try {
            val jsonObject = JSONObject(jsonContent)
            val userId = jsonObject.optString("user_id", "unknown")
            val timestamp = jsonObject.optString("timestamp", "")

            // Breakdown data
            val result = mutableMapOf<String, Any>()
            result["userId"] = userId
            result["timestamp"] = timestamp

            // Duration fields
            val durationSeconds = jsonObject.optLong("durationSeconds", 0L)
            val durationFormatted = jsonObject.optString("durationFormatted", "0s")
            result["durationSeconds"] = durationSeconds
            result["durationFormatted"] = durationFormatted
            Log.d("SessionHistory", "Duration: ${durationFormatted} (${durationSeconds}s)")

            // Height Breakdown
            val heightBreakdown = parseBreakdown(jsonObject.optJSONObject("heightBreakdown"))
            if (heightBreakdown.isNotEmpty()) {
                result["heightBreakdown"] = heightBreakdown
                Log.d("SessionHistory", "Height breakdown: ${heightBreakdown.size} items")
            }

            // Angle Breakdown
            val angleBreakdown = parseBreakdown(jsonObject.optJSONObject("angleBreakdown"))
            if (angleBreakdown.isNotEmpty()) {
                result["angleBreakdown"] = angleBreakdown
            }

            // Hand Presence Breakdown
            val handPresenceBreakdown = parseBreakdown(jsonObject.optJSONObject("handPresenceBreakdown"))
            if (handPresenceBreakdown.isNotEmpty()) {
                result["handPresenceBreakdown"] = handPresenceBreakdown
            }

            // Hand Posture Breakdown
            val handPostureBreakdown = parseBreakdown(jsonObject.optJSONObject("handPostureBreakdown"))
            if (handPostureBreakdown.isNotEmpty()) {
                result["handPostureBreakdown"] = handPostureBreakdown
            }

            // Pose Presence Breakdown
            val posePresenceBreakdown = parseBreakdown(jsonObject.optJSONObject("posePresenceBreakdown"))
            if (posePresenceBreakdown.isNotEmpty()) {
                result["posePresenceBreakdown"] = posePresenceBreakdown
            }

            // Elbow Posture Breakdown
            val elbowPostureBreakdown = parseBreakdown(jsonObject.optJSONObject("elbowPostureBreakdown"))
            if (elbowPostureBreakdown.isNotEmpty()) {
                result["elbowPostureBreakdown"] = elbowPostureBreakdown
            }

            Log.d("SessionHistory", "Successfully created result map")
            return result
        } catch (e: Exception) {
            Log.e("SessionHistory", "Error parsing summary JSON for $fileName: ${e.message}", e)
            e.printStackTrace()
            return null
        }
    }

    private fun parseBreakdown(jsonObject: JSONObject?): Map<String, Double> {
        if (jsonObject == null) return emptyMap()

        val result = mutableMapOf<String, Double>()
        val keys = jsonObject.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val value = jsonObject.optDouble(key, 0.0)
            result[key] = value
        }
        return result
    }
}