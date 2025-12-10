package expo.modules.camerax

import expo.modules.camerax.HandLandmarkerHelper.CombinedResultBundle
import expo.modules.camerax.Detector.returnBow
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class Profile {

    companion object {
        private var ts: String = ""
        private var id: String = ""

        fun setSession(userId: String, timestamp: String) {
            id = userId
            ts = timestamp
        }

        fun getTimeStamp(): String {
            return ts
        }

        fun getUserId(): String {
            return id
        }
    }

    data class SessionSummary(
        val heightBreakdown: Map<String, Double>,
        val angleBreakdown: Map<String, Double>,
        val handPresenceBreakdown: Map<String, Double>,
        val handPostureBreakdown: Map<String, Double>,
        val posePresenceBreakdown: Map<String, Double>,
        val elbowPostureBreakdown: Map<String, Double>,
        val timestamp: String
    )

    private val sessionDict: MutableMap<String, MutableList<Any>> = mutableMapOf()
    private val scheduler = Executors.newScheduledThreadPool(1)
    private val outputFiles: MutableMap<String, File> = mutableMapOf()
    private val sessionTimestamps: MutableMap<String, String> = mutableMapOf() // store timestamp when session began
    private val sessionTimestampsFormatted: MutableMap<String, String> = mutableMapOf()

    // Accumulation counter used for session summary
    private val accumulatedHeightCounts: MutableMap<String, MutableMap<String, Int>> = mutableMapOf()
    private val accumulatedAngleCounts: MutableMap<String, MutableMap<String, Int>> = mutableMapOf()
    private val accumulatedHandCounts: MutableMap<String, MutableMap<String, Int>> = mutableMapOf()
    private val accumulatedPoseCounts: MutableMap<String, MutableMap<String, Int>> = mutableMapOf()
    private val accumulatedHandPostureCounts: MutableMap<String, MutableMap<String, Int>> = mutableMapOf()
    private val accumulatedElbowPostureCounts: MutableMap<String, MutableMap<String, Int>> = mutableMapOf()

    private var ts: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
    private var id: String = ""

    fun createNewID(userId: String) {
        if (userId !in sessionDict) {
            sessionDict[userId] = mutableListOf()
            val baseDir = File(
                android.os.Environment.getExternalStoragePublicDirectory(
                    android.os.Environment.DIRECTORY_DOCUMENTS
                ), "sessions"
            )
            if (!baseDir.exists()) baseDir.mkdirs()

            val now = Date()
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(now)
            val timestampFormatted = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(now)

            setSession(userId, timestamp)

            sessionTimestamps[userId] = timestamp
            sessionTimestampsFormatted[userId] = timestampFormatted

            // initialize counters
            accumulatedHeightCounts[userId] = mutableMapOf("Top" to 0, "Middle" to 0, "Bottom" to 0, "Outside" to 0, "Unknown" to 0)
            accumulatedAngleCounts[userId] = mutableMapOf("Correct" to 0, "Wrong" to 0, "Unknown" to 0)
            accumulatedHandCounts[userId] = mutableMapOf("Detected" to 0, "None" to 0)
            accumulatedPoseCounts[userId] = mutableMapOf("Detected" to 0, "None" to 0)
            accumulatedHandPostureCounts[userId] = mutableMapOf()
            accumulatedElbowPostureCounts[userId] = mutableMapOf()

            android.util.Log.d("Profile", "Session created for $userId")
            android.util.Log.d("Profile", "Timestamp (file): $timestamp")
            android.util.Log.d("Profile", "Timestamp (json): $timestampFormatted")

            // create file with detail breakdown（without summary）
            val file = File(baseDir, "session_${userId}_$timestamp.json")

            // Write initial user ID at top of JSON
            FileWriter(file, false).use { writer ->
                writer.write("{\"user_id\":\"$userId\",\"data\":[")
            }
            outputFiles[userId] = file

            // Start 10-second breakdown scheduler
            scheduler.scheduleAtFixedRate({
                try {
                    appendNewBreakdown(userId)
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }, 10, 10, TimeUnit.SECONDS)
        }
    }

    fun addSessionData(userId: String, data: Any) {
        if (userId !in sessionDict) {
            createNewID(userId)
        }
        sessionDict[userId]?.add(data)
    }

    fun endSessionAndGetSummary(userId: String): SessionSummary? {
        val file = outputFiles[userId]
        val session = sessionDict[userId] ?: return null

        // Add remaining data in the last interval before ending to the details file
        if (session.isNotEmpty()) {
            val currentTimestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
            val windowSummary = analyzeSessionWindow(session, currentTimestamp, userId)
            val jsonSummary = formatSummaryAsJson(windowSummary, userId)

            FileWriter(file, true).use { writer ->
                writer.write("$jsonSummary,")
            }
        }

        // Finalize details file
        if (file != null) {
            FileWriter(file, true).use { writer ->
                val fileContent = file.readText()
                if (fileContent.endsWith(",")) {
                    file.writeText(fileContent.dropLast(1))
                }
                writer.write("]}")
            }
        }

        // Create session summary file
        val sessionStartTimestamp = sessionTimestampsFormatted[userId]
            ?: SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())

        val totalSummary = generateTotalSummary(userId, sessionStartTimestamp)

        saveSummaryFile(userId, totalSummary)

        // Reset
        sessionDict.remove(userId)
        outputFiles.remove(userId)
        sessionTimestamps.remove(userId)
        sessionTimestampsFormatted.remove(userId)
        accumulatedHeightCounts.remove(userId)
        accumulatedAngleCounts.remove(userId)
        accumulatedHandCounts.remove(userId)
        accumulatedPoseCounts.remove(userId)
        accumulatedHandPostureCounts.remove(userId)
        accumulatedElbowPostureCounts.remove(userId)

        return totalSummary
    }

    private fun saveSummaryFile(userId: String, summary: SessionSummary) {
        try {
            val baseDir = File(
                android.os.Environment.getExternalStoragePublicDirectory(
                    android.os.Environment.DIRECTORY_DOCUMENTS
                ), "sessions"
            )
            if (!baseDir.exists()) baseDir.mkdirs()

            val timestamp = sessionTimestamps[userId] ?: SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val summaryFile = File(baseDir, "session_${userId}_${timestamp}_summary.json")

            val jsonSummary = formatSummaryAsJson(summary, userId)
            FileWriter(summaryFile, false).use { writer ->
                writer.write(jsonSummary)
            }

            android.util.Log.d("Profile", "Summary saved to: ${summaryFile.absolutePath}")
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun appendNewBreakdown(userId: String) {
        val session = sessionDict[userId] ?: return
        if (session.isEmpty()) return
        val currentTimestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
        val summary = analyzeSessionWindow(session, currentTimestamp, userId)
        val jsonSummary = formatSummaryAsJson(summary, userId)
        val file = outputFiles[userId] ?: return

        // Append JSON segment and clear current data for next window
        FileWriter(file, true).use { writer ->
            writer.write("$jsonSummary,")
        }
        sessionDict[userId]?.clear()
    }

    private fun analyzeSessionWindow(session: List<Any>, timestamp: String, userId: String): SessionSummary {
        // analyze curren session interval
        if (session.isEmpty()) return emptySummary(timestamp)

        val bowFrames = session.filterIsInstance<returnBow>()
        val combinedFrames = session.filterIsInstance<CombinedResultBundle>()

        var heightBreakdown: Map<String, Double> = emptyMap()
        var angleBreakdown: Map<String, Double> = emptyMap()

        if (bowFrames.isNotEmpty()) {
            val total = bowFrames.size.toDouble()
            val heightCounts = mutableMapOf("Top" to 0, "Middle" to 0, "Bottom" to 0, "Outside" to 0, "Unknown" to 0)
            val angleCounts = mutableMapOf("Correct" to 0, "Wrong" to 0, "Unknown" to 0)

            bowFrames.forEach { frame ->
                when (frame.classification) {
                    3 -> heightCounts["Top"] = heightCounts["Top"]!! + 1
                    0 -> heightCounts["Middle"] = heightCounts["Middle"]!! + 1
                    2 -> heightCounts["Bottom"] = heightCounts["Bottom"]!! + 1
                    1 -> heightCounts["Outside"] = heightCounts["Outside"]!! + 1
                    else -> heightCounts["Unknown"] = heightCounts["Unknown"]!! + 1
                }
                when (frame.angle) {
                    0 -> angleCounts["Correct"] = angleCounts["Correct"]!! + 1
                    1 -> angleCounts["Wrong"] = angleCounts["Wrong"]!! + 1
                    else -> angleCounts["Unknown"] = angleCounts["Unknown"]!! + 1
                }
            }
            // Record the counts
            heightCounts.forEach { (key, value) ->
                accumulatedHeightCounts[userId]!![key] = accumulatedHeightCounts[userId]!!.getOrDefault(key, 0) + value
            }
            angleCounts.forEach { (key, value) ->
                accumulatedAngleCounts[userId]!![key] = accumulatedAngleCounts[userId]!!.getOrDefault(key, 0) + value
            }

            heightBreakdown = heightCounts.mapValues { (it.value / total) * 100 }
            angleBreakdown = angleCounts.mapValues { (it.value / total) * 100 }
        }

        var handPresenceBreakdown: Map<String, Double> = emptyMap()
        var handPostureBreakdown: Map<String, Double> = emptyMap()
        var posePresenceBreakdown: Map<String, Double> = emptyMap()
        var elbowPostureBreakdown: Map<String, Double> = emptyMap()

        if (combinedFrames.isNotEmpty()) {
            val total = combinedFrames.size.toDouble()
            val handCounts = mutableMapOf("Detected" to 0, "None" to 0)
            val poseCounts = mutableMapOf("Detected" to 0, "None" to 0)
            val handPostureCounts = mutableMapOf<String, Int>()
            val elbowPostureCounts = mutableMapOf<String, Int>()
            val classRegex = """Prediction: (\d+) \(Confidence: ([\d.]+)\)""".toRegex()

            combinedFrames.forEach { bundle ->
                if (bundle.handResults.isNotEmpty()) {
                    handCounts["Detected"] = handCounts["Detected"]!! + 1
                    val match = classRegex.find(bundle.handDetection)
                    val handClass = match?.groupValues?.get(1)?.toIntOrNull() ?: -1
                    val label = when (handClass) {
                        0 -> "Correct"
                        1 -> "Supination"
                        2 -> "Too much pronation"
                        else -> "Unknown"
                    }
                    handPostureCounts[label] = handPostureCounts.getOrDefault(label, 0) + 1
                } else {
                    handCounts["None"] = handCounts["None"]!! + 1
                }

                if (bundle.poseResults.isNotEmpty()) {
                    poseCounts["Detected"] = poseCounts["Detected"]!! + 1
                    val match = classRegex.find(bundle.poseDetection)
                    val poseClass = match?.groupValues?.get(1)?.toIntOrNull() ?: -1
                    val label = when (poseClass) {
                        0 -> "Correct"
                        1 -> "Low elbow"
                        2 -> "Elbow too high"
                        else -> "Unknown"
                    }
                    elbowPostureCounts[label] = elbowPostureCounts.getOrDefault(label, 0) + 1
                } else {
                    poseCounts["None"] = poseCounts["None"]!! + 1
                }
            }
            // Record the counts
            handCounts.forEach { (key, value) ->
                accumulatedHandCounts[userId]!![key] = accumulatedHandCounts[userId]!!.getOrDefault(key, 0) + value
            }
            poseCounts.forEach { (key, value) ->
                accumulatedPoseCounts[userId]!![key] = accumulatedPoseCounts[userId]!!.getOrDefault(key, 0) + value
            }
            handPostureCounts.forEach { (key, value) ->
                accumulatedHandPostureCounts[userId]!![key] = accumulatedHandPostureCounts[userId]!!.getOrDefault(key, 0) + value
            }
            elbowPostureCounts.forEach { (key, value) ->
                accumulatedElbowPostureCounts[userId]!![key] = accumulatedElbowPostureCounts[userId]!!.getOrDefault(key, 0) + value
            }

            handPresenceBreakdown = handCounts.mapValues { (it.value / total) * 100 }
            posePresenceBreakdown = poseCounts.mapValues { (it.value / total) * 100 }
            handPostureBreakdown = handPostureCounts.mapValues { (it.value / total) * 100 }
            elbowPostureBreakdown = elbowPostureCounts.mapValues { (it.value / total) * 100 }
        }

        return SessionSummary(
            heightBreakdown,
            angleBreakdown,
            handPresenceBreakdown,
            handPostureBreakdown,
            posePresenceBreakdown,
            elbowPostureBreakdown,
            timestamp
        )
    }

    private fun generateTotalSummary(userId: String, timestamp: String): SessionSummary {
        val heightCounts = accumulatedHeightCounts[userId] ?: mutableMapOf()
        val angleCounts = accumulatedAngleCounts[userId] ?: mutableMapOf()
        val handCounts = accumulatedHandCounts[userId] ?: mutableMapOf()
        val poseCounts = accumulatedPoseCounts[userId] ?: mutableMapOf()
        val handPostureCounts = accumulatedHandPostureCounts[userId] ?: mutableMapOf()
        val elbowPostureCounts = accumulatedElbowPostureCounts[userId] ?: mutableMapOf()

        val heightTotal = heightCounts.values.sum().toDouble()
        val heightBreakdown = if (heightTotal > 0) {
            heightCounts.mapValues { (it.value / heightTotal) * 100 }
        } else {
            emptyMap()
        }

        val angleTotal = angleCounts.values.sum().toDouble()
        val angleBreakdown = if (angleTotal > 0) {
            angleCounts.mapValues { (it.value / angleTotal) * 100 }
        } else {
            emptyMap()
        }

        val handTotal = handCounts.values.sum().toDouble()
        val handPresenceBreakdown = if (handTotal > 0) {
            handCounts.mapValues { (it.value / handTotal) * 100 }
        } else {
            emptyMap()
        }

        val poseTotal = poseCounts.values.sum().toDouble()
        val posePresenceBreakdown = if (poseTotal > 0) {
            poseCounts.mapValues { (it.value / poseTotal) * 100 }
        } else {
            emptyMap()
        }

        val handPostureTotal = handPostureCounts.values.sum().toDouble()
        val handPostureBreakdown = if (handPostureTotal > 0) {
            handPostureCounts.mapValues { (it.value / handPostureTotal) * 100 }
        } else {
            emptyMap()
        }

        val elbowPostureTotal = elbowPostureCounts.values.sum().toDouble()
        val elbowPostureBreakdown = if (elbowPostureTotal > 0) {
            elbowPostureCounts.mapValues { (it.value / elbowPostureTotal) * 100 }
        } else {
            emptyMap()
        }

        return SessionSummary(
            heightBreakdown,
            angleBreakdown,
            handPresenceBreakdown,
            handPostureBreakdown,
            posePresenceBreakdown,
            elbowPostureBreakdown,
            timestamp
        )
    }

    private fun formatSummaryAsJson(summary: SessionSummary, userId: String): String {
        val json = buildString {
            append("{")
            append("\"user_id\":\"$userId\",")
            append("\"timestamp\":\"${summary.timestamp}\",")
            append("\"heightBreakdown\":${mapToJson(summary.heightBreakdown)},")
            append("\"angleBreakdown\":${mapToJson(summary.angleBreakdown)},")
            append("\"handPresenceBreakdown\":${mapToJson(summary.handPresenceBreakdown)},")
            append("\"handPostureBreakdown\":${mapToJson(summary.handPostureBreakdown)},")
            append("\"posePresenceBreakdown\":${mapToJson(summary.posePresenceBreakdown)},")
            append("\"elbowPostureBreakdown\":${mapToJson(summary.elbowPostureBreakdown)}")
            append("}")
        }
        return json
    }

    private fun mapToJson(map: Map<String, Double>): String {
        return map.entries.joinToString(prefix = "{", postfix = "}", separator = ",") { (key, value) ->
            "\"$key\":$value"
        }
    }

    private fun emptySummary(timestamp: String): SessionSummary {
        return SessionSummary(emptyMap(), emptyMap(), emptyMap(), emptyMap(), emptyMap(), emptyMap(), timestamp)
    }
}