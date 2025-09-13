class Profile {
    data class SessionSummary(
        val heightBreakdown: Map<String, Double>, 
        val angleBreakdown: Map<String, Double>
    )
    
    private val dict: MutableMap<String, MutableList<MutableList<returnBow>>> = mutableMapOf()

    fun getHashedInfo(userId: String): MutableList<MutableList<returnBow>>? {
        return dict[userId]
    }

    fun createNewID(userId: String) {
        if (userId !in dict) {
            dict[userId] = mutableListOf()
        }
    }

    fun addSession(userId: String, data: MutableList<returnBow>) {
        if (userId !in dict) {
            createNewID(userId)
        }
        dict[userId]?.add(data)
    }

    fun analyzeSession(session: List<Detector.returnBow>): SessionSummary {
        if (session.isEmpty()) {
            return SessionSummary(emptyMap(), emptyMap())
        }

        val total = session.size.toDouble()

        // Height classification
        val heightCounts = mutableMapOf("Top" to 0, "Middle" to 0, "Bottom" to 0, "Unknown" to 0)
        session.forEach { frame ->
            when (frame.classification) {
                3 -> heightCounts["Top"] = heightCounts["Top"]!! + 1
                0 -> heightCounts["Middle"] = heightCounts["Middle"]!! + 1
                2 -> heightCounts["Bottom"] = heightCounts["Bottom"]!! + 1
                else -> heightCounts["Unknown"] = heightCounts["Unknown"]!! + 1
            }
        }
        val heightBreakdown = heightCounts.mapValues { (_, v) -> (v / total) * 100 }

        // Angle classification
        val angleCounts = mutableMapOf("Correct" to 0, "Wrong" to 0, "Unknown" to 0)
        session.forEach { frame ->
            when (frame.angle) {
                0 -> angleCounts["Correct"] = angleCounts["Correct"]!! + 1
                1 -> angleCounts["Wrong"] = angleCounts["Wrong"]!! + 1
                else -> angleCounts["Unknown"] = angleCounts["Unknown"]!! + 1
            }
        }
        val angleBreakdown = angleCounts.mapValues { (_, v) -> (v / total) * 100 }

        return SessionSummary(heightBreakdown, angleBreakdown)
    }
}
