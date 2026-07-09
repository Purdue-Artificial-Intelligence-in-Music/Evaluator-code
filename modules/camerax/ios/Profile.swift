import Foundation

class Profile {

    // MARK: - Shared session info
    private static var sharedTimestamp: String = ""
    private static var sharedUserId: String = ""

    static func setSession(userId: String, timestamp: String) {
        sharedUserId = userId
        sharedTimestamp = timestamp
    }

    static func getTimestamp() -> String { return sharedTimestamp }
    static func getUserId() -> String { return sharedUserId }

    // MARK: - Session Summary Model
    struct SessionSummary {
        let heightBreakdown: [String: Double]
        let angleBreakdown: [String: Double]
        let handPresenceBreakdown: [String: Double]
        let handPostureBreakdown: [String: Double]
        let posePresenceBreakdown: [String: Double]
        let elbowPostureBreakdown: [String: Double]
        let timestamp: String
    }

    // MARK: - Session Storage
    private var sessionDict: [String: [Any]] = [:]
    private var outputFiles: [String: URL] = [:]
    private var sessionTimestamps: [String: String] = [:]
    private var sessionTimestampsFormatted: [String: String] = [:]
    private var schedulerTimers: [String: Timer] = [:]

    // MARK: - Accumulated Counters
    private var accumulatedHeightCounts: [String: [String: Int]] = [:]
    private var accumulatedAngleCounts: [String: [String: Int]] = [:]
    private var accumulatedHandCounts: [String: [String: Int]] = [:]
    private var accumulatedPoseCounts: [String: [String: Int]] = [:]
    private var accumulatedHandPostureCounts: [String: [String: Int]] = [:]
    private var accumulatedElbowPostureCounts: [String: [String: Int]] = [:]

    // MARK: - Create New Session
    func createNewID(userId: String) {
        guard sessionDict[userId] == nil else { return }

        sessionDict[userId] = []

        let now = Date()
        let fileFormatter = DateFormatter()
        fileFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = fileFormatter.string(from: now)

        let displayFormatter = DateFormatter()
        displayFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let timestampFormatted = displayFormatter.string(from: now)

        Profile.setSession(userId: userId, timestamp: timestamp)
        sessionTimestamps[userId] = timestamp
        sessionTimestampsFormatted[userId] = timestampFormatted

        // Initialize counters
        accumulatedHeightCounts[userId] = ["Top": 0, "Middle": 0, "Bottom": 0, "Outside": 0, "Unknown": 0]
        accumulatedAngleCounts[userId] = ["Correct": 0, "Wrong": 0, "Unknown": 0]
        accumulatedHandCounts[userId] = ["Detected": 0, "None": 0]
        accumulatedPoseCounts[userId] = ["Detected": 0, "None": 0]
        accumulatedHandPostureCounts[userId] = [:]
        accumulatedElbowPostureCounts[userId] = [:]

        // Create output file in Documents/sessions/
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let sessionsDir = docsDir.appendingPathComponent("sessions")
        try? FileManager.default.createDirectory(at: sessionsDir, withIntermediateDirectories: true)

        let fileURL = sessionsDir.appendingPathComponent("session_\(userId)_\(timestamp).json")
        let initialContent = "{\"user_id\":\"\(userId)\",\"data\":["
        try? initialContent.write(to: fileURL, atomically: false, encoding: .utf8)
        outputFiles[userId] = fileURL

        // Schedule 10-second breakdown
        let timer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { [weak self] _ in
            self?.appendNewBreakdown(userId: userId)
        }
        schedulerTimers[userId] = timer
    }

    func addSessionData(userId: String, data: Any) {
        if sessionDict[userId] == nil {
            createNewID(userId: userId)
        }
        sessionDict[userId]?.append(data)
    }

    // MARK: - End Session
    func endSessionAndGetSummary(userId: String) -> SessionSummary? {
        guard let session = sessionDict[userId] else { return nil }

        // Stop the timer
        schedulerTimers[userId]?.invalidate()
        schedulerTimers.removeValue(forKey: userId)

        // Flush remaining data
        if !session.isEmpty {
            let now = Date()
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
            let currentTimestamp = formatter.string(from: now)
            let windowSummary = analyzeSessionWindow(session: session, timestamp: currentTimestamp, userId: userId)
            appendJsonToFile(userId: userId, summary: windowSummary)
        }

        // Finalize the details file
        if let fileURL = outputFiles[userId] {
            var content = (try? String(contentsOf: fileURL, encoding: .utf8)) ?? ""
            if content.hasSuffix(",") { content = String(content.dropLast()) }
            content += "]}"
            try? content.write(to: fileURL, atomically: true, encoding: .utf8)
        }

        // Generate total summary
        let sessionStartTimestamp = sessionTimestampsFormatted[userId] ?? {
            let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd HH:mm:ss"; return f.string(from: Date())
        }()
        let totalSummary = generateTotalSummary(userId: userId, timestamp: sessionStartTimestamp)
        saveSummaryFile(userId: userId, summary: totalSummary)

        // Reset
        sessionDict.removeValue(forKey: userId)
        outputFiles.removeValue(forKey: userId)
        sessionTimestamps.removeValue(forKey: userId)
        sessionTimestampsFormatted.removeValue(forKey: userId)
        accumulatedHeightCounts.removeValue(forKey: userId)
        accumulatedAngleCounts.removeValue(forKey: userId)
        accumulatedHandCounts.removeValue(forKey: userId)
        accumulatedPoseCounts.removeValue(forKey: userId)
        accumulatedHandPostureCounts.removeValue(forKey: userId)
        accumulatedElbowPostureCounts.removeValue(forKey: userId)

        return totalSummary
    }

    // MARK: - Save Summary File
    private func saveSummaryFile(userId: String, summary: SessionSummary) {
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let sessionsDir = docsDir.appendingPathComponent("sessions")
        try? FileManager.default.createDirectory(at: sessionsDir, withIntermediateDirectories: true)

        let timestamp = sessionTimestamps[userId] ?? {
            let f = DateFormatter(); f.dateFormat = "yyyyMMdd_HHmmss"; return f.string(from: Date())
        }()
        let fileURL = sessionsDir.appendingPathComponent("session_\(userId)_\(timestamp)_summary.json")
        let json = formatSummaryAsJson(summary: summary, userId: userId)
        try? json.write(to: fileURL, atomically: true, encoding: .utf8)
    }

    // MARK: - Append 10s Breakdown
    private func appendNewBreakdown(userId: String) {
        guard let session = sessionDict[userId], !session.isEmpty else { return }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let currentTimestamp = formatter.string(from: Date())
        let summary = analyzeSessionWindow(session: session, timestamp: currentTimestamp, userId: userId)
        appendJsonToFile(userId: userId, summary: summary)
        sessionDict[userId]?.removeAll()
    }

    private func appendJsonToFile(userId: String, summary: SessionSummary) {
        guard let fileURL = outputFiles[userId] else { return }
        let json = formatSummaryAsJson(summary: summary, userId: userId) + ","
        if let handle = try? FileHandle(forWritingTo: fileURL) {
            handle.seekToEndOfFile()
            if let data = json.data(using: .utf8) { handle.write(data) }
            handle.closeFile()
        }
    }

    // MARK: - Analyze Session Window
    private func analyzeSessionWindow(session: [Any], timestamp: String, userId: String) -> SessionSummary {
        if session.isEmpty { return emptySummary(timestamp: timestamp) }

        let bowFrames = session.compactMap { $0 as? BowFrame }
        let combinedFrames = session.compactMap { $0 as? CombinedFrame }

        var heightBreakdown: [String: Double] = [:]
        var angleBreakdown: [String: Double] = [:]

        if !bowFrames.isEmpty {
            let total = Double(bowFrames.count)
            var heightCounts = ["Top": 0, "Middle": 0, "Bottom": 0, "Outside": 0, "Unknown": 0]
            var angleCounts = ["Correct": 0, "Wrong": 0, "Unknown": 0]

            for frame in bowFrames {
                switch frame.heightClassification {
                case 2: heightCounts["Top"]! += 1
                case 0: heightCounts["Middle"]! += 1
                case 3: heightCounts["Bottom"]! += 1
                case 1: heightCounts["Outside"]! += 1
                default: heightCounts["Unknown"]! += 1
                }
                switch frame.angleClassification {
                case 0: angleCounts["Correct"]! += 1
                case 1: angleCounts["Wrong"]! += 1
                default: angleCounts["Unknown"]! += 1
                }
            }

            for (key, value) in heightCounts {
                accumulatedHeightCounts[userId]![key, default: 0] += value
            }
            for (key, value) in angleCounts {
                accumulatedAngleCounts[userId]![key, default: 0] += value
            }

            heightBreakdown = heightCounts.mapValues { Double($0) / total * 100 }
            angleBreakdown = angleCounts.mapValues { Double($0) / total * 100 }
        }

        var handPresenceBreakdown: [String: Double] = [:]
        var handPostureBreakdown: [String: Double] = [:]
        var posePresenceBreakdown: [String: Double] = [:]
        var elbowPostureBreakdown: [String: Double] = [:]

        if !combinedFrames.isEmpty {
            let total = Double(combinedFrames.count)
            var handCounts = ["Detected": 0, "None": 0]
            var poseCounts = ["Detected": 0, "None": 0]
            var handPostureCounts: [String: Int] = [:]
            var elbowPostureCounts: [String: Int] = [:]

            for frame in combinedFrames {
                if frame.handDetected {
                    handCounts["Detected"]! += 1
                    let label: String
                    switch frame.handPostureClass {
                    case 0: label = "Correct"
                    case 1: label = "Supination"
                    case 2: label = "Too much pronation"
                    default: label = "Unknown"
                    }
                    handPostureCounts[label, default: 0] += 1
                } else {
                    handCounts["None"]! += 1
                }

                if frame.poseDetected {
                    poseCounts["Detected"]! += 1
                    let label: String
                    switch frame.elbowPostureClass {
                    case 0: label = "Correct"
                    case 1: label = "Low elbow"
                    case 2: label = "Elbow too high"
                    default: label = "Unknown"
                    }
                    elbowPostureCounts[label, default: 0] += 1
                } else {
                    poseCounts["None"]! += 1
                }
            }

            for (key, value) in handCounts { accumulatedHandCounts[userId]![key, default: 0] += value }
            for (key, value) in poseCounts { accumulatedPoseCounts[userId]![key, default: 0] += value }
            for (key, value) in handPostureCounts { accumulatedHandPostureCounts[userId]![key, default: 0] += value }
            for (key, value) in elbowPostureCounts { accumulatedElbowPostureCounts[userId]![key, default: 0] += value }

            handPresenceBreakdown = handCounts.mapValues { Double($0) / total * 100 }
            posePresenceBreakdown = poseCounts.mapValues { Double($0) / total * 100 }
            handPostureBreakdown = handPostureCounts.mapValues { Double($0) / total * 100 }
            elbowPostureBreakdown = elbowPostureCounts.mapValues { Double($0) / total * 100 }
        }

        return SessionSummary(
            heightBreakdown: heightBreakdown,
            angleBreakdown: angleBreakdown,
            handPresenceBreakdown: handPresenceBreakdown,
            handPostureBreakdown: handPostureBreakdown,
            posePresenceBreakdown: posePresenceBreakdown,
            elbowPostureBreakdown: elbowPostureBreakdown,
            timestamp: timestamp
        )
    }

    // MARK: - Generate Total Summary
    private func generateTotalSummary(userId: String, timestamp: String) -> SessionSummary {
        func toPercentages(_ counts: [String: Int]) -> [String: Double] {
            let total = Double(counts.values.reduce(0, +))
            guard total > 0 else { return [:] }
            return counts.mapValues { Double($0) / total * 100 }
        }

        return SessionSummary(
            heightBreakdown: toPercentages(accumulatedHeightCounts[userId] ?? [:]),
            angleBreakdown: toPercentages(accumulatedAngleCounts[userId] ?? [:]),
            handPresenceBreakdown: toPercentages(accumulatedHandCounts[userId] ?? [:]),
            handPostureBreakdown: toPercentages(accumulatedHandPostureCounts[userId] ?? [:]),
            posePresenceBreakdown: toPercentages(accumulatedPoseCounts[userId] ?? [:]),
            elbowPostureBreakdown: toPercentages(accumulatedElbowPostureCounts[userId] ?? [:]),
            timestamp: timestamp
        )
    }

    // MARK: - JSON Formatting
    func formatSummaryAsJson(summary: SessionSummary, userId: String) -> String {
        func mapToJson(_ map: [String: Double]) -> String {
            let entries = map.map { "\"\($0.key)\":\($0.value)" }.joined(separator: ",")
            return "{\(entries)}"
        }
        return """
        {"user_id":"\(userId)","timestamp":"\(summary.timestamp)",\
        "heightBreakdown":\(mapToJson(summary.heightBreakdown)),\
        "angleBreakdown":\(mapToJson(summary.angleBreakdown)),\
        "handPresenceBreakdown":\(mapToJson(summary.handPresenceBreakdown)),\
        "handPostureBreakdown":\(mapToJson(summary.handPostureBreakdown)),\
        "posePresenceBreakdown":\(mapToJson(summary.posePresenceBreakdown)),\
        "elbowPostureBreakdown":\(mapToJson(summary.elbowPostureBreakdown))}
        """
    }

    func summaryToDict(summary: SessionSummary, userId: String) -> [String: Any] {
        return [
            "userId": userId,
            "user_id": userId,
            "timestamp": summary.timestamp,
            "heightBreakdown": summary.heightBreakdown,
            "angleBreakdown": summary.angleBreakdown,
            "handPresenceBreakdown": summary.handPresenceBreakdown,
            "handPostureBreakdown": summary.handPostureBreakdown,
            "posePresenceBreakdown": summary.posePresenceBreakdown,
            "elbowPostureBreakdown": summary.elbowPostureBreakdown
        ]
    }

    private func emptySummary(timestamp: String) -> SessionSummary {
        return SessionSummary(
            heightBreakdown: [:], angleBreakdown: [:],
            handPresenceBreakdown: [:], handPostureBreakdown: [:],
            posePresenceBreakdown: [:], elbowPostureBreakdown: [:],
            timestamp: timestamp
        )
    }
}

// MARK: - Frame Data Models
struct BowFrame {
    let heightClassification: Int
    let angleClassification: Int
}

struct CombinedFrame {
    let handDetected: Bool
    let handPostureClass: Int
    let poseDetected: Bool
    let elbowPostureClass: Int
}
