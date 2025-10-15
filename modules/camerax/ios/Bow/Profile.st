import Foundation

// Assuming `ReturnBow` is a Swift struct similar to your Kotlin data class
// You can adjust the properties as needed
struct ReturnBow {
    var classification: Int?
    var angle: Int?
}

class Profile {
    
    struct SessionSummary {
        let heightBreakdown: [String: Double]
        let angleBreakdown: [String: Double]
    }
    
    // Dictionary to store user data: userId -> list of sessions -> list of frames
    private var dict: [String: [[ReturnBow]]] = [:]
    
    func getHashedInfo(userId: String) -> [[ReturnBow]]? {
        return dict[userId]
    }
    
    func createNewID(userId: String) {
        if dict[userId] == nil {
            dict[userId] = []
        }
    }
    
    func addSession(userId: String, data: [ReturnBow]) {
        if dict[userId] == nil {
            createNewID(userId: userId)
        }
        dict[userId]?.append(data)
    }
    
    func analyzeSession(session: [ReturnBow]) -> SessionSummary {
        guard !session.isEmpty else {
            return SessionSummary(heightBreakdown: [:], angleBreakdown: [:])
        }
        
        let total = Double(session.count)
        
        // Height classification
        var heightCounts: [String: Int] = [
            "Top": 0,
            "Middle": 0,
            "Bottom": 0,
            "Unknown": 0
        ]
        
        for frame in session {
            switch frame.classification {
            case 3:
                heightCounts["Top", default: 0] += 1
            case 0:
                heightCounts["Middle", default: 0] += 1
            case 2:
                heightCounts["Bottom", default: 0] += 1
            default:
                heightCounts["Unknown", default: 0] += 1
            }
        }
        
        let heightBreakdown = heightCounts.mapValues { (Double($0) / total) * 100 }
        
        // Angle classification
        var angleCounts: [String: Int] = [
            "Correct": 0,
            "Wrong": 0,
            "Unknown": 0
        ]
        
        for frame in session {
            switch frame.angle {
            case 0:
                angleCounts["Correct", default: 0] += 1
            case 1:
                angleCounts["Wrong", default: 0] += 1
            default:
                angleCounts["Unknown", default: 0] += 1
            }
        }
        
        let angleBreakdown = angleCounts.mapValues { (Double($0) / total) * 100 }
        
        return SessionSummary(
            heightBreakdown: heightBreakdown,
            angleBreakdown: angleBreakdown
        )
    }
}