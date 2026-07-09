import ExpoModulesCore
import Foundation

public class CameraxModule: Module {
  public func definition() -> ModuleDefinition {
    Name("Camerax")

    AsyncFunction("getRecentSessions") { (userId: String, count: Int) -> [[String: Any]] in
      let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
      let sessionsDir = docsDir.appendingPathComponent("sessions")

      guard let files = try? FileManager.default.contentsOfDirectory(at: sessionsDir, includingPropertiesForKeys: [.creationDateKey]) else {
        return []
      }

      let summaryFiles = files
        .filter { $0.lastPathComponent.hasSuffix("_summary.json") && $0.lastPathComponent.contains(userId) }
        .sorted { a, b in
          let dateA = (try? a.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
          let dateB = (try? b.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
          return dateA > dateB
        }
        .prefix(count)

      return summaryFiles.compactMap { url -> [String: Any]? in
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return json
      }
    }

    AsyncFunction("getSessionImages") { (userId: String, timestamp: String) -> [String] in
      let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
      let sessionDir = docsDir.appendingPathComponent("sessions/\(userId)/\(timestamp)")

      guard let files = try? FileManager.default.contentsOfDirectory(at: sessionDir, includingPropertiesForKeys: nil) else {
        return []
      }

      return files
        .filter { $0.pathExtension.lowercased() == "png" }
        .map { $0.path }
    }

    View(CameraxView.self) {
      Events("onSessionEnd", "onCalibrated")

      Prop("userId") { (view: CameraxView, value: String) in
        view.userId = value
      }
      Prop("cameraActive") { (view: CameraxView, value: Bool) in
        view.cameraActive = value
      }
      Prop("detectionEnabled") { (view: CameraxView, value: Bool) in
        view.detectionEnabled = value
      }
      Prop("lensType") { (view: CameraxView, value: String) in
        view.lensType = value
      }
      Prop("maxBowAngle") { (view: CameraxView, value: Double) in
        view.maxBowAngle = value
      }
      Prop("skipCalibration") { (view: CameraxView, value: Bool) in
        view.skipCalibration = value
      }
    }
      AsyncFunction("addMockSession") { (userId: String) in
        let profile = Profile()
        profile.createNewID(userId: userId)
        
        // Mock bow frames
        let bowData: [(Int, Int)] = [(0,0),(0,0),(0,0),(2,1),(0,0),(3,0),(0,0),(0,0),(2,1),(0,0)]
        for (h, a) in bowData {
          profile.addSessionData(userId: userId, data: BowFrame(heightClassification: h, angleClassification: a))
        }
        
        // Mock combined frames
        let combinedData: [(Bool,Int,Bool,Int)] = [
          (true,0,true,0),(true,0,true,1),(true,1,true,0),
          (true,0,true,0),(false,-1,false,-1),(true,0,true,0),
          (true,2,true,0),(true,0,true,2),(true,0,true,0),(true,0,true,0)
        ]
        for (hd,hp,pd,ep) in combinedData {
          profile.addSessionData(userId: userId, data: CombinedFrame(handDetected: hd, handPostureClass: hp, poseDetected: pd, elbowPostureClass: ep))
        }
        
        let summary = profile.endSessionAndGetSummary(userId: userId)
        return summary != nil ? profile.summaryToDict(summary: summary!, userId: userId) : [:]
      }
  }
}
