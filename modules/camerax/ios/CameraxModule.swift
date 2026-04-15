import ExpoModulesCore

public class CameraxModule: Module {
  public func definition() -> ModuleDefinition {
    Name("Camerax")

    AsyncFunction("getRecentSessions") { (userId: String, count: Int) -> [[String: Any]] in
      return []
    }

    AsyncFunction("getSessionImages") { (userId: String, timestamp: String) -> [String] in
      return []
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
  }
}
