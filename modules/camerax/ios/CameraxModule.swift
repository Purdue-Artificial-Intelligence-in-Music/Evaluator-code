import ExpoModulesCore

public class CameraxModule: Module {
  public func definition() -> ModuleDefinition {
    Name("Camerax")

    Constants([
      "PI": Double.pi
    ])

    Events("onChange")

    Function("hello") {
      return "Hello from iOS Camerax"
    }

    AsyncFunction("setValueAsync") { (value: String) in
      self.sendEvent("onChange", [
        "value": value
      ])
    }

    // UI-only placeholders for parity with Android while camera logic is pending on iOS.
    AsyncFunction("getRecentSessions") { (_: String, _: Int) -> [[String: Any]] in
      return []
    }

    AsyncFunction("getSessionImages") { (_: String, _: String) -> [String] in
      return []
    }

    View(CameraxView.self) {
      Prop("userId") { (view: CameraxView, userId: String?) in
        view.setUserId(userId ?? "default_user")
      }

      Prop("cameraActive") { (view: CameraxView, active: Bool?) in
        view.setCameraActive(active ?? false)
      }

      Prop("detectionEnabled") { (view: CameraxView, enabled: Bool?) in
        view.setDetectionEnabled(enabled ?? false)
      }

      Prop("lensType") { (view: CameraxView, lensType: String?) in
        view.setLensType(lensType ?? "front")
      }

      Prop("flip") { (view: CameraxView, flip: Bool?) in
        view.setFlip(flip ?? false)
      }

      Prop("maxBowAngle") { (view: CameraxView, angle: Int?) in
        view.setMaxBowAngle(angle ?? 20)
      }

      Events("onDetectionResult", "onNoDetection", "onSessionEnd")
    }
  }
}
