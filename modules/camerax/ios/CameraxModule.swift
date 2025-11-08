import ExpoModulesCore

public class CameraxModule: Module {
  public func definition() -> ModuleDefinition {
    Name("Camerax")

    View(CameraxView.self) {

      Prop("cameraActive") { (view: CameraxView, active: Bool) in
        if active {
          view.start()
        } else {
          view.stop()
        }
      }

      Prop("lensType") { (view: CameraxView, lensType: String) in
        view.setLensType(lensType)
      }

    }
  }
}