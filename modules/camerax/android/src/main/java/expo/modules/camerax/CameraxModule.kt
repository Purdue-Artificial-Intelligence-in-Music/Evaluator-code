package expo.modules.camerax

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

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

            Prop("cameraActive") { view: CameraxView, active: Boolean ->
                view.setCameraActive(active)
            }

            Prop("maxBowAngle") { view: CameraxView, angle: Int ->
                view.setMaxBowAngle(angle)
            }

            Events("onDetectionResult", "onNoDetection", "onSessionEnd")
        }
    }
}