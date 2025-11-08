import ExpoModulesCore
import AVFoundation
import UIKit

class CameraxView: ExpoView {
  let session = AVCaptureSession()
  let previewLayer: AVCaptureVideoPreviewLayer

  required init(appContext: AppContext? = nil) {
    self.previewLayer = AVCaptureVideoPreviewLayer(session: session)
    super.init(appContext: appContext)

    previewLayer.videoGravity = .resizeAspectFill
    layer.addSublayer(previewLayer)

    let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)!
    let input = try! AVCaptureDeviceInput(device: device)
    session.addInput(input)
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    previewLayer.frame = self.bounds 
  }

  func start() {
    if !session.isRunning {
      session.startRunning()
    }
  }

  func stop() {
    if session.isRunning {
      session.stopRunning()
    }
  }

  func setLensType(_ lensType: String) {
    session.beginConfiguration()

    if let currentInput = session.inputs.first {
      session.removeInput(currentInput)
    }

    let position: AVCaptureDevice.Position = (lensType == "front") ? .front : .back

    if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) {
      let input = try! AVCaptureDeviceInput(device: device)
      session.addInput(input)
    }

    session.commitConfiguration()
  }
}