import ExpoModulesCore
import AVFoundation
import UIKit

class CameraxView: ExpoView {
  // MARK: - Events
  let onSessionEnd = EventDispatcher()
  let onCalibrated = EventDispatcher()

  // MARK: - Camera
  private var captureSession: AVCaptureSession?
  private var previewLayer: AVCaptureVideoPreviewLayer?

  // MARK: - Profile
  private let profile = Profile()

  // MARK: - Props
  var userId: String = "default_user"
  var maxBowAngle: Double = 18.0

  var cameraActive: Bool = false {
    didSet { cameraActive ? startCamera() : stopCamera() }
  }

  var lensType: String = "back" {
    didSet { switchCamera() }
  }

  var detectionEnabled: Bool = false {
    didSet {
      if detectionEnabled {
        profile.createNewID(userId: userId)
      } else {
        if let summary = profile.endSessionAndGetSummary(userId: userId) {
          let payload = profile.summaryToDict(summary: summary, userId: userId)
          DispatchQueue.main.async { self.onSessionEnd(payload) }
        }
      }
    }
  }

  var skipCalibration: Bool = false {
    didSet {
      if skipCalibration {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
          self.onCalibrated([:])
        }
      }
    }
  }

  // MARK: - Init
  required init(appContext: AppContext? = nil) {
    super.init(appContext: appContext)
    clipsToBounds = true
    checkPermissionAndSetup()
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    previewLayer?.frame = bounds
  }

  // MARK: - Permission
  private func checkPermissionAndSetup() {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
      setupSession()
    case .notDetermined:
      AVCaptureDevice.requestAccess(for: .video) { granted in
        if granted { DispatchQueue.main.async { self.setupSession() } }
      }
    default:
      break
    }
  }

  // MARK: - Setup
  private func setupSession() {
    let session = AVCaptureSession()
    session.sessionPreset = .high

    guard let device = getCamera(for: lensType),
          let input = try? AVCaptureDeviceInput(device: device),
          session.canAddInput(input) else { return }

    session.addInput(input)

    let preview = AVCaptureVideoPreviewLayer(session: session)
    preview.videoGravity = .resizeAspectFill
    preview.frame = bounds
    layer.addSublayer(preview)

    previewLayer = preview
    captureSession = session

    if cameraActive { startCamera() }
  }

  // MARK: - Start / Stop Camera
  private func startCamera() {
    guard let session = captureSession, !session.isRunning else { return }
    DispatchQueue.global(qos: .userInitiated).async { session.startRunning() }
  }

  private func stopCamera() {
    guard let session = captureSession, session.isRunning else { return }
    DispatchQueue.global(qos: .userInitiated).async { session.stopRunning() }
  }

  // MARK: - Switch Camera
  private func switchCamera() {
    guard let session = captureSession else { return }
    DispatchQueue.global(qos: .userInitiated).async {
      session.beginConfiguration()
      session.inputs.forEach { session.removeInput($0) }
      if let device = self.getCamera(for: self.lensType),
         let input = try? AVCaptureDeviceInput(device: device),
         session.canAddInput(input) {
        session.addInput(input)
      }
      session.commitConfiguration()
    }
  }

  private func getCamera(for lensType: String) -> AVCaptureDevice? {
    let position: AVCaptureDevice.Position = lensType == "front" ? .front : .back
    return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
  }

  // MARK: - Public: Add Frame Data
  func addBowFrame(heightClass: Int, angleClass: Int) {
    guard detectionEnabled else { return }
    let frame = BowFrame(heightClassification: heightClass, angleClassification: angleClass)
    profile.addSessionData(userId: userId, data: frame)
  }

  func addCombinedFrame(handDetected: Bool, handPostureClass: Int, poseDetected: Bool, elbowPostureClass: Int) {
    guard detectionEnabled else { return }
    let frame = CombinedFrame(handDetected: handDetected, handPostureClass: handPostureClass,
                               poseDetected: poseDetected, elbowPostureClass: elbowPostureClass)
    profile.addSessionData(userId: userId, data: frame)
  }
}
