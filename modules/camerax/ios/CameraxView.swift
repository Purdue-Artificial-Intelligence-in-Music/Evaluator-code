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

  // MARK: - State
  private var sessionStartTime: Date?
  private var isDetecting: Bool = false

  // MARK: - Counters for fake session summary
  private var frameCount: Int = 0
  private var heightTop = 0, heightMiddle = 0, heightBottom = 0
  private var angleCorrect = 0, angleWrong = 0
  private var handCorrect = 0, handSupination = 0, handPronation = 0
  private var elbowCorrect = 0, elbowLow = 0, elbowHigh = 0

  // MARK: - Props
  var userId: String = "default_user"
  var maxBowAngle: Double = 18.0

  var cameraActive: Bool = false {
    didSet {
      if cameraActive { startCamera() } else { stopCamera() }
    }
  }

  var lensType: String = "back" {
    didSet { switchCamera() }
  }

  var detectionEnabled: Bool = false {
    didSet {
      if detectionEnabled {
        startDetectionSession()
      } else {
        endDetectionSession()
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
        if granted {
          DispatchQueue.main.async { self.setupSession() }
        }
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
    DispatchQueue.global(qos: .userInitiated).async {
      session.startRunning()
    }
  }

  private func stopCamera() {
    guard let session = captureSession, session.isRunning else { return }
    DispatchQueue.global(qos: .userInitiated).async {
      session.stopRunning()
    }
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

  // MARK: - Detection Session
  private func startDetectionSession() {
    sessionStartTime = Date()
    frameCount = 0
    heightTop = 0; heightMiddle = 0; heightBottom = 0
    angleCorrect = 0; angleWrong = 0
    handCorrect = 0; handSupination = 0; handPronation = 0
    elbowCorrect = 0; elbowLow = 0; elbowHigh = 0
  }

  private func endDetectionSession() {
    guard let start = sessionStartTime else { return }

    let now = Date()
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
    let timestamp = formatter.string(from: now)

    let total = max(frameCount, 1)

    let payload: [String: Any] = [
      "userId": userId,
      "timestamp": timestamp,
      "heightBreakdown": [
        "Top": heightTop,
        "Middle": max(heightMiddle, total - heightTop - heightBottom),
        "Bottom": heightBottom,
        "Unknown": 0
      ],
      "angleBreakdown": [
        "Correct": max(angleCorrect, total - angleWrong),
        "Wrong": angleWrong,
        "Unknown": 0
      ],
      "handPresenceBreakdown": [
        "Detected": total,
        "None": 0
      ],
      "handPostureBreakdown": [
        "Correct": max(handCorrect, total - handSupination - handPronation),
        "Supination": handSupination,
        "Too much pronation": handPronation,
        "Unknown": 0
      ],
      "posePresenceBreakdown": [
        "Detected": total,
        "None": 0
      ],
      "elbowPostureBreakdown": [
        "Correct": max(elbowCorrect, total - elbowLow - elbowHigh),
        "Low elbow": elbowLow,
        "Elbow too high": elbowHigh,
        "Unknown": 0
      ]
    ]

    sessionStartTime = nil
    frameCount = 0

    DispatchQueue.main.async {
      self.onSessionEnd(payload)
    }
  }
}
