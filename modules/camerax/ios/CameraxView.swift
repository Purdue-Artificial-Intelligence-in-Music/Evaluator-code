import ExpoModulesCore
import AVFoundation
import UIKit

class CameraxView: ExpoView, AVCaptureVideoDataOutputSampleBufferDelegate, DetectorListener {

  // MARK: - Events
  let onSessionEnd = EventDispatcher()
  let onCalibrated = EventDispatcher()

  // MARK: - Camera
  private var captureSession: AVCaptureSession?
  private var previewLayer: AVCaptureVideoPreviewLayer?
  private let cameraQueue = DispatchQueue(label: "camera.frame.queue")

  // MARK: - Detector
  private var detector: Detector?

  // MARK: - Profile
  private let profile = Profile()

  // MARK: - Calibration
  private var calibrationCount: Int = 0
  private var calibrationCorrect: Int = 0
  private var isCalibrated: Bool = false

  // MARK: - Props
  var userId: String = "default_user"
  var maxBowAngle: Double = 18.0

    var cameraActive: Bool = false {
        didSet {
            DispatchQueue.main.async {
                if self.cameraActive {
                    self.startCamera()
                } else {
                    self.stopCamera()
                }
            }
        }
    }

  var lensType: String = "back" {
    didSet { switchCamera() }
  }

  var detectionEnabled: Bool = false {
    didSet {
      if detectionEnabled {
        isCalibrated = false
        calibrationCount = 0
        calibrationCorrect = 0
        profile.createNewID(userId: userId)
        print("✅ Detection started for user: \(userId)")
      } else {
        if let summary = profile.endSessionAndGetSummary(userId: userId) {
          let payload = profile.summaryToDict(summary: summary, userId: userId)
          DispatchQueue.main.async { self.onSessionEnd(payload) }
        }
        detector?.resetHeaps()
        print("⛔ Detection stopped")
      }
    }
  }

  var skipCalibration: Bool = false {
    didSet {
      if skipCalibration {
        isCalibrated = true
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
    setupDetector()
    checkPermissionAndSetup()
  }

  private func setupDetector() {
    do {
      detector = try Detector(listener: self)
      detector?.setMaxAngle(angle: Int(maxBowAngle))
      print("✅ Detector initialized")
    } catch {
      print("❌ Detector init failed: \(error)")
    }
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    previewLayer?.frame = bounds
  }

  // MARK: - Permission
    private func checkPermissionAndSetup() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            DispatchQueue.main.async {
                self.setupSession()
            }
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

        session.beginConfiguration()
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: cameraQueue)
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()

        let preview = AVCaptureVideoPreviewLayer(session: session)
        preview.videoGravity = .resizeAspectFill
        preview.frame = bounds
        layer.addSublayer(preview)
        previewLayer = preview

        captureSession = session

        if cameraActive {
            DispatchQueue.global(qos: .userInitiated).async {
                session.startRunning()
            }
        }
    }

  // MARK: - Start / Stop
    private func startCamera() {
        guard let session = captureSession, !session.isRunning else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            session.startRunning()
        }
    }

  private func stopCamera() {
    guard let session = captureSession, session.isRunning else { return }
    DispatchQueue.global(qos: .userInitiated).async { session.stopRunning() }
  }

  // MARK: - Switch Camera
    private func switchCamera() {
      guard let session = captureSession else { return }
      DispatchQueue.global(qos: .userInitiated).async {
        let wasRunning = session.isRunning
        if wasRunning { session.stopRunning() }
        session.beginConfiguration()
        session.inputs.forEach { session.removeInput($0) }
        if let device = self.getCamera(for: self.lensType),
           let input = try? AVCaptureDeviceInput(device: device),
           session.canAddInput(input) {
          session.addInput(input)
        }
        session.commitConfiguration()
        if wasRunning { session.startRunning() }
      }
    }

  private func getCamera(for lensType: String) -> AVCaptureDevice? {
    let position: AVCaptureDevice.Position = lensType == "front" ? .front : .back
    return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
  }

  // MARK: - Frame Processing
  func captureOutput(_ output: AVCaptureOutput,
                     didOutput sampleBuffer: CMSampleBuffer,
                     from connection: AVCaptureConnection) {
    guard detectionEnabled else { return }
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    guard let uiImage = pixelBufferToUIImage(pixelBuffer) else { return }

    print("🎥 Frame captured, running Detector...")
    detector?.detect(frame: uiImage)
  }

  private func pixelBufferToUIImage(_ pixelBuffer: CVPixelBuffer) -> UIImage? {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext()
    guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
    return UIImage(cgImage: cgImage)
  }

  // MARK: - DetectorListener
  func detected(results: Detector.YoloResults, sourceWidth: Int, sourceHeight: Int) {
    let bowResult = detector?.classify(results: results)
    print("🎯 Detected! classification: \(String(describing: bowResult?.classification)), angle: \(String(describing: bowResult?.angle))")

    if isCalibrated {
      if let bowResult = bowResult {
        let frame = BowFrame(
          heightClassification: bowResult.classification ?? -1,
          angleClassification: bowResult.angle ?? -1
        )
        profile.addSessionData(userId: userId, data: frame)
      }
    } else {
      calibrationCount += 1
      calibrationCorrect += 1
      if calibrationCount == 30 {
        let ratio = Double(calibrationCorrect) / Double(calibrationCount)
        if ratio >= 0.6 {
          isCalibrated = true
          DispatchQueue.main.async { self.onCalibrated(["calibration": true]) }
          print("✅ Calibrated!")
        } else {
          calibrationCount = 0
          calibrationCorrect = 0
        }
      }
    }
  }

  func noDetect() {
    print("❌ No detection")
    if !isCalibrated {
      calibrationCount += 1
      if calibrationCount == 30 {
        let ratio = Double(calibrationCorrect) / Double(calibrationCount)
        if ratio >= 0.6 {
          isCalibrated = true
          DispatchQueue.main.async { self.onCalibrated(["calibration": true]) }
        } else {
          calibrationCount = 0
          calibrationCorrect = 0
        }
      }
    }
  }
}
