import ExpoModulesCore
import AVFoundation
import UIKit
import MediaPipeTasksVision

class CameraxView: ExpoView, AVCaptureVideoDataOutputSampleBufferDelegate, DetectorListener, CombinedLandmarkerListener {

  // MARK: - Events
  let onSessionEnd = EventDispatcher()
  let onCalibrated = EventDispatcher()

  // MARK: - Camera
  private var captureSession: AVCaptureSession?
  private var previewLayer: AVCaptureVideoPreviewLayer?
  private let cameraQueue = DispatchQueue(label: "camera.frame.queue")
  private let sessionQueue = DispatchQueue(label: "camera.session.queue")

  // MARK: - Detection overlay
  private let boxLayer: CAShapeLayer = {
    let layer = CAShapeLayer()
    layer.fillColor = UIColor.clear.cgColor
    layer.strokeColor = UIColor.systemBlue.cgColor
    layer.lineWidth = 3
    return layer
  }()

  private let handLayer: CAShapeLayer = {
    let layer = CAShapeLayer()
    layer.fillColor = UIColor.yellow.cgColor
    layer.strokeColor = UIColor.yellow.cgColor
    layer.lineWidth = 3
    return layer
  }()

  // Same 21-point MediaPipe hand skeleton used by HandLandmarkerHelper.swift's own
  // (private) drawing code, duplicated here for the live-camera overlay.
  private static let handConnections: [(Int, Int)] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
  ]

  // MARK: - Detector
  private var detector: Detector?
  private var handLandmarkerHelper: CombinedLandmarkerHelper?
  private var frameOrientation: CGImagePropertyOrientation = .right

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
            sessionQueue.async {
                if self.cameraActive {
                    self.startCamera()
                } else {
                    self.stopCamera()
                }
            }
        }
    }

  var lensType: String = "back" {
    didSet { sessionQueue.async { self.switchCamera() } }
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
        DispatchQueue.main.async {
          self.boxLayer.path = nil
        }
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
    setupHandLandmarker()
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

  private func setupHandLandmarker() {
    handLandmarkerHelper = CombinedLandmarkerHelper(
      currentDelegate: .gpu,
      runningMode: .liveStream,
      combinedLandmarkerHelperListener: self
    )
    print("✅ HandLandmarker initialized")
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    previewLayer?.frame = bounds
    boxLayer.frame = bounds
    handLayer.frame = bounds
  }

  // MARK: - Permission
    private func checkPermissionAndSetup() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            sessionQueue.async { self.setupSession() }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    self.sessionQueue.async { self.setupSession() }
                }
            }
        default:
            break
        }
    }

  // MARK: - Setup
  // Runs on sessionQueue. All AVCaptureSession mutation (setup/start/stop/switch)
  // is serialized on this one queue so startRunning/stopRunning can never race
  // with a concurrent beginConfiguration/commitConfiguration block.
    private func setupSession() {
        let session = AVCaptureSession()
        session.sessionPreset = .high

        guard let device = getCamera(for: lensType),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else { return }

        session.beginConfiguration()
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        // MediaPipe's MPImage(sampleBuffer:) only accepts BGRA frames.
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: cameraQueue)
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()

        captureSession = session

        DispatchQueue.main.async {
            let preview = AVCaptureVideoPreviewLayer(session: session)
            preview.videoGravity = .resizeAspectFill
            preview.frame = self.bounds
            self.layer.addSublayer(preview)
            self.previewLayer = preview

            self.boxLayer.frame = self.bounds
            self.layer.addSublayer(self.boxLayer)

            self.handLayer.frame = self.bounds
            self.layer.addSublayer(self.handLayer)
        }

        if cameraActive {
            session.startRunning()
        }
    }

  // MARK: - Start / Stop
  // Must only be called on sessionQueue.
    private func startCamera() {
        guard let session = captureSession, !session.isRunning else { return }
        session.startRunning()
    }

  // Must only be called on sessionQueue.
  private func stopCamera() {
    guard let session = captureSession, session.isRunning else { return }
    session.stopRunning()
  }

  // MARK: - Switch Camera
  // Must only be called on sessionQueue.
    private func switchCamera() {
      guard let session = captureSession else { return }
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

    // Rotate the raw sensor buffer upright before detection: the YOLO model and
    // the bow height/angle classification both assume upright frames (this is
    // what the Android version does by rotating camera bitmaps).
    let cgOrientation: CGImagePropertyOrientation = lensType == "front" ? .leftMirrored : .right
    frameOrientation = cgOrientation
    guard let uiImage = pixelBufferToUIImage(pixelBuffer, orientation: cgOrientation) else { return }

    print("🎥 Frame captured, running Detector...")
    detector?.detect(frame: uiImage)

    let handOrientation: UIImage.Orientation = lensType == "front" ? .leftMirrored : .right
    handLandmarkerHelper?.detectLiveStream(
      sampleBuffer: sampleBuffer,
      orientation: handOrientation,
      isFrontCamera: lensType == "front"
    )
  }

  // A CIContext holds GPU resources; create once, not per frame.
  private static let ciContext = CIContext()

  private func pixelBufferToUIImage(_ pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) -> UIImage? {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer).oriented(orientation)
    guard let cgImage = Self.ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
    return UIImage(cgImage: cgImage)
  }

  // MARK: - Overlay drawing
  // The detector runs on frames rotated upright, but the preview layer's
  // captureDevicePoint space is the raw, un-rotated sensor space — so undo the
  // frame rotation on each normalized point before converting to layer space.
  private func uprightToDevicePoint(_ normalized: CGPoint) -> CGPoint {
    switch frameOrientation {
    case .right:
      return CGPoint(x: normalized.y, y: 1 - normalized.x)
    case .left, .leftMirrored:
      return CGPoint(x: 1 - normalized.y, y: normalized.x)
    case .down, .downMirrored:
      return CGPoint(x: 1 - normalized.x, y: 1 - normalized.y)
    default:
      return normalized
    }
  }

  private func boxPath(for points: [Detector.Point], sourceWidth: Int, sourceHeight: Int) -> UIBezierPath? {
    guard let previewLayer = previewLayer, points.count >= 3,
          sourceWidth > 0, sourceHeight > 0 else { return nil }

    let layerPoints = points.map { point -> CGPoint in
      let normalized = CGPoint(x: point.x / Double(sourceWidth), y: point.y / Double(sourceHeight))
      return previewLayer.layerPointConverted(fromCaptureDevicePoint: uprightToDevicePoint(normalized))
    }

    let path = UIBezierPath()
    path.move(to: layerPoints[0])
    layerPoints.dropFirst().forEach { path.addLine(to: $0) }
    path.close()
    return path
  }

  // MARK: - DetectorListener
  func detected(results: Detector.YoloResults, sourceWidth: Int, sourceHeight: Int) {
    let bowResult = detector?.classify(results: results)
    print("🎯 Detected! classification: \(String(describing: bowResult?.classification)), angle: \(String(describing: bowResult?.angle))")

    let hasIssue = (bowResult?.classification != nil && bowResult?.classification != 0) ||
                   (bowResult?.angle == 1)
    let combinedPath = UIBezierPath()
    if let bowPoints = results.bowResults, let bowPath = boxPath(for: bowPoints, sourceWidth: sourceWidth, sourceHeight: sourceHeight) {
      combinedPath.append(bowPath)
    }
    if let stringPoints = results.stringResults, let stringPath = boxPath(for: stringPoints, sourceWidth: sourceWidth, sourceHeight: sourceHeight) {
      combinedPath.append(stringPath)
    }
    DispatchQueue.main.async {
      self.boxLayer.strokeColor = (hasIssue ? UIColor.orange : UIColor.systemBlue).cgColor
      self.boxLayer.path = combinedPath.cgPath.isEmpty ? nil : combinedPath.cgPath
    }

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
    DispatchQueue.main.async {
      self.boxLayer.path = nil
    }
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

  // MARK: - Hand overlay drawing
  // The orientation passed into detectLiveStream only tells MediaPipe how to
  // rotate the image internally before running the model; its landmark
  // projection maps results back into the original buffer's normalized
  // coordinate space. So landmarks are already in capture-device space — the
  // same space the YOLO box points use — and need no inverse rotation here.
  private func handPath(for landmarks: [NormalizedLandmark]) -> UIBezierPath? {
    guard let previewLayer = previewLayer, !landmarks.isEmpty else { return nil }

    func rawPoint(_ landmark: NormalizedLandmark) -> CGPoint {
      let raw = CGPoint(x: Double(landmark.x), y: Double(landmark.y))
      return previewLayer.layerPointConverted(fromCaptureDevicePoint: raw)
    }

    let path = UIBezierPath()
    for (startIndex, endIndex) in Self.handConnections {
      guard startIndex < landmarks.count, endIndex < landmarks.count else { continue }
      path.move(to: rawPoint(landmarks[startIndex]))
      path.addLine(to: rawPoint(landmarks[endIndex]))
    }
    for landmark in landmarks {
      let center = rawPoint(landmark)
      path.move(to: CGPoint(x: center.x + 3, y: center.y))
      path.addArc(withCenter: center, radius: 3, startAngle: 0, endAngle: .pi * 2, clockwise: true)
    }
    return path
  }

  // MARK: - CombinedLandmarkerListener
  func onError(_ error: String, errorCode: Int) {
    print("❌ HandLandmarker error: \(error)")
  }

  func onResults(_ resultBundle: CombinedLandmarkerHelper.CombinedResultBundle) {
    guard let handResult = resultBundle.handResults.first,
          resultBundle.targetHandIndex >= 0,
          resultBundle.targetHandIndex < handResult.landmarks.count else {
      DispatchQueue.main.async { self.handLayer.path = nil }
      return
    }

    let landmarks = handResult.landmarks[resultBundle.targetHandIndex]
    let path = handPath(for: landmarks)
    DispatchQueue.main.async {
      self.handLayer.path = path?.cgPath
    }
  }
}
