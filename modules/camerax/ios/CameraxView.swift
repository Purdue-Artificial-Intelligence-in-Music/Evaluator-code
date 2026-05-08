import ExpoModulesCore
import UIKit

class CameraxView: ExpoView {
  private let onDetectionResult = EventDispatcher()
  private let onNoDetection = EventDispatcher()
  private let onSessionEnd = EventDispatcher()

  private let root = UIView()
  private let titleLabel = UILabel()
  private let subtitleLabel = UILabel()
  private let stateLabel = UILabel()
  private let chip = UILabel()

    // Our overlay sits on top of everything else
    private let overlayView   = OverlayView()

    // ── Props ───────────────────────────────────────────────────────────────
    private var userId:          String = "default_user"
    private var cameraActive:    Bool   = false
    private var detectionEnabled: Bool  = false
    private var lensType:        String = "front"
    private var flip:            Bool   = false
    private var maxBowAngle:     Int    = 20

    private var mockupTimer: Timer?


    required init(appContext: AppContext? = nil) {
        super.init(appContext: appContext)
        clipsToBounds = true

        // root (camera placeholder background)
        root.layer.cornerRadius = 12
        root.clipsToBounds = true
        addSubview(root)

        // chip
        chip.text            = "iOS UI Preview"
        chip.font            = UIFont.systemFont(ofSize: 12, weight: .semibold)
        chip.textColor       = .white
        chip.backgroundColor = UIColor(red: 0.09, green: 0.13, blue: 0.2, alpha: 0.85)
        chip.textAlignment   = .center
        chip.layer.cornerRadius = 10
        chip.clipsToBounds   = true
        root.addSubview(chip)

        // title
        titleLabel.text          = "Camera Surface"
        titleLabel.font          = UIFont.systemFont(ofSize: 24, weight: .bold)
        titleLabel.textColor     = .white
        titleLabel.textAlignment = .center
        root.addSubview(titleLabel)

        // subtitle
        subtitleLabel.text          = "Native iOS UI is wired. Camera functionality will be added next."
        subtitleLabel.font          = UIFont.systemFont(ofSize: 14, weight: .regular)
        subtitleLabel.textColor     = UIColor(white: 1.0, alpha: 0.9)
        subtitleLabel.textAlignment = .center
        subtitleLabel.numberOfLines = 0
        root.addSubview(subtitleLabel)

        // state
        stateLabel.font          = UIFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        stateLabel.textColor     = UIColor(white: 1.0, alpha: 0.92)
        stateLabel.textAlignment = .center
        stateLabel.numberOfLines = 0
        root.addSubview(stateLabel)

        // overlay — transparent, on top of root
        overlayView.backgroundColor = .clear
        overlayView.isOpaque        = false
        addSubview(overlayView)

        refreshUI()
    }

    // MARK: - Layout

    override func layoutSubviews() {
        super.layoutSubviews()
        root.frame        = bounds
        root.layer.backgroundColor = buildBackgroundColor().cgColor
        overlayView.frame = bounds   // always covers the full view

        let inset: CGFloat = 16
        chip.frame = CGRect(x: inset, y: inset, width: 110, height: 28)

        let contentWidth = bounds.width - inset * 2

        titleLabel.frame = CGRect(
            x: inset, y: bounds.midY - 72,
            width: contentWidth, height: 32
        )
        subtitleLabel.frame = CGRect(
            x: inset, y: titleLabel.frame.maxY + 8,
            width: contentWidth, height: 48
        )
        stateLabel.frame = CGRect(
            x: inset, y: bounds.height - inset - 90,
            width: contentWidth, height: 90
        )

        // Inject mockup data once layout is known
        injectMockupIfNeeded()
    }

    // MARK: - Prop setters

  func setUserId(_ value: String) {
    userId = value
    refreshUI()
  }

  func setCameraActive(_ value: Bool) {
    cameraActive = value
    refreshUI()
  }

  func setDetectionEnabled(_ value: Bool) {
    detectionEnabled = value
    if !value {
      onSessionEnd([
        "userId": userId,
        "timestamp": "",
        "durationSeconds": 0,
        "durationFormatted": "0s"
      ])
      overlayView.clear()
      stopMockupTimer()
    }
    refreshUI()
  }

  func setLensType(_ value: String) {
    lensType = value.lowercased()
    overlayView.setFrontCameraState(lensType == "front")
    refreshUI()
  }

  func setFlip(_ value: Bool) {
    flip = value
    overlayView.setFlipState(flip)
    refreshUI()
  }

  func setMaxBowAngle(_ value: Int) {
    maxBowAngle = value
    refreshUI()
  }

    // MARK: - Mockup injection

    // Change this number (1–4) to switch test scenarios
    private let mockupScenario = 2

    private var mockupInjected = false

    private func injectMockupIfNeeded() {
        guard !mockupInjected, bounds.width > 0 else { return }
        mockupInjected = true

        overlayView.setImageDimensions(imgWidth: 640, imgHeight: 640)
        overlayView.setFrontCameraState(lensType == "front")
        overlayView.setFlipState(flip)

        switch mockupScenario {
        case 1: startScenario_CorrectEverything()
        case 2: startScenario_BowTooHigh()
        case 3: startScenario_WristIssue()
        case 4: startScenario_MultipleIssues()
        default: startScenario_CorrectEverything()
        }
    }

    // ── Scenario 1: everything correct ──────────────────────────────────────
    private func startScenario_CorrectEverything() {
        var result = ReturnBow()
        result.classification = 0
        result.angle          = 0
        result.string = [
            CGPoint(x: 200, y: 180), CGPoint(x: 440, y: 175),
            CGPoint(x: 448, y: 340), CGPoint(x: 205, y: 345),
        ]
        result.bow = [
            CGPoint(x: 210, y: 360), CGPoint(x: 430, y: 355),
            CGPoint(x: 435, y: 520), CGPoint(x: 215, y: 525),
        ]
        overlayView.updateResults(
            results: result, hands: nil, pose: nil,
            handDetection: "Prediction: 0 (Confidence: 0.95)",
            poseDetection:  "Prediction: 0 (Confidence: 0.91)"
        )
        // No timer needed — correct state, no hold logic to trigger
    }

    // ── Scenario 2: bow too high ─────────────────────────────────────────────
    private func startScenario_BowTooHigh() {
        var result = ReturnBow()
        result.classification = 2      // CLASS_TOO_HIGH → "Lower the bow"
        result.angle          = 1      // ANGLE_WRONG    → "Adjust your bow angle"
        result.string = [
            CGPoint(x: 180, y: 50),  CGPoint(x: 460, y: 45),
            CGPoint(x: 465, y: 180), CGPoint(x: 185, y: 185),
        ]
        result.bow = [
            CGPoint(x: 190, y: 200), CGPoint(x: 450, y: 195),
            CGPoint(x: 455, y: 310), CGPoint(x: 195, y: 315),
        ]
        startContinuousDetection(
            result: result,
            handStr: "Prediction: 0 (Confidence: 0.88)",
            poseStr:  "Prediction: 0 (Confidence: 0.84)"
        )
    }

    // ── Scenario 3: wrist rotation issue ────────────────────────────────────
    private func startScenario_WristIssue() {
        var result = ReturnBow()
        result.classification = 0
        result.angle          = 0
        result.string = nil
        result.bow    = nil
        startContinuousDetection(
            result: result,
            handStr: "Prediction: 1 (Confidence: 0.87)",   // supination
            poseStr:  "Prediction: 2 (Confidence: 0.80)"   // elbow too high
        )
    }

    // ── Scenario 4: multiple issues at once ─────────────────────────────────
    private func startScenario_MultipleIssues() {
        var result = ReturnBow()
        result.classification = 3      // CLASS_TOO_LOW → "Lift the bow"
        result.angle          = 1      // ANGLE_WRONG   → "Adjust your bow angle"
        result.string = nil
        result.bow = [
            CGPoint(x: 100, y: 420), CGPoint(x: 380, y: 415),
            CGPoint(x: 385, y: 580), CGPoint(x: 105, y: 585),
        ]
        startContinuousDetection(
            result: result,
            handStr: "Prediction: 1 (Confidence: 0.92)",   // supination
            poseStr:  "Prediction: 1 (Confidence: 0.78)"   // elbow too low
        )
    }

    // ── Shared continuous-detection timer ────────────────────────────────────
    // OverlayView only shows text after an issue persists for 3 s.
    // We simulate that by calling updateResults every 0.1 s.
    private func startContinuousDetection(
        result: ReturnBow,
        handStr: String,
        poseStr: String
    ) {
        stopMockupTimer()
        overlayView.updateResults(
            results: result, hands: nil, pose: nil,
            handDetection: handStr,
            poseDetection: poseStr
        )
        mockupTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.overlayView.updateResults(
                results: result, hands: nil, pose: nil,
                handDetection: handStr,
                poseDetection: poseStr
            )
        }
    }

    private func stopMockupTimer() {
        mockupTimer?.invalidate()
        mockupTimer = nil
    }


  private func refreshUI() {
    stateLabel.text = "user: \(userId)\nactive: \(cameraActive)  detect: \(detectionEnabled)\nlens: \(lensType)  mirror: \(flip)  maxAngle: \(maxBowAngle)"
    chip.text = cameraActive ? "iOS UI Active" : "iOS UI Idle"
    setNeedsLayout()
  }

  private func buildBackgroundColor() -> UIColor {
    let isFront = lensType == "front"
    let base: UIColor = isFront
      ? UIColor(red: 0.09, green: 0.14, blue: 0.21, alpha: 1.0)
      : UIColor(red: 0.09, green: 0.18, blue: 0.13, alpha: 1.0)
    return cameraActive ? base : base.withAlphaComponent(0.78)
  }
}