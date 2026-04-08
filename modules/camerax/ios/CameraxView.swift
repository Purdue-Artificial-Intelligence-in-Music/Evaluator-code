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

  private var userId: String = "default_user"
  private var cameraActive: Bool = false
  private var detectionEnabled: Bool = false
  private var lensType: String = "front"
  private var flip: Bool = false
  private var maxBowAngle: Int = 20

  required init(appContext: AppContext? = nil) {
    super.init(appContext: appContext)
    clipsToBounds = true

    root.layer.cornerRadius = 12
    root.clipsToBounds = true
    addSubview(root)

    chip.text = "iOS UI Preview"
    chip.font = UIFont.systemFont(ofSize: 12, weight: .semibold)
    chip.textColor = .white
    chip.backgroundColor = UIColor(red: 0.09, green: 0.13, blue: 0.2, alpha: 0.85)
    chip.textAlignment = .center
    chip.layer.cornerRadius = 10
    chip.clipsToBounds = true
    root.addSubview(chip)

    titleLabel.text = "Camera Surface"
    titleLabel.font = UIFont.systemFont(ofSize: 24, weight: .bold)
    titleLabel.textColor = .white
    titleLabel.textAlignment = .center
    root.addSubview(titleLabel)

    subtitleLabel.text = "Native iOS UI is wired. Camera functionality will be added next."
    subtitleLabel.font = UIFont.systemFont(ofSize: 14, weight: .regular)
    subtitleLabel.textColor = UIColor(white: 1.0, alpha: 0.9)
    subtitleLabel.textAlignment = .center
    subtitleLabel.numberOfLines = 0
    root.addSubview(subtitleLabel)

    stateLabel.font = UIFont.monospacedSystemFont(ofSize: 12, weight: .regular)
    stateLabel.textColor = UIColor(white: 1.0, alpha: 0.92)
    stateLabel.textAlignment = .center
    stateLabel.numberOfLines = 0
    root.addSubview(stateLabel)

    refreshUI()
  }

  override func layoutSubviews() {
    root.frame = bounds
    root.layer.backgroundColor = buildBackgroundColor().cgColor

    let inset: CGFloat = 16
    chip.frame = CGRect(x: inset, y: inset, width: 110, height: 28)

    let contentWidth = bounds.width - inset * 2
    let titleHeight: CGFloat = 32
    titleLabel.frame = CGRect(
      x: inset,
      y: bounds.midY - 72,
      width: contentWidth,
      height: titleHeight
    )

    let subtitleHeight: CGFloat = 48
    subtitleLabel.frame = CGRect(
      x: inset,
      y: titleLabel.frame.maxY + 8,
      width: contentWidth,
      height: subtitleHeight
    )

    let stateHeight: CGFloat = 90
    stateLabel.frame = CGRect(
      x: inset,
      y: bounds.height - inset - stateHeight,
      width: contentWidth,
      height: stateHeight
    )
  }

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
    }
    refreshUI()
  }

  func setLensType(_ value: String) {
    lensType = value.lowercased()
    refreshUI()
  }

  func setFlip(_ value: Bool) {
    flip = value
    refreshUI()
  }

  func setMaxBowAngle(_ value: Int) {
    maxBowAngle = value
    refreshUI()
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
