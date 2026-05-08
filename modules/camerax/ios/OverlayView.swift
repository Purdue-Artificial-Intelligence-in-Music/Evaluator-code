import UIKit

// Replace these with real MediaPipe types once the SDK is added to the podspec

struct HandLandmarkerResult {
    var landmarks: [[MockNormalizedLandmark]] = []
}

struct PoseLandmarkerResult {
    var landmarks: [[MockNormalizedLandmark]] = []
}

struct MockNormalizedLandmark {
    var x: Float
    var y: Float
    var z: Float = 0
}

// ReturnBow

struct ReturnBow {
    var classification: Int?
    var angle: Int?
    var string: [CGPoint]?
    var bow: [CGPoint]?
}

// OverlayView

class OverlayView: UIView {

    // Detection Results

    private var results: ReturnBow?
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    private var handLandmarkerResult: HandLandmarkerResult?
    private var poseLandmarkerResult: PoseLandmarkerResult?

    private var handDetect: String = ""
    private var poseDetect: String = ""

    // Layout / Scale

    private var xOffset: Float = 0
    private var yOffset: Float = 0
    private var handsScaleFactor: Float = 0

    // Camera State

    private var isFrontCameraActive: Bool = false
    private var shouldFlipDisplay: Bool = false

    // Issue Tracking

    private var bowMessage: String = ""
    private var angleMessage: String = ""

    private var lastBowIssue: String?
    private var bowIssueStartTime: TimeInterval = 0
    private var bowIssueLastShownTime: TimeInterval = 0
    private var displayBowIssue: String?

    private var lastAngleIssue: String?
    private var angleIssueStartTime: TimeInterval = 0
    private var angleIssueLastShownTime: TimeInterval = 0
    private var displayAngleIssue: String?

    private var lastHandIssue: String?
    private var handIssueStartTime: TimeInterval = 0
    private var handIssueLastShownTime: TimeInterval = 0
    private var displayHandIssue: String?

    private var lastPoseIssue: String?
    private var poseIssueStartTime: TimeInterval = 0
    private var poseIssueLastShownTime: TimeInterval = 0
    private var displayPoseIssue: String?

    private let issueHoldDuration: TimeInterval = 3.0
    private let issueMinDisplayTime: TimeInterval = 1.0

    private var issueFrequency: [String: Int] = [:]
    private let maxLines = 2

    // MARK: - Constants

    static let classNone    = -2
    static let classPartial = -1
    static let classCorrect =  0
    static let classOutside =  1
    static let classTooHigh =  2
    static let classTooLow  =  3
    static let angleRight   =  0
    static let angleWrong   =  1
    static let landmarkStrokeWidth: CGFloat = 8

    // Init

    override init(frame: CGRect) {
        super.init(frame: frame)
        setup()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }

    private func setup() {
        backgroundColor = .clear
        isOpaque = false
    }

    // Public API

    func setFrontCameraState(_ isFront: Bool) {
        isFrontCameraActive = isFront
    }

    func setFlipState(_ flip: Bool) {
        shouldFlipDisplay = flip
        setNeedsDisplay()
    }

    func setImageDimensions(imgWidth: Int, imgHeight: Int) {
        imageWidth  = imgWidth
        imageHeight = imgHeight
        let scaleX = Float(bounds.width)  / Float(imageWidth)
        let scaleY = Float(bounds.height) / Float(imageHeight)
        handsScaleFactor = min(scaleX, scaleY)
    }

    func updateResults(
        results: ReturnBow?,
        hands: HandLandmarkerResult?,
        pose: PoseLandmarkerResult?,
        handDetection: String,
        poseDetection: String
    ) {
        self.results              = results
        self.handLandmarkerResult = hands
        self.poseLandmarkerResult = pose
        self.handDetect           = handDetection
        self.poseDetect           = poseDetection
        setNeedsDisplay()
    }

    func clear() {
        results              = nil
        handLandmarkerResult = nil
        poseLandmarkerResult = nil
        handDetect           = ""
        poseDetect           = ""
        issueFrequency.removeAll()
        setNeedsDisplay()
    }

    // Drawing

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }

        if results == nil && handLandmarkerResult == nil
            && handDetect.isEmpty && poseDetect.isEmpty {
            return
        }

        let shouldMirror = isFrontCameraActive && shouldFlipDisplay
        if shouldMirror {
            ctx.saveGState()
            ctx.translateBy(x: bounds.width, y: 0)
            ctx.scaleBy(x: -1, y: 1)
        }

        let scaleX = bounds.width
        let scaleY = bounds.height
        let now    = Date().timeIntervalSince1970

        // ── Bow / String boxes ──────────────────────────────────────────────
        if results?.classification != OverlayView.classNone {
            let hasIssue = (results?.classification != nil && results?.classification != 0)
                        || (results?.angle == OverlayView.angleWrong)
            let boxColor = hasIssue
                ? UIColor(red: 1.0, green: 0.549, blue: 0, alpha: 1)
                : UIColor.blue

            if let sb = results?.string, sb.count == 4 {
                drawQuad(ctx: ctx, points: sb, scaleX: scaleX, scaleY: scaleY,
                         color: boxColor, lineWidth: 8)
            }
            if let bb = results?.bow, bb.count == 4 {
                drawQuad(ctx: ctx, points: bb, scaleX: scaleX, scaleY: scaleY,
                         color: boxColor, lineWidth: 8)
            }

            let classificationLabels: [Int: String] = [
                0: "", 1: "Keep the bow in zone",
                2: "Lower the bow", 3: "Lift the bow"
            ]
            let angleLabels: [Int: String] = [0: "", 1: "Adjust your bow angle"]

            if let cls = results?.classification {
                bowMessage = classificationLabels[cls] ?? ""
            }
            updateIssueState(
                message: bowMessage,
                last: &lastBowIssue, startTime: &bowIssueStartTime,
                lastShown: &bowIssueLastShownTime, display: &displayBowIssue,
                now: now
            )

            if let ang = results?.angle {
                angleMessage = angleLabels[ang] ?? ""
            }
            updateIssueState(
                message: angleMessage,
                last: &lastAngleIssue, startTime: &angleIssueStartTime,
                lastShown: &angleIssueLastShownTime, display: &displayAngleIssue,
                now: now
            )
        }

        // ── Hand landmarks ──────────────────────────────────────────────────
        if let hands = handLandmarkerResult, !hands.landmarks.isEmpty {
            let handClass = parseClassification(from: handDetect)
            let lineColor = (handClass == 1 || handClass == 2)
                ? UIColor(red: 1.0, green: 0.549, blue: 0, alpha: 1)
                : UIColor.blue

            let landmarks = hands.landmarks[0]
            for conn in handConnections() {
                guard conn.0 < landmarks.count, conn.1 < landmarks.count else { continue }
                let s = landmarks[conn.0]
                let e = landmarks[conn.1]
                let sx = CGFloat(s.x) * CGFloat(imageWidth)  * CGFloat(handsScaleFactor) + CGFloat(xOffset)
                let sy = CGFloat(s.y) * CGFloat(imageHeight) * CGFloat(handsScaleFactor) + CGFloat(yOffset)
                let ex = CGFloat(e.x) * CGFloat(imageWidth)  * CGFloat(handsScaleFactor) + CGFloat(xOffset)
                let ey = CGFloat(e.y) * CGFloat(imageHeight) * CGFloat(handsScaleFactor) + CGFloat(yOffset)
                ctx.setStrokeColor(lineColor.cgColor)
                ctx.setLineWidth(OverlayView.landmarkStrokeWidth)
                ctx.move(to: CGPoint(x: sx, y: sy))
                ctx.addLine(to: CGPoint(x: ex, y: ey))
                ctx.strokePath()
            }
            for lm in landmarks {
                let px = CGFloat(lm.x) * CGFloat(imageWidth)  * CGFloat(handsScaleFactor) + CGFloat(xOffset)
                let py = CGFloat(lm.y) * CGFloat(imageHeight) * CGFloat(handsScaleFactor) + CGFloat(yOffset)
                ctx.setFillColor(UIColor.yellow.cgColor)
                ctx.fillEllipse(in: CGRect(x: px - 4, y: py - 4, width: 8, height: 8))
            }
        }

        // ── Restore mirror before drawing text ──────────────────────────────
        if shouldMirror { ctx.restoreGState() }

        // ── Hand / Pose text messages ───────────────────────────────────────
        let handClass = parseClassification(from: handDetect)
        let poseClass = parseClassification(from: poseDetect)

        if handClass == 1 || handClass == 2 {
            let msg = handClass == 1 ? "Pronate your wrist more" : "Supinate your wrist more"
            updateIssueState(
                message: msg,
                last: &lastHandIssue, startTime: &handIssueStartTime,
                lastShown: &handIssueLastShownTime, display: &displayHandIssue,
                now: now
            )
        } else {
            clearIssueIfExpired(
                display: &displayHandIssue, last: &lastHandIssue,
                startTime: &handIssueStartTime, lastShown: &handIssueLastShownTime,
                now: now
            )
        }

        if poseClass == 1 || poseClass == 2 {
            let msg = poseClass == 1 ? "Raise your elbow a bit" : "Lower your elbow a bit"
            updateIssueState(
                message: msg,
                last: &lastPoseIssue, startTime: &poseIssueStartTime,
                lastShown: &poseIssueLastShownTime, display: &displayPoseIssue,
                now: now
            )
        } else {
            clearIssueIfExpired(
                display: &displayPoseIssue, last: &lastPoseIssue,
                startTime: &poseIssueStartTime, lastShown: &poseIssueLastShownTime,
                now: now
            )
        }

        // ── Render top labels ───────────────────────────────────────────────
        var activeIssues: [String] = []
        if let b = displayBowIssue   { activeIssues.append(b) }
        if let a = displayAngleIssue { activeIssues.append(a) }
        if let h = displayHandIssue  { activeIssues.append(h) }
        if let p = displayPoseIssue  { activeIssues.append(p) }

        if !activeIssues.isEmpty {
            let sorted = activeIssues.sorted { (issueFrequency[$0] ?? 0) > (issueFrequency[$1] ?? 0) }
            var textY: CGFloat = 120
            for message in sorted.prefix(maxLines) {
                drawLabel(ctx: ctx, text: message, centerX: bounds.width / 2,
                          y: textY, padding: 16)
                textY += labelLineHeight() + 60
            }
        }
    }

    // Drawing Helpers

    private func drawQuad(ctx: CGContext, points: [CGPoint],
                          scaleX: CGFloat, scaleY: CGFloat,
                          color: UIColor, lineWidth: CGFloat) {
        let scaled = points.map {
            CGPoint(x: ($0.x / 640) * scaleX, y: ($0.y / 640) * scaleY)
        }
        ctx.setStrokeColor(color.cgColor)
        ctx.setLineWidth(lineWidth)
        ctx.move(to: scaled[0])
        for i in 1..<scaled.count { ctx.addLine(to: scaled[i]) }
        ctx.addLine(to: scaled[0])
        ctx.strokePath()
    }

    private func drawLabel(ctx: CGContext, text: String,
                           centerX: CGFloat, y: CGFloat, padding: CGFloat) {
        let attrs: [NSAttributedString.Key: Any] = [
            .font: UIFont.boldSystemFont(ofSize: 28),
            .foregroundColor: UIColor.black
        ]
        let size = text.size(withAttributes: attrs)
        let bgRect = CGRect(
            x: centerX - size.width / 2 - padding,
            y: y - size.height - padding,
            width:  size.width  + padding * 2,
            height: size.height + padding * 2
        )
        ctx.setFillColor(UIColor(white: 1, alpha: 0.7).cgColor)
        ctx.fill(bgRect)
        text.draw(at: CGPoint(x: centerX - size.width / 2, y: y - size.height),
                  withAttributes: attrs)
    }

    private func labelLineHeight() -> CGFloat {
        "A".size(withAttributes: [.font: UIFont.boldSystemFont(ofSize: 28)]).height
    }

    private func updateIssueState(
        message: String,
        last: inout String?, startTime: inout TimeInterval,
        lastShown: inout TimeInterval, display: inout String?,
        now: TimeInterval
    ) {
        if !message.isEmpty {
            recordIssue(message)
            if message == last {
                if now - startTime >= issueHoldDuration {
                    display   = message
                    lastShown = now
                }
            } else {
                last      = message
                startTime = now
                display   = nil
            }
        } else {
            clearIssueIfExpired(
                display: &display, last: &last,
                startTime: &startTime, lastShown: &lastShown, now: now
            )
        }
    }

    private func clearIssueIfExpired(
        display: inout String?, last: inout String?,
        startTime: inout TimeInterval, lastShown: inout TimeInterval,
        now: TimeInterval
    ) {
        if display != nil && now - lastShown >= issueMinDisplayTime {
            display   = nil
            last      = nil
            startTime = 0
            lastShown = 0
        }
    }

    private func recordIssue(_ message: String) {
        guard !message.isEmpty else { return }
        issueFrequency[message, default: 0] += 1
    }

    // Classification Parsing

    private func parseClassification(from text: String) -> Int {
        let pattern = #"Prediction: (\d+) \(Confidence: [\d.]+\)"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: text,
                                           range: NSRange(text.startIndex..., in: text)),
              let range = Range(match.range(at: 1), in: text),
              let cls   = Int(text[range])
        else { return -1 }
        return cls
    }

    // Hand Connections

    private func handConnections() -> [(Int, Int)] {
        return [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(0,17),(17,18),(18,19),(19,20)
        ]
    }
}