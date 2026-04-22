import Foundation
import UIKit
import AVFoundation
import CoreGraphics
import MediaPipeTasksVision
import TensorFlowLite

protocol CombinedLandmarkerListener: AnyObject {
    func onError(_ error: String, errorCode: Int)
    func onResults(_ resultBundle: CombinedLandmarkerHelper.CombinedResultBundle)
}

extension CombinedLandmarkerListener {
    func onError(_ error: String, errorCode: Int = CombinedLandmarkerHelper.otherError) {
        onError(error, errorCode: errorCode)
    }
}

private extension Array where Element == Float {
    var data: Data {
        withUnsafeBufferPointer { Data(buffer: $0) }
    }
}

private extension Data {
    func toFloatArray() -> [Float] {
        let count = self.count / MemoryLayout<Float>.stride
        return withUnsafeBytes { rawBuffer in
            let buffer = rawBuffer.bindMemory(to: Float.self)
            return Array(buffer.prefix(count))
        }
    }
}

final class CombinedLandmarkerHelper: NSObject {

    enum ComputeDelegate {
        case cpu
        case gpu

        var mediaPipeDelegate: BaseOptions.Delegate {
            switch self {
            case .cpu:
                return .CPU
            case .gpu:
                return .GPU
            }
        }
    }

    struct CombinedResultBundle {
        let handResults: [HandLandmarkerResult]
        let poseResults: [PoseLandmarkerResult]
        let inferenceTime: Double
        let inputImageHeight: Int
        let inputImageWidth: Int
        let handCoordinates: [Float]?
        let poseCoordinates: [Float]?
        let handDetection: String
        let poseDetection: String
        let targetHandIndex: Int
    }

    static let tag = "CombinedLandmarkerHelper"

    static let defaultHandDetectionConfidence: Float = 0.7
    static let defaultHandTrackingConfidence: Float = 0.7
    static let defaultHandPresenceConfidence: Float = 0.5
    static let defaultNumHands: Int = 2

    static let defaultPoseDetectionConfidence: Float = 0.7
    static let defaultPoseTrackingConfidence: Float = 0.7
    static let defaultPosePresenceConfidence: Float = 0.5
    static let defaultNumPoses: Int = 1

    static let otherError = 0
    static let gpuError = 1

    private static let handLandmarkerModelName = "hand_landmarker"
    private static let handLandmarkerModelType = "task"
    private static let poseLandmarkerModelName = "pose_landmarker_full"
    private static let poseLandmarkerModelType = "task"
    private static let handClassifierModelName = "2_19_hands"
    private static let handClassifierModelType = "tflite"
    private static let poseClassifierModelName = "keypoint_classifier (1)"
    private static let poseClassifierModelType = "tflite"

    private static let handConnections: [(Int, Int)] = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    weak var combinedLandmarkerHelperListener: CombinedLandmarkerListener?

    var minHandDetectionConfidence: Float
    var minHandTrackingConfidence: Float
    var minHandPresenceConfidence: Float
    var maxNumHands: Int
    var currentDelegate: ComputeDelegate
    var runningMode: RunningMode

    var handLandmarker: HandLandmarker?
    var poseLandmarker: PoseLandmarker?

    private var handTFLite: Interpreter?
    private var poseTFLite: Interpreter?

    private let stateLock = NSLock()
    private var isClosed = false

    private var latestHandResult: HandLandmarkerResult?
    private var latestPoseResult: PoseLandmarkerResult?
    private var latestFrameTime: Int = 0
    private var latestInputImageWidth: Int = 0
    private var latestInputImageHeight: Int = 0

    private var isFrontCameraActive = false

    init(
        minHandDetectionConfidence: Float = CombinedLandmarkerHelper.defaultHandDetectionConfidence,
        minHandTrackingConfidence: Float = CombinedLandmarkerHelper.defaultHandTrackingConfidence,
        minHandPresenceConfidence: Float = CombinedLandmarkerHelper.defaultHandPresenceConfidence,
        maxNumHands: Int = CombinedLandmarkerHelper.defaultNumHands,
        currentDelegate: ComputeDelegate = .cpu,
        runningMode: RunningMode = .image,
        combinedLandmarkerHelperListener: CombinedLandmarkerListener? = nil
    ) {
        self.minHandDetectionConfidence = minHandDetectionConfidence
        self.minHandTrackingConfidence = minHandTrackingConfidence
        self.minHandPresenceConfidence = minHandPresenceConfidence
        self.maxNumHands = maxNumHands
        self.currentDelegate = currentDelegate
        self.runningMode = runningMode
        self.combinedLandmarkerHelperListener = combinedLandmarkerHelperListener
        super.init()
        setupHandLandmarker()
        setupPoseLandmarker()
    }

    // MARK: - Simple posture checks translated from Python

    private func classifyStraightBack(landmarks: [NormalizedLandmark]) -> Int {
        let angleThreshold = 15.0
        guard landmarks.count > 12 else { return 1 }

        let rightShoulder = landmarks[11]
        let leftShoulder = landmarks[12]

        let dx = Double(rightShoulder.x - leftShoulder.x)
        let dy = Double(rightShoulder.y - leftShoulder.y)

        let angleRadians = atan2(dy, dx)
        let angleDegrees = abs(angleRadians * 180.0 / .pi)

        return angleDegrees > angleThreshold ? 0 : 1
    }

    private func classifyStraightNeck(landmarks: [NormalizedLandmark]) -> Int {
        let angleThreshold = 15.0
        guard landmarks.count > 8 else { return 1 }

        let rightEar = landmarks[7]
        let leftEar = landmarks[8]

        let dx = Double(rightEar.x - leftEar.x)
        let dy = Double(rightEar.y - leftEar.y)

        let angleRadians = atan2(dy, dx)
        let angleDegrees = abs(angleRadians * 180.0 / .pi)

        return angleDegrees > angleThreshold ? 0 : 1
    }

    // MARK: - Lifecycle

    func clearLandmarkers() {
        stateLock.lock()
        defer { stateLock.unlock() }

        isClosed = true
        handLandmarker = nil
        poseLandmarker = nil
        handTFLite = nil
        poseTFLite = nil
        latestHandResult = nil
        latestPoseResult = nil
    }

    func isClose() -> Bool {
        handLandmarker == nil && poseLandmarker == nil
    }

    // MARK: - Setup

    func setupHandLandmarker() {
        guard let modelPath = Bundle.main.path(
            forResource: Self.handLandmarkerModelName,
            ofType: Self.handLandmarkerModelType
        ) else {
            combinedLandmarkerHelperListener?.onError("Hand landmarker model not found in bundle.")
            return
        }

        if runningMode == .liveStream && combinedLandmarkerHelperListener == nil {
            fatalError("combinedLandmarkerHelperListener must be set when runningMode is .liveStream")
        }

        do {
            let options = HandLandmarkerOptions()
            options.baseOptions.modelAssetPath = modelPath
            options.baseOptions.delegate = currentDelegate.mediaPipeDelegate
            options.minHandDetectionConfidence = minHandDetectionConfidence
            options.minTrackingConfidence = minHandTrackingConfidence
            options.minHandPresenceConfidence = minHandPresenceConfidence
            options.numHands = maxNumHands
            options.runningMode = runningMode

            if runningMode == .liveStream {
                options.handLandmarkerLiveStreamDelegate = self
            }

            handLandmarker = try HandLandmarker(options: options)
        } catch {
            combinedLandmarkerHelperListener?.onError(
                "Hand Landmarker failed to initialize: \(error.localizedDescription)",
                errorCode: currentDelegate == .gpu ? Self.gpuError : Self.otherError
            )
        }
    }

    func setupPoseLandmarker() {
        guard let modelPath = Bundle.main.path(
            forResource: Self.poseLandmarkerModelName,
            ofType: Self.poseLandmarkerModelType
        ) else {
            combinedLandmarkerHelperListener?.onError("Pose landmarker model not found in bundle.")
            return
        }

        if runningMode == .liveStream && combinedLandmarkerHelperListener == nil {
            fatalError("combinedLandmarkerHelperListener must be set when runningMode is .liveStream")
        }

        do {
            let options = PoseLandmarkerOptions()
            options.baseOptions.modelAssetPath = modelPath
            options.baseOptions.delegate = currentDelegate.mediaPipeDelegate
            options.runningMode = runningMode
            options.numPoses = Self.defaultNumPoses
            options.minPoseDetectionConfidence = Self.defaultPoseDetectionConfidence
            options.minPosePresenceConfidence = Self.defaultPosePresenceConfidence
            options.minTrackingConfidence = Self.defaultPoseTrackingConfidence

            if runningMode == .liveStream {
                options.poseLandmarkerLiveStreamDelegate = self
            }

            poseLandmarker = try PoseLandmarker(options: options)
        } catch {
            combinedLandmarkerHelperListener?.onError(
                "Pose Landmarker failed to initialize: \(error.localizedDescription)",
                errorCode: currentDelegate == .gpu ? Self.gpuError : Self.otherError
            )
        }
    }

    // MARK: - Detection entry points

    func detectLiveStream(
        sampleBuffer: CMSampleBuffer,
        orientation: UIImage.Orientation,
        isFrontCamera: Bool
    ) {
        stateLock.lock()
        defer { stateLock.unlock() }

        if isClosed {
            return
        }

        guard runningMode == .liveStream else {
            combinedLandmarkerHelperListener?.onError(
                "Attempting to call detectLiveStream while not using RunningMode.liveStream"
            )
            return
        }

        isFrontCameraActive = isFrontCamera
        latestFrameTime = Self.currentTimestampMs()

        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            latestInputImageWidth = CVPixelBufferGetWidth(pixelBuffer)
            latestInputImageHeight = CVPixelBufferGetHeight(pixelBuffer)
        }

        do {
            let mpImage = try MPImage(sampleBuffer: sampleBuffer, orientation: orientation)
            try handLandmarker?.detectAsync(image: mpImage, timestampInMilliseconds: latestFrameTime)
            try poseLandmarker?.detectAsync(image: mpImage, timestampInMilliseconds: latestFrameTime)
        } catch {
            combinedLandmarkerHelperListener?.onError(error.localizedDescription)
        }
    }

    func detectBitmap(_ image: UIImage, frontCamera: Bool) {
        stateLock.lock()
        defer { stateLock.unlock() }

        if isClosed {
            return
        }

        isFrontCameraActive = frontCamera
        latestFrameTime = Self.currentTimestampMs()

        let size = Self.pixelSize(for: image)
        latestInputImageWidth = Int(size.width)
        latestInputImageHeight = Int(size.height)

        do {
            let mpImage = try MPImage(uiImage: image)
            try handLandmarker?.detectAsync(image: mpImage, timestampInMilliseconds: latestFrameTime)
            try poseLandmarker?.detectAsync(image: mpImage, timestampInMilliseconds: latestFrameTime)
        } catch {
            combinedLandmarkerHelperListener?.onError(error.localizedDescription)
        }
    }

    func detectVideoFrame(frame: UIImage?, timestampMs: Int) -> CombinedResultBundle? {
        guard runningMode == .video else {
            combinedLandmarkerHelperListener?.onError(
                "Attempting to call detectVideoFrame while not using RunningMode.video"
            )
            return nil
        }

        guard let frame else { return nil }

        let size = Self.pixelSize(for: frame)
        let startTime = Date()

        do {
            let mpImage = try MPImage(uiImage: frame)
            let handResult = try handLandmarker?.detect(videoFrame: mpImage, timestampInMilliseconds: timestampMs)
            let poseResult = try poseLandmarker?.detect(videoFrame: mpImage, timestampInMilliseconds: timestampMs)

            return buildCombinedResultBundle(
                handResult: handResult,
                poseResult: poseResult,
                inputImageWidth: Int(size.width),
                inputImageHeight: Int(size.height),
                inferenceTime: Date().timeIntervalSince(startTime) * 1000.0
            )
        } catch {
            combinedLandmarkerHelperListener?.onError(error.localizedDescription)
            return nil
        }
    }

    func detectAndDrawVideoFrame(frame: UIImage?, timestampMs: Int) -> (CombinedResultBundle?, UIImage?) {
        guard let frame else {
            return (nil, nil)
        }

        guard let resultBundle = detectVideoFrame(frame: frame, timestampMs: timestampMs) else {
            return (nil, nil)
        }

        let annotated = drawMediaPipeAnnotations(on: frame, result: resultBundle)
        return (resultBundle, annotated)
    }

    func detectImage(_ image: UIImage) -> CombinedResultBundle? {
        guard runningMode == .image else {
            combinedLandmarkerHelperListener?.onError(
                "Attempting to call detectImage while not using RunningMode.image"
            )
            return nil
        }

        let size = Self.pixelSize(for: image)
        let startTime = Date()

        do {
            let mpImage = try MPImage(uiImage: image)
            let handResult = try handLandmarker?.detect(image: mpImage)
            let poseResult = try poseLandmarker?.detect(image: mpImage)

            var handCoordinates: [Float]? = nil
            var handPrediction = "No hand detected"

            if let handResult, !handResult.landmarks.isEmpty {
                let leftHandIndex = handResult.handedness.firstIndex { categories in
                    let label = categories.first?.displayName ?? categories.first?.categoryName ?? ""
                    return label.caseInsensitiveCompare("Left") == .orderedSame
                } ?? 0

                handCoordinates = extractHandCoordinates(result: handResult, handIndex: leftHandIndex)
                if let handCoordinates {
                    handPrediction = runTFLiteInference(inputData: handCoordinates)
                }
            }

            var poseCoordinates: [Float]? = nil
            var posePrediction = "No pose detected"
            if let poseResult, !poseResult.landmarks.isEmpty {
                poseCoordinates = extractPoseCoordinates(result: poseResult)
                if let poseCoordinates {
                    posePrediction = runTFLitePoseInference(inputData: poseCoordinates)
                }
            }

            if handResult == nil && poseResult == nil {
                return nil
            }

            return CombinedResultBundle(
                handResults: handResult.map { [$0] } ?? [],
                poseResults: poseResult.map { [$0] } ?? [],
                inferenceTime: Date().timeIntervalSince(startTime) * 1000.0,
                inputImageHeight: Int(size.height),
                inputImageWidth: Int(size.width),
                handCoordinates: handCoordinates,
                poseCoordinates: poseCoordinates,
                handDetection: handPrediction,
                poseDetection: posePrediction,
                targetHandIndex: handCoordinates == nil ? -1 : 0
            )
        } catch {
            combinedLandmarkerHelperListener?.onError(error.localizedDescription)
            return nil
        }
    }

    // MARK: - Shared result build logic

    private func buildCombinedResultBundle(
        handResult: HandLandmarkerResult?,
        poseResult: PoseLandmarkerResult?,
        inputImageWidth: Int,
        inputImageHeight: Int,
        inferenceTime: Double
    ) -> CombinedResultBundle {
        var handCoords: [Float]? = nil
        var handPrediction = "No hand detected"
        var targetHandIndex = -1

        if let handResult,
           let poseResult,
           !handResult.landmarks.isEmpty,
           !poseResult.landmarks.isEmpty,
           poseResult.landmarks[0].count > 16 {

            let poseLandmarks = poseResult.landmarks[0]
            let handIndex = isFrontCameraActive ? 15 : 16
            let poseHandX = poseLandmarks[handIndex].x
            let poseHandY = poseLandmarks[handIndex].y

            var bestDistance = Float.greatestFiniteMagnitude
            let threshold: Float = 0.1

            for (index, handLandmarks) in handResult.landmarks.enumerated() {
                guard let wrist = handLandmarks.first else { continue }
                let dx = wrist.x - poseHandX
                let dy = wrist.y - poseHandY
                let distance = sqrt(dx * dx + dy * dy)

                if distance < threshold && distance < bestDistance {
                    bestDistance = distance
                    targetHandIndex = index
                }
            }

            if targetHandIndex != -1 {
                handCoords = extractHandCoordinates(result: handResult, handIndex: targetHandIndex)
                if let handCoords {
                    handPrediction = runTFLiteInference(inputData: handCoords)
                }
            }
        }

        var poseCoords: [Float]? = nil
        var posePrediction = "No pose detected"
        if let poseResult, !poseResult.landmarks.isEmpty {
            poseCoords = extractPoseCoordinates(result: poseResult)
            if let poseCoords {
                posePrediction = runTFLitePoseInference(inputData: poseCoords)
            }
        }

        return CombinedResultBundle(
            handResults: handResult.map { [$0] } ?? [],
            poseResults: poseResult.map { [$0] } ?? [],
            inferenceTime: inferenceTime,
            inputImageHeight: inputImageHeight,
            inputImageWidth: inputImageWidth,
            handCoordinates: handCoords,
            poseCoordinates: poseCoords,
            handDetection: handPrediction,
            poseDetection: posePrediction,
            targetHandIndex: targetHandIndex
        )
    }

    // MARK: - Live stream state merge

    private func maybeSendCombinedResult() {
        stateLock.lock()
        defer { stateLock.unlock() }

        guard !isClosed else { return }
        guard let latestHandResult, let latestPoseResult else { return }

        let combinedResult = buildCombinedResultBundle(
            handResult: latestHandResult,
            poseResult: latestPoseResult,
            inputImageWidth: latestInputImageWidth,
            inputImageHeight: latestInputImageHeight,
            inferenceTime: Double(Self.currentTimestampMs() - latestFrameTime)
        )

        combinedLandmarkerHelperListener?.onResults(combinedResult)

        self.latestHandResult = nil
        self.latestPoseResult = nil
    }

    // MARK: - Feature extraction

    private func extractHandCoordinates(result: HandLandmarkerResult, handIndex: Int) -> [Float]? {
        guard handIndex >= 0, handIndex < result.landmarks.count else { return nil }

        let handLandmarks = result.landmarks[handIndex]
        guard !handLandmarks.isEmpty else { return nil }

        let originX = handLandmarks[0].x
        let originY = handLandmarks[0].y

        var relativeCoordinates = Array(repeating: Float(0), count: 42)
        var maxAbsValue: Float = 0.0

        for (index, landmark) in handLandmarks.enumerated() {
            var relativeX = landmark.x - originX
            let relativeY = landmark.y - originY

            if !isFrontCameraActive {
                relativeX *= -1
            }

            let base = index * 2
            relativeCoordinates[base] = relativeX
            relativeCoordinates[base + 1] = relativeY

            maxAbsValue = max(maxAbsValue, abs(relativeX))
            maxAbsValue = max(maxAbsValue, abs(relativeY))
        }

        guard maxAbsValue > 0 else { return relativeCoordinates }
        return relativeCoordinates.map { $0 / maxAbsValue }
    }

    private func extractPoseCoordinates(result: PoseLandmarkerResult) -> [Float] {
        var coords = Array(repeating: Float(0), count: 9)
        guard !result.landmarks.isEmpty, result.landmarks[0].count > 19 else {
            return coords
        }

        let landmarks = result.landmarks[0]
        let shoulderIndex = isFrontCameraActive ? 11 : 12
        let elbowIndex = isFrontCameraActive ? 13 : 14
        let handIndex = isFrontCameraActive ? 15 : 16

        let shoulderX = isFrontCameraActive ? landmarks[shoulderIndex].x : 1.0 - landmarks[shoulderIndex].x
        let shoulderY = landmarks[shoulderIndex].y
        let shoulderZ = landmarks[shoulderIndex].z

        let elbowX = isFrontCameraActive ? landmarks[elbowIndex].x : 1.0 - landmarks[elbowIndex].x
        let elbowY = landmarks[elbowIndex].y
        let elbowZ = landmarks[elbowIndex].z

        let handX = isFrontCameraActive ? landmarks[handIndex].x : 1.0 - landmarks[handIndex].x
        let handY = landmarks[handIndex].y
        let handZ = landmarks[handIndex].z

        let shoulderElbowVector: [Float] = [
            shoulderX - elbowX,
            shoulderY - elbowY,
            shoulderZ - elbowZ
        ]

        let handElbowVector: [Float] = [
            handX - elbowX,
            handY - elbowY,
            handZ - elbowZ
        ]

        let shoulderElbowDistance = sqrt(
            shoulderElbowVector[0] * shoulderElbowVector[0] +
            shoulderElbowVector[1] * shoulderElbowVector[1] +
            shoulderElbowVector[2] * shoulderElbowVector[2]
        )

        let handElbowDistance = sqrt(
            handElbowVector[0] * handElbowVector[0] +
            handElbowVector[1] * handElbowVector[1] +
            handElbowVector[2] * handElbowVector[2]
        )

        guard shoulderElbowDistance > 0, handElbowDistance > 0 else {
            return coords
        }

        let dotProduct =
            shoulderElbowVector[0] * handElbowVector[0] +
            shoulderElbowVector[1] * handElbowVector[1] +
            shoulderElbowVector[2] * handElbowVector[2]

        let cosineTheta = max(-1.0, min(1.0, dotProduct / (shoulderElbowDistance * handElbowDistance)))
        let theta = acos(cosineTheta)

        coords[0] = shoulderElbowVector[0] / shoulderElbowDistance
        coords[1] = shoulderElbowVector[1] / shoulderElbowDistance
        coords[2] = shoulderElbowVector[2] / shoulderElbowDistance
        coords[3] = handElbowVector[0] / handElbowDistance
        coords[4] = handElbowVector[1] / handElbowDistance
        coords[5] = handElbowVector[2] / handElbowDistance
        coords[6] = theta
        coords[7] = shoulderElbowDistance
        coords[8] = handElbowDistance

        return coords
    }

    // MARK: - TFLite inference

    private func runTFLitePoseInference(inputData: [Float]) -> String {
        do {
            if poseTFLite == nil {
                poseTFLite = try Self.createInterpreterWithFallbacks(
                    modelName: Self.poseClassifierModelName,
                    modelType: Self.poseClassifierModelType
                )
            }

            guard let interpreter = poseTFLite else {
                return "Error: Pose TFLite not initialized."
            }

            let inputTensor = try interpreter.input(at: 0)
            let expectedFeatures = inputTensor.shape.dimensions.last ?? inputData.count
            var normalizedInput = Array(repeating: Float(0), count: expectedFeatures)
            for index in 0..<min(expectedFeatures, inputData.count) {
                normalizedInput[index] = inputData[index]
            }

            try interpreter.copy(normalizedInput.data, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let results = outputTensor.data.toFloatArray()
            guard !results.isEmpty else {
                return "No pose output"
            }

            let thresholdClass2: Float = 0.975
            let thresholdGeneral: Float = 0.0

            var top1Index = -1
            var top2Index = -1
            var top1Score = -Float.greatestFiniteMagnitude
            var top2Score = -Float.greatestFiniteMagnitude

            for (index, score) in results.enumerated() {
                if score > top1Score {
                    top2Score = top1Score
                    top2Index = top1Index
                    top1Score = score
                    top1Index = index
                } else if score > top2Score {
                    top2Score = score
                    top2Index = index
                }
            }

            var finalIndex = top1Index
            var finalScore = top1Score

            if finalScore < thresholdGeneral {
                finalIndex = 0
                finalScore = results.first ?? 0.0
            }

            if top1Index == 2 && top1Score < thresholdClass2 {
                finalIndex = top2Index == -1 ? 0 : top2Index
                finalScore = top2Index == -1 ? (results.first ?? 0.0) : top2Score
            }

            return String(format: "Prediction: %d (Confidence: %.2f)", finalIndex, finalScore)
        } catch {
            return "Error: \(error.localizedDescription)"
        }
    }

    private func runTFLiteInference(inputData: [Float]) -> String {
        do {
            if handTFLite == nil {
                handTFLite = try Self.createInterpreterWithFallbacks(
                    modelName: Self.handClassifierModelName,
                    modelType: Self.handClassifierModelType
                )
            }

            guard let interpreter = handTFLite else {
                return "Error: Hand TFLite not initialized."
            }

            let inputTensor = try interpreter.input(at: 0)
            let expectedFeatures = inputTensor.shape.dimensions.last ?? inputData.count
            var normalizedInput = Array(repeating: Float(0), count: expectedFeatures)
            for index in 0..<min(expectedFeatures, inputData.count) {
                normalizedInput[index] = inputData[index]
            }

            try interpreter.copy(normalizedInput.data, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            var results = outputTensor.data.toFloatArray()
            guard !results.isEmpty else {
                return "No hand output"
            }

            let supinationIndex = 1
            if results.indices.contains(supinationIndex) {
                results[supinationIndex] *= 0.7
            }

            guard let maxIndex = results.indices.max(by: { results[$0] < results[$1] }) else {
                return "No hand output"
            }
            let confidence = results[maxIndex]

            if maxIndex == supinationIndex && confidence < 0.60 {
                return String(format: "Prediction: %d (Confidence: %.2f)", 0, confidence)
            }

            return String(format: "Prediction: %d (Confidence: %.2f)", maxIndex, confidence)
        } catch {
            return "Error: \(error.localizedDescription)"
        }
    }

    private static func createInterpreterWithFallbacks(modelName: String, modelType: String) throws -> Interpreter {
        guard let modelPath = Bundle.main.path(forResource: modelName, ofType: modelType) else {
            throw NSError(
                domain: Self.tag,
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model file \(modelName).\(modelType) not found in bundle"]
            )
        }

        var options = Interpreter.Options()
        options.threadCount = 4

        if let coreMLDelegate = CoreMLDelegate() {
            do {
                let interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: [coreMLDelegate])
                try interpreter.allocateTensors()
                return interpreter
            } catch {
                // fall through
            }
        }

        do {
            let metalDelegate = MetalDelegate()
            let interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: [metalDelegate])
            try interpreter.allocateTensors()
            return interpreter
        } catch {
            let interpreter = try Interpreter(modelPath: modelPath, options: options)
            try interpreter.allocateTensors()
            return interpreter
        }
    }

    // MARK: - Drawing

    private func drawMediaPipeAnnotations(on image: UIImage, result: CombinedResultBundle) -> UIImage {
        let imageSize = CGSize(width: CGFloat(result.inputImageWidth), height: CGFloat(result.inputImageHeight))
        let renderer = UIGraphicsImageRenderer(size: imageSize)

        return renderer.image { rendererContext in
            image.draw(in: CGRect(origin: .zero, size: imageSize))
            let context = rendererContext.cgContext

            let centerX = imageSize.width / 2.0
            var currentY: CGFloat = 120.0
            let lineSpacing: CGFloat = 60.0
            let padding: CGFloat = 16.0

            if let poseResult = result.poseResults.first, !poseResult.landmarks.isEmpty {
                let poseLandmarks = poseResult.landmarks[0]
                let backStatus = classifyStraightBack(landmarks: poseLandmarks)
                let neckStatus = classifyStraightNeck(landmarks: poseLandmarks)
                print("Posture Check -> Back: \(backStatus == 1 ? "GOOD" : "BAD") (\(backStatus)) | Neck: \(neckStatus == 1 ? "GOOD" : "BAD") (\(neckStatus))")
            }

            let handClass = Self.extractPredictedClass(from: result.handDetection)
            let poseClass = Self.extractPredictedClass(from: result.poseDetection)

            let handHasIssue = handClass == 1 || handClass == 2
            let poseHasIssue = poseClass == 1 || poseClass == 2

            if let handClass {
                let handMessage: String
                switch handClass {
                case 1:
                    handMessage = "Pronate your wrist more"
                case 2:
                    handMessage = "Supinate your wrist more"
                default:
                    handMessage = ""
                }

                if !handMessage.isEmpty {
                    currentY = Self.drawCenteredLabel(
                        handMessage,
                        centerX: centerX,
                        y: currentY,
                        padding: padding,
                        in: context
                    ) + lineSpacing
                }
            }

            if let poseClass {
                let poseMessage: String
                switch poseClass {
                case 1:
                    poseMessage = "Raise your elbow a bit"
                case 2:
                    poseMessage = "Lower your elbow a bit"
                default:
                    poseMessage = ""
                }

                if !poseMessage.isEmpty {
                    _ = Self.drawCenteredLabel(
                        poseMessage,
                        centerX: centerX,
                        y: currentY,
                        padding: padding,
                        in: context
                    )
                }
            }

            if let handResult = result.handResults.first,
               result.targetHandIndex >= 0,
               result.targetHandIndex < handResult.landmarks.count {
                let targetLandmarks = handResult.landmarks[result.targetHandIndex]

                let lineColor = handHasIssue ? UIColor.orange : UIColor.systemBlue
                let pointColor = handHasIssue ? UIColor(red: 1.0, green: 0.71, blue: 0.2, alpha: 1.0) : UIColor.cyan

                context.setLineWidth(4.0)
                context.setStrokeColor(lineColor.cgColor)

                for (startIndex, endIndex) in Self.handConnections {
                    guard startIndex < targetLandmarks.count, endIndex < targetLandmarks.count else { continue }
                    let start = CGPoint(
                        x: CGFloat(targetLandmarks[startIndex].x) * imageSize.width,
                        y: CGFloat(targetLandmarks[startIndex].y) * imageSize.height
                    )
                    let end = CGPoint(
                        x: CGFloat(targetLandmarks[endIndex].x) * imageSize.width,
                        y: CGFloat(targetLandmarks[endIndex].y) * imageSize.height
                    )
                    context.move(to: start)
                    context.addLine(to: end)
                    context.strokePath()
                }

                context.setFillColor(pointColor.cgColor)
                for landmark in targetLandmarks {
                    let point = CGPoint(
                        x: CGFloat(landmark.x) * imageSize.width,
                        y: CGFloat(landmark.y) * imageSize.height
                    )
                    let radius: CGFloat = 5.0
                    context.fillEllipse(in: CGRect(
                        x: point.x - radius,
                        y: point.y - radius,
                        width: radius * 2.0,
                        height: radius * 2.0
                    ))
                }
            }

            _ = poseHasIssue
        }
    }

    private static func drawCenteredLabel(
        _ message: String,
        centerX: CGFloat,
        y: CGFloat,
        padding: CGFloat,
        in context: CGContext
    ) -> CGFloat {
        let font = UIFont.boldSystemFont(ofSize: 28)
        let textColor = UIColor.black
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: textColor
        ]

        let textSize = (message as NSString).size(withAttributes: attributes)
        let rect = CGRect(
            x: centerX - textSize.width / 2.0 - padding,
            y: y - textSize.height + 4.0 - padding,
            width: textSize.width + 2.0 * padding,
            height: textSize.height + 2.0 * padding
        )

        context.setFillColor(UIColor.white.withAlphaComponent(0.72).cgColor)
        let path = UIBezierPath(roundedRect: rect, cornerRadius: 10.0)
        context.addPath(path.cgPath)
        context.fillPath()

        let textRect = CGRect(
            x: centerX - textSize.width / 2.0,
            y: y - textSize.height,
            width: textSize.width,
            height: textSize.height
        )
        (message as NSString).draw(in: textRect, withAttributes: attributes)

        return y + textSize.height
    }

    private static func extractPredictedClass(from predictionText: String) -> Int? {
        let pattern = #"Prediction:\s*(\d+)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(predictionText.startIndex..<predictionText.endIndex, in: predictionText)
        guard let match = regex.firstMatch(in: predictionText, range: range),
              let classRange = Range(match.range(at: 1), in: predictionText) else {
            return nil
        }
        return Int(predictionText[classRange])
    }

    // MARK: - Small utilities

    private static func currentTimestampMs() -> Int {
        Int(Date().timeIntervalSince1970 * 1000.0)
    }

    private static func pixelSize(for image: UIImage) -> CGSize {
        if let cgImage = image.cgImage {
            return CGSize(width: cgImage.width, height: cgImage.height)
        }
        return CGSize(width: image.size.width * image.scale, height: image.size.height * image.scale)
    }
}

// MARK: - MediaPipe live stream delegates

extension CombinedLandmarkerHelper: HandLandmarkerLiveStreamDelegate {
    func handLandmarker(
        _ handLandmarker: HandLandmarker,
        didFinishDetection result: HandLandmarkerResult?,
        timestampInMilliseconds: Int,
        error: Error?
    ) {
        if let error {
            combinedLandmarkerHelperListener?.onError(error.localizedDescription)
            return
        }

        latestHandResult = result
        maybeSendCombinedResult()
    }
}

extension CombinedLandmarkerHelper: PoseLandmarkerLiveStreamDelegate {
    func poseLandmarker(
        _ poseLandmarker: PoseLandmarker,
        didFinishDetection result: PoseLandmarkerResult?,
        timestampInMilliseconds: Int,
        error: Error?
    ) {
        if let error {
            combinedLandmarkerHelperListener?.onError(error.localizedDescription)
            return
        }

        latestPoseResult = result
        maybeSendCombinedResult()
    }
}
