import Foundation
import UIKit
import TensorFlowLite
import Accelerate

//Interface for detector listener
protocol DetectorListener: AnyObject {
    func noDetect()
    func detected(results: YoloResults, sourceWidth: Int, sourceHeight: Int)
}

class Detector {
    private var interpreter: Interpreter
    private weak var listener: DetectorListener?
    private var labels: [String] = []
    
    private var tensorWidth: Int = 0
    private var tensorHeight: Int = 0
    private var numChannel: Int = 0
    private var numElements: Int = 0
    
    private var bowRepeat: Int = 0
    private var stringRepeat: Int = 0
    private var bowPoints: [Point]?
    private var stringPoints: [Point]?
    private var yLocked: Bool = false
    private var yAvg: [Double]?
    private var frameCounter: Int = 0
    private var stringYCoordHeights: [[Int]] = []
    private let numWaitFrames: Int = 5
    
    // Constsants
    private static let INPUT_MEAN: Float = 0.0
    private static let INPUT_STANDARD_DEVIATION: Float = 255.0
    private static let CONFIDENCE_THRESHOLD: Float = 0.1
    
    // Custom data structs
    struct YoloResults {
        var bowResults: [Point]?
        var stringResults: [Point]?
    }
    
    struct Point {
        var x: Double
        var y: Double
    }
    
    struct OrientedBoundingBox {
        let x: Float
        let y: Float
        let height: Float
        let width: Float
        let conf: Float
        let cls: Int
        let angle: Float
    }
    
    struct ReturnBow {
        var classification: Int?
        var bow: [Point]?
        var string: [Point]?
        var angle: Int?
    }
    
    init(listener: DetectorListener? = nil) throws {
        self.listener = listener
        
        var options = Interpreter.Options()
        options.threadCount = 4
        
        if let metalDelegate = try? MetalDelegate() {
            options.addDelegate(metalDelegate)
        }
        // Load model
        guard let modelPath = Bundle.main.path(forResource: "best_nano_float16", ofType: "tflite") else {
            throw NSError(domain: "Detector", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file not found"])
        }
        
        // Initialize interpreter
        interpreter = try Interpreter(modelPath: modelPath, options: options)
        try interpreter.allocateTensors()
        
        // Get input/output shapes
        let inputShape = try interpreter.input(at: 0).shape
        let outputShape = try interpreter.output(at: 0).shape
        
        // Handle both NCHW and NHWC formats
        if inputShape.dimensions[1] == 3 {
            tensorWidth = inputShape.dimensions[2]
            tensorHeight = inputShape.dimensions[3]
        } else {
            tensorWidth = inputShape.dimensions[1]
            tensorHeight = inputShape.dimensions[2]
        }
        
        numChannel = outputShape.dimensions[1]
        numElements = outputShape.dimensions[2]
    }
    
    func detect(frame: UIImage) -> YoloResults {
        let inferenceStartTime = Date()
        var results = YoloResults(bowResults: nil, stringResults: nil)

        if(tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0) {
            print("MODEL ERROR")
            return results
        }
        
        // Resize image
        guard let resizedImage = frame.resize(to: CGSize(width: tensorWidth, height: tensorHeight)),
              let pixelBuffer = resizedImage.pixelBuffer() else {
            print("Failed to Resize Image")
            return results
        }
        
        guard let inputData = preprocessImage(pixelBuffer: pixelBuffer) else {
            return results
        }
        
        // Run inference
        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            
            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            let floatArray = outputData.toArray(type: Float.self)
            
            let bestBoxes = newBestBox(array: floatArray)
            
            var bowConf: Float = 0
            var stringConf: Float = 0
            let ogWidth = Float(frame.size.width)
            let ogHeight = Float(frame.size.height)
            
            for box in bestBoxes {
                if box.cls == 0 && box.conf > bowConf {
                    results.bowResults = rotatedRectToPoints(
                        cx: box.x * ogWidth,
                        cy: box.y * ogHeight,
                        w: box.width * ogWidth,
                        h: box.height * ogHeight,
                        angleRad: box.angle
                    )
                    bowConf = box.conf
                } else if box.cls == 1 && box.conf > stringConf {
                    let angle = box.width > box.height ? box.angle + Float.pi / 2 : box.angle
                    let points = rotatedRectToPoints(
                        cx: box.x * ogWidth,
                        cy: box.y * ogHeight,
                        w: box.width * ogWidth,
                        h: box.height * ogHeight,
                        angleRad: angle
                    )
                    results.stringResults = sortStringPoints(pts: points)
                    stringConf = box.conf
                }
            }
            
            let inferenceTime = Date().timeIntervalSince(inferenceStartTime)
            
            if results.bowResults == nil && results.stringResults == nil {
                listener?.noDetect()
            } else {
                listener?.detected(results: results, sourceWidth: Int(frame.size.width), sourceHeight: Int(frame.size.height))
                print(results)
            }
            
        } catch {
            print("Inference error: \(error)")
        }
        
        return results
    }
    
    private func preprocessImage(pixelBuffer: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        var floatData = [Float](repeating: 0, count: width * height * 3)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixel = baseAddress.advanced(by: y * bytesPerRow + x * 4)
                let r = Float(pixel.load(fromByteOffset: 0, as: UInt8.self))
                let g = Float(pixel.load(fromByteOffset: 1, as: UInt8.self))
                let b = Float(pixel.load(fromByteOffset: 2, as: UInt8.self))
                
                let index = (y * width + x) * 3
                floatData[index] = (r - Detector.INPUT_MEAN) / Detector.INPUT_STANDARD_DEVIATION
                floatData[index + 1] = (g - Detector.INPUT_MEAN) / Detector.INPUT_STANDARD_DEVIATION
                floatData[index + 2] = (b - Detector.INPUT_MEAN) / Detector.INPUT_STANDARD_DEVIATION
            }
        }
        
        return Data(bytes: &floatData, count: floatData.count * MemoryLayout<Float>.size)
    }
    
    func drawPointsOnBitmap(
        bitmap: UIImage,
        points: YoloResults,
        classification: Int?,
        angle: Int?
    ) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(bitmap.size, false, bitmap.scale)
        defer { UIGraphicsEndImageContext() }
        
        guard let context = UIGraphicsGetCurrentContext() else {
            return bitmap
        }
        
        bitmap.draw(at: .zero)
        
        // Determine if there's an issue
        let hasIssue = (classification != nil && classification != 0) ||
                       (angle != nil && angle == 1)
        
        // Choose colors
        let boxColor = hasIssue ? UIColor(red: 1.0, green: 140/255.0, blue: 0, alpha: 1.0) : UIColor.blue
        
        context.setStrokeColor(boxColor.cgColor)
        context.setLineWidth(8.0)
        
        // Draw string box
        if let stringBox = points.stringResults, stringBox.count >= 4 {
            context.move(to: CGPoint(x: stringBox[0].x, y: stringBox[0].y))
            context.addLine(to: CGPoint(x: stringBox[1].x, y: stringBox[1].y))
            context.addLine(to: CGPoint(x: stringBox[2].x, y: stringBox[2].y))
            context.addLine(to: CGPoint(x: stringBox[3].x, y: stringBox[3].y))
            context.addLine(to: CGPoint(x: stringBox[0].x, y: stringBox[0].y))
            context.strokePath()
        }
        
        // Draw bow box
        if let bowBox = points.bowResults, bowBox.count >= 4 {
            context.move(to: CGPoint(x: bowBox[0].x, y: bowBox[0].y))
            context.addLine(to: CGPoint(x: bowBox[1].x, y: bowBox[1].y))
            context.addLine(to: CGPoint(x: bowBox[2].x, y: bowBox[2].y))
            context.addLine(to: CGPoint(x: bowBox[3].x, y: bowBox[3].y))
            context.addLine(to: CGPoint(x: bowBox[0].x, y: bowBox[0].y))
            context.strokePath()
        }
        
        // Classification labels
        let classificationLabels: [Int: String] = [
            0: "",
            1: "Bow outside zone",
            2: "Bow too high",
            3: "Bow too low"
        ]
        
        let angleLabels: [Int: String] = [
            0: "",
            1: "Incorrect bow angle"
        ]
        
        // Text attributes
        let orangeColor = UIColor(red: 1.0, green: 140/255.0, blue: 0, alpha: 1.0)
        let darkOrange = UIColor(red: 204/255.0, green: 85/255.0, blue: 0, alpha: 1.0)
        
        let textAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.boldSystemFont(ofSize: 56),
            .foregroundColor: orangeColor,
            .strokeColor: darkOrange,
            .strokeWidth: -6.0
        ]
        
        let topMargin: CGFloat = 300
        let lineSpacing: CGFloat = 70
        let centerX = bitmap.size.width / 2
        var currentY = topMargin
        
        // Draw classification message
        if let classification = classification, classification != 0,
           let message = classificationLabels[classification], !message.isEmpty {
            let attributedString = NSAttributedString(string: message, attributes: textAttributes)
            let textSize = attributedString.size()
            attributedString.draw(at: CGPoint(x: centerX - textSize.width / 2, y: currentY))
            currentY += lineSpacing
        }
        
        // Draw angle message
        if let angle = angle, angle == 1,
           let message = angleLabels[angle], !message.isEmpty {
            let attributedString = NSAttributedString(string: message, attributes: textAttributes)
            let textSize = attributedString.size()
            attributedString.draw(at: CGPoint(x: centerX - textSize.width / 2, y: currentY))
        }
        
        guard let annotatedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            return bitmap
        }
        
        return annotatedImage
    }
    
    func processFrame(bitmap: UIImage) -> UIImage {
        let classificationResult = classify(results: detect(frame: bitmap))
        let annotatedBitmap = drawPointsOnBitmap(
            bitmap: bitmap,
            points: YoloResults(
                bowResults: classificationResult.bow,
                stringResults: classificationResult.string
            ),
            classification: classificationResult.classification,
            angle: classificationResult.angle
        )
        return annotatedBitmap
    }

    private func rotatedRectToPoints(cx: Float, cy: Float, w: Float, h: Float, angleRad: Float) -> [Point] {
        let halfW = w / 2
        let halfH = h / 2
        print("ANGLE \(angleRad)")
        let cosA = cos(angleRad - Float.pi / 2)
        let sinA = sin(angleRad - Float.pi / 2)
        
        let corners: [(Float, Float)] = [
            (-halfW, -halfH),
            (halfW, -halfH),
            (halfW, halfH),
            (-halfW, halfH)
        ]
        
        return corners.map { (x, y) in
            let xRot = x * cosA - y * sinA + cx
            let yRot = x * sinA + y * cosA + cy
            return Point(x: Double(xRot), y: Double(yRot))
        }
    }
    
    private func newBestBox(array: [Float]) -> [OrientedBoundingBox] {
        var boundingBoxes: [OrientedBoundingBox] = []
        
        for r in 0..<numElements {
            let stringCnf = array[5 * numElements + r]
            let bowCnf = array[4 * numElements + r]
            let cls = stringCnf > bowCnf ? 1 : 0
            let cnf = stringCnf > bowCnf ? stringCnf : bowCnf
            
            if cnf > Detector.CONFIDENCE_THRESHOLD {
                let x = array[r]
                let y = array[1 * numElements + r]
                let h = array[2 * numElements + r]
                let w = array[3 * numElements + r]
                let angle = array[6 * numElements + r]
                
                boundingBoxes.append(
                    OrientedBoundingBox(
                        x: x, y: y, height: h, width: w,
                        conf: cnf, cls: cls, angle: angle
                    )
                )
            }
        }
        
        return boundingBoxes
    }
    
    func updatePoints(stringBox: [Point], bowBox: [Point]) {
        bowPoints = bowBox
        
        if !yLocked {
            stringPoints = stringBox
        } else if let yAvg = yAvg {
            let sortedString = sortStringPoints(pts: stringBox)
            stringPoints = sortedString
        }
    }
    
    func sortStringPoints(pts: [Point]) -> [Point] {
        let sortedPoints = pts.sorted { $0.y < $1.y }
        let topPoints = Array(sortedPoints.prefix(2)).sorted { $0.x < $1.x }
        let bottomPoints = Array(sortedPoints.suffix(2)).sorted { $0.x > $1.x }
        return topPoints + bottomPoints
    }
    
    func getMidline() -> [Int] {
        func distance(pt1: Point, pt2: Point) -> Double {
            return (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y)
        }
        
        guard let bowPoints = bowPoints else { return [] }
        
        let d1 = distance(pt1: bowPoints[0], pt2: bowPoints[1])
        let d2 = distance(pt1: bowPoints[1], pt2: bowPoints[2])
        let d3 = distance(pt1: bowPoints[2], pt2: bowPoints[3])
        let d4 = distance(pt1: bowPoints[3], pt2: bowPoints[0])
        let distances = [d1, d2, d3, d4]
        
        guard let minDistance = distances.min(),
              let minIndex = distances.firstIndex(of: minDistance) else {
            return []
        }
        
        let (pair1, pair2): ((Point, Point), (Point, Point))
        switch minIndex {
        case 0:
            pair1 = (bowPoints[0], bowPoints[1])
            pair2 = (bowPoints[2], bowPoints[3])
        case 1:
            pair1 = (bowPoints[1], bowPoints[2])
            pair2 = (bowPoints[3], bowPoints[0])
        case 2:
            pair1 = (bowPoints[2], bowPoints[3])
            pair2 = (bowPoints[0], bowPoints[1])
        default:
            pair1 = (bowPoints[3], bowPoints[0])
            pair2 = (bowPoints[1], bowPoints[2])
        }
        
        let mid1 = [(pair1.0.x + pair1.1.x) / 2, (pair1.0.y + pair1.1.y) / 2]
        let mid2 = [(pair2.0.x + pair2.1.x) / 2, (pair2.0.y + pair2.1.y) / 2]
        
        let dy = mid1[1] - mid2[1]
        let dx = mid1[0] - mid2[0]
        
        if dx == 0.0 {
            return [Int.max, Int(mid1[0])]
        } else {
            let slope = dy / dx
            let intercept = mid1[1] - slope * mid1[0]
            return [Int(slope), Int(intercept)]
        }
    }
    
    private func getVerticalLines() -> [[Double]] {
        guard let stringPoints = stringPoints else { return [] }
        
        let topLeft = stringPoints[0]
        let topRight = stringPoints[1]
        let botRight = stringPoints[2]
        let botLeft = stringPoints[3]
        
        let dxLeft = topLeft.x - botLeft.x
        let leftSlope: Double
        let leftYint: Double
        
        if dxLeft == 0.0 {
            leftSlope = Double.infinity
            leftYint = -1.0
        } else {
            leftSlope = (topLeft.y - botLeft.y) / dxLeft
            leftYint = topLeft.y - leftSlope * topLeft.x
        }
        
        let dxRight = topRight.x - botRight.x
        let rightSlope: Double
        let rightYint: Double
        
        if dxRight == 0.0 {
            rightSlope = Double.infinity
            rightYint = -1.0
        } else {
            rightSlope = (topRight.y - botRight.y) / dxRight
            rightYint = topRight.y - rightSlope * topRight.x
        }
        
        let leftLine = [leftSlope, leftYint, topLeft.y, botLeft.y]
        let rightLine = [rightSlope, rightYint, topRight.y, botRight.y]
        
        return [leftLine, rightLine]
    }
    
    private func intersectsVertical(linearLine: [Int], verticalLines: [[Double]]) -> Int {
        let m = Double(linearLine[0])
        let b = Double(linearLine[1])
        
        let verticalOne = verticalLines[0]
        let verticalTwo = verticalLines[1]
        
        func getIntersection(vLine: [Double], xRef: Double) -> Point? {
            let slopeV = vLine[0]
            let interceptV = vLine[1]
            let topY = vLine[2]
            let botY = vLine[3]
            
            let x: Double
            let y: Double
            
            if slopeV == Double.infinity || interceptV == -1.0 {
                x = xRef
                if m == Double.infinity { return nil }
                y = m * x + b
            } else if m == Double.infinity {
                x = b
                y = slopeV * x + interceptV
            } else if abs(m - slopeV) < 1e-6 {
                return nil
            } else {
                x = (interceptV - b) / (m - slopeV)
                y = m * x + b
            }
            
            let yMin = min(topY, botY)
            let yMax = max(topY, botY)
            
            if yMin > y || y > yMax {
                return nil
            }
            
            return Point(x: x, y: y)
        }
        
        guard let stringPoints = stringPoints else { return 1 }
        
        let xLeft = stringPoints[0].x
        let xRight = stringPoints[1].x
        
        var pt1 = getIntersection(vLine: verticalOne, xRef: xLeft)
        var pt2 = getIntersection(vLine: verticalTwo, xRef: xRight)
        
        if pt1 == nil && pt2 == nil {
            print("BOW: INVALID INTERSECTION")
            return 1
        }
        if pt1 == nil { pt1 = pt2 }
        if pt2 == nil { pt2 = pt1 }
        
        return bowHeightIntersection(intersectionPoints: [pt1!, pt2!], verticalLines: [verticalOne, verticalTwo])
    }
    
    private func bowHeightIntersection(intersectionPoints: [Point], verticalLines: [[Double]]) -> Int {
        let topZonePercentage = 0.15
        let bottomZonePercentage = 0.15
        
        let verticalOne = verticalLines[0]
        let verticalTwo = verticalLines[1]
        
        let topY1 = verticalOne[2]
        let topY2 = verticalTwo[2]
        let botY1 = verticalOne[3]
        let botY2 = verticalTwo[3]
        
        let height = abs(((botY1 - topY1) + (botY2 - topY2)) / 2.0)
        if height == 0.0 { return 0 }
        
        let avgTopY = (topY1 + topY2) / 2.0
        let avgBotY = (botY1 + botY2) / 2.0
        
        let tooHighThreshold = avgTopY + height * topZonePercentage
        let tooLowThreshold = avgBotY - height * bottomZonePercentage
        
        let intersectionY = intersectionPoints.map { $0.y }.reduce(0, +) / Double(intersectionPoints.count)
        
        if intersectionY <= tooHighThreshold {
            return 2
        }
        
        if intersectionY >= tooLowThreshold {
            return 3
        }
        
        return 0
    }
    
    private func averageYCoordinate(stringBoxCoords: [Point]) {
        frameCounter += 1
        let yCoords = stringBoxCoords.map { Int($0.y) }
        stringYCoordHeights.append(yCoords)
        
        if frameCounter % numWaitFrames == 0 {
            let topLeftAvg = median(values: stringYCoordHeights.map { $0[0] })
            let topRightAvg = median(values: stringYCoordHeights.map { $0[1] })
            let botRightAvg = median(values: stringYCoordHeights.map { $0[2] })
            let botLeftAvg = median(values: stringYCoordHeights.map { $0[3] })
            
            stringPoints?[0].y = topLeftAvg
            stringPoints?[1].y = topRightAvg
            stringPoints?[2].y = botRightAvg
            stringPoints?[3].y = botLeftAvg
            yAvg = [topLeftAvg, topRightAvg]
            yLocked = true
            stringYCoordHeights = []
        }
    }
    
    private func median(values: [Int]) -> Double {
        guard !values.isEmpty else { fatalError("Empty list has no median") }
        
        let sorted = values.sorted()
        let middle = sorted.count / 2
        
        if sorted.count % 2 == 0 {
            return Double(sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            return Double(sorted[middle])
        }
    }
    
    private func degrees(radians: Double) -> Double {
        return radians * (180.0 / Double.pi)
    }
    
    private func bowAngle(bowLine: [Int], verticalLines: [[Double]]) -> Int {
        let maxAngle = 15.0
        
        let mBow = Double(bowLine[0])
        let m1 = verticalLines[0][0]
        let m2 = verticalLines[1][0]
        
        let angleOne = abs(degrees(radians: atan(abs(mBow - m2) / (1 + mBow * m2))))
        let angleTwo = abs(degrees(radians: atan(abs(m1 - mBow) / (1 + m1 * mBow))))
        
        let minAngle = abs(90 - min(angleOne, angleTwo))
        
        return minAngle > maxAngle ? 1 : 0
    }
    
    func classify(results: YoloResults) -> ReturnBow {
        var classResults = ReturnBow(classification: nil, bow: nil, string: nil, angle: nil)
        var mutableResults = results
        
        if mutableResults.stringResults != nil {
            stringRepeat = 0
            stringPoints = mutableResults.stringResults
        } else if stringRepeat < 5, let stringPoints = stringPoints {
            classResults.classification = -1
            stringRepeat += 1
            mutableResults.stringResults = stringPoints
        } else {
            stringPoints = nil
        }
        
        if mutableResults.bowResults != nil {
            bowRepeat = 0
            bowPoints = mutableResults.bowResults
        } else if bowRepeat < 5, let bowPoints = bowPoints {
            classResults.classification = -1
            bowRepeat += 1
            mutableResults.bowResults = bowPoints
        } else {
            bowPoints = nil
        }
        
        if stringPoints == nil && bowPoints == nil {
            classResults.classification = -2
            return classResults
        }
        
        if mutableResults.stringResults == nil {
            classResults.classification = -1
            classResults.bow = mutableResults.bowResults
            return classResults
        } else if mutableResults.bowResults == nil {
            classResults.classification = -1
            classResults.string = mutableResults.stringResults
            return classResults
        } else {
            classResults.string = mutableResults.stringResults
            classResults.bow = mutableResults.bowResults
            updatePoints(stringBox: mutableResults.stringResults!, bowBox: mutableResults.bowResults!)
            let midlines = getMidline()
            let vertLines = getVerticalLines()
            let intersectPoints = intersectsVertical(linearLine: midlines, verticalLines: vertLines)
            classResults.angle = bowAngle(bowLine: midlines, verticalLines: vertLines)
            classResults.classification = intersectPoints
            print("BOW: \(String(describing: classResults.classification))")
            return classResults
        }
    }
}

// Extensions
extension Data {
    func toArray<T>(type: T.Type) -> [T] where T: ExpressibleByIntegerLiteral {
        var array = [T](repeating: 0, count: count / MemoryLayout<T>.stride)
        _ = array.withUnsafeMutableBytes { copyBytes(to: $0) }
        return array
    }
}

extension UIImage {
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    // Convert UIImage to CVPixelBuffer for ML 
    func pixelBuffer() -> CVPixelBuffer? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attributes as CFDictionary,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let context = CGContext(
            data: pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }
        
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        
        return buffer
    }
}