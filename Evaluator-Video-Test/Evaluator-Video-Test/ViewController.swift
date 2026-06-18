import UIKit
import AVFoundation
import MediaPipeTasksVision

class ViewController: UIViewController {

    // MARK: - UI
    private let imageView = UIImageView()

    // MARK: - Detectors
    private lazy var detector: Detector = {
        do {
            return try Detector()
        } catch {
            fatalError("Failed to initialize Detector: \(error)")
        }
    }()

    // MediaPipe hand + pose landmarker (elbows are pose landmarks 13 & 14).
    // .gpu runs inference on the Metal-backed GPU delegate.
    private lazy var landmarker = CombinedLandmarkerHelper(currentDelegate: .gpu, runningMode: .video)

    override func viewDidLoad() {
        super.viewDidLoad()

        setupUI()
        processVideo(named: "supination-slow")
    }

    // MARK: - UI Setup
    private func setupUI() {
        imageView.frame = view.bounds
        imageView.contentMode = .scaleAspectFit
        imageView.backgroundColor = .black
        view.addSubview(imageView)
    }

    // MARK: - Video Processing
    private func processVideo(named name: String) {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mp4") else {
            fatalError("Video file not found")
        }

        let asset = AVURLAsset(url: url)

        Task {
            do {
                let tracks = try await asset.loadTracks(withMediaType: .video)
                guard let track = tracks.first else {
                    fatalError("No video track found")
                }

                let reader = try AVAssetReader(asset: asset)

                // Use the track's stored orientation so portrait clips display upright.
                let transform = try await track.load(.preferredTransform)
                let orientation = imageOrientation(from: transform)
                print("preferredTransform: \(transform) -> orientation: \(orientation.rawValue)")

                let outputSettings: [String: Any] = [
                    kCVPixelBufferPixelFormatTypeKey as String:
                        kCVPixelFormatType_32BGRA
                ]

                let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
                reader.add(output)

                reader.startReading()

                var frameTimestampMs = 0
                var frameIndex = 0
                let landmarkEvery = 1
                var lastBundle: CombinedLandmarkerHelper.CombinedResultBundle?

                while reader.status == .reading,
                      let sampleBuffer = output.copyNextSampleBuffer(),
                      let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

                    let frame = pixelBufferToUIImage(pixelBuffer, orientation: orientation)

                    let annotated = self.detector.processFrame(bitmap: frame)

                    
                    lastBundle = self.landmarker.detectVideoFrame(
                        frame: frame, timestampMs: frameTimestampMs)
                    frameTimestampMs += 33
                    frameIndex += 1

                    let finalImage = lastBundle.map {
                        self.landmarker.drawMediaPipeAnnotations(on: annotated, result: $0)
                    } ?? annotated

                    await MainActor.run {
                        self.imageView.image = finalImage
                    }
                }

            } catch {
                fatalError("Video processing failed: \(error)")
            }
        }
    }

}
