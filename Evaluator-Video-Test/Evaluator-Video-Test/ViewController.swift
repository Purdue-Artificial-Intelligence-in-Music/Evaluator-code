import UIKit
import AVFoundation

class ViewController: UIViewController {

    // MARK: - UI
    private let imageView = UIImageView()

    // MARK: - Detector
    private lazy var detector: Detector = {
        do {
            return try Detector()
        } catch {
            fatalError("Failed to initialize Detector: \(error)")
        }
    }()
    override func viewDidLoad() {
        super.viewDidLoad()

        setupUI()
        processVideo(named: "test_video")
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

                let outputSettings: [String: Any] = [
                    kCVPixelBufferPixelFormatTypeKey as String:
                        kCVPixelFormatType_32BGRA
                ]

                let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
                reader.add(output)
                reader.startReading()

                while reader.status == .reading,
                      let sampleBuffer = output.copyNextSampleBuffer(),
                      let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

                    let frame = pixelBufferToUIImage(pixelBuffer)
                    let annotated = self.detector.processFrame(bitmap: frame)

                    await MainActor.run {
                        self.imageView.image = annotated
                    }

                    try await Task.sleep(nanoseconds: 33_000_000) // ~30 FPS
                }

            } catch {
                fatalError("Video processing failed: \(error)")
            }
        }
    }
}
