import RealityKit
import AVFoundation
import CoreImage
import Metal
import Photos
import UIKit

/// High-resolution AR recorder. While recording, the ARView's render scale
/// is raised so the whole AR pipeline (camera feed, occlusion, Pipo, toon
/// outline) is rendered at ~4K, and every frame's framebuffer texture is
/// tapped via renderCallbacks.postProcess into an HEVC writer. The on-screen
/// image is unchanged (downscaled by the display); UI is hidden by
/// ContentView so footage contains only the AR view.
final class ScreenRecorder: NSObject, ObservableObject {

    @Published var isRecording = false
    @Published var toast: String?

    /// Recording height in pixels; width follows the screen aspect.
    private let targetHeight: CGFloat = 3840

    private weak var arView: ARView?
    private var originalScaleFactor: CGFloat = 0

    private let device = MTLCreateSystemDefaultDevice()!
    private lazy var ciContext = CIContext(mtlDevice: device,
                                           options: [.cacheIntermediates: false])
    private var ringTextures: [MTLTexture] = []
    private var ringIndex = 0

    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var outputURL: URL?
    private var recordingStart: CFTimeInterval?
    private let queue = DispatchQueue(label: "pipo.recorder")

    // MARK: - Control

    func start(arView: ARView) {
        guard !isRecording, writer == nil else { return }
        self.arView = arView

        originalScaleFactor = arView.contentScaleFactor
        let pointHeight = max(arView.bounds.height, 1)
        arView.contentScaleFactor = targetHeight / pointHeight

        recordingStart = nil
        isRecording = true
        arView.renderCallbacks.postProcess = { [weak self] context in
            self?.captureFrame(context)
        }
    }

    func stop() {
        guard isRecording else { return }
        isRecording = false
        arView?.renderCallbacks.postProcess = nil
        arView?.contentScaleFactor = originalScaleFactor
        queue.async { [weak self] in self?.finish() }
    }

    // MARK: - Per-frame capture (render thread)

    private func captureFrame(_ context: ARView.PostProcessContext) {
        let source = context.sourceColorTexture

        // Mandatory passthrough: with a postProcess callback installed we
        // are responsible for filling the output texture.
        if let blit = context.commandBuffer.makeBlitCommandEncoder() {
            blit.copy(from: source, to: context.targetColorTexture)
            blit.endEncoding()
        }
        guard isRecording else { return }

        if writer == nil {
            setUpWriter(width: source.width, height: source.height,
                        format: source.pixelFormat)
        }
        guard !ringTextures.isEmpty else { return }

        // Copy into our own ring texture so the renderer can recycle its
        // drawable; encode to video once the GPU finishes this frame.
        let capture = ringTextures[ringIndex]
        ringIndex = (ringIndex + 1) % ringTextures.count
        if let blit = context.commandBuffer.makeBlitCommandEncoder() {
            blit.copy(from: source, to: capture)
            blit.endEncoding()
        }
        let timestamp = CACurrentMediaTime()
        context.commandBuffer.addCompletedHandler { [weak self] _ in
            self?.queue.async { self?.append(texture: capture, at: timestamp) }
        }
    }

    private func setUpWriter(width: Int, height: Int, format: MTLPixelFormat) {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pipo-\(Int(Date().timeIntervalSince1970)).mov")
        do {
            let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
            let settings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.hevc,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: 55_000_000,
                    AVVideoExpectedSourceFrameRateKey: 60,
                ],
            ]
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
            input.expectsMediaDataInRealTime = true
            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: width,
                    kCVPixelBufferHeightKey as String: height,
                ])
            writer.add(input)
            self.writer = writer
            self.input = input
            self.adaptor = adaptor
            self.outputURL = url

            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: false)
            descriptor.usage = [.shaderRead]
            descriptor.storageMode = .private
            ringTextures = (0..<3).compactMap { _ in device.makeTexture(descriptor: descriptor) }
        } catch {
            showToast("Could not start recording")
        }
    }

    // MARK: - Encoding (serial queue)

    private func append(texture: MTLTexture, at timestamp: CFTimeInterval) {
        guard isRecording || recordingStart != nil,
              let writer, let input, let adaptor else { return }

        if recordingStart == nil {
            recordingStart = timestamp
            writer.startWriting()
            writer.startSession(atSourceTime: .zero)
        }
        guard writer.status == .writing, input.isReadyForMoreMediaData,
              let pool = adaptor.pixelBufferPool else { return }

        var maybeBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &maybeBuffer)
        guard let buffer = maybeBuffer else { return }

        // Metal textures are top-left origin; Core Image is bottom-left.
        var image = CIImage(mtlTexture: texture, options: [:])!
        image = image.transformed(by: CGAffineTransform(scaleX: 1, y: -1)
            .translatedBy(x: 0, y: -image.extent.height))
        ciContext.render(image, to: buffer, bounds: image.extent,
                         colorSpace: CGColorSpaceCreateDeviceRGB())

        let seconds = timestamp - (recordingStart ?? timestamp)
        adaptor.append(buffer, withPresentationTime:
                        CMTime(seconds: seconds, preferredTimescale: 600))
    }

    private func finish() {
        guard let writer, let url = outputURL else { return }
        input?.markAsFinished()
        if writer.status == .writing {
            writer.finishWriting { [weak self] in
                self?.saveToPhotos(url)
            }
        } else {
            showToast("Recording failed")
        }
        self.writer = nil
        self.input = nil
        self.adaptor = nil
        self.outputURL = nil
        self.recordingStart = nil
        self.ringTextures = []
        self.ringIndex = 0
    }

    private func saveToPhotos(_ url: URL) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { [weak self] status in
            guard status == .authorized || status == .limited else {
                self?.showToast("Allow Photos access in Settings to save clips")
                return
            }
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
            }, completionHandler: { success, _ in
                self?.showToast(success ? "Saved to Photos ✓" : "Save failed")
                try? FileManager.default.removeItem(at: url)
            })
        }
    }

    private func showToast(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.toast = message
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                if self.toast == message { self.toast = nil }
            }
        }
    }
}
