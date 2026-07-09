import ReplayKit
import AVFoundation
import Photos
import UIKit

/// Records the app's screen with ReplayKit and saves the result to the
/// Photos library. UI is hidden by ContentView before capture starts, so
/// the footage contains only the AR view.
final class ScreenRecorder: NSObject, ObservableObject {

    @Published var isRecording = false
    @Published var toast: String?

    private var writer: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var outputURL: URL?
    private let queue = DispatchQueue(label: "pipo.recorder")

    func start() {
        guard !isRecording else { return }
        let recorder = RPScreenRecorder.shared()
        guard recorder.isAvailable else {
            showToast("Screen recording unavailable")
            return
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pipo-\(Int(Date().timeIntervalSince1970)).mov")
        do {
            let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
            let scale = UIScreen.main.scale
            let size = UIScreen.main.bounds.size
            let settings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.hevc,
                AVVideoWidthKey: Int(size.width * scale),
                AVVideoHeightKey: Int(size.height * scale),
            ]
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
            input.expectsMediaDataInRealTime = true
            writer.add(input)
            self.writer = writer
            self.videoInput = input
            self.outputURL = url
        } catch {
            showToast("Could not start recording")
            return
        }

        recorder.isMicrophoneEnabled = false
        recorder.startCapture(handler: { [weak self] buffer, type, error in
            guard error == nil, type == .video else { return }
            self?.queue.async { self?.append(buffer) }
        }, completionHandler: { [weak self] error in
            DispatchQueue.main.async {
                if let error {
                    self?.showToast(error.localizedDescription)
                } else {
                    self?.isRecording = true
                }
            }
        })
    }

    func stop() {
        guard isRecording else { return }
        isRecording = false
        RPScreenRecorder.shared().stopCapture { [weak self] _ in
            self?.queue.async { self?.finish() }
        }
    }

    private func append(_ buffer: CMSampleBuffer) {
        guard let writer, let input = videoInput,
              CMSampleBufferDataIsReady(buffer) else { return }
        if writer.status == .unknown {
            writer.startWriting()
            writer.startSession(atSourceTime: CMSampleBufferGetPresentationTimeStamp(buffer))
        }
        guard writer.status == .writing, input.isReadyForMoreMediaData else { return }
        input.append(buffer)
    }

    private func finish() {
        guard let writer, let url = outputURL else { return }
        videoInput?.markAsFinished()
        if writer.status == .writing {
            writer.finishWriting { [weak self] in
                self?.saveToPhotos(url)
            }
        } else {
            showToast("Recording failed")
        }
        self.writer = nil
        self.videoInput = nil
        self.outputURL = nil
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
