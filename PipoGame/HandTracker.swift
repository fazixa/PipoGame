import ARKit
import Vision
import simd

/// Tracks the user's palm in world space: Vision hand-pose landmarks on the
/// camera image + LiDAR scene depth at the palm pixel, unprojected through
/// the camera intrinsics. Runs only while `enabled` (hand mode).
final class HandTracker: NSObject, ARSessionDelegate {

    /// Set by PipoController when hand mode toggles.
    var enabled = false

    /// Smoothed palm position in world space; nil until first detection.
    private(set) var palmWorldPosition: SIMD3<Float>?
    /// Smoothed horizontal direction the fingers point (wrist -> middle
    /// knuckle) in world space; drives Pipo's yaw so he turns with the hand.
    private(set) var palmDirection: SIMD3<Float>?
    private var lastDetectionTime: CFTimeInterval = 0

    var isTracking: Bool {
        palmWorldPosition != nil && CACurrentMediaTime() - lastDetectionTime < 0.4
    }

    private let queue = DispatchQueue(label: "pipo.handtracking", qos: .userInteractive)
    private var inFlight = false
    private let request: VNDetectHumanHandPoseRequest = {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 1
        return request
    }()

    func reset() {
        palmWorldPosition = nil
        palmDirection = nil
        lastDetectionTime = 0
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard enabled, !inFlight, let depthMap = frame.sceneDepth?.depthMap else { return }
        inFlight = true

        // Retain only the buffers and camera math, never the ARFrame itself.
        let image = frame.capturedImage
        let intrinsics = frame.camera.intrinsics
        let cameraTransform = frame.camera.transform
        let imageWidth = Float(CVPixelBufferGetWidth(image))
        let imageHeight = Float(CVPixelBufferGetHeight(image))

        queue.async { [weak self] in
            defer { self?.inFlight = false }
            self?.process(image: image, depthMap: depthMap,
                          intrinsics: intrinsics, cameraTransform: cameraTransform,
                          imageWidth: imageWidth, imageHeight: imageHeight)
        }
    }

    private func process(image: CVPixelBuffer, depthMap: CVPixelBuffer,
                         intrinsics: simd_float3x3, cameraTransform: simd_float4x4,
                         imageWidth: Float, imageHeight: Float) {
        // Portrait phone: the sensor buffer is landscape; .right tells Vision
        // how to rotate it upright so hand detection is reliable.
        let handler = VNImageRequestHandler(cvPixelBuffer: image, orientation: .right)
        try? handler.perform([request])
        guard let hand = request.results?.first else { return }

        // Palm center: wrist + finger base knuckles.
        let jointNames: [VNHumanHandPoseObservation.JointName] =
            [.wrist, .indexMCP, .middleMCP, .ringMCP, .littleMCP]
        var sum = SIMD2<Float>.zero
        var count: Float = 0
        for name in jointNames {
            guard let point = try? hand.recognizedPoint(name),
                  point.confidence > 0.3 else { continue }
            sum += SIMD2<Float>(Float(point.location.x), Float(point.location.y))
            count += 1
        }
        guard count >= 3 else { return }
        let palmOriented = sum / count

        func unproject(_ oriented: SIMD2<Float>) -> SIMD3<Float>? {
            // Vision coords are in the ROTATED (portrait) image, lower-left
            // origin. Map back to the raw landscape buffer, top-left origin:
            // for a 90° CW display rotation, (xo, yo) -> (1 - yo, 1 - xo).
            let u = 1 - oriented.y
            let v = 1 - oriented.x

            CVPixelBufferLockBaseAddress(depthMap, .readOnly)
            let dw = CVPixelBufferGetWidth(depthMap)
            let dh = CVPixelBufferGetHeight(depthMap)
            let dx = min(max(Int(u * Float(dw)), 0), dw - 1)
            let dy = min(max(Int(v * Float(dh)), 0), dh - 1)
            let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)
            let base = CVPixelBufferGetBaseAddress(depthMap)!
            let depth = base.advanced(by: dy * rowBytes + dx * MemoryLayout<Float32>.size)
                .assumingMemoryBound(to: Float32.self).pointee
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)

            guard depth.isFinite, depth > 0.1, depth < 3.0 else { return nil }

            // Through the intrinsics into ARKit camera space
            // (+x right, +y up, -z forward), then into world space.
            let fx = intrinsics[0][0], fy = intrinsics[1][1]
            let cx = intrinsics[2][0], cy = intrinsics[2][1]
            let px = u * imageWidth
            let py = v * imageHeight
            let cameraPoint = SIMD4<Float>((px - cx) * depth / fx,
                                           -(py - cy) * depth / fy,
                                           -depth, 1)
            let world4 = cameraTransform * cameraPoint
            return SIMD3<Float>(world4.x, world4.y, world4.z)
        }

        guard let world = unproject(palmOriented) else { return }

        // Hand orientation: wrist -> middle knuckle, flattened to the ground
        // plane. Falls back to the previous direction when either joint's
        // depth sample is unusable.
        var direction: SIMD3<Float>?
        if let wrist = try? hand.recognizedPoint(.wrist), wrist.confidence > 0.3,
           let middle = try? hand.recognizedPoint(.middleMCP), middle.confidence > 0.3,
           let wristWorld = unproject(SIMD2(Float(wrist.location.x), Float(wrist.location.y))),
           let middleWorld = unproject(SIMD2(Float(middle.location.x), Float(middle.location.y))) {
            var d = middleWorld - wristWorld
            d.y = 0
            let len = simd_length(d)
            if len > 0.02 {
                direction = d / len
            }
        }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            if let previous = self.palmWorldPosition,
               simd_length(world - previous) < 0.6 {
                self.palmWorldPosition = simd_mix(previous, world,
                                                  SIMD3<Float>(repeating: 0.35))
            } else {
                self.palmWorldPosition = world
            }
            if let direction {
                if let previous = self.palmDirection {
                    let blended = simd_mix(previous, direction, SIMD3<Float>(repeating: 0.25))
                    let len = simd_length(blended)
                    if len > 0.001 { self.palmDirection = blended / len }
                } else {
                    self.palmDirection = direction
                }
            }
            self.lastDetectionTime = CACurrentMediaTime()
        }
    }
}
