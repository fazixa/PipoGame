import ARCore
import ARKit
import CoreLocation
import Foundation
import RealityKit
import UIKit
import simd

/// Feeds ARKit frames into ARCore's Geospatial API and reports whether the
/// device has localized against Google's VPS. This is Phase 4's first
/// milestone — confirm outdoor tracking actually works before building
/// anchor placement ("put Pipo on that building") on top of it.
final class GeospatialManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    @Published private(set) var statusText = "Geospatial off"
    @Published private(set) var isHighAccuracy = false

    var enabled = false {
        didSet {
            guard enabled != oldValue else { return }
            enabled ? start() : stop()
        }
    }

    private var locationManager: CLLocationManager?
    private var garSession: GARSession?
    /// The anchor Pipo is currently pinned to, if any. Its `.transform`
    /// keeps refining as ARCore's tracking of that real-world point
    /// improves — read it fresh every frame rather than snapshotting once.
    private(set) var placedAnchor: GARAnchor?

    // Debug visualization of Streetscape Geometry — the real building/
    // terrain mesh ARCore raycasts against, normally invisible.
    private var debugAnchor: AnchorEntity?
    private var streetscapeGeometryModels: [UUID: Entity] = [:]
    /// True while screen recording — hides only the translucent tint (the
    /// real occluder stays active, so Pipo is still correctly hidden behind
    /// buildings in the recording, just without the debug overlay showing).
    private var debugVisualsHidden = false
    /// Manual world-space correction for VPS's small (~1-2m) positioning
    /// error, applied to the whole geometry mesh set as a rigid group (both
    /// the debug tint and the real occluder, since they share debugAnchor).
    @Published private(set) var geometryCalibrationOffset: SIMD3<Float> = .zero

    private func start() {
        statusText = "Requesting location…"
        let manager = CLLocationManager()
        manager.delegate = self
        locationManager = manager
        manager.requestWhenInUseAuthorization()
    }

    private func stop() {
        garSession = nil
        locationManager = nil
        isHighAccuracy = false
        statusText = "Geospatial off"
        debugAnchor?.removeFromParent()
        debugAnchor = nil
        streetscapeGeometryModels.removeAll()
        geometryCalibrationOffset = .zero
    }

    /// Nudges the whole geometry mesh set by 0.5m along one world axis —
    /// manual correction for VPS's positioning error, since the mesh itself
    /// carries no auto-calibration.
    func nudgeGeometryCalibration(x: Float = 0, y: Float = 0, z: Float = 0) {
        geometryCalibrationOffset += SIMD3<Float>(x, y, z)
        debugAnchor?.setPosition(geometryCalibrationOffset, relativeTo: nil)
    }

    /// Hides (or restores) the translucent building/terrain tint across all
    /// currently-visualized geometry — call around screen recording so the
    /// debug overlay never ends up in captured footage.
    func setDebugVisualsHidden(_ hidden: Bool) {
        debugVisualsHidden = hidden
        for container in streetscapeGeometryModels.values {
            container.findEntity(named: "debugVisual")?.isEnabled = !hidden
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        switch manager.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            guard manager.accuracyAuthorization == .fullAccuracy else {
                statusText = "Enable Precise Location for Pipo in Settings"
                return
            }
            setupGARSession()
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        default:
            statusText = "Location permission denied"
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: any Error) {
        statusText = "Location error: \(error.localizedDescription)"
    }

    private func setupGARSession() {
        guard garSession == nil else { return }
        let session: GARSession
        do {
            session = try GARSession(apiKey: Secrets.arCoreAPIKey, bundleIdentifier: nil)
        } catch {
            statusText = "GARSession failed: \(error.localizedDescription)"
            return
        }
        guard session.isGeospatialModeSupported(.enabled) else {
            statusText = "Geospatial not supported on this device"
            return
        }
        let configuration = GARSessionConfiguration()
        configuration.geospatialMode = .enabled
        // Real 3D building/terrain mesh from VPS around the device — needed
        // to raycast a screen tap against an actual distant building rather
        // than the near-field LiDAR plane ARKit already knows about.
        configuration.streetscapeGeometryMode = .enabled
        var error: NSError?
        session.setConfiguration(configuration, error: &error)
        if let error {
            statusText = "GARSession config failed: \(error.localizedDescription)"
            return
        }
        garSession = session
        statusText = "Localizing…"
    }

    /// Raycasts a screen tap against real building/terrain geometry (not
    /// ARKit's near-field plane, which has no idea what's a block away) and
    /// pins Pipo's anchor there. Returns false if the tap didn't land on any
    /// known streetscape geometry (open sky, or VPS hasn't meshed that
    /// spot yet).
    @discardableResult
    func placeAnchor(from arView: ARView, at point: CGPoint) -> Bool {
        guard let garSession,
              let query = arView.makeRaycastQuery(from: point, allowing: .estimatedPlane, alignment: .any)
        else { return false }
        do {
            let results = try garSession.raycastStreetscapeGeometry(origin: query.origin,
                                                                     direction: query.direction)
            guard let result = results.first else {
                statusText = "No building there — try aiming at a building surface"
                return false
            }
            if let placedAnchor {
                garSession.remove(placedAnchor)
            }
            placedAnchor = try garSession.createAnchor(geometry: result.streetscapeGeometry,
                                                        transform: result.worldTransform)
            return true
        } catch {
            statusText = "Couldn't place there: \(error.localizedDescription)"
            return false
        }
    }

    /// Call every frame while `enabled`; cheap no-op otherwise.
    func update(frame: ARFrame, in arView: ARView) {
        guard enabled, let garSession else { return }
        guard let garFrame = try? garSession.update(frame) else { return }
        updateStreetscapeDebugMesh(garFrame: garFrame, arView: arView)
        guard let earth = garFrame.earth else { return }

        guard earth.earthState == .enabled else {
            statusText = "Earth error (\(earth.earthState.rawValue))"
            isHighAccuracy = false
            return
        }
        guard earth.trackingState == .tracking,
              let transform = earth.cameraGeospatialTransform else {
            statusText = "Not tracking — point camera at buildings/signs"
            isHighAccuracy = false
            return
        }

        isHighAccuracy = transform.horizontalAccuracy < 10 && transform.orientationYawAccuracy < 15
        statusText = String(format: "%.6f, %.6f  ±%.1fm  alt %.1fm",
                            transform.coordinate.latitude, transform.coordinate.longitude,
                            transform.horizontalAccuracy, transform.altitude)
    }

    /// Renders ARCore's Streetscape Geometry — the real building/terrain
    /// mesh `placeAnchor` raycasts against — as translucent shapes, so you
    /// can see what area VPS has actually meshed instead of it being
    /// invisible collision-only geometry.
    private func updateStreetscapeDebugMesh(garFrame: GARFrame, arView: ARView) {
        if debugAnchor == nil {
            let anchor = AnchorEntity(world: matrix_identity_float4x4)
            anchor.position = geometryCalibrationOffset
            arView.scene.addAnchor(anchor)
            debugAnchor = anchor
        }
        guard let debugAnchor else { return }

        guard let geometries = garFrame.streetscapeGeometries else {
            for model in streetscapeGeometryModels.values { model.removeFromParent() }
            streetscapeGeometryModels.removeAll()
            return
        }

        var seen = Set<UUID>()
        for geometry in geometries {
            seen.insert(geometry.identifier)
            if streetscapeGeometryModels[geometry.identifier] == nil {
                guard let container = Self.buildGeometryContainer(geometry) else { continue }
                container.setParent(debugAnchor)
                container.findEntity(named: "debugVisual")?.isEnabled = !debugVisualsHidden
                streetscapeGeometryModels[geometry.identifier] = container
            }
            guard let model = streetscapeGeometryModels[geometry.identifier] else { continue }
            model.transform = Transform(matrix: geometry.meshTransform)
            if geometry.trackingState == .stopped {
                streetscapeGeometryModels.removeValue(forKey: geometry.identifier)
                model.removeFromParent()
            } else {
                model.isEnabled = geometry.trackingState == .tracking
            }
        }
        for id in streetscapeGeometryModels.keys where !seen.contains(id) {
            streetscapeGeometryModels[id]?.removeFromParent()
            streetscapeGeometryModels.removeValue(forKey: id)
        }
    }

    /// Two entities sharing one mesh: a real (invisible) occluder so Pipo
    /// correctly disappears behind whatever part of a building is actually
    /// in front of him — and stays visible above/beside it, like he would
    /// against any real 3D geometry — plus a visible translucent tint on
    /// top purely for debug display. The tint doesn't write depth, so it
    /// can't double up with or fight the occluder's own depth writes.
    private static func buildGeometryContainer(_ geometry: GARStreetscapeGeometry) -> Entity? {
        var descriptor = MeshDescriptor()
        var vertices: [SIMD3<Float>] = []
        for i in 0..<Int(geometry.mesh.vertexCount) {
            let v = geometry.mesh.vertices[i]
            vertices.append(SIMD3<Float>(v.x, v.y, v.z))
        }
        descriptor.positions = MeshBuffers.Positions(vertices)

        var indices: [UInt32] = []
        for i in 0..<Int(geometry.mesh.triangleCount) {
            let t = geometry.mesh.triangles[i]
            indices.append(t.indices.0)
            indices.append(t.indices.1)
            indices.append(t.indices.2)
        }
        descriptor.primitives = .triangles(indices)

        guard let mesh = try? MeshResource.generate(from: [descriptor]) else { return nil }

        let container = Entity()
        container.addChild(ModelEntity(mesh: mesh, materials: [OcclusionMaterial()]))

        let color: UIColor = geometry.type == .terrain
            ? UIColor(red: 0, green: 0.6, blue: 0, alpha: 0.35)
            : UIColor(red: 0.6, green: 0.2, blue: 0.9, alpha: 0.35)
        var debugMaterial = UnlitMaterial(color: color)
        debugMaterial.blending = .transparent(opacity: .init(floatLiteral: 0.35))
        debugMaterial.writesDepth = false
        let debugVisual = ModelEntity(mesh: mesh, materials: [debugMaterial])
        debugVisual.name = "debugVisual"
        container.addChild(debugVisual)

        return container
    }
}
