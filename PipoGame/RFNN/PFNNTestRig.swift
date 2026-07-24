import RealityKit
import ARKit
import simd

/// TEMP: minimal 2-joint skinned test rig (see scratchpad/build_test_rig.py)
/// to nail RealityKit's exact jointTransforms convention with far less
/// noise than the full 31-joint PFNN character. Cycles the child joint
/// between identity and a known 90-degree bend, translation always zero,
/// to see directly whether that hinges naturally at the joint the way a
/// real elbow would.
final class PFNNTestRig: ObservableObject {
    @Published var isPlaced = false
    weak var arView: ARView?
    private var anchor: AnchorEntity?
    private var meshEntity: ModelEntity?
    private var elapsed: Float = 0

    func toggle() {
        if isPlaced { remove() } else { spawn() }
    }

    private func spawn() {
        guard !isPlaced, let arView, let rig = try? Entity.load(named: "TestRig") else {
            PipoLog.log("PFNNTestRig: failed to load TestRig.usdz")
            return
        }
        guard let mesh = Self.firstMeshEntity(in: rig) else {
            PipoLog.log("PFNNTestRig: no mesh entity")
            return
        }
        // Documented RealityKit limitation: culling bounds for a skinned
        // mesh are computed from the REST pose, not the deformed one, so
        // posing a joint away from rest can move geometry outside those
        // bounds and get the whole mesh culled -- invisible from every
        // angle, exactly matching what was observed. boundsMargin is
        // Apple's own recommended fix (RealityKit engineer, dev forums).
        if var model = mesh.components[ModelComponent.self] {
            model.boundsMargin = 2.0
            mesh.components.set(model)
        }
        let cam = arView.cameraTransform
        var forward = SIMD3<Float>(-cam.matrix.columns.2.x, 0, -cam.matrix.columns.2.z)
        forward = simd_length_squared(forward) > 0.0001 ? simd_normalize(forward) : SIMD3<Float>(0, 0, 1)
        let spawnPosition = cam.translation + forward * 1.0

        let placementAnchor = AnchorEntity(world: spawnPosition)
        rig.scale = SIMD3<Float>(repeating: 0.3) // 2-unit-tall rig -> 0.6m, comfortable to view
        placementAnchor.addChild(rig)
        arView.scene.addAnchor(placementAnchor)

        self.anchor = placementAnchor
        self.meshEntity = mesh
        elapsed = 0
        isPlaced = true
        PipoLog.log("PFNNTestRig: spawned, jointNames=\(mesh.jointNames)")
    }

    private func remove() {
        anchor?.removeFromParent()
        anchor = nil
        meshEntity = nil
        isPlaced = false
    }

    // DEBUG: never touch jointTransforms at all, to see whether the
    // periodic full disappearance is inherent to this mesh/skeleton
    // (would still happen) or actually triggered by writing poses (would
    // not happen).
    enum DebugMode { case untouched, staticIdentityEveryFrame, bendCycle }
    var debugMode: DebugMode = .staticIdentityEveryFrame

    func update(deltaTime dt: Float) {
        guard isPlaced, let meshEntity else { return }
        if debugMode == .untouched { return }
        elapsed += dt

        if debugMode == .staticIdentityEveryFrame {
            // Calls the setter every single frame, but both joints are
            // ALWAYS identity -- isolates whether merely calling the
            // jointTransforms setter repeatedly (regardless of whether
            // any value actually changes) is enough to trigger the
            // disappearance, vs. it requiring an actual value change.
            var transforms = meshEntity.jointTransforms
            guard transforms.count >= 2 else { return }
            transforms[0] = Transform(rotation: simd_quatf(angle: 0, axis: [1, 0, 0]))
            transforms[1] = Transform(rotation: simd_quatf(angle: 0, axis: [1, 0, 0]))
            meshEntity.jointTransforms = transforms
            return
        }

        // Cycles every 3s: identity for 1.5s, then a 90-degree bend around
        // X (zero translation) for 1.5s -- should hinge cleanly at the
        // joint (y=1 in rest space) if "bone length structurally fixed,
        // rotation-only jointTransforms" is the correct model.
        let bent = elapsed.truncatingRemainder(dividingBy: 3.0) > 1.5
        var transforms = meshEntity.jointTransforms
        guard transforms.count >= 2 else { return }
        transforms[0] = Transform(rotation: simd_quatf(angle: 0, axis: [1, 0, 0]))
        transforms[1] = bent
            ? Transform(rotation: simd_quatf(angle: .pi / 2, axis: [1, 0, 0]))
            : Transform(rotation: simd_quatf(angle: 0, axis: [1, 0, 0]))
        meshEntity.jointTransforms = transforms
    }

    private static func firstMeshEntity(in entity: Entity) -> ModelEntity? {
        if let model = entity as? ModelEntity, model.model != nil {
            return model
        }
        for child in entity.children {
            if let found = firstMeshEntity(in: child) { return found }
        }
        return nil
    }
}
