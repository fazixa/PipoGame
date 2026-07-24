import RealityKit
import ARKit
import simd
import Combine

/// Orchestrates the PFNN puppet: owns the network/trajectory/character
/// state, assembles the network's 342-dim input vector and decodes its
/// 311-dim output exactly per `sreyafrancis/PFNN`'s `demo/pfnn.cpp`
/// (`pre_render()`/`post_render()`), and drives the puppet's skeleton via
/// RealityKit's own `jointTransforms` every frame — see `writeJointTransforms`'s
/// doc comment for why the skeleton/mesh are built NATIVELY in Swift from
/// the raw PFNN binaries rather than loaded from a USDZ.
///
/// Everything here (trajectory, joints, IK) runs in the PUPPET'S OWN
/// native units (matching its bundled rig/mesh, NOT meters) except at the
/// very edges: `nativeToMeters` scales the placed entity for real-world
/// display, and `groundHeightNative` is the only place that talks to the
/// real LiDAR mesh (in meters) and converts back. This mirrors PipoAsset's
/// own `worldScale` pattern. `Areas` (walls/crouch-zones/jump-pads) from
/// the original demo are dropped entirely -- no equivalent here, and out
/// of scope for this first "bring the puppet in and test it" phase.
final class PFNNController: ObservableObject {
    @Published var isPlaced = false
    var joystickInput: SIMD2<Float> = .zero

    private let joystickDeadzone: Float = 0.15
    // Native units per network step (matches pfnn.cpp's own
    // target_vel_speed range, 2.5...5.0) -- NOT derived from a real-world
    // m/s target. The network has no dt/timestep input anywhere; it was
    // trained assuming one prediction per displayed frame, so its own
    // notion of "a reasonable walk speed" IS this native-unit constant,
    // not something to re-derive from real-world units.
    private let nativeWalkSpeed: Float = 3.5

    // Puppet mesh height is ~157 native units (measured directly off
    // character_vertices.bin); scaling so that renders at ~1.7m, then
    // scaled down further to 0.3x (~0.5m) per request.
    private let nativeToMeters: Float = 0.0108 * 0.3

    private var network: PFNNNetwork?
    private let trajectory = PFNNTrajectory()
    private var character: PFNNCharacter?

    weak var arView: ARView?
    private var anchor: AnchorEntity?
    private var meshEntity: ModelEntity?
    private var anchorPosition: SIMD3<Float> = .zero

    // Joint names in the exact order character_parents.bin/build_puppet.py
    // use (independently verified earlier this session against parent
    // indices, e.g. parents[21]=20 LeftFingerBase->LeftHand).
    private static let jointNames = [
        "Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
        "RHipJoint", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
        "LowerBack", "Spine", "Spine1", "Neck", "Neck1", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftFingerBase",
        "LeftHandIndex1", "LThumb",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightFingerBase",
        "RightHandIndex1", "RThumb",
    ]

    init() {
        network = PFNNNetwork()
        if network == nil {
            PipoLog.log("PFNNController: failed to load PFNNWeights.bin")
        }
    }

    func toggle() {
        if isPlaced {
            remove()
        } else {
            spawn()
        }
    }

    private func spawn() {
        guard !isPlaced, let arView else { return }

        guard let (parents, restLocal) = Self.loadRig() else {
            PipoLog.log("PFNNController: failed to load rig binaries")
            return
        }
        guard let rawMesh = Self.loadRawMesh() else {
            PipoLog.log("PFNNController: failed to load raw mesh binaries")
            return
        }

        let newCharacter = PFNNCharacter(parents: parents, restLocal: restLocal)
        if let network {
            // pfnn.cpp's reset() seeds joint state from the network's own
            // mean output (a neutral standing pose), not zeros — see
            // PFNNCharacter.resetJointState's comment for why this matters.
            newCharacter.resetJointState(usingMeanOutput: network.yMean, rootPosition: .zero, rootYaw: 0)
        }
        character = newCharacter

        let cam = arView.cameraTransform
        var forward = SIMD3<Float>(-cam.matrix.columns.2.x, 0, -cam.matrix.columns.2.z)
        forward = simd_length_squared(forward) > 0.0001 ? simd_normalize(forward) : SIMD3<Float>(0, 0, 1)
        var spawnPosition = cam.translation + forward * 1.5
        // Screen-center raycast via ARKit's own query (allowing
        // .estimatedPlane), matching how Pipo itself is placed
        // (ARViewContainer.Coordinator.handleTap) — far more forgiving
        // than RealityKit's scene.raycast against .sceneUnderstanding
        // only, which requires the LiDAR mesh to already be fully
        // reconstructed at that exact spot.
        let screenCenter = CGPoint(x: arView.bounds.midX, y: arView.bounds.midY)
        let hits = arView.raycast(from: screenCenter, allowing: .estimatedPlane, alignment: .horizontal)
        if let hit = hits.first {
            let t = hit.worldTransform.columns.3
            spawnPosition = SIMD3<Float>(t.x, t.y, t.z)
        }
        anchorPosition = spawnPosition

        guard let mesh = Self.buildMeshResource(parents: parents, restLocal: restLocal, rawMesh: rawMesh) else {
            PipoLog.log("PFNNController: failed to build native skeleton/mesh resource")
            return
        }
        let entity = ModelEntity(mesh: mesh, materials: [SimpleMaterial(color: .white, isMetallic: false)])
        entity.scale = SIMD3<Float>(repeating: nativeToMeters)

        let placementAnchor = AnchorEntity(world: spawnPosition)
        placementAnchor.addChild(entity)
        arView.scene.addAnchor(placementAnchor)

        self.anchor = placementAnchor
        self.meshEntity = entity
        debugLoggedFirstFrame = false

        trajectory.reset(at: .zero, groundHeight: { _ in 0 })
        isPlaced = true
    }

    private func remove() {
        anchor?.removeFromParent()
        anchor = nil
        meshEntity = nil
        character = nil
        isPlaced = false
    }

    enum DebugMode { case decodeOnly, full }
    var debugMode: DebugMode = .full
    private var debugLoggedFirstFrame = false

    func update(deltaTime dt: Float) {
        guard isPlaced, let network, let character, arView != nil else { return }

        updateTargetFromJoystick()
        trajectory.updateGait(gaitSmooth: 0.1)
        trajectory.predictFuture(strafeAmount: 0, responsive: false, groundHeight: groundHeightNative)

        // pre_render(): decode using the CURRENT (not-yet-advanced) root.
        let rootPosition = trajectory.rootPosition
        let rootYaw = trajectory.rootYaw
        let xp = assembleInput(character: character)
        let yp = network.predict(xp, phase: character.phase)

        if !debugLoggedFirstFrame {
            debugLoggedFirstFrame = true
            PipoLog.log("PFNN frame1: xp[0..6)=\(xp[0..<6]) yp[0..8)=\(yp[0..<8])")
        }

        character.decode(yp: yp, rootPosition: rootPosition, rootYaw: rootYaw)

        if debugMode == .full {
            character.applyIK(yp: yp, groundHeight: groundHeightNative)
        }

        writeJointTransforms(character: character)

        // post_render(): advance the trajectory/phase for the NEXT frame.
        trajectory.rollPastForward()
        let standAmount = powf(1 - trajectory.gaitStand[PFNNTrajectory.half], 0.25)
        trajectory.updateCurrent(rootDelta: SIMD2(yp[0], yp[1]), rootTurn: yp[2], standAmount: standAmount)
        updateFutureTrajectoryVerbatim(yp: yp)

        character.phase = character.phase + (standAmount * 0.9 + 0.1) * 2 * .pi * yp[3]
        character.phase = character.phase.truncatingRemainder(dividingBy: 2 * .pi)
        if character.phase < 0 { character.phase += 2 * .pi }
    }

    private func updateTargetFromJoystick() {
        guard let arView else { return }
        let magnitude = simd_length(joystickInput)
        let cam = arView.cameraTransform.matrix
        var forward = SIMD3<Float>(-cam.columns.2.x, 0, -cam.columns.2.z)
        if simd_length_squared(forward) < 0.0001 {
            forward = SIMD3<Float>(cam.columns.1.x, 0, cam.columns.1.z)
        }
        forward = simd_normalize(forward)
        let right = SIMD3<Float>(-forward.z, 0, forward.x)

        guard magnitude > joystickDeadzone else {
            trajectory.targetVel = .zero
            return
        }
        let direction = simd_normalize(right * joystickInput.x + forward * joystickInput.y)
        let throttle = min((magnitude - joystickDeadzone) / (1 - joystickDeadzone), 1)
        trajectory.targetDir = direction
        trajectory.targetVel = direction * (nativeWalkSpeed * throttle)
    }

    /// Converts a native-space XZ into the real AR world, raycasts the
    /// real LiDAR scene mesh (mirroring `PipoController.meshGroundHeight`),
    /// and converts the hit back into native-space height. Falls back to
    /// native Y=0 (the anchor's own placement height) if nothing is hit.
    private func groundHeightNative(_ native: SIMD2<Float>) -> Float {
        guard let arView else { return 0 }
        let worldX = anchorPosition.x + native.x * nativeToMeters
        let worldZ = anchorPosition.z + native.y * nativeToMeters
        let from = SIMD3<Float>(worldX, anchorPosition.y + 1.0, worldZ)
        let to = SIMD3<Float>(worldX, anchorPosition.y - 2.0, worldZ)
        guard let hit = arView.scene.raycast(from: from, to: to, query: .nearest, mask: .sceneUnderstanding).first else {
            return 0
        }
        return (hit.position.y - anchorPosition.y) / nativeToMeters
    }

    /// pre_render()'s full Xp assembly -- trajectory pos/dir/gait sampled
    /// every 10th frame across the whole 120-sample window (w=12), the
    /// previous frame's joint positions/velocities (31 joints x 3 x 2), and
    /// terrain heights sampled either side of the path (w=12 again).
    /// Index math ported verbatim from pfnn.cpp -- see PFNNNetwork.swift's
    /// header for the exact XDIM=342 breakdown this reproduces.
    private func assembleInput(character: PFNNCharacter) -> [Float] {
        var xp = [Float](repeating: 0, count: PFNNNetwork.xDim)
        let w = PFNNTrajectory.length / 10 // 12
        let rootPosition = trajectory.rootPosition
        let rootYaw = trajectory.rootYaw

        var i = 0
        while i < PFNNTrajectory.length {
            let idx = i / 10
            let posLocal = rotateY(trajectory.positions[i] - rootPosition, by: -rootYaw)
            let dirLocal = rotateY(trajectory.directions[i], by: -rootYaw)
            xp[w * 0 + idx] = posLocal.x
            xp[w * 1 + idx] = posLocal.z
            xp[w * 2 + idx] = dirLocal.x
            xp[w * 3 + idx] = dirLocal.z
            xp[w * 4 + idx] = trajectory.gaitStand[i]
            xp[w * 5 + idx] = trajectory.gaitWalk[i]
            xp[w * 6 + idx] = trajectory.gaitJog[i]
            xp[w * 7 + idx] = trajectory.gaitCrouch[i]
            xp[w * 8 + idx] = trajectory.gaitJump[i]
            xp[w * 9 + idx] = 0 // unused, matches pfnn.cpp
            i += 10
        }

        let prevRootPosition = trajectory.prevRootPosition
        let prevRootYaw = trajectory.prevRootYaw
        let o1 = (PFNNTrajectory.length / 10) * 10 // 120
        let jc = PFNNCharacter.jointCount
        for j in 0..<jc {
            let pos = rotateY(character.jointPositions[j] - prevRootPosition, by: -prevRootYaw)
            let vel = rotateY(character.jointVelocities[j], by: -prevRootYaw)
            xp[o1 + jc * 3 * 0 + j * 3 + 0] = pos.x
            xp[o1 + jc * 3 * 0 + j * 3 + 1] = pos.y
            xp[o1 + jc * 3 * 0 + j * 3 + 2] = pos.z
            xp[o1 + jc * 3 * 1 + j * 3 + 0] = vel.x
            xp[o1 + jc * 3 * 1 + j * 3 + 1] = vel.y
            xp[o1 + jc * 3 * 1 + j * 3 + 2] = vel.z
        }

        let o2 = o1 + jc * 3 * 2 // 306
        i = 0
        while i < PFNNTrajectory.length {
            let idx = i / 10
            let yaw = yawAngle(of: trajectory.directions[i])
            let posR = trajectory.positions[i] + rotateY(SIMD3<Float>(trajectory.width, 0, 0), by: yaw)
            let posL = trajectory.positions[i] + rotateY(SIMD3<Float>(-trajectory.width, 0, 0), by: yaw)
            xp[o2 + w * 0 + idx] = groundHeightNative(SIMD2(posR.x, posR.z)) - rootPosition.y
            xp[o2 + w * 1 + idx] = trajectory.positions[i].y - rootPosition.y
            xp[o2 + w * 2 + idx] = groundHeightNative(SIMD2(posL.x, posL.z)) - rootPosition.y
            i += 10
        }

        return xp
    }

    /// post_render()'s "Update Future Trajectory" -- reconstructs the
    /// future half of the trajectory window from the network's predicted
    /// Yp[8...] samples. Ported as a direct, literal index-for-index copy
    /// (see PFNNTrajectory.swift for why this isn't abstracted further).
    private func updateFutureTrajectoryVerbatim(yp: [Float]) {
        let w = PFNNTrajectory.half / 10 // 6
        let currentRootPosition = trajectory.positions[PFNNTrajectory.half]
        let currentRootYaw = trajectory.rootYaw
        for i in (PFNNTrajectory.half + 1)..<PFNNTrajectory.length {
            let m = (Float(i - PFNNTrajectory.half)).truncatingRemainder(dividingBy: 10) / 10
            let lo = i / 10 - w
            var pos = SIMD3<Float>(
                (1 - m) * yp[8 + w * 0 + lo] + m * yp[8 + w * 0 + lo + 1],
                0,
                (1 - m) * yp[8 + w * 1 + lo] + m * yp[8 + w * 1 + lo + 1])
            var dir = SIMD3<Float>(
                (1 - m) * yp[8 + w * 2 + lo] + m * yp[8 + w * 2 + lo + 1],
                0,
                (1 - m) * yp[8 + w * 3 + lo] + m * yp[8 + w * 3 + lo + 1])
            pos = rotateY(pos, by: currentRootYaw) + currentRootPosition
            dir = simd_normalize(rotateY(dir, by: currentRootYaw))
            trajectory.positions[i] = pos
            trajectory.directions[i] = dir
        }
    }

    /// Writes each joint's FULL local transform (translation AND rotation,
    /// via `Transform(matrix:)`) into the puppet's `jointTransforms` every
    /// frame, driving RealityKit's own GPU skeletal skinning directly.
    ///
    /// This works cleanly ONLY because the skeleton/mesh are built NATIVELY
    /// in Swift (see `buildMeshResource`) instead of loaded from a USDZ.
    /// An extensive investigation this session (a dedicated Swift/
    /// RealityKit sandbox testing well over a dozen jointTransforms
    /// conventions against a USDZ-loaded version of this same rig/mesh —
    /// rotation-only, full transforms, absolute vs. delta rotation, both
    /// multiplication orders, in-place mutation, zero-length connector
    /// bones, non-identity rest rotations, a known CAMDM precedent from
    /// another branch) found that RealityKit's import of THIS specific
    /// hand-authored USDZ silently drops/mishandles joint rest translation
    /// (its own default `jointTransforms` always read back (0,0,0)
    /// regardless of what was authored), causing torso/collar distortion
    /// no matter how jointTransforms was written on that asset. Building
    /// the IDENTICAL skeleton and mesh natively via `MeshResource.Skeleton`/
    /// `MeshResource.Contents` (bypassing USD import entirely) reports the
    /// correct rest translation by default and renders perfectly cleanly
    /// with this same full-transform write, across an entire walk cycle.
    private func writeJointTransforms(character: PFNNCharacter) {
        guard let meshEntity else { return }
        var transforms = meshEntity.jointTransforms
        for i in 0..<min(PFNNCharacter.jointCount, transforms.count) {
            transforms[i] = Transform(matrix: character.jointAnimLocal[i])
        }
        meshEntity.jointTransforms = transforms
    }

    /// Builds the puppet's skeleton and skinned mesh entirely in Swift from
    /// the raw PFNN binaries -- no USDZ/USD import involved at all (see
    /// `writeJointTransforms`'s doc comment for why this matters).
    private static func buildMeshResource(
        parents: [Int], restLocal: [simd_float4x4],
        rawMesh: (positions: [SIMD3<Float>], normals: [SIMD3<Float>], weights: [SIMD4<Float>], jointIndices: [SIMD4<Int32>], triangles: [UInt32])
    ) -> MeshResource? {
        let jointCount = PFNNCharacter.jointCount
        var worldRest = [simd_float4x4](repeating: matrix_identity_float4x4, count: jointCount)
        for i in 0..<jointCount {
            let p = parents[i]
            worldRest[i] = p == -1 ? restLocal[i] : worldRest[p] * restLocal[i]
        }
        let inverseBindPoseMatrices = worldRest.map { $0.inverse }
        let restPoseTransforms = restLocal.map { Transform(matrix: $0) }
        let parentIndices: [Int?] = parents.map { $0 == -1 ? nil : $0 }

        guard let skeleton = MeshResource.Skeleton(
            id: "PFNNSkeleton", jointNames: jointNames,
            inverseBindPoseMatrices: inverseBindPoseMatrices,
            restPoseTransforms: restPoseTransforms, parentIndices: parentIndices
        ) else { return nil }

        var influencesFlat: [MeshJointInfluence] = []
        influencesFlat.reserveCapacity(rawMesh.positions.count * 4)
        for v in 0..<rawMesh.positions.count {
            let w = rawMesh.weights[v], idx = rawMesh.jointIndices[v]
            for k in 0..<4 {
                influencesFlat.append(MeshJointInfluence(jointIndex: Int(idx[k]), weight: w[k]))
            }
        }

        var descriptor = MeshDescriptor(name: "PFNNPuppet")
        descriptor.positions = MeshBuffer(rawMesh.positions)
        descriptor.normals = MeshBuffer(rawMesh.normals)
        descriptor.primitives = .triangles(rawMesh.triangles)

        guard var model = try? MeshResource.Model(id: "PFNNPuppetModel", descriptors: [descriptor]) else { return nil }
        var newParts: [MeshResource.Part] = []
        for var part in model.parts {
            part.jointInfluences = MeshResource.JointInfluences(influences: MeshBuffer(influencesFlat), influencesPerVertex: 4)
            part.skeletonID = "PFNNSkeleton"
            newParts.append(part)
        }
        model.parts = MeshPartCollection(newParts)

        var contents = MeshResource.Contents()
        contents.models = MeshModelCollection([model])
        contents.skeletons = MeshSkeletonCollection([skeleton])
        contents.instances = MeshInstanceCollection([MeshResource.Instance(id: "PFNNPuppetInstance", model: "PFNNPuppetModel")])

        return try? MeshResource.generate(from: contents)
    }

    /// Parses `character_vertices.bin`/`character_triangles.bin` directly
    /// (bundled as PFNNCharacterVertices.bin / PFNNCharacterTriangles.bin).
    /// Each vertex is 15 floats: pos.xyz, normal.xyz, ao, weight x4,
    /// jointIndex x4 (see build_puppet.py's own header).
    private static func loadRawMesh() -> (positions: [SIMD3<Float>], normals: [SIMD3<Float>], weights: [SIMD4<Float>], jointIndices: [SIMD4<Int32>], triangles: [UInt32])? {
        guard let vertsURL = Bundle.main.url(forResource: "PFNNCharacterVertices", withExtension: "bin"),
              let trisURL = Bundle.main.url(forResource: "PFNNCharacterTriangles", withExtension: "bin"),
              let vertsData = try? Data(contentsOf: vertsURL),
              let trisData = try? Data(contentsOf: trisURL) else {
            return nil
        }

        let floatCount = vertsData.count / MemoryLayout<Float>.size
        guard floatCount % 15 == 0 else { return nil }
        let vertexCount = floatCount / 15
        var floats = [Float](repeating: 0, count: floatCount)
        _ = floats.withUnsafeMutableBytes { vertsData.copyBytes(to: $0) }

        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var weights: [SIMD4<Float>] = []
        var jointIndices: [SIMD4<Int32>] = []
        positions.reserveCapacity(vertexCount)
        normals.reserveCapacity(vertexCount)
        weights.reserveCapacity(vertexCount)
        jointIndices.reserveCapacity(vertexCount)
        for v in 0..<vertexCount {
            let o = v * 15
            positions.append(SIMD3<Float>(floats[o], floats[o + 1], floats[o + 2]))
            normals.append(SIMD3<Float>(floats[o + 3], floats[o + 4], floats[o + 5]))
            weights.append(SIMD4<Float>(floats[o + 7], floats[o + 8], floats[o + 9], floats[o + 10]))
            jointIndices.append(SIMD4<Int32>(Int32(floats[o + 11]), Int32(floats[o + 12]), Int32(floats[o + 13]), Int32(floats[o + 14])))
        }

        guard trisData.count % MemoryLayout<UInt32>.size == 0 else { return nil }
        let triCount = trisData.count / MemoryLayout<UInt32>.size
        var triangles = [UInt32](repeating: 0, count: triCount)
        _ = triangles.withUnsafeMutableBytes { trisData.copyBytes(to: $0) }

        return (positions, normals, weights, jointIndices, triangles)
    }

    /// Parses `character_parents.bin`/`character_xforms.bin` directly
    /// (bundled as PFNNCharacterParents.bin / PFNNCharacterRestXforms.bin).
    private static func loadRig() -> (parents: [Int], restLocal: [simd_float4x4])? {
        guard let parentsURL = Bundle.main.url(forResource: "PFNNCharacterParents", withExtension: "bin"),
              let xformsURL = Bundle.main.url(forResource: "PFNNCharacterRestXforms", withExtension: "bin"),
              let parentsData = try? Data(contentsOf: parentsURL),
              let xformsData = try? Data(contentsOf: xformsURL) else {
            return nil
        }

        let jointCount = PFNNCharacter.jointCount
        guard parentsData.count == jointCount * MemoryLayout<Float>.size,
              xformsData.count == jointCount * 16 * MemoryLayout<Float>.size else {
            return nil
        }

        var parentFloats = [Float](repeating: 0, count: jointCount)
        _ = parentFloats.withUnsafeMutableBytes { parentsData.copyBytes(to: $0) }
        let parents = parentFloats.map { Int($0) }

        var xformFloats = [Float](repeating: 0, count: jointCount * 16)
        _ = xformFloats.withUnsafeMutableBytes { xformsData.copyBytes(to: $0) }

        var restLocal: [simd_float4x4] = []
        restLocal.reserveCapacity(jointCount)
        for j in 0..<jointCount {
            let o = j * 16
            // The file stores each matrix row-major; building each column
            // by taking the same position from every 4-float row-group
            // performs the transpose simd_float4x4 needs (verified this
            // session against the raw bytes directly).
            let m = simd_float4x4(
                SIMD4<Float>(xformFloats[o + 0], xformFloats[o + 4], xformFloats[o + 8], xformFloats[o + 12]),
                SIMD4<Float>(xformFloats[o + 1], xformFloats[o + 5], xformFloats[o + 9], xformFloats[o + 13]),
                SIMD4<Float>(xformFloats[o + 2], xformFloats[o + 6], xformFloats[o + 10], xformFloats[o + 14]),
                SIMD4<Float>(xformFloats[o + 3], xformFloats[o + 7], xformFloats[o + 11], xformFloats[o + 15]))
            restLocal.append(m)
        }

        return (parents, restLocal)
    }
}
