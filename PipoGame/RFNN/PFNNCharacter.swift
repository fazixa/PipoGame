import simd

/// pfnn.cpp's `quat_exp` — decodes a 3-component exponential-map rotation
/// (the network's per-joint rotation output) into a quaternion.
func quatExp(_ l: SIMD3<Float>) -> simd_quatf {
    let w = simd_length(l)
    if w < 0.01 { return simd_quatf(real: 1, imag: .zero) }
    let s = sin(w) / w
    return simd_normalize(simd_quatf(real: cos(w), imag: l * s))
}

/// A pure yaw rotation matrix, matching `glm::rotate(angle, (0,1,0))`.
func yawMatrix(_ angle: Float) -> simd_float3x3 {
    let c = cos(angle), s = sin(angle)
    return simd_float3x3(columns: (
        SIMD3<Float>(c, 0, -s),
        SIMD3<Float>(0, 1, 0),
        SIMD3<Float>(s, 0, c)))
}

func embed4x4(_ rotation: simd_float3x3) -> simd_float4x4 {
    var m = matrix_identity_float4x4
    m.columns.0 = SIMD4<Float>(rotation.columns.0, 0)
    m.columns.1 = SIMD4<Float>(rotation.columns.1, 0)
    m.columns.2 = SIMD4<Float>(rotation.columns.2, 0)
    return m
}

func makeTransform(_ rotation: simd_float3x3, _ translation: SIMD3<Float>) -> simd_float4x4 {
    var m = embed4x4(rotation)
    m.columns.3 = SIMD4<Float>(translation, 1)
    return m
}

func upperLeft3x3(_ m: simd_float4x4) -> simd_float3x3 {
    simd_float3x3(
        SIMD3<Float>(m.columns.0.x, m.columns.0.y, m.columns.0.z),
        SIMD3<Float>(m.columns.1.x, m.columns.1.y, m.columns.1.z),
        SIMD3<Float>(m.columns.2.x, m.columns.2.y, m.columns.2.z))
}

/// Overwrites only the rotation (upper-left 3x3) of a transform, leaving
/// its translation untouched — matches pfnn.cpp's `two_joint()` writing
/// only `a_lR[x][y]` for x,y in 0..2.
func replacingRotation(_ m: simd_float4x4, with rot: simd_float3x3) -> simd_float4x4 {
    var result = m
    result.columns.0 = SIMD4<Float>(rot.columns.0, m.columns.0.w)
    result.columns.1 = SIMD4<Float>(rot.columns.1, m.columns.1.w)
    result.columns.2 = SIMD4<Float>(rot.columns.2, m.columns.2.w)
    return result
}

func mix(_ a: SIMD4<Float>, _ b: SIMD4<Float>, t: Float) -> SIMD4<Float> { a * (1 - t) + b * t }

/// pfnn.cpp's `mix_transforms` — slerps rotation, lerps translation.
func mixTransforms(_ x: simd_float4x4, _ y: simd_float4x4, _ a: Float) -> simd_float4x4 {
    let qx = simd_quatf(upperLeft3x3(x))
    let qy = simd_quatf(upperLeft3x3(y))
    let q = simd_slerp(qx, qy, a)
    var out = embed4x4(simd_float3x3(q))
    out.columns.3 = mix(x.columns.3, y.columns.3, t: a)
    return out
}

/// Ports the `Character` struct (31-joint CMU-style skeleton state +
/// forward kinematics) and the `IK` struct (built-in two-bone leg IK +
/// heel/toe foot-plant locking) from `sreyafrancis/PFNN`'s `demo/pfnn.cpp`.
/// The IK is driven by the network's OWN predicted per-foot contact
/// confidence (`Yp[4..<8)`), locking a foot's world position the instant
/// that confidence crosses a threshold and fading it back out — the lock
/// only ever reads a fresh measurement at the moment it engages, avoiding
/// the feedback-loop bug hit earlier this session with a hand-built,
/// live-pose-driven foot lock.
final class PFNNCharacter {
    static let jointCount = 31

    // Named joint indices, matching skeletondef.py's 31-joint ordering
    // (independently verified against character_parents.bin this session:
    // e.g. parents[21]=20 LeftFingerBase->LeftHand, parents[17]=13
    // LeftShoulder->Spine1, consistent throughout).
    static let rootL = 1, hipL = 2, kneeL = 3, heelL = 4, toeL = 5
    static let rootR = 6, hipR = 7, kneeR = 8, heelR = 9, toeR = 10

    let jointParents: [Int]
    let jointRestLocal: [simd_float4x4]     // loaded once, local/parent-relative
    var jointAnimLocal: [simd_float4x4]     // per-frame, local/parent-relative
    var jointGlobalRest: [simd_float4x4]
    var jointGlobalAnim: [simd_float4x4]

    // Smoothed pos/vel, kept ONLY for next frame's input-vector self
    // feedback (pre_render()'s "Input Joint Previous Positions/Velocities")
    // -- the pose actually rendered/IK'd each frame uses the network's raw
    // predicted position/rotation directly, not this smoothed value.
    var jointPositions: [SIMD3<Float>]
    var jointVelocities: [SIMD3<Float>]

    var phase: Float = 0
    var jointSmooth: Float = 0.5 // pfnn.cpp's extra_joint_smooth

    // IK lock state, indices HL=0, HR=1, TL=2, TR=3.
    private var ikLock = [Float](repeating: 0, count: 4)
    private var ikLockPosition = [SIMD3<Float>](repeating: .zero, count: 4)
    private var ikHeight = [Float](repeating: 0, count: 4)
    private let ikFade: Float = 0.075
    private let ikThreshold: Float = 0.8
    private let ikSmoothness: Float = 0.5
    private let ikHeelHeight: Float = 5.0
    private let ikToeHeight: Float = 4.0

    init(parents: [Int], restLocal: [simd_float4x4]) {
        precondition(parents.count == Self.jointCount && restLocal.count == Self.jointCount)
        jointParents = parents
        jointRestLocal = restLocal
        jointAnimLocal = restLocal
        jointGlobalRest = Array(repeating: matrix_identity_float4x4, count: Self.jointCount)
        jointGlobalAnim = Array(repeating: matrix_identity_float4x4, count: Self.jointCount)
        jointPositions = Array(repeating: .zero, count: Self.jointCount)
        jointVelocities = Array(repeating: .zero, count: Self.jointCount)
        forwardKinematics()
    }

    /// pfnn.cpp's `reset()` seeds `joint_positions`/`joint_velocities` from
    /// the network's own trained MEAN output (`Ymean` — a neutral standing
    /// pose baked into the normalization statistics), not zeros. Skipping
    /// this and zero-initializing instead means frame 1 feeds the network
    /// a "previous pose" with every joint coincident at the root — wildly
    /// outside anything it was trained on, which is what an exploded first
    /// frame of skinning looks like.
    func resetJointState(usingMeanOutput yMean: [Float], rootPosition: SIMD3<Float>, rootYaw: Float) {
        let rootRot = yawMatrix(rootYaw)
        let opos = 32, ovel = 32 + Self.jointCount * 3
        for i in 0..<Self.jointCount {
            let pos = rootRot * SIMD3<Float>(yMean[opos + i * 3], yMean[opos + i * 3 + 1], yMean[opos + i * 3 + 2]) + rootPosition
            let vel = rootRot * SIMD3<Float>(yMean[ovel + i * 3], yMean[ovel + i * 3 + 1], yMean[ovel + i * 3 + 2])
            jointPositions[i] = pos
            jointVelocities[i] = vel
        }
        phase = 0
    }

    /// pfnn.cpp's `Character::forward_kinematics()`. Joints are stored
    /// parent-before-child (verified against character_parents.bin), so a
    /// single forward pass suffices instead of walking to the root for
    /// every joint.
    func forwardKinematics() {
        for i in 0..<Self.jointCount {
            let p = jointParents[i]
            jointGlobalAnim[i] = p == -1 ? jointAnimLocal[i] : jointGlobalAnim[p] * jointAnimLocal[i]
            jointGlobalRest[i] = p == -1 ? jointRestLocal[i] : jointGlobalRest[p] * jointRestLocal[i]
        }
    }

    /// `pre_render()`'s "Build Local Transforms" + "Convert to local
    /// space": decodes the network's predicted per-joint position/
    /// velocity/rotation (root-relative) into world-space joint state,
    /// then derives the LOCAL (parent-relative) anim transforms
    /// RealityKit's jointTransforms need.
    func decode(yp: [Float], rootPosition: SIMD3<Float>, rootYaw: Float) {
        let rootRot = yawMatrix(rootYaw)
        let opos = 32
        let ovel = 32 + Self.jointCount * 3
        let orot = 32 + Self.jointCount * 3 * 2

        for i in 0..<Self.jointCount {
            let pos = rootRot * SIMD3<Float>(yp[opos + i * 3], yp[opos + i * 3 + 1], yp[opos + i * 3 + 2]) + rootPosition
            let vel = rootRot * SIMD3<Float>(yp[ovel + i * 3], yp[ovel + i * 3 + 1], yp[ovel + i * 3 + 2])
            let localRot = simd_float3x3(quatExp(SIMD3<Float>(yp[orot + i * 3], yp[orot + i * 3 + 1], yp[orot + i * 3 + 2])))
            let rot = rootRot * localRot

            jointPositions[i] = mix(jointPositions[i] + vel, pos, t: jointSmooth)
            jointVelocities[i] = vel
            jointGlobalAnim[i] = makeTransform(rot, pos)
        }

        for i in 0..<Self.jointCount {
            jointAnimLocal[i] = i == 0 ? jointGlobalAnim[i] : jointGlobalAnim[jointParents[i]].inverse * jointGlobalAnim[i]
        }
        forwardKinematics()
    }

    /// pfnn.cpp's `IK::two_joint` — analytic two-bone solve rotating
    /// joints `a` (hip) and `b` (knee) so that `c` (heel), a fixed distance
    /// from `b`, reaches world target `t`.
    private func twoJoint(a: SIMD3<Float>, b: SIMD3<Float>, c: SIMD3<Float>, t: SIMD3<Float>, eps: Float,
                          aParentGlobal: simd_float4x4, bParentGlobal: simd_float4x4,
                          aGlobal: simd_float4x4, bGlobal: simd_float4x4,
                          aLocal: inout simd_float4x4, bLocal: inout simd_float4x4) {
        let lc = simd_length(b - a)
        let la = simd_length(b - c)
        let lt = simd_clamp(simd_length(t - a), eps, lc + la - eps)

        if simd_length(c - t) < eps { return }

        let acAb0 = acos(simd_clamp(simd_dot(simd_normalize(c - a), simd_normalize(b - a)), -1, 1))
        let baBc0 = acos(simd_clamp(simd_dot(simd_normalize(a - b), simd_normalize(c - b)), -1, 1))
        let acAt0 = acos(simd_clamp(simd_dot(simd_normalize(c - a), simd_normalize(t - a)), -1, 1))

        let acAb1 = acos(simd_clamp((la * la - lc * lc - lt * lt) / (-2 * lc * lt), -1, 1))
        let baBc1 = acos(simd_clamp((lt * lt - lc * lc - la * la) / (-2 * lc * la), -1, 1))

        let a0 = simd_normalize(simd_cross(b - a, c - a))
        let a1 = simd_normalize(simd_cross(t - a, c - a))

        let r0 = simd_float3x3(simd_quatf(angle: acAb1 - acAb0, axis: -a0))
        let r1 = simd_float3x3(simd_quatf(angle: baBc1 - baBc0, axis: -a0))
        let r2 = simd_float3x3(simd_quatf(angle: acAt0, axis: -a1))

        let aLocalRot = upperLeft3x3(aParentGlobal).inverse * (r2 * r0 * upperLeft3x3(aGlobal))
        let bLocalRot = upperLeft3x3(bParentGlobal).inverse * (r1 * upperLeft3x3(bGlobal))

        aLocal = replacingRotation(aLocal, with: aLocalRot)
        bLocal = replacingRotation(bLocal, with: bLocalRot)
    }

    /// The full "Perform IK" block from `pre_render()`: hip/knee two-bone
    /// solve toward a ground-corrected heel target, heel/toe rotation to
    /// align with the ground plane, and the lock/threshold/fade foot-plant
    /// state machine, all driven by the network's own predicted per-foot
    /// contact confidence (`yp[4..<8)`).
    func applyIK(yp: [Float], groundHeight: (SIMD2<Float>) -> Float) {
        let ikWeight = SIMD4<Float>(yp[4], yp[5], yp[6], yp[7])

        var keyHL = SIMD3<Float>(jointGlobalAnim[Self.heelL].columns.3.x, jointGlobalAnim[Self.heelL].columns.3.y, jointGlobalAnim[Self.heelL].columns.3.z)
        var keyTL = SIMD3<Float>(jointGlobalAnim[Self.toeL].columns.3.x, jointGlobalAnim[Self.toeL].columns.3.y, jointGlobalAnim[Self.toeL].columns.3.z)
        var keyHR = SIMD3<Float>(jointGlobalAnim[Self.heelR].columns.3.x, jointGlobalAnim[Self.heelR].columns.3.y, jointGlobalAnim[Self.heelR].columns.3.z)
        var keyTR = SIMD3<Float>(jointGlobalAnim[Self.toeR].columns.3.x, jointGlobalAnim[Self.toeR].columns.3.y, jointGlobalAnim[Self.toeR].columns.3.z)

        keyHL = mix(keyHL, ikLockPosition[0], t: ikLock[0])
        keyTL = mix(keyTL, ikLockPosition[2], t: ikLock[2])
        keyHR = mix(keyHR, ikLockPosition[1], t: ikLock[1])
        keyTR = mix(keyTR, ikLockPosition[3], t: ikLock[3])

        ikHeight[0] = mix(ikHeight[0], groundHeight(SIMD2(keyHL.x, keyHL.z)) + ikHeelHeight, t: ikSmoothness)
        ikHeight[2] = mix(ikHeight[2], groundHeight(SIMD2(keyTL.x, keyTL.z)) + ikToeHeight, t: ikSmoothness)
        ikHeight[1] = mix(ikHeight[1], groundHeight(SIMD2(keyHR.x, keyHR.z)) + ikHeelHeight, t: ikSmoothness)
        ikHeight[3] = mix(ikHeight[3], groundHeight(SIMD2(keyTR.x, keyTR.z)) + ikToeHeight, t: ikSmoothness)

        keyHL.y = max(keyHL.y, ikHeight[0])
        keyTL.y = max(keyTL.y, ikHeight[2])
        keyHR.y = max(keyHR.y, ikHeight[1])
        keyTR.y = max(keyTR.y, ikHeight[3])

        // Rotate hip/knee.
        do {
            let hipL3 = column3(jointGlobalAnim[Self.hipL]), kneeL3 = column3(jointGlobalAnim[Self.kneeL]), heelL3 = column3(jointGlobalAnim[Self.heelL])
            let hipR3 = column3(jointGlobalAnim[Self.hipR]), kneeR3 = column3(jointGlobalAnim[Self.kneeR]), heelR3 = column3(jointGlobalAnim[Self.heelR])

            var hipLocal = jointAnimLocal[Self.hipL], kneeLocal = jointAnimLocal[Self.kneeL]
            twoJoint(a: hipL3, b: kneeL3, c: heelL3, t: keyHL, eps: 1.0,
                     aParentGlobal: jointGlobalAnim[Self.rootL], bParentGlobal: jointGlobalAnim[Self.hipL],
                     aGlobal: jointGlobalAnim[Self.hipL], bGlobal: jointGlobalAnim[Self.kneeL],
                     aLocal: &hipLocal, bLocal: &kneeLocal)
            jointAnimLocal[Self.hipL] = hipLocal
            jointAnimLocal[Self.kneeL] = kneeLocal

            var hipRLocal = jointAnimLocal[Self.hipR], kneeRLocal = jointAnimLocal[Self.kneeR]
            twoJoint(a: hipR3, b: kneeR3, c: heelR3, t: keyHR, eps: 1.0,
                     aParentGlobal: jointGlobalAnim[Self.rootR], bParentGlobal: jointGlobalAnim[Self.hipR],
                     aGlobal: jointGlobalAnim[Self.hipR], bGlobal: jointGlobalAnim[Self.kneeR],
                     aLocal: &hipRLocal, bLocal: &kneeRLocal)
            jointAnimLocal[Self.hipR] = hipRLocal
            jointAnimLocal[Self.kneeR] = kneeRLocal

            forwardKinematics()
        }

        // Rotate heel to align with the ground plane under it.
        do {
            let bendS: Float = 4, bendU: Float = 4, bendD: Float = 4
            let blend = simd_clamp(ikWeight * 2.5, SIMD4<Float>(repeating: 0), SIMD4<Float>(repeating: 1))

            let heelL = column3(jointGlobalAnim[Self.heelL])
            let side0L4 = jointGlobalAnim[Self.heelL] * SIMD4<Float>(10, 0, 0, 1)
            let side1L4 = jointGlobalAnim[Self.heelL] * SIMD4<Float>(-10, 0, 0, 1)
            var side0L = SIMD3<Float>(side0L4.x, side0L4.y, side0L4.z) / side0L4.w
            var side1L = SIMD3<Float>(side1L4.x, side1L4.y, side1L4.z) / side1L4.w
            var floorL = keyTL

            side0L.y = simd_clamp(groundHeight(SIMD2(side0L.x, side0L.z)) + ikToeHeight, heelL.y - bendS, heelL.y + bendS)
            side1L.y = simd_clamp(groundHeight(SIMD2(side1L.x, side1L.z)) + ikToeHeight, heelL.y - bendS, heelL.y + bendS)
            floorL.y = simd_clamp(floorL.y, heelL.y - bendD, heelL.y + bendU)

            let targZL = simd_normalize(floorL - heelL)
            var targXL = simd_normalize(side0L - side1L)
            let targYL = simd_normalize(simd_cross(targXL, targZL))
            targXL = simd_cross(targZL, targYL)

            jointAnimLocal[Self.heelL] = mixTransforms(
                jointAnimLocal[Self.heelL],
                jointGlobalAnim[Self.kneeL].inverse * simd_float4x4(
                    SIMD4<Float>(targXL, 0), SIMD4<Float>(-targYL, 0), SIMD4<Float>(targZL, 0), SIMD4<Float>(heelL, 1)),
                blend.y)

            let heelR = column3(jointGlobalAnim[Self.heelR])
            let side0R4 = jointGlobalAnim[Self.heelR] * SIMD4<Float>(10, 0, 0, 1)
            let side1R4 = jointGlobalAnim[Self.heelR] * SIMD4<Float>(-10, 0, 0, 1)
            var side0R = SIMD3<Float>(side0R4.x, side0R4.y, side0R4.z) / side0R4.w
            var side1R = SIMD3<Float>(side1R4.x, side1R4.y, side1R4.z) / side1R4.w
            var floorR = keyTR

            side0R.y = simd_clamp(groundHeight(SIMD2(side0R.x, side0R.z)) + ikToeHeight, heelR.y - bendS, heelR.y + bendS)
            side1R.y = simd_clamp(groundHeight(SIMD2(side1R.x, side1R.z)) + ikToeHeight, heelR.y - bendS, heelR.y + bendS)
            floorR.y = simd_clamp(floorR.y, heelR.y - bendD, heelR.y + bendU)

            let targZR = simd_normalize(floorR - heelR)
            var targXR = simd_normalize(side0R - side1R)
            let targYR = simd_normalize(simd_cross(targZR, targXR))
            targXR = simd_cross(targZR, targYR)

            jointAnimLocal[Self.heelR] = mixTransforms(
                jointAnimLocal[Self.heelR],
                jointGlobalAnim[Self.kneeR].inverse * simd_float4x4(
                    SIMD4<Float>(-targXR, 0), SIMD4<Float>(targYR, 0), SIMD4<Float>(targZR, 0), SIMD4<Float>(heelR, 1)),
                blend.w)

            forwardKinematics()
        }

        // Rotate toe to align with the ground plane under it.
        do {
            let bendD: Float = 0, bendU: Float = 10
            let blend = simd_clamp(ikWeight * 2.5, SIMD4<Float>(repeating: 0), SIMD4<Float>(repeating: 1))

            let toeL = column3(jointGlobalAnim[Self.toeL])
            let fwrdL4 = jointGlobalAnim[Self.toeL] * SIMD4<Float>(0, 0, 10, 1)
            let side0L4 = jointGlobalAnim[Self.toeL] * SIMD4<Float>(10, 0, 0, 1)
            let side1L4 = jointGlobalAnim[Self.toeL] * SIMD4<Float>(-10, 0, 0, 1)
            var fwrdL = SIMD3<Float>(fwrdL4.x, fwrdL4.y, fwrdL4.z) / fwrdL4.w
            var side0L = SIMD3<Float>(side0L4.x, side0L4.y, side0L4.z) / side0L4.w
            var side1L = SIMD3<Float>(side1L4.x, side1L4.y, side1L4.z) / side1L4.w

            fwrdL.y = simd_clamp(groundHeight(SIMD2(fwrdL.x, fwrdL.z)) + ikToeHeight, toeL.y - bendD, toeL.y + bendU)
            side0L.y = simd_clamp(groundHeight(SIMD2(side0L.x, side0L.z)) + ikToeHeight, toeL.y - bendD, toeL.y + bendU)
            side1L.y = simd_clamp(groundHeight(SIMD2(side0L.x, side1L.z)) + ikToeHeight, toeL.y - bendD, toeL.y + bendU)

            var sideL = simd_normalize(side0L - side1L)
            fwrdL = simd_normalize(fwrdL - toeL)
            let upwrL = simd_normalize(simd_cross(sideL, fwrdL))
            sideL = simd_cross(fwrdL, upwrL)

            jointAnimLocal[Self.toeL] = mixTransforms(
                jointAnimLocal[Self.toeL],
                jointGlobalAnim[Self.heelL].inverse * simd_float4x4(
                    SIMD4<Float>(sideL, 0), SIMD4<Float>(-upwrL, 0), SIMD4<Float>(fwrdL, 0), SIMD4<Float>(toeL, 1)),
                blend.y)

            let toeR = column3(jointGlobalAnim[Self.toeR])
            let fwrdR4 = jointGlobalAnim[Self.toeR] * SIMD4<Float>(0, 0, 10, 1)
            let side0R4 = jointGlobalAnim[Self.toeR] * SIMD4<Float>(10, 0, 0, 1)
            let side1R4 = jointGlobalAnim[Self.toeR] * SIMD4<Float>(-10, 0, 0, 1)
            var fwrdR = SIMD3<Float>(fwrdR4.x, fwrdR4.y, fwrdR4.z) / fwrdR4.w
            var side0R = SIMD3<Float>(side0R4.x, side0R4.y, side0R4.z) / side0R4.w
            var side1R = SIMD3<Float>(side1R4.x, side1R4.y, side1R4.z) / side1R4.w

            fwrdR.y = simd_clamp(groundHeight(SIMD2(fwrdR.x, fwrdR.z)) + ikToeHeight, toeR.y - bendD, toeR.y + bendU)
            side0R.y = simd_clamp(groundHeight(SIMD2(side0R.x, side0R.z)) + ikToeHeight, toeR.y - bendD, toeR.y + bendU)
            side1R.y = simd_clamp(groundHeight(SIMD2(side1R.x, side1R.z)) + ikToeHeight, toeR.y - bendD, toeR.y + bendU)

            var sideR = simd_normalize(side0R - side1R)
            fwrdR = simd_normalize(fwrdR - toeR)
            let upwrR = simd_normalize(simd_cross(sideR, fwrdR))
            sideR = simd_cross(fwrdR, upwrR)

            jointAnimLocal[Self.toeR] = mixTransforms(
                jointAnimLocal[Self.toeR],
                jointGlobalAnim[Self.heelR].inverse * simd_float4x4(
                    SIMD4<Float>(sideR, 0), SIMD4<Float>(-upwrR, 0), SIMD4<Float>(fwrdR, 0), SIMD4<Float>(toeR, 1)),
                blend.w)

            forwardKinematics()
        }

        // Update locks.
        if ikLock[0] == 0, ikWeight.y >= ikThreshold {
            ikLock[0] = 1; ikLockPosition[0] = column3(jointGlobalAnim[Self.heelL])
            ikLock[2] = 1; ikLockPosition[2] = column3(jointGlobalAnim[Self.toeL])
        }
        if ikLock[1] == 0, ikWeight.w >= ikThreshold {
            ikLock[1] = 1; ikLockPosition[1] = column3(jointGlobalAnim[Self.heelR])
            ikLock[3] = 1; ikLockPosition[3] = column3(jointGlobalAnim[Self.toeR])
        }
        if ikLock[0] > 0, ikWeight.y < ikThreshold {
            ikLock[0] = simd_clamp(ikLock[0] - ikFade, 0, 1)
            ikLock[2] = simd_clamp(ikLock[2] - ikFade, 0, 1)
        }
        if ikLock[1] > 0, ikWeight.w < ikThreshold {
            ikLock[1] = simd_clamp(ikLock[1] - ikFade, 0, 1)
            ikLock[3] = simd_clamp(ikLock[3] - ikFade, 0, 1)
        }
    }
}

private func column3(_ m: simd_float4x4) -> SIMD3<Float> {
    SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
}
