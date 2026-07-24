import simd

/// A yaw-only rotation (rotation around world Y) — every `mat3` in
/// pfnn.cpp's `Trajectory`/`Character` is built via
/// `glm::rotate(atan2f(dir.x, dir.z), (0,1,0))`, i.e. always a pure heading
/// angle, never pitch/roll. Storing the angle directly instead of a full
/// rotation matrix/quaternion keeps the port simple without losing anything
/// pfnn.cpp actually uses.
func rotateY(_ v: SIMD3<Float>, by angle: Float) -> SIMD3<Float> {
    let s = sin(angle), c = cos(angle)
    return SIMD3<Float>(v.x * c + v.z * s, v.y, -v.x * s + v.z * c)
}

func yawAngle(of direction: SIMD3<Float>) -> Float {
    atan2(direction.x, direction.z)
}

/// Blends two facing directions by slerping the shortest-path yaw between
/// them — ported from pfnn.cpp's `mix_directions` (quaternion slerp of two
/// yaw-only rotations reduces to a plain angle lerp along the shortest arc).
func mixDirections(_ x: SIMD3<Float>, _ y: SIMD3<Float>, _ a: Float) -> SIMD3<Float> {
    let ax = yawAngle(of: x), ay = yawAngle(of: y)
    var delta = ay - ax
    while delta > .pi { delta -= 2 * .pi }
    while delta < -.pi { delta += 2 * .pi }
    let angle = ax + delta * a
    return SIMD3<Float>(sin(angle), 0, cos(angle))
}

/// Ports the `Trajectory` struct and its `pre_render()`/`post_render()`
/// update math from `sreyafrancis/PFNN`'s `demo/pfnn.cpp`. The `Areas`
/// system (walls/crouch-zones/jump-pads) is dropped entirely — no
/// equivalent in this app, and out of scope for "bring the puppet in and
/// test it directly." Terrain height comes from a caller-supplied
/// `groundHeight` closure (a real LiDAR raycast in practice) instead of
/// pfnn.cpp's synthetic heightmap texture.
final class PFNNTrajectory {
    static let length = 120
    static let half = length / 2

    var positions = [SIMD3<Float>](repeating: .zero, count: PFNNTrajectory.length)
    var directions = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 1), count: PFNNTrajectory.length)
    var heights = [Float](repeating: 0, count: PFNNTrajectory.length)

    // Gait one-hot-ish weights per trajectory sample. Jog/crouch/jump are
    // always kept at 0 for this phase (no run-trigger or crouch control
    // wired up yet) but the slots stay present — the network's 342-dim
    // input vector was trained expecting exactly this many gait channels
    // in this order, even if some are always zero.
    var gaitStand = [Float](repeating: 0, count: PFNNTrajectory.length)
    var gaitWalk = [Float](repeating: 0, count: PFNNTrajectory.length)
    var gaitJog = [Float](repeating: 0, count: PFNNTrajectory.length)
    var gaitCrouch = [Float](repeating: 0, count: PFNNTrajectory.length)
    var gaitJump = [Float](repeating: 0, count: PFNNTrajectory.length)

    var targetDir = SIMD3<Float>(0, 0, 1)
    var targetVel = SIMD3<Float>.zero

    let width: Float = 0.25 // trajectory "shoulder width" sample offset, meters (pfnn.cpp: 25 in ~cm-scale units)

    func reset(at position: SIMD3<Float>, groundHeight: (SIMD2<Float>) -> Float) {
        let y = groundHeight(SIMD2(position.x, position.z))
        let root = SIMD3<Float>(position.x, y, position.z)
        for i in 0..<Self.length {
            positions[i] = root
            directions[i] = SIMD3<Float>(0, 0, 1)
            heights[i] = y
            gaitStand[i] = 0; gaitWalk[i] = 0; gaitJog[i] = 0; gaitCrouch[i] = 0; gaitJump[i] = 0
        }
        targetDir = SIMD3<Float>(0, 0, 1)
        targetVel = .zero
    }

    /// Simplified gait model for this phase: standing vs. walking, purely
    /// from target speed. (pfnn.cpp also branches on a run trigger and a
    /// crouch toggle — neither has a control wired up yet.)
    func updateGait(gaitSmooth: Float) {
        let standing = simd_length(targetVel) < 0.1
        let standAmount = standing ? 1 - simd_clamp(simd_length(targetVel) / 0.1, 0, 1) : 0
        gaitStand[Self.half] = mix(gaitStand[Self.half], standAmount, t: gaitSmooth)
        gaitWalk[Self.half] = mix(gaitWalk[Self.half], standing ? 0 : 1, t: gaitSmooth)
        gaitJog[Self.half] = mix(gaitJog[Self.half], 0, t: gaitSmooth)
        gaitCrouch[Self.half] = mix(gaitCrouch[Self.half], 0, t: gaitSmooth)
        gaitJump[Self.half] = mix(gaitJump[Self.half], 0, t: gaitSmooth)
    }

    /// Ports the "Predict Future Trajectory" section of `pre_render()`
    /// (minus wall collision, which needs the dropped `Areas` system).
    func predictFuture(strafeAmount: Float, responsive: Bool, groundHeight: (SIMD2<Float>) -> Float) {
        var blended = positions
        for i in (Self.half + 1)..<Self.length {
            let biasPos: Float = responsive ? 2.0 : mix(0.5, 1.0, t: strafeAmount)
            let biasDir: Float = responsive ? mix(5.0, 3.0, t: strafeAmount) : mix(2.0, 0.5, t: strafeAmount)

            let scalePos = 1 - powf(1 - Float(i - Self.half) / Float(Self.half), biasPos)
            let scaleDir = 1 - powf(1 - Float(i - Self.half) / Float(Self.half), biasDir)

            blended[i] = blended[i - 1] + mix(positions[i] - positions[i - 1], targetVel, t: scalePos)
            directions[i] = mixDirections(directions[i], targetDir, scaleDir)

            heights[i] = heights[Self.half]
            gaitStand[i] = gaitStand[Self.half]
            gaitWalk[i] = gaitWalk[Self.half]
            gaitJog[i] = gaitJog[Self.half]
            gaitCrouch[i] = gaitCrouch[Self.half]
            gaitJump[i] = gaitJump[Self.half]
        }
        for i in (Self.half + 1)..<Self.length { positions[i] = blended[i] }

        for i in Self.half..<Self.length {
            positions[i].y = groundHeight(SIMD2(positions[i].x, positions[i].z))
        }

        heights[Self.half] = 0
        var i = 0
        while i < Self.length {
            heights[Self.half] += positions[i].y / Float(Self.length / 10)
            i += 10
        }
    }

    /// The root position/rotation the network's input vector is expressed
    /// relative to — the trajectory sample at the CURRENT frame (index
    /// `half`), matching pfnn.cpp's `root_position`/`root_rotation`.
    var rootPosition: SIMD3<Float> {
        SIMD3<Float>(positions[Self.half].x, heights[Self.half], positions[Self.half].z)
    }
    var rootYaw: Float { yawAngle(of: directions[Self.half]) }

    var prevRootPosition: SIMD3<Float> {
        SIMD3<Float>(positions[Self.half - 1].x, heights[Self.half - 1], positions[Self.half - 1].z)
    }
    var prevRootYaw: Float { yawAngle(of: directions[Self.half - 1]) }

    /// "Update Past Trajectory" — rolls the whole window one sample
    /// forward in time.
    func rollPastForward() {
        for i in 0..<Self.half {
            positions[i] = positions[i + 1]
            directions[i] = directions[i + 1]
            heights[i] = heights[i + 1]
            gaitStand[i] = gaitStand[i + 1]
            gaitWalk[i] = gaitWalk[i + 1]
            gaitJog[i] = gaitJog[i + 1]
            gaitCrouch[i] = gaitCrouch[i + 1]
            gaitJump[i] = gaitJump[i + 1]
        }
    }

    /// "Update Current Trajectory" — advances the current-frame sample by
    /// the network's own predicted root velocity/turn (`Yp[0], Yp[1],
    /// Yp[2]`), scaled down while standing so a stationary idle doesn't
    /// creep.
    func updateCurrent(rootDelta: SIMD2<Float>, rootTurn: Float, standAmount: Float) {
        let yaw = rootYaw
        let update = rotateY(SIMD3<Float>(rootDelta.x, 0, rootDelta.y), by: yaw)
        positions[Self.half] += standAmount * update
        let angle = yaw + standAmount * (-rootTurn)
        directions[Self.half] = SIMD3<Float>(sin(angle), 0, cos(angle))
    }

    // "Update Future Trajectory" (the block that reconstructs indices
    // half+1..<length from the network's predicted Yp[8...] future
    // samples) is deliberately NOT ported here as a helper — the raw
    // index arithmetic in pfnn.cpp for that block is stride-based in a
    // way that's easy to subtly break by abstracting (verified by hand:
    // the last blend sample of the last quantity's block legitimately
    // reads one index past where the block "should" conceptually end,
    // because that's just what this specific trained network's output
    // layout is — not a bug to clean up). PFNNController ported it as a
    // direct, literal, index-for-index copy of pfnn.cpp's loop instead,
    // operating straight on `positions`/`directions` here.
}

func mix(_ a: Float, _ b: Float, t: Float) -> Float { a * (1 - t) + b * t }
func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, t: Float) -> SIMD3<Float> { a * (1 - t) + b * t }
