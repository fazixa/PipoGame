import RealityKit
import ARKit
import Combine
import simd

/// Owns Pipo's state machine and drives him every frame.
///
/// Two animation paths:
/// - Rigged Pipo (Pipo.usdz): the baked walk clip plays while code drives
///   locomotion; on arrival the clip freezes at the nearest foot-contact
///   frame (no idle clip authored yet).
/// - Primitive fallback (PipoBuilder): fully procedural idle/walk/sit,
///   used only if the USDZ fails to load.
final class PipoController: ObservableObject {

    enum State {
        case unplaced
        case idle
        case walking(target: SIMD3<Float>)
        /// Rigged path only: finish the current step, then freeze the clip.
        case stoppingWalk
        case sitting
        /// TEMP: fall/landing prototype. Dropping past a look-ahead-detected
        /// edge; groundY is where it'll land, resumeTarget is the original
        /// walk destination to continue toward once landed.
        case falling(resumeTarget: SIMD3<Float>, groundY: Float)
        /// TEMP: hard-landing clip playing on impact, then resumes walking.
        case landing(resumeTarget: SIMD3<Float>)
    }

    @Published var isPlaced = false
    @Published var isSitting = false
    /// Sit is procedural-only until a sit clip is authored for the rigged Pipo.
    @Published var supportsSit = true

    /// Set by ContentView; used to stop recording without on-screen UI.
    var onLongPress: (() -> Void)?

    weak var arView: ARView?

    private var state: State = .unplaced
    private var anchor: AnchorEntity?
    private var pipo: Entity?

    // Rigged-clip playback
    private var animationOwner: Entity?
    private var walkClip: AnimationResource?
    private var walkClipDuration: TimeInterval = 1
    private var walkPlayback: AnimationPlaybackController?
    private var usesClips: Bool { walkClip != nil }

    // TEMP: fall/landing prototype
    private var landClip: AnimationResource?
    private var landClipDuration: TimeInterval = 1
    private var landPlayback: AnimationPlaybackController?
    /// Keeps landClip's source entity retained — see LoadedPipo.landEntity.
    private var landEntity: Entity?
    private var fallVelocity: Float = 0
    private let gravity: Float = 9.8                          // m/s^2, real-world scale
    // TEMP: fixed real-world distance, deliberately NOT scaled by
    // scaleFactor. This is "how far ahead to peek for an edge," which
    // should track how far a step actually covers, not the character's
    // rendered size — scaling it by pinch level meant that at the ~9x
    // scaleFactor implied by testing at 20cm+ tall, it was looking ~45cm
    // ahead and detecting the table's edge while still comfortably mid-table.
    private let lookAheadDistance: Float = 0.03                 // m, fixed
    /// Trigger a fall when the look-ahead ground drop exceeds this many
    /// character-heights, so small steps/bumps don't trigger it. This DOES
    /// scale correctly with size, via characterHeight below.
    private let fallEdgeHeightMultiplier: Float = 1.0
    private var characterHeight: Float {
        guard let pipo else { return 0.02 }
        return pipo.visualBounds(relativeTo: nil).extents.y
    }
    // Minor safety net against a single noisy LiDAR-mesh raycast (not the
    // main fix — the look-ahead distance above was) — require the drop to
    // read consistently for a few frames before committing.
    private var edgeConfirmFrames = 0
    private let edgeConfirmThreshold = 6  // ~0.1s at 60fps

    private var time: Float = 0
    private var walkPhase: Float = 0
    private var baseScale: Float = 1

    // Ground speed matches the walk cycle's stride at natural size; both
    // scale with pinch so feet don't slide when Pipo is giant or tiny.
    private var scaleFactor: Float {
        guard let pipo, baseScale > 0 else { return 1 }
        return pipo.scale.x / baseScale
    }
    // TEMP: replaces the Pipo-tuned 0.16 for WalkTest.usdz (Mixamo,
    // exported "in place" with no baked root motion). The clip's own
    // stance-phase kinematics (both feet's ground-contact windows, cross-
    // checked two ways) imply ~1.15 m/s at the model's NATURAL/raw size —
    // but PipoAsset.load() renders WalkTest at 0.015x scale, and walkSpeed
    // is applied at that already-scaled size (scaleFactor normalizes to 1.0
    // at placement scale, not true 1:1), so the constant needs that same
    // 0.015 factor folded in: 1.15 * 0.015 ≈ 0.0173.
    // Revert to 0.16 when swapping back to Pipo.
    private var walkSpeed: Float { (usesClips ? 0.0173 : 0.22) * scaleFactor }  // m/s
    private let stepRate: Float = 11                          // procedural gait, rad/s
    private var arrivalDistance: Float { 0.03 * scaleFactor }

    // MARK: - Input

    func handleTap(on result: ARRaycastResult) {
        switch state {
        case .unplaced:
            place(at: result.worldTransform)
        case .sitting:
            break // hint tells the user to stand first
        case .falling, .landing:
            break // TEMP: fall/landing prototype — ignore taps mid-sequence
        case .idle, .walking, .stoppingWalk:
            let t = result.worldTransform.columns.3
            state = .walking(target: [t.x, t.y, t.z])
            startWalkClipIfNeeded()
        }
    }

    func toggleSit() {
        guard supportsSit else { return }
        switch state {
        case .sitting:
            state = .idle
            isSitting = false
        case .idle, .walking, .stoppingWalk:
            state = .sitting
            isSitting = true
        case .unplaced, .falling, .landing:
            break // TEMP: can't sit mid-air or mid-landing
        }
    }

    /// Pinch-to-scale, clamped to 0.2x–10x of Pipo's natural size.
    func pinch(by factor: Float) {
        guard let pipo else { return }
        let scaled = min(max(pipo.scale.x * factor, baseScale * 0.2), baseScale * 10)
        pipo.scale = SIMD3<Float>(repeating: scaled)
    }

    func reset() {
        anchor?.removeFromParent()
        anchor = nil
        pipo = nil
        animationOwner = nil
        walkClip = nil
        walkPlayback = nil
        landClip = nil
        landPlayback = nil
        landEntity = nil
        fallVelocity = 0
        edgeConfirmFrames = 0
        state = .unplaced
        isPlaced = false
        isSitting = false
        supportsSit = true
    }

    private func place(at transform: simd_float4x4) {
        guard let arView else { return }

        let pipo: Entity
        if let loaded = PipoAsset.load() {
            pipo = loaded.root
            animationOwner = loaded.animationOwner
            walkClip = loaded.walkClip
            walkClipDuration = loaded.walkClip.definition.duration
            landClip = loaded.landClip
            landClipDuration = loaded.landClip?.definition.duration ?? 1
            landEntity = loaded.landEntity
            supportsSit = false
        } else {
            pipo = PipoBuilder.build()
            supportsSit = true
        }

        let anchor = AnchorEntity(world: transform)

        // Face the camera when appearing.
        let position = SIMD3<Float>(transform.columns.3.x,
                                    transform.columns.3.y,
                                    transform.columns.3.z)
        let toCamera = arView.cameraTransform.translation - position
        pipo.orientation = simd_quatf(angle: atan2(toCamera.x, toCamera.z), axis: [0, 1, 0])

        let finalScale = pipo.scale
        baseScale = finalScale.x
        pipo.scale = finalScale * 0.01
        anchor.addChild(pipo)
        arView.scene.addAnchor(anchor)

        var appeared = pipo.transform
        appeared.scale = finalScale
        pipo.move(to: appeared, relativeTo: anchor, duration: 0.35, timingFunction: .easeOut)

        self.anchor = anchor
        self.pipo = pipo
        state = .idle
        isPlaced = true
    }

    // MARK: - Per-frame update

    func update(deltaTime dt: Float) {
        guard let pipo else { return }
        time += dt

        switch state {
        case .unplaced:
            break

        case .idle:
            if !usesClips { animateIdle(pipo: pipo, dt: dt) }

        case .walking(let target):
            walk(pipo: pipo, toward: target, dt: dt)

        case .stoppingWalk:
            freezeWalkClipAtContactFrame()

        case .sitting:
            if !usesClips { animateSit(pipo: pipo, dt: dt) }

        case .falling(let resumeTarget, let groundY):
            fall(pipo: pipo, resumeTarget: resumeTarget, groundY: groundY, dt: dt)

        case .landing(let resumeTarget):
            updateLanding(pipo: pipo, resumeTarget: resumeTarget)
        }
    }

    private func walk(pipo: Entity, toward target: SIMD3<Float>, dt: Float) {
        var position = pipo.position(relativeTo: nil)
        var heading = target - position
        heading.y = 0
        let distance = simd_length(heading)

        if distance < arrivalDistance {
            state = usesClips ? .stoppingWalk : .idle
            return
        }

        let direction = heading / distance

        // TEMP: fall/landing prototype. Look ahead of the current position
        // rather than reacting after already stepping into open air — if
        // the ground there drops more than ~1 character-height, fall.
        // lookAheadDistance is a fixed distance (not scaled by pinch size);
        // the confirm-frames counter is a minor safety net against a single
        // noisy LiDAR-mesh raycast.
        if usesClips, landClip != nil {
            let lookAheadPoint = position + direction * lookAheadDistance
            if let aheadGroundY = groundHeight(at: lookAheadPoint),
               position.y - aheadGroundY > characterHeight * fallEdgeHeightMultiplier {
                edgeConfirmFrames += 1
                if edgeConfirmFrames >= edgeConfirmThreshold {
                    edgeConfirmFrames = 0
                    // Fully stop (not pause) the walk clip — a paused-but-
                    // present controller can still compete with the land
                    // clip on the same entity, since RealityKit layers
                    // animations by default rather than replacing them.
                    walkPlayback?.stop()
                    walkPlayback = nil
                    startLandClip()
                    state = .falling(resumeTarget: target, groundY: aheadGroundY)
                    return
                }
            } else {
                edgeConfirmFrames = 0
            }
        }

        let step = min(walkSpeed * dt, distance)
        position += direction * step

        // Re-ground on the scene mesh so slopes and small bumps are followed,
        // but ignore hits far below (walking near a shelf edge shouldn't snap
        // Pipo to the floor).
        if let groundY = groundHeight(at: position), abs(groundY - position.y) < 0.15 * scaleFactor {
            position.y = damp(position.y, groundY, rate: 12, dt: dt)
        }
        pipo.setPosition(position, relativeTo: nil)

        let desired = simd_quatf(angle: atan2(direction.x, direction.z), axis: [0, 1, 0])
        let current = pipo.orientation(relativeTo: nil)
        pipo.setOrientation(simd_slerp(current, desired, smoothing(rate: 10, dt: dt)), relativeTo: nil)

        if usesClips {
            startWalkClipIfNeeded()
        } else {
            animateGait(pipo: pipo, dt: dt)
        }
    }

    private func groundHeight(at position: SIMD3<Float>) -> Float? {
        guard let arView else { return nil }

        // Prefer raycasting against the actual reconstructed LiDAR mesh
        // (real triangle geometry) over ARKit's estimated-plane model, which
        // extrapolates/smooths past real edges like a table boundary and
        // was causing Pipo to "float" past them.
        let from = position + SIMD3<Float>(0, 0.25, 0)
        let to = position - SIMD3<Float>(0, 2.0, 0)
        if let hit = arView.scene.raycast(from: from, to: to, query: .nearest,
                                          mask: .sceneUnderstanding).first {
            return hit.position.y
        }

        // Fallback: estimated plane, for areas the mesh hasn't scanned yet.
        let query = ARRaycastQuery(origin: position + [0, 0.25, 0],
                                   direction: [0, -1, 0],
                                   allowing: .estimatedPlane,
                                   alignment: .any)
        return arView.session.raycast(query).first?.worldTransform.columns.3.y
    }

    // MARK: - Fall/landing prototype

    private func fall(pipo: Entity, resumeTarget: SIMD3<Float>, groundY: Float, dt: Float) {
        fallVelocity += gravity * dt
        var position = pipo.position(relativeTo: nil)
        position.y -= fallVelocity * dt
        if position.y <= groundY {
            position.y = groundY
            pipo.setPosition(position, relativeTo: nil)
            fallVelocity = 0
            // Resume the SAME controller created back in walk() rather than
            // tearing it down and starting a new one — recreating the
            // controller mid-sequence (stop + fresh playAnimation) was the
            // likely source of the landing clip playing "randomly": whether
            // the new controller was fully hooked up before the very next
            // frame's isValid/time check was timing-dependent.
            landPlayback?.speed = 1
            landPlayback?.resume()
            print("DEBUG landing resume: playback=\(landPlayback != nil) valid=\(landPlayback?.isValid ?? false) time=\(landPlayback?.time ?? -1)")
            state = .landing(resumeTarget: resumeTarget)
            return
        }
        pipo.setPosition(position, relativeTo: nil)
    }

    private func updateLanding(pipo: Entity, resumeTarget: SIMD3<Float>) {
        guard let playback = landPlayback, playback.isValid,
              playback.time < landClipDuration - 0.05 else {
            print("DEBUG updateLanding: bailing to walk, playback=\(landPlayback != nil) valid=\(landPlayback?.isValid ?? false) time=\(landPlayback?.time ?? -1)")
            landPlayback?.stop()
            landPlayback = nil
            state = .walking(target: resumeTarget)
            startWalkClipIfNeeded()
            return
        }
    }

    /// Creates the land-clip controller paused at frame 0 (the braced-in-air
    /// pose), held as a static visual during the fall itself. fall() resumes
    /// this SAME controller on impact rather than recreating it.
    private func startLandClip() {
        guard let clip = landClip, let owner = animationOwner else {
            print("DEBUG startLandClip: missing clip or owner, clip=\(landClip != nil) owner=\(animationOwner != nil)")
            return
        }
        landPlayback?.stop()
        landPlayback = owner.playAnimation(clip, transitionDuration: 0, startsPaused: true)
        print("DEBUG startLandClip: created paused playback, valid=\(landPlayback?.isValid ?? false)")
    }

    // MARK: - Rigged clip playback

    private func startWalkClipIfNeeded() {
        guard let clip = walkClip, let owner = animationOwner else { return }
        if let playback = walkPlayback, playback.isValid {
            playback.speed = 1
            playback.resume()  // TEMP: un-pauses if this resumed from a fall
        } else {
            walkPlayback = owner.playAnimation(clip.repeat(),
                                               transitionDuration: 0.25,
                                               startsPaused: false)
        }
    }

    /// Without an idle clip, stopping mid-stride looks broken. Let the walk
    /// cycle run until it passes a foot-contact pose (start, middle or end of
    /// the cycle), then freeze there.
    private func freezeWalkClipAtContactFrame() {
        guard let playback = walkPlayback, playback.isValid else {
            state = .idle
            return
        }
        let t = playback.time.truncatingRemainder(dividingBy: walkClipDuration)
        let half = walkClipDuration / 2
        let distanceToContact = min(abs(t), abs(t - half), abs(t - walkClipDuration))
        if distanceToContact < 0.06 {
            playback.speed = 0
            state = .idle
        }
    }

    // MARK: - Procedural poses (primitive fallback)

    private func animateIdle(pipo: Entity, dt: Float) {
        guard let body = pipo.findEntity(named: "bodyPivot"),
              let head = pipo.findEntity(named: "head"),
              let footL = pipo.findEntity(named: "footL"),
              let footR = pipo.findEntity(named: "footR") else { return }

        let bob = PipoBuilder.bodyRestY + sin(time * 2.2) * 0.0035
        body.position.y = damp(body.position.y, bob, rate: 8, dt: dt)
        body.orientation = simd_slerp(body.orientation, simd_quatf(angle: 0, axis: [1, 0, 0]),
                                      smoothing(rate: 6, dt: dt))

        head.orientation = simd_quatf(angle: sin(time * 0.6) * 0.12, axis: [0, 1, 0])

        for foot in [footL, footR] {
            foot.position.y = damp(foot.position.y, PipoBuilder.footRestY, rate: 10, dt: dt)
            foot.position.z = damp(foot.position.z, 0, rate: 10, dt: dt)
        }
    }

    private func animateGait(pipo: Entity, dt: Float) {
        guard let body = pipo.findEntity(named: "bodyPivot"),
              let footL = pipo.findEntity(named: "footL"),
              let footR = pipo.findEntity(named: "footR") else { return }

        walkPhase += stepRate * dt

        footL.position.z = sin(walkPhase) * 0.018
        footL.position.y = PipoBuilder.footRestY + max(0, sin(walkPhase)) * 0.01
        footR.position.z = sin(walkPhase + .pi) * 0.018
        footR.position.y = PipoBuilder.footRestY + max(0, sin(walkPhase + .pi)) * 0.01

        body.position.y = PipoBuilder.bodyRestY + abs(sin(walkPhase)) * 0.004
        body.orientation = simd_quatf(angle: sin(walkPhase) * 0.05, axis: [0, 0, 1])
    }

    private func animateSit(pipo: Entity, dt: Float) {
        guard let body = pipo.findEntity(named: "bodyPivot"),
              let head = pipo.findEntity(named: "head"),
              let footL = pipo.findEntity(named: "footL"),
              let footR = pipo.findEntity(named: "footR") else { return }

        body.position.y = damp(body.position.y, PipoBuilder.bodySitY, rate: 8, dt: dt)
        body.orientation = simd_slerp(body.orientation,
                                      simd_quatf(angle: -0.18, axis: [1, 0, 0]),
                                      smoothing(rate: 8, dt: dt))
        head.orientation = simd_slerp(head.orientation,
                                      simd_quatf(angle: 0.1, axis: [1, 0, 0]),
                                      smoothing(rate: 8, dt: dt))

        for foot in [footL, footR] {
            foot.position.z = damp(foot.position.z, 0.028, rate: 8, dt: dt)
            foot.position.y = damp(foot.position.y, PipoBuilder.footRestY * 0.8, rate: 8, dt: dt)
        }
    }

    // MARK: - Helpers

    private func smoothing(rate: Float, dt: Float) -> Float {
        1 - exp(-rate * dt)
    }

    private func damp(_ current: Float, _ target: Float, rate: Float, dt: Float) -> Float {
        current + (target - current) * smoothing(rate: rate, dt: dt)
    }
}
