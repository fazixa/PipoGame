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

    /// TEMP: trajectory-drawing prototype. A drawn waypoint plus whether the
    /// surface it was tapped on is vertical (a wall) or horizontal (ground/
    /// ledge) — read from the tap's surface normal and shown as a red vs.
    /// yellow marker. Reaching a vertical point triggers climbing (straight
    /// up the wall face to whatever's next), reaching a horizontal one
    /// resumes normal walking.
    struct PathPoint {
        let position: SIMD3<Float>
        let isVertical: Bool
    }

    enum State {
        case unplaced
        case idle
        case walking(target: PathPoint)
        /// Rigged path only: finish the current step, then freeze the clip.
        case stoppingWalk
        case sitting
        /// TEMP: fall/landing prototype. Dropping past a look-ahead-detected
        /// edge; landingPosition is where it'll land (offset forward from
        /// the edge, in the direction Pipo was walking, so he doesn't drop
        /// straight down off the lip), resumeTarget is the original walk
        /// destination to continue toward once landed.
        case falling(resumeTarget: PathPoint, landingPosition: SIMD3<Float>)
        /// TEMP: hard-landing clip playing on impact, then resumes walking.
        case landing(resumeTarget: PathPoint)
        /// TEMP: climb prototype. Climbing straight up the wall face from
        /// startPosition to target's position (X/Z pinned to the wall's own
        /// point — see climb() for why). startTime is a snapshot of `time`
        /// when this climb segment began, used to compute progress.
        case climbing(target: PathPoint, startPosition: SIMD3<Float>, startTime: Float)
    }

    @Published var isPlaced = false
    @Published var isSitting = false
    /// Sit is procedural-only until a sit clip is authored for the rigged Pipo.
    @Published var supportsSit = true
    /// TEMP: trajectory-drawing prototype. While true, taps add path points
    /// (visualized with markers) instead of moving Pipo immediately.
    @Published var isDrawingPath = false

    /// Set by ContentView; used to stop recording without on-screen UI.
    var onLongPress: (() -> Void)?

    weak var arView: ARView?

    private var state: State = .unplaced
    private var anchor: AnchorEntity?
    private var pipo: Entity?

    // TEMP: trajectory-drawing prototype
    /// Points placed so far this drawing session, in tap order.
    private var drawnPoints: [PathPoint] = []
    /// Visual marker entities, one per drawn point that hasn't been reached
    /// yet — kept in the same order as drawnPoints/pathQueue so arriving at
    /// a waypoint always removes pathMarkers.first.
    private var pathMarkers: [AnchorEntity] = []
    /// Remaining waypoints once a drawn path is committed and being walked;
    /// the current walk target (state's .walking payload) is NOT in this
    /// queue — only the ones still to come after it.
    private var pathQueue: [PathPoint] = []

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

    // TEMP: climb prototype. Reaching a red (vertical) path point triggers
    // climbing to whatever's next in the queue, instead of the normal
    // grounded walk.
    private var climbClip: AnimationResource?
    private var climbPlayback: AnimationPlaybackController?
    /// Keeps climbClip's source entity retained — see landEntity for why.
    private var climbEntity: Entity?

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
    // TEMP: climb prototype — fallback lerp speed if no climb clip loaded.
    private var climbSpeed: Float { walkSpeed * 0.5 }

    // MARK: - Input

    func handleTap(on result: ARRaycastResult) {
        if isDrawingPath {
            addPathPoint(at: result.worldTransform)
            return
        }
        switch state {
        case .unplaced:
            place(at: result.worldTransform)
        case .sitting:
            break // hint tells the user to stand first
        case .falling, .landing, .climbing:
            break // TEMP: fall/landing/climb prototype — ignore taps mid-sequence
        case .idle, .walking, .stoppingWalk:
            let t = result.worldTransform.columns.3
            state = .walking(target: PathPoint(position: [t.x, t.y, t.z], isVertical: false))
            startWalkClipIfNeeded()
        }
    }

    // MARK: - TEMP: trajectory-drawing prototype

    /// Enters drawing mode (only from .idle, with Pipo already placed) or,
    /// if already drawing, commits the drawn points as a multi-waypoint
    /// walk and starts moving through them in order.
    func toggleDrawPath() {
        guard isPlaced else { return }
        if isDrawingPath {
            isDrawingPath = false
            guard !drawnPoints.isEmpty else { return }
            pathQueue = drawnPoints
            drawnPoints = []
            let first = pathQueue.removeFirst()
            state = .walking(target: first)
            startWalkClipIfNeeded()
        } else {
            guard case .idle = state else { return }
            clearPathMarkers()
            drawnPoints = []
            pathQueue = []
            isDrawingPath = true
        }
    }

    private func addPathPoint(at transform: simd_float4x4) {
        guard let arView else { return }
        let t = transform.columns.3

        // Surface orientation read from the raycast result's own up-axis
        // (column 1) — a tap that landed on a wall reads as vertical (red),
        // ground/ledge taps as horizontal (yellow).
        let upY = transform.columns.1.y
        let isVertical = abs(upY) < 0.5
        drawnPoints.append(PathPoint(position: [t.x, t.y, t.z], isVertical: isVertical))

        let marker = ModelEntity(mesh: .generateSphere(radius: 0.004 * scaleFactor),
                                 materials: [UnlitMaterial(color: isVertical ? .systemRed : .systemYellow)])
        let markerAnchor = AnchorEntity(world: transform)
        markerAnchor.addChild(marker)
        arView.scene.addAnchor(markerAnchor)
        pathMarkers.append(markerAnchor)
    }

    private func clearPathMarkers() {
        for marker in pathMarkers {
            marker.removeFromParent()
        }
        pathMarkers.removeAll()
    }

    func toggleSit() {
        guard supportsSit, !isDrawingPath else { return }
        switch state {
        case .sitting:
            state = .idle
            isSitting = false
        case .idle, .walking, .stoppingWalk:
            state = .sitting
            isSitting = true
        case .unplaced, .falling, .landing, .climbing:
            break // TEMP: can't sit mid-air, mid-landing, or mid-climb
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
        climbClip = nil
        climbPlayback = nil
        climbEntity = nil
        clearPathMarkers()
        drawnPoints = []
        pathQueue = []
        isDrawingPath = false
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
            climbClip = loaded.climbClip
            climbEntity = loaded.climbEntity
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

        case .falling(let resumeTarget, let landingPosition):
            fall(pipo: pipo, resumeTarget: resumeTarget, landingPosition: landingPosition, dt: dt)

        case .landing(let resumeTarget):
            updateLanding(pipo: pipo, resumeTarget: resumeTarget)

        case .climbing(let target, let startPosition, let startTime):
            climb(pipo: pipo, target: target, startPosition: startPosition, startTime: startTime)
        }
    }

    private func walk(pipo: Entity, toward target: PathPoint, dt: Float) {
        var position = pipo.position(relativeTo: nil)
        var heading = target.position - position
        heading.y = 0
        let distance = simd_length(heading)

        // Arrival is detected once the normal walked step below has closed
        // the gap to essentially nothing — it clamps to never overshoot, so
        // Pipo naturally walks all the way to the point's exact center
        // instead of stopping early or being snapped the rest of the way.
        if distance < 0.0005 {
            // TEMP: trajectory-drawing prototype — remove the marker for the
            // waypoint just reached, then continue toward the next queued
            // point if this was part of a multi-point drawn path.
            if !pathMarkers.isEmpty {
                pathMarkers.removeFirst().removeFromParent()
            }
            // TEMP: climb prototype — a red (vertical) point marks the base
            // of a wall. Climb straight up it toward whatever's next in the
            // queue instead of continuing to walk.
            if target.isVertical, !pathQueue.isEmpty {
                let next = pathQueue.removeFirst()
                // Deliberately NOT stopping walkPlayback here — leaving it
                // playing lets startClimbClipIfNeeded's playAnimation crossfade
                // directly from the walk pose into the climb pose. Stopping it
                // first was cutting the walk clip before the blend had anything
                // to blend from, so Pipo snapped to a static bind pose for a
                // frame before the climb clip faded in.
                state = .climbing(target: next, startPosition: position, startTime: time)
                return
            }
            if !pathQueue.isEmpty {
                let next = pathQueue.removeFirst()
                state = .walking(target: next)
                return
            }
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
                    // Land a bit forward of the edge, in the direction Pipo
                    // was walking, rather than dropping straight down from
                    // where the edge was detected — half a character-height
                    // of forward carry reads as falling with momentum
                    // instead of stepping straight off a cliff.
                    let forwardCarry = direction * (characterHeight * 0.5)
                    let landingPosition = SIMD3<Float>(position.x + forwardCarry.x,
                                                        aheadGroundY,
                                                        position.z + forwardCarry.z)
                    state = .falling(resumeTarget: target, landingPosition: landingPosition)
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

    private func fall(pipo: Entity, resumeTarget: PathPoint, landingPosition: SIMD3<Float>, dt: Float) {
        fallVelocity += gravity * dt
        var position = pipo.position(relativeTo: nil)
        position.y -= fallVelocity * dt
        // Drift horizontally toward the forward-offset landing spot while
        // falling, carrying the walking momentum through the drop instead
        // of dropping straight down from the edge.
        position.x = damp(position.x, landingPosition.x, rate: 6, dt: dt)
        position.z = damp(position.z, landingPosition.z, rate: 6, dt: dt)
        if position.y <= landingPosition.y {
            position.y = landingPosition.y
            position.x = landingPosition.x
            position.z = landingPosition.z
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

    private func updateLanding(pipo: Entity, resumeTarget: PathPoint) {
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

    // MARK: - Climb prototype

    /// Climbs straight up the wall face: horizontal position stays pinned
    /// to the wall's own point (`target`, where the red marker was placed)
    /// while only height changes over time — mirrors how walking only ever
    /// moves X/Z and lets Y follow the ground, so this doesn't cut a
    /// diagonal path through the air between wherever the walk stopped and
    /// the target.
    private func climb(pipo: Entity, target: PathPoint, startPosition: SIMD3<Float>, startTime: Float) {
        let elapsed = time - startTime
        // The climb clip is a self-contained reach cycle that returns
        // exactly to its start pose (verified against the exported hip
        // bone: first and last frame match) — it's meant to be looped
        // continuously while code drives the actual climb rate, the same
        // way the walk clip loops while walkSpeed drives ground movement.
        // Locking total movement duration to one non-repeating playthrough
        // of the clip made taller climbs cover way more distance than a
        // single reach should, which read as much too fast.
        let verticalDistance = abs(target.position.y - startPosition.y)
        let duration = max(verticalDistance / climbSpeed, 0.01)
        let t = min(elapsed / duration, 1.0)
        let newY = simd_mix(startPosition.y, target.position.y, t)
        let newPosition = SIMD3<Float>(target.position.x, newY, target.position.z)
        pipo.setPosition(newPosition, relativeTo: nil)

        startClimbClipIfNeeded()

        guard t >= 1.0 else { return }

        if !pathMarkers.isEmpty {
            pathMarkers.removeFirst().removeFromParent()
        }

        if target.isVertical {
            // Still on the wall — keep climbing straight to whatever's next.
            if !pathQueue.isEmpty {
                let next = pathQueue.removeFirst()
                state = .climbing(target: next, startPosition: newPosition, startTime: time)
                return
            }
            climbPlayback?.stop()
            climbPlayback = nil
            state = usesClips ? .stoppingWalk : .idle
            return
        }

        // Reached the top (a horizontal point) — resume walking toward
        // whatever's next, or stop here if this was the last point.
        climbPlayback?.stop()
        climbPlayback = nil
        if !pathQueue.isEmpty {
            let next = pathQueue.removeFirst()
            state = .walking(target: next)
            startWalkClipIfNeeded()
            return
        }
        state = usesClips ? .stoppingWalk : .idle
    }

    private func startClimbClipIfNeeded() {
        guard let clip = climbClip, let owner = animationOwner else { return }
        if let playback = climbPlayback, playback.isValid {
            playback.speed = 1
            playback.resume()
        } else {
            climbPlayback = owner.playAnimation(clip.repeat(),
                                                transitionDuration: 0.25,
                                                startsPaused: false)
        }
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
