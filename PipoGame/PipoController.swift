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
        /// Hand mode: Pipo rides the tracked palm.
        case handFollowing
    }

    @Published var isPlaced = false
    @Published var isSitting = false
    /// Sit is procedural-only until a sit clip is authored for the rigged Pipo.
    @Published var supportsSit = true
    @Published var isToon = false
    /// Toon mode needs the clip-driven Pipo (the outline shell syncs clips).
    @Published var supportsToon = false
    @Published var isHandMode = false
    /// True while hand mode is on but no palm is currently detected.
    @Published var searchingForHand = false

    weak var handTracker: HandTracker?

    /// Set by ContentView; used to stop recording without on-screen UI.
    var onLongPress: (() -> Void)?

    weak var arView: ARView?

    private var state: State = .unplaced
    private var anchor: AnchorEntity?
    private var pipo: Entity?

    // Rigged-clip playback. The current model ships only a looping sit clip;
    // walking is a glide until a walk cycle is authored (usesClips false).
    private var isRigged = false
    private var animationOwner: Entity?
    private var walkClip: AnimationResource?
    private var walkClipDuration: TimeInterval = 1
    private var walkPlayback: AnimationPlaybackController?
    private var sitClip: AnimationResource?
    private var sitPlayback: AnimationPlaybackController?
    private var usesClips: Bool { walkClip != nil }

    // Toon look
    private let toonStyle = ToonStyle()
    private var outlineAnimOwner: Entity?
    private var outlineClip: AnimationResource?
    private var outlinePlayback: AnimationPlaybackController?

    private var time: Float = 0
    private var walkPhase: Float = 0
    private var baseScale: Float = 1

    // Ground speed matches the walk cycle's stride at natural size; both
    // scale with pinch so feet don't slide when Pipo is giant or tiny.
    private var scaleFactor: Float {
        guard let pipo, baseScale > 0 else { return 1 }
        return pipo.scale.x / baseScale
    }
    /// Playback rate of the walk clip; ground speed scales with it so the
    /// feet stay planted. Tune this to change how fast Pipo walks.
    private let walkPace: Float = 1.5
    private var walkSpeed: Float { (usesClips ? 0.16 : 0.22) * scaleFactor * walkPace }  // m/s
    private let stepRate: Float = 11                          // procedural gait, rad/s
    private var arrivalDistance: Float { 0.03 * scaleFactor }

    // MARK: - Input

    func handleTap(on result: ARRaycastResult) {
        switch state {
        case .unplaced:
            place(at: result.worldTransform)
        case .sitting, .handFollowing:
            break // sitting: stand first; hand mode: taps don't steer
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
            if isRigged {
                sitPlayback?.stop(blendOutDuration: 0.3)
                sitPlayback = nil
                outlinePlayback?.stop(blendOutDuration: 0.3)
                outlinePlayback = nil
            }
            state = .idle
            isSitting = false
        case .idle, .walking, .stoppingWalk:
            if isRigged {
                guard let clip = sitClip, let owner = animationOwner else { return }
                walkPlayback?.stop(blendOutDuration: 0.2)
                walkPlayback = nil
                sitPlayback = owner.playAnimation(clip.repeat(),
                                                  transitionDuration: 0.3,
                                                  startsPaused: false)
                // Outline shell loops the same clip in lockstep
                if let oOwner = outlineAnimOwner, let oClip = outlineClip {
                    outlinePlayback?.stop(blendOutDuration: 0.2)
                    let playback = oOwner.playAnimation(oClip.repeat(),
                                                        transitionDuration: 0.3,
                                                        startsPaused: false)
                    playback.time = sitPlayback?.time ?? 0
                    outlinePlayback = playback
                }
            }
            state = .sitting
            isSitting = true
        case .unplaced, .handFollowing:
            break
        }
    }

    // MARK: - Hand mode

    func toggleHandMode() {
        isHandMode ? exitHandMode() : enterHandMode()
    }

    private func enterHandMode() {
        guard let arView else { return }
        if pipo == nil {
            // Entering hand mode before placement: spawn Pipo hidden; he
            // appears on the palm at first detection.
            place(at: matrix_identity_float4x4)
            pipo?.isEnabled = false
        }
        walkPlayback?.speed = 0
        outlinePlayback?.speed = 0
        handTracker?.reset()
        handTracker?.enabled = true
        state = .handFollowing
        isHandMode = true
        searchingForHand = true
        _ = arView // silence unused warning paths
    }

    private func exitHandMode() {
        handTracker?.enabled = false
        isHandMode = false
        searchingForHand = false
        guard let pipo else {
            state = .unplaced
            return
        }
        pipo.isEnabled = true
        // Settle onto whatever surface is below (tabletop, floor)
        var position = pipo.position(relativeTo: nil)
        if let groundY = groundHeight(at: position), position.y - groundY < 2.5 {
            position.y = groundY
            var transform = Transform(matrix: pipo.transformMatrix(relativeTo: nil))
            transform.translation = position
            pipo.move(to: transform, relativeTo: nil, duration: 0.3, timingFunction: .easeIn)
        }
        state = .idle
    }

    func toggleToon() {
        guard let pipo, supportsToon else { return }
        if isToon {
            toonStyle.remove()
            outlineAnimOwner = nil
            outlineClip = nil
            outlinePlayback = nil
            isToon = false
            return
        }
        guard let clone = toonStyle.apply(to: pipo) else { return }
        outlineAnimOwner = firstEntityWithAnimations(in: clone)
        outlineClip = outlineAnimOwner?.availableAnimations.first
        // Match whatever the body is doing right now (sitting loop, walking,
        // or a frozen pose)
        if let bodyPlayback = sitPlayback ?? walkPlayback, bodyPlayback.isValid,
           let owner = outlineAnimOwner, let clip = outlineClip {
            let playback = owner.playAnimation(clip.repeat(),
                                               transitionDuration: 0,
                                               startsPaused: false)
            playback.time = bodyPlayback.time
            playback.speed = bodyPlayback.speed
            outlinePlayback = playback
        }
        isToon = true
    }

    private func firstEntityWithAnimations(in entity: Entity) -> Entity? {
        if !entity.availableAnimations.isEmpty { return entity }
        for child in entity.children {
            if let found = firstEntityWithAnimations(in: child) { return found }
        }
        return nil
    }

    /// Pinch-to-scale, clamped to 0.2x–10x of Pipo's natural size.
    func pinch(by factor: Float) {
        guard let pipo else { return }
        let scaled = min(max(pipo.scale.x * factor, baseScale * 0.2), baseScale * 10)
        pipo.scale = SIMD3<Float>(repeating: scaled)
    }

    func reset() {
        handTracker?.enabled = false
        isHandMode = false
        searchingForHand = false
        toonStyle.remove()
        outlineAnimOwner = nil
        outlineClip = nil
        outlinePlayback = nil
        isToon = false
        supportsToon = false
        anchor?.removeFromParent()
        anchor = nil
        pipo = nil
        isRigged = false
        animationOwner = nil
        walkClip = nil
        walkPlayback = nil
        sitClip = nil
        sitPlayback = nil
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
            isRigged = true
            animationOwner = loaded.animationOwner
            walkClip = loaded.walkClip
            walkClipDuration = loaded.walkClip?.definition.duration ?? 1
            sitClip = loaded.sitClip
            supportsSit = loaded.sitClip != nil
            supportsToon = true
        } else {
            pipo = PipoBuilder.build()
            isRigged = false
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

        if isToon, let arView {
            toonStyle.updateThickness(cameraWorldPosition: arView.cameraTransform.translation,
                                      worldScale: pipo.scale.x)
        }

        switch state {
        case .unplaced:
            break

        case .idle:
            if !isRigged { animateIdle(pipo: pipo, dt: dt) }

        case .walking(let target):
            walk(pipo: pipo, toward: target, dt: dt)

        case .stoppingWalk:
            freezeWalkClipAtContactFrame()

        case .sitting:
            if !isRigged { animateSit(pipo: pipo, dt: dt) }

        case .handFollowing:
            followHand(pipo: pipo, dt: dt)
        }
    }

    private func followHand(pipo: Entity, dt: Float) {
        guard let tracker = handTracker, tracker.isTracking,
              let palm = tracker.palmWorldPosition else {
            searchingForHand = true
            return
        }
        if searchingForHand { searchingForHand = false }
        if !pipo.isEnabled {
            // First detection after spawning hidden: appear on the palm
            pipo.setPosition(palm, relativeTo: nil)
            pipo.isEnabled = true
        }

        let current = pipo.position(relativeTo: nil)
        // Lift so his feet clear the palm rather than clipping into it —
        // ARKit's hand depth estimate is noisy enough at this close range
        // that a small offset isn't reliable margin against person-
        // occlusion misreading his feet as behind the hand. Scales with
        // pinch so a giant/tiny Pipo still clears proportionally.
        let target = palm + SIMD3<Float>(0, 0.02 * scaleFactor, 0)
        pipo.setPosition(current + (target - current) * smoothing(rate: 16, dt: dt),
                         relativeTo: nil)

        // Turn with the hand: Pipo faces the wrist (back toward the holder).
        if let direction = tracker.palmDirection {
            let desired = simd_quatf(angle: atan2(-direction.x, -direction.z), axis: [0, 1, 0])
            pipo.setOrientation(simd_slerp(pipo.orientation(relativeTo: nil), desired,
                                           smoothing(rate: 10, dt: dt)),
                                relativeTo: nil)
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
        } else if !isRigged {
            animateGait(pipo: pipo, dt: dt)
        }
        // Rigged model without a walk clip: he glides in rest pose for now.
    }

    private func groundHeight(at position: SIMD3<Float>) -> Float? {
        guard let arView else { return nil }
        let query = ARRaycastQuery(origin: position + [0, 0.25, 0],
                                   direction: [0, -1, 0],
                                   allowing: .estimatedPlane,
                                   alignment: .any)
        return arView.session.raycast(query).first?.worldTransform.columns.3.y
    }

    // MARK: - Rigged clip playback

    private func startWalkClipIfNeeded() {
        guard let clip = walkClip, let owner = animationOwner else { return }
        if let playback = walkPlayback, playback.isValid {
            playback.speed = walkPace
        } else {
            walkPlayback = owner.playAnimation(clip.repeat(),
                                               transitionDuration: 0.25,
                                               startsPaused: false)
            walkPlayback?.speed = walkPace
        }

        // Keep the outline shell walking in lockstep with the body
        if let oOwner = outlineAnimOwner, let oClip = outlineClip {
            if let playback = outlinePlayback, playback.isValid {
                playback.speed = walkPace
            } else {
                outlinePlayback = oOwner.playAnimation(oClip.repeat(),
                                                       transitionDuration: 0.25,
                                                       startsPaused: false)
                outlinePlayback?.speed = walkPace
            }
            if let body = walkPlayback, let outline = outlinePlayback {
                outline.time = body.time
            }
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
            outlinePlayback?.speed = 0
            if let outline = outlinePlayback { outline.time = playback.time }
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
