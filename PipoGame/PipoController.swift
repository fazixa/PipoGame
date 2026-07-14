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
    /// Textured-skin look (fingerprint bump + noise displacement) — mutually
    /// exclusive with toon, picked from the same brush menu.
    @Published var isBumpy = false
    @Published var isHandMode = false
    /// True while hand mode is on but no palm is currently detected.
    @Published var searchingForHand = false
    @Published var isGeoActive = false
    /// When on, the one-finger drag pushes/pulls Pipo along camera depth
    /// instead of moving him left/right/up — a single 2D drag can't
    /// disambiguate 3 axes at once, so this is an explicit mode switch
    /// rather than trying to read it from a second simultaneous gesture.
    @Published var isZAxisDragMode = false

    weak var handTracker: HandTracker?
    let geospatial = GeospatialManager()

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
    private let bumpyStyle = BumpyStyle()
    private var outlineAnimOwner: Entity?
    private var outlineClip: AnimationResource?
    private var outlinePlayback: AnimationPlaybackController?

    private var time: Float = 0
    private var walkPhase: Float = 0
    private var baseScale: Float = 1

    // Geospatial ("giant Pipo among buildings") mode
    private var isGeoAnchored = false
    /// Manual drag offset (world-space), on top of wherever ARCore's live
    /// tracking puts the anchor — reset on each new placement.
    private var geoOffset: SIMD3<Float> = .zero
    /// Multiplier over his natural (~15cm) size when pinned to a distant
    /// building — otherwise he'd be sub-pixel at that range.
    private let geoGiantScale: Float = 150
    /// Extra multiplier on top of the base drag sensitivity, just for
    /// Z-axis depth — moving him between distant buildings means covering
    /// city-block distances, not fine nudges.
    private let zAxisDragMultiplier: Float = 25

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

    private var isDragging = false

    /// Starts a one-finger drag anywhere on screen — doesn't matter where
    /// relative to Pipo.
    func beginDrag(at screenPoint: CGPoint) -> Bool {
        guard pipo != nil, arView != nil else { return false }
        switch state {
        case .unplaced, .handFollowing:
            return false
        default:
            break
        }

        if case .walking = state {
            walkPlayback?.stop(blendOutDuration: 0.2)
            walkPlayback = nil
            state = .idle
        } else if case .stoppingWalk = state {
            walkPlayback?.stop(blendOutDuration: 0.2)
            walkPlayback = nil
            state = .idle
        }

        isDragging = true
        return true
    }

    func toggleZAxisDragMode() {
        isZAxisDragMode.toggle()
    }

    /// One-finger drag. Normal placement always keeps him grounded: his root
    /// snaps directly to wherever the ARKit plane raycast hits under the
    /// touch point, every frame, no matter where on his body you grabbed —
    /// dragging him across a surface should never leave him floating above
    /// it. Geospatial placement has no nearby surface to slide along, so it
    /// instead uses camera-relative offset math: left/right/up normally, or
    /// push/pull along camera depth in Z-axis mode (drag up = away, drag
    /// down = closer) — an explicit mode switch rather than trying to read
    /// a third axis out of one 2D gesture.
    func dragBy(_ delta: CGPoint, at screenPoint: CGPoint) {
        guard isDragging else { return }
        guard isGeoAnchored else {
            guard let arView,
                  let result = arView.raycast(from: screenPoint, allowing: .estimatedPlane,
                                              alignment: .any).first else { return }
            let t = result.worldTransform.columns.3
            pipo?.setPosition(SIMD3<Float>(t.x, t.y, t.z), relativeTo: nil)
            return
        }
        if isZAxisDragMode {
            // Depth needs to cover city-block/kilometer distances, not the
            // fine nudges lateral positioning wants, so it gets a much
            // bigger multiplier than the shared base sensitivity.
            applyCameraRelativeOffset(right: 0, up: 0, forward: Float(-delta.y) * zAxisDragMultiplier)
        } else {
            applyCameraRelativeOffset(right: Float(delta.x), up: Float(-delta.y), forward: 0)
        }
    }

    func endDrag() {
        isDragging = false
    }

    /// Geospatial drag math only (see dragBy) — accumulates into `geoOffset`,
    /// which update() re-applies on top of ARCore's live tracking every
    /// frame. Writing directly to the anchor's position here would just get
    /// overwritten next frame.
    private func applyCameraRelativeOffset(right: Float, up: Float, forward: Float) {
        guard let arView else { return }
        let cam = arView.cameraTransform.matrix
        let rightAxis = SIMD3<Float>(cam.columns.0.x, cam.columns.0.y, cam.columns.0.z)
        let upAxis = SIMD3<Float>(cam.columns.1.x, cam.columns.1.y, cam.columns.1.z)
        let forwardAxis = -SIMD3<Float>(cam.columns.2.x, cam.columns.2.y, cam.columns.2.z)
        let sensitivity: Float = 0.0015 * scaleFactor
        geoOffset += (rightAxis * right + upAxis * up + forwardAxis * forward) * sensitivity
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

    func toggleGeospatial() {
        isGeoActive.toggle()
        geospatial.enabled = isGeoActive
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
        if isBumpy {
            bumpyStyle.remove()
            isBumpy = false
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

    func toggleBumpy() {
        guard let pipo, supportsToon else { return }
        if isBumpy {
            bumpyStyle.remove()
            isBumpy = false
            return
        }
        if isToon {
            toonStyle.remove()
            outlineAnimOwner = nil
            outlineClip = nil
            outlinePlayback = nil
            isToon = false
        }
        bumpyStyle.apply(to: pipo)
        isBumpy = true
    }

    private func firstEntityWithAnimations(in entity: Entity) -> Entity? {
        if !entity.availableAnimations.isEmpty { return entity }
        for child in entity.children {
            if let found = firstEntityWithAnimations(in: child) { return found }
        }
        return nil
    }

    /// Pinch-to-scale, clamped to 0.2x–400x of Pipo's natural size normally.
    /// Geospatial mode gets a much higher ceiling (5000x, i.e. skyscraper
    /// scale) — standing him among distant buildings is the whole point,
    /// and 400x wasn't enough headroom for that.
    func pinch(by factor: Float) {
        guard let pipo else { return }
        let maxMultiplier: Float = isGeoAnchored ? 5000 : 400
        let scaled = min(max(pipo.scale.x * factor, baseScale * 0.2), baseScale * maxMultiplier)
        pipo.scale = SIMD3<Float>(repeating: scaled)
    }

    /// Two-finger rotate gesture, turning Pipo in place around the vertical
    /// (world up) axis.
    func rotate(by radians: Float) {
        guard let pipo else { return }
        let spin = simd_quatf(angle: -radians, axis: [0, 1, 0])
        pipo.setOrientation(spin * pipo.orientation(relativeTo: nil), relativeTo: nil)
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
        bumpyStyle.remove()
        isBumpy = false
        supportsToon = false
        anchor?.removeFromParent()
        anchor = nil
        pipo = nil
        isGeoAnchored = false
        geoOffset = .zero
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

    private func loadFreshPipoEntity() -> Entity {
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
        return pipo
    }

    private func place(at transform: simd_float4x4) {
        guard let arView else { return }
        let pipo = loadFreshPipoEntity()

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

    /// Pins Pipo (giant-scaled) to a real building, via a raycast against
    /// ARCore's Streetscape Geometry rather than ARKit's near-field plane —
    /// that's what makes "stand him between two distant buildings" possible
    /// at all. Re-tapping while already geo-anchored moves him to the new
    /// spot instead of spawning a second Pipo.
    func placeGeospatial(from arView: ARView, at point: CGPoint) {
        guard geospatial.placeAnchor(from: arView, at: point),
              let hit = geospatial.placedAnchor?.transform else { return }
        // Use only the hit's POSITION — its rotation encodes the building
        // facade's surface normal, which pinned Pipo flat against the wall
        // at whatever angle that surface happened to be (read as "placed
        // perpendicularly"). He should stand upright, not lie on the facade.
        geoOffset = .zero
        let hitPos = SIMD3<Float>(hit.columns.3.x, hit.columns.3.y, hit.columns.3.z)
        let upright = uprightTransform(at: hitPos, arView: arView)

        if let pipo, let anchor {
            anchor.setTransformMatrix(upright, relativeTo: nil)
            let giant = baseScale * geoGiantScale
            pipo.scale = SIMD3<Float>(repeating: giant)
        } else {
            let pipo = loadFreshPipoEntity()
            let anchor = AnchorEntity(world: upright)
            let finalScale = pipo.scale
            baseScale = finalScale.x
            pipo.scale = finalScale * geoGiantScale
            anchor.addChild(pipo)
            arView.scene.addAnchor(anchor)
            self.anchor = anchor
            self.pipo = pipo
            state = .idle
            isPlaced = true
        }
        isGeoAnchored = true
    }

    /// World transform standing upright at `position`, facing the camera —
    /// same convention as the near-field place(at:) pop-in.
    private func uprightTransform(at position: SIMD3<Float>, arView: ARView) -> simd_float4x4 {
        let toCamera = arView.cameraTransform.translation - position
        let yaw = atan2(toCamera.x, toCamera.z)
        let rotation = simd_quatf(angle: yaw, axis: [0, 1, 0])
        return Transform(scale: .one, rotation: rotation, translation: position).matrix
    }

    // MARK: - Per-frame update

    func update(deltaTime dt: Float) {
        guard let pipo else { return }
        time += dt

        if isToon, let arView {
            toonStyle.updateThickness(cameraWorldPosition: arView.cameraTransform.translation,
                                      worldScale: pipo.scale.x)
        }

        // A geospatial anchor's position keeps refining as ARCore's tracking
        // of that real-world point improves, so re-apply it every frame
        // rather than just once at placement. Only the position is taken —
        // see placeGeospatial's uprightTransform for why rotation isn't.
        if isGeoAnchored, let anchor, let arView, let hit = geospatial.placedAnchor?.transform {
            let hitPos = SIMD3<Float>(hit.columns.3.x, hit.columns.3.y, hit.columns.3.z) + geoOffset
            anchor.setTransformMatrix(uprightTransform(at: hitPos, arView: arView), relativeTo: nil)
        }

        // Keep the outline shell locked to whichever clip the body is
        // playing. Walking already re-syncs every frame via
        // startWalkClipIfNeeded(), but the sitting loop only synced once at
        // the moment it started — fine for a few seconds, but two separate
        // AnimationPlaybackControllers looping the same clip drift apart
        // over time, which read as the outline trailing the body by a
        // couple frames the longer he sits.
        if let outline = outlinePlayback, outline.isValid,
           let body = sitPlayback ?? walkPlayback, body.isValid {
            outline.time = body.time
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
