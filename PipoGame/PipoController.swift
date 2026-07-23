import RealityKit
import ARKit
import Combine
import simd
import UIKit

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
        /// Game mode: the on-screen joystick steers him directly. Same
        /// kinematic grounding/fall physics as .walking — only the source
        /// of the heading changes.
        case driving
    }

    @Published var isPlaced = false
    @Published var isSitting = false
    /// Sit is procedural-only until a sit clip is authored for the rigged Pipo.
    @Published var supportsSit = true
    /// TEMP: trajectory-drawing prototype. While true, taps add path points
    /// (visualized with markers) instead of moving Pipo immediately.
    @Published var isDrawingPath = false
    /// TEMP: freehand-move prototype. While true, a 3-axis gizmo (a child
    /// of Pipo, so it inherits his position/rotation/scale automatically)
    /// is shown, and dragging one of its arms moves him along that axis,
    /// independent of any detected surface.
    @Published var isFreehand = false
    /// Game mode: a floating on-screen joystick drives Pipo directly,
    /// camera-relative, through the same kinematic controller as walking
    /// (grounded footprint sweep, falls, landing clip).
    @Published var isGameMode = false
    /// Live joystick vector from the UI overlay — x = screen-right,
    /// y = screen-up, magnitude 0...1. Deliberately not @Published: it
    /// changes every drag tick and only update() reads it.
    var joystickInput: SIMD2<Float> = .zero
    private let joystickDeadzone: Float = 0.15
    /// Full stick deflection drives him at this multiple of natural walk
    /// pace — the walk clip is sped up by the same factor, so feet stay
    /// planted even in overdrive.
    private let driveTopSpeedMultiplier: Float = 1.5

    /// Toon look: flat unlit color + inverted-hull outline (see ToonStyle).
    @Published var isToon = false
    /// Toon needs the rigged Pipo (the outline shell mirrors its joints).
    @Published var supportsToon = false

    /// Set by ContentView; used to stop recording without on-screen UI.
    var onLongPress: (() -> Void)?

    weak var arView: ARView?

    private var state: State = .unplaced
    private var anchor: AnchorEntity?
    private var pipo: Entity?

    // TEMP: freehand-move prototype. gizmoRoot is a child of pipo (see
    // createGizmo) so it automatically inherits his position, rotation,
    // and scale — no manual per-frame syncing needed.
    private var gizmoRoot: Entity?
    private var gizmoArms: [ModelEntity] = []

    // Toon look. The outline shell is a clone of Pipo whose pose is synced
    // by copying the body's live jointTransforms every frame in update() —
    // no playback mirroring, so it stays locked to the body across
    // walk/sit/land clips, crossfades, and freezes alike. (Main's original
    // implementation mirrored playback controllers instead; with multiple
    // clips from separate USDZs, joint copying is simpler and drift-free.)
    private let toonStyle = ToonStyle()
    private var outlineMeshEntity: ModelEntity?
    private var freehandDragAxis: SIMD3<Float>?
    private var freehandDragStartPosition: SIMD3<Float> = .zero
    private var freehandDragScreenDirection: CGVector = .zero
    private var freehandDragMetersPerPoint: Float = 0

    // Pose-reactive footprint. meshEntity is cached at placement so
    // refreshFootprint doesn't need to re-search pipo's hierarchy every
    // frame. footprintOffsets is the XZ convex hull of his current joint
    // positions, relative to his own position — groundedHeight's sweep
    // below uses these instead of a fixed circle, so the grounded-check
    // literally follows his real silhouette every frame, in whatever pose
    // he's currently in (walking, sitting, climbing, ...).
    private var meshEntity: ModelEntity?
    private var footprintOffsets: [SIMD2<Float>] = []

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

    // TEMP: edge-dangling sit prototype. When toggleSit() finds a nearby
    // ledge, it walks Pipo there via the normal .walking state instead of
    // sitting in place — these carry the "sit once you arrive, facing this
    // direction" intent through that walk, since PathPoint/State don't
    // otherwise have anywhere to store it.
    private var sitAfterWalkArrival = false
    private var pendingSitFacing: SIMD3<Float>?
    /// The edge's real surface height — the target the one-shot pelvis
    /// correction below aligns the live Hips joint to once the sit clip
    /// has actually taken over the pose.
    private var pendingSitTargetY: Float?
    /// Counts up once .sitting begins; at pelvisCorrectionDelay the sit
    /// clip's crossfade (see startSitClipIfNeeded) has finished blending
    /// in, so meshEntity.jointTransforms reflects the real seated pose and
    /// it's safe to sample the Hips joint for the one-shot correction.
    private var pelvisCorrectionTimer: Float?
    private let pelvisCorrectionDelay: Float = 0.3
    /// The Hips joint's own pivot isn't exactly at the lowest point of his
    /// seated geometry, so aligning it exactly to the surface leaves him
    /// sinking in a bit — small upward bias on top of the precise
    /// correction to compensate. Tune directly if he's still in/above the
    /// surface.
    private var pelvisHeightBias: Float { characterHeight * 0.11 }
    /// Same idea horizontally: nudges him slightly further out over the
    /// drop (along the sit facing) so his butt sits right at the lip
    /// instead of a touch behind it. Tune directly if he's over/short.
    private var pelvisForwardBias: Float { characterHeight * 0.11 }
    /// DEBUG: visualizes findDanglingEdge's search — small dots at every
    /// probed sample point, colored by outcome, plus the winning edge.
    /// Cleared and rebuilt on each search.
    private var edgeDebugMarkers: [AnchorEntity] = []
    /// TEMP DEBUG: while true, tapping Sit only runs findDanglingEdge for
    /// its marker visualization — Pipo doesn't actually walk or sit. Flip
    /// to false to restore the real behavior once the search looks right.
    private let edgeSearchDebugOnly = false

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
    private var characterHeight: Float {
        guard let pipo else { return 0.02 }
        return pipo.visualBounds(relativeTo: nil).extents.y
    }
    /// How close beneath the footprint's own base a hit has to be to count as
    /// "standing on it," vs. "that's an edge, fall" — the same role a
    /// CharacterController's step-offset plays: a fraction of the
    /// character's OWN height (not a fixed distance scaled by zoom level),
    /// so small bumps/steps still get smoothly walked over but a real edge
    /// — even a modest one, well under a full character-height drop — is
    /// still recognized as one. (0.15 * scaleFactor previously worked out
    /// to roughly a FULL character-height at placement scale, since
    /// characterHeight itself is already close to that same 0.15m — meaning
    /// almost nothing could ever register as "off the edge.")
    private var groundedTolerance: Float { characterHeight * 0.2 }

    // TEMP: climb prototype. Reaching a red (vertical) path point triggers
    // climbing to whatever's next in the queue, instead of the normal
    // grounded walk.
    private var climbClip: AnimationResource?
    private var climbPlayback: AnimationPlaybackController?
    /// Keeps climbClip's source entity retained — see landEntity for why.
    private var climbEntity: Entity?

    private var sitClip: AnimationResource?
    private var sitPlayback: AnimationPlaybackController?
    /// Keeps sitClip's source entity retained — see landEntity for why.
    private var sitEntity: Entity?

    private var time: Float = 0
    private var walkPhase: Float = 0
    private var baseScale: Float = 1

    // Ground speed matches the walk cycle's stride at natural size; both
    // scale with pinch so feet don't slide when Pipo is giant or tiny.
    private var scaleFactor: Float {
        guard let pipo, baseScale > 0 else { return 1 }
        return pipo.scale.x / baseScale
    }
    // Tuned for Pipo's walk (Universal6 rerig, Mixamo chain): clean flat
    // stance on both feet — measured directly off the Walk action (57-joint
    // rig, 31-frame/30fps loop) by sampling mixamorig:LeftFoot/RightFoot
    // world position through their stance windows (frames 11-20 and 25-5
    // respectively, where foot height is pinned flat at 0.3918). Both feet
    // slide back at the same CONSTANT rate, ~4.72 m/s at the model's raw
    // ~4.62 m size, so that IS the slide-free ground speed. PipoAsset
    // renders at 0.034x, and walkSpeed applies at that already-scaled size
    // (scaleFactor normalizes to 1.0 at placement scale), so fold it in:
    // 4.72 * 0.034 ≈ 0.161.
    private var walkSpeed: Float { (usesClips ? 0.161 : 0.22) * scaleFactor }  // m/s
    private let stepRate: Float = 11                          // procedural gait, rad/s
    // TEMP: climb prototype — fallback lerp speed if no climb clip loaded.
    private var climbSpeed: Float { walkSpeed * 0.5 }
    // MARK: - Input

    /// Only places Pipo on first tap — tap-to-walk is disabled in the main
    /// view after that (dragging/twisting are how he's repositioned now).
    /// Drawing mode still uses taps to place path points, handled above
    /// this in the switch since it applies regardless of placement state.
    func handleTap(on result: ARRaycastResult) {
        if isDrawingPath {
            addPathPoint(at: result.worldTransform)
            return
        }
        guard case .unplaced = state else { return }
        place(at: result.worldTransform)
    }

    /// Drags Pipo to wherever the touch currently raycasts, only while
    /// he's stationary (idle/sitting) — dragging mid-walk, mid-fall, or
    /// mid-climb would fight the position that state's own update logic is
    /// actively driving.
    func drag(to result: ARRaycastResult) {
        guard let pipo, !isDrawingPath, !isFreehand else { return }
        switch state {
        case .idle, .sitting:
            let t = result.worldTransform.columns.3
            pipo.setPosition([t.x, t.y, t.z], relativeTo: nil)
        case .unplaced, .walking, .stoppingWalk, .falling, .landing, .climbing, .driving:
            break
        }
    }

    /// Twists Pipo in place by a delta rotation around the vertical axis,
    /// same state restriction as drag() above.
    func rotate(by deltaRadians: Float) {
        guard let pipo, !isDrawingPath else { return }
        switch state {
        case .idle, .sitting:
            let delta = simd_quatf(angle: -deltaRadians, axis: [0, 1, 0])
            pipo.setOrientation(delta * pipo.orientation(relativeTo: nil), relativeTo: nil)
        case .unplaced, .walking, .stoppingWalk, .falling, .landing, .climbing, .driving:
            break
        }
    }

    // MARK: - TEMP: freehand-move prototype

    /// Enters/exits freehand mode (only from .idle/.sitting, same
    /// restriction as drag/rotate). Shows/hides the 3-axis gizmo.
    func toggleToon() {
        guard let pipo, supportsToon else { return }
        if isToon {
            toonStyle.remove()
            outlineMeshEntity = nil
            isToon = false
            return
        }
        // The freehand gizmo lives as a child of pipo — detach it while the
        // style clones him so its arms don't get swept into the outline
        // shell / flat-color material swap.
        let gizmo = gizmoRoot
        gizmo?.removeFromParent()
        let clone = toonStyle.apply(to: pipo)
        if let gizmo { pipo.addChild(gizmo) }
        guard let clone else { return }
        outlineMeshEntity = firstMeshEntity(in: clone)
        isToon = true
    }

    /// Enters/exits joystick game mode. Entering while sitting stands him
    /// up first; exiting mid-drive lets the walk clip freeze naturally.
    func toggleGameMode() {
        guard isPlaced, !isDrawingPath, !isFreehand else { return }
        if isGameMode {
            isGameMode = false
            joystickInput = .zero
            if case .driving = state {
                state = usesClips ? .stoppingWalk : .idle
            }
        } else {
            if case .sitting = state { toggleSit() }
            isGameMode = true
        }
    }

    func toggleFreehand() {
        guard isPlaced, !isDrawingPath else { return }
        if isFreehand {
            isFreehand = false
            removeGizmo()
        } else {
            switch state {
            case .idle, .sitting:
                isFreehand = true
                createGizmo()
            case .unplaced, .walking, .stoppingWalk, .falling, .landing, .climbing, .driving:
                break
            }
        }
    }

    /// Gizmo arms are children of pipo, so their length/radius are computed
    /// in HIS OWN local (unscaled-by-his-current-transform) space — that
    /// way, since children inherit their parent's scale automatically,
    /// the gizmo stays correctly proportioned as he's pinched from 0.2x up
    /// to 500x without any manual scale-factor math here.
    private func createGizmo() {
        guard let pipo else { return }
        let localHeight = pipo.visualBounds(relativeTo: pipo).extents.y
        let root = Entity()
        // Lift up from his origin (feet) to roughly waist height, so the
        // gizmo doesn't sit buried at ground level / overlapping his legs.
        root.position = [0, localHeight * 0.5, 0]

        let axes: [(SIMD3<Float>, UIColor, String)] = [
            ([1, 0, 0], .systemRed, "gizmoX"),
            ([0, 1, 0], .systemGreen, "gizmoY"),
            ([0, 0, 1], .systemBlue, "gizmoZ"),
        ]
        let length: Float = localHeight * 0.5
        let radius: Float = localHeight * 0.004
        // Touch target is a separate, much fatter invisible capsule over
        // the same length — grabbing a visually thin arm was unreliable
        // since the auto-generated collision shape hugged its tiny radius
        // exactly. This doesn't change how thin the arm looks, only how
        // generous the hit area around it is.
        let touchRadius: Float = localHeight * 0.04
        var arms: [ModelEntity] = []
        for (direction, color, name) in axes {
            let arm = ModelEntity(mesh: .generateCylinder(height: length, radius: radius),
                                  materials: [UnlitMaterial(color: color)])
            arm.name = name
            // generateCylinder is aligned along +Y by default; rotate to
            // point along this axis, then offset so it extends OUTWARD
            // from the origin rather than being centered on it.
            arm.orientation = simd_quaternion([0, 1, 0], direction)
            arm.position = direction * (length / 2)
            arm.collision = CollisionComponent(shapes: [.generateCapsule(height: length, radius: touchRadius)])
            root.addChild(arm)
            arms.append(arm)
        }
        pipo.addChild(root)
        gizmoRoot = root
        gizmoArms = arms
    }

    private func removeGizmo() {
        gizmoRoot?.removeFromParent()
        gizmoRoot = nil
        gizmoArms = []
        freehandDragAxis = nil
    }

    /// Called on pan-gesture-began at the touch point; returns true if a
    /// gizmo arrow was grabbed (in which case the caller should route
    /// subsequent pan updates to updateFreehandDrag instead of the normal
    /// surface-snapping drag()).
    func beginFreehandDrag(at point: CGPoint) -> Bool {
        guard isFreehand, let arView, let pipo,
              let hitEntity = arView.entity(at: point) else { return false }
        let localAxis: SIMD3<Float>
        switch hitEntity.name {
        case "gizmoX": localAxis = [1, 0, 0]
        case "gizmoY": localAxis = [0, 1, 0]
        case "gizmoZ": localAxis = [0, 0, 1]
        default: return false
        }
        // The gizmo is a child of pipo, so its arms rotate along with him —
        // resolve the grabbed arm's CURRENT world direction through his
        // orientation, not a fixed world axis, so dragging a tilted arm
        // moves him along where it's actually pointing.
        let axis = pipo.orientation(relativeTo: nil).act(localAxis)
        let worldPosition = pipo.position(relativeTo: nil)
        // Project the axis into screen space (a short probe point along it)
        // so a 2D finger drag can be resolved into a 1D distance along that
        // axis, regardless of camera angle — the same technique real 3D
        // move gizmos use, rather than assuming the axis is screen-aligned.
        guard let originScreen = arView.project(worldPosition) else { return false }
        let probeDistance: Float = 0.05
        guard let probeScreen = arView.project(worldPosition + axis * probeDistance) else { return false }
        let dx = Float(probeScreen.x - originScreen.x)
        let dy = Float(probeScreen.y - originScreen.y)
        let screenDistance = sqrt(dx * dx + dy * dy)
        guard screenDistance > 1 else { return false } // axis pointing at/away from camera
        freehandDragAxis = axis
        freehandDragStartPosition = worldPosition
        freehandDragScreenDirection = CGVector(dx: CGFloat(dx / screenDistance), dy: CGFloat(dy / screenDistance))
        freehandDragMetersPerPoint = probeDistance / screenDistance
        return true
    }

    /// `translation` is the pan gesture's cumulative translation since it
    /// began (UIPanGestureRecognizer.translation(in:)), in points.
    func updateFreehandDrag(translation: CGPoint) {
        guard let axis = freehandDragAxis, let pipo else { return }
        let projected = Float(translation.x) * Float(freehandDragScreenDirection.dx)
                       + Float(translation.y) * Float(freehandDragScreenDirection.dy)
        let worldDelta = projected * freehandDragMetersPerPoint
        let newPosition = freehandDragStartPosition + axis * worldDelta
        // gizmoRoot is a child of pipo, so it moves (and rotates, and
        // scales) with him automatically — nothing else to sync here.
        pipo.setPosition(newPosition, relativeTo: nil)
    }

    func endFreehandDrag() {
        freehandDragAxis = nil
    }

    // MARK: - TEMP: trajectory-drawing prototype

    /// Enters drawing mode (only from .idle, with Pipo already placed) or,
    /// if already drawing, commits the drawn points as a multi-waypoint
    /// walk and starts moving through them in order.
    func toggleDrawPath() {
        guard isPlaced, !isFreehand else { return }
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
        guard supportsSit, !isDrawingPath, !isFreehand else { return }
        switch state {
        case .sitting:
            // Force a fresh walk-clip crossfade out of the sit pose rather
            // than resuming any stale frozen walk controller from before
            // sitting — resuming a dormant controller doesn't blend, it
            // just pops straight to wherever it was left off (the same
            // class of bug fixed for the land clip earlier).
            walkPlayback?.stop()
            walkPlayback = nil
            state = usesClips ? .stoppingWalk : .idle
            startWalkClipIfNeeded()
            isSitting = false
        case .idle, .walking, .stoppingWalk:
            // TEMP DEBUG: edgeSearchDebugOnly gate below — set to false to
            // restore the real walk-there-and-sit behavior.
            guard let edge = findDanglingEdge() else {
                if !edgeSearchDebugOnly {
                    // Deliberately NOT stopping walkPlayback here — leaving
                    // it playing lets startSitClipIfNeeded's playAnimation
                    // crossfade directly from the walk/idle pose into the
                    // sit pose.
                    state = .sitting
                    isSitting = true
                }
                return
            }
            guard !edgeSearchDebugOnly else { return }
            pendingSitFacing = edge.facing
            pendingSitTargetY = edge.position.y
            sitAfterWalkArrival = true
            state = .walking(target: PathPoint(position: edge.position, isVertical: false))
            startWalkClipIfNeeded()
        case .unplaced, .falling, .landing, .climbing, .driving:
            break // TEMP: can't sit mid-air, mid-landing, mid-climb, or mid-drive
        }
    }

    /// Pinch-to-scale, clamped to 0.2x–500x of Pipo's natural size.
    func pinch(by factor: Float) {
        guard let pipo else { return }
        let scaled = min(max(pipo.scale.x * factor, baseScale * 0.2), baseScale * 500)
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
        climbClip = nil
        climbPlayback = nil
        climbEntity = nil
        sitClip = nil
        sitPlayback = nil
        sitEntity = nil
        toonStyle.remove()
        outlineMeshEntity = nil
        isToon = false
        supportsToon = false
        // gizmoRoot is a child of pipo, so it's already gone once pipo is
        // released above — just reset the tracking state here.
        gizmoRoot = nil
        gizmoArms = []
        isFreehand = false
        meshEntity = nil
        footprintOffsets = []
        sitAfterWalkArrival = false
        pendingSitFacing = nil
        pendingSitTargetY = nil
        pelvisCorrectionTimer = nil
        clearEdgeDebugMarkers()
        clearPathMarkers()
        drawnPoints = []
        pathQueue = []
        isDrawingPath = false
        state = .unplaced
        isPlaced = false
        isSitting = false
        supportsSit = true
        isGameMode = false
        joystickInput = .zero
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
            sitClip = loaded.sitClip
            sitEntity = loaded.sitEntity
            supportsSit = sitClip != nil
            supportsToon = true
        } else {
            pipo = PipoBuilder.build()
            supportsSit = true
            supportsToon = false
        }

        let anchor = AnchorEntity(world: transform)

        // Face the camera when appearing.
        let position = SIMD3<Float>(transform.columns.3.x,
                                    transform.columns.3.y,
                                    transform.columns.3.z)
        let toCamera = arView.cameraTransform.translation - position
        pipo.orientation = simd_quatf(angle: atan2(toCamera.x, toCamera.z), axis: [0, 1, 0])

        // Starting size is 10x Pipo's natural scale; baseScale stays at the
        // natural (un-multiplied) reference so scaleFactor/pinch clamping
        // continue to mean "relative to his actual natural size."
        let naturalScale = pipo.scale
        baseScale = naturalScale.x
        let finalScale = naturalScale * 10
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

        meshEntity = firstMeshEntity(in: pipo)
        refreshFootprint()
    }

    /// Rebuilds the footprint used by groundedHeight's sweep from the mesh
    /// entity's CURRENT joint transforms (HasModel.jointTransforms —
    /// live/animated, unlike ModelComponent's own mesh resource, which
    /// stays at bind pose since skinning deforms vertices on the GPU rather
    /// than mutating CPU-side geometry). Called every frame from update()
    /// so the grounded-check follows his actual current pose/silhouette,
    /// not a fixed circle, and not frozen at whatever pose he was in when
    /// placed.
    private func refreshFootprint() {
        guard let pipo, let meshEntity else { return }
        let localPoints = meshEntity.jointTransforms.map(\.translation)
        guard localPoints.count >= 4 else { return } // convex hull needs a non-degenerate point set
        let worldPoints = localPoints.map { meshEntity.convert(position: $0, to: nil) }

        // XZ offsets relative to Pipo's own position, so groundedHeight can
        // apply them to whatever position it's testing (which may be a
        // hypothetical next-step position, not necessarily where his
        // joints currently are).
        let pipoPosition = pipo.position(relativeTo: nil)
        let worldXZOffsets = worldPoints.map { SIMD2<Float>($0.x - pipoPosition.x, $0.z - pipoPosition.z) }
        footprintOffsets = convexHullXZ(worldXZOffsets)
    }

    /// One-shot correction, run once the sit clip's crossfade has finished
    /// (see pelvisCorrectionTimer), that reads the LIVE Hips joint's actual
    /// world position — same jointTransforms data refreshFootprint uses —
    /// and shifts Pipo's root vertically so the real pelvis lands exactly
    /// on pendingSitTargetY (the edge's true surface height), instead of
    /// guessing a fixed downward offset that doesn't account for however
    /// this specific clip actually holds the pelvis relative to the root.
    private func correctPelvisHeight(pipo: Entity) {
        // Match by suffix, not exact name — the USD export fix sanitizes
        // ":" to "_" in joint names ("mixamorig:Hips" -> "mixamorig_Hips"),
        // so an exact-name lookup for the original Mixamo name silently
        // never matched, meaning this whole function was a no-op.
        // "root_x" is the pelvis on Pipo's ARP rig (Blender "root.x").
        guard let meshEntity, let targetY = pendingSitTargetY,
              let hipsIndex = meshEntity.jointNames.firstIndex(where: {
                  $0.hasSuffix("Hips") || $0.hasSuffix("root_x")
              }) else {
            pendingSitTargetY = nil
            return
        }
        let hipsLocal = meshEntity.jointTransforms[hipsIndex].translation
        let hipsWorldBefore = meshEntity.convert(position: hipsLocal, to: nil)
        // DEBUG: purple = where the Hips joint was actually sampled BEFORE
        // correcting — compare against the green edge marker to see
        // whether the measurement itself lines up with the target. Marker
        // material has depth-testing disabled (see addEdgeDebugMarker) so
        // it renders through his own body instead of being hidden inside it.
        addEdgeDebugMarker(at: hipsWorldBefore, color: .systemPurple, radius: characterHeight * 0.04)

        var position = pipo.position(relativeTo: nil)
        position.y -= (hipsWorldBefore.y - targetY) - pelvisHeightBias
        // Forward bias goes along his current facing, which at this point
        // IS the edge's outward direction — the arrival handler snapped him
        // to it before sitting began.
        let facing = pipo.orientation(relativeTo: nil).act(SIMD3<Float>(0, 0, 1))
        position.x += facing.x * pelvisForwardBias
        position.z += facing.z * pelvisForwardBias
        pipo.setPosition(position, relativeTo: nil)

        // DEBUG: cyan = where the Hips joint ends up AFTER correcting —
        // should land exactly on the green target if the math is right.
        let hipsWorldAfter = meshEntity.convert(position: hipsLocal, to: nil)
        addEdgeDebugMarker(at: hipsWorldAfter, color: .systemCyan, radius: characterHeight * 0.04)
        pendingSitTargetY = nil
    }

    private func firstMeshEntity(in entity: Entity) -> ModelEntity? {
        if let model = entity as? ModelEntity, model.model != nil {
            return model
        }
        for child in entity.children {
            if let found = firstMeshEntity(in: child) { return found }
        }
        return nil
    }

    /// Standard 2D convex hull (Andrew's monotone chain) over a point set's
    /// XZ coordinates — used to reduce ~65 joint positions down to the
    /// handful that actually define his current footprint's outline.
    private func convexHullXZ(_ points: [SIMD2<Float>]) -> [SIMD2<Float>] {
        guard points.count >= 3 else { return points }
        let sorted = points.sorted { $0.x != $1.x ? $0.x < $1.x : $0.y < $1.y }

        func cross(_ o: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }

        var lower: [SIMD2<Float>] = []
        for p in sorted {
            while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 {
                lower.removeLast()
            }
            lower.append(p)
        }
        var upper: [SIMD2<Float>] = []
        for p in sorted.reversed() {
            while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 {
                upper.removeLast()
            }
            upper.append(p)
        }
        lower.removeLast()
        upper.removeLast()
        return lower + upper
    }

    // MARK: - Per-frame update

    func update(deltaTime dt: Float) {
        guard let pipo else { return }
        time += dt

        // Every frame — groundedHeight's sweep depends on footprintOffsets
        // being current.
        refreshFootprint()

        switch state {
        case .unplaced:
            break

        case .idle:
            if isGameMode, simd_length(joystickInput) > joystickDeadzone {
                state = .driving
            } else if !usesClips {
                animateIdle(pipo: pipo, dt: dt)
            }

        case .walking(let target):
            walk(pipo: pipo, toward: target, dt: dt)

        case .stoppingWalk:
            freezeWalkClipAtContactFrame()

        case .sitting:
            if usesClips {
                startSitClipIfNeeded()
            } else {
                animateSit(pipo: pipo, dt: dt)
            }
            if var timer = pelvisCorrectionTimer {
                timer += dt
                if timer >= pelvisCorrectionDelay {
                    correctPelvisHeight(pipo: pipo)
                    pelvisCorrectionTimer = nil
                } else {
                    pelvisCorrectionTimer = timer
                }
            }

        case .falling(let resumeTarget, let landingPosition):
            fall(pipo: pipo, resumeTarget: resumeTarget, landingPosition: landingPosition, dt: dt)

        case .landing(let resumeTarget):
            updateLanding(pipo: pipo, resumeTarget: resumeTarget)

        case .climbing(let target, let startPosition, let startTime):
            climb(pipo: pipo, target: target, startPosition: startPosition, startTime: startTime)

        case .driving:
            drive(pipo: pipo, dt: dt)
        }

        // Toon: mirror the body's live pose onto the outline shell. All
        // toon materials are static now (no camera-dependent params), so
        // pose sync is the only per-frame work.
        if isToon, let body = meshEntity, let outline = outlineMeshEntity {
            outline.jointTransforms = body.jointTransforms
            // Joints and blend shape weights are separate channels: the
            // knee correctives ride the body's playing clip as
            // blendShapeWeights, and the shell plays no clips — copy
            // the live weights across too or its knees stay uncorrected.
            if let weights = body.components[BlendShapeWeightsComponent.self] {
                outline.components.set(weights)
            }
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
            // TEMP: edge-dangling sit prototype — arrived at the ledge
            // toggleSit() found; face outward over the drop (the direction
            // findDanglingEdge recorded) and sit there, instead of the
            // normal stop-walking fallback.
            if sitAfterWalkArrival {
                sitAfterWalkArrival = false
                if let facing = pendingSitFacing {
                    pipo.setOrientation(simd_quatf(angle: atan2(facing.x, facing.z), axis: [0, 1, 0]),
                                        relativeTo: nil)
                }
                pendingSitFacing = nil
                pelvisCorrectionTimer = 0
                PipoLog.log("arrived at edge -> .sitting. walkPlayback valid=\(walkPlayback?.isValid ?? false) playing=\(walkPlayback?.isPlaying ?? false) usesClips=\(usesClips) sitClip=\(sitClip != nil)")
                state = .sitting
                isSitting = true
                return
            }
            state = usesClips ? .stoppingWalk : .idle
            return
        }

        let direction = heading / distance

        let step = min(walkSpeed * dt, distance)
        position += resolveHorizontalCollision(position: position,
                                                displacement: direction * step,
                                                target: target)

        if sitAfterWalkArrival {
            // This walk is a deliberate approach to a ledge already
            // validated by findDanglingEdge — the normal grounded-check
            // below would misfire here, since part of his footprint is
            // SUPPOSED to end up overhanging the drop as he nears the
            // edge. Skip it and just glide his height toward the
            // already-known target Y instead of re-querying the ground
            // (and potentially reading "not supported") every step.
            position.y = damp(position.y, target.position.y, rate: 12, dt: dt)
            pipo.setPosition(position, relativeTo: nil)
        } else if let groundY = groundedHeight(at: position) {
            position.y = damp(position.y, groundY, rate: 12, dt: dt)
            pipo.setPosition(position, relativeTo: nil)
        } else if usesClips, landClip != nil {
            // Fully stop (not pause) the walk clip — a paused-but-present
            // controller can still compete with the land clip on the same
            // entity, since RealityKit layers animations by default rather
            // than replacing them. NOT pre-creating a paused land-clip
            // controller here anymore (startLandClip's old braced-pose
            // trick) — fall() now creates it fresh exactly at impact
            // instead, since stopping that pre-created controller and
            // immediately replacing it in the same call was a likely
            // culprit for the impact animation silently not taking effect.
            walkPlayback?.stop()
            walkPlayback = nil
            // Land a bit forward of the edge, in the direction Pipo was
            // walking, rather than dropping straight down from where he
            // left the ground — half a character-height of forward carry
            // reads as falling with momentum instead of stepping straight
            // off a cliff. eventualGroundY is wherever the mesh is found
            // below, however far down that turns out to be (the floor, most
            // likely) — the full-range downward search groundHeight already
            // does, just without the close-tolerance grounded check.
            let eventualGroundY = groundHeight(at: position) ?? position.y - characterHeight
            let forwardCarry = direction * (characterHeight * 0.5)
            let landingPosition = SIMD3<Float>(position.x + forwardCarry.x,
                                                eventualGroundY,
                                                position.z + forwardCarry.z)
            pipo.setPosition(position, relativeTo: nil)
            print("PIPOFALLDBG trigger: from=\(position) landingPosition=\(landingPosition) landClip=\(landClip != nil) owner=\(animationOwner != nil)")
            state = .falling(resumeTarget: target, landingPosition: landingPosition)
            return
        } else {
            pipo.setPosition(position, relativeTo: nil)
        }

        let desired = simd_quatf(angle: atan2(direction.x, direction.z), axis: [0, 1, 0])
        let current = pipo.orientation(relativeTo: nil)
        pipo.setOrientation(simd_slerp(current, desired, smoothing(rate: 10, dt: dt)), relativeTo: nil)

        if usesClips {
            startWalkClipIfNeeded()
        } else {
            animateGait(pipo: pipo, dt: dt)
        }
    }

    // MARK: - Game mode (joystick drive)

    /// Direct joystick locomotion: camera-relative heading, analog speed,
    /// and the SAME grounded-footprint sweep + fall trigger walk() uses —
    /// game mode changes who steers, not the physics.
    private func drive(pipo: Entity, dt: Float) {
        let magnitude = simd_length(joystickInput)
        guard isGameMode, magnitude > joystickDeadzone, let arView else {
            state = usesClips ? .stoppingWalk : .idle
            return
        }

        // Camera-relative frame flattened to XZ: stick-up = away from the
        // camera, stick-right = screen right.
        let cam = arView.cameraTransform.matrix
        var forward = SIMD3<Float>(-cam.columns.2.x, 0, -cam.columns.2.z)
        if simd_length_squared(forward) < 0.0001 {
            // Looking straight down — the camera's up axis is the best
            // remaining hint for "away from me."
            forward = SIMD3<Float>(cam.columns.1.x, 0, cam.columns.1.z)
        }
        forward = simd_normalize(forward)
        let right = SIMD3<Float>(-forward.z, 0, forward.x)
        let direction = simd_normalize(right * joystickInput.x + forward * joystickInput.y)

        // Analog: deadzone..full remaps to 0..topSpeed and drives both
        // ground speed and clip pace, so feet stay planted at any
        // deflection — including overdrive past natural walk pace.
        let throttle = min((magnitude - joystickDeadzone) / (1 - joystickDeadzone), 1)
            * driveTopSpeedMultiplier
        var position = pipo.position(relativeTo: nil)
        position += resolveHorizontalCollision(position: position,
                                               displacement: direction * (walkSpeed * throttle * dt))

        if let groundY = groundedHeight(at: position) {
            position.y = damp(position.y, groundY, rate: 12, dt: dt)
            pipo.setPosition(position, relativeTo: nil)
        } else if usesClips, landClip != nil {
            // Same fall-with-momentum trigger as walk(). The resume target
            // IS the landing spot, so after the landing clip plays he just
            // stands there awaiting further stick input.
            walkPlayback?.stop()
            walkPlayback = nil
            let eventualGroundY = groundHeight(at: position) ?? position.y - characterHeight
            let forwardCarry = direction * (characterHeight * 0.5)
            let landingPosition = SIMD3<Float>(position.x + forwardCarry.x,
                                               eventualGroundY,
                                               position.z + forwardCarry.z)
            pipo.setPosition(position, relativeTo: nil)
            state = .falling(resumeTarget: PathPoint(position: landingPosition, isVertical: false),
                             landingPosition: landingPosition)
            return
        } else {
            pipo.setPosition(position, relativeTo: nil)
        }

        let desired = simd_quatf(angle: atan2(direction.x, direction.z), axis: [0, 1, 0])
        let current = pipo.orientation(relativeTo: nil)
        pipo.setOrientation(simd_slerp(current, desired, smoothing(rate: 10, dt: dt)), relativeTo: nil)

        if usesClips {
            startWalkClipIfNeeded()
            walkPlayback?.speed = throttle
        } else {
            animateGait(pipo: pipo, dt: dt)
        }
    }

    // MARK: - Wall collision (kinematic probe + slide)

    /// Games-style lateral collision: probe the LiDAR mesh along the
    /// intended horizontal displacement at two body heights, using the
    /// pose-reactive footprint's extent along the motion direction as his
    /// radius. Blocked motion clamps at the wall (minus a skin margin) and
    /// the remainder slides along the wall plane — one re-probe on the
    /// slide direction, same as a CharacterController's collide-and-slide.
    /// Vertical (climbable) walk targets are exempt: the climb flow is
    /// SUPPOSED to reach the wall base.
    private func resolveHorizontalCollision(position: SIMD3<Float>,
                                            displacement: SIMD3<Float>,
                                            target: PathPoint? = nil) -> SIMD3<Float> {
        if let target, target.isVertical { return displacement }
        let horizontal = SIMD3<Float>(displacement.x, 0, displacement.z)
        var remaining = simd_length(horizontal)
        guard remaining > 1e-6 else { return displacement }
        var dir = horizontal / remaining
        var moved = SIMD3<Float>.zero

        for _ in 0..<2 {
            let dxz = SIMD2<Float>(dir.x, dir.z)
            let extent = footprintOffsets.map { simd_dot($0, dxz) }.max() ?? 0
            let radius = max(extent, characterHeight * 0.08) + characterHeight * 0.04

            var wall: (allowed: Float, normal: SIMD3<Float>)?
            for heightFraction: Float in [0.25, 0.6] {
                let origin = position + moved
                    + SIMD3<Float>(0, characterHeight * heightFraction, 0)
                guard let hit = meshRaycastHit(from: origin,
                                               to: origin + dir * (remaining + radius))
                else { continue }
                var normal = SIMD3<Float>(hit.normal.x, 0, hit.normal.z)
                let steepness = simd_length(normal)
                // mostly-horizontal normal = a wall; gentle slopes are the
                // ground system's business
                guard steepness > 0.6 else { continue }
                normal /= steepness
                if simd_dot(normal, dir) > 0 { normal = -normal }
                let allowed = simd_distance(origin, hit.position) - radius
                if wall == nil || allowed < wall!.allowed {
                    wall = (allowed, normal)
                }
            }

            guard let wall, wall.allowed < remaining else {
                moved += dir * remaining
                break
            }
            let step = max(wall.allowed, 0)
            moved += dir * step
            let leftover = remaining - step
            var slide = dir - wall.normal * simd_dot(dir, wall.normal)
            let slideLength = simd_length(slide)
            guard slideLength > 1e-4, leftover > 1e-6 else { break }
            slide /= slideLength
            dir = slide
            remaining = leftover * slideLength
        }
        return SIMD3<Float>(moved.x, displacement.y, moved.z)
    }

    private func groundHeight(at position: SIMD3<Float>) -> Float? {
        guard let arView else { return nil }

        // Prefer raycasting against the actual reconstructed LiDAR mesh
        // (real triangle geometry) over ARKit's estimated-plane model, which
        // extrapolates/smooths past real edges like a table boundary and
        // was causing Pipo to "float" past them.
        if let hitY = meshGroundHeight(at: position) {
            return hitY
        }

        // Fallback: estimated plane, for areas the mesh hasn't scanned yet.
        let query = ARRaycastQuery(origin: position + [0, 0.25, 0],
                                   direction: [0, -1, 0],
                                   allowing: .estimatedPlane,
                                   alignment: .any)
        return arView.session.raycast(query).first?.worldTransform.columns.3.y
    }

    /// Raycasts straight down against only the real reconstructed LiDAR
    /// mesh — no ARKit estimated-plane fallback. The grounded-check sweep
    /// below needs this distinction: an estimated plane extrapolates past a
    /// real edge like a table boundary, so if the sweep fell back to it,
    /// every sample would keep finding phantom "ground" past the edge and
    /// Pipo would never detect he'd walked off it.
    private func meshGroundHeight(at position: SIMD3<Float>) -> Float? {
        guard let arView else { return nil }
        let from = position + SIMD3<Float>(0, 0.25, 0)
        let to = position - SIMD3<Float>(0, 2.0, 0)
        return arView.scene.raycast(from: from, to: to, query: .nearest,
                                    mask: .sceneUnderstanding).first?.position.y
    }

    /// Arbitrary-direction raycast against the real LiDAR mesh, returning
    /// the full hit (position AND surface normal) — the ledge face
    /// confirmation in findDanglingEdge needs the normal, which the
    /// height-only helpers above throw away.
    private func meshRaycastHit(from: SIMD3<Float>, to: SIMD3<Float>) -> CollisionCastHit? {
        guard let arView else { return nil }
        return arView.scene.raycast(from: from, to: to, query: .nearest,
                                    mask: .sceneUnderstanding).first
    }

    /// Searches outward from Pipo's current position for the nearest edge
    /// suitable for the legs-dangling sit clip, using the same cascade
    /// real ledge-grab/climbing systems use: a coarse downward-sample scan
    /// only APPROXIMATES where the floor stops; the edge is then CONFIRMED
    /// by raycasting horizontally against the ledge's actual vertical face
    /// and resolved as the intersection of that face with the top surface.
    /// This is what the old sample-differencing version couldn't do —
    /// a LiDAR hole looks identical to a real drop from above, but a hole
    /// has no face to hit; and the face's surface normal gives the true
    /// outward direction, instead of whatever oblique angle the scan ray
    /// happened to approach the edge from (which had him sitting twisted
    /// relative to the lip). Returns the lip point (where his pelvis
    /// should end up) and the outward facing (the direction his legs
    /// dangle over), or nil if nothing qualifies within the search radius.
    private func findDanglingEdge() -> (position: SIMD3<Float>, facing: SIMD3<Float>)? {
        guard let pipo else { return nil }
        let origin = pipo.position(relativeTo: nil)

        clearEdgeDebugMarkers()
        guard let originGroundY = meshGroundHeight(at: origin) else { return nil }

        let searchRadius = characterHeight * 6
        let stepSize = characterHeight * 0.3
        let directionCount = 12
        // Only search the half facing whichever way Pipo's currently
        // facing — no reason to consider ledges behind him.
        let forward = pipo.orientation(relativeTo: nil).act(SIMD3<Float>(0, 0, 1))
        // The sit clip's legs hang roughly half a character-height below
        // the pelvis (baked into the origin shift done on export) — a
        // "step" shallower than this isn't a real dangling edge, since his
        // legs would just come to rest on whatever's immediately below
        // instead of clearing it and hanging in open air. Also protects
        // against small height noise in the LiDAR mesh being misread as an
        // edge, which was causing the search to stop almost immediately.
        let minimumDropDepth = characterHeight * 0.4
        // A rise taller than this is an obstacle, not a step Pipo could
        // actually walk up — searching wasn't checking for this, so a
        // direction that walked up onto something raised (e.g. climbing
        // onto a table's own surface) would keep going from THAT height
        // and could report a "candidate edge" measured above where he
        // started, which he could never actually walk to.
        let walkableRiseTolerance = characterHeight * 0.2

        var best: (position: SIMD3<Float>, facing: SIMD3<Float>, distance: Float)?

        for i in 0..<directionCount {
            let angle = Float(i) / Float(directionCount) * 2 * .pi
            let direction = SIMD3<Float>(cos(angle), 0, sin(angle))
            guard simd_dot(direction, forward) > 0 else { continue } // behind him — skip

            var lastSolidPoint = origin
            var lastSolidGroundY = originGroundY
            var distance = stepSize

            while distance <= searchRadius {
                let samplePoint = origin + direction * distance
                var sampleGroundY = meshGroundHeight(at: samplePoint)
                var usedEstimatedFallback = false

                if sampleGroundY == nil {
                    // No real-mesh hit — could be genuine open space (an
                    // edge) or just an area the LiDAR scan hasn't finished
                    // reconstructing yet. Cross-check ARKit's estimated-
                    // plane fallback: if it still finds a surface near the
                    // last known height, this is probably just unscanned
                    // rather than a real drop, so treat it as still-solid
                    // and keep going instead of reading it as an edge.
                    if let estimated = groundHeight(at: samplePoint),
                       abs(estimated - lastSolidGroundY) < minimumDropDepth {
                        sampleGroundY = estimated
                        usedEstimatedFallback = true
                    }
                }
                let drop = sampleGroundY.map { lastSolidGroundY - $0 } ?? Float.greatestFiniteMagnitude
                let rise = sampleGroundY.map { $0 - lastSolidGroundY } ?? 0

                if rise > walkableRiseTolerance {
                    // Too tall to just walk up (e.g. the side of a table) —
                    // a dead end for this direction. Don't continue "onto"
                    // it and search from up there; stop with no edge found.
                    addEdgeDebugMarker(at: samplePoint, color: .systemOrange,
                                       radius: characterHeight * 0.02)
                    break
                }

                if drop < minimumDropDepth {
                    // Same surface, a small step, or even a slight rise —
                    // not a real dangling edge. Keep probing outward from
                    // here, whatever height it settles at.
                    let markerY = sampleGroundY ?? samplePoint.y
                    addEdgeDebugMarker(at: SIMD3<Float>(samplePoint.x, markerY, samplePoint.z),
                                       color: usedEstimatedFallback ? .systemBlue : .systemGray,
                                       radius: characterHeight * 0.02)
                    if let sampleGroundY {
                        lastSolidPoint = samplePoint
                        lastSolidGroundY = sampleGroundY
                    }
                    distance += stepSize
                    continue
                }
                // A drop registered just before samplePoint.
                // Now CONFIRM it the way ledge-grab systems do, instead of
                // trusting the height differencing:
                //
                // Phase 2 — face confirmation at SEVERAL depths below the
                // top surface, shallowest first: a box or wall has a tall
                // vertical face (any depth hits), but a tabletop is a thin
                // slab with nothing but air beneath — its only vertical
                // geometry is the few-cm band of the slab's own side, so
                // only the probe just under the lip can catch it.
                let probeX = samplePoint.x + direction.x * stepSize
                let probeZ = samplePoint.z + direction.z * stepSize
                var faceHit: CollisionCastHit?
                for depthFactor: Float in [0.08, 0.25, 0.45] {
                    let start = SIMD3<Float>(probeX,
                                             lastSolidGroundY - characterHeight * depthFactor,
                                             probeZ)
                    if let hit = meshRaycastHit(from: start, to: start - direction * (stepSize * 4)) {
                        faceHit = hit
                        break
                    }
                }

                var facing = direction
                let lipX: Float
                let lipZ: Float
                let topY: Float

                if let faceHit {
                    // Face found. Its surface normal, flattened to the
                    // horizontal plane, is the true outward direction over
                    // the drop — flipped if the reconstructed triangle
                    // happened to wind the other way. The face X/Z plus the
                    // top surface just inside it are the exact edge.
                    var n = faceHit.normal
                    n.y = 0
                    let normalLength = simd_length(n)
                    if normalLength > 0.001 { facing = n / normalLength }
                    if simd_dot(facing, direction) < 0 { facing = -facing }
                    addEdgeDebugMarker(at: faceHit.position, color: .systemTeal,
                                       radius: characterHeight * 0.02)
                    let inset = characterHeight * 0.05
                    lipX = faceHit.position.x - facing.x * inset
                    lipZ = faceHit.position.z - facing.z * inset
                    guard let lipTopY = meshGroundHeight(at: SIMD3<Float>(lipX, lastSolidGroundY, lipZ)),
                          abs(lipTopY - lastSolidGroundY) < walkableRiseTolerance else {
                        break
                    }
                    topY = lipTopY
                } else if sampleGroundY != nil {
                    // No face at any depth, but the sample beyond the drop
                    // hit REAL mesh far below (e.g. the floor under a
                    // table) — that's a genuine edge whose vertical side
                    // just isn't reconstructed, not a scan hole. Fall back
                    // to refining the lip by binary search between the last
                    // solid sample and this one; facing stays the scan
                    // direction since there's no normal to read.
                    var solid = lastSolidPoint
                    var solidY = lastSolidGroundY
                    var far = samplePoint
                    for _ in 0..<6 {
                        let mid = (solid + far) / 2
                        if let midY = meshGroundHeight(at: mid),
                           solidY - midY < minimumDropDepth {
                            solid = mid
                            solidY = midY
                        } else {
                            far = mid
                        }
                    }
                    lipX = solid.x
                    lipZ = solid.z
                    topY = solidY
                    addEdgeDebugMarker(at: SIMD3<Float>(lipX, topY, lipZ), color: .systemIndigo,
                                       radius: characterHeight * 0.02)
                } else {
                    // No face AND no mesh below at all — indistinguishable
                    // from an unscanned hole; reject.
                    addEdgeDebugMarker(at: SIMD3<Float>(probeX, lastSolidGroundY, probeZ),
                                       color: .systemYellow,
                                       radius: characterHeight * 0.02)
                    break
                }

                let edgePosition = SIMD3<Float>(lipX, topY, lipZ)

                // Phase 4 — clearance: just outside the lip, where his
                // legs will hang, re-verify the drop really is deep enough.
                let clearInset = characterHeight * 0.15
                if let outsideY = meshGroundHeight(at: SIMD3<Float>(lipX + facing.x * clearInset,
                                                                    topY,
                                                                    lipZ + facing.z * clearInset)),
                   topY - outsideY < minimumDropDepth {
                    break
                }

                addEdgeDebugMarker(at: edgePosition, color: .systemRed, radius: characterHeight * 0.03)
                // Keep whichever confirmed edge is closest to Pipo's start
                // position across all directions.
                let edgeDistance = simd_length(SIMD3<Float>(edgePosition.x - origin.x, 0,
                                                            edgePosition.z - origin.z))
                if best == nil || edgeDistance < best!.distance {
                    best = (position: edgePosition, facing: facing, distance: edgeDistance)
                }
                break
            }
        }

        guard let found = best else { return nil }
        addEdgeDebugMarker(at: found.position, color: .systemGreen, radius: characterHeight * 0.05)
        addEdgeDebugLine(from: origin, to: found.position, color: .systemGreen)
        return (found.position, found.facing)
    }

    /// DEBUG: small sphere marker used by findDanglingEdge's visualization.
    /// Depth-testing disabled so it renders through occluding geometry
    /// (e.g. Pipo's own body) instead of being hidden inside/behind it.
    private func addEdgeDebugMarker(at position: SIMD3<Float>, color: UIColor, radius: Float) {
        guard let arView else { return }
        var material = UnlitMaterial(color: color)
        material.readsDepth = false
        material.writesDepth = false
        let marker = ModelEntity(mesh: .generateSphere(radius: max(radius, 0.001)),
                                 materials: [material])
        let anchor = AnchorEntity(world: position)
        anchor.addChild(marker)
        arView.scene.addAnchor(anchor)
        edgeDebugMarkers.append(anchor)
    }

    /// DEBUG: thin line from Pipo's start position to the winning edge, so
    /// the picked direction is visible at a glance.
    private func addEdgeDebugLine(from: SIMD3<Float>, to: SIMD3<Float>, color: UIColor) {
        guard let arView else { return }
        let delta = to - from
        let length = simd_length(delta)
        guard length > 0.001 else { return }
        var material = UnlitMaterial(color: color)
        material.readsDepth = false
        material.writesDepth = false
        let line = ModelEntity(mesh: .generateCylinder(height: length, radius: characterHeight * 0.006),
                               materials: [material])
        line.orientation = simd_quaternion([0, 1, 0], delta / length)
        let anchor = AnchorEntity(world: from + delta / 2)
        anchor.addChild(line)
        arView.scene.addAnchor(anchor)
        edgeDebugMarkers.append(anchor)
    }

    private func clearEdgeDebugMarkers() {
        for marker in edgeDebugMarkers {
            marker.removeFromParent()
        }
        edgeDebugMarkers.removeAll()
    }

    /// Sweeps Pipo's ACTUAL current footprint (the convex hull of his live
    /// joint positions, recomputed every frame by refreshFootprint — not a
    /// fixed circle) straight down and returns the height to stand
    /// on if any sample is close enough beneath to count as support, or nil
    /// if the whole footprint has walked past an edge. Using several
    /// samples across the real footprint, rather than a single center ray,
    /// is what makes this reliable without a debounce counter — a lone
    /// raycast can catch a stray hole in the LiDAR mesh, or miss a real
    /// edge because the footprint's center is still over solid ground
    /// while its rim already isn't; averaging that out is the same reason
    /// real character controllers sweep a shape instead of a ray.
    private func groundedHeight(at position: SIMD3<Float>) -> Float? {
        let offsets = footprintOffsets.isEmpty ? [.zero] : footprintOffsets

        var supportedHits: [Float] = []
        for offset in offsets {
            let samplePosition = SIMD3<Float>(position.x + offset.x, position.y, position.z + offset.y)
            if let hitY = meshGroundHeight(at: samplePosition), position.y - hitY < groundedTolerance {
                supportedHits.append(hitY)
            }
        }

        guard !supportedHits.isEmpty else { return nil }
        // Median rather than max: max matches a CharacterController's
        // step-offset behavior for a real small lip under one edge of the
        // footprint, but it also means a single bad sample — e.g. a
        // raycast near furniture (a sofa, a table) clipping the object's
        // own surface instead of the real floor, which LiDAR often can't
        // see under/behind occluded furniture in the first place — can
        // override every other correctly-grounded sample and drag his
        // whole height up to match that one outlier. Median stays robust
        // to one stray high (or low) reading as long as most of the
        // footprint agrees, at the cost of being slightly less eager about
        // stepping onto a real small lip that only one sample touches.
        let sorted = supportedHits.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
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
            // Play the impact animation fresh, right at the moment of
            // landing — landPlayback is untouched before this point (no
            // pre-created paused controller to stop first), since stopping
            // one controller and immediately creating another on the same
            // entity within a single call was a likely reason the clip
            // wasn't visually taking effect.
            if let clip = landClip, let owner = animationOwner {
                landPlayback = owner.playAnimation(clip, transitionDuration: 0, startsPaused: false)
            }
            print("PIPOFALLDBG impact: landPlayback=\(landPlayback != nil) valid=\(landPlayback?.isValid ?? false) time=\(landPlayback?.time ?? -1) landClipDuration=\(landClipDuration)")
            state = .landing(resumeTarget: resumeTarget)
            return
        }
        pipo.setPosition(position, relativeTo: nil)
    }

    private func updateLanding(pipo: Entity, resumeTarget: PathPoint) {
        guard let playback = landPlayback, playback.isValid,
              playback.time < landClipDuration - 0.05 else {
            print("PIPOFALLDBG bailing to walk: playback=\(landPlayback != nil) valid=\(landPlayback?.isValid ?? false) time=\(landPlayback?.time ?? -1) landClipDuration=\(landClipDuration)")
            landPlayback?.stop()
            landPlayback = nil
            state = .walking(target: resumeTarget)
            startWalkClipIfNeeded()
            return
        }
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

    private func startSitClipIfNeeded() {
        guard let clip = sitClip, let owner = animationOwner else {
            PipoLog.log("startSitClipIfNeeded: bail — clip=\(sitClip != nil) owner=\(animationOwner != nil)")
            return
        }
        if let playback = sitPlayback, playback.isValid {
            PipoLog.log("startSitClipIfNeeded: resuming existing playback, isPlaying=\(playback.isPlaying)")
            playback.speed = 1
            playback.resume()
        } else {
            PipoLog.log("startSitClipIfNeeded: starting fresh playback on owner=\(owner.name) clip.duration=\(clip.definition.duration)")
            sitPlayback = owner.playAnimation(clip.repeat(),
                                              transitionDuration: 0.25,
                                              startsPaused: false)
            PipoLog.log("startSitClipIfNeeded: sitPlayback created=\(sitPlayback != nil) isValid=\(sitPlayback?.isValid ?? false)")
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
