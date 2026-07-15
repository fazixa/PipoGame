import RealityKit
import Foundation

/// Loads the real rigged Pipo from Pipo.usdz (exported from Blender with the
/// walk cycle baked in). Falls back to nil if the asset is missing or broken;
/// the caller then uses the primitive PipoBuilder placeholder.
enum PipoAsset {

    /// Model is ~4.5 m tall in the USDZ; this brings him to roughly 15 cm.
    static let worldScale: Float = 0.034

    struct LoadedPipo {
        let root: Entity
        /// The entity that owns the animation clips (usually a child SkelRoot).
        let animationOwner: Entity
        let walkClip: AnimationResource
        /// TEMP: hard-landing clip, played on impact after a fall. Loaded
        /// from a separate USDZ (LandTest.usdz) sharing the same Mixamo
        /// joint names/hierarchy as WalkTest, so its AnimationResource can
        /// be played directly on the walk entity's skeleton.
        let landClip: AnimationResource?
        /// TEMP: keeps the LandTest entity that landClip came from alive.
        /// Never added to the scene — extracting the clip and letting this
        /// go out of scope was unreliable on a second load (worked once,
        /// then landClip silently came back functional-but-dead after a
        /// reset+reload). Held here so it stays retained for as long as
        /// the character using its clip does.
        let landEntity: Entity?
        /// TEMP: climb clip, played while scaling a wall marked with a red
        /// (vertical) path point. Same loading pattern as landClip.
        let climbClip: AnimationResource?
        /// TEMP: keeps the ClimbTest entity that climbClip came from alive —
        /// see landEntity's doc comment for why this matters.
        let climbEntity: Entity?
        /// TEMP: sit clip. Same loading pattern as landClip/climbClip. Its
        /// export shifted the whole rig so the seated butt height — not the
        /// standard Mixamo feet-at-origin convention — is the local origin,
        /// so placing this entity puts his butt on the tapped surface.
        let sitClip: AnimationResource?
        /// TEMP: keeps the SittingTest entity that sitClip came from alive —
        /// see landEntity's doc comment for why this matters.
        let sitEntity: Entity?
    }

    static func load() -> LoadedPipo? {
        // TEMP: loading WalkTest.usdz (Mixamo rig prototype) instead of Pipo
        // to sanity-check the joint-skeleton export pipeline on this clean,
        // pre-regression branch. Revert to "Pipo" / worldScale afterward.
        guard let root = try? Entity.load(named: "WalkTest") else { return nil }
        root.scale = SIMD3<Float>(repeating: 0.015)

        applyGroundingShadow(root)

        guard let owner = firstEntityWithAnimations(in: root),
              let clip = owner.availableAnimations.first else {
            return nil
        }

        // TEMP: LandTest.usdz is loaded to pull its AnimationResource out —
        // its own entity hierarchy is never added to the scene, but IS kept
        // alive via LoadedPipo.landEntity (see its doc comment for why).
        // Same skeleton/joint names as WalkTest, so the clip plays fine on
        // the walk entity directly.
        let landEntity = try? Entity.load(named: "LandTest")
        print("DEBUG landEntity loaded:", landEntity != nil)
        let landOwner = landEntity.flatMap(firstEntityWithAnimations)
        print("DEBUG landOwner found:", landOwner != nil,
              "availableAnimations:", landOwner?.availableAnimations.count ?? -1)
        let landClip = landOwner?.availableAnimations.first
        print("DEBUG landClip extracted:", landClip != nil,
              "duration:", landClip?.definition.duration ?? -1)

        // TEMP: same loading pattern as landClip — ClimbTest.usdz shares the
        // WalkTest skeleton, so its clip plays directly on the walk entity.
        let climbEntity = try? Entity.load(named: "ClimbTest")
        let climbClip = climbEntity.flatMap(firstEntityWithAnimations)?.availableAnimations.first

        let sitEntity = try? Entity.load(named: "SittingTest")
        let sitClip = sitEntity.flatMap(firstEntityWithAnimations)?.availableAnimations.first

        return LoadedPipo(root: root, animationOwner: owner, walkClip: clip,
                          landClip: landClip, landEntity: landEntity,
                          climbClip: climbClip, climbEntity: climbEntity,
                          sitClip: sitClip, sitEntity: sitEntity)
    }

    private static func firstEntityWithAnimations(in entity: Entity) -> Entity? {
        if !entity.availableAnimations.isEmpty { return entity }
        for child in entity.children {
            if let found = firstEntityWithAnimations(in: child) { return found }
        }
        return nil
    }

    private static func applyGroundingShadow(_ entity: Entity) {
        if entity is ModelEntity || entity.components.has(ModelComponent.self) {
            entity.components.set(GroundingShadowComponent(castsShadow: true))
        }
        for child in entity.children {
            applyGroundingShadow(child)
        }
    }
}
