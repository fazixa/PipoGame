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
        // Real Pipo: ARP rig exported to a deform-only joint skeleton
        // (tools/blender_export.py + fix_usdz_for_realitykit.py) with the
        // walk cycle baked onto the joints and knee-corrective blend shapes
        // riding along. Land/climb/sit clips were authored on the Mixamo
        // test skeleton and can't play on this rig — they stay nil until
        // re-authored on Pipo's rig; the controller falls back gracefully
        // (glide-climb, no land clip, sit disabled via supportsSit).
        guard let root = try? Entity.load(named: "Pipo") else { return nil }
        root.scale = SIMD3<Float>(repeating: worldScale)

        applyGroundingShadow(root)
        flattenFaceSlots(root)

        guard let owner = firstEntityWithAnimations(in: root),
              let clip = owner.availableAnimations.first else {
            return nil
        }

        // PipoSit.usdz / PipoLand.usdz: clips exported off the same rig
        // (v32 sit loop, v31 land one-shot), so they play directly on the
        // walk entity's skeleton. Loaded to pull the AnimationResource out;
        // their entity hierarchies are never added to the scene but must
        // stay retained (sitEntity/landEntity) or the clip silently dies
        // on a second load.
        let sitEntity = try? Entity.load(named: "PipoSit")
        let sitClip = sitEntity.flatMap(firstEntityWithAnimations)?.availableAnimations.first

        let landEntity = try? Entity.load(named: "PipoLand")
        let landClip = landEntity.flatMap(firstEntityWithAnimations)?.availableAnimations.first

        return LoadedPipo(root: root, animationOwner: owner, walkClip: clip,
                          landClip: landClip, landEntity: landEntity,
                          climbClip: nil, climbEntity: nil,
                          sitClip: sitClip, sitEntity: sitEntity)
    }

    /// Eyes/mouth (identified by their dark near-black tint) read as flat
    /// unlit color rather than catching PBR specular/shadowing — small,
    /// near-black features where PBR shading mostly just adds noise
    /// instead of readable shape.
    private static func flattenFaceSlots(_ entity: Entity) {
        if var model = entity.components[ModelComponent.self] {
            model.materials = model.materials.map { material in
                guard let pbr = material as? PhysicallyBasedMaterial else { return material }
                var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
                pbr.baseColor.tint.getRed(&r, green: &g, blue: &b, alpha: &a)
                guard (0.299 * r + 0.587 * g + 0.114 * b) < 0.25 else { return material }
                return UnlitMaterial(color: pbr.baseColor.tint)
            }
            entity.components.set(model)
        }
        for child in entity.children {
            flattenFaceSlots(child)
        }
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
