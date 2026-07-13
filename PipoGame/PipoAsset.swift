import RealityKit
import Foundation

/// Loads the real rigged Pipo from Pipo.usdz (exported from Blender with the
/// walk cycle baked in). Falls back to nil if the asset is missing or broken;
/// the caller then uses the primitive PipoBuilder placeholder.
enum PipoAsset {

    /// Model is ~4.5 m tall in the USDZ; this brings him to roughly 15 cm.
    static let worldScale: Float = 0.034

    /// What the single baked clip inside Pipo.usdz is. The current model
    /// ships with only a sit animation (a head-sway loop); flip this back
    /// to .walk (or extend to multiple files) when more clips are authored.
    enum BundledClip { case walk, sit }
    static let bundledClip: BundledClip = .sit

    struct LoadedPipo {
        let root: Entity
        /// The entity that owns the animation clips (usually a child SkelRoot).
        let animationOwner: Entity
        let walkClip: AnimationResource?
        let sitClip: AnimationResource?
    }

    static func load() -> LoadedPipo? {
        guard let root = try? Entity.load(named: "Pipo") else { return nil }
        root.scale = SIMD3<Float>(repeating: worldScale)

        applyGroundingShadow(root)
        flattenFaceSlots(root)

        guard let owner = firstEntityWithAnimations(in: root),
              let clip = owner.availableAnimations.first else {
            return nil
        }
        return LoadedPipo(root: root,
                          animationOwner: owner,
                          walkClip: bundledClip == .walk ? clip : nil,
                          sitClip: bundledClip == .sit ? clip : nil)
    }

    private static func firstEntityWithAnimations(in entity: Entity) -> Entity? {
        if !entity.availableAnimations.isEmpty { return entity }
        for child in entity.children {
            if let found = firstEntityWithAnimations(in: child) { return found }
        }
        return nil
    }

    /// Eyes/mouth (identified by their dark near-black tint, same check
    /// ToonStyle uses) read as flat unlit color rather than catching PBR
    /// specular/shadowing — small, near-black features where PBR shading
    /// mostly just adds noise instead of readable shape.
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

    private static func applyGroundingShadow(_ entity: Entity) {
        if entity is ModelEntity || entity.components.has(ModelComponent.self) {
            entity.components.set(GroundingShadowComponent(castsShadow: true))
        }
        for child in entity.children {
            applyGroundingShadow(child)
        }
    }
}
