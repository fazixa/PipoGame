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

    private static func applyGroundingShadow(_ entity: Entity) {
        if entity is ModelEntity || entity.components.has(ModelComponent.self) {
            entity.components.set(GroundingShadowComponent(castsShadow: true))
        }
        for child in entity.children {
            applyGroundingShadow(child)
        }
    }
}
