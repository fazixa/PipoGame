import RealityKit
import Metal

/// Textured-skin look for Pipo: a fingerprint-derived normal map and
/// roughness map on his PBR skin material, plus a subtle cloud-noise vertex
/// displacement for larger-scale surface undulation. Bump (normal/roughness)
/// handles fine detail without moving geometry; displacement handles the
/// larger shapes — see Shaders.metal's pipoNoiseDisplaceGeometry. Only
/// touches PhysicallyBasedMaterial slots, so eyes/mouth (Unlit) are
/// untouched, same as ToonStyle's face-slot handling.
final class BumpyStyle {
    private let noiseScale: Float = 6.0
    private let noiseStrength: Float = 0.015

    private var swapped: [(Entity, [any RealityKit.Material])] = []
    private var geometryModifier: CustomMaterial.GeometryModifier?
    private var normalTexture: TextureResource?
    private var roughnessTexture: TextureResource?

    var isActive: Bool { !swapped.isEmpty }

    func apply(to pipo: Entity) {
        guard !isActive else { return }
        guard let modifier = makeGeometryModifier(),
              let normal = loadTexture("PipoBumpNormal"),
              let roughness = loadTexture("PipoBumpRoughness") else { return }

        forEachModel(in: pipo) { entity, model in
            swapped.append((entity, model.materials))
            model.materials = model.materials.map { material in
                guard var pbr = material as? PhysicallyBasedMaterial else { return material }
                pbr.normal = .init(texture: .init(normal))
                pbr.roughness.texture = .init(roughness)
                guard var custom = try? CustomMaterial(from: pbr, geometryModifier: modifier) else {
                    return pbr
                }
                custom.custom.value = SIMD4<Float>(noiseScale, noiseStrength, 0, 0)
                return custom
            }
        }
    }

    func remove() {
        for (entity, materials) in swapped {
            if var model = entity.components[ModelComponent.self] {
                model.materials = materials
                entity.components.set(model)
            }
        }
        swapped = []
    }

    private func makeGeometryModifier() -> CustomMaterial.GeometryModifier? {
        if let geometryModifier { return geometryModifier }
        guard let device = MTLCreateSystemDefaultDevice(),
              let library = device.makeDefaultLibrary() else { return nil }
        let modifier = CustomMaterial.GeometryModifier(named: "pipoNoiseDisplaceGeometry", in: library)
        geometryModifier = modifier
        return modifier
    }

    private func loadTexture(_ name: String) -> TextureResource? {
        if name == "PipoBumpNormal", let normalTexture { return normalTexture }
        if name == "PipoBumpRoughness", let roughnessTexture { return roughnessTexture }
        guard let texture = try? TextureResource.load(named: name) else {
            print("BumpyStyle: failed to load texture \(name)")
            return nil
        }
        if name == "PipoBumpNormal" { normalTexture = texture } else { roughnessTexture = texture }
        return texture
    }

    private func forEachModel(in entity: Entity, _ body: (Entity, inout ModelComponent) -> Void) {
        if var model = entity.components[ModelComponent.self] {
            body(entity, &model)
            entity.components.set(model)
        }
        for child in entity.children {
            forEachModel(in: child, body)
        }
    }
}
