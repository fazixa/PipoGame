import RealityKit
import Metal
import UIKit

/// Toon look for Pipo: body materials become flat unlit color, and an
/// inverted-hull clone (normals-pushed shell, front faces culled) draws the
/// outline. The clone plays the same clips as the body; PipoController keeps
/// them in sync.
final class ToonStyle {

    /// Outline thickness in model units — a fixed fraction of Pipo's own
    /// size (he is ~5.4 units tall), so the line scales with him like part
    /// of the drawing and the art style stays put at any distance or pinch.
    private let lineWidth: Float = 0.012

    // RealityKit merges Pipo's three meshes into ONE ModelComponent with a
    // material slot per part, so eyes/mouth cannot be excluded by removing
    // entities — they are identified per SLOT by their dark tint (body is
    // pink, face features are near-black plum).
    private func isFaceSlot(_ material: any RealityKit.Material) -> Bool {
        let tint: UIColor
        if let pbr = material as? PhysicallyBasedMaterial {
            tint = pbr.baseColor.tint
        } else if let unlit = material as? UnlitMaterial {
            // PipoAsset.flattenFaceSlots already converts eyes/mouth to
            // Unlit at load time (outside toon mode too) — still need to
            // recognize them here by tint so they get the face-pull
            // treatment instead of being swept into the outline hull as a
            // body slot (which would wrap an ink ring around them).
            tint = unlit.color.tint
        } else {
            return false
        }
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        tint.getRed(&r, green: &g, blue: &b, alpha: &a)
        return (0.299 * r + 0.587 * g + 0.114 * b) < 0.25
    }

    private var swapped: [(Entity, [any RealityKit.Material])] = []
    /// Entities + slot indices the per-frame update must touch.
    private var outlineSlots: [(Entity, [Int])] = []
    private var faceSlots: [(Entity, [Int])] = []
    private var outlineMaterial: CustomMaterial?
    private var faceMaterial: CustomMaterial?
    private(set) var outlineRoot: Entity?

    var isActive: Bool { outlineRoot != nil }

    /// Returns the outline clone so the controller can sync its animation.
    @discardableResult
    func apply(to pipo: Entity) -> Entity? {
        guard !isActive else { return outlineRoot }

        // 1. Flat unlit body, preserving each slot's base color. Face slots
        //    get the depth-pulled material instead (see Shaders.metal).
        let facePull = makeFaceMaterial()
        forEachModel(in: pipo) { entity, model in
            swapped.append((entity, model.materials))
            var faceIndices: [Int] = []
            model.materials = model.materials.enumerated().map { index, material in
                if self.isFaceSlot(material), let facePull {
                    faceIndices.append(index)
                    return facePull
                }
                if let pbr = material as? PhysicallyBasedMaterial {
                    return UnlitMaterial(color: pbr.baseColor.tint)
                }
                if let simple = material as? SimpleMaterial {
                    return UnlitMaterial(color: simple.color.tint)
                }
                return material
            }
            if !faceIndices.isEmpty {
                faceSlots.append((entity, faceIndices))
            }
        }

        // 2. Inverted-hull outline shell: body slots get the ink material,
        //    face slots become fully transparent (no ring around eyes/mouth).
        guard let material = makeOutlineMaterial() else { return nil }
        var invisible = UnlitMaterial(color: .clear)
        invisible.blending = .transparent(opacity: .init(scale: 0))
        let clone = pipo.clone(recursive: true)
        clone.transform = Transform.identity
        forEachModel(in: clone) { entity, model in
            entity.components.remove(GroundingShadowComponent.self)
            var bodyIndices: [Int] = []
            // Note: the clone's slots were already swapped in step 1, so face
            // slots are recognized by holding the facePull CustomMaterial.
            model.materials = model.materials.enumerated().map { index, slot in
                if slot is CustomMaterial {
                    return invisible
                }
                bodyIndices.append(index)
                return material
            }
            if !bodyIndices.isEmpty {
                outlineSlots.append((entity, bodyIndices))
            }
        }
        pipo.addChild(clone)
        outlineRoot = clone
        return clone
    }

    func remove() {
        outlineRoot?.removeFromParent()
        outlineRoot = nil
        outlineSlots = []
        faceSlots = []
        for (entity, materials) in swapped {
            if var model = entity.components[ModelComponent.self] {
                model.materials = materials
                entity.components.set(model)
            }
        }
        swapped = []
    }

    /// Per frame: width is a model-space constant, so this only refreshes
    /// the camera position the shaders use for their depth pulls. Only the
    /// tracked slots are touched.
    func updateThickness(cameraWorldPosition: SIMD3<Float>, worldScale: Float) {
        guard isActive else { return }
        if var material = outlineMaterial {
            apply(&material, to: outlineSlots, cameraWorldPosition: cameraWorldPosition)
            outlineMaterial = material
        }
        if var material = faceMaterial {
            apply(&material, to: faceSlots, cameraWorldPosition: cameraWorldPosition)
            faceMaterial = material
        }
    }

    private func apply(_ material: inout CustomMaterial, to slots: [(Entity, [Int])],
                       cameraWorldPosition: SIMD3<Float>) {
        for (entity, indices) in slots {
            let camModel = entity.convert(position: cameraWorldPosition, from: nil)
            material.custom.value = SIMD4<Float>(lineWidth,
                                                 camModel.x, camModel.y, camModel.z)
            if var model = entity.components[ModelComponent.self] {
                for index in indices where index < model.materials.count {
                    model.materials[index] = material
                }
                entity.components.set(model)
            }
        }
    }

    private func makeFaceMaterial() -> CustomMaterial? {
        if let cached = faceMaterial { return cached }
        guard let device = MTLCreateSystemDefaultDevice(),
              let library = device.makeDefaultLibrary() else { return nil }
        do {
            let surface = CustomMaterial.SurfaceShader(named: "pipoFaceSurface", in: library)
            let geometry = CustomMaterial.GeometryModifier(named: "pipoFacePullGeometry", in: library)
            var material = try CustomMaterial(surfaceShader: surface,
                                              geometryModifier: geometry,
                                              lightingModel: .unlit)
            material.custom.value = SIMD4<Float>(0.02, 0, 0, 0)
            faceMaterial = material
            return material
        } catch {
            print("face material creation failed: \(error)")
            return nil
        }
    }

    private func makeOutlineMaterial() -> CustomMaterial? {
        if let cached = outlineMaterial { return cached }
        guard let device = MTLCreateSystemDefaultDevice(),
              let library = device.makeDefaultLibrary() else { return nil }
        do {
            let surface = CustomMaterial.SurfaceShader(named: "pipoOutlineSurface", in: library)
            let geometry = CustomMaterial.GeometryModifier(named: "pipoOutlineGeometry", in: library)
            var material = try CustomMaterial(surfaceShader: surface,
                                              geometryModifier: geometry,
                                              lightingModel: .unlit)
            material.faceCulling = .front
            material.custom.value = SIMD4<Float>(0.02, 0, 0, 0)
            // Baseline multiplier of 1; the shader's set_opacity() (taper
            // weight) supplies the actual per-fragment value, fading the ink
            // to fully transparent in step with the geometry push near a
            // hull-taper seam (see pipoOutlineSurface).
            material.blending = .transparent(opacity: .init(scale: 1))
            outlineMaterial = material
            return material
        } catch {
            print("outline material creation failed: \(error)")
            return nil
        }
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
