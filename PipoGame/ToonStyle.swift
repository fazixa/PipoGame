import RealityKit
import Metal
import UIKit

/// Toon look for Pipo: body materials become flat unlit color, and an
/// inverted-hull clone (normals-pushed shell, front faces culled) draws the
/// outline. PipoController keeps the clone's pose in sync (joint transforms
/// + blend shape weights copied every frame).
final class ToonStyle {

    /// Outline thickness in model units — a fixed fraction of Pipo's own
    /// size (he is ~4.6 units tall), so the line scales with him like part
    /// of the drawing and the art style stays put at any distance or pinch.
    private let lineWidth: Float = 0.026

    // RealityKit merges Pipo's meshes into ModelComponents with a material
    // slot per part — eyes/mouth are identified per SLOT by their dark tint
    // (body is pink, face features are near-black plum).
    private func isFaceSlot(_ material: any RealityKit.Material) -> Bool {
        let tint: UIColor
        if let pbr = material as? PhysicallyBasedMaterial {
            tint = pbr.baseColor.tint
        } else if let unlit = material as? UnlitMaterial {
            // PipoAsset.flattenFaceSlots already converts eyes/mouth to
            // Unlit at load time (outside toon mode too) — still need to
            // recognize them here by tint so they get the face material
            // instead of the body treatment.
            tint = unlit.color.tint
        } else {
            return false
        }
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        tint.getRed(&r, green: &g, blue: &b, alpha: &a)
        // 0.45, not 0.25: RealityKit reports tints sRGB-encoded, so the
        // Blender-exported linear face color (dark plum) reads as ~0.28
        // luminance — the old 0.25 threshold silently missed it (body
        // pink reads ~0.61, so 0.45 separates them comfortably).
        return (0.299 * r + 0.587 * g + 0.114 * b) < 0.45
    }

    private var swapped: [(Entity, [any RealityKit.Material])] = []
    private var outlineMaterial: CustomMaterial?
    private var faceMaterial: CustomMaterial?
    private(set) var outlineRoot: Entity?

    var isActive: Bool { outlineRoot != nil }

    /// Returns the outline clone so the controller can sync its pose.
    @discardableResult
    func apply(to pipo: Entity) -> Entity? {
        guard !isActive else { return outlineRoot }

        // 1. Flat unlit displaced body, preserving each slot's base color.
        //    Face slots get the face material (same displacement, no
        //    outline participation) instead. Face indices are recorded per
        //    model IN TRAVERSAL ORDER so step 2 can identify the same slots
        //    on the clone positionally — every slot is a CustomMaterial
        //    after this pass, so they can't be told apart by type there.
        let face = makeFaceMaterial()
        var faceIndicesPerModel: [[Int]] = []
        forEachModel(in: pipo) { entity, model in
            swapped.append((entity, model.materials))
            var faceIndices: [Int] = []
            model.materials = model.materials.enumerated().map { index, material in
                if self.isFaceSlot(material), let face {
                    faceIndices.append(index)
                    return face
                }
                if let pbr = material as? PhysicallyBasedMaterial {
                    return self.makeBodyMaterial(tint: pbr.baseColor.tint) ?? material
                }
                if let simple = material as? SimpleMaterial {
                    return self.makeBodyMaterial(tint: simple.color.tint) ?? material
                }
                if let unlit = material as? UnlitMaterial {
                    return self.makeBodyMaterial(tint: unlit.color.tint) ?? material
                }
                return material
            }
            faceIndicesPerModel.append(faceIndices)
        }

        // 2. Inverted-hull outline shell: body slots get the ink material,
        //    face slots become fully transparent (no ring around eyes/mouth).
        guard let material = makeOutlineMaterial() else { return nil }
        var invisible = UnlitMaterial(color: .clear)
        invisible.blending = .transparent(opacity: .init(scale: 0))
        let clone = pipo.clone(recursive: true)
        clone.transform = Transform.identity
        var modelIndex = 0
        var pureFaceClones: [Entity] = []
        forEachModel(in: clone) { entity, model in
            entity.components.remove(GroundingShadowComponent.self)
            let faceIndices = modelIndex < faceIndicesPerModel.count
                ? faceIndicesPerModel[modelIndex] : []
            modelIndex += 1
            // The hull is the BODY only: a model that is nothing but face
            // slots is an eye/mouth decal mesh — it contributes nothing to
            // the outline, and its inflated clone would ring the feature.
            if !faceIndices.isEmpty && faceIndices.count == model.materials.count {
                pureFaceClones.append(entity)
                return
            }
            model.materials = model.materials.enumerated().map { index, _ -> any RealityKit.Material in
                faceIndices.contains(index) ? invisible : material
            }
        }
        for entity in pureFaceClones {
            entity.removeFromParent()
        }
        pipo.addChild(clone)
        outlineRoot = clone
        return clone
    }

    func remove() {
        outlineRoot?.removeFromParent()
        outlineRoot = nil
        for (entity, materials) in swapped {
            if var model = entity.components[ModelComponent.self] {
                model.materials = materials
                entity.components.set(model)
            }
        }
        swapped = []
    }

    /// Flat unlit color + soft-noise displacement, tinted per slot.
    private func makeBodyMaterial(tint: UIColor) -> CustomMaterial? {
        guard let device = MTLCreateSystemDefaultDevice(),
              let library = device.makeDefaultLibrary() else { return nil }
        do {
            let surface = CustomMaterial.SurfaceShader(named: "pipoToonBodySurface", in: library)
            let geometry = CustomMaterial.GeometryModifier(named: "pipoToonBodyGeometry", in: library)
            var material = try CustomMaterial(surfaceShader: surface,
                                              geometryModifier: geometry,
                                              lightingModel: .unlit)
            material.baseColor = .init(tint: tint)
            material.custom.value = SIMD4<Float>(lineWidth, 0, 0, 0)
            return material
        } catch {
            print("toon body material creation failed: \(error)")
            return nil
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
            material.custom.value = SIMD4<Float>(0, 0, 0, 0)
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
            material.custom.value = SIMD4<Float>(lineWidth, 0, 0, 0)
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
