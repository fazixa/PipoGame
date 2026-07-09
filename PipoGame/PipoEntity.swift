import RealityKit
import UIKit

/// Builds the placeholder Pipo: a small primitive-based robot (~15 cm tall).
/// Named parts ("bodyPivot", "head", "footL", "footR") are driven procedurally
/// by PipoController. The real rigged Pipo will replace this via USDZ later.
enum PipoBuilder {

    static let footRestY: Float = 0.012
    static let bodyRestY: Float = 0.058
    static let bodySitY: Float = 0.034

    static func build() -> Entity {
        let root = Entity()
        root.name = "pipo"

        let bodyColor = UIColor(red: 0.98, green: 0.93, blue: 0.85, alpha: 1)
        let darkColor = UIColor(red: 0.15, green: 0.13, blue: 0.12, alpha: 1)
        let accentColor = UIColor(red: 0.95, green: 0.55, blue: 0.25, alpha: 1)

        let bodyMaterial = SimpleMaterial(color: bodyColor, roughness: 0.6, isMetallic: false)
        let darkMaterial = SimpleMaterial(color: darkColor, roughness: 0.4, isMetallic: false)
        let accentMaterial = SimpleMaterial(color: accentColor, roughness: 0.6, isMetallic: false)

        // Body pivot bobs up and down; body mesh, head and eyes ride along.
        let bodyPivot = Entity()
        bodyPivot.name = "bodyPivot"
        bodyPivot.position = [0, bodyRestY, 0]
        root.addChild(bodyPivot)

        let body = ModelEntity(
            mesh: .generateBox(size: [0.07, 0.08, 0.055], cornerRadius: 0.022),
            materials: [bodyMaterial]
        )
        body.name = "body"
        bodyPivot.addChild(body)

        let head = ModelEntity(
            mesh: .generateSphere(radius: 0.034),
            materials: [bodyMaterial]
        )
        head.name = "head"
        head.position = [0, 0.062, 0]
        bodyPivot.addChild(head)

        // Eyes face +Z; locomotion treats +Z as forward.
        for x in [Float(-0.013), 0.013] {
            let eye = ModelEntity(
                mesh: .generateSphere(radius: 0.0045),
                materials: [darkMaterial]
            )
            eye.position = [x, 0.008, 0.031]
            head.addChild(eye)
        }

        let belly = ModelEntity(
            mesh: .generateSphere(radius: 0.012),
            materials: [accentMaterial]
        )
        belly.position = [0, -0.005, 0.026]
        bodyPivot.addChild(belly)

        for (name, x) in [("footL", Float(-0.022)), ("footR", Float(0.022))] {
            let foot = ModelEntity(
                mesh: .generateSphere(radius: 0.013),
                materials: [darkMaterial]
            )
            foot.name = name
            foot.position = [x, footRestY, 0]
            root.addChild(foot)
        }

        for case let model as ModelEntity in allDescendants(of: root) {
            model.components.set(GroundingShadowComponent(castsShadow: true))
        }

        return root
    }

    private static func allDescendants(of entity: Entity) -> [Entity] {
        entity.children.flatMap { [$0] + allDescendants(of: $0) }
    }
}
