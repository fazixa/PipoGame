import SwiftUI
import RealityKit
import ARKit
import Combine

struct ARViewContainer: UIViewRepresentable {
    @ObservedObject var controller: PipoController

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)

        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        config.environmentTexturing = .automatic
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            config.frameSemantics.insert(.personSegmentationWithDepth)
        }
        arView.session.run(config)

        arView.environment.sceneUnderstanding.options.insert(.occlusion)
        arView.environment.sceneUnderstanding.options.insert(.receivesLighting)
        // Lets groundHeight() raycast against the actual reconstructed LiDAR
        // mesh instead of ARKit's simplified/extrapolated estimated-plane
        // model, which was bleeding past real edges (e.g. a table boundary).
        arView.environment.sceneUnderstanding.options.insert(.collision)

        // DEBUG: visualize the LiDAR scene reconstruction mesh (wireframe
        // over real-world geometry) so surfaces/heights ARKit actually
        // detected are visible. Remove once done checking table/floor
        // detection for the walk-off-edge falling behavior.
        arView.debugOptions.insert(.showSceneUnderstanding)

        let coaching = ARCoachingOverlayView()
        coaching.session = arView.session
        coaching.goal = .anyPlane
        coaching.frame = arView.bounds
        coaching.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        arView.addSubview(coaching)

        let tap = UITapGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.handleTap(_:)))
        arView.addGestureRecognizer(tap)

        let longPress = UILongPressGestureRecognizer(target: context.coordinator,
                                                     action: #selector(Coordinator.handleLongPress(_:)))
        longPress.minimumPressDuration = 0.6
        arView.addGestureRecognizer(longPress)

        let pinch = UIPinchGestureRecognizer(target: context.coordinator,
                                             action: #selector(Coordinator.handlePinch(_:)))
        arView.addGestureRecognizer(pinch)

        controller.arView = arView
        context.coordinator.updateSubscription = arView.scene.subscribe(to: SceneEvents.Update.self) { [weak controller] event in
            controller?.update(deltaTime: Float(event.deltaTime))
        }

        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(controller: controller)
    }

    final class Coordinator {
        let controller: PipoController
        var updateSubscription: Cancellable?

        init(controller: PipoController) {
            self.controller = controller
        }

        @objc func handleTap(_ recognizer: UITapGestureRecognizer) {
            guard let arView = controller.arView else { return }
            let point = recognizer.location(in: arView)
            guard let result = arView.raycast(from: point,
                                              allowing: .estimatedPlane,
                                              alignment: .any).first else { return }
            controller.handleTap(on: result)
        }

        @objc func handleLongPress(_ recognizer: UILongPressGestureRecognizer) {
            guard recognizer.state == .began else { return }
            controller.onLongPress?()
        }

        @objc func handlePinch(_ recognizer: UIPinchGestureRecognizer) {
            guard recognizer.state == .changed else { return }
            controller.pinch(by: Float(recognizer.scale))
            recognizer.scale = 1
        }
    }
}
