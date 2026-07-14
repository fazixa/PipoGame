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
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        arView.session.run(config)
        arView.session.delegate = context.coordinator.handTracker
        controller.handTracker = context.coordinator.handTracker

        arView.environment.sceneUnderstanding.options.insert(.occlusion)
        arView.environment.sceneUnderstanding.options.insert(.receivesLighting)
        // ARKit's estimated room lighting is often conservative, so
        // PhysicallyBasedMaterial-lit virtual objects (Pipo's skin) can
        // read darker than intended. Boost how strongly they respond to it.
        arView.environment.lighting.intensityExponent = 1.8

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
        pinch.delegate = context.coordinator
        arView.addGestureRecognizer(pinch)

        let rotation = UIRotationGestureRecognizer(target: context.coordinator,
                                                    action: #selector(Coordinator.handleRotation(_:)))
        rotation.delegate = context.coordinator
        arView.addGestureRecognizer(rotation)

        let pan = UIPanGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.handlePan(_:)))
        pan.maximumNumberOfTouches = 1
        arView.addGestureRecognizer(pan)

        controller.arView = arView
        context.coordinator.updateSubscription = arView.scene.subscribe(to: SceneEvents.Update.self) { [weak controller, weak arView] event in
            controller?.update(deltaTime: Float(event.deltaTime))
            if let arView, let frame = arView.session.currentFrame {
                controller?.geospatial.update(frame: frame, in: arView)
            }
        }

        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(controller: controller)
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        let controller: PipoController
        let handTracker = HandTracker()
        var updateSubscription: Cancellable?

        init(controller: PipoController) {
            self.controller = controller
            super.init()
        }

        @objc func handleTap(_ recognizer: UITapGestureRecognizer) {
            guard let arView = controller.arView else { return }
            let point = recognizer.location(in: arView)
            if controller.isGeoActive {
                // Geospatial mode: taps target real buildings meters/blocks
                // away, which ARKit's own near-field plane raycast (below)
                // has no way to hit — route through ARCore's Streetscape
                // Geometry raycast instead.
                controller.placeGeospatial(from: arView, at: point)
                return
            }
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

        @objc func handleRotation(_ recognizer: UIRotationGestureRecognizer) {
            guard recognizer.state == .changed else { return }
            controller.rotate(by: Float(recognizer.rotation))
            recognizer.rotation = 0
        }

        private var isDraggingPipo = false

        @objc func handlePan(_ recognizer: UIPanGestureRecognizer) {
            guard let arView = controller.arView, let view = recognizer.view else { return }
            switch recognizer.state {
            case .began:
                isDraggingPipo = controller.beginDrag(at: recognizer.location(in: arView))
            case .changed:
                guard isDraggingPipo else { return }
                controller.dragBy(recognizer.translation(in: view), at: recognizer.location(in: arView))
                recognizer.setTranslation(.zero, in: view)
            case .ended, .cancelled, .failed:
                if isDraggingPipo {
                    controller.endDrag()
                    isDraggingPipo = false
                }
            default:
                break
            }
        }

        // Let two-finger pinch and rotate run together instead of one
        // gesture stealing the touches from the other.
        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                               shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
            true
        }
    }
}
