import SwiftUI

struct ContentView: View {
    @StateObject private var controller = PipoController()
    @StateObject private var rfnnController = PFNNController()
    @StateObject private var testRig = PFNNTestRig()
    @StateObject private var recorder = ScreenRecorder()
    @State private var uiVisible = true

    var body: some View {
        ZStack {
            ARViewContainer(controller: controller, rfnnController: rfnnController, testRig: testRig)
                .ignoresSafeArea()

            // Game mode: floating joystick over the lower-left region,
            // kept clear of the control bar below. Sits UNDER the control
            // VStack in the ZStack so the buttons stay tappable.
            if controller.isGameMode && uiVisible {
                VStack {
                    Spacer()
                    HStack {
                        JoystickOverlay { controller.joystickInput = $0 }
                            .frame(width: 240, height: 280)
                        Spacer()
                    }
                }
                .padding(.leading, 8)
                .padding(.bottom, 140)
            }

            // TEMP: PFNN puppet spike -- own joystick, independent of Pipo's
            // own game mode, so the two characters can be tested/compared
            // side by side.
            if rfnnController.isPlaced && uiVisible {
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        JoystickOverlay { rfnnController.joystickInput = $0 }
                            .frame(width: 240, height: 280)
                    }
                }
                .padding(.trailing, 8)
                .padding(.bottom, 140)
            }

            if uiVisible {
                VStack {
                    Spacer()

                    Text(hint)
                        .font(.footnote)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.bottom, 8)

                    HStack(spacing: 12) {
                        Button {
                            controller.toggleSit()
                        } label: {
                            Image(systemName: controller.isSitting ? "figure.stand" : "chair")
                                .frame(maxWidth: .infinity)
                        }
                        .disabled(!controller.isPlaced || !controller.supportsSit || controller.isDrawingPath || controller.isFreehand)

                        // TEMP: trajectory-drawing prototype
                        Button {
                            controller.toggleDrawPath()
                        } label: {
                            Image(systemName: controller.isDrawingPath ? "checkmark.circle.fill" : "scribble")
                                .frame(maxWidth: .infinity)
                        }
                        .disabled(!controller.isPlaced || controller.isSitting || controller.isFreehand || controller.isGameMode)

                        // TEMP: freehand-move prototype
                        Button {
                            controller.toggleFreehand()
                        } label: {
                            Image(systemName: controller.isFreehand ? "checkmark.circle.fill" : "move.3d")
                                .frame(maxWidth: .infinity)
                        }
                        .disabled(!controller.isPlaced || controller.isDrawingPath || controller.isGameMode)

                        Button {
                            controller.toggleToon()
                        } label: {
                            Image(systemName: controller.isToon ? "paintbrush.fill" : "paintbrush")
                                .frame(maxWidth: .infinity)
                        }
                        .disabled(!controller.isPlaced || !controller.supportsToon)

                        Button {
                            controller.toggleGameMode()
                        } label: {
                            Image(systemName: controller.isGameMode ? "gamecontroller.fill" : "gamecontroller")
                                .frame(maxWidth: .infinity)
                        }
                        .disabled(!controller.isPlaced || controller.isDrawingPath || controller.isFreehand)

                        // TEMP: PFNN puppet spike -- own rig/mesh, no
                        // retargeting onto Pipo, just to test the ported
                        // network directly.
                        Button {
                            rfnnController.toggle()
                        } label: {
                            Image(systemName: rfnnController.isPlaced ? "figure.walk.circle.fill" : "figure.walk.circle")
                                .frame(maxWidth: .infinity)
                        }

                        // TEMP: minimal 2-joint test rig, isolating
                        // RealityKit's exact jointTransforms convention.
                        Button {
                            testRig.toggle()
                        } label: {
                            Image(systemName: testRig.isPlaced ? "checkmark.circle.fill" : "figure.stand")
                                .frame(maxWidth: .infinity)
                        }

                        Button {
                            startRecording()
                        } label: {
                            Image(systemName: "record.circle.fill")
                                .foregroundStyle(.red)
                                .frame(maxWidth: .infinity)
                        }

                        Button(role: .destructive) {
                            controller.reset()
                        } label: {
                            Image(systemName: "arrow.counterclockwise")
                                .frame(maxWidth: .infinity)
                        }
                        .disabled(!controller.isPlaced)
                    }
                    .buttonStyle(.bordered)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                    .padding(.horizontal, 24)

                    Text("Record hides the UI • long-press the screen to stop & save")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .padding(.top, 6)
                        .padding(.bottom, 12)
                }
                .transition(.opacity)
            }

            if let toast = recorder.toast {
                VStack {
                    Text(toast)
                        .font(.footnote.weight(.medium))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.top, 8)
                    Spacer()
                }
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: uiVisible)
        .animation(.easeInOut(duration: 0.25), value: recorder.toast)
        .onAppear {
            controller.onLongPress = { [weak recorder] in
                if let recorder, recorder.isRecording {
                    recorder.stop()
                }
                // Also restores the UI if recording failed to start.
                uiVisible = true
            }
        }
    }

    private func startRecording() {
        uiVisible = false
        // Let the UI fade out fully before capture begins so it never
        // appears in the footage.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            recorder.start()
        }
    }

    /// Floating virtual joystick: the base circle appears wherever the
    /// touch lands inside this view's frame and follows classic virtual-
    /// gamepad rules — knob clamped to the base radius, normalized vector
    /// reported on every change, zero on release (which also hides it).
    private struct JoystickOverlay: View {
        var onChange: (SIMD2<Float>) -> Void
        @State private var origin: CGPoint?
        @State private var knobOffset: CGSize = .zero
        private let radius: CGFloat = 55

        var body: some View {
            ZStack(alignment: .topLeading) {
                Color.clear
                if let origin {
                    Circle()
                        .fill(.ultraThinMaterial)
                        .overlay(Circle().strokeBorder(.white.opacity(0.25)))
                        .frame(width: radius * 2, height: radius * 2)
                        .position(origin)
                    Circle()
                        .fill(.white.opacity(0.65))
                        .frame(width: 44, height: 44)
                        .position(x: origin.x + knobOffset.width,
                                  y: origin.y + knobOffset.height)
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        let base = origin ?? value.startLocation
                        if origin == nil { origin = base }
                        var dx = value.location.x - base.x
                        var dy = value.location.y - base.y
                        let length = sqrt(dx * dx + dy * dy)
                        if length > radius {
                            dx *= radius / length
                            dy *= radius / length
                        }
                        knobOffset = CGSize(width: dx, height: dy)
                        // Screen-up is negative dy; the controller wants +y = up.
                        onChange(SIMD2<Float>(Float(dx / radius), Float(-dy / radius)))
                    }
                    .onEnded { _ in
                        origin = nil
                        knobOffset = .zero
                        onChange(.zero)
                    }
            )
        }
    }

    private var hint: String {
        if !controller.isPlaced {
            return "Tap a surface to place Pipo"
        }
        if controller.isDrawingPath {
            return "Tap to add path points • tap Go when done"
        }
        if controller.isFreehand {
            return "Drag the colored arrows to move Pipo in 3D • tap Done"
        }
        if controller.isGameMode {
            return "Touch the lower-left area and drag to drive Pipo"
        }
        if controller.isSitting {
            return "Pipo is sitting — tap Stand first to move him"
        }
        return "Drag to move Pipo • twist to turn him"
    }
}
