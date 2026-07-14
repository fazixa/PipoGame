import SwiftUI

struct ContentView: View {
    @StateObject private var controller: PipoController
    @ObservedObject private var geospatial: GeospatialManager
    @StateObject private var recorder = ScreenRecorder()
    @State private var uiVisible = true

    init() {
        let controller = PipoController()
        _controller = StateObject(wrappedValue: controller)
        _geospatial = ObservedObject(wrappedValue: controller.geospatial)
    }

    var body: some View {
        ZStack {
            ARViewContainer(controller: controller)
                .ignoresSafeArea()

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
                            Label(controller.isSitting ? "Stand" : "Sit",
                                  systemImage: controller.isSitting ? "figure.stand" : "chair")
                                .frame(maxWidth: .infinity)
                        }
                        .disabled(!controller.isPlaced || !controller.supportsSit)

                        Button {
                            startRecording()
                        } label: {
                            Image(systemName: "record.circle.fill")
                                .font(.title2)
                                .foregroundStyle(.red)
                                .frame(maxWidth: .infinity)
                        }

                        Button {
                            controller.toggleToon()
                        } label: {
                            Image(systemName: controller.isToon
                                  ? "paintbrush.fill" : "paintbrush")
                                .frame(maxWidth: .infinity)
                        }
                        .disabled(!controller.isPlaced || !controller.supportsToon)

                        Button {
                            controller.toggleHandMode()
                        } label: {
                            Image(systemName: controller.isHandMode
                                  ? "hand.raised.fill" : "hand.raised")
                                .frame(maxWidth: .infinity)
                        }

                        Button {
                            controller.toggleGeospatial()
                        } label: {
                            Image(systemName: controller.isGeoActive
                                  ? "globe.americas.fill" : "globe.americas")
                                .frame(maxWidth: .infinity)
                        }

                        Button {
                            controller.toggleZAxisDragMode()
                        } label: {
                            Image(systemName: controller.isZAxisDragMode
                                  ? "arrow.up.and.down.circle.fill" : "arrow.left.and.right.circle")
                                .frame(maxWidth: .infinity)
                        }
                        .disabled(!controller.isGeoActive)

                        Button(role: .destructive) {
                            controller.reset()
                        } label: {
                            Label("Reset", systemImage: "arrow.counterclockwise")
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

            if controller.isGeoActive {
                VStack {
                    Text(geospatial.statusText)
                        .font(.caption.monospaced())
                        .foregroundStyle(geospatial.isHighAccuracy ? .green : .primary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.top, 54)
                    Spacer()
                }
                .transition(.opacity)
            }

            if controller.isGeoActive {
                VStack {
                    HStack {
                        Spacer()
                        calibrationPanel
                            .padding(.trailing, 12)
                    }
                    .padding(.top, 100)
                    Spacer()
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
            controller.onLongPress = { [weak recorder, weak geospatial] in
                if let recorder, recorder.isRecording {
                    recorder.stop()
                }
                geospatial?.setDebugVisualsHidden(false)
                // Also restores the UI if recording failed to start.
                uiVisible = true
            }
        }
    }

    private func startRecording() {
        uiVisible = false
        geospatial.setDebugVisualsHidden(true)
        // Let the UI fade out fully before capture begins so it never
        // appears in the footage.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            guard let arView = controller.arView else {
                uiVisible = true
                geospatial.setDebugVisualsHidden(false)
                return
            }
            recorder.start(arView: arView)
        }
    }

    /// Manual X/Y/Z nudge for the Streetscape Geometry mesh (both the
    /// invisible occluder and the debug tint move together) — corrects
    /// VPS's small positioning error by hand, 0.5m per tap.
    private var calibrationPanel: some View {
        VStack(spacing: 6) {
            Text("Building calibration")
                .font(.caption2)
                .foregroundStyle(.secondary)
            calibrationRow(axis: "X", onMinus: { geospatial.nudgeGeometryCalibration(x: -0.5) },
                          onPlus: { geospatial.nudgeGeometryCalibration(x: 0.5) })
            calibrationRow(axis: "Y", onMinus: { geospatial.nudgeGeometryCalibration(y: -0.5) },
                          onPlus: { geospatial.nudgeGeometryCalibration(y: 0.5) })
            calibrationRow(axis: "Z", onMinus: { geospatial.nudgeGeometryCalibration(z: -0.5) },
                          onPlus: { geospatial.nudgeGeometryCalibration(z: 0.5) })
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func calibrationRow(axis: String, onMinus: @escaping () -> Void,
                                onPlus: @escaping () -> Void) -> some View {
        HStack(spacing: 8) {
            Text(axis)
                .font(.caption.monospaced())
                .frame(width: 14)
            Button(action: onMinus) {
                Image(systemName: "minus.circle.fill")
            }
            Button(action: onPlus) {
                Image(systemName: "plus.circle.fill")
            }
        }
        .buttonStyle(.plain)
        .font(.title3)
    }

    private var hint: String {
        if controller.isHandMode {
            return controller.searchingForHand
                ? "Show your open palm to the camera"
                : "Pipo is riding your hand — tap ✋ to set him down"
        }
        if !controller.isPlaced {
            return "Tap a surface to place Pipo"
        }
        if controller.isSitting {
            return "Pipo is sitting — tap Stand first to move him"
        }
        return "Tap anywhere to make Pipo walk there"
    }
}
