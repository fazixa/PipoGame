import SwiftUI

struct ContentView: View {
    @StateObject private var controller = PipoController()
    @StateObject private var recorder = ScreenRecorder()
    @State private var uiVisible = true

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

    private var hint: String {
        if !controller.isPlaced {
            return "Tap a surface to place Pipo"
        }
        if controller.isSitting {
            return "Pipo is sitting — tap Stand first to move him"
        }
        return "Tap anywhere to make Pipo walk there"
    }
}
