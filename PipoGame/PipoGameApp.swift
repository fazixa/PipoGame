import SwiftUI

@main
struct PipoGameApp: App {
    init() {
        // DEBUG: stdout is block-buffered (not line-buffered) when piped
        // through devicectl's console, so print() output sat invisible
        // until the buffer filled or the process fully exited. Disable
        // buffering so debug prints show up immediately while diagnosing
        // the fall/landing prototype.
        setvbuf(stdout, nil, _IONBF, 0)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
