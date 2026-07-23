import Foundation

/// TEMP DIAGNOSTIC: file-based logging, since live console-attach via
/// devicectl has proven unreliable in this environment — write to a file
/// in Documents instead, pulled after each test via
/// `xcrun devicectl device copy from --domain-type appDataContainer
/// --domain-identifier me.faeshayesteh.PipoGame --source "Documents"
/// --destination <path>`.
enum PipoLog {
    private static let url: URL = {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return dir.appendingPathComponent("pipo_debug.log")
    }()

    static func log(_ message: String) {
        let line = "[\(Date())] \(message)\n"
        print(line)
        guard let data = line.data(using: .utf8) else { return }
        if FileManager.default.fileExists(atPath: url.path) {
            if let handle = try? FileHandle(forWritingTo: url) {
                handle.seekToEndOfFile()
                handle.write(data)
                try? handle.close()
            }
        } else {
            try? data.write(to: url)
        }
    }
}
