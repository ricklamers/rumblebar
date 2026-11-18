// RumbleBar — menu bar dictation using Reson8/Rumble-2-Flash running locally.
//
// The app records the microphone while dictation is active (⌥D toggles,
// Esc cancels), sends 16 kHz mono float32 audio to a local Python server
// (server/rumble_server.py, launched and owned by this app), and pastes the
// transcript into the focused application.

import AVFoundation
import AppKit
import Carbon.HIToolbox
import ServiceManagement

// MARK: - Configuration

enum Config {
    static let info = Bundle.main.infoDictionary ?? [:]
    static var python: String { info["RumbleBarPython"] as? String ?? "" }
    static var serverScript: String { info["RumbleBarServerScript"] as? String ?? "" }
    static var port: Int {
        if let p = info["RumbleBarPort"] as? Int { return p }
        if let s = info["RumbleBarPort"] as? String, let p = Int(s) { return p }
        return 8765
    }
    static var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }

    static let supportDir: URL = {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("RumbleBar", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()
    static var debugDir: URL { supportDir.appendingPathComponent("Debug Recordings", isDirectory: true) }
    static let logDir: URL = {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/RumbleBar", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    // From the model's config.json `support_languages`.
    static let languages = [
        "Arabic", "Cantonese", "Chinese", "Czech", "Danish", "Dutch", "English",
        "Filipino", "Finnish", "French", "German", "Greek", "Hindi", "Hungarian",
        "Indonesian", "Italian", "Japanese", "Korean", "Macedonian", "Malay",
        "Persian", "Polish", "Portuguese", "Romanian", "Russian", "Spanish",
        "Swedish", "Thai", "Turkish", "Vietnamese",
    ]
}

// MARK: - Settings

final class Settings {
    private let d = UserDefaults.standard

    init() {
        d.register(defaults: [
            "language": "auto", "digits": true, "filler": false, "verbatim": true,
            "context": "", "sounds": false, "saveDebug": true,
        ])
    }

    var language: String { get { d.string(forKey: "language") ?? "auto" } set { d.set(newValue, forKey: "language") } }
    var digits: Bool { get { d.bool(forKey: "digits") } set { d.set(newValue, forKey: "digits") } }
    var filler: Bool { get { d.bool(forKey: "filler") } set { d.set(newValue, forKey: "filler") } }
    var verbatim: Bool { get { d.bool(forKey: "verbatim") } set { d.set(newValue, forKey: "verbatim") } }
    var context: String { get { d.string(forKey: "context") ?? "" } set { d.set(newValue, forKey: "context") } }
    var sounds: Bool { get { d.bool(forKey: "sounds") } set { d.set(newValue, forKey: "sounds") } }
    var saveDebug: Bool { get { d.bool(forKey: "saveDebug") } set { d.set(newValue, forKey: "saveDebug") } }
}

// MARK: - Server process

final class ServerManager {
    enum State: Equatable { case stopped, starting, loading, ready(device: String), failed(String) }

    var onChange: ((State) -> Void)?
    private(set) var state: State = .stopped { didSet { if state != oldValue { onChange?(state) } } }
    private var process: Process?
    private var pollTimer: Timer?

    func start() {
        state = .starting
        // Reuse a server that is already up on our port (e.g. started by hand).
        health { [weak self] status in
            guard let self else { return }
            if status == nil { self.spawn() }
            self.beginPolling()
        }
    }

    private func spawn() {
        guard FileManager.default.isExecutableFile(atPath: Config.python),
              FileManager.default.fileExists(atPath: Config.serverScript) else {
            state = .failed("Python env not found — run scripts/setup.sh and rebuild")
            return
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: Config.python)
        p.arguments = ["-I", Config.serverScript, "--port", String(Config.port),
                       "--parent-pid", String(ProcessInfo.processInfo.processIdentifier)]
        var env = ProcessInfo.processInfo.environment
        env["HF_HUB_OFFLINE"] = "1"          // model is fetched by setup.sh; never hit the network
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        p.environment = env
        let logURL = Config.logDir.appendingPathComponent("server.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        if let fh = try? FileHandle(forWritingTo: logURL) {
            p.standardOutput = fh
            p.standardError = fh
        }
        p.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                guard let self, self.process === proc else { return }
                self.process = nil
                self.state = .failed("Server exited (code \(proc.terminationStatus)) — see server log")
            }
        }
        do {
            try p.run()
            process = p
        } catch {
            state = .failed("Could not launch server: \(error.localizedDescription)")
        }
    }

    private func beginPolling() {
        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.health { status in
                guard let self else { return }
                if case .failed = self.state { self.pollTimer?.invalidate(); return }
                guard let status else { return }  // not listening yet
                switch status["status"] as? String {
                case "ready":
                    self.state = .ready(device: (status["device"] as? String ?? "?").uppercased())
                    self.pollTimer?.invalidate()
                case "error":
                    self.state = .failed(status["error"] as? String ?? "model failed to load")
                    self.pollTimer?.invalidate()
                default:
                    self.state = .loading
                }
            }
        }
    }

    func health(_ done: @escaping ([String: Any]?) -> Void) {
        var req = URLRequest(url: Config.baseURL.appendingPathComponent("health"))
        req.timeoutInterval = 2
        URLSession.shared.dataTask(with: req) { data, _, _ in
            let obj = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
            DispatchQueue.main.async { done(obj) }
        }.resume()
    }

    func restart() {
        stop()
        // Give the old process a moment to release the port.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { self.start() }
    }

    func stop() {
        pollTimer?.invalidate()
        if let p = process {
            process = nil
            p.terminationHandler = nil
            p.terminate()
        }
        state = .stopped
    }
}

// MARK: - Audio capture

final class Recorder {
    static let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                      channels: 1, interleaved: false)!
    private var engine: AVAudioEngine?
    private var converter: AVAudioConverter?
    private var samples: [Float] = []
    private let lock = NSLock()

    var isRunning: Bool { engine?.isRunning ?? false }

    func start() throws {
        lock.lock(); samples.removeAll(keepingCapacity: true); lock.unlock()
        // A fresh engine per session picks up input-device changes cleanly.
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw NSError(domain: "RumbleBar", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No audio input device available"])
        }
        converter = AVAudioConverter(from: format, to: Recorder.target)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            self?.append(buffer)
        }
        engine.prepare()
        try engine.start()
        self.engine = engine
    }

    private func append(_ buffer: AVAudioPCMBuffer) {
        guard let converter else { return }
        let ratio = Recorder.target.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 64)
        guard let out = AVAudioPCMBuffer(pcmFormat: Recorder.target, frameCapacity: capacity) else { return }
        var supplied = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, let ch = out.floatChannelData, out.frameLength > 0 else { return }
        lock.lock()
        samples.append(contentsOf: UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)))
        lock.unlock()
    }

    /// Stops capture and returns everything recorded since `start()`.
    func stop() -> [Float] {
        if let engine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        engine = nil
        converter = nil
        lock.lock(); defer { lock.unlock() }
        let out = samples
        samples = []
        return out
    }
}

// MARK: - Global hotkeys (Carbon; no Accessibility permission needed)

final class HotKeys {
    enum ID: UInt32 { case toggle = 1, cancel = 2 }
    static var handler: ((ID) -> Void)?
    private var refs: [ID: EventHotKeyRef] = [:]
    private static let signature: OSType = 0x524D_4252  // 'RMBR'

    init() {
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, event, _ -> OSStatus in
            var hk = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                              nil, MemoryLayout<EventHotKeyID>.size, nil, &hk)
            if let id = HotKeys.ID(rawValue: hk.id) {
                DispatchQueue.main.async { HotKeys.handler?(id) }
            }
            return noErr
        }, 1, &spec, nil, nil)
    }

    @discardableResult
    func register(_ id: ID, keyCode: Int, modifiers: Int) -> Bool {
        unregister(id)
        var ref: EventHotKeyRef?
        let status = RegisterEventHotKey(UInt32(keyCode), UInt32(modifiers),
                                         EventHotKeyID(signature: HotKeys.signature, id: id.rawValue),
                                         GetApplicationEventTarget(), 0, &ref)
        if status == noErr, let ref { refs[id] = ref; return true }
        return false
    }

    func unregister(_ id: ID) {
        if let ref = refs.removeValue(forKey: id) { UnregisterEventHotKey(ref) }
    }
}

// MARK: - Text insertion

enum Inserter {
    /// Pastes `text` into the focused app via the clipboard, then restores the
    /// previous clipboard contents. Falls back to leaving text on the clipboard
    /// if Accessibility access has not been granted.
    @discardableResult
    static func insert(_ text: String) -> Bool {
        let pb = NSPasteboard.general
        let saved: [NSPasteboardItem] = (pb.pasteboardItems ?? []).map { item in
            let copy = NSPasteboardItem()
            for type in item.types {
                if let data = item.data(forType: type) { copy.setData(data, forType: type) }
            }
            return copy
        }
        pb.clearContents()
        pb.setString(text, forType: .string)
        guard AXIsProcessTrusted() else { return false }

        let ourChange = pb.changeCount
        let src = CGEventSource(stateID: .combinedSessionState)
        let vKey: CGKeyCode = 9
        let down = CGEvent(keyboardEventSource: src, virtualKey: vKey, keyDown: true)
        let up = CGEvent(keyboardEventSource: src, virtualKey: vKey, keyDown: false)
        down?.flags = .maskCommand
        up?.flags = .maskCommand
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            // Only restore if nobody else touched the clipboard meanwhile.
            guard pb.changeCount == ourChange else { return }
            pb.clearContents()
            if !saved.isEmpty { pb.writeObjects(saved) }
        }
        return true
    }
}

// MARK: - Debug recordings

enum DebugStore {
    static func save(samples: [Float], text: String) {
        let dir = Config.debugDir
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let base = dir.appendingPathComponent(f.string(from: Date()))
        try? wav(samples).write(to: base.appendingPathExtension("wav"))
        try? text.write(to: base.appendingPathExtension("txt"), atomically: true, encoding: .utf8)
    }

    /// 32-bit float, mono, 16 kHz WAV — the exact samples sent to the model.
    static func wav(_ samples: [Float]) -> Data {
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        let dataBytes = UInt32(samples.count * 4)
        d.append(contentsOf: Array("RIFF".utf8)); u32(36 + dataBytes)
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16)
        u16(3); u16(1); u32(16_000); u32(16_000 * 4); u16(4); u16(32)
        d.append(contentsOf: Array("data".utf8)); u32(dataBytes)
        samples.withUnsafeBufferPointer { d.append(UnsafeBufferPointer(start: UnsafeRawPointer($0.baseAddress!)
            .assumingMemoryBound(to: UInt8.self), count: samples.count * 4)) }
        return d
    }
}

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    private var statusItem: NSStatusItem!
    private let settings = Settings()
    private let server = ServerManager()
    private let recorder = Recorder()
    private let hotkeys = HotKeys()
    private var recordingStart: Date?
    private var pending = 0
    private var lastTranscript: String?
    private var lastMessage: String?
    private var messageTimer: Timer?

    func applicationDidFinishLaunching(_ note: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        let menu = NSMenu()
        menu.delegate = self
        statusItem.menu = menu

        HotKeys.handler = { [weak self] id in
            switch id {
            case .toggle: self?.toggleRecording()
            case .cancel: self?.cancelRecording()
            }
        }
        if !hotkeys.register(.toggle, keyCode: kVK_ANSI_D, modifiers: optionKey) {
            flash("⌥D is taken by another app")
        }

        server.onChange = { [weak self] _ in self?.refreshIcon() }
        server.start()
        refreshIcon()

        AVCaptureDevice.requestAccess(for: .audio) { _ in }
        let prompt = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        _ = AXIsProcessTrustedWithOptions([prompt: true] as CFDictionary)
    }

    func applicationWillTerminate(_ note: Notification) {
        _ = recorder.stop()
        server.stop()
    }

    // MARK: Dictation

    @objc func toggleRecording() {
        recorder.isRunning ? finishRecording() : startRecording()
    }

    private func startRecording() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .denied, .restricted:
            flash("Microphone access denied — enable it in System Settings")
            openPrivacy("Privacy_Microphone")
            return
        default: break
        }
        do {
            try recorder.start()
        } catch {
            flash("Mic error: \(error.localizedDescription)", beep: true)
            return
        }
        recordingStart = Date()
        hotkeys.register(.cancel, keyCode: kVK_Escape, modifiers: 0)
        if settings.sounds { NSSound(named: "Bottle")?.play() }
        refreshIcon()
    }

    @objc func cancelRecording() {
        guard recorder.isRunning else { return }
        _ = recorder.stop()
        hotkeys.unregister(.cancel)
        if settings.sounds { NSSound(named: "Funk")?.play() }
        flash("Dictation cancelled")
    }

    private func finishRecording() {
        let samples = recorder.stop()
        hotkeys.unregister(.cancel)
        if settings.sounds { NSSound(named: "Pop")?.play() }
        guard samples.count >= 16_000 / 4 else {
            flash("Recording too short")
            return
        }
        pending += 1
        refreshIcon()
        transcribe(samples, attempt: 0)
    }

    private func transcribe(_ samples: [Float], attempt: Int) {
        var comps = URLComponents(url: Config.baseURL.appendingPathComponent("transcribe"),
                                  resolvingAgainstBaseURL: false)!
        comps.queryItems = [
            URLQueryItem(name: "language", value: settings.language),
            URLQueryItem(name: "digits", value: settings.digits ? "1" : "0"),
            URLQueryItem(name: "filler", value: settings.filler ? "1" : "0"),
            URLQueryItem(name: "verbatim", value: settings.verbatim ? "1" : "0"),
            URLQueryItem(name: "context", value: settings.context),
        ]
        var req = URLRequest(url: comps.url!)
        req.httpMethod = "POST"
        req.timeoutInterval = 600
        req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        req.setValue("1", forHTTPHeaderField: "X-Rumble-Client")
        req.httpBody = samples.withUnsafeBufferPointer { Data(buffer: $0) }

        URLSession.shared.dataTask(with: req) { [weak self] data, response, error in
            DispatchQueue.main.async {
                guard let self else { return }
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                let obj = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
                // Model still loading (or server still starting): wait and retry for up to 2 minutes.
                if (code == 503 || error != nil) && attempt < 120 {
                    if attempt == 0 { self.flash("Waiting for model to load…") }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                        self.transcribe(samples, attempt: attempt + 1)
                    }
                    return
                }
                self.pending -= 1
                defer { self.refreshIcon() }
                guard code == 200, let text = obj?["text"] as? String else {
                    let msg = obj?["error"] as? String ?? error?.localizedDescription ?? "HTTP \(code)"
                    self.flash("Transcription failed: \(msg)", beep: true)
                    if self.settings.saveDebug { DebugStore.save(samples: samples, text: "") }
                    return
                }
                if self.settings.saveDebug { DebugStore.save(samples: samples, text: text) }
                guard !text.isEmpty else { self.flash("No speech detected"); return }
                self.lastTranscript = text
                if !Inserter.insert(text) {
                    self.flash("Copied to clipboard — grant Accessibility to auto-paste")
                }
            }
        }.resume()
    }

    // MARK: Status icon

    private func refreshIcon() {
        guard let button = statusItem.button else { return }
        let symbol: String
        var tint: NSColor? = nil
        if recorder.isRunning {
            symbol = "mic.fill"; tint = .systemRed
        } else if pending > 0 {
            symbol = "ellipsis.circle"
        } else {
            switch server.state {
            case .ready: symbol = "waveform"
            case .failed: symbol = "exclamationmark.triangle"
            default: symbol = "hourglass"
            }
        }
        var img = NSImage(systemSymbolName: symbol, accessibilityDescription: "RumbleBar")
        if let tint {
            // contentTintColor on a template image renders near-black in the
            // macOS 26 menu bar (invisible on dark backgrounds); bake the color
            // into the symbol instead and opt out of template rendering.
            img = img?.withSymbolConfiguration(NSImage.SymbolConfiguration(paletteColors: [tint]))
            img?.isTemplate = false
        } else {
            img?.isTemplate = true
        }
        button.image = img
        button.toolTip = statusLine()
    }

    private func statusLine() -> String {
        if let m = lastMessage { return m }
        if recorder.isRunning {
            let secs = Int(Date().timeIntervalSince(recordingStart ?? Date()))
            return "Recording… \(secs)s (⌥D to finish, Esc to cancel)"
        }
        if pending > 0 { return "Transcribing…" }
        switch server.state {
        case .stopped: return "Server stopped"
        case .starting: return "Starting server…"
        case .loading: return "Loading Rumble-2-Flash…"
        case .ready(let device): return "Ready — Rumble-2-Flash on \(device)"
        case .failed(let msg): return "Error: \(msg)"
        }
    }

    /// Shows a transient message in the menu's status line and the icon tooltip.
    private func flash(_ message: String, beep: Bool = false) {
        lastMessage = message
        messageTimer?.invalidate()
        messageTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: false) { [weak self] _ in
            self?.lastMessage = nil
            self?.refreshIcon()
        }
        if beep { NSSound.beep() }
        refreshIcon()
    }

    // MARK: Menu

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        let status = NSMenuItem(title: statusLine(), action: nil, keyEquivalent: "")
        status.isEnabled = false
        menu.addItem(status)
        menu.addItem(.separator())

        let toggle = item(recorder.isRunning ? "Stop Dictation" : "Start Dictation", #selector(toggleRecording))
        toggle.keyEquivalent = "d"
        toggle.keyEquivalentModifierMask = [.option]
        menu.addItem(toggle)
        if recorder.isRunning { menu.addItem(item("Cancel Dictation (Esc)", #selector(cancelRecording))) }
        if lastTranscript != nil { menu.addItem(item("Copy Last Transcript", #selector(copyLast))) }
        menu.addItem(.separator())

        // Language submenu
        let langItem = NSMenuItem(title: "Language: \(settings.language == "auto" ? "Auto-detect" : settings.language)",
                                  action: nil, keyEquivalent: "")
        let langMenu = NSMenu()
        for lang in ["auto"] + Config.languages {
            let i = item(lang == "auto" ? "Auto-detect" : lang, #selector(pickLanguage(_:)))
            i.representedObject = lang
            i.state = settings.language == lang ? .on : .off
            langMenu.addItem(i)
            if lang == "auto" { langMenu.addItem(.separator()) }
        }
        langItem.submenu = langMenu
        menu.addItem(langItem)

        menu.addItem(check("Numbers as Digits", settings.digits, #selector(toggleDigits)))
        menu.addItem(check("Keep Filler Words (um, uh)", settings.filler, #selector(toggleFiller)))
        menu.addItem(check("Verbatim (keep repeated words)", settings.verbatim, #selector(toggleVerbatim)))
        let vocab = settings.context.isEmpty ? "Custom Vocabulary…" : "Custom Vocabulary (set)…"
        menu.addItem(item(vocab, #selector(editVocabulary)))
        menu.addItem(.separator())

        menu.addItem(check("Sound Effects", settings.sounds, #selector(toggleSounds)))
        menu.addItem(check("Save Debug Recordings", settings.saveDebug, #selector(toggleSaveDebug)))
        menu.addItem(check("Launch at Login", SMAppService.mainApp.status == .enabled, #selector(toggleLogin)))
        if !AXIsProcessTrusted() {
            menu.addItem(item("Grant Accessibility (for auto-paste)…", #selector(openAccessibility)))
        }
        menu.addItem(.separator())
        menu.addItem(item("Open Debug Recordings…", #selector(openDebug)))
        menu.addItem(item("Open Server Log…", #selector(openLog)))
        menu.addItem(item("Restart Server", #selector(restartServer)))
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Quit RumbleBar", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
    }

    private func item(_ title: String, _ action: Selector) -> NSMenuItem {
        let i = NSMenuItem(title: title, action: action, keyEquivalent: "")
        i.target = self
        return i
    }

    private func check(_ title: String, _ on: Bool, _ action: Selector) -> NSMenuItem {
        let i = item(title, action)
        i.state = on ? .on : .off
        return i
    }

    @objc private func pickLanguage(_ sender: NSMenuItem) {
        settings.language = sender.representedObject as? String ?? "auto"
    }
    @objc private func toggleDigits() { settings.digits.toggle() }
    @objc private func toggleFiller() { settings.filler.toggle() }
    @objc private func toggleVerbatim() { settings.verbatim.toggle() }
    @objc private func toggleSounds() { settings.sounds.toggle() }
    @objc private func toggleSaveDebug() { settings.saveDebug.toggle() }

    @objc private func toggleLogin() {
        do {
            if SMAppService.mainApp.status == .enabled {
                try SMAppService.mainApp.unregister()
            } else {
                try SMAppService.mainApp.register()
            }
        } catch {
            flash("Launch at Login failed: \(error.localizedDescription)")
        }
    }

    @objc private func copyLast() {
        guard let t = lastTranscript else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(t, forType: .string)
    }

    @objc private func editVocabulary() {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = "Custom Vocabulary"
        alert.informativeText = "Names, jargon, or phrases to bias recognition toward (comma separated). "
            + "Sent to the model as context with every dictation."
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 360, height: 24))
        field.stringValue = settings.context
        field.placeholderString = "e.g. Reson8, Rumble-2-Flash, Kubernetes"
        alert.accessoryView = field
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Cancel")
        alert.window.initialFirstResponder = field
        if alert.runModal() == .alertFirstButtonReturn {
            settings.context = field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    @objc private func openAccessibility() {
        let prompt = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        _ = AXIsProcessTrustedWithOptions([prompt: true] as CFDictionary)
        openPrivacy("Privacy_Accessibility")
    }

    private func openPrivacy(_ anchor: String) {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func openDebug() {
        try? FileManager.default.createDirectory(at: Config.debugDir, withIntermediateDirectories: true)
        NSWorkspace.shared.open(Config.debugDir)
    }

    @objc private func openLog() {
        NSWorkspace.shared.open(Config.logDir.appendingPathComponent("server.log"))
    }

    @objc private func restartServer() { server.restart() }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
