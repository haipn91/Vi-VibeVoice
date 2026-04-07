const { app, BrowserWindow, clipboard, globalShortcut, ipcMain, Menu, nativeImage, nativeTheme, screen } = require("electron");
const path = require("path");
const fs = require("fs");
const os = require("os");
const { execFileSync, spawn } = require("child_process");

const CONFIG_FILE = "vi-vibevoice.config.json";
/**
 * Default: Ctrl + ` (two keys) — works with Electron globalShortcut on Windows; Ctrl+Win often fails.
 */
const DEFAULT_SHORTCUT = "Control+Backquote";
/** Windows: migrate shortcuts that were unreliable or 3-key defaults. */
const UNSUPPORTED_WIN_SUPER_HINT = /control.*\+.*super|super.*\+.*control|^super$/i;
/** Prefer two-key modifiers; keep one 3-key option as last resort. */
const FALLBACK_SHORTCUTS = [
  "Control+Semicolon",
  "Control+Period",
  "Control+Slash",
  "Control+Space",
  "CommandOrControl+Shift+Space"
];
/** Bản gọn: chỉ ASR qua HTTP API (không wake word, không Python / ASR cục bộ trong bộ cài). */
const DEFAULT_CONFIG = {
  api: {
    endpoint: "",
    method: "POST",
    audioField: "audio",
    responseTextPath: "text",
    requestTimeoutMs: 90000,
    /** Nếu đặt, renderer thêm Authorization: Bearer + apiKey khi headers JSON chưa có Authorization. */
    apiKey: "",
    headersJson: "{}",
    extraFormDataJson: "{\"language\":\"vi\",\"model\":\"g-group-ai-lab/gipformer-65M-rnnt\"}"
  },
  recording: {
    sampleRate: 16000,
    channelCount: 1,
    mimeType: "audio/webm;codecs=opus",
    silenceThreshold: 0.018,
    silenceDurationMs: 1600
  },
  shortcuts: {
    toggleRecording: DEFAULT_SHORTCUT
  },
  output: {
    copyAsrToClipboard: true,
    pasteToForegroundAfterAsr: true,
    minimizeBeforePaste: true,
    pasteDelayMs: 800
  },
  asr: {
    mode: "remote"
  },
  ui: {
    /** Floating always-on-top HUD when recording / processing (visible even if main window minimized). */
    recordingHud: true
  }
};

/** Đổi tên file so với bản cũ: cài mới / lần đầu dùng build này → All time bắt đầu từ 0. */
const STATS_FILE = "vi-vibevoice-usage-stats.json";

let mainWindow = null;
let recordingHudWindow = null;
let recordingHudHideTimer = null;
let currentShortcut = DEFAULT_SHORTCUT;
/** Last globalShortcut registration outcome (for UI: conflicts with other apps e.g. Microsoft VibeVoice). */
let lastShortcutRegistration = { ok: false, requested: "", active: null };
/** Bundled `local_asr_server.py` child (fat Windows installer only). */
let localAsrProcess = null;

const LOCAL_ASR_PORT = 18765;

const HUD_W = 300;
const HUD_H = 88;
const HUD_MARGIN = 20;

/** Same artwork as sidebar (logo.svg), rasterized for Windows/Linux window & taskbar icons. */
function getAppWindowIcon() {
  const iconPath = path.join(app.getAppPath(), "assets", "icon.png");
  if (!fs.existsSync(iconPath)) {
    return undefined;
  }
  try {
    const img = nativeImage.createFromPath(iconPath);
    return img.isEmpty() ? undefined : img;
  } catch (_e) {
    return undefined;
  }
}

function repositionRecordingHud(win) {
  if (!win || win.isDestroyed()) {
    return;
  }
  const wa = screen.getPrimaryDisplay().workArea;
  const x = wa.x + wa.width - HUD_W - HUD_MARGIN;
  const y = wa.y + wa.height - HUD_H - HUD_MARGIN;
  win.setBounds({ x, y, width: HUD_W, height: HUD_H });
}

function destroyRecordingHudWindow() {
  if (recordingHudHideTimer) {
    clearTimeout(recordingHudHideTimer);
    recordingHudHideTimer = null;
  }
  if (recordingHudWindow && !recordingHudWindow.isDestroyed()) {
    try {
      recordingHudWindow.close();
    } catch (_e) {
      /* ignore */
    }
  }
  recordingHudWindow = null;
}

function ensureRecordingHudWindow() {
  if (recordingHudWindow && !recordingHudWindow.isDestroyed()) {
    repositionRecordingHud(recordingHudWindow);
    return recordingHudWindow;
  }
  recordingHudWindow = new BrowserWindow({
    width: HUD_W,
    height: HUD_H,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    movable: true,
    maximizable: false,
    minimizable: false,
    fullscreenable: false,
    focusable: false,
    hasShadow: true,
    show: false,
    backgroundColor: "#00000000",
    title: "Vi-VibeVoice · Recording",
    icon: getAppWindowIcon(),
    webPreferences: {
      preload: path.join(__dirname, "hud-preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  try {
    recordingHudWindow.setAlwaysOnTop(true, "floating");
  } catch (_e) {
    recordingHudWindow.setAlwaysOnTop(true);
  }
  if (process.platform === "darwin") {
    try {
      recordingHudWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
    } catch (_e) {
      /* optional API */
    }
  }
  recordingHudWindow.loadFile(path.join(app.getAppPath(), "src", "recording-hud.html"));
  repositionRecordingHud(recordingHudWindow);
  recordingHudWindow.on("closed", () => {
    recordingHudWindow = null;
  });
  return recordingHudWindow;
}

function sendRecordingHudRender(payload) {
  const win = ensureRecordingHudWindow();
  if (!win || win.isDestroyed()) {
    return;
  }
  const send = () => {
    if (win && !win.isDestroyed() && win.webContents && !win.webContents.isDestroyed()) {
      win.webContents.send("hud:render", payload);
    }
  };
  if (win.webContents.isLoading()) {
    win.webContents.once("did-finish-load", send);
  } else {
    send();
  }
}

function showRecordingHudMode(opts) {
  const { mode, detail } = opts || {};
  const cfg = loadConfig();
  if (cfg.ui && cfg.ui.recordingHud === false) {
    if (recordingHudWindow && !recordingHudWindow.isDestroyed()) {
      recordingHudWindow.hide();
    }
    return;
  }
  if (recordingHudHideTimer) {
    clearTimeout(recordingHudHideTimer);
    recordingHudHideTimer = null;
  }

  if (!mode || mode === "off") {
    if (recordingHudWindow && !recordingHudWindow.isDestroyed()) {
      recordingHudWindow.hide();
    }
    return;
  }

  const labels = {
    recording: {
      title: "Recording",
      subtitle: "Press the shortcut again or Stop to send audio",
      pulse: true
    },
    processing: {
      title: "Processing",
      subtitle: "Sending audio to ASR…",
      pulse: false
    },
    success: {
      title: "Done",
      subtitle: "Transcript received",
      pulse: false
    },
    error: {
      title: "Error",
      subtitle: "See the main window",
      pulse: false
    }
  };

  const pack = labels[mode] || labels.error;
  const dot = mode === "success" ? "success" : mode === "error" ? "error" : "";
  sendRecordingHudRender({
    title: pack.title,
    subtitle: detail || pack.subtitle,
    pulse: !!pack.pulse,
    dot
  });

  const win = ensureRecordingHudWindow();
  repositionRecordingHud(win);
  if (!win.isVisible()) {
    win.showInactive();
  }
}

function scheduleRecordingHudHide(ms) {
  if (recordingHudHideTimer) {
    clearTimeout(recordingHudHideTimer);
  }
  recordingHudHideTimer = setTimeout(() => {
    recordingHudHideTimer = null;
    showRecordingHudMode({ mode: "off" });
  }, Math.min(Math.max(Number(ms) || 1800, 400), 8000));
}

ipcMain.handle("recording-hud:update", (_event, payload) => {
  const mode = payload && typeof payload.mode === "string" ? payload.mode : "off";
  const detail = payload && payload.detail != null ? String(payload.detail).slice(0, 220) : "";
  const dismissMs = payload && payload.dismissMs != null ? Number(payload.dismissMs) : NaN;

  showRecordingHudMode({ mode, detail });

  if (
    Number.isFinite(dismissMs) &&
    dismissMs > 0 &&
    mode !== "recording" &&
    mode !== "processing" &&
    mode !== "off"
  ) {
    scheduleRecordingHudHide(dismissMs);
  }
  return { ok: true };
});

function statsPath() {
  return path.join(app.getPath("userData"), STATS_FILE);
}

function todayKey() {
  return new Date().toISOString().slice(0, 10);
}

function loadStats() {
  const defaults = {
    todayKey: todayKey(),
    todayWords: 0,
    todayUses: 0,
    totalWords: 0,
    totalUses: 0
  };
  try {
    const raw = fs.readFileSync(statsPath(), "utf8");
    const s = { ...defaults, ...JSON.parse(raw) };
    const d = todayKey();
    if (s.todayKey !== d) {
      s.todayKey = d;
      s.todayWords = 0;
      s.todayUses = 0;
    }
    return s;
  } catch (_e) {
    return { ...defaults };
  }
}

function saveStats(s) {
  fs.writeFileSync(statsPath(), JSON.stringify(s, null, 2), "utf8");
}

function getConfigPath() {
  return path.join(app.getPath("userData"), CONFIG_FILE);
}

function hasBundledPythonRuntime() {
  if (process.platform !== "win32") {
    return false;
  }
  try {
    return fs.existsSync(path.join(process.resourcesPath, "python-runtime", "python.exe"));
  } catch (_e) {
    return false;
  }
}

function getAppAsarUnpackRoot() {
  return path.join(process.resourcesPath, "app.asar.unpacked");
}

function defaultConfigForNewInstall() {
  const o = JSON.parse(JSON.stringify(DEFAULT_CONFIG));
  if (hasBundledPythonRuntime()) {
    o.api = { ...o.api, endpoint: `http://127.0.0.1:${LOCAL_ASR_PORT}/asr/transcribe` };
  }
  return o;
}

function endpointTargetsEmbeddedLocalAsr(endpoint) {
  const ep = typeof endpoint === "string" ? endpoint : "";
  return (
    new RegExp(`127\\.0\\.0\\.1:${LOCAL_ASR_PORT}|localhost:${LOCAL_ASR_PORT}`, "i").test(ep) &&
    /asr\/transcribe/i.test(ep)
  );
}

function stopEmbeddedLocalAsr() {
  if (localAsrProcess && !localAsrProcess.killed) {
    try {
      localAsrProcess.kill();
    } catch (_e) {
      /* ignore */
    }
  }
  localAsrProcess = null;
}

function startEmbeddedLocalAsrIfNeeded() {
  stopEmbeddedLocalAsr();
  if (!hasBundledPythonRuntime()) {
    return;
  }
  let cfg;
  try {
    cfg = loadConfig();
  } catch (_e) {
    return;
  }
  const ep = cfg.api && cfg.api.endpoint;
  if (!endpointTargetsEmbeddedLocalAsr(ep)) {
    return;
  }
  const py = path.join(process.resourcesPath, "python-runtime", "python.exe");
  const script = path.join(getAppAsarUnpackRoot(), "python", "local_asr_server.py");
  if (!fs.existsSync(script)) {
    console.warn("[Vi-VibeVoice] Embedded local ASR script missing:", script);
    return;
  }
  const cwd = getAppAsarUnpackRoot();
  const env = {
    ...process.env,
    HF_HUB_DISABLE_SYMLINKS: "1",
    PYTHONUTF8: "1"
  };
  try {
    localAsrProcess = spawn(
      py,
      [script, "--host", "127.0.0.1", "--port", String(LOCAL_ASR_PORT)],
      {
        cwd,
        env,
        stdio: "ignore",
        windowsHide: true
      }
    );
    localAsrProcess.on("error", (err) => {
      console.error("[Vi-VibeVoice] Embedded local ASR failed to start:", err.message);
    });
    localAsrProcess.on("exit", (code, signal) => {
      if (code !== 0 && code != null) {
        console.warn("[Vi-VibeVoice] Embedded local ASR exited", code, signal || "");
      }
      localAsrProcess = null;
    });
    console.log(`[Vi-VibeVoice] Embedded local ASR server (127.0.0.1:${LOCAL_ASR_PORT})`);
  } catch (e) {
    console.error("[Vi-VibeVoice] Embedded local ASR spawn error:", e);
  }
}

function syncEmbeddedLocalAsrToSavedConfig(savedConfig) {
  const ep = savedConfig && savedConfig.api ? savedConfig.api.endpoint : "";
  if (hasBundledPythonRuntime() && endpointTargetsEmbeddedLocalAsr(ep)) {
    startEmbeddedLocalAsrIfNeeded();
  } else {
    stopEmbeddedLocalAsr();
  }
}

function ensureUserConfig() {
  const configPath = getConfigPath();
  const dirPath = path.dirname(configPath);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
  if (!fs.existsSync(configPath)) {
    fs.writeFileSync(configPath, JSON.stringify(defaultConfigForNewInstall(), null, 2), "utf8");
  }
}

function deepMerge(base, override) {
  if (Array.isArray(base)) {
    return Array.isArray(override) ? override : base;
  }
  if (typeof base !== "object" || base === null) {
    return override === undefined ? base : override;
  }
  const result = { ...base };
  for (const [key, value] of Object.entries(override || {})) {
    result[key] = key in base ? deepMerge(base[key], value) : value;
  }
  return result;
}

function normalizeSlimConfig(merged) {
  if (merged && typeof merged === "object") {
    delete merged.wakeWord;
    merged.asr = { mode: "remote" };
  }
  return merged;
}

function loadConfig() {
  ensureUserConfig();
  try {
    const raw = fs.readFileSync(getConfigPath(), "utf8");
    const merged = normalizeSlimConfig(deepMerge(DEFAULT_CONFIG, JSON.parse(raw)));
    const tr = merged.shortcuts && merged.shortcuts.toggleRecording;
    if (process.platform === "win32" && typeof tr === "string") {
      const n = tr.replace(/\s+/g, "").toLowerCase();
      const obsolete =
        UNSUPPORTED_WIN_SUPER_HINT.test(n) ||
        n === "commandorcontrol+shift+space" ||
        n === "control+shift+space";
      if (obsolete) {
        merged.shortcuts.toggleRecording = "Control+Backquote";
        saveConfig(merged);
      }
    }
    return merged;
  } catch (error) {
    return DEFAULT_CONFIG;
  }
}

function saveConfig(nextConfig) {
  const merged = normalizeSlimConfig(deepMerge(DEFAULT_CONFIG, nextConfig));
  fs.writeFileSync(getConfigPath(), JSON.stringify(merged, null, 2), "utf8");
  return merged;
}

function sendToRenderer(channel, payload = {}) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

function registerShortcut(shortcut) {
  const requested = (shortcut || DEFAULT_SHORTCUT).trim() || DEFAULT_SHORTCUT;
  globalShortcut.unregisterAll();
  const handler = () => {
    sendToRenderer("shortcut:toggle-recording", { source: "shortcut" });
  };
  const candidates = [requested, ...FALLBACK_SHORTCUTS.filter((a) => a !== requested)];
  let active = requested;
  let ok = false;
  for (const accel of candidates) {
    try {
      if (globalShortcut.register(accel, handler)) {
        ok = true;
        active = accel;
        break;
      }
    } catch (_e) {
      /* try next */
    }
    console.warn(`[Vi-VibeVoice] globalShortcut register failed: ${accel}`);
  }
  if (!ok) {
    console.error("[Vi-VibeVoice] No global shortcut registered; use the Record button or fix shortcut in Settings.");
  }
  lastShortcutRegistration = {
    ok,
    requested,
    active: ok ? active : null
  };
  currentShortcut = ok ? active : requested;
  if (ok && active !== requested) {
    try {
      const cfg = loadConfig();
      if (cfg.shortcuts && cfg.shortcuts.toggleRecording !== active) {
        saveConfig({ ...cfg, shortcuts: { ...cfg.shortcuts, toggleRecording: active } });
      }
    } catch (_e) {
      /* ignore */
    }
  }
  return currentShortcut;
}

function sendPasteShortcut() {
  if (process.platform === "win32") {
    execFileSync(
      "powershell.exe",
      [
        "-NoProfile",
        "-STA",
        "-WindowStyle",
        "Hidden",
        "-Command",
        "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^v')"
      ],
      { windowsHide: true, timeout: 8000, stdio: "pipe" }
    );
    return;
  }
  if (process.platform === "darwin") {
    execFileSync(
      "osascript",
      ["-e", 'tell application "System Events" to keystroke "v" using command down'],
      { timeout: 8000, stdio: "pipe" }
    );
    return;
  }
  execFileSync("xdotool", ["key", "ctrl+v"], { timeout: 5000, stdio: "pipe" });
}

function createWindow() {
  nativeTheme.themeSource = "light";
  mainWindow = new BrowserWindow({
    width: 1040,
    height: 700,
    minWidth: 880,
    minHeight: 560,
    backgroundColor: "#081521",
    title: "Vi-VibeVoice",
    icon: getAppWindowIcon(),
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.setMenuBarVisibility(false);
  mainWindow.loadFile(path.join(app.getAppPath(), "src", "index.html"));
}

function formatLogDetail(detail) {
  if (detail === undefined) {
    return "";
  }
  if (typeof detail === "string") {
    return detail.length > 6000 ? `${detail.slice(0, 6000)}…(truncated)` : detail;
  }
  try {
    const s = JSON.stringify(detail, null, 2);
    return s.length > 6000 ? `${s.slice(0, 6000)}…(truncated)` : s;
  } catch (_e) {
    return String(detail);
  }
}

ipcMain.handle("output:copy-text", (_event, text) => {
  if (typeof text !== "string") {
    return { ok: false, error: "Not a string" };
  }
  clipboard.writeText(text);
  return { ok: true };
});

ipcMain.handle("output:paste-text-to-foreground", async (_event, payload) => {
  const text = payload && typeof payload.text === "string" ? payload.text : "";
  const options = payload && payload.options ? payload.options : {};
  const minimizeFirst = options.minimizeFirst !== false;
  const delayMs = Math.min(Math.max(Number(options.delayMs) || 800, 0), 5000);

  if (!text) {
    return { ok: false, error: "No text to paste" };
  }

  clipboard.writeText(text);

  if (minimizeFirst && mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.minimize();
  }

  await new Promise((resolve) => {
    setTimeout(resolve, delayMs);
  });

  try {
    sendPasteShortcut();
    return { ok: true };
  } catch (err) {
    const hint =
      process.platform === "linux"
        ? " (Linux: install xdotool, or paste manually with Ctrl+V)"
        : process.platform === "darwin"
          ? " (macOS: grant Accessibility to the app running Electron)"
          : "";
    return { ok: false, error: `${err.message || String(err)}${hint}` };
  }
});

ipcMain.handle("app:debug-log", (_event, payload) => {
  const scope = payload && payload.scope ? String(payload.scope) : "app";
  const message = payload && payload.message != null ? String(payload.message) : "";
  const detail = payload ? payload.detail : undefined;
  const prefix = `[Vi-VibeVoice:${scope}]`;
  if (detail !== undefined) {
    console.log(prefix, message, formatLogDetail(detail));
  } else {
    console.log(prefix, message);
  }
  return true;
});

ipcMain.handle("app:get-config", () => {
  const config = loadConfig();
  return {
    config,
    configPath: getConfigPath(),
    platform: os.platform(),
    appVersion: app.getVersion(),
    shortcut: currentShortcut,
    shortcutRegistration: lastShortcutRegistration,
    stats: loadStats(),
    bundledPythonRuntime: hasBundledPythonRuntime(),
    embeddedLocalAsrPort: LOCAL_ASR_PORT
  };
});

ipcMain.handle("app:bundled-python", () => ({
  available: hasBundledPythonRuntime(),
  defaultLocalUrl: hasBundledPythonRuntime()
    ? `http://127.0.0.1:${LOCAL_ASR_PORT}/asr/transcribe`
    : ""
}));

ipcMain.handle("app:save-config", (_event, nextConfig) => {
  const config = saveConfig(nextConfig);
  const shortcut = registerShortcut(config.shortcuts.toggleRecording);
  syncEmbeddedLocalAsrToSavedConfig(config);
  return {
    config,
    shortcut,
    shortcutRegistration: lastShortcutRegistration
  };
});

ipcMain.handle("stats:get", () => loadStats());

ipcMain.handle("stats:add", (_event, payload) => {
  const text = payload && typeof payload.text === "string" ? payload.text : "";
  const words = text.trim() ? text.trim().split(/\s+/).filter(Boolean).length : 0;
  const s = loadStats();
  const d = todayKey();
  if (s.todayKey !== d) {
    s.todayKey = d;
    s.todayWords = 0;
    s.todayUses = 0;
  }
  s.todayWords += words;
  s.todayUses += 1;
  s.totalWords += words;
  s.totalUses += 1;
  saveStats(s);
  return s;
});

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);
  ensureUserConfig();
  if (process.platform === "darwin" && app.dock) {
    const dockIcon = path.join(app.getAppPath(), "assets", "icon.png");
    if (fs.existsSync(dockIcon)) {
      try {
        app.dock.setIcon(dockIcon);
      } catch (_e) {
        /* optional */
      }
    }
  }
  createWindow();
  const config = loadConfig();
  registerShortcut(config.shortcuts.toggleRecording);
  startEmbeddedLocalAsrIfNeeded();

  try {
    screen.on("display-metrics-changed", () => {
      if (recordingHudWindow && !recordingHudWindow.isDestroyed()) {
        repositionRecordingHud(recordingHudWindow);
      }
    });
  } catch (_e) {
    /* optional */
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("will-quit", () => {
  stopEmbeddedLocalAsr();
  globalShortcut.unregisterAll();
  destroyRecordingHudWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
