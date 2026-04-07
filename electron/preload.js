const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("viVibeVoice", {
  copyText: (text) => ipcRenderer.invoke("output:copy-text", text),
  pasteTextToForeground: (text, options) =>
    ipcRenderer.invoke("output:paste-text-to-foreground", { text, options }),
  logDebug: (scope, message, detail) => ipcRenderer.invoke("app:debug-log", { scope, message, detail }),
  getConfig: () => ipcRenderer.invoke("app:get-config"),
  saveConfig: (config) => ipcRenderer.invoke("app:save-config", config),
  getStats: () => ipcRenderer.invoke("stats:get"),
  addStats: (payload) => ipcRenderer.invoke("stats:add", payload),
  onShortcutToggle: (callback) => ipcRenderer.on("shortcut:toggle-recording", (_event, payload) => callback(payload)),
  updateRecordingHud: (payload) => ipcRenderer.invoke("recording-hud:update", payload),
  bundledPythonInfo: () => ipcRenderer.invoke("app:bundled-python")
});
