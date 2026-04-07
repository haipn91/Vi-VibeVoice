const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("hudApi", {
  onRender: (cb) => {
    ipcRenderer.on("hud:render", (_event, data) => cb(data));
  }
});
