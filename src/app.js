function getDefaultToggleShortcut() {
  return "Control+Backquote";
}

const state = {
  config: null,
  configPath: "",
  platform: "",
  transcripts: [],
  mediaRecorder: null,
  chunks: [],
  audioStream: null,
  analyser: null,
  audioContext: null,
  levelTimer: null,
  isRecording: false,
  recordingStopReason: "manual"
};

const els = {};

function cacheElements() {
  const ids = [
    "recordButton",
    "stopButton",
    "copyButton",
    "clearButton",
    "saveSettingsButton",
    "recordingStatus",
    "shortcutLabel",
    "shortcutKbdHome",
    "levelLabel",
    "levelBar",
    "transcriptList",
    "transcriptItemTemplate",
    "apiEndpoint",
    "apiKey",
    "apiMethod",
    "audioField",
    "responseTextPath",
    "headersJson",
    "extraFormDataJson",
    "shortcutInput",
    "shortcutRegisterHint",
    "copyAsrToClipboard",
    "pasteToForegroundAfterAsr",
    "pasteDelayMs",
    "minimizeBeforePaste",
    "appVersionLabel",
    "statTodayWords",
    "statTodayUses",
    "statTotalWords",
    "statTotalUses",
    "uiRecordingHud",
    "settingsSaveFeedback",
    "embeddedAsrHint"
  ];
  for (const id of ids) {
    els[id] = document.getElementById(id);
  }
}

function formatShortcutDisplay(s) {
  if (!s) {
    return "";
  }
  let t = s
    .replace(/CommandOrControl/gi, "Ctrl")
    .replace(/\bControl\b/gi, "Ctrl")
    .replace(/Command/gi, "Cmd");
  const superLabel =
    state.platform === "darwin" ? "Cmd" : state.platform === "win32" ? "Win" : "Super";
  t = t.replace(/\bSuper\b/gi, superLabel);
  t = t.replace(/\bBackquote\b/gi, "`");
  t = t.replace(/\bSemicolon\b/gi, ";");
  t = t.replace(/\bPeriod\b/gi, ".");
  t = t.replace(/\bSlash\b/gi, "/");
  return t.replace(/\+/g, " + ");
}

function showView(name) {
  document.querySelectorAll(".view").forEach((v) => {
    const active = v.dataset.view === name;
    v.classList.toggle("is-active", active);
    v.hidden = !active;
  });
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.view === name);
  });
}

function installNavigation() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });
}

function renderStats(s) {
  if (!s) {
    return;
  }
  els.statTodayWords.textContent = String(s.todayWords ?? 0);
  els.statTodayUses.textContent = String(s.todayUses ?? 0);
  els.statTotalWords.textContent = String(s.totalWords ?? 0);
  els.statTotalUses.textContent = String(s.totalUses ?? 0);
}

function getEffectiveAsrEndpoint() {
  return (state.config.api.endpoint || "").trim();
}

function buildRequestHeaders() {
  const raw = safeJsonParse(state.config.api.headersJson, {});
  const headers =
    raw && typeof raw === "object" && !Array.isArray(raw) ? { ...raw } : {};
  const key = typeof state.config.api.apiKey === "string" ? state.config.api.apiKey.trim() : "";
  const hasAuth = Object.keys(headers).some(
    (k) => k.toLowerCase() === "authorization" && String(headers[k]).trim()
  );
  if (key && !hasAuth) {
    headers.Authorization = `Bearer ${key}`;
  }
  return headers;
}

function setRecordingStatus(text) {
  els.recordingStatus.textContent = text;
}

function syncRecordingHud(payload) {
  if (state.config?.ui?.recordingHud === false) {
    return;
  }
  if (!window.viVibeVoice.updateRecordingHud) {
    return;
  }
  window.viVibeVoice.updateRecordingHud(payload).catch(() => {});
}

function updateControls() {
  els.recordButton.disabled = state.isRecording;
  els.stopButton.disabled = !state.isRecording;
}

function safeJsonParse(value, fallback) {
  if (!value || !value.trim()) {
    return fallback;
  }
  return JSON.parse(value);
}

function getByPath(target, pathStr) {
  if (!pathStr) {
    return target;
  }
  return pathStr.split(".").reduce((acc, key) => (acc == null ? undefined : acc[key]), target);
}

function formatNow() {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "short",
    timeStyle: "medium"
  }).format(new Date());
}

function roundMs(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n)) {
    return "—";
  }
  return n.toFixed(1);
}

function extractServerTiming(payload, responseTextPath) {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  if (payload.timing && typeof payload.timing === "object") {
    return payload.timing;
  }
  const pathStr = (responseTextPath || "text").trim();
  if (!pathStr.includes(".")) {
    return null;
  }
  const parts = pathStr.split(".").filter(Boolean);
  parts.pop();
  if (parts.length === 0) {
    return null;
  }
  const parent = getByPath(payload, parts.join("."));
  if (parent && typeof parent === "object" && parent.timing && typeof parent.timing === "object") {
    return parent.timing;
  }
  return null;
}

/**
 * Log for History / export: end-to-end client steps + optional server breakdown (local ASR).
 */
function buildTranscriptTimingLog(client, serverTiming) {
  const lines = [
    "⏱ Đánh giá tốc độ (ms)",
    `Máy khách — tổng tới khi có text: ${roundMs(client.totalMs)} ms`,
    `  · Chuẩn bị FormData & bắt đầu gửi: ${roundMs(client.prepareFormMs)} ms`,
    `  · HTTP (upload + chờ phản hồi): ${roundMs(client.httpMs)} ms`,
    `  · Parse JSON & lấy text: ${roundMs(client.parseMs)} ms`
  ];
  if (serverTiming && typeof serverTiming === "object") {
    const s = serverTiming;
    const hasServer =
      s.server_total_ms != null ||
      s.read_upload_ms != null ||
      s.asr_ms != null ||
      s.capu_ms != null;
    if (hasServer) {
      lines.push("Máy chủ ASR (chi tiết — khi API trả về timing):");
      if (s.server_total_ms != null) {
        lines.push(`  · Tổng phía server: ${roundMs(s.server_total_ms)} ms`);
      }
      if (s.read_upload_ms != null) {
        lines.push(`  · Đọc body upload: ${roundMs(s.read_upload_ms)} ms`);
      }
      if (s.asr_ms != null) {
        lines.push(`  · ASR (gipformer): ${roundMs(s.asr_ms)} ms`);
      }
      if (s.capu_ms != null) {
        lines.push(`  · CAPU / hậu xử lý: ${roundMs(s.capu_ms)} ms`);
      }
    }
  }
  lines.push(
    "Ghi chú: thời gian HTTP ở máy khách gồm xử lý trên server + độ trễ mạng (loopback thường rất nhỏ)."
  );
  return lines.join("\n");
}

function renderTranscripts() {
  els.transcriptList.innerHTML = "";

  if (state.transcripts.length === 0) {
    const empty = document.createElement("div");
    empty.className = "note-card";
    empty.innerHTML =
      "<h3>No transcripts yet</h3><p>Record from Home — transcripts will show up here.</p>";
    els.transcriptList.appendChild(empty);
    return;
  }

  for (const item of state.transcripts) {
    const fragment = els.transcriptItemTemplate.content.cloneNode(true);
    fragment.querySelector(".transcript-title").textContent = item.title;
    fragment.querySelector(".transcript-meta").textContent = item.meta;
    fragment.querySelector(".transcript-tag").textContent = item.tag;
    fragment.querySelector(".transcript-content").textContent = item.content;
    const timingEl = fragment.querySelector(".transcript-timing");
    if (timingEl) {
      if (item.timingLog) {
        timingEl.textContent = item.timingLog;
        timingEl.hidden = false;
      } else {
        timingEl.remove();
      }
    }
    const actions = fragment.querySelector(".transcript-actions");
    const pasteBtn = fragment.querySelector(".paste-foreground-button");
    if (item.pasteable) {
      pasteBtn.addEventListener("click", () => {
        pasteIntoForegroundApp(item.content);
      });
    } else if (actions) {
      actions.remove();
    }
    els.transcriptList.appendChild(fragment);
  }
}

function addTranscript(item) {
  state.transcripts.unshift(item);
  renderTranscripts();
}

function updateShortcutRegisterHint(payload) {
  const sr = payload.shortcutRegistration;
  const el = els.shortcutRegisterHint;
  if (!el || !sr) {
    return;
  }
  if (!sr.requested) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  if (!sr.ok) {
    el.hidden = false;
    el.classList.add("shortcut-register-hint-warn");
    el.textContent =
      "No global shortcut is active. Another app may already use the same key combination. Quit that app or pick a different two-key shortcut below (e.g. Control+Semicolon).";
    return;
  }
  if (sr.active && sr.requested !== sr.active) {
    el.hidden = false;
    el.classList.remove("shortcut-register-hint-warn");
    const got = formatShortcutDisplay(sr.active);
    const wanted = formatShortcutDisplay(sr.requested);
    el.textContent = `Active: ${got}. "${wanted}" was not available — usually another program already registered the same combination. Close that app (or exit VibeVoice) if you need "${wanted}".`;
    return;
  }
  el.hidden = true;
  el.textContent = "";
  el.classList.remove("shortcut-register-hint-warn");
}

function fillSettings(payload) {
  const { config, shortcut, stats } = payload;
  state.config = config;

  els.apiEndpoint.value = config.api.endpoint || "";
  els.apiKey.value = config.api.apiKey != null ? config.api.apiKey : "";
  els.apiMethod.value = config.api.method || "POST";
  els.audioField.value = config.api.audioField || "audio";
  els.responseTextPath.value = config.api.responseTextPath || "text";
  els.headersJson.value = config.api.headersJson || "{}";
  els.extraFormDataJson.value = config.api.extraFormDataJson || "{}";
  const effectiveShortcut = shortcut || config.shortcuts.toggleRecording;
  els.shortcutInput.value = effectiveShortcut;

  const shortDisp = formatShortcutDisplay(shortcut || config.shortcuts.toggleRecording);
  els.shortcutLabel.textContent = shortDisp;
  els.shortcutKbdHome.textContent = shortDisp;

  const out = config.output || {};
  els.copyAsrToClipboard.checked = out.copyAsrToClipboard !== false;
  els.pasteToForegroundAfterAsr.checked = out.pasteToForegroundAfterAsr !== false;
  els.minimizeBeforePaste.checked = out.minimizeBeforePaste !== false;
  els.pasteDelayMs.value = String(out.pasteDelayMs != null ? out.pasteDelayMs : 800);

  const ui = config.ui || {};
  els.uiRecordingHud.checked = ui.recordingHud !== false;

  els.appVersionLabel.textContent = `v${payload.appVersion || "0.1.0"}`;
  if (stats) {
    renderStats(stats);
  }
  updateShortcutRegisterHint(payload);

  if (els.embeddedAsrHint) {
    if (payload.bundledPythonRuntime) {
      const port = payload.embeddedLocalAsrPort != null ? payload.embeddedLocalAsrPort : 18765;
      els.embeddedAsrHint.hidden = false;
      els.embeddedAsrHint.textContent =
        `This build includes embedded Python. Point the URL to http://127.0.0.1:${port}/asr/transcribe to run on-device ASR; the app starts that server automatically when this URL is saved.`;
    } else {
      els.embeddedAsrHint.hidden = true;
      els.embeddedAsrHint.textContent = "";
    }
  }
}

function collectSettings() {
  return {
    api: {
      endpoint: els.apiEndpoint.value.trim(),
      apiKey: els.apiKey.value.trim(),
      method: els.apiMethod.value,
      audioField: els.audioField.value.trim() || "audio",
      responseTextPath: els.responseTextPath.value.trim() || "text",
      requestTimeoutMs: state.config.api.requestTimeoutMs,
      headersJson: els.headersJson.value.trim() || "{}",
      extraFormDataJson: els.extraFormDataJson.value.trim() || "{}"
    },
    recording: state.config.recording,
    shortcuts: {
      toggleRecording: els.shortcutInput.value.trim() || getDefaultToggleShortcut()
    },
    output: {
      ...(state.config.output || {}),
      copyAsrToClipboard: els.copyAsrToClipboard.checked,
      pasteToForegroundAfterAsr: els.pasteToForegroundAfterAsr.checked,
      minimizeBeforePaste: els.minimizeBeforePaste.checked,
      pasteDelayMs: (() => {
        const n = Number(els.pasteDelayMs.value);
        return Number.isFinite(n) ? Math.min(Math.max(n, 0), 5000) : 800;
      })()
    },
    asr: {
      mode: "remote"
    },
    ui: {
      ...(state.config.ui || {}),
      recordingHud: els.uiRecordingHud.checked
    }
  };
}

let settingsSaveFeedbackTimer = null;

function showSettingsSaveFeedback(kind, text, autoHideMs = 0) {
  const el = els.settingsSaveFeedback;
  if (!el) {
    return;
  }
  if (settingsSaveFeedbackTimer) {
    clearTimeout(settingsSaveFeedbackTimer);
    settingsSaveFeedbackTimer = null;
  }
  el.hidden = false;
  el.textContent = text;
  el.className = `settings-save-feedback ${kind}`;
  if (autoHideMs > 0) {
    settingsSaveFeedbackTimer = setTimeout(() => {
      el.hidden = true;
      el.textContent = "";
      el.className = "settings-save-feedback";
      settingsSaveFeedbackTimer = null;
    }, autoHideMs);
  }
}

async function saveSettings() {
  try {
    if (settingsSaveFeedbackTimer) {
      clearTimeout(settingsSaveFeedbackTimer);
      settingsSaveFeedbackTimer = null;
    }
    els.saveSettingsButton.disabled = true;
    showSettingsSaveFeedback("working", "Đang lưu…", 0);

    const nextConfig = collectSettings();
    const payload = await window.viVibeVoice.saveConfig(nextConfig);
    fillSettings(payload);
    if (payload.config?.ui?.recordingHud === false && window.viVibeVoice.updateRecordingHud) {
      window.viVibeVoice.updateRecordingHud({ mode: "off" }).catch(() => {});
    }

    showSettingsSaveFeedback("ok", "Đã lưu cài đặt.", 5500);

    const content = `Vietnamese ASR\nEndpoint: ${getEffectiveAsrEndpoint()}`;
    addTranscript({
      title: "Settings updated",
      meta: `${formatNow()} | saved`,
      tag: "config",
      content
    });
  } catch (error) {
    showSettingsSaveFeedback("err", `Không lưu được: ${error.message}`, 9000);
    addTranscript({
      title: "Settings error",
      meta: formatNow(),
      tag: "error",
      content: error.message
    });
  } finally {
    els.saveSettingsButton.disabled = false;
  }
}

function stopLevelMonitor() {
  if (state.levelTimer) {
    cancelAnimationFrame(state.levelTimer);
    state.levelTimer = null;
  }
}

function monitorInputLevel() {
  if (!state.analyser) {
    return;
  }

  const buffer = new Uint8Array(state.analyser.fftSize);

  const tick = () => {
    if (!state.analyser) {
      return;
    }

    state.analyser.getByteTimeDomainData(buffer);
    let sumSquares = 0;
    for (const sample of buffer) {
      const centered = (sample - 128) / 128;
      sumSquares += centered * centered;
    }
    const rms = Math.sqrt(sumSquares / buffer.length);
    const percent = Math.min(100, Math.round(rms * 280));
    els.levelBar.style.width = `${percent}%`;
    els.levelLabel.textContent = `${percent}%`;

    state.levelTimer = requestAnimationFrame(tick);
  };

  stopLevelMonitor();
  tick();
}

async function ensureAudioStream() {
  if (state.audioStream) {
    return state.audioStream;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: state.config.recording.channelCount || 1,
      sampleRate: state.config.recording.sampleRate || 16000,
      noiseSuppression: true,
      echoCancellation: true
    }
  });

  state.audioStream = stream;
  state.audioContext = new AudioContext();
  if (state.audioContext.state === "suspended") {
    await state.audioContext.resume();
  }
  const source = state.audioContext.createMediaStreamSource(stream);
  state.analyser = state.audioContext.createAnalyser();
  state.analyser.fftSize = 2048;
  source.connect(state.analyser);
  monitorInputLevel();
  return stream;
}

async function releaseAudioCapture() {
  stopLevelMonitor();
  state.analyser = null;
  if (state.audioContext) {
    try {
      await state.audioContext.close();
    } catch (_e) {
      /* ignore */
    }
    state.audioContext = null;
  }
  if (state.audioStream) {
    try {
      state.audioStream.getTracks().forEach((t) => t.stop());
    } catch (_e) {
      /* ignore */
    }
    state.audioStream = null;
  }
  if (els.levelBar) {
    els.levelBar.style.width = "0%";
  }
  if (els.levelLabel) {
    els.levelLabel.textContent = "—";
  }
}

async function startRecording() {
  if (state.isRecording) {
    return;
  }

  try {
    const stream = await ensureAudioStream();
    state.chunks = [];
    state.recordingStopReason = "manual";

    const mimeType = MediaRecorder.isTypeSupported(state.config.recording.mimeType)
      ? state.config.recording.mimeType
      : "audio/webm";

    state.mediaRecorder = new MediaRecorder(stream, { mimeType });
    state.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        state.chunks.push(event.data);
      }
    };
    state.mediaRecorder.onstop = async () => {
      const blob = new Blob(state.chunks, { type: state.mediaRecorder.mimeType });
      const stopReason = state.recordingStopReason || "manual";
      state.recordingStopReason = "manual";
      try {
        await transcribeBlob(blob, stopReason);
      } finally {
        await releaseAudioCapture();
        state.mediaRecorder = null;
        state.chunks = [];
      }
    };

    state.mediaRecorder.start(250);
    state.isRecording = true;
    updateControls();
    setRecordingStatus("Recording…");
    syncRecordingHud({ mode: "recording" });
  } catch (error) {
    setRecordingStatus("Could not open microphone");
    syncRecordingHud({
      mode: "error",
      detail: error.message || "Could not open microphone",
      dismissMs: 3400
    });
    addTranscript({
      title: "Microphone error",
      meta: formatNow(),
      tag: "error",
      content: error.message
    });
  }
}

function stopRecording(reason = "manual") {
  if (!state.isRecording || !state.mediaRecorder) {
    return;
  }

  state.recordingStopReason = reason;
  state.isRecording = false;
  updateControls();
  const statusMsg = reason === "silence" ? "Sending (silence)…" : "Sending audio…";
  setRecordingStatus(statusMsg);
  syncRecordingHud({
    mode: "processing",
    detail:
      reason === "silence"
        ? "Stopped on silence — sending to ASR…"
        : "Sending to ASR…"
  });
  state.mediaRecorder.stop();
}

async function applyOutputAfterAsr(text, options = {}) {
  const { recordingStoppedViaShortcut = false } = options;
  const out = state.config.output || {};
  if (out.copyAsrToClipboard !== false && window.viVibeVoice.copyText) {
    await window.viVibeVoice.copyText(text);
  }
  if (out.pasteToForegroundAfterAsr && window.viVibeVoice.pasteTextToForeground) {
    const minimizeFirst =
      out.minimizeBeforePaste !== false && !recordingStoppedViaShortcut;
    const res = await window.viVibeVoice.pasteTextToForeground(text, {
      minimizeFirst,
      delayMs: Number(out.pasteDelayMs) || 800
    });
    if (!res.ok) {
      setRecordingStatus("Copied to clipboard — paste manually (Ctrl+V)");
      await logAsr("output", "Auto-paste failed", { error: res.error });
    }
  }
}

async function pasteIntoForegroundApp(text) {
  if (!text || !window.viVibeVoice.pasteTextToForeground) {
    return;
  }
  const out = state.config.output || {};
  try {
    const res = await window.viVibeVoice.pasteTextToForeground(text, {
      minimizeFirst: out.minimizeBeforePaste !== false,
      delayMs: Number(out.pasteDelayMs) || 800
    });
    if (!res.ok) {
      addTranscript({
        title: "Paste to app",
        meta: formatNow(),
        tag: "error",
        content:
          res.error ||
          "Could not send Ctrl+V. Copied to clipboard — focus a field and press Ctrl+V."
      });
    }
  } catch (error) {
    addTranscript({
      title: "Paste to app",
      meta: formatNow(),
      tag: "error",
      content: error.message
    });
  }
}

async function logAsr(scope, message, detail) {
  try {
    if (window.viVibeVoice.logDebug) {
      await window.viVibeVoice.logDebug(scope, message, detail);
    }
  } catch (_e) {
    /* ignore */
  }
}

function transcriptMetaFromStopReason(stopReason) {
  if (stopReason === "shortcut") {
    return `${formatNow()} | stopped with shortcut`;
  }
  if (stopReason === "silence") {
    return `${formatNow()} | auto-stop (silence)`;
  }
  return `${formatNow()} | stopped with button`;
}

async function transcribeBlob(blob, stopReason) {
  const title = "Transcript";
  const tStart = performance.now();
  try {
    const headers = buildRequestHeaders();
    const extraFormData = safeJsonParse(state.config.api.extraFormDataJson, {});
    const endpoint = getEffectiveAsrEndpoint();
    const textPath = state.config.api.responseTextPath || "text";

    if (!endpoint) {
      throw new Error("Set the ASR endpoint URL in Settings, then Save.");
    }

    await logAsr("ASR", "Transcribe started", {
      endpoint,
      mode: "remote",
      method: state.config.api.method || "POST",
      audioField: state.config.api.audioField || "audio",
      responseTextPath: textPath,
      blobBytes: blob.size,
      blobType: blob.type || "",
      stopReason
    });

    if (endpoint && !/\/asr\/transcribe\b/.test(endpoint) && !endpoint.includes("transcribe")) {
      await logAsr("ASR", "Warning: endpoint may need a transcribe path", { endpoint });
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), state.config.api.requestTimeoutMs || 90000);

    const formData = new FormData();
    const extension = blob.type.includes("ogg") ? "ogg" : "webm";
    formData.append(state.config.api.audioField || "audio", blob, `capture.${extension}`);
    for (const [key, value] of Object.entries(extraFormData)) {
      formData.append(key, typeof value === "string" ? value : JSON.stringify(value));
    }

    const tAfterForm = performance.now();

    const response = await fetch(endpoint, {
      method: state.config.api.method || "POST",
      headers,
      body: formData,
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    const rawText = await response.text();
    const tAfterHttp = performance.now();
    await logAsr("ASR", `HTTP ${response.status} ${response.statusText}`, {
      contentType: response.headers.get("content-type") || "",
      bodyPreview: rawText.length > 8000 ? `${rawText.slice(0, 8000)}…` : rawText
    });

    if (!response.ok) {
      throw new Error(`ASR API returned ${response.status} ${response.statusText}`);
    }

    let payload;
    try {
      payload = JSON.parse(rawText);
    } catch (parseErr) {
      await logAsr("ASR", "Response is not valid JSON", { error: parseErr.message });
      throw new Error("ASR response is not valid JSON.");
    }

    const tAfterParse = performance.now();
    const clientTiming = {
      prepareFormMs: tAfterForm - tStart,
      httpMs: tAfterHttp - tAfterForm,
      parseMs: tAfterParse - tAfterHttp,
      totalMs: tAfterParse - tStart
    };

    const transcriptText = getByPath(payload, textPath);
    const serverTiming = extractServerTiming(payload, textPath);
    await logAsr("ASR", "JSON parsed", {
      responseTextPath: textPath,
      extracted: transcriptText != null ? String(transcriptText).slice(0, 2000) : null,
      keysTopLevel: payload && typeof payload === "object" ? Object.keys(payload) : []
    });

    if (!transcriptText && transcriptText !== 0) {
      await logAsr("ASR", "No text at path", { payload });
      throw new Error("No transcript text found in response JSON.");
    }

    const textStr = String(transcriptText).trim();
    const timingLog = buildTranscriptTimingLog(clientTiming, serverTiming);
    await logAsr("ASR", "Pipeline timing (ms)", {
      client: {
        prepareForm: roundMs(clientTiming.prepareFormMs),
        http: roundMs(clientTiming.httpMs),
        parse: roundMs(clientTiming.parseMs),
        total: roundMs(clientTiming.totalMs)
      },
      server: serverTiming
    });
    addTranscript({
      title,
      meta: `${transcriptMetaFromStopReason(stopReason)} | pipeline ${Math.round(clientTiming.totalMs)} ms`,
      tag: "manual",
      content: textStr,
      pasteable: true,
      timingLog
    });
    if (window.viVibeVoice.addStats) {
      try {
        const s = await window.viVibeVoice.addStats({ text: textStr });
        renderStats(s);
      } catch (_e) {
        /* ignore */
      }
    }
    await applyOutputAfterAsr(textStr, {
      recordingStoppedViaShortcut: stopReason === "shortcut"
    });
    syncRecordingHud({
      mode: "success",
      detail: "Text may have been pasted into the focused app",
      dismissMs: 2000
    });
    setRecordingStatus("Ready");
  } catch (error) {
    await logAsr("ASR", "Transcribe error", { message: error.message, name: error.name });
    addTranscript({
      title: "ASR error",
      meta: formatNow(),
      tag: "error",
      content: `${title}: ${error.message}`
    });
    syncRecordingHud({
      mode: "error",
      detail: error.message.slice(0, 180) || "ASR error",
      dismissMs: 4200
    });
    setRecordingStatus("Error while sending audio");
  }
}

function copyAllTranscripts() {
  const text = state.transcripts
    .map((item) => {
      let block = `[${item.meta}] ${item.content}`;
      if (item.timingLog) {
        block += `\n${item.timingLog}`;
      }
      return block;
    })
    .join("\n\n");
  navigator.clipboard.writeText(text || "");
}

function installEvents() {
  els.recordButton.addEventListener("click", () => startRecording());
  els.stopButton.addEventListener("click", () => stopRecording("manual"));
  els.copyButton.addEventListener("click", () => copyAllTranscripts());
  els.clearButton.addEventListener("click", () => {
    state.transcripts = [];
    renderTranscripts();
  });
  els.saveSettingsButton.addEventListener("click", () => saveSettings());

  window.viVibeVoice.onShortcutToggle(() => {
    if (state.isRecording) {
      stopRecording("shortcut");
    } else {
      startRecording();
    }
  });

}

async function bootstrap() {
  cacheElements();
  installNavigation();
  const payload = await window.viVibeVoice.getConfig();
  state.platform = payload.platform || "";
  state.config = payload.config;
  state.configPath = payload.configPath;
  fillSettings(payload);
  renderTranscripts();
  installEvents();
  updateControls();
  showView("home");
}

bootstrap();
