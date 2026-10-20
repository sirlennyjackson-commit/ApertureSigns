// Aperture Signs web client.
// Camera + MediaPipe run in the browser; only landmark numbers go to the server.
import { HAND_CONNECTIONS, POSE_CONNECTIONS, activeItem, backoff, buildFrame } from "./payload.js";

const VISION = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21";
const MODELS = {
  hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
};
// Same origin by default. For development against another host: ?api=http://localhost:8000
const API = (new URLSearchParams(location.search).get("api") || location.origin).replace(/\/$/, "");
const WS_URL = API.replace(/^http/, "ws") + "/ws/recognize";
const FRAME_MS = 33; // ~30 fps

const $ = (id) => document.getElementById(id);
const stage = $("stage");
const ctx = stage.getContext("2d");
const video = Object.assign(document.createElement("video"), { playsInline: true, muted: true });

let handLm = null;
let poseLm = null;
let stream = null;
let running = false;
let lastTs = 0;
let fps = 0;
let lastFpsPaint = 0;
let tab = "recognize";
let socket = null;
let attempt = 0;
let noModel = false;
let recording = null;
let words = [];

// ---------------------------------------------------------------- helpers --
const setChip = (id, text, cls = "") => {
  const el = $(id);
  el.textContent = text;
  el.className = `chip ${cls}`.trim();
};
const setStageMsg = (text) => { $("stage-msg").textContent = text; };

async function api(path, options = {}) {
  const res = await fetch(API + path, options);
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw Object.assign(new Error(body?.detail || `HTTP ${res.status}`), { status: res.status });
  return body;
}

const friendly = (err) => {
  if (!window.isSecureContext) return "The camera needs HTTPS (or localhost).";
  if (err?.name === "NotAllowedError") return "Camera permission was denied. Allow it in the browser address bar and try again.";
  if (err?.name === "NotFoundError") return "No camera was found.";
  return `Could not start: ${err?.message ?? err}`;
};

// ------------------------------------------------------------------ health --
async function refreshHealth() {
  try {
    const h = await api("/health");
    $("chip-model").textContent = h.model_loaded ? `model: ${h.model_labels} signs` : "model: none yet";
    $("chip-model").className = `chip ${h.model_loaded ? "ok" : "bad"}`;
    noModel = !h.model_loaded;
  } catch {
    setChip("chip-model", "server unreachable", "bad");
  }
}

// ------------------------------------------------------------------ models --
async function loadModels() {
  if (handLm && poseLm) return;
  const { FilesetResolver, HandLandmarker, PoseLandmarker } = await import(`${VISION}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${VISION}/wasm`);
  const make = async (Task, url, extra) => {
    const options = (delegate) => ({ baseOptions: { modelAssetPath: url, delegate }, runningMode: "VIDEO", ...extra });
    try { return await Task.createFromOptions(fileset, options("GPU")); } catch { return Task.createFromOptions(fileset, options("CPU")); }
  };
  handLm = await make(HandLandmarker, MODELS.hand, { numHands: 2 });
  poseLm = await make(PoseLandmarker, MODELS.pose, { numPoses: 1 });
}

async function startCamera() {
  $("btn-camera").disabled = true;
  setStageMsg("Loading hand and pose models… (a few seconds the first time)");
  try {
    await loadModels();
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    running = true;
    setStageMsg("");
    $("btn-camera").textContent = "Stop camera";
    syncSocket();
    requestAnimationFrame(loop);
  } catch (err) {
    setStageMsg(friendly(err));
  } finally {
    $("btn-camera").disabled = false;
  }
}

function stopCamera() {
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  recording = null;
  ctx.clearRect(0, 0, stage.width, stage.height);
  setStageMsg("Camera is off.");
  $("btn-camera").textContent = "Start camera";
  $("fps").textContent = "";
  syncSocket();
}

// -------------------------------------------------------------------- loop --
function loop(ts) {
  if (!running) return;
  requestAnimationFrame(loop);
  if (ts - lastTs < FRAME_MS || video.readyState < 2) return;
  const dt = ts - lastTs;
  lastTs = ts;
  fps = fps ? 0.9 * fps + 0.1 * (1000 / dt) : 1000 / dt;
  if (ts - lastFpsPaint > 500) { $("fps").textContent = `${fps.toFixed(0)} fps`; lastFpsPaint = ts; }

  // Mirror (selfie view) BEFORE detection, exactly like the Python tracker does.
  ctx.save();
  ctx.translate(stage.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, stage.width, stage.height);
  ctx.restore();

  const frame = buildFrame(handLm.detectForVideo(stage, ts), poseLm.detectForVideo(stage, ts));
  drawFrame(frame);
  onFrame(frame);
}

function drawFrame(frame) {
  const w = stage.width;
  const h = stage.height;
  const line = (pts, edges, visible) => {
    ctx.beginPath();
    for (const [a, b] of edges) {
      if (visible && !(visible(pts[a]) && visible(pts[b]))) continue;
      ctx.moveTo(pts[a][0] * w, pts[a][1] * h);
      ctx.lineTo(pts[b][0] * w, pts[b][1] * h);
    }
    ctx.stroke();
  };
  ctx.lineWidth = 3;
  if (frame.pose) { ctx.strokeStyle = "#5cc8ff"; line(frame.pose, POSE_CONNECTIONS, (p) => p[3] >= 0.3); }
  for (const key of ["left", "right"]) {
    if (!frame[key]) continue;
    ctx.strokeStyle = key === "left" ? "#4ade80" : "#fbbf24";
    line(frame[key], HAND_CONNECTIONS);
  }
}

function onFrame(frame) {
  if (recording) {
    recording.frames.push(frame);
    const left = Math.max(0, recording.until - performance.now());
    $("rec-status").textContent = `Recording… ${(left / 1000).toFixed(1)}s`;
    if (left === 0) finishRecording();
  }
  if (tab === "recognize" && socket?.readyState === WebSocket.OPEN && socket.bufferedAmount < 100000) {
    socket.send(JSON.stringify(frame));
  }
}

// --------------------------------------------------------------- websocket --
function syncSocket() {
  const want = running && tab === "recognize" && !noModel;
  if (want && !socket) openSocket();
  else if (!want && socket) closeSocket();
  if (!want && !socket) setChip("chip-ws", running && tab === "recognize" && noModel ? "live: no model" : "live: idle");
}

function openSocket() {
  setChip("chip-ws", "live: connecting…");
  const ws = new WebSocket(WS_URL);
  socket = ws;
  ws.onopen = () => { attempt = 0; };
  ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
  ws.onclose = () => {
    if (socket !== ws) return;
    socket = null;
    setChip("chip-ws", "live: reconnecting…", "bad");
    if (running && tab === "recognize" && !noModel) setTimeout(() => { if (!socket) syncSocket(); }, backoff(attempt++));
  };
}

function closeSocket() {
  const ws = socket;
  socket = null;
  ws?.close();
  setChip("chip-ws", "live: idle");
}

function handleMessage(msg) {
  switch (msg.type) {
    case "ready":
      setChip("chip-ws", `live: ${msg.labels.length} signs`, "ok");
      $("guess-hint").textContent = "Sign clearly; rest your hands between signs.";
      break;
    case "sign":
      words.push(msg.label);
      renderTranscript();
      break;
    case "state":
      $("guess-label").textContent = msg.hands ? (msg.label ?? "…") : "—";
      $("guess-bar").style.width = `${Math.round((msg.hands ? msg.confidence : 0) * 100)}%`;
      if (!msg.hands) $("guess-hint").textContent = "No hands in view.";
      break;
    case "error":
      if (msg.detail === "model_not_loaded") {
        noModel = true;
        $("guess-hint").textContent = "No trained model on the server yet. Use “Record data”, train, then reload the model.";
        syncSocket();
      } else {
        $("guess-hint").textContent = `Server: ${msg.detail}`;
      }
      break;
    default:
  }
}

function renderTranscript() {
  const el = $("transcript");
  el.textContent = words.length ? words.join(" ") : "";
  if (!words.length) el.innerHTML = '<span class="muted">Confirmed signs appear here.</span>';
}

$("btn-clear").onclick = () => {
  words = [];
  renderTranscript();
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "reset" }));
};
$("btn-copy").onclick = () => navigator.clipboard?.writeText(words.join(" "));
$("btn-speak").onclick = () => {
  if (words.length && "speechSynthesis" in window) speechSynthesis.speak(new SpeechSynthesisUtterance(words.join(" ")));
};

// -------------------------------------------------------------- text → sign --
let timeline = [];
let playTimers = [];

async function translate() {
  const text = $("text-in").value.trim();
  if (!text) return;
  $("text-note").textContent = "Translating…";
  try {
    const { translation, avatar } = await api("/pipeline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    timeline = avatar.timeline;
    $("chips").replaceChildren(...timeline.map((t) => {
      const span = document.createElement("span");
      span.className = `gloss${t.type === "fingerspelling" ? " fs" : ""}`;
      span.textContent = t.gloss;
      return span;
    }));
    const unknown = translation.fingerspelled_words.length;
    $("text-note").textContent = timeline.length
      ? `${translation.status}${unknown ? ` · fingerspelled: ${translation.fingerspelled_words.join(", ")}` : ""}. Word-for-word gloss, not full KSL grammar.`
      : "Nothing to sign.";
    $("btn-play").classList.toggle("hidden", !timeline.length);
  } catch (err) {
    $("text-note").textContent = `Error: ${err.message}`;
  }
}

function play() {
  playTimers.forEach(clearTimeout);
  playTimers = [];
  $("player").classList.remove("hidden");
  const chips = [...$("chips").children];
  const show = (i) => {
    chips.forEach((c, k) => c.classList.toggle("on", k === i));
    const item = timeline[i];
    $("player-gloss").textContent = item ? item.gloss : "";
    $("player-letters").textContent = "";
    if (item?.letters) {
      const step = item.duration_ms / item.letters.length;
      item.letters.forEach((letter, k) => {
        playTimers.push(setTimeout(() => { $("player-letters").textContent = item.letters.slice(0, k + 1).join(" "); }, k * step));
      });
    }
  };
  timeline.forEach((t, i) => playTimers.push(setTimeout(() => show(i), t.start_ms)));
  const end = timeline.length ? timeline.at(-1).start_ms + timeline.at(-1).duration_ms : 0;
  playTimers.push(setTimeout(() => show(-1), end));
  void activeItem; // exported helper kept for custom renderers
}

// ------------------------------------------------------------------ dictate --
let recorder = null;
async function toggleMic() {
  const btn = $("btn-mic");
  if (recorder) { recorder.stop(); return; }
  try {
    const audio = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    recorder = new MediaRecorder(audio);
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      audio.getTracks().forEach((t) => t.stop());
      recorder = null;
      btn.setAttribute("aria-pressed", "false");
      btn.textContent = "🎙 Dictate";
      $("text-note").textContent = "Transcribing…";
      try {
        const lang = $("stt-lang").value;
        const { text } = await api(`/stt${lang ? `?language=${lang}` : ""}`, { method: "POST", body: new Blob(chunks) });
        $("text-in").value = text;
        $("text-note").textContent = "";
        translate();
      } catch (err) {
        $("text-note").textContent = err.status === 501
          ? "Offline dictation is not installed on the server (pip install -r requirements-stt.txt). You can still type."
          : `Dictation failed: ${err.message}`;
      }
    };
    recorder.start();
    btn.setAttribute("aria-pressed", "true");
    btn.textContent = "■ Stop";
  } catch (err) {
    $("text-note").textContent = friendly(err);
  }
}

// ------------------------------------------------------------------- record --
const LABEL_RE = /^[A-Za-z0-9_-]{1,40}$/;
const SIGNER_RE = /^[A-Za-z0-9-]{1,40}$/;

async function startRecording() {
  const status = $("rec-status");
  const label = $("rec-label").value.trim();
  const signer = $("rec-signer").value.trim();
  const seconds = Math.min(8, Math.max(0.5, Number($("rec-seconds").value) || 1.5));
  if (!running) return void (status.textContent = "Start the camera first.");
  if (!LABEL_RE.test(label)) return void (status.textContent = "Label: letters, numbers, - and _ only.");
  if (!SIGNER_RE.test(signer)) return void (status.textContent = "Signer id: letters, numbers and - only.");
  if (!$("rec-consent").checked) return void (status.textContent = "Please confirm the signer consents.");
  localStorage.setItem("aperture.signer", signer);
  $("btn-record").disabled = true;
  for (let n = 3; n > 0; n--) {
    status.textContent = `Get ready… ${n}`;
    await new Promise((r) => setTimeout(r, 1000));
    if (!running) { $("btn-record").disabled = false; return; }
  }
  recording = { frames: [], until: performance.now() + seconds * 1000, label, signer };
}

async function finishRecording() {
  const { frames, label, signer } = recording;
  recording = null;
  $("rec-status").textContent = "Saving…";
  try {
    const res = await api("/collect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, signer, consent: true, frames: frames.slice(0, 300) }),
    });
    $("rec-status").textContent = `Saved clip ${res.count} for “${res.label}”.`;
  } catch (err) {
    $("rec-status").textContent = err.status === 403
      ? "Recording is disabled on this server (set APERTURE_ENABLE_COLLECT=1)."
      : `Not saved: ${err.message}`;
  } finally {
    $("btn-record").disabled = false;
    refreshStats();
  }
}

async function refreshStats() {
  const list = $("rec-stats");
  try {
    const { labels, signers } = await api("/collect/stats");
    const rows = Object.entries(labels).map(([k, v]) => {
      const li = document.createElement("li");
      li.textContent = `${k}`;
      const n = document.createElement("span");
      n.textContent = `${v} ${v < 10 ? "(record more)" : ""}`;
      li.append(n);
      return li;
    });
    list.replaceChildren(...rows);
    if (signers.length) $("rec-stats").dataset.signers = signers.join(", ");
  } catch {
    list.replaceChildren();
  }
}

async function reloadModel() {
  try {
    const r = await api("/model/reload", { method: "POST" });
    $("rec-status").textContent = r.model_loaded ? `Model loaded: ${r.labels.length} signs.` : "No model file found. Run the training command first.";
    await refreshHealth();
  } catch (err) {
    $("rec-status").textContent = `Reload failed: ${err.message}`;
  }
}

// --------------------------------------------------------------------- tabs --
function setTab(name) {
  tab = name;
  for (const btn of document.querySelectorAll("[data-tab]")) {
    btn.setAttribute("aria-selected", String(btn.dataset.tab === name));
    $(`tab-${btn.dataset.tab}`).classList.toggle("hidden", btn.dataset.tab !== name);
  }
  syncSocket();
  if (name === "record") refreshStats();
}

document.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
$("btn-camera").onclick = () => (running ? stopCamera() : startCamera());
$("btn-translate").onclick = translate;
$("btn-play").onclick = play;
$("btn-mic").onclick = toggleMic;
$("btn-record").onclick = startRecording;
$("btn-reload").onclick = reloadModel;
$("rec-signer").value = localStorage.getItem("aperture.signer") || "signer1";
$("text-in").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); translate(); } });
refreshHealth();
