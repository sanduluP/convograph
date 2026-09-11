/* Convograph v2 frontend — the event-driven live app.
   A state object fed by SSE events; render functions per panel; inputs
   (live mic / upload / paste) all append to one running session. */

import { createGraph } from "/assets/graph3d.js";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const S = {
  sid: null,
  meta: null,
  state: "idle",         // idle|warming|listening|paused|ending|ended|failed
  turns: [],
  partials: new Map(),   // id -> turn_partial (the sentence being spoken)
  speakers: {},
  speakerOrder: [],
  episodes: new Map(),
  graph: null,
  renders: new Map(),
  latestRenderEp: null,
  recView: "latest",
  board: null,
  renaming: null,
  compare: null,
  view: "live",          // live | compare
  audioSeconds: 0,
  hasLiveAudio: false,
  ingestPending: 0,
  livePhase: "off",      // off | connecting | on
  detail: "",
};

let graph3d = null;
let compareGraph = null;
let ws = null;           // live audio socket
let recorder = null;     // MediaRecorder
let micStream = null;    // getUserMedia stream — survives pause/resume

/* ── boot: a session always exists; the app opens on the live view ────────── */

boot();

async function boot() {
  const params = new URLSearchParams(location.search);
  const existing = params.get("s");
  if (existing) {
    const res = await fetch(`/api/session/${existing}/meta`);
    if (res.ok) {
      S.sid = existing;
      connect();
      render();
      return;
    }
    toast("That session is gone (server restarted) — starting a fresh one.");
  }
  const created = await api("", { method: "POST", body: JSON.stringify({}) });
  S.sid = created.id;
  history.replaceState(null, "", `?s=${S.sid}`);
  connect();
  render();
}

async function api(path, opts = {}) {
  const res = await fetch(`/api/session${path}`, {
    headers: { "Content-Type": "application/json" }, ...opts,
  });
  if (!res.ok) throw new Error((await res.text()).slice(0, 300));
  return res.json();
}

function connect() {
  const es = new EventSource(`/api/session/${S.sid}/events`);
  const on = (type, fn) => es.addEventListener(type, (e) => {
    fn(JSON.parse(e.data)); render();
  });
  on("session", (d) => {
    if (d.state) S.state = d.state;
    if (d.title) S.meta = { ...(S.meta || {}), title: d.title,
                            started_at: d.started_at ?? S.meta?.started_at };
    if (d.started_at) S.meta = { ...(S.meta || {}), started_at: d.started_at };
    if (d.settings) S.meta = { ...(S.meta || {}), settings: d.settings };
    S.detail = d.detail || "";
  });
  on("turn", (d) => {
    S.partials.delete(d.id);           // the final replaces its partial
    S.turns.push(d);
    if (!S.speakerOrder.includes(d.speaker)) S.speakerOrder.push(d.speaker);
    S.speakers[d.speaker] = d.name;
  });
  es.addEventListener("turn_partial", (e) => {   // transient, high-rate
    const d = JSON.parse(e.data);
    S.partials.set(d.id, d);
    if (!S.speakerOrder.includes(d.speaker)) S.speakerOrder.push(d.speaker);
    S.speakers[d.speaker] ??= d.name;
    renderTranscript(); renderTopbar();          // cheap partial re-render
  });
  es.addEventListener("clock", (e) => {
    S.audioSeconds = JSON.parse(e.data).audio_seconds;
    S.hasLiveAudio = true;
    const el = $("#timer"); if (el) el.textContent = fmtSecs(S.audioSeconds);
  });
  on("queue", (d) => { S.ingestPending = d.ingest_pending; });
  on("speaker", (d) => {
    S.speakers[d.label] = d.name;
    S.turns.forEach((t) => { if (t.speaker === d.label) t.name = d.name; });
  });
  on("episode", (d) => {
    const ep = S.episodes.get(d.index) || {};
    ep[d.stage] = d;
    S.episodes.set(d.index, ep);
  });
  on("graph", (d) => { S.graph = d; });
  on("graph_delta", () => {});
  on("render", (d) => {
    S.renders.set(d.episode, d);
    if (d.status === "done" && d.images.length) S.latestRenderEp = d.episode;
  });
  on("board", (d) => { S.board = { ...d, v: (S.board?.v || 0) + 1 }; });
  on("error", (d) => toast(`${d.stage}: ${d.message}`));
  es.onerror = async () => {
    // EventSource auto-reconnects; detect a server restart (session gone)
    try {
      const res = await fetch(`/api/session/${S.sid}/meta`);
      if (res.status === 404) {
        es.close();
        toast("Session lost — the server restarted. Reload to start fresh.");
      }
    } catch { /* network blip; let EventSource retry */ }
  };
}

/* ── live microphone (recording happens in THIS browser) ──────────────────── */

function startRecorder() {
  /* A FRESH MediaRecorder per (re)start: each one emits a complete webm
     header, and the server pairs it with a fresh ffmpeg. Decoding across a
     pause gap in one long stream is what used to kill the chain silently. */
  recorder = new MediaRecorder(micStream, { mimeType: "audio/webm;codecs=opus" });
  recorder.ondataavailable = (e) => {
    if (e.data.size && ws?.readyState === WebSocket.OPEN) ws.send(e.data);
  };
  recorder.start(250);
}

async function startLive() {
  if (S.livePhase !== "off") { stopLiveCapture(); return; }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    toast("Microphone blocked. Live capture needs a localhost or HTTPS "
        + "origin — open the app through your SSH tunnel.");
    return;
  }
  S.livePhase = "connecting";
  render();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/api/session/${S.sid}/live`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    startRecorder();
    S.livePhase = "on";
    render();
  };
  ws.onmessage = (e) => {
    if (typeof e.data === "string") {
      const m = JSON.parse(e.data);
      if (m.error) { toast(m.message || m.error); stopLiveCapture(); }
    }
  };
  ws.onclose = () => {
    if (S.livePhase === "on" && !["ending", "ended"].includes(S.state)) {
      toast("Live connection dropped — reconnecting…");
      stopRecorderOnly();
      S.livePhase = "off";
      setTimeout(() => startLive().catch(() => {}), 1500);
    }
  };
}

function stopRecorderOnly() {
  try { if (recorder && recorder.state !== "inactive") recorder.stop(); } catch {}
  recorder = null;
}

function releaseMic() {
  micStream?.getTracks().forEach((t) => t.stop());
  micStream = null;
}

function stopLiveCapture() {   // user toggled 🎙 off: end the session's audio
  try { ws?.send(JSON.stringify({ type: "stop" })); } catch {}
  stopRecorderOnly();
  releaseMic();
  S.livePhase = "off";
  render();
}

async function pauseOrResume() {
  if (S.state === "listening") {
    // stop (not pause) the recorder: flushes its tail; mic stream stays live
    stopRecorderOnly();
    if (ws?.readyState === WebSocket.OPEN)
      ws.send(JSON.stringify({ type: "pause" }));
    else await api(`/${S.sid}/pause`, { method: "POST" });
  } else if (S.state === "paused") {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "resume" }));  // server restarts ffmpeg
      if (micStream) startRecorder();               // fresh recorder = fresh header
    } else {
      await api(`/${S.sid}/resume`, { method: "POST" });
    }
  }
}

async function endSession() {
  if (S.livePhase !== "off") stopLiveCapture();
  else await api(`/${S.sid}/end`, { method: "POST" });
}

/* ── helpers ──────────────────────────────────────────────────────────────── */

function colorClass(label) {
  const i = S.speakerOrder.indexOf(label);
  return `c${((i < 0 ? 0 : i)) % 4}`;
}
function initial(label) {
  return (S.speakers[label] || label).trim()[0]?.toUpperCase() || "?";
}
function named(label) { return S.speakers[label] !== label; }
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function fmtSecs(s) {
  const p = (n) => String(n).padStart(2, "0");
  s = Math.max(0, Math.floor(s));
  return `${p(Math.floor(s / 3600))}:${p(Math.floor((s % 3600) / 60))}:${p(s % 60)}`;
}
setInterval(() => {
  const el = $("#timer"); if (!el) return;
  if (S.hasLiveAudio) { el.textContent = fmtSecs(S.audioSeconds); return; }
  if (S.meta?.started_at && !["ended", "failed"].includes(S.state))
    el.textContent = fmtSecs(Date.now() / 1000 - S.meta.started_at);
}, 1000);

function toast(msg) {
  const box = $("#toasts");
  const el = document.createElement("div");
  el.className = "toast"; el.textContent = msg;
  box.appendChild(el);
  setTimeout(() => el.remove(), 9000);
}

/* ── render root ──────────────────────────────────────────────────────────── */

function render() {
  const app = $("#app");
  app.classList.toggle("compare-mode", S.view === "compare");
  app.classList.toggle("paused",
    ["paused", "ended", "failed"].includes(S.state));
  renderTopbar();
  $("#view-live").hidden = S.view !== "live";
  $("#view-compare").hidden = S.view !== "compare";
  $("#track").hidden = S.view !== "live";
  if (S.view === "live") {
    renderTranscript(); renderGraphPanel(); renderRecording(); renderTrack();
  } else {
    renderCompare();
  }
}

function renderTopbar() {
  const t = $("#session-title");
  if (t.querySelector("input")) {
    // a rename is in progress — don't clobber the input mid-typing
  } else if (S.meta?.title && S.meta.title !== "Untitled session") {
    t.textContent = S.meta.title; t.classList.remove("placeholder");
  } else {
    t.textContent = "Untitled session — name it when you like";
    t.classList.add("placeholder");
  }
  const label = {
    idle: "Idle — add audio or text on the left",
    warming: "Warming up",
    listening: S.livePhase === "on" ? "Listening" : "Processing",
    paused: "Paused",
    ending: `Ending — ${S.ingestPending} episode(s) queued`,
    ended: "Session complete",
    failed: "Failed",
  }[S.state] || S.state;
  $("#live-label").textContent = label;
  $("#live-detail").textContent = S.detail || "";

  const unnamed = S.speakerOrder.filter((l) => !named(l)).length;
  const tag = $("#ident-tag");
  if (S.speakerOrder.length && unnamed) {
    tag.hidden = false;
    tag.textContent = `${S.speakerOrder.length - unnamed} of `
                    + `${S.speakerOrder.length} voices identified`;
  } else tag.hidden = true;

  $("#stack").innerHTML = S.speakerOrder.slice(0, 4).map((l) =>
    `<span class="monogram ${colorClass(l)} ${named(l) ? "" : "unnamed"}"
       title="${esc(S.speakers[l])}">${esc(initial(l))}</span>`).join("");

  $("#btn-compare").disabled = !(S.meta?.episodes_snapshotted >= 2 ||
                                 countDone("graphed") >= 2);
  const busyEnd = ["ending", "ended", "failed", "idle"].includes(S.state);
  $("#btn-end").disabled = busyEnd && S.state !== "idle" || S.state === "idle";
  $("#btn-pause").disabled = !["listening", "paused"].includes(S.state);
  $("#btn-pause").textContent = S.state === "paused" ? "Resume" : "Pause";
  const liveBtn = $("#btn-live");
  liveBtn.textContent = { off: "🎙 Live", connecting: "⏳ Starting…",
                          on: "■ Stop live" }[S.livePhase];
  liveBtn.disabled = S.livePhase === "connecting" ||
                     ["ending", "ended", "failed"].includes(S.state);
  $("#btn-back").hidden = S.view !== "compare";
  $("#btn-compare").hidden = S.view === "compare";
}

function countDone(stage) {
  let n = 0;
  for (const ep of S.episodes.values())
    if (ep[stage]?.status === "done") n++;
  return n;
}

/* ── transcript panel (finals + live partials) ────────────────────────────── */

function renderTranscript() {
  const body = $("#transcript-body");
  let html = "", lastEp = 0;
  const currentEp = Math.max(0, ...[...S.episodes.keys()]);
  S.turns.forEach((t) => {
    if (t.episode !== lastEp) {
      const cur = t.episode >= currentEp ? " current" : "";
      html += `<div class="ep-boundary${cur}"><span>Episode ${t.episode}</span></div>`;
      lastEp = t.episode;
    }
    const dim = t.episode < currentEp - 1 ? " dim" : "";
    const renaming = S.renaming === t.speaker;
    const nameHtml = renaming
      ? `<input class="rename-input" data-label="${esc(t.speaker)}"
           value="${esc(named(t.speaker) ? t.name : "")}"
           placeholder="${esc(t.speaker)}" />
         <span class="rename-hint">↵ to save</span>`
      : `<span class="name ${named(t.speaker) ? "named" : ""}"
           data-label="${esc(t.speaker)}" title="Click to rename">${esc(t.name)}</span>`;
    const ts = t.t_start != null ? `${t.t_start.toFixed(1)}s` : t.id;
    html += `
      <div class="turn${dim}">
        <span class="monogram ${colorClass(t.speaker)}
          ${named(t.speaker) ? "" : "unnamed"}">${esc(initial(t.speaker))}</span>
        <div>
          <div class="who">${nameHtml}<span class="ts tabular">${esc(ts)}</span></div>
          <div>${esc(t.text)}</div>
        </div>
      </div>`;
  });

  // open partials: the sentences still being spoken (design 1a "live turn")
  const partials = [...S.partials.values()].sort((a, b) => a.t_start - b.t_start);
  for (const p of partials) {
    html += `
      <div class="turn partial">
        <span class="monogram speaking ${colorClass(p.speaker)}
          ${named(p.speaker) ? "" : "unnamed"}">${esc(initial(p.speaker))}</span>
        <div>
          <div class="who">
            <span class="name ${named(p.speaker) ? "named" : ""}"
              data-label="${esc(p.speaker)}">${esc(S.speakers[p.speaker] || p.speaker)}</span>
            <span class="ts speaking">speaking</span>
          </div>
          <div>${esc(p.text)}<span class="caret"></span></div>
        </div>
      </div>`;
  }

  body.innerHTML = `<div class="turns">${html}</div>`;
  $("#word-count").textContent =
    `${S.turns.reduce((n, t) => n + t.text.split(/\s+/).length, 0)} words`;

  if (S.renaming) {
    const input = $(".rename-input", body);
    input?.focus();
    input?.addEventListener("keydown", async (e) => {
      if (e.key === "Enter") {
        const name = input.value.trim();
        S.renaming = null;
        if (name) await api(`/${S.sid}/speaker`, {
          method: "POST",
          body: JSON.stringify({ label: input.dataset.label, name }),
        });
        else render();
      }
      if (e.key === "Escape") { S.renaming = null; render(); }
    });
  } else {
    body.scrollTop = body.scrollHeight;
  }
  $$(".name", body).forEach((el) => el.addEventListener("click", () => {
    S.renaming = el.dataset.label; render();
  }));
}

/* ── knowledge graph panel ────────────────────────────────────────────────── */

function renderGraphPanel() {
  if (!graph3d) graph3d = createGraph($("#graph-svg"));
  if (S.graph) {
    graph3d.update(S.graph.nodes, S.graph.edges,
                   S.graph.node_status, S.graph.edge_status);
    const c = S.graph.counters;
    $("#graph-meta").textContent =
      `${c.facts} facts · ${c.relations} relations` +
      (c.invalidated ? ` · ${c.invalidated} invalidated` : "");
    $("#graph-status").textContent = S.graph.status_line;
  } else {
    $("#graph-meta").textContent = "waiting for the first episode";
    $("#graph-status").textContent =
      S.state === "listening" ? "Extracting episode 1 …" : "";
  }
}

/* ── graphic recording panel ──────────────────────────────────────────────── */

const EXCALIDRAW_VERSION = "0.18.0";
const REACT_VERSION = "18.3.1";
let canvasMountedV = 0;

function excalidrawSrcdoc(boardUrl) {
  return `<!doctype html><html><head><meta charset="utf-8">
  <style>html,body{margin:0;height:100%;background:#fff}#root{height:100%}</style>
  <link rel="stylesheet"
    href="https://unpkg.com/@excalidraw/excalidraw@${EXCALIDRAW_VERSION}/dist/prod/index.css">
  <script>window.EXCALIDRAW_ASSET_PATH =
    "https://esm.sh/@excalidraw/excalidraw@${EXCALIDRAW_VERSION}/dist/prod/";<\/script>
  </head><body><div id="root"></div>
  <script type="module">
    import React from "https://esm.sh/react@${REACT_VERSION}";
    import { createRoot } from "https://esm.sh/react-dom@${REACT_VERSION}/client";
    import { Excalidraw } from
      "https://esm.sh/@excalidraw/excalidraw@${EXCALIDRAW_VERSION}?deps=react@${REACT_VERSION},react-dom@${REACT_VERSION}";
    const scene = await (await fetch("${boardUrl}?t=" + Date.now())).json();
    scene.scrollToContent = true;
    createRoot(document.getElementById("root"))
      .render(React.createElement(Excalidraw, { initialData: scene }));
  <\/script></body></html>`;
}

function renderCanvasView(plate) {
  if (!S.board) {
    plate.innerHTML = `<div class="empty"><h3>No board yet</h3>
      <p>The Excalidraw board is composed after the first episode renders.</p></div>`;
    canvasMountedV = 0;
    return;
  }
  if (canvasMountedV === S.board.v && $("#exc-frame", plate)) return;
  plate.innerHTML = `
    <iframe id="exc-frame" style="position:absolute;inset:0;width:100%;height:100%;
      border:0;background:#fff"></iframe>
    <a class="btn ghost" style="position:absolute;right:10px;top:8px;z-index:5;
      background:#ffffffd0;border-radius:8px"
      href="${S.board.url}" download="convograph_board.excalidraw">Download .excalidraw</a>`;
  $("#exc-frame", plate).srcdoc = excalidrawSrcdoc(S.board.url);
  canvasMountedV = S.board.v;
}

function renderRecording() {
  const plate = $("#plate");
  const chip = $("#status-chip");
  const ep = S.latestRenderEp;
  const running = [...S.renders.values()].find((r) => r.status !== "done");

  if (S.recView === "canvas") {
    renderCanvasView(plate);
    $("#rec-meta").textContent = S.board
      ? `Editable board · ${S.board.elements} elements · ${S.board.episodes} episode(s)`
      : "Editable board";
    chip.hidden = !running;
    return;
  }
  canvasMountedV = 0;

  if (S.recView === "frames") {
    const eps = [...S.renders.entries()].sort((a, b) => a[0] - b[0])
      .filter(([, r]) => r.images?.length);
    plate.innerHTML = eps.length ? `<div class="collage" style="overflow:auto">${
      eps.flatMap(([i, r]) => r.images.map((u, j) => `
        <figure><img src="${u}" alt="">
          <figcaption>ep ${i} · ${esc((r.facts || r.captions)[j] || "")}</figcaption>
        </figure>`)).join("")}</div>`
      : `<div class="empty"><h3>No frames yet</h3></div>`;
    $("#rec-meta").textContent = "All episodes";
    chip.hidden = !running;
    return;
  }

  if (ep != null) {
    const r = S.renders.get(ep);
    plate.innerHTML = `
      <div class="collage">${r.images.map((u, i) => `
        <figure><img src="${u}" alt="">
          <figcaption>${esc((r.facts || r.captions)[i] || "")}</figcaption>
        </figure>`).join("")}</div>
      ${running ? '<div class="sweep"></div>' : ""}`;
    $("#rec-meta").textContent = `Pictogram · FLUX schnell · episode ${ep}`;
  } else {
    plate.innerHTML = `
      <div class="empty">
        <div class="spinner">
          <svg viewBox="0 0 72 72" width="72" height="72" fill="none">
            <circle cx="36" cy="36" r="22" stroke="#c4c4ca" stroke-width="1.5"/>
            <circle cx="36" cy="36" r="30" stroke="#0a66ff" stroke-width="1.5"
              stroke-dasharray="40 150" stroke-linecap="round"/>
          </svg>
        </div>
        <h3>The first plate arrives with episode 1</h3>
        <p>Facts are extracted per episode; each new decision becomes one
           drawn pictogram. Rendering starts as soon as episode 1's graph lands.</p>
      </div>`;
    $("#rec-meta").textContent = "Pictogram · FLUX schnell";
  }

  if (running) {
    chip.hidden = false;
    const pct = Math.round((running.progress || 0) * 100);
    $("#chip-text").textContent = running.status === "captioning"
      ? `Captioning episode ${running.episode}`
      : `Rendering episode ${running.episode}`;
    $("#chip-bar").style.width = `${pct}%`;
    $("#chip-pct").textContent = `${pct}%`;
  } else chip.hidden = true;

  const thumbs = $("#thumbs");
  const eps = [...S.renders.entries()].sort((a, b) => a[0] - b[0]);
  let h = eps.map(([i, r]) => r.images[0]
    ? `<img class="thumb ${i === ep ? "latest" : ""}" src="${r.images[0]}"
         data-ep="${i}" title="episode ${i}">`
    : `<span class="thumb placeholder">${i} …</span>`).join("");
  const next = (eps.at(-1)?.[0] || 0) + 1;
  if (S.state === "listening")
    h += `<span class="thumb placeholder">${next} …</span>`;
  thumbs.innerHTML = h;
  $$("img.thumb", thumbs).forEach((el) => el.addEventListener("click", () => {
    S.latestRenderEp = Number(el.dataset.ep); render();
  }));
}

/* ── episode track ────────────────────────────────────────────────────────── */

function renderTrack() {
  const idxs = [...S.episodes.keys()].sort((a, b) => a - b);
  const done = (st) =>
    idxs.filter((i) => S.episodes.get(i)[st]?.status === "done").length;
  const queued = S.ingestPending
    ? ` · <span style="color:var(--a700)">${S.ingestPending} queued for extraction</span>`
    : "";
  $("#track-summary").innerHTML =
    `<b>${idxs.length}</b> episodes · ${done("graphed")} graphed · `
    + `${done("rendered")} rendered${queued}`;
  $("#lanes").innerHTML = idxs.map((i) => {
    const ep = S.episodes.get(i);
    const bar = (st) => {
      const s = ep[st] || {};
      const cls = s.status === "done" ? "done" : s.status === "running"
        ? "running" : s.status === "failed" ? "failed" : "";
      const w = s.status === "running"
        ? `width:${Math.round((s.progress || 0.2) * 100)}%` : "";
      return `<div class="bar ${s.status ? "" : "dashed"}"
                title="${st}${s.detail ? ": " + esc(s.detail) : ""}">
                <i class="${cls}" style="${w}"></i></div>`;
    };
    const live = ep.graphed?.status === "running" ||
                 ep.rendered?.status === "running";
    return `<div class="ep-cell ${live ? "live" : ""}">
      <div class="bars">${bar("transcribed")}${bar("graphed")}${bar("rendered")}</div>
      <div class="lbl tabular">${i}</div></div>`;
  }).join("");
}

/* ── compare episodes ─────────────────────────────────────────────────────── */

async function openCompare() {
  const n = Math.max(S.meta?.episodes_snapshotted || 0, countDone("graphed"));
  if (n < 2) return;
  S.compare = { a: 1, b: n, n, data: null };
  S.view = "compare";
  await loadDiff();
}

async function loadDiff() {
  const { a, b } = S.compare;
  S.compare.data = await api(`/${S.sid}/diff?a=${a}&b=${b}`);
  render();
}

function renderCompare() {
  const c = S.compare;
  if (!c?.data) return;
  $("#cmp-title").innerHTML =
    `Episode ${c.a} <span class="arrow">→</span> Episode ${c.b}`;
  const s = c.data.summary;
  $("#cmp-summary").innerHTML =
    `<span class="tabular">+${s.added} added · ${s.revised} revised · ` +
    `<s>${s.invalidated}</s> invalidated · ${s.unchanged} unchanged</span>`;

  const track = $("#scrub");
  const pos = (i) => 4 + ((i - 1) / Math.max(1, c.n - 1)) * 92;
  let h = `<div class="line"></div>
    <div class="span" style="left:${pos(c.a)}%; width:${pos(c.b) - pos(c.a)}%"></div>`;
  for (let i = 1; i <= c.n; i++) {
    const cls = i >= c.a && i <= c.b ? "inrange" : "";
    h += `<div class="scrub-dot ${cls}" data-i="${i}" style="left:${pos(i)}%">
            <span class="dlbl tabular">${i}</span></div>`;
  }
  h += `<div class="scrub-handle a" style="left:${pos(c.a)}%" title="A = episode ${c.a}"></div>
        <div class="scrub-handle b" style="left:${pos(c.b)}%" title="B = episode ${c.b}"></div>`;
  track.innerHTML = h;
  $$(".scrub-dot", track).forEach((el) => el.addEventListener("click", async () => {
    const i = Number(el.dataset.i);
    if (Math.abs(i - c.a) <= Math.abs(i - c.b)) c.a = Math.min(i, c.b - 1);
    else c.b = Math.max(i, c.a + 1);
    await loadDiff();
  }));

  const changesOnly = $("#cmp-filter .on")?.dataset.f !== "all";
  const rows = c.data.rows.filter((r) => !changesOnly || r.change !== "unchanged");
  $("#ledger-body").innerHTML = rows.map((r) => `
    <tr>
      <td>${esc(r.fact)}</td>
      <td>${r.at_a == null ? '<span class="absent">—</span>'
            : r.change === "revised" ? `<span class="old">${esc(r.at_a)}</span>`
            : esc(r.at_a)}</td>
      <td>${r.at_b == null ? '<span class="absent">—</span>' : esc(r.at_b)}</td>
      <td><span class="chg ${r.change}">${r.change[0].toUpperCase() + r.change.slice(1)}</span></td>
    </tr>`).join("");

  if (!compareGraph) compareGraph = createGraph($("#cmp-svg"));
  if (S.graph) {
    const changedFacts = new Set(c.data.rows
      .filter((r) => r.change !== "unchanged")
      .flatMap((r) => [r.at_a, r.at_b, r.fact]).filter(Boolean));
    const edgeStatus = {}, nodeStatus = {}, changedNodes = new Set();
    for (const e of S.graph.edges) {
      if (changedFacts.has(e.fact)) {
        edgeStatus[e.id] = e.invalid_at ? "invalidated" : "added";
        changedNodes.add(e.source); changedNodes.add(e.target);
      } else edgeStatus[e.id] = "confirmed";
    }
    for (const n of S.graph.nodes)
      nodeStatus[n.id] = changedNodes.has(n.id)
        ? (S.graph.node_status?.[n.id] === "invalidated" ? "invalidated" : "added")
        : "confirmed";
    compareGraph.update(
      S.graph.nodes.map((n) => changedNodes.has(n.id) ? n : { ...n, label: "" }),
      S.graph.edges, nodeStatus, edgeStatus);
  }
}

/* ── settings sheet ───────────────────────────────────────────────────────── */

function openSheet() {
  const st = S.meta?.settings || {};
  $("#set-episode-turns").value = st.episode_turns ?? 12;
  $("#set-episode-seconds").value = st.episode_seconds ?? 120;
  $("#set-window-lines").value = st.window_lines ?? 5;
  $("#set-max-facts").value = st.max_render_facts ?? 6;
  $("#set-render").checked = st.render !== false;
  $("#sheet").hidden = false; $("#scrim").hidden = false;
  $("#app").classList.add("dimmed");
}
function closeSheet() {
  $("#sheet").hidden = true; $("#scrim").hidden = true;
  $("#app").classList.remove("dimmed");
}
async function saveSheet() {
  const body = {
    episode_turns: Number($("#set-episode-turns").value) || 12,
    episode_seconds: Number($("#set-episode-seconds").value) || 120,
    window_lines: Number($("#set-window-lines").value) || 5,
    max_render_facts: Number($("#set-max-facts").value) || 6,
    render: $("#set-render").checked,
  };
  if (S.sid) {
    const r = await api(`/${S.sid}/settings`, {
      method: "POST", body: JSON.stringify(body) });
    S.meta = { ...(S.meta || {}), settings: r.settings };
  }
  closeSheet();
}

/* ── wiring ───────────────────────────────────────────────────────────────── */

/* ── sessions panel + title rename ────────────────────────────────────────── */

async function openSessionsPanel() {
  const panel = $("#sessions-panel");
  if (!panel.hidden) { panel.hidden = true; return; }
  const res = await fetch("/api/sessions");
  const { sessions } = await res.json();
  $("#sessions-list").innerHTML = sessions.map((s) => {
    const when = new Date(s.started_at * 1000)
      .toLocaleString([], { dateStyle: "short", timeStyle: "short" });
    return `<div class="session-row ${s.id === S.sid ? "current" : ""}"
              data-sid="${s.id}">
      <span class="srow-title">${esc(s.title)}</span>
      <span class="srow-state">${esc(s.state)}</span>
      <span class="srow-meta tabular">${when} · ${s.episodes} episode(s)
        · ${s.id}</span>
    </div>`;
  }).join("") || `<div style="padding:12px 16px;font-size:12px;
    color:var(--n600)">No sessions yet.</div>`;
  panel.hidden = false;
  $$(".session-row", panel).forEach((el) => el.addEventListener("click", () => {
    if (el.dataset.sid !== S.sid) location.href = `?s=${el.dataset.sid}`;
    panel.hidden = true;
  }));
}

function renameTitle() {
  const t = $("#session-title");
  if (t.querySelector("input")) return;
  const current = S.meta?.title && S.meta.title !== "Untitled session"
    ? S.meta.title : "";
  t.innerHTML = `<input class="title-input" value="${esc(current)}"
    placeholder="Session title">`;
  const input = t.querySelector("input");
  input.focus();
  input.addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      const title = input.value.trim();
      if (title) {
        await api(`/${S.sid}/title`, { method: "POST",
                                       body: JSON.stringify({ title }) });
        S.meta = { ...(S.meta || {}), title };
      }
      render();
    }
    if (e.key === "Escape") render();
  });
  input.addEventListener("blur", () => setTimeout(render, 150));
}

$("#btn-sessions").addEventListener("click", () =>
  openSessionsPanel().catch((e) => toast(e.message)));
$("#btn-new-session").addEventListener("click", () => { location.href = "/"; });
$("#session-title").addEventListener("click", renameTitle);
document.addEventListener("click", (e) => {
  const panel = $("#sessions-panel");
  if (!panel.hidden && !panel.contains(e.target) &&
      e.target !== $("#btn-sessions")) panel.hidden = true;
});

$("#btn-live").addEventListener("click", () =>
  startLive().catch((e) => toast(e.message)));
$("#btn-upload").addEventListener("click", () => $("#upload-file").click());
$("#upload-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file, file.name);
  const res = await fetch(`/api/session/${S.sid}/audio`, {
    method: "POST", body: fd });
  if (!res.ok) toast((await res.text()).slice(0, 200));
  e.target.value = "";
});
$("#btn-paste-toggle").addEventListener("click", () => {
  $("#paste-drawer").hidden = !$("#paste-drawer").hidden;
});
$("#btn-paste-go").addEventListener("click", async () => {
  const text = $("#paste-text").value.trim();
  if (!text) return;
  try {
    await api(`/${S.sid}/text`, { method: "POST",
                                  body: JSON.stringify({ text }) });
    $("#paste-text").value = "";
    $("#paste-drawer").hidden = true;
  } catch (e) { toast(e.message); }
});
$("#btn-example").addEventListener("click", async () => {
  $("#paste-text").value =
    (await (await fetch("/assets/example.txt")).text()).trim();
});
$("#btn-pause").addEventListener("click", () =>
  pauseOrResume().catch((e) => toast(e.message)));
$("#btn-end").addEventListener("click", () =>
  endSession().catch((e) => toast(e.message)));
$("#btn-compare").addEventListener("click", () =>
  openCompare().catch((e) => toast(e.message)));
$("#btn-back").addEventListener("click", () => { S.view = "live"; render(); });
$("#btn-settings").addEventListener("click", openSheet);
$("#btn-sheet-close").addEventListener("click", closeSheet);
$("#scrim").addEventListener("click", closeSheet);
$("#btn-sheet-save").addEventListener("click", () =>
  saveSheet().catch((e) => toast(e.message)));
$("#btn-export").addEventListener("click", () => {
  if (!S.graph && !S.turns.length) return;
  const blob = new Blob(
    [JSON.stringify({ turns: S.turns, graph: S.graph }, null, 2)],
    { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `convograph_${S.sid}.json`;
  a.click();
});
$("#rec-tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#rec-tabs button").forEach((x) => x.classList.remove("on"));
  b.classList.add("on");
  S.recView = b.dataset.v;
  render();
});
$("#cmp-filter").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#cmp-filter button").forEach((x) => x.classList.remove("on"));
  b.classList.add("on"); render();
});
$("#graph-zoom-in").addEventListener("click", () => graph3d?.zoom(+0.05));
$("#graph-zoom-out").addEventListener("click", () => graph3d?.zoom(-0.05));
$("#graph-compare-link").addEventListener("click", () =>
  openCompare().catch((e) => toast(e.message)));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeSheet();
});
