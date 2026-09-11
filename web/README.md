# web — the async app (design: `new_ui_template_design/`)

The three modules run on three different processors — module 1 on the Spark's
GB10, module 2's extraction on SAIA in the cloud, module 3's FLUX on the H100
tunnel — so this app runs them **concurrently**: episode k renders while
episode k+1 is still extracting, and the browser watches every stage over one
SSE stream. This is the pipelining the Streamlit `ui/` cannot express (its
model is "rerun the whole script"); `ui/` stays untouched as the batch tool.

```
browser (static/, no build step)
   ▲  SSE: session/turn/turn_partial/clock/queue/episode/graph/render/board/error
   │  WS (live only): binary webm audio down, pause/resume/stop commands
   │
FastAPI (server/main.py) ── event bus (server/bus.py, backlog + transient)
   │
   │   ALL inputs feed one turn stream; stages are long-lived workers:
   │
   ├─ paste ──► parse_turns ─────────┐
   ├─ upload ─► ui/module1.py batch ─┤ turn_q ─► EPISODIZER ─► ingest_q
   ├─ live ──► ws ► ffmpeg ►         │  (12 turns or 120 s)      │
   │    docker 09_live_stt.py ►      │                           ▼
   │    TurnSettler (partials) ──────┘                    INGEST worker (serial)
   │        module 1, persistent GPU container             ui_ingest.py → SAIA
   │                                                       → AuraDB; snapshot +
   │                                                       diff per episode
   └─ render worker: per-episode task (captions + FLUX H100) — overlaps the
      next episode's ingest — then compose_board → the Excalidraw canvas
```

## Run it

```bash
cd web
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt   # once
./.venv/bin/python -m uvicorn server.main:app --port 8700
```

Open http://localhost:8700 (through your SSH tunnel — the mic recorder needs a
`localhost` or HTTPS origin, so the tunnel is required, not optional).
Backends expected (same as `ui/`): module 2's `.venv` + `.env` (AuraDB, SAIA),
the `serv-7101` tunnel for ports 11435 (captions/embedder) and 8500 (FLUX),
and module 1's Docker image for the audio path.

Pipeline check without a browser: `POST /api/session`, `POST .../start` with
`{"text": "Name: words\n..."}`, then `GET .../events` and watch the stages
overlap. `GET .../diff?a=1&b=2` is the compare-episodes ledger.

## How the design handoff maps (wired / adapted / not wired)

| Design element (screens 1a–1d) | Status |
|---|---|
| Transcript rail: turns, monograms, episode boundaries, fading, word count | **wired** |
| Click-to-rename speakers, applies to all past turns | **wired** (turns only; facts already extracted keep the old name) |
| 3D knowledge graph: Fibonacci sphere, rotation, depth opacity, status styles (added pulse / revised ring / invalidated dashed+strikethrough), zoom, legend | **wired**, live data |
| Episode track: 3 stage-bars per episode, live progress | **wired** — this row IS the async pipeline |
| Graphic recording panel: latest render, frames strip, status chip with real progress, brushing-in sweep, empty state with spinner | **wired** (collage of per-fact pictograms — see below) |
| Compare episodes: scrubber, A/B, ledger with change tags and struck-through old values, changed-region graph | **wired** (click dots to move A/B; drag not implemented) |
| Settings sheet: episodes, render on/off, max images, extraction window; "applies from next episode" | **wired** for the knobs that exist |
| Live mic, streaming transcript with "speaking" partials + caret | **wired**: browser MediaRecorder → WebSocket → ffmpeg → a persistent GPU container running module 1's `09_live_stt.py` (NeMo's chunked streaming API, ~1–2.5 s latency). Sentences appear as mutable `turn_partial`s and settle into final turns that feed the episodizer continuously. Pause/resume/reconnect handled; one live session at a time (single GPU). |
| Session lifecycle (Pause / End, live timer) | **wired**: pause stops audio intake and flushes the open episode (models stay warm); End drains the extraction queue visibly (`ending` state); the timer counts session audio time during live capture. |
| Input in the main view | **wired**: no entry screen — a session exists on page load; 🎙 Live / Upload / Paste live in the transcript rail and all append to the running session. |
| One painterly plate per episode with sketch layer + positioned labels | **adapted**: our module 3 renders one pictogram per fact, so the plate shows a captioned collage. Painterly/sketch-layer swatches are visible but disabled, labeled with what they'd need (FLUX 1.1 pro backend / Excalidraw layer). |
| "Render pass 2 of 3" | **adapted**: schnell is single-pass; the chip shows caption/render progress instead. |
| Whisper-large badge, voice-print card, waveform, "Detect" speakers | **not wired** — no voiceprint backend; ASR is Parakeet, not Whisper (badge says so). |
| Topic-shift episodizer | **not wired** — episodes cut on turn count (setting). The boundary label in the transcript is honest about it. |
| Ledger "By" column (who caused a change) | **not wired** — module 2 doesn't attribute facts to speakers yet. |
| Export | **adapted**: exports turns+graph JSON (not the design's unspecified format). |

## Live-path notes

- A live session keeps one warm GPU container for its whole life (~5.5 GB;
  model load ~1 min, shown as the `warming` state). `make live-stt-sim
  WAV=examples/2spk.wav FAST=1` exercises the container loop without the app.
- `web/tests/live_client.py` is the no-microphone integration test: it streams
  `tests/fixture_60s.webm` over the WebSocket at real-time pace and asserts
  the whole event contract (partial→final per turn id, pause freezes the
  clock, stop → ending → drained → ended).
- A speaker who never pauses ≥4 s produces one ever-growing partial that only
  settles at a sentence break — natural turn-taking settles within ~3–5 s.
  Known limitation, same root as the batch monologue caveat.

## Verified (2026-09-10, on this machine)

Robot-launch test transcript, 2 episodes: extraction on SAIA, 6 FLUX images,
and the money shot — `diff?a=1&b=2` returned `1 revised + 2 invalidated`
(deployment date revised Kaiserslautern→Mannheim, crew booking superseded),
rendered struck-through in the ledger and dashed in the graph. Zero errors.

Note the sibling `output/` dir is gitignored (rendered PNGs + snapshots land
there per session).
