"""
FastAPI backend for the async Convograph app (design: new_ui_template_design).

    cd web && ../ui/.venv/bin/python -m uvicorn server.main:app --port 8700

Endpoints:
    POST /api/session                       {title?, settings?} -> {id}
    POST /api/session/{id}/start            {text} -> 202 (paste-transcript path)
    POST /api/session/{id}/audio            multipart wav/m4a -> 202 (module 1 path)
    GET  /api/session/{id}/events           SSE: full backlog, then live events
    POST /api/session/{id}/speaker          {label, name} -> rename everywhere
    POST /api/session/{id}/settings         merge; applies from the next episode
    GET  /api/session/{id}/diff?a=1&b=3     compare-episodes ledger (screen 1c)
    GET  /                                  the app (web/static)
    GET  /output/...                        rendered images
"""
from __future__ import annotations

import asyncio
import json
import os

from fastapi import (FastAPI, File, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .bus import STORE, Session
from . import live_stt, pipeline

INPUT_OK_STATES = ("idle", "listening", "paused", "warming")

WEB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI(title="convograph-web")


def _get(sid: str) -> Session:
    s = STORE.get(sid)
    if not s:
        raise HTTPException(404, "no such session")
    return s


class NewSession(BaseModel):
    title: str = ""
    settings: dict = {}


class StartBody(BaseModel):
    text: str


class SpeakerBody(BaseModel):
    label: str
    name: str


@app.post("/api/session")
def create_session(body: NewSession):
    s = STORE.create(body.title, body.settings)
    return {"id": s.id, "group_id": s.group_id, "settings": s.settings}


@app.post("/api/session/{sid}/text", status_code=202)
@app.post("/api/session/{sid}/start", status_code=202)   # pre-worker alias
async def add_text(sid: str, body: StartBody):
    s = _get(sid)
    if s.state not in INPUT_OK_STATES:
        raise HTTPException(409, f"session is {s.state}")
    try:
        await pipeline.feed_text(s, body.text)
    except pipeline.StageError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"ok": True}


@app.post("/api/session/{sid}/audio", status_code=202)
async def add_audio(sid: str, file: UploadFile = File(...)):
    s = _get(sid)
    if s.state not in INPUT_OK_STATES:
        raise HTTPException(409, f"session is {s.state}")
    if s.live is not None:
        raise HTTPException(409, "live capture is active on this session — "
                                 "stop it before uploading")
    audio = await file.read()
    s.tasks.append(asyncio.create_task(
        pipeline.feed_audio_file(s, audio, file.filename or "rec.wav")))
    return {"ok": True}


@app.post("/api/session/{sid}/pause")
async def pause(sid: str):
    s = _get(sid)
    await pipeline.pause_session(s)
    return {"state": s.state}


@app.post("/api/session/{sid}/resume")
async def resume(sid: str):
    s = _get(sid)
    await pipeline.resume_session(s)
    return {"state": s.state}


@app.post("/api/session/{sid}/end")
async def end(sid: str):
    s = _get(sid)
    await pipeline.end_session(s)
    return {"state": s.state}


@app.websocket("/api/session/{sid}/live")
async def live(ws: WebSocket, sid: str):
    """Binary frames: webm/opus audio from the browser's MediaRecorder.
    Text frames: {"type": "pause" | "resume" | "stop"}."""
    s = STORE.get(sid)
    if s is None:
        await ws.close(code=4004)
        return
    await ws.accept()

    if s.live is None:
        if live_stt.GPU_LOCK.locked():
            await ws.send_text(json.dumps(
                {"error": "gpu_busy",
                 "message": "another live session holds the GPU"}))
            await ws.close(code=1013)
            return
        await live_stt.GPU_LOCK.acquire()
        s.live = live_stt.LiveSTT(s)
        pipeline.ensure_workers(s)
        await s.live.start()
    else:
        # reconnect: fresh MediaRecorder -> fresh webm header -> new ffmpeg
        s.live.touch_grace(cancel_only=True)
        await s.live.restart_ffmpeg()
        await pipeline.resume_session(s)

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code") or 1000)
            if msg.get("bytes") is not None:
                if s.live is None or s.live.stopped:
                    break
                if not s.live.paused:
                    try:
                        s.live.audio_q.put_nowait(msg["bytes"])
                    except asyncio.QueueFull:
                        pass          # drop rather than stall the socket
            elif msg.get("text"):
                cmd = json.loads(msg["text"]).get("type")
                if cmd == "pause":
                    await pipeline.pause_session(s)
                elif cmd == "resume":
                    # The client restarts its MediaRecorder on resume (fresh
                    # webm header), so the decoder must restart too — decoding
                    # ACROSS a pause boundary is exactly what broke silently.
                    if s.live is not None:
                        await s.live.restart_ffmpeg()
                    await pipeline.resume_session(s)
                elif cmd == "stop":
                    await pipeline.end_session(s)
                    break
    except WebSocketDisconnect:
        if s.live is not None and not s.live.stopped:
            await pipeline.pause_session(s)
            s.live.touch_grace()      # auto-end after the grace period
        return
    try:
        await ws.close()
    except Exception:                                             # noqa: BLE001
        pass


@app.get("/api/session/{sid}/events")
async def events(sid: str):
    s = _get(sid)

    async def gen():
        async for ev in s.stream():
            yield ev.sse()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/session/{sid}/speaker")
def rename_speaker(sid: str, body: SpeakerBody):
    s = _get(sid)
    s.speakers[body.label] = body.name
    for t in s.turns:
        if t["speaker"] == body.label:
            t["name"] = body.name
    # Renames apply to all past and future TURNS (as the design promises).
    # Facts already extracted keep the old name — extraction has happened;
    # episodes ingested after the rename use the new one.
    s.emit("speaker", {"label": body.label, "name": body.name})
    return {"ok": True}


@app.post("/api/session/{sid}/settings")
def update_settings(sid: str, body: dict):
    s = _get(sid)
    allowed = {"episode_turns", "episode_seconds", "settle_steps",
               "window_lines", "max_render_facts", "render", "style"}
    if "episode_lines" in body:          # pre-worker alias
        body.setdefault("episode_turns", body["episode_lines"])
    s.settings.update({k: v for k, v in body.items() if k in allowed})
    s.emit("session", {"state": s.state, "settings": s.settings})
    return {"ok": True, "settings": s.settings}


@app.get("/api/session/{sid}/diff")
def diff(sid: str, a: int, b: int):
    """Ledger between the graph as of episode a and episode b (1-based)."""
    s = _get(sid)
    if not (1 <= a <= len(s.snapshots) and 1 <= b <= len(s.snapshots) and a < b):
        raise HTTPException(400, f"need 1 <= a < b <= {len(s.snapshots)}")
    A = {e["id"]: e for e in s.snapshots[a - 1]["edges"]}
    B = {e["id"]: e for e in s.snapshots[b - 1]["edges"]}

    rows, pair_old = [], {}
    for eid, e in A.items():
        if not e.get("invalid_at") and (eid not in B or B[eid].get("invalid_at")):
            pair_old[(e["source"], e["target"])] = e

    used_old = set()
    for eid, e in B.items():
        if eid in A and not A[eid].get("invalid_at") and not e.get("invalid_at"):
            rows.append({"fact": e["fact"], "at_a": e["fact"], "at_b": e["fact"],
                         "change": "unchanged"})
        elif eid not in A:
            old = pair_old.get((e["source"], e["target"]))
            if old is not None and old["id"] not in used_old:
                used_old.add(old["id"])
                rows.append({"fact": e["fact"], "at_a": old["fact"],
                             "at_b": e["fact"], "change": "revised"})
            elif e.get("invalid_at"):
                # Born and struck between A and B — the reversal landed in the
                # same episode as the decision. Still an invalidation.
                rows.append({"fact": e["fact"], "at_a": None, "at_b": e["fact"],
                             "change": "invalidated"})
            else:
                rows.append({"fact": e["fact"], "at_a": None, "at_b": e["fact"],
                             "change": "added"})
    for eid, e in A.items():
        if eid in used_old:
            continue
        gone = eid not in B or (B[eid].get("invalid_at") and not e.get("invalid_at"))
        if gone and not e.get("invalid_at"):
            rows.append({"fact": e["fact"], "at_a": e["fact"], "at_b": None,
                         "change": "invalidated"})

    order = {"added": 0, "revised": 1, "invalidated": 2, "unchanged": 3}
    rows.sort(key=lambda r: order[r["change"]])
    summary = {c: sum(1 for r in rows if r["change"] == c) for c in order}
    return {"a": a, "b": b, "rows": rows, "summary": summary}


@app.get("/api/session/{sid}/meta")
def meta(sid: str):
    s = _get(sid)
    return {"id": s.id, "title": s.title, "state": s.state,
            "started_at": s.started_at, "settings": s.settings,
            "speakers": s.speakers, "episodes_snapshotted": len(s.snapshots),
            "audio_seconds": round(s.audio_seconds, 1),
            "ingest_pending": s.ingest_q.qsize(),
            "live": s.live is not None}


os.makedirs(os.path.join(WEB_DIR, "output"), exist_ok=True)
app.mount("/output", StaticFiles(directory=os.path.join(WEB_DIR, "output")),
          name="output")
app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIR, "static")),
          name="assets")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "static", "index.html"))
