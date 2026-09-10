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
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .bus import STORE, Session
from . import pipeline

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


@app.post("/api/session/{sid}/start", status_code=202)
async def start_text(sid: str, body: StartBody):
    s = _get(sid)
    if s.state not in ("idle",):
        raise HTTPException(409, f"session is {s.state}")
    s.tasks.append(asyncio.create_task(pipeline.run_session(s, text=body.text)))
    return {"ok": True}


@app.post("/api/session/{sid}/audio", status_code=202)
async def start_audio(sid: str, file: UploadFile = File(...)):
    s = _get(sid)
    if s.state not in ("idle",):
        raise HTTPException(409, f"session is {s.state}")
    audio = await file.read()
    s.tasks.append(asyncio.create_task(
        pipeline.run_session(s, audio=audio, audio_name=file.filename or "rec.wav")))
    return {"ok": True}


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
    allowed = {"episode_lines", "window_lines", "max_render_facts", "render", "style"}
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
            "speakers": s.speakers, "episodes_snapshotted": len(s.snapshots)}


os.makedirs(os.path.join(WEB_DIR, "output"), exist_ok=True)
app.mount("/output", StaticFiles(directory=os.path.join(WEB_DIR, "output")),
          name="output")
app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIR, "static")),
          name="assets")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "static", "index.html"))
