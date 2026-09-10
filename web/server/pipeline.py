"""
The async three-module pipeline. This file is the reason web/ exists.

The three stages run on three different processors, so they overlap:

    transcribe (module 1)      Spark GB10, Docker subprocess
    graph      (module 2)      SAIA cloud LLM via ui_ingest.py subprocess
    render     (module 3)      caption model + FLUX on the H100 tunnel

Episodes chain through ingest SEQUENTIALLY (each Graphiti episode builds on
the previous one — that is what makes supersession detection work), but the
moment episode k's ingest lands, its render fires as an independent task while
episode k+1's ingest is already running. The episode track in the UI shows
exactly this overlap.

Every stage only talks to the modules over the same subprocess seams the
Streamlit UI uses (ADR 0003 spirit): ui_ingest.py in module 2's venv,
kg_to_caption + the FLUX HTTP server for module 3, ui/module1.py for audio.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from typing import List, Optional

from .bus import Session

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KG_MODULE = os.path.join(REPO_ROOT, "modules", "kg-agent-memory")
KG_PYTHON = os.path.join(KG_MODULE, ".venv", "bin", "python")
IMG_MODULE = os.path.join(REPO_ROOT, "modules", "graphic-generation")
UI_DIR = os.path.join(REPO_ROOT, "ui")
OUTPUT_DIR = os.path.join(REPO_ROOT, "web", "output")
KG_QUERY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_query.py")

# kg_to_caption reads these at import time (same note as ui/orchestrator.py).
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11435")
os.environ.setdefault("CAPTION_MODEL", "qwen3:4b-instruct")
os.environ.setdefault("FLUX_SERVER_URL", "http://localhost:8500")

sys.path.insert(0, IMG_MODULE)
sys.path.insert(0, UI_DIR)
import kg_to_caption  # noqa: E402  (stdlib-only)
import compose_board  # noqa: E402  (module 3's Excalidraw composer; needs PIL)


TURN_RE = re.compile(r"^\s*([^:\n]{1,60}?)\s*:\s*(.+)$")


class StageError(RuntimeError):
    pass


# ── input parsing ─────────────────────────────────────────────────────────────

def parse_turns(text: str) -> List[dict]:
    """'Name (Role): words' lines -> turn dicts. Lines without a speaker tag
    continue the previous turn (people paste transcripts with wrapped lines)."""
    turns: List[dict] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = TURN_RE.match(raw)
        if m and turns is not None and len(m.group(1).split()) <= 6:
            turns.append({"speaker": m.group(1).strip(), "text": m.group(2).strip()})
        elif turns:
            turns[-1]["text"] += " " + raw.strip()
    return turns


# ── stage: transcribe (module 1) ──────────────────────────────────────────────

async def transcribe_audio(session: Session, audio: bytes, name: str) -> str:
    """Audio bytes -> speaker-tagged transcript text, via ui/module1.py
    (ffmpeg -> Docker Sortformer+Parakeet -> schema export -> lines)."""
    import module1  # ui/module1.py — needs the Docker image + GPU

    loop = asyncio.get_running_loop()
    session.emit("episode", {"index": 0, "stage": "transcribed",
                             "status": "running", "progress": 0.1,
                             "detail": "module 1: diarizing + transcribing on the GB10"})

    def _progress(msg: str) -> None:
        loop.call_soon_threadsafe(
            session.emit, "session", {"state": "transcribing", "detail": msg})

    res = await loop.run_in_executor(
        None, lambda: module1.transcribe(audio, name, progress_cb=_progress))
    text = await loop.run_in_executor(None, lambda: module1.to_text(res.transcript_json))
    return text


# ── stage: episodize + emit turns ─────────────────────────────────────────────

def cut_episodes(session: Session, turns: List[dict]) -> List[List[dict]]:
    n = max(2, int(session.settings["episode_lines"]))
    return [turns[i:i + n] for i in range(0, len(turns), n)]


def emit_turns(session: Session, episodes: List[List[dict]]) -> None:
    t = 0
    for ep_idx, ep in enumerate(episodes, start=1):
        for turn in ep:
            label = turn["speaker"]
            session.speakers.setdefault(label, label)
            record = {"id": f"t{t:04d}", "speaker": label,
                      "name": session.speakers[label],
                      "text": turn["text"], "episode": ep_idx}
            session.turns.append(record)
            session.emit("turn", record)
            t += 1
        session.emit("episode", {"index": ep_idx, "stage": "transcribed",
                                 "status": "done", "progress": 1.0,
                                 "detail": f"{len(ep)} turns"})


# ── stage: graph (module 2) ───────────────────────────────────────────────────

async def _run(cmd: List[str], env: Optional[dict] = None, timeout: int = 1800) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=KG_MODULE, env={**os.environ, **(env or {})})
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise StageError(f"timed out after {timeout}s: {' '.join(cmd[:3])}")
    if proc.returncode != 0:
        tail = "\n".join((out.decode() + "\n" + err.decode()).strip().splitlines()[-8:])
        raise StageError(tail)
    return out.decode()


async def snapshot_graph(session: Session) -> dict:
    out = await _run([KG_PYTHON, KG_QUERY, session.group_id], timeout=120)
    return json.loads(out.strip().splitlines()[-1])


def diff_snapshots(prev: dict, cur: dict, episode: int) -> dict:
    """Statuses per the design: added / revised / invalidated / confirmed."""
    prev_edges = {e["id"]: e for e in prev.get("edges", [])}
    added, invalidated = [], []
    for e in cur["edges"]:
        old = prev_edges.get(e["id"])
        if old is None:
            added.append(e)
            # Extracted and superseded within the same episode (the reversal
            # landed in the same window as the decision) — still an
            # invalidation, and the UI must strike it through.
            if e.get("invalid_at"):
                invalidated.append(e)
        elif not old.get("invalid_at") and e.get("invalid_at"):
            invalidated.append(e)
    # revised = the same entity pair lost one fact and gained another this episode
    inv_pairs = {(e["source"], e["target"]) for e in invalidated}
    revised_ids = {e["id"] for e in added if (e["source"], e["target"]) in inv_pairs}

    prev_nodes = {n["id"] for n in prev.get("nodes", [])}
    edge_status = {}
    for e in cur["edges"]:
        if e["id"] in revised_ids:
            edge_status[e["id"]] = "revised"
        elif any(e["id"] == a["id"] for a in added):
            edge_status[e["id"]] = "added"
        elif e.get("invalid_at"):
            edge_status[e["id"]] = "invalidated"
        else:
            edge_status[e["id"]] = "confirmed"

    node_status = {}
    for n in cur["nodes"]:
        incident = [e for e in cur["edges"] if n["id"] in (e["source"], e["target"])]
        if incident and all(e.get("invalid_at") for e in incident):
            node_status[n["id"]] = "invalidated"
        elif any(edge_status[e["id"]] == "revised" for e in incident):
            node_status[n["id"]] = "revised"
        elif n["id"] not in prev_nodes:
            node_status[n["id"]] = "added"
        else:
            node_status[n["id"]] = "confirmed"

    return {
        "episode": episode,
        "added": [e["fact"] for e in added],
        "invalidated": [e["fact"] for e in invalidated],
        "revised_count": len(revised_ids),
        "edge_status": edge_status,
        "node_status": node_status,
    }


async def ingest_episode(session: Session, ep_idx: int, ep_turns: List[dict]) -> List[str]:
    """Feed one episode's lines into module 2; returns facts ADDED this episode
    (what the render stage should draw). Emits graph snapshot + delta events."""
    session.emit("episode", {"index": ep_idx, "stage": "graphed",
                             "status": "running", "progress": 0.05,
                             "detail": "extracting entities + relations (SAIA)"})
    lines = "\n".join(
        f"{session.speakers.get(t['speaker'], t['speaker'])}: {t['text']}"
        for t in ep_turns)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(lines)
        path = fh.name
    try:
        await _run(
            [KG_PYTHON, os.path.join(KG_MODULE, "ui_ingest.py"),
             "--text-file", path, "--group-id", session.group_id],
            env={"UI_INGEST_WINDOW_LINES": str(session.settings["window_lines"])})
    finally:
        os.unlink(path)

    cur = await snapshot_graph(session)
    prev = session.snapshots[-1] if session.snapshots else {"nodes": [], "edges": []}
    delta = diff_snapshots(prev, cur, ep_idx)
    session.snapshots.append(cur)
    # Persist snapshots: debuggability, and the compare screen survives restarts.
    snap_dir = os.path.join(OUTPUT_DIR, session.id)
    os.makedirs(snap_dir, exist_ok=True)
    with open(os.path.join(snap_dir, f"snap_ep{ep_idx:02d}.json"), "w") as fh:
        json.dump(cur, fh, ensure_ascii=False)

    valid = [e for e in cur["edges"] if not e.get("invalid_at")]
    inv_names = ", ".join(delta["invalidated"][:1])
    add_names = ", ".join(delta["added"][:1])
    bits = []
    if delta["invalidated"]:
        bits.append(f"“{inv_names[:60]}” marked invalid")
    if delta["added"]:
        bits.append(f"“{add_names[:60]}” added")
    session.emit("graph", {
        "nodes": cur["nodes"], "edges": cur["edges"],
        "node_status": delta["node_status"], "edge_status": delta["edge_status"],
        "counters": {"facts": len(valid), "relations": len(cur["edges"]),
                     "invalidated": len(cur["edges"]) - len(valid)},
        "status_line": f"Extracted episode {ep_idx}"
                       + (" — " + "; ".join(bits) if bits else ""),
    })
    session.emit("graph_delta", delta)
    session.emit("episode", {"index": ep_idx, "stage": "graphed", "status": "done",
                             "progress": 1.0,
                             "detail": f"+{len(delta['added'])} facts, "
                                       f"{len(delta['invalidated'])} invalidated"})
    return delta["added"]


# ── stage: render (module 3) ──────────────────────────────────────────────────

def _caption(fact: str) -> str:
    return kg_to_caption.caption_from_fact(fact)


def _flux_batch(captions: List[str]) -> List[Optional[bytes]]:
    base = os.environ["FLUX_SERVER_URL"].rstrip("/")
    payload = json.dumps({"captions": [" ".join(c.split()) for c in captions]}).encode()
    req = urllib.request.Request(f"{base}/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        result = json.loads(r.read())
    by_index = {img.get("index"): img for img in result.get("images", [])}
    out: List[Optional[bytes]] = []
    for i in range(len(captions)):
        img = by_index.get(i)
        out.append(base64.b64decode(img["png_b64"])
                   if img and "png_b64" in img else None)
    return out


async def render_episode(session: Session, ep_idx: int, facts: List[str]) -> None:
    if not session.settings.get("render", True):
        session.emit("episode", {"index": ep_idx, "stage": "rendered",
                                 "status": "done", "progress": 1.0,
                                 "detail": "rendering off in settings"})
        return
    facts = facts[: int(session.settings["max_render_facts"])]
    if not facts:
        session.emit("episode", {"index": ep_idx, "stage": "rendered",
                                 "status": "done", "progress": 1.0,
                                 "detail": "nothing new to draw"})
        return

    loop = asyncio.get_running_loop()
    session.emit("render", {"episode": ep_idx, "status": "captioning",
                            "progress": 0.05, "images": [], "captions": []})
    session.emit("episode", {"index": ep_idx, "stage": "rendered",
                             "status": "running", "progress": 0.1,
                             "detail": f"captioning {len(facts)} fact(s)"})
    captions = []
    for i, fact in enumerate(facts):
        captions.append(await loop.run_in_executor(None, _caption, fact))
        session.emit("render", {"episode": ep_idx, "status": "captioning",
                                "progress": 0.05 + 0.35 * (i + 1) / len(facts),
                                "images": [], "captions": captions})

    session.emit("episode", {"index": ep_idx, "stage": "rendered",
                             "status": "running", "progress": 0.5,
                             "detail": f"FLUX: {len(captions)} image(s) on the H100"})
    session.emit("render", {"episode": ep_idx, "status": "rendering",
                            "progress": 0.5, "images": [], "captions": captions})
    images = await loop.run_in_executor(None, _flux_batch, captions)

    ep_dir = os.path.join(OUTPUT_DIR, session.id)
    os.makedirs(ep_dir, exist_ok=True)
    urls = []
    for i, png in enumerate(images):
        if png is None:
            continue
        fname = f"ep{ep_idx:02d}_{i:02d}.png"
        with open(os.path.join(ep_dir, fname), "wb") as fh:
            fh.write(png)
        urls.append(f"/output/{session.id}/{fname}")

    session.emit("render", {"episode": ep_idx, "status": "done", "progress": 1.0,
                            "images": urls, "captions": captions,
                            "facts": facts})
    session.emit("episode", {"index": ep_idx, "stage": "rendered", "status": "done",
                             "progress": 1.0, "detail": f"{len(urls)} image(s)"})

    # ── the Excalidraw board: module 3's composer over everything so far ──
    # One editable scene per session, regrown after each episode. Renders can
    # finish out of order, so composition is serialized per session and the
    # entries carry the episode index for stable ordering.
    kept = [i for i, png in enumerate(images) if png is not None]
    for j, i in enumerate(kept):
        session.board_entries.append({
            "image": os.path.join(ep_dir, f"ep{ep_idx:02d}_{i:02d}.png"),
            "caption": facts[i] if i < len(facts) else captions[i],
            "label": f"episode {ep_idx}",
            "timestamp": f"{ep_idx:03d}-{j:02d}",
        })
    async with session.board_lock:
        board_path = os.path.join(ep_dir, "board.excalidraw")
        entries = list(session.board_entries)
        scene = await loop.run_in_executor(
            None, lambda: compose_board.compose(entries, board_path))
        scene.pop("_layout_debug", None)
        with open(board_path, "w") as fh:
            json.dump(scene, fh)
        session.emit("board", {"url": f"/output/{session.id}/board.excalidraw",
                               "episodes": ep_idx,
                               "elements": len(scene.get("elements", []))})


# ── the conductor ─────────────────────────────────────────────────────────────

async def run_session(session: Session, text: Optional[str] = None,
                      audio: Optional[bytes] = None,
                      audio_name: str = "recording.wav") -> None:
    try:
        if audio is not None:
            session.state = "transcribing"
            session.emit("session", {"state": "transcribing", "title": session.title})
            text = await transcribe_audio(session, audio, audio_name)

        turns = parse_turns(text or "")
        if not turns:
            raise StageError("no speaker-tagged turns found in the input")

        session.state = "listening"
        session.emit("session", {"state": "listening", "title": session.title,
                                 "started_at": session.started_at})

        episodes = cut_episodes(session, turns)
        for i in range(1, len(episodes) + 1):
            for stage in ("transcribed", "graphed", "rendered"):
                session.emit("episode", {"index": i, "stage": stage,
                                         "status": "pending", "progress": 0.0})
        emit_turns(session, episodes)

        # Ingest sequentially; fire each episode's render as its own task the
        # moment its ingest lands. Renders overlap the NEXT ingest — that is
        # the whole point of this app.
        render_tasks: List[asyncio.Task] = []
        for ep_idx, ep_turns in enumerate(episodes, start=1):
            try:
                added = await ingest_episode(session, ep_idx, ep_turns)
            except StageError as exc:
                session.emit("error", {"stage": "graph", "episode": ep_idx,
                                       "message": str(exc)})
                session.emit("episode", {"index": ep_idx, "stage": "graphed",
                                         "status": "failed", "progress": 0.0,
                                         "detail": str(exc)[:160]})
                continue
            t = asyncio.create_task(render_episode(session, ep_idx, added))
            render_tasks.append(t)
            session.tasks.append(t)

        for t in render_tasks:
            try:
                await t
            except Exception as exc:                              # noqa: BLE001
                session.emit("error", {"stage": "render", "message": str(exc)})

        session.state = "ended"
        session.emit("session", {"state": "ended"})
    except Exception as exc:                                      # noqa: BLE001
        session.state = "failed"
        session.emit("error", {"stage": "pipeline", "message": str(exc)})
        session.emit("session", {"state": "failed"})
