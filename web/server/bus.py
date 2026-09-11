"""
Session store + event bus for the async pipeline.

One Session = one meeting. Stage workers publish typed events; every
subscriber (the browser, over SSE) gets the full backlog on connect and live
events after — so a page reload never loses state, and the UI is a pure
function of the event log. Single-process, in-memory: this app runs on the
Spark next to the modules, not in a fleet.

Event types (mirroring the design handoff's "Streams" section):
    session   {state, title, started_at, detail?}   states: idle | warming |
              listening | paused | ending | ended | failed
    turn      {id, speaker, name, t_start, t_end, text, episode, source}
    turn_partial {id, speaker, name, text, t_start, t_end}   TRANSIENT — the
              sentence still being spoken; a later `turn` with the same id
              replaces it
    clock     {audio_seconds}                        TRANSIENT, ~1/s while live
    queue     {ingest_pending}                       ingest backlog depth
    episode   {index, stage: transcribed|graphed|rendered,
               status: pending|running|done|failed, progress, detail}
    graph     {nodes, edges, counters, status_line}       (full snapshot)
    graph_delta {added_facts, invalidated_facts, revised_pairs, episode}
    render    {episode, status, progress, images: [urls], captions}
    board     {url, episodes, elements}
    error     {stage, message}

Transient events are pushed to live subscribers but NOT kept in the backlog:
a reconnecting client re-syncs from the next snapshot within ~1 s, and the
backlog stays a faithful, replayable record of what actually happened.
"""
from __future__ import annotations

import asyncio
import glob
import itertools
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


@dataclass
class Event:
    seq: int
    type: str
    data: dict

    def sse(self) -> str:
        return (f"id: {self.seq}\nevent: {self.type}\n"
                f"data: {json.dumps(self.data, ensure_ascii=False)}\n\n")


@dataclass
class Session:
    id: str
    title: str
    started_at: float
    group_id: str
    settings: Dict[str, Any]
    state: str = "idle"          # see module docstring for the state machine
    events: List[Event] = field(default_factory=list)
    subscribers: List[asyncio.Queue] = field(default_factory=list)
    _seq: itertools.count = field(default_factory=lambda: itertools.count(1))

    # ── the event-driven spine ────────────────────────────────────────────
    # All inputs (paste / upload / live) put tagged messages on turn_q; the
    # episodizer worker is its only consumer and feeds ingest_q; the ingest
    # worker is ingest_q's only consumer. Workers live as long as the session.
    turn_q: asyncio.Queue = field(default_factory=asyncio.Queue)
    ingest_q: asyncio.Queue = field(default_factory=asyncio.Queue)
    workers: List[asyncio.Task] = field(default_factory=list)
    next_episode: int = 1
    audio_seconds: float = 0.0   # session AUDIO time (live); wall time otherwise
    live: Optional[Any] = None   # live_stt.LiveSTT handle while capturing

    # pipeline bookkeeping (also derivable from events; kept for the workers)
    turns: List[dict] = field(default_factory=list)
    speakers: Dict[str, str] = field(default_factory=dict)   # label -> display name
    episodes: List[dict] = field(default_factory=list)
    snapshots: List[dict] = field(default_factory=list)      # graph state per episode
    tasks: List[asyncio.Task] = field(default_factory=list)
    board_entries: List[dict] = field(default_factory=list)  # compose_board input
    board_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _disk_fh: Optional[Any] = None

    def emit(self, type_: str, data: dict, transient: bool = False) -> None:
        ev = Event(next(self._seq), type_, data)
        if not transient:
            self.events.append(ev)
            self._persist(ev)
        for q in list(self.subscribers):
            q.put_nowait(ev)

    def _persist(self, ev: Event) -> None:
        """The durable event log IS the session: replaying it rebuilds the UI
        and the worker bookkeeping after a server restart."""
        if self._disk_fh is None:
            os.makedirs(os.path.join(OUTPUT_DIR, self.id), exist_ok=True)
            self._disk_fh = open(
                os.path.join(OUTPUT_DIR, self.id, "events.jsonl"), "a",
                buffering=1)
        self._disk_fh.write(json.dumps(
            {"seq": ev.seq, "type": ev.type, "data": ev.data},
            ensure_ascii=False) + "\n")

    def write_meta(self) -> None:
        os.makedirs(os.path.join(OUTPUT_DIR, self.id), exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, self.id, "meta.json"), "w") as fh:
            json.dump({"id": self.id, "title": self.title,
                       "group_id": self.group_id,
                       "started_at": self.started_at,
                       "settings": self.settings}, fh)

    async def stream(self) -> AsyncIterator[Event]:
        """Backlog first, then live events, until the client disconnects."""
        q: asyncio.Queue = asyncio.Queue()
        backlog = list(self.events)
        self.subscribers.append(q)
        try:
            for ev in backlog:
                yield ev
            while True:
                yield await q.get()
        finally:
            self.subscribers.remove(q)


def _normalize_settings(settings: dict) -> dict:
    """Accept the pre-worker name `episode_lines` as an alias for
    `episode_turns`, so old clients and scripts keep working."""
    out = dict(settings)
    if "episode_lines" in out and "episode_turns" not in out:
        out["episode_turns"] = out.pop("episode_lines")
    else:
        out.pop("episode_lines", None)
    return out


class Store:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def create(self, title: str, settings: Optional[dict] = None) -> Session:
        sid = uuid.uuid4().hex[:10]
        s = Session(
            id=sid,
            title=title or "Untitled session",
            started_at=time.time(),
            group_id=f"web_{time.strftime('%Y%m%d')}_{sid}",
            settings={
                # applied "from the next episode", as the design promises.
                # Smaller episodes = more supersession opportunities: Graphiti
                # invalidates most reliably when the reversal arrives in a
                # LATER ingest call than the decision (observed 2/2 across
                # episodes vs 0/2 within one episode, 2026-09-11).
                "episode_turns": 8,        # cut after this many FINAL turns…
                "episode_seconds": 75,     # …or this much audio, whichever first
                "settle_steps": 3,         # live: snapshots unchanged -> final
                "window_lines": 5,         # UI_INGEST_WINDOW_LINES for module 2
                "max_render_facts": 6,     # images per episode ceiling
                "render": True,            # render stage on/off
                "style": "pictogram",      # the only wired style (FLUX schnell)
                **_normalize_settings(settings or {}),
            },
        )
        self._sessions[sid] = s
        s.write_meta()
        return s

    def get(self, sid: str) -> Optional[Session]:
        s = self._sessions.get(sid)
        if s is None:
            s = self._load(sid)
            if s is not None:
                self._sessions[sid] = s
        return s

    def _load(self, sid: str) -> Optional[Session]:
        """Rebuild a session from its durable event log after a restart.

        Everything the UI and the workers need replays from events; graph
        snapshots reload from their files. Processes and queues are gone, so a
        session that was mid-flight comes back as 'paused' — adding input (or
        going live again) re-arms the workers, and the same group_id means the
        knowledge graph simply continues where it left off. Episodes that were
        queued-but-not-ingested when the server died are lost; the graph holds
        exactly what was committed."""
        meta_path = os.path.join(OUTPUT_DIR, sid, "meta.json")
        if not os.path.exists(meta_path):
            return None
        meta = json.load(open(meta_path))
        s = Session(id=sid, title=meta["title"], started_at=meta["started_at"],
                    group_id=meta["group_id"],
                    settings={**meta.get("settings", {})})
        events_path = os.path.join(OUTPUT_DIR, sid, "events.jsonl")
        max_ep = 0
        if os.path.exists(events_path):
            for line in open(events_path):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = Event(next(s._seq), raw["type"], raw["data"])
                s.events.append(ev)
                d = ev.data
                if ev.type == "turn":
                    s.turns.append(dict(d))
                    s.speakers.setdefault(d["speaker"], d["name"])
                    max_ep = max(max_ep, d.get("episode", 0))
                elif ev.type == "speaker":
                    s.speakers[d["label"]] = d["name"]
                elif ev.type == "episode":
                    max_ep = max(max_ep, d.get("index", 0))
                elif ev.type == "session" and "state" in d:
                    s.state = d["state"]
                elif ev.type == "render" and d.get("status") == "done":
                    ep_dir = os.path.join(OUTPUT_DIR, sid)
                    for url, cap in zip(d.get("images", []),
                                        d.get("facts") or d.get("captions") or []):
                        s.board_entries.append({
                            "image": os.path.join(OUTPUT_DIR, sid,
                                                  os.path.basename(url)),
                            "caption": cap,
                            "label": f"episode {d['episode']}",
                            "timestamp": f"{d['episode']:03d}",
                        })
        s.next_episode = max_ep + 1
        for snap in sorted(glob.glob(os.path.join(OUTPUT_DIR, sid, "snap_ep*.json"))):
            try:
                s.snapshots.append(json.load(open(snap)))
            except json.JSONDecodeError:
                pass
        if s.state not in ("ended", "failed", "idle"):
            # it was mid-flight when the server went away
            s.state = "paused"
            s.emit("session", {"state": "paused",
                               "detail": "restored after a server restart — "
                                         "add input or go live to continue"})
        return s

    def list_all(self) -> List[dict]:
        """Sessions in memory plus every persisted one on disk."""
        rows: Dict[str, dict] = {}
        for meta_path in glob.glob(os.path.join(OUTPUT_DIR, "*", "meta.json")):
            try:
                m = json.load(open(meta_path))
            except json.JSONDecodeError:
                continue
            rows[m["id"]] = {"id": m["id"], "title": m["title"],
                             "started_at": m["started_at"], "state": "on disk",
                             "episodes": len(glob.glob(os.path.join(
                                 os.path.dirname(meta_path), "snap_ep*.json")))}
        for s in self._sessions.values():
            rows[s.id] = {"id": s.id, "title": s.title,
                          "started_at": s.started_at, "state": s.state,
                          "episodes": len(s.snapshots)}
        return sorted(rows.values(), key=lambda r: r["started_at"], reverse=True)


STORE = Store()
