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
import itertools
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


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

    def emit(self, type_: str, data: dict, transient: bool = False) -> None:
        ev = Event(next(self._seq), type_, data)
        if not transient:
            self.events.append(ev)
        for q in list(self.subscribers):
            q.put_nowait(ev)

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
                # applied "from the next episode", as the design promises
                "episode_turns": 12,       # cut after this many FINAL turns…
                "episode_seconds": 120,    # …or this much audio, whichever first
                "settle_steps": 3,         # live: snapshots unchanged -> final
                "window_lines": 5,         # UI_INGEST_WINDOW_LINES for module 2
                "max_render_facts": 6,     # images per episode ceiling
                "render": True,            # render stage on/off
                "style": "pictogram",      # the only wired style (FLUX schnell)
                **_normalize_settings(settings or {}),
            },
        )
        self._sessions[sid] = s
        return s

    def get(self, sid: str) -> Optional[Session]:
        return self._sessions.get(sid)

    def all(self) -> List[Session]:
        return list(self._sessions.values())


STORE = Store()
