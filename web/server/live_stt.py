"""
Live capture: browser microphone -> module 1 -> the session's turn stream.

The chain is three processes per live session, each doing the one thing it is
good at:

    WebSocket (binary webm/opus from MediaRecorder)
      -> ffmpeg  (-i pipe:0 -f s16le -ar 16000 -ac 1 pipe:1)   host process
        -> docker run -i ... scripts/09_live_stt.py            GPU container
          -> JSONL sentence snapshots on stdout
            -> TurnSettler -> turn_partial events + FINAL turns onto turn_q

Why ffmpeg sits in the middle: MediaRecorder emits ONE continuous webm
stream (chunks are not independently decodable), and the models need raw
16 kHz mono PCM — exactly the conversion ffmpeg does statefully on a pipe.
A page reload creates a NEW MediaRecorder and therefore a fresh webm header,
which a mid-stream ffmpeg cannot parse: restart_ffmpeg() swaps only that
process; the GPU container keeps its models and its monotonically increasing
step counter.

One live session at a time: the GB10 is a single GPU and module 1's streaming
state is per-process. GPU_LOCK is shared with the batch upload path.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Optional

from .bus import Session

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODULE1_DIR = os.path.join(REPO_ROOT, "modules", "asr-diarization")
DOCKER_IMAGE = os.environ.get("MODULE1_DOCKER_IMAGE", "sortformer-spark")

GPU_LOCK = asyncio.Lock()
GRACE_SECONDS = 600            # WS gone this long -> auto-end the session


class TurnSettler:
    """Turns module 1's mutable sentence snapshots into immutable turns.

    Sentence identity: (speaker, start_time) — start_time is stable from the
    moment a sentence appears; words and end_time keep changing. A sentence is
    FINAL when (a) a newer sentence by the same speaker exists, (b) it went
    `settle_steps` consecutive snapshots without changing, (c) the stream hit
    EOF, or (d) it scrolled off the snapshot tail. Only finals become turns;
    the one still being spoken is emitted as a transient `turn_partial`.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.seen: dict = {}        # id -> {"row": ..., "stable": n, "final": bool}
        self.order: list = []       # ids in first-seen order

    @staticmethod
    def _id(row: dict) -> str:
        return f"lt-{row['speaker']}-{int(row['start_time'] * 100)}"

    def _finalize(self, sid: str) -> None:
        entry = self.seen[sid]
        if entry["final"]:
            return
        entry["final"] = True
        row = entry["row"]
        # The final streaming flush can stamp times past the real audio end
        # (padding frames) — clamp, as the batch path (08) does.
        clamp = self.session.audio_seconds or row["end_time"]
        self.session.turn_q.put_nowait({"kind": "turn", "turn": {
            "id": sid, "speaker": row["speaker"], "text": row["words"],
            "t_start": min(row["start_time"], clamp),
            "t_end": min(row["end_time"], clamp),
            "source": "live",
        }})

    def feed(self, snapshot: dict) -> None:
        rows = snapshot.get("seglst", [])
        latest_by_speaker: dict = {}
        for row in rows:
            sid = self._id(row)
            if sid not in self.seen:
                self.seen[sid] = {"row": row, "stable": 0, "final": False}
                self.order.append(sid)
            else:
                old = self.seen[sid]["row"]
                if (old["words"], old["end_time"]) == (row["words"], row["end_time"]):
                    self.seen[sid]["stable"] += 1
                else:
                    self.seen[sid]["stable"] = 0
                self.seen[sid]["row"] = row
            prev = latest_by_speaker.get(row["speaker"])
            if prev is None or row["start_time"] > prev[1]:
                latest_by_speaker[row["speaker"]] = (self._id(row), row["start_time"])

        in_snapshot = {self._id(r) for r in rows}
        settle_after = int(self.session.settings.get("settle_steps", 3))
        last_ids = {sid for sid, _ in latest_by_speaker.values()}

        for sid in self.order:
            entry = self.seen[sid]
            if entry["final"]:
                continue
            if snapshot.get("eof"):
                self._finalize(sid)                       # rule (c)
            elif sid not in in_snapshot and snapshot.get("dropped_before", 0) > 0:
                self._finalize(sid)                       # rule (d): off the tail
            elif sid not in last_ids:
                self._finalize(sid)                       # rule (a): superseded
            elif entry["stable"] >= settle_after:
                self._finalize(sid)                       # rule (b): settled
            else:
                # the sentence still being spoken -> transient partial
                row = entry["row"]
                label = row["speaker"]
                self.session.speakers.setdefault(label, label)
                self.session.emit("turn_partial", {
                    "id": sid, "speaker": label,
                    "name": self.session.speakers[label],
                    "text": row["words"], "t_start": row["start_time"],
                    "t_end": row["end_time"],
                }, transient=True)


class LiveSTT:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.audio_q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.settler = TurnSettler(session)
        self.docker: Optional[asyncio.subprocess.Process] = None
        self.ffmpeg: Optional[asyncio.subprocess.Process] = None
        self.paused = False
        self.ready = asyncio.Event()
        self.stopped = False
        self._tasks: list = []
        self._grace: Optional[asyncio.TimerHandle] = None
        self._last_clock = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        s = self.session
        s.state = "warming"
        s.emit("session", {"state": "warming",
                           "detail": "starting module 1 — loading speech "
                                     "models (~1 min on first run)"})
        self.docker = await asyncio.create_subprocess_exec(
            "docker", "run", "--rm", "-i", "--gpus", "all",
            "--ipc=host", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
            "-e", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
            "-v", os.path.expanduser("~/.cache/huggingface") + ":/root/.cache/huggingface",
            "-v", f"{MODULE1_DIR}:/work", "-w", "/work",
            # --entrypoint python skips the NGC banner that the image's
            # entrypoint prints to STDOUT, keeping the JSONL stream clean.
            "--entrypoint", "python",
            DOCKER_IMAGE, "scripts/09_live_stt.py",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await self._spawn_ffmpeg()
        self._tasks += [
            asyncio.create_task(self._read_stdout()),
            asyncio.create_task(self._read_stderr()),
            asyncio.create_task(self._pump_webm()),
        ]

    async def _spawn_ffmpeg(self) -> None:
        self.ffmpeg = await asyncio.create_subprocess_exec(
            "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
            "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._tasks.append(asyncio.create_task(self._pump_pcm(self.ffmpeg)))

    async def restart_ffmpeg(self) -> None:
        """A reconnecting client brings a fresh MediaRecorder = fresh webm
        header; only ffmpeg needs replacing. The container just sees a gap."""
        old = self.ffmpeg
        if old is not None:
            try:
                old.stdin.close()
            except Exception:                                     # noqa: BLE001
                pass
        await self._spawn_ffmpeg()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def touch_grace(self, cancel_only: bool = False) -> None:
        """(Re)arm the auto-end timer; called on WS disconnect/connect."""
        if self._grace is not None:
            self._grace.cancel()
            self._grace = None
        if cancel_only:
            return
        loop = asyncio.get_running_loop()

        def _expire() -> None:
            from . import pipeline
            asyncio.create_task(pipeline.end_session(self.session))

        self._grace = loop.call_later(GRACE_SECONDS, _expire)

    async def stop(self) -> None:
        """Orderly teardown; the settler puts the final {'kind':'end'} when
        the container's eof snapshot has been consumed."""
        if self.stopped:
            return
        self.stopped = True
        self.touch_grace(cancel_only=True)
        await self.audio_q.put(None)          # -> ffmpeg stdin EOF -> chain drains

    # ── pumps and readers ─────────────────────────────────────────────────

    async def _pump_webm(self) -> None:
        assert self.ffmpeg is not None
        while True:
            item = await self.audio_q.get()
            if item is None:
                break
            if self.ffmpeg.stdin is None:
                continue
            try:
                self.ffmpeg.stdin.write(item)
                await self.ffmpeg.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                break
        try:
            self.ffmpeg.stdin.close()
        except Exception:                                         # noqa: BLE001
            pass

    async def _pump_pcm(self, ffmpeg: asyncio.subprocess.Process) -> None:
        s = self.session
        while True:
            data = await ffmpeg.stdout.read(4096)
            if not data:
                break
            s.audio_seconds += len(data) / 2 / 16000
            if s.audio_seconds - self._last_clock >= 1.0:
                self._last_clock = s.audio_seconds
                s.emit("clock", {"audio_seconds": round(s.audio_seconds, 1)},
                       transient=True)
            if self.docker is None or self.docker.stdin is None:
                break
            try:
                self.docker.stdin.write(data)
                await self.docker.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                return
        # This ffmpeg is done. Close the container's stdin ONLY on real stop —
        # a restart_ffmpeg swap ends this pump without ending the session.
        if self.stopped and self.docker is not None and self.docker.stdin:
            try:
                self.docker.stdin.close()
            except Exception:                                     # noqa: BLE001
                pass

    async def _read_stderr(self) -> None:
        assert self.docker is not None
        async for raw in self.docker.stderr:
            line = raw.decode(errors="replace").strip()
            if line.startswith("STATUS "):
                self.session.emit("session", {"state": self.session.state,
                                              "detail": line[7:][:200]})

    async def _read_stdout(self) -> None:
        s = self.session
        got_eof = False
        assert self.docker is not None
        async for raw in self.docker.stdout:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("ready"):
                self.ready.set()
                if s.state == "warming":
                    s.state = "listening"
                    s.emit("session", {"state": "listening",
                                       "started_at": s.started_at,
                                       "detail": "live — speak now"})
                continue
            self.settler.feed(msg)
            if msg.get("eof"):
                got_eof = True
        # container exited
        if got_eof or self.stopped:
            s.turn_q.put_nowait({"kind": "end"})
        else:
            s.emit("error", {"stage": "live",
                             "message": "live transcription process exited "
                                        "unexpectedly — session paused"})
            from . import pipeline
            await pipeline.pause_session(s)
        if GPU_LOCK.locked():
            GPU_LOCK.release()
        s.live = None
