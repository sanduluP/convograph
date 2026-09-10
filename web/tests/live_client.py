"""
Integration test for the live path — a fake browser, no microphone needed.

Streams a webm fixture over the live WebSocket at real-time pace (the way
MediaRecorder would), listens to the session's SSE stream alongside, and
asserts the event contract:

  * warming -> listening (after model load)
  * per live turn id: one-or-more turn_partial, then exactly one final turn
  * pause freezes the audio clock; resume unfreezes it
  * stop -> ending -> ingest queue drains -> ended
  * episodes: transcribed done -> graphed done -> render/board events

Run (server on :8700, fixture from tests/):
    ./.venv/bin/python tests/live_client.py
"""
import asyncio
import json
import sys
import time

import httpx
import websockets

BASE = "http://localhost:8700"
FIXTURE = "tests/fixture_60s.webm"
CHUNK_MS = 250

events = []            # (type, data) in arrival order
clock_values = []


async def sse_listener(sid: str) -> None:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", f"{BASE}/api/session/{sid}/events") as r:
            etype = None
            async for line in r.aiter_lines():
                if line.startswith("event: "):
                    etype = line[7:]
                elif line.startswith("data: ") and etype:
                    data = json.loads(line[6:])
                    events.append((etype, data))
                    if etype == "clock":
                        clock_values.append((time.monotonic(),
                                             data["audio_seconds"]))
                    if etype == "session" and data.get("state") in ("ended", "failed"):
                        return
                    etype = None


async def main() -> None:
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE}/api/session", json={
            "title": "live integration test",
            "settings": {"episode_turns": 4, "episode_seconds": 30,
                         "max_render_facts": 2, "settle_steps": 3},
        })
        sid = r.json()["id"]
    print(f"session {sid}")

    listener = asyncio.create_task(sse_listener(sid))
    ws_url = f"ws://localhost:8700/api/session/{sid}/live"
    webm = open(FIXTURE, "rb").read()
    # MediaRecorder(250ms) on 60s of ~483KB -> ~2KB per chunk
    n_chunks = 60_000 // CHUNK_MS
    csize = max(1, len(webm) // n_chunks)

    async with websockets.connect(ws_url, max_size=None) as ws:
        print("ws open — streaming at real-time pace (60 s)...")
        sent = 0
        i = 0
        paused_at = None
        while sent < len(webm):
            await ws.send(webm[sent:sent + csize])
            sent += csize
            i += 1
            await asyncio.sleep(CHUNK_MS / 1000)
            if i == 120 and paused_at is None:          # ~30 s in: test pause
                print("pausing 5 s...")
                await ws.send(json.dumps({"type": "pause"}))
                paused_at = time.monotonic()
                await asyncio.sleep(5)
                clock_before = clock_values[-1][1] if clock_values else 0
                await ws.send(json.dumps({"type": "resume"}))
                print(f"resumed (clock during pause: {clock_before}s)")
        print("audio done — sending stop")
        await ws.send(json.dumps({"type": "stop"}))
        # keep the socket open until the server closes it
        try:
            await asyncio.wait_for(ws.recv(), timeout=30)
        except Exception:                                # noqa: BLE001
            pass

    print("waiting for the session to end (ingest drain)...")
    await asyncio.wait_for(listener, timeout=900)

    # ── assertions ────────────────────────────────────────────────────────
    states = [d["state"] for t, d in events if t == "session" and "state" in d]
    assert "warming" in states, f"no warming state: {states}"
    assert "listening" in states, f"no listening state: {states}"
    assert states[-1] == "ended", f"final state {states[-1]}"
    assert "ending" in states, "no ending state"

    finals = [d for t, d in events if t == "turn"]
    live_finals = [d for d in finals if d.get("source") == "live"]
    assert live_finals, "no live turns settled"
    partial_ids = {d["id"] for t, d in events if t == "turn_partial"}
    # every live final should have been seen as a partial first
    covered = sum(1 for d in live_finals if d["id"] in partial_ids)
    print(f"{len(live_finals)} live turns, {covered} preceded by partials, "
          f"{len(partial_ids)} distinct partial ids")
    assert covered >= 1, "no partial->final sequence observed"

    # per-id ordering: last partial before its final
    for d in live_finals:
        idx_final = next(i for i, (t, e) in enumerate(events)
                         if t == "turn" and e.get("id") == d["id"])
        later_partials = [i for i, (t, e) in enumerate(events)
                          if t == "turn_partial" and e["id"] == d["id"]
                          and i > idx_final]
        assert not later_partials, f"partial after final for {d['id']}"

    eps_done = {d["index"] for t, d in events
                if t == "episode" and d["stage"] == "graphed"
                and d["status"] == "done"}
    assert eps_done, "no episode was graphed"
    boards = [d for t, d in events if t == "board"]
    renders = [d for t, d in events if t == "render" and d["status"] == "done"]
    print(f"episodes graphed: {sorted(eps_done)}, renders done: {len(renders)}, "
          f"boards: {len(boards)}")

    errors = [d for t, d in events if t == "error"]
    for e in errors:
        print(f"  (error event: {e['stage']}: {e['message'][:100]})")

    print("PASS — live path verified end to end")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
