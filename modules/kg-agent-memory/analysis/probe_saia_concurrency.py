#!/usr/bin/env python3
"""
probe_saia_concurrency.py — how many PARALLEL requests will SAIA accept?

WHY THIS QUESTION AND NOT "how many tokens do we have"
------------------------------------------------------
Graphiti does not send its extraction calls one at a time; it fans them out per
episode. So the limit that actually bites during an end-to-end run is
requests-in-flight, not total volume. A quota measured in tokens-per-day says
nothing about whether call 6 of 8 comes back 429.

This ramps 1 -> 2 -> 4 -> 8 -> 12 concurrent calls and STOPS at the first
rejection, so the answer costs a couple of dozen four-token replies. It is a
probe, not a benchmark: the point is to find the wall, not to measure
throughput, and burning quota to characterise quota would be self-defeating.

The number it prints is what goes in the rate-limit increase email (the SAIA
docs say limits "can be increased upon email request"), instead of asking for
an unspecified "more".

Usage:
    python analysis/probe_saia_concurrency.py            # ramp to 12
    python analysis/probe_saia_concurrency.py --max 4    # stay small
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = "https://chat-ai.academiccloud.de/v1"
MODEL = "qwen3-30b-a3b-instruct-2507"

# The vault's git-ignored .env is the single source of truth for this key
# (CLAUDE.md rule 4b — never borrow one from a neighbouring repo).
ENV_PATH = os.path.expanduser("~/Documents/Obsidian Vault/.env")


def _load_key() -> str:
    if os.getenv("SAIA_API_KEY"):
        return os.environ["SAIA_API_KEY"]
    with open(ENV_PATH) as fh:
        for line in fh:
            if line.startswith("SAIA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"❌ SAIA_API_KEY not found in {ENV_PATH}")


def one_call(key: str, i: int) -> dict:
    """One deliberately tiny request. max_tokens=4 keeps the cost of finding the
    concurrency wall close to zero — we care about whether it is ACCEPTED, not
    what it says. Safe at 4 tokens because this is an INSTRUCT build; a
    reasoning build would spend the whole budget thinking and return nothing."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 4, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            r.read()
        return {"i": i, "ok": True, "secs": time.time() - t0}
    except urllib.error.HTTPError as exc:
        return {"i": i, "ok": False, "code": exc.code,
                "secs": time.time() - t0,
                "body": exc.read().decode(errors="replace")[:200]}
    except Exception as exc:                                   # noqa: BLE001
        return {"i": i, "ok": False, "code": None,
                "secs": time.time() - t0, "body": repr(exc)[:200]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max", type=int, default=12,
                    help="largest concurrency level to try (default 12)")
    args = ap.parse_args()

    key = _load_key()
    print(f"🔑 key loaded from {ENV_PATH}")
    print(f"🎯 {MODEL} @ {BASE_URL}")
    print(f"📏 ramping concurrency, stopping at the first rejection\n")

    levels = [n for n in (1, 2, 4, 8, 12, 16, 24) if n <= args.max]
    total_calls = 0
    wall = None

    for n in levels:
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(lambda i: one_call(key, i), range(n)))
        total_calls += n
        elapsed = time.time() - t0
        ok = [r for r in results if r["ok"]]
        bad = [r for r in results if not r["ok"]]
        lat = sorted(r["secs"] for r in ok)
        median = lat[len(lat) // 2] if lat else float("nan")

        status = "✅" if not bad else "❌"
        print(f"{status} {n:>2} parallel — {len(ok)}/{n} ok, "
              f"wall {elapsed:5.2f}s, median call {median:5.2f}s")
        for r in bad:
            print(f"      ↳ call {r['i']}: HTTP {r['code']} {r['body']}")
        if bad:
            wall = n
            break

    print()
    if wall:
        print(f"🚧 first rejection at {wall} concurrent requests.")
        print(f"   Ask SAIA to raise the limit above {wall}.")
    else:
        print(f"✅ no rejection up to {levels[-1]} concurrent requests.")
        print(f"   Graphiti's fan-out fits under the current limit at that width.")
    print(f"🧾 {total_calls} calls spent, 4 tokens each — negligible quota.")


if __name__ == "__main__":
    main()
