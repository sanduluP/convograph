#!/usr/bin/env python3
"""Survey the H2HMem MULTI-PARTY subset — is it a viable second corpus for us?

WHY
---
GroupMemBench's Finance domain is 30,000 messages and costs ~38 h of GPU to
ingest, which is why we had to shard it across 8 SLURM jobs. H2HMem
(arXiv 2606.09461) is a candidate SECOND evaluation corpus. Before committing to
it we want the numbers that actually decide the question:

  1. How big is the multi-party subset really — i.e. would one GPU reservation
     be enough, or are we back to sharding?
  2. How many questions target the thing our temporal KG is FOR (superseded
     facts, evolution, conflicts, ordering) versus plain lookup?
  3. How many questions need an IMAGE, i.e. cannot be answered by a text-only
     pipeline without a captioning pass?
  4. Do the answers span multiple sessions (long-range memory) or sit inside one?

Everything is read straight from the HuggingFace repo over HTTPS — no clone, no
GPU, no API key (the dataset is public, MIT).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List
from urllib.request import urlopen

# HF serves raw files from /resolve/main/<path>. The dataset ships one folder per
# dialogue, and inside it one folder per session holding session.json (the turns)
# and questions.json (the QA pairs).
BASE = "https://huggingface.co/datasets/varib/H2HMEM/resolve/main"
SPLIT = "multi-party"          # the subset that matches our target: >2 speakers

# The task types whose whole point is that a fact CHANGES over time — the reason
# a bi-temporal KG should beat verbatim chunk retrieval. Named exactly as the
# paper's sub_type strings appear in the JSON (matched case-insensitively and by
# prefix, because the files are not perfectly consistent).
TEMPORAL_SUBTYPES = [
    "knowledge resolution",          # currently-correct fact across updates
    "reference & evolution tracking",  # how an entity evolved
    "temporal reasoning",            # order events
    "conflict detection",            # contradicts stored memory
]


def fetch_json(path: str) -> Any:
    """GET one JSON file from the HF repo (raises on any HTTP problem)."""
    with urlopen(f"{BASE}/{path}", timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_repo_files() -> List[str]:
    """Every file path in the dataset repo, via the HF tree API (paginated)."""
    url = ("https://huggingface.co/api/datasets/varib/H2HMEM/tree/main/"
           f"{SPLIT}?recursive=true&expand=false")
    with urlopen(url, timeout=60) as resp:
        entries = json.loads(resp.read().decode("utf-8"))
    return [e["path"] for e in entries if e.get("type") == "file"]


def is_temporal(sub_type: str) -> bool:
    """True when this question type depends on facts changing over time."""
    s = (sub_type or "").strip().lower()
    return any(s.startswith(t) for t in TEMPORAL_SUBTYPES)


def main() -> int:
    print(f"🌐 listing {SPLIT} files on HuggingFace …", flush=True)
    files = list_repo_files()
    session_files = sorted(f for f in files if f.endswith("session.json"))
    question_files = sorted(f for f in files if f.endswith("questions.json"))
    image_files = [f for f in files if f.endswith((".jpg", ".jpeg", ".png"))]
    print(f"   {len(session_files)} session.json, {len(question_files)} "
          f"questions.json, {len(image_files)} images", flush=True)

    # ── 1. corpus size ───────────────────────────────────────────────────────
    # "Rounds" in the paper = one utterance by one speaker, i.e. exactly what we
    # call a MESSAGE. That makes the comparison to GroupMemBench direct.
    print("\n📥 reading sessions …", flush=True)
    turns_per_dialogue: Dict[str, int] = defaultdict(int)
    speakers_per_dialogue: Dict[str, set] = defaultdict(set)
    turns_with_image = 0
    total_turns = 0
    total_chars = 0
    for path in session_files:
        dialogue = path.split("/")[1]          # multi-party/dialogueN/scenes/...
        data = fetch_json(path)
        for turn in data.get("dialogue", []):
            total_turns += 1
            turns_per_dialogue[dialogue] += 1
            speakers_per_dialogue[dialogue].add(turn.get("role", ""))
            content = turn.get("content", {}) or {}
            total_chars += len(content.get("text", "") or "")
            if (content.get("image") or "").strip():
                turns_with_image += 1

    # ── 2. questions ─────────────────────────────────────────────────────────
    print("📥 reading questions …", flush=True)
    main_types: Counter = Counter()
    sub_types: Counter = Counter()
    difficulty: Counter = Counter()
    # Sub-type breakdown restricted to the cross-session bank, because that is
    # the only place a fact can be superseded BETWEEN sessions — i.e. the only
    # place our bi-temporal layer can possibly earn its keep.
    cross_sub_types: Counter = Counter()
    q_with_image = 0
    q_multi_session = 0
    q_total = 0
    temporal_q = 0
    for path in question_files:
        data = fetch_json(path)
        for q in data.get("questions", []):
            q_total += 1
            qt = q.get("question_type", {}) or {}
            main_types[qt.get("main_type", "?")] += 1
            sub = qt.get("sub_type", "?")
            sub_types[sub] += 1
            difficulty[q.get("difficulty", "?")] += 1
            if is_temporal(sub):
                temporal_q += 1
            if ((q.get("question", {}) or {}).get("image") or "").strip():
                q_with_image += 1
            # CROSS-SESSION detection — subtler than it looks.
            #
            # `answer_session` never holds more than one entry, so a naive
            # len(...) > 1 test reports 0% cross-session and is WRONG. The
            # dataset encodes scope by LOCATION instead: each dialogue has a
            # pseudo-session folder `session0` holding the questions that span
            # the whole dialogue ("In Session 1 X thought A; by Session 3 how had
            # he revised it?"). Those files carry answer_session == ['session0'],
            # a sentinel, not a real session. That is why there are 30
            # questions.json files for 25 real sessions: the extra 5 are one
            # session0 bank per multi-party dialogue.
            sessions = q.get("answer_session") or []
            if "session0" in sessions or "/session0/" in f"/{path}":
                q_multi_session += 1
                cross_sub_types[sub] += 1

    # ── report ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 66)
    print(f"📊 H2HMem — {SPLIT} subset")
    print("═" * 66)

    print(f"\n🗣️  CORPUS")
    print(f"   dialogues            : {len(turns_per_dialogue)}")
    print(f"   sessions             : {len(session_files)}")
    print(f"   turns (= messages)   : {total_turns}")
    print(f"   turns carrying image : {turns_with_image} "
          f"({100*turns_with_image/max(total_turns,1):.1f}%)")
    print(f"   images on disk       : {len(image_files)}")
    print(f"   total text           : {total_chars:,} chars "
          f"(~{total_chars//4:,} tokens)")
    print(f"\n   per dialogue:")
    for d in sorted(turns_per_dialogue):
        print(f"     {d:<12} {turns_per_dialogue[d]:>5} turns   "
              f"{len(speakers_per_dialogue[d])} speakers")

    print(f"\n❓ QUESTIONS  (total {q_total})")
    print(f"   need an image in the QUESTION itself : {q_with_image} "
          f"({100*q_with_image/max(q_total,1):.1f}%)")
    print(f"   CROSS-SESSION (the session0 bank)    : {q_multi_session} "
          f"({100*q_multi_session/max(q_total,1):.1f}%)")
    print(f"   ⭐ TEMPORAL types (our differentiator): {temporal_q} "
          f"({100*temporal_q/max(q_total,1):.1f}%)")

    print(f"\n   cross-session bank by sub type "
          f"(where supersession can actually happen):")
    for k, v in cross_sub_types.most_common():
        star = " ⭐" if is_temporal(k) else ""
        print(f"     {k:<40} {v:>5}{star}")

    print(f"\n   by main type:")
    for k, v in main_types.most_common():
        print(f"     {k:<28} {v:>5}")
    print(f"\n   by sub type:")
    for k, v in sub_types.most_common():
        star = " ⭐" if is_temporal(k) else ""
        print(f"     {k:<40} {v:>5}{star}")
    print(f"\n   by difficulty:")
    for k, v in difficulty.most_common():
        print(f"     {k:<12} {v:>5}")

    # ── the actual decision: what would ingest cost us? ──────────────────────
    # Measured on the Finance runs: ~44 s per 5-message window on an H100.
    windows = -(-total_turns // 5)             # ceiling division, window = 5
    hours = windows * 44 / 3600
    print("\n" + "═" * 66)
    print("⏱️  INGEST COST ESTIMATE (window=5, ~44 s/window measured on H100)")
    print(f"   windows              : {windows}")
    print(f"   estimated GPU time   : {hours:.1f} h")
    print(f"   GroupMemBench Finance: 6,002 windows ≈ 38 h  (needed 8 shards)")
    if hours < 20:
        print(f"   ✅ fits in ONE reservation — no sharding needed")
    else:
        print(f"   ⚠️  would need sharding like Finance did")
    print("═" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
