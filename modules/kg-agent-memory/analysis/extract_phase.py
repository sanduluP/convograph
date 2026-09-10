#!/usr/bin/env python3
"""
extract_phase.py — pull ONE meeting-sized unit out of the GroupMemBench source.

WHY THIS EXISTS
---------------
Every board we have made so far was planned from an arbitrary 2-window slice of
a SIX-WEEK corpus. `gmb_finance_full` is not a meeting: 5,810 episodes,
2025-06-19 to 2025-07-28, six parallel project channels. Summarising that is not
the task; summarising a meeting is.

The source JSON has the structure we threw away at ingestion. It is
`{channel: [messages]}`, and each message carries `phase_name`, `topic`,
`is_decision_point`, `decision_type` and `decision_change_metadata`. So a
meeting-sized, coherent, BOUNDED unit already exists: one channel × one phase,
typically 260-630 messages (~50-130 windows).

AND IT COMES WITH GROUND TRUTH
------------------------------
That is the part that matters. `topic` gives the topics a digest should
surface; `decision_type == "changed"` with `decision_change_metadata`
(`original_decision` -> `changed_to`) gives the revisions our supersession query
tries to recover. So a digest built on this unit can be SCORED, not eyeballed.

Note `is_decision_point` is true for ~66% of messages — too generous to be a
label. `decision_type == "changed"` is the sharp one, and it is rare (11 in a
5,205-message channel), which is itself worth knowing before we promise a
revision-heavy board.

Output is the plain-text `Speaker: utterance` format `ui_ingest.py` reads, plus
a sidecar JSON holding the ground truth for that phase.

Usage:
    python analysis/extract_phase.py --list
    python analysis/extract_phase.py --channel "Treasury Management System Implementation" \\
        --phase "Production Deployment Readiness" --out tmp/phase.txt
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import os

DEFAULT_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "final", "Finance",
    "synthetic_domain_channels_rolevariants_Finance.json")


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def phase_messages(data: dict, channel: str, phase: str) -> list[dict]:
    msgs = [m for m in data.get(channel, []) if m.get("phase_name") == phase]
    # Chronological, with msg_node as the tiebreak — the same ordering the
    # benchmark loaders use, so our unit is the benchmark's unit.
    msgs.sort(key=lambda m: (m.get("timestamp", ""), m.get("msg_node", "")))
    return msgs


def ground_truth(msgs: list[dict]) -> dict:
    """The labels a digest can be scored against."""
    changed = [m for m in msgs
               if str(m.get("decision_type", "")).lower() == "changed"]
    revisions = []
    for m in changed:
        meta = m.get("decision_change_metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta.replace("'", '"'))
            except (json.JSONDecodeError, AttributeError):
                meta = {"raw": meta}
        revisions.append({
            "msg_node": m.get("msg_node"),
            "timestamp": m.get("timestamp"),
            "author": m.get("author"),
            "decision_label": m.get("decision_label"),
            "original_decision": (meta or {}).get("original_decision"),
            "changed_to": (meta or {}).get("changed_to"),
        })
    return {
        "topics": [{"topic": t, "messages": n}
                   for t, n in collections.Counter(
                       m.get("topic") for m in msgs).most_common()],
        "participants": [{"author": a, "role": r, "messages": n}
                         for (a, r), n in collections.Counter(
                             (m.get("author"), m.get("role")) for m in msgs).most_common()],
        "revisions": revisions,
        "n_messages": len(msgs),
        "n_decision_points": sum(
            1 for m in msgs if str(m.get("is_decision_point", "")).lower() == "true"),
    }


def render(msgs: list[dict]) -> list[str]:
    """`Author: content`, one line per message — ui_ingest's input format.

    Speakers stay IN the text on purpose. Speaker-freeness is not achieved by
    removing names: that was tried and moved person-rooted facts from 89.6% to
    90.4%. It is achieved by GRAPHITI_EXCLUDE_SPEAKERS, a hard type filter
    enforced inside graphiti-core. Stripping the names here as well would only
    destroy the attribution a reader needs, for no measured benefit.
    """
    lines = []
    for m in msgs:
        text = html.unescape(str(m.get("content", ""))).strip()
        if not text:
            continue
        # Newlines inside a message would become extra "utterances" downstream.
        text = " ".join(text.split())
        who = m.get("author") or "Unknown"
        role = m.get("role")
        lines.append(f"{who} ({role}): {text}" if role else f"{who}: {text}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--list", action="store_true",
                    help="show channels and their phases with message counts")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--phase", default=None)
    ap.add_argument("--max-messages", type=int, default=0,
                    help="keep only the first N messages — for timing a small "
                         "slice before committing to the whole phase (0 = all)")
    ap.add_argument("--out", default=None, help="write the transcript here")
    args = ap.parse_args()

    data = load(args.src)

    if args.list or not (args.channel and args.phase):
        for ch, msgs in sorted(data.items(), key=lambda kv: -len(kv[1])):
            ph = collections.Counter(m.get("phase_name") for m in msgs)
            chg = sum(1 for m in msgs
                      if str(m.get("decision_type", "")).lower() == "changed")
            print(f"\n📁 {ch}  —  {len(msgs)} msgs, {len(ph)} phases, "
                  f"{chg} changed decision(s)")
            for name, n in ph.most_common():
                pm = [m for m in msgs if m.get("phase_name") == name]
                c = sum(1 for m in pm
                        if str(m.get("decision_type", "")).lower() == "changed")
                flag = f"  ⭐ {c} revision(s)" if c else ""
                print(f"     {n:>5} msgs · ~{-(-n // 5):>3} windows   {name}{flag}")
        if not (args.channel and args.phase):
            print("\nPick one with --channel and --phase.")
        return

    msgs = phase_messages(data, args.channel, args.phase)
    if not msgs:
        raise SystemExit(f"❌ no messages for channel={args.channel!r} "
                         f"phase={args.phase!r} — run --list")
    if args.max_messages:
        msgs = msgs[:args.max_messages]

    gt = ground_truth(msgs)
    lines = render(msgs)

    print(f"📁 {args.channel}")
    print(f"📑 {args.phase}")
    print(f"💬 {len(lines)} utterances → ~{-(-len(lines) // 5)} windows of 5")
    print(f"👥 {len(gt['participants'])} participants")
    print(f"🏷️  {len(gt['topics'])} topic(s): "
          + ", ".join(t["topic"] for t in gt["topics"]))
    print(f"🔄 {len(gt['revisions'])} ground-truth revision(s) "
          f"(decision_type == 'changed')")
    print(f"   {gt['n_decision_points']} of {gt['n_messages']} messages are "
          f"flagged is_decision_point — too broad to use as a label")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        gt_path = os.path.splitext(args.out)[0] + ".ground_truth.json"
        with open(gt_path, "w", encoding="utf-8") as fh:
            json.dump({"channel": args.channel, "phase": args.phase, **gt},
                      fh, indent=2)
        print(f"\n📝 transcript   : {args.out}")
        print(f"📝 ground truth : {gt_path}")


if __name__ == "__main__":
    main()
