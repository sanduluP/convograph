#!/usr/bin/env python3
"""
fact_redundancy.py — how many DISTINCT ideas are in a set of extracted facts?

WHY
---
Hop 2's planner was dropping ~37 of 40 facts, which read as the planner being
too selective. Reading the dropped list said otherwise:

    [0] User_12 requests Vendor to post the endpoint, cadence, and named owner...
    [2] User_12 requires Vendor to post the endpoint in the spec by EOD July 16.
    [5] User_12 requires Vendor to post endpoint, cadence, and named owner by...

Those are one statement extracted several times. So the question is not "is the
planner too aggressive" but "how many of these 40 facts are actually different
from each other" — a MODULE 2 question about extraction, measured from module
3's input.

The number matters beyond the board: a retrieval benchmark over a fact set that
is 4x redundant is measuring something different from what it claims to, because
one retrieved idea can occupy four result slots.

METHOD
------
Token-set Jaccard on content words, single-link clustering above a threshold.
Deliberately lexical rather than embedding-based: these near-duplicates differ by
a word or two, so they are trivially separable without a model, and a lexical
measure is reproducible with no GPU, no API key, and no drift between runs. It
UNDER-counts redundancy (two paraphrases with no shared words look distinct), so
the figure it reports is a floor, not an estimate.

Usage:
    python analysis/fact_redundancy.py --plan <run_dir>/plan.json
    python analysis/fact_redundancy.py --group-id gmb_finance_full --windows 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

# Words carrying no distinguishing content here. Kept SHORT on purpose: an
# aggressive stop list would manufacture similarity that is not there.
STOP = {
    "the", "a", "an", "to", "in", "on", "at", "by", "of", "for", "and", "or",
    "is", "are", "be", "that", "this", "it", "with", "as", "from", "will",
}


def tokens(fact: str) -> set[str]:
    """Content words, lowercased. URLs collapse to one token: two facts citing
    the same SharePoint page differ by the page, not by every path segment."""
    fact = re.sub(r"https?://\S+", " URL ", fact)
    return {w for w in re.findall(r"[a-zA-Z_0-9]+", fact.lower())
            if w not in STOP and len(w) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster(facts: list[str], threshold: float) -> list[list[int]]:
    """Single-link clustering: i and j join if they are similar enough, and
    clusters merge transitively. Single-link is the right choice for detecting
    RESTATEMENT — a chain a~b~c means all three describe one idea even if a and
    c share little directly."""
    toks = [tokens(f) for f in facts]
    parent = list(range(len(facts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            if jaccard(toks[i], toks[j]) >= threshold:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(facts)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def load_facts(args) -> tuple[list[dict], str]:
    if args.plan:
        with open(args.plan) as fh:
            plan = json.load(fh)
        return plan.get("_facts", []), f"plan {args.plan}"

    module_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = os.path.join(module_root, ".venv", "bin", "python")
    proc = subprocess.run(
        [py, os.path.join(module_root, "ui_ingest.py"), "--query-only",
         "--group-id", args.group_id, "--windows", str(args.windows),
         "--max-facts", str(args.max_facts)],
        cwd=module_root, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise SystemExit(f"query failed:\n{proc.stderr[-2000:]}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    return result.get("facts", []), f"group {args.group_id}, {args.windows} window(s)"


def dedupe(records: list[dict], threshold: float = 0.6) -> tuple[list[dict], int]:
    """Collapse restatements, keeping one record per idea. Returns (kept, removed).

    WHICH MEMBER SURVIVES, and why:
      1. a SUPERSEDED one (invalid_at set) if the cluster has any — a fact the
         conversation later overturned is the one worth putting on a board;
      2. otherwise the LONGEST, because the restatement pattern is list
         decomposition ("post primary, backup, trigger text and artifact type"
         emitted alongside one fact per item), and the combined member is the
         one that carries the whole statement.

    Order is preserved by the surviving member's original index, so downstream
    fact indices still read in conversation order.
    """
    facts = [r["fact"] for r in records]
    kept_idx = []
    for group in cluster(facts, threshold):
        superseded = [i for i in group if records[i].get("invalid_at")]
        pool = superseded or group
        kept_idx.append(max(pool, key=lambda i: len(facts[i])))
    kept_idx.sort()
    return [records[i] for i in kept_idx], len(records) - len(kept_idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default=None, help="a run's plan.json (reads _facts)")
    ap.add_argument("--group-id", default=None, help="query the graph directly instead")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--max-facts", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="Jaccard above which two facts are the same idea (0.6)")
    args = ap.parse_args()
    if not args.plan and not args.group_id:
        ap.error("give --plan or --group-id")

    records, source = load_facts(args)
    facts = [r["fact"] for r in records]
    if not facts:
        raise SystemExit("no facts to analyse")

    groups = cluster(facts, args.threshold)
    n, k = len(facts), len(groups)

    print(f"📥 source        : {source}")
    print(f"🔢 facts         : {n}")
    print(f"🧠 distinct ideas: {k}  (single-link Jaccard ≥ {args.threshold})")
    print(f"📉 redundancy    : {n / k:.2f}x — each idea appears {n / k:.2f} times "
          f"on average")
    print(f"♻️  {n - k} of {n} facts ({100 * (n - k) / n:.0f}%) restate another fact\n")

    shown = [g for g in groups if len(g) > 1]
    print(f"── the {len(shown)} restated idea(s), largest first ──")
    for g in shown[:8]:
        print(f"\n  ×{len(g)}")
        for i in g:
            print(f"    [{i}] {facts[i][:104]}")
    singles = [g for g in groups if len(g) == 1]
    print(f"\n  …plus {len(singles)} fact(s) that stand alone")

    print(f"\n💡 A planner keeping ~{k} of {n} is DEDUPLICATING, not over-selecting.")


if __name__ == "__main__":
    main()
