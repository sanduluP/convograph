#!/usr/bin/env python3
"""Compare two retrievers on the SAME questions — properly, with a paired test.

WHY THIS EXISTS
---------------
"Graphiti 28.1 % vs BM25 46.9 %" is a difference of six questions out of 32. On a
sample that small, the honest question is not "which is bigger" but "could this
gap be noise?". Reporting a raw percentage gap to a supervisor without a paired
test is how a project talks itself into a wrong conclusion — in either direction.

Both retrievers answered the IDENTICAL question set, so the comparison is PAIRED
and McNemar's exact test is the right tool: it looks only at the questions where
the two systems DISAGREE, which is where all the information about a difference
actually lives.

It also prints the per-question disagreement table, because with n=32 the useful
next step is reading the ~10 questions that differ, not staring at the average.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> List[dict]:
    """Read a run's JSONL result file (one record per question, in order)."""
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def is_correct(rec: dict) -> bool:
    """The harness writes the judge's verdict as free text ('correct'/'incorrect')."""
    return str(rec.get("judge_answer", "")).strip().lower().startswith("correct")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two DISCORDANT counts.

    b = A right / B wrong, c = A wrong / B right. Under H0 each discordant pair is
    a fair coin, so the count follows Binomial(b+c, 0.5). Concordant pairs (both
    right, both wrong) carry no information about a difference and are excluded —
    that is the whole point of a paired test.
    """
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % interval for a proportion — correct at small n, unlike normal
    approximation, which happily produces intervals outside [0, 1] here."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    if len(sys.argv) >= 3:
        path_a, path_b = Path(sys.argv[1]), Path(sys.argv[2])
        name_a, name_b = path_a.stem.split("_")[0], path_b.stem.split("_")[0]
    else:
        base = REPO_ROOT / "results" / "cluster"
        path_a = base / "graphiti_gmb_finance_full_knowledge_update.jsonl"
        path_b = base / "bm25_gmb_finance_full_knowledge_update.jsonl"
        name_a, name_b = "graphiti", "bm25"

    a, b_recs = load(path_a), load(path_b)
    if len(a) != len(b_recs):
        print(f"❌ different question counts: {len(a)} vs {len(b_recs)}")
        return 1

    # Pair on the query text, not on file order — order equality is an assumption
    # worth verifying rather than trusting.
    by_query_b: Dict[str, dict] = {r.get("query", ""): r for r in b_recs}
    if len(by_query_b) != len(b_recs):
        print("⚠️  duplicate queries in the second file; falling back to file order")
        by_query_b = {}

    both_right = a_only = b_only = neither = 0
    disagreements = []
    for i, rec_a in enumerate(a):
        rec_b = by_query_b.get(rec_a.get("query", ""), b_recs[i]) if by_query_b \
            else b_recs[i]
        ca, cb = is_correct(rec_a), is_correct(rec_b)
        if ca and cb:
            both_right += 1
        elif ca and not cb:
            a_only += 1
            disagreements.append((i + 1, name_a, rec_a.get("query", "")))
        elif cb and not ca:
            b_only += 1
            disagreements.append((i + 1, name_b, rec_a.get("query", "")))
        else:
            neither += 1

    n = len(a)
    ka = sum(1 for r in a if is_correct(r))
    kb = sum(1 for r in b_recs if is_correct(r))
    lo_a, hi_a = wilson(ka, n)
    lo_b, hi_b = wilson(kb, n)
    p = mcnemar_exact(a_only, b_only)

    print("═" * 68)
    print(f"  {name_a}  vs  {name_b}      (n = {n}, paired)")
    print("═" * 68)
    print(f"\n{name_a:>10}: {ka}/{n} = {ka/n:.1%}   95% CI [{lo_a:.1%}, {hi_a:.1%}]")
    print(f"{name_b:>10}: {kb}/{n} = {kb/n:.1%}   95% CI [{lo_b:.1%}, {hi_b:.1%}]")
    print(f"\n  contingency (the paired view):")
    print(f"    both correct            : {both_right}")
    print(f"    only {name_a:<18}: {a_only}")
    print(f"    only {name_b:<18}: {b_only}")
    print(f"    neither                 : {neither}")
    print(f"\n  McNemar exact p = {p:.4f}   "
          f"({'SIGNIFICANT' if p < 0.05 else 'NOT significant'} at α=0.05)")
    print(f"  discordant pairs = {a_only + b_only} — all the evidence about a "
          f"difference lives here")

    if p >= 0.05:
        print(f"\n  ⚠️  With {a_only + b_only} discordant pairs, this sample cannot "
              f"distinguish\n      the two systems. Report the direction, not a "
              f"verdict.")

    print(f"\n  questions where they disagree (read these, not the average):")
    for idx, winner, query in disagreements:
        print(f"    Q{idx:<3} won by {winner:<10} {query[:64]}")

    # Overlap in CIs is the other honest way to say the same thing.
    if not (hi_a < lo_b or hi_b < lo_a):
        print(f"\n  note: the two 95% CIs OVERLAP — another sign the gap is not "
              f"resolved at this n.")
    print("═" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
