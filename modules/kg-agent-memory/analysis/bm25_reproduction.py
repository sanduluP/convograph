#!/usr/bin/env python3
"""
bm25_reproduction.py — can our harness reproduce the paper's BM25 row?

THE QUESTION THIS ANSWERS
-------------------------
We want to put our Graphiti number next to Table 2 of the GroupMemBench paper
(arXiv 2605.14498). That is only meaningful if our measuring instrument agrees
with theirs. BM25 is the instrument check: it appears in both tables, and it is
deterministic and parameter-free. It has no temperature, no prompt, no embedding
model, no index tuning. So every point of difference between their BM25 row and
ours comes from the HARNESS (answering model, judge model, prompts, top-k) or from
the DOMAIN mix — never from the retriever itself.

Run it on all four domains, average, and compare to their published 43.22:
  - if we land near 43.22, our harness is calibrated and our other numbers can be
    read against the paper's directly;
  - if we land far from it, the offset is the harness, and NO absolute number of
    ours (including Graphiti's) can be compared to the paper without stating it.

WHY THE AVERAGE IS COMPUTED TWO WAYS
------------------------------------
The paper does not state how its "Average" column is formed, and neither the macro
nor the micro average of its own BM25 row reproduces the printed 43.22 exactly
(macro gives 42.20). So we print both and label them, rather than silently picking
whichever is flattering:
  macro = mean of the six per-category percentages (each CATEGORY weighs equally)
  micro = total correct / total questions (each QUESTION weighs equally)

INPUT
-----
  results/bm25_<Domain>/<qtype>.jsonl   -- one scored question per line,
                                           `judge_answer` is "correct"/"incorrect"
Domains with no results directory are reported as missing rather than skipped, so
a partial batch can never be mistaken for a complete four-domain average.

Usage:
  python analysis/bm25_reproduction.py
  python analysis/bm25_reproduction.py --out reports/bm25_reproduction.md
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

# Column order matches the paper's table exactly, so our rows can be read directly
# against a screenshot of it. Keys are our filenames; labels are the paper's names.
CATEGORIES: List[Tuple[str, str]] = [
    ("multi_hop", "Multi-Hop"),
    ("knowledge_update", "Update"),
    ("term_ambiguity", "Ambiguity"),
    ("user_implicit", "Implicit"),
    ("temporal", "Temporal"),
    ("abstention", "Abstention"),
]

DOMAINS = ["Finance", "Healthcare", "Manufacturing", "Technology"]

# The paper's BM25 row from Table 2 (accuracy %, over the union of all four
# domains). Quoted data, not something we can recompute.
PAPER_BM25: Dict[str, float] = {
    "multi_hop": 40.11,
    "knowledge_update": 25.23,
    "term_ambiguity": 14.15,
    "user_implicit": 40.82,
    "temporal": 54.94,
    "abstention": 77.98,
}
PAPER_BM25_PUBLISHED_AVG = 43.22

# The paper's PER-DOMAIN BM25 rows, from Tables 7-10 in Appendix G. These matter
# more than the Table 2 row above: comparing our Finance numbers against a
# four-domain average confounds "our harness differs" with "Finance differs from
# the other three". Comparing Finance to Finance removes that confound entirely.
# Order of values matches CATEGORIES. Last value is the paper's printed average.
PAPER_BM25_BY_DOMAIN: Dict[str, Tuple[List[float], float]] = {
    "Technology":    ([43.90, 30.60, 23.30, 35.70, 73.00, 78.60], 46.00),  # Table 7
    "Finance":       ([47.90, 34.40, 11.10, 53.30, 46.70, 75.90], 42.10),  # Table 8
    "Healthcare":    ([37.50, 11.80,  0.00, 100.00, 45.90, 89.50], 43.10),  # Table 9
    "Manufacturing": ([31.10, 13.60,  0.00,  0.00, 55.80, 72.70], 41.10),  # Table 10
}

# Cells where the paper's own per-domain sample is so small that the value is
# degenerate (exact 0 % or 100 %). The paper explicitly flags this for Healthcare's
# Ambiguity and Implicit columns. A delta against such a cell is not evidence of a
# harness difference, so we mark it rather than let it inflate the story.
def _is_degenerate(v: float) -> bool:
    return v in (0.0, 100.0)

# Cell type used throughout: (n_correct, n_total), or None when not measured.
Cell = Optional[Tuple[int, int]]


def score_file(path: str) -> Cell:
    """Return (correct, total) for one scored JSONL, or None if it is absent/empty.

    An existing-but-empty file means a run that died, which is not the same thing
    as a genuine 0 % — so it returns None rather than (0, 0).
    """
    if not os.path.exists(path):
        return None
    correct = total = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            verdict = str(json.loads(line).get("judge_answer", "")).strip().lower()
            if verdict in {"correct", "yes", "true", "1"}:
                correct += 1
    return (correct, total) if total else None


def collect(results_root: str) -> Dict[str, Dict[str, Cell]]:
    """domain -> category -> (correct, total)."""
    return {
        d: {k: score_file(os.path.join(results_root, f"bm25_{d}", f"{k}.jsonl"))
            for k, _ in CATEGORIES}
        for d in DOMAINS
    }


def pool(cells: List[Cell]) -> Cell:
    """Sum a list of cells, ignoring the missing ones. None if nothing is present.

    Used to pool one category ACROSS domains. Pooling counts (rather than
    averaging percentages) is the honest way to combine unequal domain sizes when
    reporting a micro figure.
    """
    present = [c for c in cells if c is not None]
    if not present:
        return None
    return (sum(c for c, _ in present), sum(n for _, n in present))


def pct(cell: Cell) -> str:
    return "--" if cell is None else f"{100.0 * cell[0] / cell[1]:.2f}"


def macro_micro(cells: List[Cell]) -> Tuple[Optional[float], Optional[float]]:
    """(macro, micro) over whichever cells are present."""
    present = [c for c in cells if c is not None]
    if not present:
        return None, None
    macro = sum(100.0 * c / n for c, n in present) / len(present)
    micro = 100.0 * sum(c for c, _ in present) / sum(n for _, n in present)
    return macro, micro


def fmt(x: Optional[float]) -> str:
    return "--" if x is None else f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default=None, help="also write the markdown here")
    args = ap.parse_args()

    data = collect(args.results_root)
    keys = [k for k, _ in CATEGORIES]
    labels = [lab for _, lab in CATEGORIES]

    lines: List[str] = []
    add = lines.append

    add("## BM25 reproduction — our harness vs. GroupMemBench Table 2")
    add("")
    add("BM25 is deterministic and parameter-free, so any gap below is the "
        "**harness** (answering model, judge, prompts, top-k) or the **domain mix** "
        "— never the retriever.")
    add("")

    # ── per-domain, ours vs theirs, domain-matched ──────────────────────────
    # This is the primary comparison. Appendix G gives the paper's BM25 per
    # domain, so each of our rows is checked against the SAME domain rather than
    # against a four-domain average.
    add("### Per domain — ours vs. the paper's own per-domain BM25 (Appendix G)")
    add("")
    add("| Domain | " + " | ".join(labels) + " | Micro | n |")
    add("|---" * (len(labels) + 3) + "|")
    for d in DOMAINS:
        cells = [data[d][k] for k in keys]
        ma, mi = macro_micro(cells)
        n = sum(c[1] for c in cells if c is not None)
        paper_vals, paper_avg = PAPER_BM25_BY_DOMAIN[d]

        add(f"| **{d}** — paper | "
            + " | ".join(f"{v:.2f}" + (" ⚠️" if _is_degenerate(v) else "")
                         for v in paper_vals)
            + f" | {paper_avg:.2f} | |")

        flag = "" if all(c is not None for c in cells) else " ⚠️ incomplete"
        add(f"| {d} — ours{flag} | " + " | ".join(pct(c) for c in cells)
            + f" | {fmt(mi)} | {n or '--'} |")

        # The delta row is suppressed where the paper's cell is degenerate (0/100
        # on a handful of questions) — a delta there measures their sample size,
        # not our harness.
        add(f"| *Δ* | " + " | ".join(
            "n/a" if _is_degenerate(pv) else
            ("--" if c is None else f"{100.0 * c[0] / c[1] - pv:+.2f}")
            for pv, c in zip(paper_vals, cells))
            + (f" | {mi - paper_avg:+.2f} | |" if mi is not None else " | -- | |"))
    add("")
    add("⚠️ = the paper's own cell is exactly 0 % or 100 %, i.e. degenerate on a "
        "very small per-domain sample (it flags this for Healthcare). Deltas "
        "against those cells are omitted — they would measure their sample size, "
        "not our harness.")
    add("")

    # ── the pooled four-domain row, which is what the paper reports ─────────
    pooled = [pool([data[d][k] for d in DOMAINS]) for k in keys]
    p_macro, p_micro = macro_micro(pooled)
    have_all = all(
        data[d][k] is not None for d in DOMAINS for k in keys)

    add("### All domains vs. the paper")
    add("")
    add("| BM25 | " + " | ".join(labels) + " | Average |")
    add("|---" * (len(labels) + 2) + "|")
    add("| Paper (Table 2) | " + " | ".join(f"{PAPER_BM25[k]:.2f}" for k in keys)
        + f" | {PAPER_BM25_PUBLISHED_AVG:.2f} |")
    # Micro, not macro: Appendix G states the Table 2 averages are micro-averages
    # over the union of the filtered evaluation set across the four domains.
    add("| **Ours (pooled)** | " + " | ".join(pct(c) for c in pooled)
        + f" | {fmt(p_micro)} |")
    add("| *Δ (ours − paper)* | " + " | ".join(
        "--" if c is None else f"{100.0 * c[0] / c[1] - PAPER_BM25[k]:+.2f}"
        for k, c in zip(keys, pooled)) + " | |")
    add("")

    # Appendix G: "The cross-domain micro-averages summarized in Table 2 are
    # computed over the union of the filtered evaluation set across these four
    # domains." So micro is the like-for-like figure; macro is printed only as a
    # secondary reference.
    add(f"Our pooled **micro {fmt(p_micro)}** vs the paper's **"
        f"{PAPER_BM25_PUBLISHED_AVG:.2f}** (macro, for reference: {fmt(p_macro)}). "
        f"Table 2's averages are micro-averages over the union of the filtered "
        f"evaluation set (Appendix G), so micro is the like-for-like number.")
    add("")

    if not have_all:
        add("> ⚠️ **Incomplete.** Some domain/category cells are missing, so the "
            "pooled row is not yet a four-domain average. Do not quote it.")
        add("")

    # ── the verdict, stated as a rule for reading every other number ────────
    # Three tiers rather than a pass/fail: the harness swaps GPT-5 for a 30B open
    # model on BOTH the answering agent and the judge, so an exact match was never
    # the expectation. What matters is the SIZE of the offset and whether it is
    # stated whenever one of our absolute numbers is quoted.
    if p_micro is not None:
        delta = p_micro - PAPER_BM25_PUBLISHED_AVG
        if abs(delta) <= 3:
            verdict = ("✅ **Close reproduction.**", "Absolute numbers can be read "
                       "against the paper's with the per-column deltas noted.")
        elif abs(delta) <= 8:
            verdict = ("🟡 **Approximate reproduction.**", "Close enough to treat "
                       "the paper's numbers as a sanity reference, but every "
                       "absolute number of ours should be quoted WITH this offset.")
        else:
            verdict = ("⛔ **Not reproduced.**", "Our absolute numbers cannot be "
                       "placed in Table 2 at all without stating this offset.")
        head, tail = verdict
        add(f"> {head} Our pooled BM25 micro is **{delta:+.1f} points** from the "
            f"paper's {PAPER_BM25_PUBLISHED_AVG:.2f}, using a different agent and "
            f"judge (Qwen3-30B vs GPT-5). {tail} The offset-free comparison is "
            f"always ours-vs-ours: any retriever against OUR BM25, measured in the "
            f"same harness.")
    add("")

    # ── raw counts, because several categories have small n ────────────────
    add("### Raw counts (pooled across domains)")
    add("")
    add("| " + " | ".join(labels) + " |")
    add("|---" * len(labels) + "|")
    add("| " + " | ".join("--" if c is None else f"{c[0]}/{c[1]}" for c in pooled) + " |")

    md = "\n".join(lines)
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(md + "\n")
        print(f"\n📝 written to {args.out}")


if __name__ == "__main__":
    main()
