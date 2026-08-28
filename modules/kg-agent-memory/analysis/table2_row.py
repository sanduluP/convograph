#!/usr/bin/env python3
"""
table2_row.py — reproduce Table 2 of the GroupMemBench paper and append OUR rows.

WHAT THIS IS FOR
----------------
Table 2 of the paper reports accuracy (%) of eight memory systems across six query
categories, averaged over all four domains. We have measured our own Graphiti
pipeline on ONE domain (Finance). This script prints the published table verbatim
and appends our measured rows underneath it, so the comparison can be read at a
glance — with the caveats printed right next to the numbers rather than lost in a
conversation.

THE CALIBRATION IDEA — WHY WE ALWAYS PRINT OUR *BM25* ROW TOO
-------------------------------------------------------------
Our numbers come from a different harness than the paper's: different answering
model (Qwen3-30B via vLLM), different judge, our own prompts, top-k 10. So a raw
"our 28.1 beats their 27.10" comparison is not sound on its own — the two numbers
were produced by different machinery.

BM25 is the fix. It appears in BOTH the paper's table and our runs, and it is a
deterministic, parameter-free retriever: any difference between their BM25 row and
ours is attributable to the harness and the domain, NOT to the retriever. That
makes our BM25 row a per-column calibration anchor. Read the Δ row before reading
any of our other numbers — a column where our BM25 sits far from theirs is a column
where our absolute numbers cannot be compared to theirs at face value.

(As of 2026-08-12 the anchor showed four of six columns reproducing within ~3
points, with knowledge_update running ~15 points HOT and temporal ~30 points COLD.)

INPUTS
------
Our rows are read from result JSONLs, each line a scored question with a
`judge_answer` field of "correct" / "incorrect":
  results/cluster/graphiti_<group>_<qtype>.jsonl   -- the Graphiti row
  results/cluster/bm25_<group>_<qtype>.jsonl       -- the same-job BM25 control
Missing files are reported as "--" rather than silently skipped, so a half-finished
batch can never masquerade as a complete row.

Usage:
  python analysis/table2_row.py                       # markdown to stdout
  python analysis/table2_row.py --out table2.md       # also write a file
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

# ── The six query categories ─────────────────────────────────────────────────
# Our filenames use the benchmark's snake_case names; the paper's table uses
# prettier display names. Order matches the paper's column order exactly so the
# rows line up visually when pasted next to a screenshot of the table.
CATEGORIES: List[Tuple[str, str]] = [
    ("multi_hop", "Multi-Hop"),
    ("knowledge_update", "Update"),
    ("term_ambiguity", "Ambiguity"),
    ("user_implicit", "Implicit"),
    ("temporal", "Temporal"),
    ("abstention", "Abstention"),
]

# ── Table 2 of the paper, transcribed verbatim ───────────────────────────────
# Values are accuracy (%) averaged over all FOUR domains. Kept here as literal
# data (rather than re-derived) because they are a published result we are
# quoting, not something we can recompute. Column order matches CATEGORIES.
# Source: GroupMemBench (arXiv 2605.14498), Table 2.
PAPER_TABLE: List[Tuple[str, str, List[float], float]] = [
    # (section, method, [MH, Upd, Amb, Impl, Temp, Abst], published average)
    ("RAG-based Methods", "BM25", [40.11, 25.23, 14.15, 40.82, 54.94, 77.98], 43.22),
    ("RAG-based Methods", "text-embedding-3-large", [36.26, 23.36, 21.70, 46.94, 32.72, 75.23], 38.04),
    ("RAG-based Methods", "GraphRAG", [12.09, 14.02, 19.81, 14.29, 5.56, 66.97], 20.56),
    ("Agent Memory Systems", "Mem0", [21.98, 4.67, 11.32, 20.41, 16.67, 82.57], 25.73),
    ("Agent Memory Systems", "MemGPT", [22.53, 17.76, 20.75, 28.57, 12.42, 77.98], 28.15),
    ("Agent Memory Systems", "A-Mem", [35.16, 22.43, 26.42, 46.94, 23.46, 67.89], 35.10),
    ("Agent Memory Systems", "HippoRAG", [39.56, 27.10, 30.19, 42.86, 29.63, 75.23], 39.72),
    ("Agent Memory Systems", "Hindsight", [42.31, 17.76, 37.74, 40.82, 54.94, 77.06], 46.01),
]

# The paper's own BM25 row, pulled out by name — this is the calibration reference
# every one of our numbers gets compared against.
PAPER_BM25 = dict(zip([k for k, _ in CATEGORIES], PAPER_TABLE[0][2]))


def score_file(path: str) -> Optional[Tuple[int, int]]:
    """Return (n_correct, n_total) for one scored result JSONL, or None if absent.

    The judge writes a free-text verdict; everything that is not an explicit
    positive counts as incorrect. We normalise case/whitespace because the judge's
    output has drifted between runs ("correct" vs "Correct").
    """
    if not os.path.exists(path):
        return None
    correct = total = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            if str(row.get("judge_answer", "")).strip().lower() in {
                "correct", "yes", "true", "1",
            }:
                correct += 1
    # A file that exists but holds zero rows is a failed run, not a 0 % result.
    return (correct, total) if total else None


def collect_row(results_dir: str, retriever: str, group: str) -> Dict[str, Optional[Tuple[int, int]]]:
    """Gather one retriever's (correct, total) for every category."""
    out: Dict[str, Optional[Tuple[int, int]]] = {}
    for key, _ in CATEGORIES:
        out[key] = score_file(os.path.join(results_dir, f"{retriever}_{group}_{key}.jsonl"))
    return out


def pct(cell: Optional[Tuple[int, int]]) -> str:
    """Format a cell as 'NN.N' — or '--' when that category has not been run yet."""
    if cell is None:
        return "--"
    c, n = cell
    return f"{100.0 * c / n:.1f}"


def frac(cell: Optional[Tuple[int, int]]) -> str:
    """Format a cell as the raw fraction 'c/n', which n=15 categories badly need."""
    return "--" if cell is None else f"{cell[0]}/{cell[1]}"


def averages(row: Dict[str, Optional[Tuple[int, int]]]) -> Tuple[str, str]:
    """Return (macro, micro) averages over the categories that actually ran.

    Two averages, because they answer different questions and the paper does not
    say which one its 'Average' column uses:
      macro = mean of the six per-category percentages (each category weighs the
              same, regardless of how many questions it has)
      micro = total correct / total questions (each QUESTION weighs the same, so
              the 48-question multi_hop dominates the 15-question user_implicit)
    Neither reproduces the paper's published average exactly from its own row, so
    both are printed and labelled rather than one being presented as definitive.
    """
    present = [c for c in row.values() if c is not None]
    if not present:
        return "--", "--"
    macro = sum(100.0 * c / n for c, n in present) / len(present)
    micro = 100.0 * sum(c for c, _ in present) / sum(n for _, n in present)
    return f"{macro:.1f}", f"{micro:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/cluster",
                    help="where the scored JSONLs live (default: results/cluster)")
    ap.add_argument("--group", default="gmb_finance_full",
                    help="group_id embedded in the result filenames")
    ap.add_argument("--out", default=None, help="also write the markdown here")
    args = ap.parse_args()

    graphiti = collect_row(args.results_dir, "graphiti", args.group)
    bm25 = collect_row(args.results_dir, "bm25", args.group)

    lines: List[str] = []
    add = lines.append

    add("## Table 2 (GroupMemBench, arXiv 2605.14498) + our Finance rows")
    add("")
    header = "| Method | " + " | ".join(d for _, d in CATEGORIES) + " | Average |"
    add(header)
    add("|---" * (len(CATEGORIES) + 2) + "|")

    # ── the published rows, verbatim ────────────────────────────────────────
    section = None
    for sec, method, vals, avg in PAPER_TABLE:
        if sec != section:
            add(f"| *{sec}* |" + " |" * (len(CATEGORIES) + 1))
            section = sec
        add(f"| {method} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {avg:.2f} |")

    # ── our rows ────────────────────────────────────────────────────────────
    add(f"| *Ours — Finance domain only* |" + " |" * (len(CATEGORIES) + 1))

    b_macro, b_micro = averages(bm25)
    add("| **BM25 (our harness)** | "
        + " | ".join(pct(bm25[k]) for k, _ in CATEGORIES)
        + f" | {b_macro} |")

    g_macro, g_micro = averages(graphiti)
    add("| **Graphiti (ours)** | "
        + " | ".join(pct(graphiti[k]) for k, _ in CATEGORIES)
        + f" | {g_macro} |")

    # The calibration row: how far our BM25 sits from the paper's BM25, per column.
    # This is the row that tells a reader which of our columns are comparable.
    add("| *Δ our BM25 − paper BM25* | "
        + " | ".join(
            "--" if bm25[k] is None
            else f"{100.0 * bm25[k][0] / bm25[k][1] - PAPER_BM25[k]:+.1f}"
            for k, _ in CATEGORIES)
        + " | |")
    add("")

    # ── raw counts, so nobody has to trust a percentage on n=15 ─────────────
    add("### Raw counts (ours)")
    add("")
    add("| Retriever | " + " | ".join(d for _, d in CATEGORIES) + " |")
    add("|---" * (len(CATEGORIES) + 1) + "|")
    add("| BM25 | " + " | ".join(frac(bm25[k]) for k, _ in CATEGORIES) + " |")
    add("| Graphiti | " + " | ".join(frac(graphiti[k]) for k, _ in CATEGORIES) + " |")
    add("")
    add(f"Micro-average (per question): BM25 {b_micro}, Graphiti {g_micro}. "
        f"Macro (per category): BM25 {b_macro}, Graphiti {g_macro}.")
    add("")

    # ── the caveats, printed WITH the table so they cannot be separated ─────
    add("> **Read this before comparing.** Our two rows cover the **Finance domain "
        "only** (214 questions); every published row is averaged over all four "
        "domains. They also come from a different harness — Qwen3-30B as both "
        "answering agent and judge, our own prompts, top-k 10 — so absolute values "
        "are not directly comparable to the published ones.")
    add(">")
    add("> The **Δ row** is the calibration: it is how far our BM25 lands from the "
        "paper's BM25, per column. BM25 is deterministic and parameter-free, so that "
        "gap measures the harness plus the domain, not the retriever. A column with a "
        "small Δ is one where our numbers can be read against the paper's; a column "
        "with a large Δ is not.")
    add(">")
    add("> The graph Graphiti was scored on is at **96.8 % window coverage** (192 of "
        "6,002 windows failed extraction and were skipped), so its row is a lower bound.")

    md = "\n".join(lines)
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(md + "\n")
        print(f"\n📝 written to {args.out}")


if __name__ == "__main__":
    main()
