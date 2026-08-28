"""Summarize typed-question evaluation results across (baseline, question_type).

Walks a results directory containing files named ``{baseline}__{type}.jsonl``
(written by ``eval_typed_baselines.sh``). For each file, counts how many
records have ``judge_answer`` parsed as Correct vs Incorrect and prints a
Markdown accuracy table.

The judge writes a free-form line that ends with "Final: Correct" or
"Final: Incorrect" (per [prompts/hipporag_judge_system.txt]); the baselines
already extract that into ``judge_answer``. We treat ``judge_answer`` matching
``Correct`` (case-insensitive, after stripping non-letters) as a hit and
everything else as a miss. Records with empty/missing ``judge_answer`` count
as misses but are also reported in a "skipped" column.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, Tuple

_NORM_RE = re.compile(r"[^a-zA-Z]+")


def _is_correct(judge_answer: str) -> bool:
    norm = _NORM_RE.sub("", (judge_answer or "")).lower()
    return norm == "correct"


def _scan(results_dir: str) -> Dict[Tuple[str, str], Dict[str, int]]:
    out: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0, "missing_judge": 0})
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".jsonl"):
            continue
        if "__" not in name:
            continue
        baseline, rest = name.split("__", 1)
        qtype = rest[:-len(".jsonl")]
        path = os.path.join(results_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                judge = rec.get("judge_answer", "") or ""
                key = (baseline, qtype)
                out[key]["total"] += 1
                if not judge.strip():
                    out[key]["missing_judge"] += 1
                if _is_correct(judge):
                    out[key]["correct"] += 1
    return out


def _format_markdown(stats, baselines, qtypes) -> str:
    header = "| baseline | " + " | ".join(qtypes) + " | overall |"
    sep = "|" + "---|" * (len(qtypes) + 2)
    lines = [header, sep]
    for b in baselines:
        row = [b]
        total_correct = 0
        total_total = 0
        for t in qtypes:
            s = stats.get((b, t), {"correct": 0, "total": 0})
            if s["total"] == 0:
                row.append("—")
            else:
                acc = s["correct"] / s["total"]
                row.append(f"{acc * 100:.1f}% ({s['correct']}/{s['total']})")
            total_correct += s["correct"]
            total_total += s["total"]
        if total_total == 0:
            row.append("—")
        else:
            row.append(f"**{total_correct / total_total * 100:.1f}% ({total_correct}/{total_total})**")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_tsv(stats, baselines, qtypes) -> str:
    header = ["baseline"] + list(qtypes) + ["overall_correct", "overall_total", "overall_accuracy"]
    lines = ["\t".join(header)]
    for b in baselines:
        row = [b]
        total_correct = 0
        total_total = 0
        for t in qtypes:
            s = stats.get((b, t), {"correct": 0, "total": 0})
            row.append(f"{s['correct']}/{s['total']}" if s["total"] else "")
            total_correct += s["correct"]
            total_total += s["total"]
        row.append(str(total_correct))
        row.append(str(total_total))
        row.append(f"{total_correct / total_total:.4f}" if total_total else "")
        lines.append("\t".join(row))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="Dir containing {baseline}__{type}.jsonl files")
    ap.add_argument("--out-markdown", default=None, help="Optional path to write Markdown table")
    ap.add_argument("--out-tsv", default=None, help="Optional path to write TSV")
    ap.add_argument(
        "--baselines",
        default="mem0,hindsight,hipporag,memgpt",
        help="Comma-separated baseline names in display order",
    )
    ap.add_argument(
        "--question-types",
        default="multi_hop,knowledge_update,temporal,user_implicit,term_ambiguity,abstention",
        help="Comma-separated question types in display order",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"results dir not found: {args.results_dir}")
        return 2

    stats = _scan(args.results_dir)
    baselines = [s.strip() for s in args.baselines.split(",") if s.strip()]
    qtypes = [s.strip() for s in args.question_types.split(",") if s.strip()]

    md = _format_markdown(stats, baselines, qtypes)
    print(md)
    print()

    missing_judge_total = sum(s["missing_judge"] for s in stats.values())
    if missing_judge_total:
        print(f"# {missing_judge_total} records had empty judge_answer (counted as Incorrect).")

    if args.out_markdown:
        os.makedirs(os.path.dirname(args.out_markdown) or ".", exist_ok=True)
        with open(args.out_markdown, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"wrote markdown -> {args.out_markdown}")

    if args.out_tsv:
        os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
        with open(args.out_tsv, "w", encoding="utf-8") as f:
            f.write(_format_tsv(stats, baselines, qtypes) + "\n")
        print(f"wrote tsv -> {args.out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
