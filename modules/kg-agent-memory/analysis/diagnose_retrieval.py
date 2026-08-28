#!/usr/bin/env python3
"""Why did a retriever lose? Separate RETRIEVAL failure from ANSWERING failure.

THE QUESTION THIS ANSWERS
-------------------------
Graphiti scored 28.1 % against BM25's 46.9 % on the same 32 questions. "It is
worse" is not actionable. There are two completely different failures hiding
behind one number, and they have opposite fixes:

  RETRIEVAL failure  — the evidence was never handed to the agent. The graph, the
                       search, or the fact→message mapping is at fault. Fix the
                       retriever.
  ANSWERING failure  — the evidence WAS in the retrieved passages and the agent
                       still got it wrong. The retriever did its job. Fix the
                       prompt, the top-k, or the ordering.

GroupMemBench ships a gold `answer` string per question but no gold message id,
so we approximate "was the evidence retrieved?" by lexical overlap between the
gold answer's CONTENT words and the retrieved passages. That is a proxy, and it
is stated as one — but it is a sharp one here, because these answers name
concrete artefacts ("dual-key approval", "T+2 settlement") that appear verbatim
in the source messages.

Reported per retriever, and — most usefully — for the questions where the two
retrievers DISAGREE.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]

# Words that carry no evidence signal. Kept deliberately small: the goal is to
# strip grammar, not to guess which domain terms matter.
_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "by", "from",
    "as", "that", "this", "these", "those", "it", "its", "will", "would",
    "should", "can", "could", "has", "have", "had", "do", "does", "did", "not",
    "no", "yes", "we", "they", "he", "she", "you", "i", "our", "their", "team",
    "current", "currently", "approach", "now", "using", "use", "used", "which",
    "what", "how", "when", "who", "there", "then", "than", "so", "if", "all",
    "any", "each", "more", "most", "other", "some", "such", "only", "own",
    "same", "also", "into", "over", "after", "before", "during", "while",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def content_words(text: str) -> Set[str]:
    """Lowercased content tokens (stopwords and 1-char tokens dropped)."""
    return {t.lower() for t in _TOKEN_RE.findall(text or "")
            if len(t) > 1 and t.lower() not in _STOP}


def load(path: Path) -> List[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def is_correct(rec: dict) -> bool:
    return str(rec.get("judge_answer", "")).strip().lower().startswith("correct")


def coverage(gold: str, docs: List[str]) -> float:
    """Fraction of the gold answer's content words present in the passages."""
    gw = content_words(gold)
    if not gw:
        return 0.0
    dw = content_words(" ".join(docs))
    return len(gw & dw) / len(gw)


def summarise(name: str, recs: List[dict], gold_by_q: Dict[str, str],
              threshold: float) -> Dict[str, object]:
    """Split this run's questions into the four diagnostic buckets."""
    retrieved_and_right = retrieved_and_wrong = 0
    missed_and_right = missed_and_wrong = 0
    covs: List[float] = []

    for rec in recs:
        gold = gold_by_q.get(rec.get("query", ""), "")
        cov = coverage(gold, rec.get("retrieved_docs", []))
        covs.append(cov)
        found = cov >= threshold
        if found and is_correct(rec):
            retrieved_and_right += 1
        elif found:
            retrieved_and_wrong += 1
        elif is_correct(rec):
            missed_and_right += 1
        else:
            missed_and_wrong += 1

    n = len(recs)
    found_total = retrieved_and_right + retrieved_and_wrong
    print(f"\n  {name}")
    print(f"    mean gold-answer coverage in top-10 : "
          f"{sum(covs)/max(len(covs),1):.1%}")
    print(f"    evidence RETRIEVED (cov ≥ {threshold:.0%})     : "
          f"{found_total}/{n} = {found_total/n:.1%}")
    print(f"      ├─ and answered correctly         : {retrieved_and_right}")
    print(f"      └─ and answered WRONG             : {retrieved_and_wrong}  "
          f"← answering failure")
    print(f"    evidence NOT retrieved              : "
          f"{missed_and_right + missed_and_wrong}/{n}")
    print(f"      ├─ answered correctly anyway      : {missed_and_right}  "
          f"← parametric knowledge / lucky")
    print(f"      └─ answered wrong                 : {missed_and_wrong}  "
          f"← retrieval failure")
    return {
        "mean_coverage": sum(covs) / max(len(covs), 1),
        "retrieved": found_total,
        "retrieval_failures": missed_and_wrong,
        "answering_failures": retrieved_and_wrong,
        "n": n,
    }


def main() -> int:
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    base = REPO_ROOT / "results" / "cluster"
    g = load(base / "graphiti_gmb_finance_full_knowledge_update.jsonl")
    b = load(base / "bm25_gmb_finance_full_knowledge_update.jsonl")
    questions = load(REPO_ROOT / "questions" / "Finance" / "knowledge_update.jsonl")
    gold_by_q = {q["question"]: q["answer"] for q in questions}

    missing_gold = sum(1 for r in g if r.get("query", "") not in gold_by_q)
    print("═" * 70)
    print(f"  Retrieval diagnosis — knowledge_update, Finance (n = {len(g)})")
    print(f"  coverage threshold = {threshold:.0%} of gold-answer content words")
    print("═" * 70)
    if missing_gold:
        print(f"⚠️  {missing_gold} questions had no gold answer matched by text")

    sg = summarise("GRAPHITI", g, gold_by_q, threshold)
    sb = summarise("BM25", b, gold_by_q, threshold)

    print("\n" + "─" * 70)
    print("  VERDICT")
    print("─" * 70)
    dr = sg["retrieval_failures"] - sb["retrieval_failures"]
    da = sg["answering_failures"] - sb["answering_failures"]
    print(f"    Graphiti has {dr:+d} retrieval failures vs BM25")
    print(f"    Graphiti has {da:+d} answering failures vs BM25")
    if dr > da:
        print(f"\n    → The gap is mostly a RETRIEVAL problem: Graphiti is not "
              f"putting\n      the evidence in front of the agent as often as "
              f"BM25 does.")
    elif da > dr:
        print(f"\n    → The gap is mostly an ANSWERING problem: Graphiti retrieves "
              f"the\n      evidence but the agent fails to use it — look at "
              f"ordering / top-k.")
    else:
        print(f"\n    → Retrieval and answering contribute about equally.")

    print(f"\n    Mean coverage: Graphiti {sg['mean_coverage']:.1%} vs "
          f"BM25 {sb['mean_coverage']:.1%}")
    print("═" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
