#!/usr/bin/env python3
"""
score_digest.py — does the digest actually surface what the meeting was about?

WHY THIS EXISTS
---------------
"The digest board looks better than the window board" is an impression. The
GroupMemBench source ships labels that turn it into a measurement, and they are
labels nobody wrote for us — they are how the corpus was generated:

    topic                    what the phase was about (1 per phase here)
    decision_type=="changed" the decision the phase reversed, with
    decision_change_metadata  original_decision -> changed_to spelled out

That second one is exactly what Q1's revision spine claims to recover. So the
question stops being "does this look right" and becomes "is the ground-truth
revision in the digest's top N, and where".

WHAT IS AND IS NOT MEASURED
---------------------------
Matching is LEXICAL — token-set F1 over content words, the same approach
fact_redundancy uses. Deliberately not an embedding model: this has to be
reproducible with no GPU, no API key and no drift between runs, and the failure
mode is the safe one. A paraphrase with no shared vocabulary scores 0 here even
though a human would call it a hit, so every number this prints is a FLOOR.

A per-phase score is also weak on its own: one revision and one topic is a
sample of one. The output is built to aggregate across phases, which is what
makes it a table for a paper rather than an anecdote.

Usage:
    python analysis/score_digest.py \\
        --digest ui/output/<run>/digest/digest.json \\
        --ground-truth tmp/phase_prod_deploy.ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.fact_redundancy import tokens  # noqa: E402  (one tokeniser, reused)


def f1(a: set[str], b: set[str]) -> float:
    """Token-set F1. Symmetric, and it does not reward a long text for
    containing a short one — which plain overlap would, so a 40-word digest
    entry would 'match' every short ground-truth line."""
    if not a or not b:
        return 0.0
    hit = len(a & b)
    if not hit:
        return 0.0
    p, r = hit / len(b), hit / len(a)
    return 2 * p * r / (p + r)


def best_match(target: str, candidates: list[str]) -> tuple[int, float]:
    """(index, score) of the candidate closest to target. (-1, 0.0) if none."""
    t = tokens(target)
    best, best_i = 0.0, -1
    for i, c in enumerate(candidates):
        s = f1(t, tokens(c))
        if s > best:
            best, best_i = s, i
    return best_i, best


def score(digest: dict, gt: dict, threshold: float = 0.25) -> dict:
    """Compare one digest against one phase's ground truth."""
    q = digest.get("queries", {})
    out: dict = {"phase": gt.get("phase"), "channel": gt.get("channel"),
                 "threshold": threshold}

    # ── Q1: is the phase's REVERSED decision in the revision spine, and where? ──
    # Rank matters as much as presence. A board shows a handful of revisions, so
    # a ground-truth hit at position 24 of 25 would not reach the board even
    # though recall counts it.
    revisions = q.get("revisions", [])
    rev_rows = [f"{r.get('was','')} {r.get('became','') or ''}" for r in revisions]
    rev_scores = []
    for g in gt.get("revisions", []):
        target = " ".join(str(x) for x in
                          (g.get("original_decision"), g.get("changed_to")) if x)
        i, s = best_match(target, rev_rows)
        rev_scores.append({
            "ground_truth": target[:160],
            "found": bool(s >= threshold),
            "rank": (i + 1) if s >= threshold else None,
            "score": round(s, 3),
            "matched": rev_rows[i][:160] if i >= 0 else None,
        })
    hits = [r for r in rev_scores if r["found"]]
    out["revisions"] = {
        "n_ground_truth": len(rev_scores),
        "n_found": len(hits),
        "recall": round(len(hits) / len(rev_scores), 3) if rev_scores else None,
        "best_rank": min((r["rank"] for r in hits), default=None),
        "n_returned": len(revisions),
        "detail": rev_scores,
    }

    # ── Q2: do the digest's topics cover the phase's labelled topic? ───────────
    # The two vocabularies do not line up by construction — ground truth says
    # "Deployment and Operations Readiness" while the graph says "compliance",
    # "owner", "timestamp". A topic counts as covered if ANY digest topic or its
    # neighbours overlap it, because a board names an idea through a cluster of
    # entities rather than one string.
    topic_rows = []
    for t in q.get("topics", []):
        topic_rows.append(" ".join([t.get("topic", "")]
                                   + list(t.get("neighbours") or [])))
    topic_scores = []
    for g in gt.get("topics", []):
        i, s = best_match(g["topic"], topic_rows)
        topic_scores.append({
            "ground_truth": g["topic"],
            "covered": bool(s >= threshold),
            "rank": (i + 1) if s >= threshold else None,
            "score": round(s, 3),
            "matched": topic_rows[i][:120] if i >= 0 else None,
        })
    covered = [t for t in topic_scores if t["covered"]]
    out["topics"] = {
        "n_ground_truth": len(topic_scores),
        "n_covered": len(covered),
        "coverage": round(len(covered) / len(topic_scores), 3) if topic_scores else None,
        "n_returned": len(q.get("topics", [])),
        "detail": topic_scores,
    }

    # ── Q3: participants are checkable exactly — names, not prose ─────────────
    gt_people = {p["author"] for p in gt.get("participants", [])}
    got = {p["speaker"] for p in q.get("participants", [])}
    out["participants"] = {
        "n_ground_truth": len(gt_people),
        "n_found": len(gt_people & got),
        "recall": round(len(gt_people & got) / len(gt_people), 3) if gt_people else None,
        "spurious": sorted(got - gt_people),
        "missed": sorted(gt_people - got),
    }
    return out


def render(s: dict) -> str:
    L = [f"# Digest score — {s['channel']} / {s['phase']}", ""]
    r, t, p = s["revisions"], s["topics"], s["participants"]

    L += ["## Q1 · revision spine", ""]
    L.append(f"- ground-truth revisions: **{r['n_ground_truth']}**, "
             f"found **{r['n_found']}** (recall {r['recall']}), "
             f"out of {r['n_returned']} returned")
    if r["best_rank"]:
        L.append(f"- best hit at **rank {r['best_rank']}** of {r['n_returned']}")
    for d in r["detail"]:
        mark = "✅" if d["found"] else "❌"
        L.append(f"  - {mark} score {d['score']} — {d['ground_truth'][:110]}")
        if d["matched"]:
            L.append(f"    matched: {d['matched'][:110]}")

    L += ["", "## Q2 · topics", ""]
    L.append(f"- ground-truth topics: **{t['n_ground_truth']}**, "
             f"covered **{t['n_covered']}** (coverage {t['coverage']})")
    for d in t["detail"]:
        mark = "✅" if d["covered"] else "❌"
        L.append(f"  - {mark} score {d['score']} — {d['ground_truth']}")
        if d["matched"]:
            L.append(f"    by: {d['matched'][:110]}")

    L += ["", "## Q3 · participants", ""]
    L.append(f"- **{p['n_found']}/{p['n_ground_truth']}** found "
             f"(recall {p['recall']})")
    # This number is worth far less than the two above it and should not be
    # quoted beside them. Speaker names repeat across the whole corpus, so a
    # digest of a COMPLETELY DIFFERENT slice still scored 5/7 here (measured as
    # a negative control, 2026-09-08, while Q1 and Q2 correctly scored 0). It
    # says the participant parser works; it says nothing about whether the
    # digest is about this meeting.
    L.append("  - ⚠️ weak metric: speaker names recur corpus-wide, so an "
             "unrelated digest still scores well. Q1 and Q2 are the real tests.")
    if p["spurious"]:
        L.append(f"- spurious: {', '.join(p['spurious'])}")
    if p["missed"]:
        L.append(f"- missed: {', '.join(p['missed'])}")

    L += ["", "> Matching is lexical token-set F1, so a paraphrase sharing no "
          "vocabulary scores 0. Every number here is a FLOOR."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--digest", required=True, help="a run's digest/digest.json")
    ap.add_argument("--ground-truth", required=True,
                    help="the phase's .ground_truth.json from extract_phase.py")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="token-set F1 above which two texts are the same thing")
    ap.add_argument("--out", default=None, help="write the score JSON here")
    args = ap.parse_args()

    with open(args.digest) as fh:
        digest = json.load(fh)
    with open(args.ground_truth) as fh:
        gt = json.load(fh)

    s = score(digest, gt, args.threshold)
    print(render(s))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(s, fh, indent=2)
        print(f"📝 {args.out}")


if __name__ == "__main__":
    main()
