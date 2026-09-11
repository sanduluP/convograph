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

HOW MATCHING WORKS, AND WHY IT CHANGED
-------------------------------------
SEMANTIC, via bge-m3 on unicorn. It was lexical token-set F1 first, chosen for
reproducibility, and that was the wrong call: the two sides share MEANING, not
words. Ground truth is a human-written label ("Deployment and Operations
Readiness"); the digest speaks the graph's vocabulary ("monitoring, Ops,
runbook, deployment support"). Measured 2026-09-08 on the real phase, the
lexical scorer rated the CORRECT revision at 0.151 and the CORRECT topic at
0.20, both under its own 0.25 threshold, and reported recall 0.0 for a digest
that had actually found the right material.

A lexical fallback remains for when the embedder is unreachable, and the output
always says which mode produced the numbers — the two are not comparable and a
mixed table would be worse than no table.

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
from analysis.fact_redundancy import tokens  # noqa: E402  (lexical fallback)

# bge-m3, the same embedder Graphiti uses for this corpus — so "similar" means
# the same thing here as it does inside the graph.
EMBED_URL = os.getenv("GRAPHITI_EMBED_BASE_URL", "http://localhost:11435/v1")
EMBED_MODEL = os.getenv("GRAPHITI_EMBED_MODEL", "bge-m3")


def embed(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch. None if the embedder is unreachable, so the caller can
    fall back rather than fail — a scorer that needs a tunnel to run at all is
    a scorer that will not be run."""
    import json as _json
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{EMBED_URL.rstrip('/')}/embeddings",
            data=_json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer ollama"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = _json.loads(r.read())
        return [d["embedding"] for d in
                sorted(body["data"], key=lambda d: d["index"])]
    except Exception:                                          # noqa: BLE001
        return None


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


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


def best_match(target: str, candidates: list[str],
               vectors: dict | None = None) -> tuple[int, float]:
    """(index, score) of the candidate closest to target. (-1, 0.0) if none.

    Uses the precomputed embedding table when there is one, cosine on bge-m3;
    otherwise token-set F1. The scales are NOT comparable — cosine on this model
    puts unrelated text around 0.4-0.6, so the thresholds differ by mode.
    """
    if vectors is not None:
        tv = vectors.get(target)
        if tv is not None:
            best, best_i = 0.0, -1
            for i, c in enumerate(candidates):
                cv = vectors.get(c)
                if cv is None:
                    continue
                sc = cosine(tv, cv)
                if sc > best:
                    best, best_i = sc, i
            return best_i, best
    t = tokens(target)
    best, best_i = 0.0, -1
    for i, c in enumerate(candidates):
        sc = f1(t, tokens(c))
        if sc > best:
            best, best_i = sc, i
    return best_i, best


# Thresholds are per-MODE because the two scales have nothing to do with each
# other. Token-set F1 is ~0 for unrelated text; bge-m3 cosine puts unrelated
# text around 0.4-0.6 because every sentence shares the same embedding
# neighbourhood of "English about work". 0.70 was chosen against the negative
# control, not the positive one: the unrelated corpus digest scores below it.
DEFAULT_THRESHOLD = {"embedding": 0.70, "lexical": 0.25}


def score(digest: dict, gt: dict, threshold: float | None = None) -> dict:
    """Compare one digest against one phase's ground truth."""
    q = digest.get("queries", {})

    # ── embed everything in ONE batch ───────────────────────────────────────
    # Collected first so the embedder is called once rather than per comparison:
    # the alternative is O(candidates x ground truth) round trips through an SSH
    # tunnel, which is slow enough that nobody would run the scorer twice.
    rev_rows = [f"{r.get('was','')} {r.get('became','') or ''}"
                for r in q.get("revisions", [])]
    topic_rows = [" ".join([t.get("topic", "")] + list(t.get("neighbours") or []))
                  for t in q.get("topics", [])]
    gt_revs = [" ".join(str(x) for x in
                        (g.get("original_decision"), g.get("changed_to")) if x)
               for g in gt.get("revisions", [])]
    gt_topics = [g["topic"] for g in gt.get("topics", [])]

    corpus = [t for t in (rev_rows + topic_rows + gt_revs + gt_topics) if t]
    vecs = embed(corpus) if corpus else None
    vectors = dict(zip(corpus, vecs)) if vecs else None
    mode = "embedding" if vectors else "lexical"
    threshold = threshold if threshold is not None else DEFAULT_THRESHOLD[mode]

    out: dict = {"phase": gt.get("phase"), "channel": gt.get("channel"),
                 "threshold": threshold, "mode": mode,
                 "embed_model": EMBED_MODEL if vectors else None}

    # ── Q1: is the phase's REVERSED decision in the revision spine, and where? ──
    # Rank matters as much as presence. A board shows a handful of revisions, so
    # a ground-truth hit at position 24 of 25 would not reach the board even
    # though recall counts it.
    revisions = q.get("revisions", [])
    rev_scores = []
    for target in gt_revs:
        i, s = best_match(target, rev_rows, vectors)
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
    topic_scores = []
    for gt_topic in gt_topics:
        i, s = best_match(gt_topic, topic_rows, vectors)
        topic_scores.append({
            "ground_truth": gt_topic,
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
    L = [f"# Digest score — {s['channel']} / {s['phase']}", "",
         f"matching: **{s['mode']}**"
         + (f" (`{s['embed_model']}`)" if s.get("embed_model") else "")
         + f", threshold {s['threshold']}", ""]
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

    if s["mode"] == "lexical":
        L += ["", "> ⚠️ The embedder was unreachable, so this fell back to lexical "
              "token-set F1 — which rates a correct paraphrase at ~0.15 and calls "
              "it a miss. Start the tunnel and re-run before quoting anything."]
    else:
        L += ["", f"> Cosine on `{s['embed_model']}`. Unrelated work text sits "
              f"around 0.4-0.6 on this model, so {s['threshold']} is the bar, not 0."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--digest", required=True, help="a run's digest/digest.json")
    ap.add_argument("--ground-truth", required=True,
                    help="the phase's .ground_truth.json from extract_phase.py")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the per-mode default (0.70 embedding, 0.25 lexical)")
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
