"""
Which CHANNEL does each GroupMemBench question actually live in?

WHY THIS EXISTS
---------------
The 2026-08-03 full-AML run truncated the Finance corpus to its first 4,877
messages (= exactly the "AML (Anti-Money Laundering) Project" channel) so that a
Graphiti KG could be built in one 12 h job. But the QUESTION set was NOT
truncated: all 32 knowledge_update questions were still asked.

If most of those questions are about the OTHER five channels, then their evidence
simply is not in the corpus and NO retriever can answer them. That would put a
hard ceiling on both the Graphiti score (15.6%) and the BM25 control (18.8%) and
make the whole comparison uninformative — the two systems would mostly be tying
at "impossible".

This script measures that ceiling. It has no LLM in the loop and no ground-truth
channel label to lean on (the questions ship only id/question/answer/asking_user_id),
so it uses BM25: score each question's text against each channel treated as one
big document, and take the argmax as the channel the question is "about".

Reading the output
------------------
  * "questions whose best channel is AML" is the OPTIMISTIC ceiling — the most
    questions any retriever could plausibly get right on the truncated corpus.
  * If that number is close to 6/32, then BM25's 6/32 is already AT the ceiling
    and Graphiti's 5/32 is one question behind it, not 20 points behind.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi


# Tokeniser matching the one the bm25 baseline uses: lowercase alphanumeric runs.
# Kept deliberately dumb so this audit cannot flatter itself with clever NLP.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase the text and keep alphanumeric runs — nothing else."""
    return _TOKEN_RE.findall(text.lower())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conversation-json", required=True,
                    help="the synthetic_domain_channels_*.json the eval loads")
    ap.add_argument("--questions-jsonl", required=True,
                    help="one question per line: id / question / answer")
    ap.add_argument("--truncated-channel", default="AML (Anti-Money Laundering) Project",
                    help="the channel the truncated corpus actually contains")
    args = ap.parse_args()

    # ── 1. group every message by its channel ────────────────────────────────
    # The loader flattens the JSON into per-message dicts carrying a "_channel"
    # key; here we only need the raw text per channel, so we walk the JSON
    # directly rather than importing the eval's loader (fewer moving parts).
    print(f"📥 reading corpus  {args.conversation_json}", flush=True)
    with open(args.conversation_json) as fh:
        raw = json.load(fh)

    # The file is a dict of channel-name -> list-of-message-records. Each record
    # holds the utterance under "content" (older dumps used "text"), so accept both.
    per_channel_tokens: dict[str, list[str]] = defaultdict(list)
    per_channel_count: dict[str, int] = defaultdict(int)
    for channel, records in raw.items():
        for rec in records:
            body = rec.get("content") or rec.get("text") or ""
            per_channel_tokens[channel].extend(tokenize(body))
            per_channel_count[channel] += 1

    channels = sorted(per_channel_tokens)
    print(f"✅ {len(channels)} channels, "
          f"{sum(per_channel_count.values())} messages total", flush=True)
    for ch in channels:
        marker = "  ← truncated corpus" if ch == args.truncated_channel else ""
        print(f"   • {per_channel_count[ch]:>5} msgs  {ch}{marker}")

    # ── 2. one BM25 index whose "documents" are whole channels ───────────────
    # Six documents, one per channel. Scoring a question against this index asks
    # "which channel's vocabulary does this question look like?" — exactly the
    # question we need answered.
    bm25 = BM25Okapi([per_channel_tokens[ch] for ch in channels])

    # ── 3. attribute each question to its argmax channel ─────────────────────
    print(f"\n📥 reading questions  {args.questions_jsonl}", flush=True)
    questions = [json.loads(line) for line in open(args.questions_jsonl)]
    print(f"✅ {len(questions)} questions\n", flush=True)

    hits: dict[str, int] = defaultdict(int)
    in_scope_ids: list[str] = []
    for q in questions:
        # Score the question AND its gold answer together: the answer text names
        # the entities the evidence messages talk about, which is a much stronger
        # channel signal than the question alone.
        probe = tokenize(f"{q['question']} {q.get('answer', '')}")
        scores = bm25.get_scores(probe)
        best = channels[max(range(len(channels)), key=lambda i: scores[i])]
        hits[best] += 1
        if best == args.truncated_channel:
            in_scope_ids.append(q["id"])

    # ── 4. report ────────────────────────────────────────────────────────────
    print("📊 questions attributed per channel")
    for ch in sorted(hits, key=lambda c: -hits[c]):
        marker = "  ← IN SCOPE for the truncated run" if ch == args.truncated_channel else ""
        print(f"   {hits[ch]:>3}  {ch}{marker}")

    n_in = len(in_scope_ids)
    n_all = len(questions)
    print(f"\n🎯 CEILING for the truncated corpus: {n_in}/{n_all} "
          f"({100.0 * n_in / n_all:.1f}%)")
    print(f"   in-scope question ids: {', '.join(in_scope_ids) or '(none)'}")
    print("\n   Any score above this ceiling is luck (a guess that happened to")
    print("   match); any score at it means the retriever found everything that")
    print("   was findable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
