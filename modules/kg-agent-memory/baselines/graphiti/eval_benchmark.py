#!/usr/bin/env python3
"""Graphiti temporal-KG RAG baseline for GroupMemBench.

WHERE THIS FITS
---------------
GroupMemBench evaluates a memory system by the SINGLE function it must provide:

    retrieve(query, k) -> List[int]     # indices into the channel's `messages` list

Everything else — the QA *agent* LLM, the *judge* LLM, and scoring — is shared
harness code in `baselines/rag_common/eval_lib.run_qa`, reused verbatim (that is
exactly why we build inside the GroupMemBench repo rather than a separate one).

TWO-STEP PLAN (this file is the step-1 scaffold)
------------------------------------------------
  • STEP 1 (now): `retrieve()` is a BM25 stub. Goal = prove the harness runs
    end-to-end on ONE channel and writes results — AND give us a first BM25
    number on the same channel to compare against later. No Graphiti yet.
  • STEP 2 (next): replace the stub body with the real Graphiti temporal-KG
    retriever (see the big TODO block in `build_retriever`). The differentiator
    lives there: filter to currently-VALID facts (invalid_at IS NULL) and map
    each fact back to its `source_msg` index — so `knowledge_update` questions
    get the latest decision and `temporal` questions get the resolved date.

This mirrors `baselines/bm25/eval_benchmark.py` deliberately: same CLI, same
`run_qa` call, so the comparison is apples-to-apples.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Set

# The harness uses package-internal imports; make the repo root importable so
# `python baselines/graphiti/eval_benchmark.py` works from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.rag_common.eval_lib import (
    load_conversation_messages,
    load_env_file,
    load_questions,
    message_index_text,
    read_text,
    run_qa,
)
from llm_utils import (
    create_chat_client,
    normalize_llm_provider,
    resolve_api_key,
    resolve_base_url,
)

# Defaults point at our local DFKI stack (serv-3306 qwen2.5:32b via the .env),
# NOT Azure gpt-5 — so the smoke test runs on-prem out of the box.
API_VERSION_DEFAULT = "2024-02-15-preview"
AGENT_MODEL_DEFAULT = "qwen2.5:32b"
JUDGE_MODEL_DEFAULT = "qwen2.5:32b"
RETRIEVE_TOP_K = 10

# Word-level tokenizer shared by the BM25 stub (kept identical to the BM25
# baseline so the step-1 numbers are directly comparable).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Lowercase word-level split; keeps numbers/dates/user-IDs intact, no
    stopword removal (fully deterministic, parameter-free)."""
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


# Grammar words carry no evidence signal. Kept identical to the list in
# `analysis/diagnose_retrieval.py` so the two analyses agree on what "overlap"
# means — a proxy is only defensible if it is applied consistently.
_STOP_WORDS = {
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


def _content_words(text: str) -> Set[str]:
    """Lowercased content tokens (grammar words and 1-char tokens dropped)."""
    return {t for t in tokenize(text) if len(t) > 1 and t not in _STOP_WORDS}


def build_bm25_retriever(messages: List[dict]) -> Callable[[str, int], List[int]]:
    """BM25 baseline retriever — a runnable placeholder that also gives us the
    lexical number the Graphiti retriever must beat (kept for apples-to-apples)."""
    from rank_bm25 import BM25Okapi  # local import: only needed for the stub

    corpus_tokens = [tokenize(message_index_text(m)) for m in messages]
    bm25 = BM25Okapi(corpus_tokens)
    print(f"[graphiti-stub] BM25 index built over {len(messages)} messages")

    def retrieve(query: str, k: int) -> List[int]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return order[:k]

    return retrieve


def wrap_channel_scoped(
    retrieve: Callable[[str, int], List[int]],
    messages: List[dict],
    rule: str,
    oversample: int,
    gold_by_question: Dict[str, str],
) -> Callable[[str, int], List[int]]:
    """Insert a CHANNEL-FILTERING step between retrieval and the QA agent.

    WHY
    ---
    The 2026-08-05 diagnosis showed Graphiti retrieves the evidence about as often
    as BM25 and then fails to *use* it, and that BM25 wins precisely when its ten
    passages come from ONE channel (1.73 distinct channels when correct vs 2.71
    when wrong). The six Finance channels are parallel projects sharing vocabulary,
    so an off-channel passage is a pure distractor.

    This wrapper tests that directly, without re-ingesting anything:

        retrieve k*oversample  →  pick ONE target channel  →  drop everything else
                               →  hand the top k survivors to the agent

    THE TWO RULES (why both)
    ------------------------
    ``oracle``    picks the channel using the GOLD answer. This is not a system —
                  it is a CEILING. It answers "if channel identification were
                  perfect, what would we score?". If the ceiling is not clearly
                  above the unscoped baseline, then building a channel classifier
                  is not worth anyone's time and we stop here. That negative result
                  is the point of running it.
    ``majority``  picks the channel most represented among the oversampled hits.
                  Uses no gold answer, so it is actually buildable today. The gap
                  between ``majority`` and ``oracle`` is how much a smarter
                  disambiguator could still buy us.

    NOTE ON WHY NOT `asking_user_id`: 10 of the 12 Finance users belong to more
    than one channel (three of them to five), so the asker does not identify a
    channel. That is also why per-channel ``group_id`` namespacing is not a fix.
    """
    # Precompute each message's channel once; the loader stores it as `_channel`.
    channel_of = [m.get("_channel", "") for m in messages]

    def pick_oracle(indices: List[int], gold: str) -> str:
        """Channel whose retrieved messages best cover the gold answer's wording.

        We do not have a gold message id in GroupMemBench, only a gold answer
        string, so "the right channel" is approximated by lexical coverage — the
        same proxy `analysis/diagnose_retrieval.py` already uses and defends.
        """
        gold_words = _content_words(gold)
        if not gold_words:
            return ""
        best, best_score = "", -1.0
        by_channel: Dict[str, List[int]] = {}
        for i in indices:
            by_channel.setdefault(channel_of[i], []).append(i)
        for chan, idxs in by_channel.items():
            text = " ".join(str(messages[i].get("content", "")) for i in idxs)
            score = len(gold_words & _content_words(text)) / len(gold_words)
            if score > best_score:
                best, best_score = chan, score
        return best

    def pick_majority(indices: List[int]) -> str:
        """Channel contributing the most of the oversampled hits (ties → best-ranked)."""
        counts: Dict[str, int] = {}
        for i in indices:
            counts[channel_of[i]] = counts.get(channel_of[i], 0) + 1
        if not counts:
            return ""
        top = max(counts.values())
        # Tie-break by rank: whichever tied channel appears earliest in the ranking.
        for i in indices:
            if counts[channel_of[i]] == top:
                return channel_of[i]
        return ""

    def wrapped(query: str, k: int) -> List[int]:
        # Oversample first — filtering a top-10 down to one channel would leave
        # 3-4 passages and conflate "scoping" with "fewer passages".
        wide = retrieve(query, k * oversample)
        if not wide:
            return wide

        if rule == "oracle":
            # run_qa passes "<asker> <question>"; recover the gold answer by
            # matching the question text as a suffix of the search query.
            gold = ""
            for qtext, ans in gold_by_question.items():
                if query.endswith(qtext):
                    gold = ans
                    break
            target = pick_oracle(wide, gold)
        else:
            target = pick_majority(wide)

        kept = [i for i in wide if channel_of[i] == target][:k]
        # Never hand the agent an empty context: if the filter wipes everything
        # out, fall back to the unfiltered ranking so the question is still
        # comparable to the baseline run.
        return kept if kept else wide[:k]

    return wrapped


def build_retriever(messages: List[dict], args) -> Callable[[str, int], List[int]]:
    """Dispatch to the requested retriever (`--retriever bm25|graphiti`).

    STEP 1 = bm25 (proves the harness + the number to beat).
    STEP 2b = graphiti (the real temporal-KG retriever in graphiti_retriever.py):
      windowed ingest → Graphiti hybrid search → currently-valid facts first →
      map each fact back to its precise source message index.
    """
    if args.retriever == "bm25":
        return build_bm25_retriever(messages)

    # The extraction-vs-retrieval ablation: same graph, same facts, but indexed by
    # fact TEXT instead of by entity embeddings. See kgfacts_retriever.py.
    if args.retriever == "kgfacts":
        from baselines.graphiti.kgfacts_retriever import build_kgfacts_retriever
        return build_kgfacts_retriever(
            messages, window=args.window, group_id=args.group_id)

    # Graphiti temporal-KG retriever. Imported lazily so a plain BM25 run never
    # needs graphiti-core / Neo4j installed.
    from baselines.graphiti.graphiti_retriever import build_graphiti_retriever

    # Optionally install the meeting-aware resolve_edge prompt (implicit-revision
    # contradiction detection) BEFORE the Graphiti client resolves any edge.
    if args.strict_prompt:
        from baselines.graphiti import prompts_override
        prompts_override.apply_overrides()

    # --window-range is given as "START:END" on the CLI; parse it to a tuple here
    # (1-based START, exclusive END) or None when the job ingests everything.
    window_range = None
    if args.window_range:
        lo, hi = args.window_range.split(":")
        window_range = (int(lo), int(hi))

    return build_graphiti_retriever(
        messages,
        window=args.window,
        group_id=args.group_id,
        max_ingest_messages=args.max_ingest_messages,
        window_range=window_range,
        resume=args.resume,
        retrieve_only=args.retrieve_only,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversation-json", required=True)
    ap.add_argument("--questions-jsonl", required=True)
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--agent-model", default=AGENT_MODEL_DEFAULT)
    ap.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    ap.add_argument("--agent-prompt", default="prompts/hipporag_agent_system.txt")
    ap.add_argument("--judge-prompt", default="prompts/hipporag_judge_system.txt")
    ap.add_argument("--retrieve-top-k", type=int, default=RETRIEVE_TOP_K)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--llm-provider", default=None)
    ap.add_argument("--ingest-only", action="store_true",
                    help="Build the retriever/index and exit without running QA.")
    # Smoke-test convenience (not in the BM25 baseline): cap the number of
    # questions so step-1 plumbing checks finish in a minute instead of hitting
    # the whole question file against the (slow) serv-3306 agent+judge.
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, run only the first N questions (0 = all).")
    # --- Step 2b: Graphiti temporal-KG retriever knobs -----------------------
    ap.add_argument("--retriever", choices=["bm25", "graphiti", "kgfacts"], default="bm25",
                    help="bm25 = lexical baseline; graphiti = temporal-KG retriever.")
    ap.add_argument("--window", type=int, default=10,
                    help="Graphiti: messages per episode (swept knob; 1 = one per msg).")
    ap.add_argument("--group-id", default="gmb",
                    help="Graphiti: Neo4j namespace for this KG (wiped + rebuilt each run).")
    ap.add_argument("--max-ingest-messages", type=int, default=0,
                    help="Graphiti: cap ingested messages for a fast smoke test (0 = all).")
    # FAIRNESS KNOB (applies to BOTH retrievers, unlike --max-ingest-messages).
    # Truncating the corpus GLOBALLY — before any retriever is built — is what
    # makes a scoped Graphiti run comparable to BM25: both index exactly the same
    # messages and both answer from the same evidence pool. Comparing Graphiti on
    # a 1k slice against BM25 on all 30k would be apples-to-oranges, since most
    # questions' evidence would simply be absent from the slice.
    ap.add_argument("--max-corpus-messages", type=int, default=0,
                    help="Truncate the corpus to the first N messages for ALL "
                         "retrievers (0 = all). Use for scope-matched comparisons.")
    # --- Parallel / resumable ingest -----------------------------------------
    # The Finance corpus needs ~38 h of GPU to ingest end-to-end, but the cluster
    # caps a reservation at 1 day. So we shard: every job loads the SAME full
    # corpus (identical message indices, identical global window numbering) and
    # ingests only its own slice of windows. Six ~13 h jobs run at once instead of
    # one impossible 38 h job.
    ap.add_argument("--window-range", default=None, metavar="START:END",
                    help="Graphiti: ingest only global windows [START, END) "
                         "(1-based, END exclusive). Omit to ingest all.")
    ap.add_argument("--resume", action="store_true",
                    help="Graphiti: do NOT wipe the group; skip windows whose "
                         "episode already exists. Required for sharded runs (a "
                         "wipe would delete a sibling shard) and for restarting "
                         "after a walltime kill.")
    ap.add_argument("--retrieve-only", action="store_true",
                    help="Graphiti: SCORE an already-built graph. Ingests "
                         "nothing — just maps existing episodes back to message "
                         "indices, then runs QA. Implies --resume. Use this "
                         "against the merged full-corpus store; without it the "
                         "run would pay to fill every window ingest skipped.")
    # --- channel scoping (the 2026-08-05 contamination experiment) -----------
    # `none` reproduces every run so far. `oracle` measures the CEILING using the
    # gold answer; `majority` is the buildable version that uses no gold data.
    ap.add_argument("--channel-scope", choices=["none", "oracle", "majority"],
                    default="none",
                    help="filter retrieval to a single channel before answering")
    ap.add_argument("--channel-oversample", type=int, default=3,
                    help="retrieve k*N before filtering, so the surviving set is "
                         "still k passages deep (default 3)")
    ap.add_argument("--strict-prompt", action="store_true",
                    help="Graphiti: install the meeting-aware resolve_edge (implicit-"
                         "revision) contradiction prompt before ingest.")
    args = ap.parse_args()

    load_env_file(args.env_file)
    llm_provider = normalize_llm_provider(args.llm_provider)
    base_endpoint = resolve_base_url(llm_provider)
    api_key = resolve_api_key(llm_provider)

    messages = load_conversation_messages(args.conversation_json)
    print(f"[graphiti] loaded {len(messages)} messages")

    # Scope-match the corpus BEFORE building any retriever, so the retriever and
    # the passages shown to the QA agent come from the identical message pool.
    # (The loader sorts by (_channel, timestamp, msg_node), so the first N
    # messages are a contiguous slice of the alphabetically-first channel — i.e.
    # one real conversation, not a random mix.)
    if args.max_corpus_messages and args.max_corpus_messages > 0:
        messages = messages[: args.max_corpus_messages]
        channels = {m.get("_channel", "") for m in messages}
        print(f"[graphiti] --max-corpus-messages={args.max_corpus_messages} → "
              f"corpus scoped to {len(messages)} messages "
              f"across {len(channels)} channel(s): {sorted(channels)}")

    retrieve = build_retriever(messages, args)

    if args.ingest_only:
        print("[graphiti] ingest-only: retriever built, exiting before QA")
        return 0

    questions = load_questions(args.questions_jsonl)
    # Optionally truncate for a fast smoke test (see --limit).
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
        print(f"[graphiti] --limit={args.limit} → running {len(questions)} questions only")
    # Channel scoping is applied HERE, not in build_retriever, because the oracle
    # rule needs the gold answers and the questions are only loaded above.
    if args.channel_scope != "none":
        gold_by_question = {q["question"]: q.get("answer", "") for q in questions}
        retrieve = wrap_channel_scoped(
            retrieve,
            messages,
            rule=args.channel_scope,
            oversample=args.channel_oversample,
            gold_by_question=gold_by_question,
        )
        print(f"[graphiti] channel scoping ON — rule='{args.channel_scope}', "
              f"oversample={args.channel_oversample}x "
              f"(retrieve {args.retrieve_top_k * args.channel_oversample}, "
              f"keep top {args.retrieve_top_k} from one channel)", flush=True)

    agent_system = read_text(args.agent_prompt)
    judge_system = read_text(args.judge_prompt)
    client = create_chat_client(
        provider=llm_provider,
        azure_endpoint=base_endpoint,
        base_url=base_endpoint,
        api_version=API_VERSION_DEFAULT,
        api_key=api_key,
    )

    correct, total = run_qa(
        questions=questions,
        messages=messages,
        retrieve=retrieve,
        top_k=args.retrieve_top_k,
        client=client,
        agent_model=args.agent_model,
        judge_model=args.judge_model,
        agent_system=agent_system,
        judge_system=judge_system,
        output_jsonl=args.output_jsonl,
    )

    accuracy = correct / total if total else 0.0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
