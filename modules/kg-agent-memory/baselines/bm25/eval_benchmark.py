#!/usr/bin/env python3
"""BM25 RAG baseline.

Indexes one document per message (content only — metadata is attached when
the passage is shown to the gpt-5 agent, not at retrieval time). Pure CPU,
no model calls during ingest.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rank_bm25 import BM25Okapi

from baselines.rag_common.eval_lib import (
    call_chat,  # noqa: F401  (kept available for callers)
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

API_VERSION_DEFAULT = "2024-02-15-preview"
AGENT_MODEL_DEFAULT = "gpt-5"
JUDGE_MODEL_DEFAULT = "gpt-5"
RETRIEVE_TOP_K = 10

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Lowercase + word-level split. Keeps numbers, dates, user IDs intact;
    no stopword removal so the tokenization is fully deterministic and
    parameter-free."""
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


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
                    help="Build the BM25 index and exit without running QA.")
    args = ap.parse_args()

    load_env_file(args.env_file)
    llm_provider = normalize_llm_provider(args.llm_provider)
    base_endpoint = resolve_base_url(llm_provider)
    api_key = resolve_api_key(llm_provider)

    messages = load_conversation_messages(args.conversation_json)
    print(f"[bm25] loaded {len(messages)} messages")

    corpus_tokens = [tokenize(message_index_text(m)) for m in messages]
    bm25 = BM25Okapi(corpus_tokens)
    print(f"[bm25] index built (vocab~={sum(len(t) for t in corpus_tokens)} tokens)")

    if args.ingest_only:
        print("[bm25] ingest-only: index built, exiting before QA")
        return 0

    questions = load_questions(args.questions_jsonl)
    agent_system = read_text(args.agent_prompt)
    judge_system = read_text(args.judge_prompt)
    client = create_chat_client(
        provider=llm_provider,
        azure_endpoint=base_endpoint,
        base_url=base_endpoint,
        api_version=API_VERSION_DEFAULT,
        api_key=api_key,
    )

    def retrieve(query: str, k: int) -> List[int]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = bm25.get_scores(q_tokens)
        # argsort descending, take top-k
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return order[:k]

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
