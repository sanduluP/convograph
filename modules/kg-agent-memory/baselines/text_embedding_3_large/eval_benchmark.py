#!/usr/bin/env python3
"""Dense-embedding RAG baseline (text-embedding-3-large).

Indexes one document per message (content only). Embeddings are computed once
per conversation and cached on disk under ``--store-dir`` so re-runs across
question types skip the embed pass. Retrieval is exact cosine similarity over
the in-memory matrix — no ANN library needed at this scale (<= ~50k messages).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import AzureOpenAI, OpenAI
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.rag_common.eval_lib import (
    conversation_fingerprint,
    conversation_id_from_path,
    load_conversation_messages,
    load_env_file,
    load_questions,
    message_index_text,
    read_text,
    run_qa,
)
from llm_utils import (
    AZURE_OPENAI_PROVIDER,
    OPENAI_PROVIDER,
    create_chat_client,
    normalize_llm_provider,
    resolve_api_key,
    resolve_base_url,
    with_retry,
)

API_VERSION_DEFAULT = "2024-02-15-preview"
AGENT_MODEL_DEFAULT = "gpt-5"
JUDGE_MODEL_DEFAULT = "gpt-5"
EMBEDDING_MODEL_DEFAULT = "text-embedding-3-large"
EMBEDDING_DIMS_DEFAULT = 3072
EMBEDDING_BATCH_DEFAULT = 64
RETRIEVE_TOP_K = 10
INGEST_CONFIG_VERSION = "rag_dense_raw_message_v1"


def normalize_embedding_provider(provider: Optional[str]) -> str:
    value = (provider or os.environ.get("EMBEDDING_PROVIDER") or AZURE_OPENAI_PROVIDER).strip().lower()
    aliases = {
        "azure": AZURE_OPENAI_PROVIDER,
        "azure_openai": AZURE_OPENAI_PROVIDER,
        "aoai": AZURE_OPENAI_PROVIDER,
        "openai": OPENAI_PROVIDER,
        "oai": OPENAI_PROVIDER,
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError(
            f"Unsupported embedding provider: {provider!r}. Expected azure_openai or openai."
        )
    return normalized


def build_embedding_client(provider: str, default_api_version: str) -> Tuple[Any, str]:
    """Returns (client, deployment_or_model_string)."""
    if provider == AZURE_OPENAI_PROVIDER:
        endpoint = (
            os.environ.get("EMBEDDING_AZURE_ENDPOINT")
            or os.environ.get("EMBEDDING_AZURE_OPENAI_ENDPOINT")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        if not endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT for Azure OpenAI embeddings.")
        api_key = (
            os.environ.get("EMBEDDING_AZURE_OPENAI_API_KEY")
            or os.environ.get("EMBEDDING_AZURE_API_KEY")
            or os.environ.get("AZURE_OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing AZURE_OPENAI_API_KEY for Azure OpenAI embeddings.")
        api_version = os.environ.get("EMBEDDING_AZURE_API_VERSION") or default_api_version
        client = AzureOpenAI(
            azure_endpoint=endpoint.rstrip("/"),
            api_version=api_version,
            api_key=api_key,
            max_retries=0,
        )
        return client, ""

    if provider == OPENAI_PROVIDER:
        base_url = os.environ.get("EMBEDDING_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("EMBEDDING_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY for OpenAI embeddings.")
        kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        return OpenAI(**kwargs), ""

    raise ValueError(f"Unsupported embedding provider: {provider}")


def embed_batch(client: Any, model: str, texts: List[str]) -> np.ndarray:
    """One Azure/OpenAI embedding call with built-in retry."""
    response = with_retry(
        client.embeddings.create,
        model=model,
        input=texts,
    )
    vectors = np.asarray([d.embedding for d in response.data], dtype=np.float32)
    return vectors


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def manifest_path(store_dir: str) -> str:
    return os.path.join(store_dir, "manifest.json")


def embeddings_path(store_dir: str) -> str:
    return os.path.join(store_dir, "embeddings.npy")


def load_manifest(store_dir: str) -> Optional[Dict[str, Any]]:
    p = manifest_path(store_dir)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(store_dir: str, payload: Dict[str, Any]) -> None:
    with open(manifest_path(store_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_or_load_embeddings(
    *,
    messages: List[Dict[str, Any]],
    store_dir: str,
    embedding_client: Any,
    embedding_model: str,
    embedding_provider: str,
    embedding_dims: int,
    batch_size: int,
    conversation_path: str,
) -> np.ndarray:
    """If the manifest matches conversation + config, load embeddings.npy.
    Otherwise embed every message content and persist."""
    os.makedirs(store_dir, exist_ok=True)
    fp = conversation_fingerprint(conversation_path)
    existing = load_manifest(store_dir)
    same_conv = existing and existing.get("conversation") == fp
    same_cfg = (
        existing
        and existing.get("ingest_config_version") == INGEST_CONFIG_VERSION
        and existing.get("embedding_provider") == embedding_provider
        and existing.get("embedding_model") == embedding_model
        and existing.get("embedding_dims") == embedding_dims
        and existing.get("total_messages") == len(messages)
    )
    if (
        same_conv
        and same_cfg
        and existing.get("status") == "ready"
        and os.path.exists(embeddings_path(store_dir))
    ):
        emb = np.load(embeddings_path(store_dir))
        if emb.shape == (len(messages), embedding_dims):
            print(f"[embed] reusing cached embeddings: {embeddings_path(store_dir)}")
            return emb
        print("[embed] cached embeddings shape mismatch; rebuilding")

    texts = [message_index_text(m) or " " for m in messages]
    out = np.zeros((len(texts), embedding_dims), dtype=np.float32)
    pbar = tqdm(range(0, len(texts), batch_size), desc="Embedding messages", unit="batch", dynamic_ncols=True)
    for start in pbar:
        end = min(start + batch_size, len(texts))
        chunk = texts[start:end]
        vecs = embed_batch(embedding_client, embedding_model, chunk)
        if vecs.shape[1] != embedding_dims:
            raise ValueError(
                f"Embedding model returned dim={vecs.shape[1]}, expected {embedding_dims}"
            )
        out[start:end] = vecs

    np.save(embeddings_path(store_dir), out)
    write_manifest(
        store_dir,
        {
            "ingest_config_version": INGEST_CONFIG_VERSION,
            "conversation": fp,
            "conversation_id": conversation_id_from_path(conversation_path),
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dims": embedding_dims,
            "total_messages": len(messages),
            "status": "ready",
        },
    )
    print(f"[embed] wrote {len(texts)} embeddings to {embeddings_path(store_dir)}")
    return out


def default_store_dir(conversation_path: str) -> str:
    cid = conversation_id_from_path(conversation_path)
    return os.path.join("stores", "text_embedding_3_large", cid)


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
    ap.add_argument("--embedding-provider", default=None)
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL_DEFAULT)
    ap.add_argument("--embedding-dims", type=int, default=EMBEDDING_DIMS_DEFAULT)
    ap.add_argument("--embedding-batch-size", type=int, default=EMBEDDING_BATCH_DEFAULT)
    ap.add_argument("--store-dir", default=None)
    ap.add_argument("--ingest-only", action="store_true",
                    help="Build/refresh the embedding cache and exit before QA.")
    args = ap.parse_args()

    load_env_file(args.env_file)
    llm_provider = normalize_llm_provider(args.llm_provider)
    base_endpoint = resolve_base_url(llm_provider)
    api_key = resolve_api_key(llm_provider)
    embedding_provider = normalize_embedding_provider(args.embedding_provider)

    messages = load_conversation_messages(args.conversation_json)
    print(f"[embed] loaded {len(messages)} messages")

    store_dir = args.store_dir or default_store_dir(args.conversation_json)
    embedding_client, _ = build_embedding_client(embedding_provider, API_VERSION_DEFAULT)
    doc_emb = build_or_load_embeddings(
        messages=messages,
        store_dir=store_dir,
        embedding_client=embedding_client,
        embedding_model=args.embedding_model,
        embedding_provider=embedding_provider,
        embedding_dims=args.embedding_dims,
        batch_size=args.embedding_batch_size,
        conversation_path=args.conversation_json,
    )
    doc_emb_norm = l2_normalize(doc_emb)

    if args.ingest_only:
        print("[embed] ingest-only: cache built, exiting before QA")
        return 0

    questions = load_questions(args.questions_jsonl)
    agent_system = read_text(args.agent_prompt)
    judge_system = read_text(args.judge_prompt)
    chat_client = create_chat_client(
        provider=llm_provider,
        azure_endpoint=base_endpoint,
        base_url=base_endpoint,
        api_version=API_VERSION_DEFAULT,
        api_key=api_key,
    )

    def retrieve(query: str, k: int) -> List[int]:
        if not query.strip():
            return []
        q_vec = embed_batch(embedding_client, args.embedding_model, [query])
        q_vec = l2_normalize(q_vec)[0]
        scores = doc_emb_norm @ q_vec
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            top_unsorted = np.argpartition(-scores, k)[:k]
            order = top_unsorted[np.argsort(-scores[top_unsorted])]
        return [int(i) for i in order[:k]]

    correct, total = run_qa(
        questions=questions,
        messages=messages,
        retrieve=retrieve,
        top_k=args.retrieve_top_k,
        client=chat_client,
        agent_model=args.agent_model,
        judge_model=args.judge_model,
        agent_system=agent_system,
        judge_system=judge_system,
        output_jsonl=args.output_jsonl,
    )

    accuracy = correct / total if total else 0.0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Embedding cache: {store_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
