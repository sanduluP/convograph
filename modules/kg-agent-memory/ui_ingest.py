#!/usr/bin/env python3
"""
ui_ingest.py — take ONE fresh transcript (not a pre-existing corpus) and
ingest it into the temporal KG, then return the facts extracted for it.

This is the "single transcript in, facts out" entrypoint module 2 did not
already have (baselines/graphiti/graphiti_retriever.py only ever windows a
full pre-existing corpus for eval). Run inside module 2's own venv:

    .venv/bin/python ui_ingest.py --text-file transcript.txt --group-id ui_20260831

LLM / EMBEDDER BACKEND
-----------------------
Same env vars as baselines/graphiti/graphiti_retriever.py's _build_client()
(GRAPHITI_LLM_BASE_URL / GRAPHITI_LLM_MODEL / GRAPHITI_EMBED_BASE_URL /
GRAPHITI_EMBED_MODEL), so pointing this at the cluster vLLM tunnel vs. local
ollama is a caller-side env change, not a code change — see README for both.

Applies baselines/graphiti/prompts_override.py's meeting-aware contradiction
prompt (without it, implicit decision revisions never get flagged as
superseding an earlier fact — see that file's docstring).

Neo4j: reads NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE from
.env (AuraDB or local Docker both work — AuraDB needs NEO4J_DATABASE passed
explicitly since graphiti-core's driver defaults to a database literally
named "neo4j", which an Aura instance does not use).

Output: one JSON object to stdout —
    {"group_id": ..., "episodes": N, "facts": [{"fact", "valid_at", "invalid_at"}, ...]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baselines", "graphiti"))

import prompts_override  # noqa: E402
from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # noqa: E402
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # noqa: E402
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient  # noqa: E402
from graphiti_core.driver.neo4j_driver import Neo4jDriver  # noqa: E402

WINDOW_LINES = int(os.getenv("UI_INGEST_WINDOW_LINES", "12"))  # lines per episode


def _load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _build_client() -> Graphiti:
    llm_config = LLMConfig(
        api_key=os.getenv("GRAPHITI_LLM_API_KEY", "ollama"),
        model=os.getenv("GRAPHITI_LLM_MODEL", "qwen2.5:3b-instruct"),
        small_model=os.getenv("GRAPHITI_LLM_MODEL", "qwen2.5:3b-instruct"),
        base_url=os.getenv("GRAPHITI_LLM_BASE_URL", "http://localhost:11434/v1"),
        temperature=float(os.getenv("GRAPHITI_TEMPERATURE", "0.2")),
        max_tokens=4096,
    )
    llm_client = OpenAIGenericClient(
        config=llm_config, max_tokens=int(os.getenv("GRAPHITI_MAX_TOKENS", "4096"))
    )
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key=os.getenv("GRAPHITI_EMBED_API_KEY", "ollama"),
        embedding_model=os.getenv("GRAPHITI_EMBED_MODEL", "bge-m3:latest"),
        embedding_dim=int(os.getenv("GRAPHITI_EMBED_DIM", "1024")),
        base_url=os.getenv("GRAPHITI_EMBED_BASE_URL", "http://localhost:11434/v1"),
    ))
    cross_encoder = OpenAIRerankerClient(config=llm_config)

    driver = Neo4jDriver(
        os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"],
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    return Graphiti(graph_driver=driver, llm_client=llm_client,
                     embedder=embedder, cross_encoder=cross_encoder)


def _windows(lines: list[str], size: int) -> list[list[str]]:
    return [lines[i:i + size] for i in range(0, len(lines), size)] or [[]]


async def ingest(text: str, group_id: str) -> dict:
    prompts_override.apply_overrides()
    graphiti = _build_client()
    await graphiti.build_indices_and_constraints()

    lines = [ln for ln in text.splitlines() if ln.strip()]
    windows = _windows(lines, WINDOW_LINES)

    prev_uuids: list[str] = []
    now = datetime.now(timezone.utc)
    for i, window in enumerate(windows):
        if not window:
            continue
        res = await graphiti.add_episode(
            name=f"{group_id}_w{i}",
            episode_body="\n".join(window),
            source=EpisodeType.message,
            source_description="UI-submitted transcript",
            reference_time=now,
            group_id=group_id,
            previous_episode_uuids=prev_uuids,
        )
        prev_uuids = [res.episode.uuid]

    records, _, _ = await graphiti.driver.execute_query(
        "MATCH ()-[r:RELATES_TO {group_id: $gid}]->() WHERE r.fact IS NOT NULL "
        "RETURN r.fact AS fact, r.valid_at AS valid_at, r.invalid_at AS invalid_at "
        "ORDER BY r.valid_at ASC",
        gid=group_id,
    )
    facts = [
        {
            "fact": r["fact"],
            "valid_at": str(r["valid_at"]) if r["valid_at"] else None,
            "invalid_at": str(r["invalid_at"]) if r["invalid_at"] else None,
        }
        for r in records
    ]

    await graphiti.close()
    return {"group_id": group_id, "episodes": len(windows), "facts": facts}


def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--env-file", default=os.path.join(os.path.dirname(__file__), ".env"))
    args = parser.parse_args()

    _load_env_file(args.env_file)
    text = open(args.text_file).read()

    result = asyncio.run(ingest(text, args.group_id))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
