#!/usr/bin/env python3
"""Kuzu compatibility check — can we drop the Neo4j SERVER entirely?

WHY THIS EXISTS
---------------
Our ingest currently needs a Neo4j server. Neo4j runs on Faris's laptop, the LLM
runs on the cluster, so the ingest process sits in the middle and the NETWORK
becomes load-bearing: a VPN blip kills a multi-hour run. That is unacceptable for
the ~7h full-channel run (nobody should have to keep a laptop awake all day).

Kuzu is an EMBEDDED graph database — the whole KG is just a FILE, no server, no
port, no tunnel. If Graphiti works on Kuzu, the entire run collapses into ONE
self-contained SLURM job on Pegasus: submit it, close the laptop, done.

WHAT WE MUST PROVE (three things, in order of risk)
--------------------------------------------------
  1. INGEST works        — add_episode against a Kuzu-backed Graphiti.
  2. BI-TEMPORAL works   — a later episode contradicting an earlier fact sets
                           `invalid_at`. This is our whole contribution; if Kuzu
                           loses it, Kuzu is useless to us.
  3. HYBRID SEARCH works — search() returns edges. The source shows Kuzu-specific
                           branches for full-text ("Kuzu only supports simple
                           queries"), so BM25+semantic should both run, but
                           "should" is not evidence.

We plant a deliberate deadline revision so check 2 has something to catch.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

from baselines.rag_common.eval_lib import load_env_file

# Three tiny episodes. Episode 2 REVISES episode 1's deadline — that is the
# contradiction we expect Graphiti to catch by stamping `invalid_at` on the old
# fact rather than deleting it.
EPISODES = [
    ("User_1: The AML model review deadline is June 30.\n"
     "User_2: Agreed, June 30 it is."),
    ("User_1: We need to move the AML model review deadline to July 15.\n"
     "User_2: Understood, July 15 is the new deadline."),
    ("User_3: Who owns the AML model review?\n"
     "User_1: Sarah owns it."),
]


def build_client(db_path: str) -> Graphiti:
    """Same LLM/embedder wiring as the real retriever, but Kuzu instead of Neo4j."""
    llm_config = LLMConfig(
        api_key=os.getenv("GRAPHITI_LLM_API_KEY", "ollama"),
        model=os.getenv("GRAPHITI_LLM_MODEL", "qwen2.5:32b"),
        small_model=os.getenv("GRAPHITI_LLM_MODEL", "qwen2.5:32b"),
        base_url=os.getenv("GRAPHITI_LLM_BASE_URL", "http://localhost:8000/v1"),
        temperature=float(os.getenv("GRAPHITI_TEMPERATURE", "0.2")),
    )
    llm_client = OpenAIGenericClient(config=llm_config, max_tokens=16384)
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key=os.getenv("GRAPHITI_EMBED_API_KEY", "ollama"),
        embedding_model=os.getenv("GRAPHITI_EMBED_MODEL", "bge-m3:latest"),
        embedding_dim=int(os.getenv("GRAPHITI_EMBED_DIM", "1024")),
        base_url=os.getenv("GRAPHITI_EMBED_BASE_URL",
                           "http://serv-3306.kl.dfki.de:8000/v1"),
    ))
    # THE ONE LINE THAT MATTERS: an embedded file-backed driver, no server.
    driver = KuzuDriver(db=db_path)
    return Graphiti(graph_driver=driver, llm_client=llm_client, embedder=embedder)


async def main() -> int:
    load_env_file(".env")
    db_path = os.path.join(tempfile.mkdtemp(prefix="kuzu_check_"), "graph.kuzu")
    print(f"🗂️  Kuzu db file: {db_path}")

    graphiti = build_client(db_path)
    ok = {"ingest": False, "bitemporal": False, "search": False}
    try:
        # --- 1) schema + ingest -------------------------------------------------
        await graphiti.build_indices_and_constraints()
        print("✅ 1/3 schema created (build_indices_and_constraints)")

        base = datetime(2026, 6, 1, tzinfo=timezone.utc)
        prev: list[str] = []
        for i, body in enumerate(EPISODES, start=1):
            res = await graphiti.add_episode(
                name=f"kuzu_check_ep{i}",
                episode_body=body,
                source=EpisodeType.message,
                source_description="kuzu compatibility check",
                reference_time=base + timedelta(days=7 * (i - 1)),
                group_id="kuzu_check",
                previous_episode_uuids=prev or None,
            )
            prev.append(res.episode.uuid)
            print(f"   • episode {i}/{len(EPISODES)} ingested")
        ok["ingest"] = True
        print("✅ 2/3 INGEST works on Kuzu")

        # --- 2) bi-temporal: did the deadline revision invalidate the old fact? --
        # Query the Kuzu store directly; RELATES_TO is reified as RelatesToNode_.
        recs, _, _ = await graphiti.driver.execute_query(
            "MATCH (e:RelatesToNode_) RETURN e.fact AS fact, e.valid_at AS valid_at, "
            "e.invalid_at AS invalid_at"
        )
        facts = [dict(r) for r in recs]
        invalidated = [f for f in facts if f.get("invalid_at")]
        print(f"\n📊 facts extracted: {len(facts)}   invalidated: {len(invalidated)}")
        for f in facts[:12]:
            mark = "🔴 SUPERSEDED" if f.get("invalid_at") else "🟢 valid"
            print(f"   {mark}  {str(f.get('fact'))[:88]}")
        ok["bitemporal"] = len(invalidated) > 0
        print("✅ 3/3 BI-TEMPORAL invalidation fired on Kuzu"
              if ok["bitemporal"] else
              "⚠️  no invalid_at set — bi-temporal NOT demonstrated on this sample")

        # --- 3) hybrid search (semantic + BM25 full-text + RRF) -----------------
        edges = await graphiti.search("What is the AML model review deadline?",
                                      num_results=5)
        print(f"\n🔎 hybrid search returned {len(edges)} edges")
        for e in edges[:5]:
            state = "SUPERSEDED" if getattr(e, "invalid_at", None) else "valid"
            print(f"   [{state}] {getattr(e, 'fact', '')[:88]}")
        ok["search"] = len(edges) > 0
    finally:
        await graphiti.close()

    print("\n" + "=" * 62)
    for k, v in ok.items():
        print(f"  {'✅' if v else '❌'}  {k}")
    print("=" * 62)
    # Ingest + search are hard requirements. Bi-temporal is what makes Kuzu USEFUL
    # to us specifically, so treat a miss as a failure worth investigating.
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
