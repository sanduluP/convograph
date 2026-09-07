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
import re
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

    # Fail with an instruction, not a bare KeyError. This module historically had
    # NO .env at all - every run got its Neo4j connection from the cluster job
    # scripts, which start their own Neo4j and export the URI - so "NEO4J_URI is
    # missing" is the FIRST thing a new local/UI user hits, and the raw traceback
    # says nothing about how to fix it.
    missing = [k for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
               if not os.getenv(k)]
    if missing:
        raise SystemExit(
            f"missing {', '.join(missing)}.\n\n"
            f"This module needs a git-ignored .env next to this file. For a local UI run:\n\n"
            f"    docker start neo4j-convograph-ui     # bolt 7690, browser http://localhost:7476\n\n"
            f"    # modules/kg-agent-memory/.env\n"
            f"    NEO4J_URI=bolt://localhost:7690\n"
            f"    NEO4J_USER=neo4j\n"
            f"    NEO4J_PASSWORD=<the container's password>\n"
            f"    NEO4J_DATABASE=neo4j\n\n"
            f"Do NOT point this at neo4j-gmb-full (port 7688): that container holds the\n"
            f"merged full-corpus graph and every score measured on it. UI runs are\n"
            f"throwaway and belong in their own store."
        )

    driver = Neo4jDriver(
        os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"],
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    return Graphiti(graph_driver=driver, llm_client=llm_client,
                     embedder=embedder, cross_encoder=cross_encoder)


def _windows(lines: list[str], size: int) -> list[list[str]]:
    return [lines[i:i + size] for i in range(0, len(lines), size)] or [[]]


async def _read_facts(graphiti, group_id: str, limit: int | None = None,
                     prefer_superseded: bool = False) -> list[dict]:
    """Return facts from a group, oldest first.

    Split out of ingest() so the SAME cypher serves both callers: the ingest
    path, and the query-only path that reads a graph somebody built earlier.
    One query, one shape, no chance of the two drifting apart.

    `limit` is pushed into the CYPHER, not applied afterwards in Python. That
    matters: gmb_finance_full holds 111,258 facts, and slicing client-side meant
    dragging all of them across the wire from Aura to take six — measured 19.7 s
    for a query that should be well under a second.

    `prefer_superseded` puts the facts the conversation later OVERTURNED first.
    Those are the ones worth drawing: a fact with invalid_at set is a decision
    that changed, which is the whole point of a temporal KG and the thing a flat
    summary cannot show. Without it a board of the first six facts is just the
    oldest six, which is rarely the interesting six.
    """
    # Sorting by whether invalid_at exists, THEN by time, keeps the result
    # deterministic - two runs on the same graph must produce the same board.
    order = ("ORDER BY (r.invalid_at IS NULL), r.valid_at ASC"
             if prefer_superseded else "ORDER BY r.valid_at ASC")
    cypher = (
        "MATCH ()-[r:RELATES_TO {group_id: $gid}]->() WHERE r.fact IS NOT NULL "
        "RETURN r.fact AS fact, r.valid_at AS valid_at, r.invalid_at AS invalid_at "
        f"{order}"
    )
    if limit is not None:
        cypher += " LIMIT $lim"
    records, _, _ = await graphiti.driver.execute_query(
        cypher, gid=group_id, **({"lim": limit} if limit is not None else {})
    )
    return [
        {
            "fact": r["fact"],
            "valid_at": str(r["valid_at"]) if r["valid_at"] else None,
            "invalid_at": str(r["invalid_at"]) if r["invalid_at"] else None,
        }
        for r in records
    ]


async def _read_window_run(graphiti, group_id: str, windows: int,
                           max_facts: int = 40) -> dict:
    """Return a CONTIGUOUS run of `windows` episodes: their raw text and facts.

    WHY A RUN OF EPISODES AND NOT "THE FIRST N FACTS"
    -------------------------------------------------
    A single fact carries almost no meaning alone. Read one in isolation -
    "waiting one week too long on a setup-question spike caused the fix to
    become a training scramble" - and even a human cannot tell what was decided.
    Handing that to an LLM and asking for a picture is hopeless.

    An EPISODE is the natural unit we have: one Graphiti episode is a 5-message
    window of the conversation. Consecutive episodes are consecutive in the
    conversation, so a run of them is a genuine slice of one discussion.

    ORDERING: by the window NUMBER parsed out of the episode name
    ("<group>_w<N>"), never by valid_at. Many episodes share a timestamp - the
    first attempt ordered by valid_at and picked w5418 next to w3349, which are
    two unrelated parts of the corpus that merely happen to fall on the same
    day.

    WHICH run: the one with the most SUPERSEDED facts - decisions the
    conversation later overturned, the one thing a flat summary cannot
    reconstruct. Ties break on the earliest window, so the choice is stable.

    FACT COUNT: capped. A Graphiti fact edge can be attached to MANY episodes,
    so a 2-window run pulled 152 facts rather than the ~26 those 10 messages
    produced. Superseded facts are kept first, then the rest chronologically.

    This is a workaround for the corpus, not a design: GroupMemBench is a
    scrubbed group chat with no meeting boundaries. When module 1 supplies one
    real meeting, the unit becomes the meeting and only this function changes.
    """
    records, _, _ = await graphiti.driver.execute_query(
        "MATCH (e:Episodic {group_id: $gid}) "
        "RETURN e.uuid AS uuid, e.name AS name, e.content AS content, "
        "       e.valid_at AS valid_at",
        gid=group_id,
    )
    def window_no(name: str) -> int:
        m = re.search(r"_w(\d+)$", name or "")
        return int(m.group(1)) if m else -1

    eps = sorted((dict(r) for r in records), key=lambda e: window_no(e["name"]))
    eps = [e for e in eps if window_no(e["name"]) >= 0]
    if not eps:
        return {"episodes": [], "facts": []}
    windows = max(1, min(windows, len(eps)))

    # Score every candidate run by supersessions, but do the counting IN THE
    # DATABASE: pulling all 111,258 facts to the client to build the mapping
    # took 44 s. This asks only for the (episode, superseded?) pairs.
    srecords, _, _ = await graphiti.driver.execute_query(
        "MATCH ()-[r:RELATES_TO {group_id: $gid}]->() "
        "WHERE r.fact IS NOT NULL AND r.invalid_at IS NOT NULL "
        "UNWIND r.episodes AS eu RETURN eu AS uuid, count(*) AS n",
        gid=group_id,
    )
    sup_by_uuid = {r["uuid"]: r["n"] for r in srecords}
    sup = [sup_by_uuid.get(e["uuid"], 0) for e in eps]

    best_start, best_score = 0, -1
    for i in range(0, len(eps) - windows + 1):
        score = sum(sup[i:i + windows])
        if score > best_score:
            best_start, best_score = i, score

    chosen = eps[best_start:best_start + windows]
    uuids = [e["uuid"] for e in chosen]

    # Fetch ONLY the facts belonging to the chosen episodes, ranked so the
    # overturned ones survive the cap.
    frecords, _, _ = await graphiti.driver.execute_query(
        "MATCH ()-[r:RELATES_TO {group_id: $gid}]->() "
        "WHERE r.fact IS NOT NULL AND any(u IN r.episodes WHERE u IN $uu) "
        "RETURN DISTINCT r.fact AS fact, r.valid_at AS valid_at, "
        "       r.invalid_at AS invalid_at "
        "ORDER BY (r.invalid_at IS NULL), r.valid_at ASC LIMIT $lim",
        gid=group_id, uu=uuids, lim=max_facts,
    )

    return {
        "episodes": [{"name": e["name"], "valid_at": str(e["valid_at"]),
                      "content": e["content"]} for e in chosen],
        "facts": [{"fact": r["fact"],
                   "valid_at": str(r["valid_at"]) if r["valid_at"] else None,
                   "invalid_at": str(r["invalid_at"]) if r["invalid_at"] else None}
                  for r in frecords],
        "window_start": window_no(chosen[0]["name"]),
        "superseded_in_run": best_score,
    }


async def query_only(group_id: str, limit: int | None = None,
                     windows: int | None = None,
                     max_facts: int = 40) -> dict:
    """Read facts from a graph that ALREADY EXISTS. Extracts nothing.

    Module 2's extraction is the slow, expensive stage: on a laptop CPU one
    short transcript took ~10 minutes, and a benchmark-quality graph needs the
    30B model on the cluster, not the 4B we serve for the UI. But module 3 does
    not need to re-derive any of that — it needs FACTS, and a graph full of them
    already exists.

    So this treats module 2's output as FIXED INPUT: point at a group_id that is
    already in Neo4j and go straight to captions and images. It is the shape
    module 3 was always meant to have (a cypher query against the temporal KG),
    and it takes the extraction cost out of the iteration loop entirely.

    Re-running extraction later is just not passing --query-only.
    """
    graphiti = _build_client()

    if windows:
        # Window mode: a contiguous slice of the conversation, with the raw
        # messages alongside the facts. See _read_window_run for why.
        run = await _read_window_run(graphiti, group_id, windows, max_facts)
        await graphiti.close()
        return {"group_id": group_id, "source": "existing-graph",
                "mode": "windows", "windows": windows,
                "window_start": run.get("window_start"),
                "superseded_in_run": run.get("superseded_in_run"),
                "episode_texts": run["episodes"],
                "episodes": len(run["episodes"]), "facts": run["facts"]}

    # prefer_superseded: a board should show what CHANGED, not merely what is
    # oldest. See _read_facts.
    facts = await _read_facts(graphiti, group_id, limit=limit,
                              prefer_superseded=True)
    await graphiti.close()
    return {"group_id": group_id, "episodes": 0, "facts": facts,
            "source": "existing-graph", "mode": "facts"}


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

    facts = await _read_facts(graphiti, group_id)
    await graphiti.close()
    return {"group_id": group_id, "episodes": len(windows), "facts": facts,
            "source": "fresh-ingest"}


def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file",
                        help="Transcript to ingest. Not needed with --query-only.")
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--query-only", action="store_true",
                        help="Do NOT extract. Read the facts of an EXISTING group "
                             "and return them. This is how module 3 is meant to be "
                             "driven: module 2's output is fixed input, so the slow "
                             "extraction stays out of the loop.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap how many facts come back (query-only).")
    parser.add_argument("--windows", type=int, default=None,
                        help="Return a CONTIGUOUS run of N episodes - their raw "
                             "messages AND their facts - instead of loose facts. "
                             "One episode is a 5-message window (~13 facts). The "
                             "run with the most superseded facts is chosen.")
    parser.add_argument("--max-facts", type=int, default=40,
                        help="Cap how many facts a WINDOW run returns (--limit "
                             "is the equivalent for plain fact mode, which "
                             "window mode ignores). Fact edges are reused across "
                             "episodes, so 2 windows can pull back 150+ facts "
                             "without this - superseded ones come first.")
    parser.add_argument("--env-file", default=os.path.join(os.path.dirname(__file__), ".env"))
    args = parser.parse_args()

    _load_env_file(args.env_file)

    if args.query_only:
        result = asyncio.run(query_only(args.group_id, args.limit, args.windows,
                                        args.max_facts))
    else:
        if not args.text_file:
            parser.error("--text-file is required unless --query-only is given")
        text = open(args.text_file).read()
        result = asyncio.run(ingest(text, args.group_id))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
