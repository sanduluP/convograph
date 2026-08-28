#!/usr/bin/env python3
"""The real Graphiti temporal-KG retriever for GroupMemBench (Step 2b).

BIG PICTURE
-----------
GroupMemBench asks every memory system for ONE function:

    retrieve(query, k) -> List[int]      # indices into the channel's `messages`

This module builds that function backed by a Graphiti temporal knowledge graph
instead of BM25. The two milestones (see the 2026-07-30 Obsidian note) live here:

  MILESTONE 1 — BUILD THE KG (`_ingest`)
    Walk the messages in chronological order, grouped into WINDOWS of `window`
    consecutive messages (never crossing a channel boundary). Each window becomes
    ONE Graphiti *episode*. Graphiti LLM-extracts entities + facts and, when a
    later window contradicts an earlier fact, sets `invalid_at` on the old edge
    (bi-temporal) — that timeline is the whole point. Windowing (≈10 msgs/episode
    vs 1) cuts LLM rounds ~10x and gives the extractor real conversational context.

  MILESTONE 2 — RETRIEVE (`retrieve`)
    We do NOT hand-roll Cypher or a 2-hop walk. `graphiti.search(query)` already
    IS a hybrid retriever (semantic over fact embeddings + BM25 + graph rerank).
    On top of it we add the ONE thing BM25 cannot do — a TEMPORAL preference:
    currently-valid facts (`invalid_at IS NULL`) are ranked above superseded ones,
    so "what is the CURRENT X?" questions get the latest decision while historical
    facts stay retrievable for `temporal`/`multi_hop` questions.

    ⚠️ WHAT THE DEFAULT SEARCH ACTUALLY DOES (verified 2026-08-13, read the source)
    The sentence above oversells it. `graphiti.search()` uses the recipe
    EDGE_HYBRID_SEARCH_RRF = [BM25 over `e.fact`, cosine over `e.fact_embedding`]
    fused by RRF. There is NO "graph rerank" and NO traversal: both channels read
    the SAME fact SENTENCE, node names are never matched, and the relation name
    (`REQUIRES_MAPPING` etc.) is never read. In other words our KG retrieval is
    BM25-over-extracted-sentences plus an embedding — which is exactly why it
    scores level with plain BM25 over the raw messages (112 vs 115 of 214).

    GRAPHITI_BFS=1 turns on the graph walk that was missing (see `_search_config`).

DECOUPLING INGEST-SIZE FROM RETURN-SIZE (the key trick)
-------------------------------------------------------
A windowed fact maps to a *window* of ~10 messages, not one. We keep an
`episode_uuid -> [global message indices]` map built during ingestion. At query
time a hit fact -> its episode(s) -> the window's messages -> a cheap lexical
re-rank of those messages against the fact/query -> the single best message index.
So we ingest coarsely (cheap) but RETURN precise message indices. This is safe
because GroupMemBench judges the agent's ANSWER, not exact index match.
"""

from __future__ import annotations

import asyncio
import atexit
import html
import os
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# Word-level tokenizer reused for the within-window re-rank (identical rule to the
# BM25 baseline so the lexical scoring is consistent across the two retrievers).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> List[str]:
    """Lowercase word-level split (numbers/dates/user-IDs kept intact)."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _msg_text(message: dict) -> str:
    """The message content only (HTML-unescaped), same unit BM25 indexes."""
    content = message.get("content", "")
    return html.unescape(content).strip() if isinstance(content, str) else ""


def _parse_time(ts: str) -> datetime:
    """ISO8601 timestamp -> tz-aware UTC datetime (Graphiti's reference_time).
    Falls back to 'now' so a malformed/missing timestamp never crashes ingest."""
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# WINDOWING — group the flattened message list into per-channel episodes.
# ---------------------------------------------------------------------------
def _build_windows(messages: List[dict], window: int) -> List[List[int]]:
    """Return a list of windows, each a list of GLOBAL message indices.

    Windows are `window` consecutive messages that share the same `_channel`
    (a window never spans two channels — that would fuse unrelated projects).
    The indices are positions in the flattened `messages` list, i.e. exactly the
    integers `retrieve()` must return.
    """
    windows: List[List[int]] = []
    current: List[int] = []
    current_channel: Optional[str] = None
    for idx, m in enumerate(messages):
        ch = m.get("_channel", "")
        # Start a fresh window when the channel changes or the window is full.
        if ch != current_channel or len(current) >= window:
            if current:
                windows.append(current)
            current = []
            current_channel = ch
        current.append(idx)
    if current:
        windows.append(current)
    return windows


def _episode_body(messages: List[dict], idxs: List[int]) -> str:
    """Render a window of messages as the text Graphiti will extract from.

    TWO FRAMINGS, SELECTED BY `GRAPHITI_SPEAKER_FRAMING`
    ----------------------------------------------------
    `speaker` (default, the original) — speaker IS the sentence subject:
        User_1: we should freeze v1 evidence fields now
    `saidby` (the experiment) — speaker demoted to a bracketed attribution:
        [said by User_1] we should freeze v1 evidence fields now

    The value names describe the literal output format, so a graph named after one
    of them can be reconstructed from its name alone.

    WHY THIS ONE FUNCTION IS THE CAUSE OF THE STAR GRAPH
    ---------------------------------------------------
    Measured on the merged full-Finance KG: **94.1 % of all 111,258 facts originate
    at a person**, only 0.4 % link two domain concepts, and the twelve
    highest-degree nodes in the entire graph ARE the twelve users (`User_13` alone
    has degree 17,814). That shape is not a Graphiti defect — it is the direct,
    predictable consequence of the `speaker` framing above, via this chain:

      1. `extract_message` sees `User_1` at the head of a line. It is a properly
         named entity, so extracting it is CORRECT by that prompt's own rules.
      2. `extract_edges` is then told: "Extract all factual relationships between
         the given ENTITIES", with the rule that every fact must join two DISTINCT
         entities, plus: "When a sentence describes a specific detail about a
         single entity, do NOT drop it — look for a second entity to anchor it."
      3. Every sentence in the window was uttered BY `User_1`, and `User_1` is
         right there in the ENTITIES list. So it becomes the anchor for everything.

    The extractor is obeying its instructions exactly. We are the ones who made the
    speaker the grammatical subject of every sentence in the corpus.

    WHAT THE `saidby` FRAMING CHANGES
    ---------------------------------
    The speaker is demoted to a bracketed attribution rather than the subject of
    the line. Attribution is still present — so "who said what" questions remain
    answerable and `user_implicit` should not collapse — but the sentence's natural
    subject becomes the thing being discussed, which is what we want anchoring the
    facts.

    NOTE: this changes the FACT SENTENCES themselves, not just the graph topology.
    Since retrieval is BM25 + cosine over fact text, this affects the ordinary
    retrieval path too, not only the BFS walk.
    """
    framing = os.getenv("GRAPHITI_SPEAKER_FRAMING", "speaker")
    lines = []
    for i in idxs:
        m = messages[i]
        author = m.get("author") or "?"
        text = _msg_text(m)
        if framing == "saidby":
            # Attribution kept, but moved OUT of subject position so the extractor
            # anchors facts on what is being discussed instead of on who spoke.
            lines.append(f"[said by {author}] {text}")
        else:
            lines.append(f"{author}: {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OPTIONAL GRAPH TRAVERSAL (the thing the default recipe leaves switched off)
# ---------------------------------------------------------------------------
def _build_bfs_config():
    """Return a SearchConfig that ADDS breadth-first traversal, or None if off.

    WHY THIS EXISTS
    ---------------
    Graphiti's default recipe, EDGE_HYBRID_SEARCH_RRF, is
        [EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity]
    i.e. two lexical/semantic lookups over the fact SENTENCE and nothing else. The
    graph is never walked, so the KG behaves as a flat bag of sentences. That is
    consistent with every null result we have measured: entity stitching, group_id
    namespacing and channel scoping all changed the graph's STRUCTURE, and none of
    them moved the score, because structure is not read at query time.

    HOW THE WALK IS SEEDED (no origin-selection component needed)
    ------------------------------------------------------------
    In graphiti_core/search/search.py, when `EdgeSearchMethod.bfs` is present and
    no explicit origins are passed, Graphiti uses the SOURCE NODES of the edges
    that the first-pass BM25+cosine search already returned:

        source_node_uuids = [edge.source_node_uuid
                             for result in search_results for edge in result]

    So BFS is layered ON TOP of the existing search rather than replacing it.

    ⚠️ WHY THE DEPTH DEFAULT IS DANGEROUS **FOR OUR GRAPH SPECIFICALLY**
    --------------------------------------------------------------------
    The walk is one Cypher variable-length match, not a Python recursion:

        MATCH path = (origin {uuid: $u})-[:RELATES_TO|MENTIONS*1..DEPTH]->(:Entity)
        ... LIMIT $limit

    Graphiti's default is MAX_SEARCH_DEPTH = 3. That is fine on a normal graph and
    reckless on ours: 94 % of our facts originate at a PERSON, so the seed nodes
    collapse onto the 12 `User_N` hubs — and `User_13` alone has degree 17,814.
    Neo4j expands every path BEFORE `LIMIT` is applied, so depth 3 from a hub of
    that size is a combinatorial explosion that can hang the query or OOM the
    database. We therefore default to depth 1 and step up deliberately.

    KNOBS
    -----
      GRAPHITI_BFS=1        enable traversal (default 0 = unchanged behaviour)
      GRAPHITI_BFS_DEPTH=N  hops, default 1. Do NOT jump to 3 on this graph.
    """
    if os.getenv("GRAPHITI_BFS", "0") != "1":
        return None

    # Imported lazily so the default (no-BFS) path never depends on these symbols.
    from graphiti_core.search.search_config import (
        EdgeReranker,
        EdgeSearchConfig,
        EdgeSearchMethod,
        SearchConfig,
    )

    depth = int(os.getenv("GRAPHITI_BFS_DEPTH", "1"))
    print(f"[graphiti] 🔎 BFS traversal ENABLED (bfs_max_depth={depth}) — "
          f"search = bm25 + cosine + graph walk, fused by RRF", flush=True)

    return SearchConfig(
        edge_config=EdgeSearchConfig(
            # The two default channels are KEPT — BFS is additive, so any change in
            # the score is attributable to the walk alone.
            search_methods=[
                EdgeSearchMethod.bm25,
                EdgeSearchMethod.cosine_similarity,
                EdgeSearchMethod.bfs,
            ],
            reranker=EdgeReranker.rrf,
            bfs_max_depth=depth,
        )
    )


# ---------------------------------------------------------------------------
# GRAPHITI CLIENT — chat/extraction LLM + embedder + reranker, all from .env.
# ---------------------------------------------------------------------------
def _build_client() -> Graphiti:
    """Wire Graphiti to Neo4j + the serv-3306 (or cluster vLLM) OpenAI-compatible
    endpoints. Mirrors graphiti-eval/ingest_meetings.build_graphiti."""
    llm_config = LLMConfig(
        api_key=os.getenv("GRAPHITI_LLM_API_KEY", "ollama"),
        model=os.getenv("GRAPHITI_LLM_MODEL", "qwen2.5:32b"),
        small_model=os.getenv("GRAPHITI_LLM_MODEL", "qwen2.5:32b"),
        base_url=os.getenv("GRAPHITI_LLM_BASE_URL", "http://serv-3306.kl.dfki.de:8000/v1"),
        # NOT 0.0 on purpose. Greedy decoding occasionally falls into a REPETITION
        # LOOP on dense windows: the extractor emits thousands of near-duplicate
        # JSON lines, blows past max_tokens, and the truncated string fails to
        # parse. At temperature 0 every retry regenerates the byte-identical bad
        # response (we observed three retries failing at the exact same character
        # offset), so Graphiti's retry logic cannot recover and the whole ingest
        # dies. A small temperature makes retries genuinely different, which is
        # what lets them succeed. Extraction is already stochastic run-to-run, so
        # this costs us no reproducibility we actually had.
        temperature=float(os.getenv("GRAPHITI_TEMPERATURE", "0.2")),
        max_tokens=4096,  # superseded by the client arg below; kept for clarity
    )
    # NOTE: OpenAIGenericClient takes its OWN `max_tokens` constructor arg which
    # OVERRIDES LLMConfig.max_tokens (its default is 16384). Dense GroupMemBench
    # windows can make the extractor emit >12k tokens of JSON for a single
    # episode; hitting the ceiling truncates the response mid-string and
    # `json.loads` raises "Unterminated string", which killed the first 1k-message
    # ingest outright. Raise the ceiling, but stay well under max-model-len so the
    # PROMPT still fits alongside the completion.
    llm_client = OpenAIGenericClient(
        config=llm_config,
        # Budget: prompt + completion must fit max-model-len. The output ceiling is
        # NOT the only consumer — Graphiti's node-dedup prompt embeds the existing
        # entity candidates, so the PROMPT itself grows with the graph. On the
        # 2026-08-02 full-channel run it reached 16,385 input tokens, which against
        # a 16,384 output reservation overflowed the 32k context and 400'd 96
        # windows. The cluster job now serves 65,536 max-model-len and sets this to
        # 8192, leaving ~57k of prompt headroom.
        max_tokens=int(os.getenv("GRAPHITI_MAX_TOKENS", "8192")),
    )
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key=os.getenv("GRAPHITI_EMBED_API_KEY", "ollama"),
        embedding_model=os.getenv("GRAPHITI_EMBED_MODEL", "bge-m3:latest"),
        embedding_dim=int(os.getenv("GRAPHITI_EMBED_DIM", "1024")),
        base_url=os.getenv("GRAPHITI_EMBED_BASE_URL", "http://serv-3306.kl.dfki.de:8000/v1"),
    ))
    # Reranker points at the same LLM (default would call real OpenAI and fail).
    cross_encoder = OpenAIRerankerClient(config=llm_config)
    return Graphiti(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "graphiti123"),
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )


async def _wipe_group(graphiti: Graphiti, group_id: str) -> None:
    """Delete just this group's KG so every run starts clean (idempotent re-runs)."""
    await graphiti.driver.execute_query(
        "MATCH (n {group_id: $g}) DETACH DELETE n", g=group_id
    )


async def _existing_episodes(graphiti: Graphiti, group_id: str) -> Dict[str, str]:
    """Map episode NAME -> uuid for episodes already stored in this group.

    This is what makes an ingest resumable. Episode names are deterministic
    (`{group_id}_w{N}`, N counted over the FULL corpus), so a name that is already
    in Neo4j means that window's LLM extraction has already been paid for and must
    not be paid for again — a 13 h job that dies at hour 12 should cost us hour 12,
    not hours 1-12.
    """
    records, _, _ = await graphiti.driver.execute_query(
        "MATCH (e:Episodic {group_id: $g}) RETURN e.name AS name, e.uuid AS uuid",
        g=group_id,
    )
    return {r["name"]: r["uuid"] for r in records if r.get("name")}


async def _ingest(
    graphiti: Graphiti,
    messages: List[dict],
    windows: List[Tuple[int, List[int]]],
    group_id: str,
    resume: bool = False,
    retrieve_only: bool = False,
) -> Dict[str, List[int]]:
    """Ingest every window as one episode; return episode_uuid -> [msg indices].

    `windows` is a list of (GLOBAL window number, message indices). The number is
    global — i.e. counted over the whole corpus, not over this job's slice — so
    that N parallel jobs each handling a different range produce episode names
    that never collide and that line up with the same corpus at merge time.
    """
    await graphiti.build_indices_and_constraints()
    ep_to_indices: Dict[str, List[int]] = {}
    prev_uuids: List[str] = []  # chain episodes so coreference survives windows
    total = len(windows)
    failed: List[int] = []  # windows we had to skip (see the except block)
    consecutive = 0          # current UNBROKEN run of failures (outage detector)
    done = 0                 # episodes actually ingested by THIS job (for progress)
    max_consecutive_failures = int(os.getenv("GRAPHITI_MAX_CONSECUTIVE_FAILURES", "5"))

    # On a resume, recover what is already in the graph. These windows are then
    # replayed from Neo4j for free: we still need their uuid -> indices entry so
    # retrieval can map a fact back to a message, but we skip the LLM entirely.
    already: Dict[str, str] = {}
    if resume:
        already = await _existing_episodes(graphiti, group_id)
        print(f"[graphiti] resume: {len(already)} episodes already in group "
              f"'{group_id}' — those windows will be skipped", flush=True)

    # retrieve_only turns this into a pure MAPPING pass over an existing graph:
    # rebuild episode_uuid -> [message indices] from what is already stored and
    # ingest nothing. That is what a QA run against an already-built graph (e.g.
    # the merged full-corpus store) needs — it must NOT pay to fill the windows
    # that ingest skipped, because that would turn a 30-minute scoring run into a
    # multi-hour one. Gap-filling is a separate, deliberate pass.
    if retrieve_only:
        missing = 0
        for w_no, idxs in windows:
            uuid = already.get(f"{group_id}_w{w_no}")
            if uuid is None:
                missing += 1
                continue
            ep_to_indices[uuid] = idxs
        print(f"[graphiti] retrieve-only: mapped {len(ep_to_indices)} episodes, "
              f"{missing} windows absent from the graph "
              f"({100*missing/max(len(windows),1):.2f}% of the corpus unretrievable)",
              flush=True)
        return ep_to_indices

    for w_no, idxs in windows:
        name = f"{group_id}_w{w_no}"
        if name in already:
            # Already ingested by an earlier attempt: reuse its uuid, no LLM call.
            ep_to_indices[already[name]] = idxs
            prev_uuids.append(already[name])
            continue
        body = _episode_body(messages, idxs)
        # reference_time = the window's FIRST message timestamp (its position in
        # the decision timeline) — this is what Graphiti stamps facts with.
        ref = _parse_time(messages[idxs[0]].get("timestamp", ""))
        try:
            res = await graphiti.add_episode(
                name=name,
                episode_body=body,
                source=EpisodeType.message,
                source_description="GroupMemBench group-chat window",
                reference_time=ref,
                group_id=group_id,
                previous_episode_uuids=prev_uuids[-4:] or None,
            )
        except Exception as exc:  # noqa: BLE001 — one bad window must not be fatal
            # A single window whose extraction never parses (repetition loop ->
            # truncated JSON) should not kill the ENTIRE ingest. Skipping it costs
            # that window's facts; aborting costs every window.
            #
            # BUT skipping must NEVER paper over a systemic outage. On 2026-08-01
            # the DFKI VPN dropped mid-run: windows 37-200 all failed, and a naive
            # skip-everything loop cheerfully reported "KG built" over a graph with
            # 35/200 episodes. A silently degraded result is worse than a crash,
            # because it looks like data. So: tolerate ISOLATED failures, abort on a
            # RUN of them (that is an endpoint/network problem, not bad content).
            failed.append(w_no)
            consecutive += 1
            print(f"[graphiti] ⚠️  window {w_no}/{total} FAILED, skipping "
                  f"({type(exc).__name__}: {str(exc)[:120]})", flush=True)
            if consecutive >= max_consecutive_failures:
                raise RuntimeError(
                    f"ABORTING ingest: {consecutive} consecutive windows failed "
                    f"(at window {w_no}/{total}). This is an infrastructure failure, "
                    f"not bad content — check the LLM endpoint, the SSH tunnel and "
                    f"the VPN. {len(ep_to_indices)} episodes were ingested before "
                    f"the failure; the graph is INCOMPLETE and must not be used."
                ) from exc
            continue
        # A success resets the run counter — only an unbroken streak means outage.
        consecutive = 0
        ep_to_indices[res.episode.uuid] = idxs
        prev_uuids.append(res.episode.uuid)
        # Progress is counted over THIS job's slice (done/total), while the window
        # id printed alongside it is the GLOBAL one — so a log line identifies the
        # window unambiguously across all parallel jobs.
        done += 1
        if done % 10 == 0 or done == total:
            print(f"[graphiti] ingested episode {done}/{total} (window {w_no})",
                  flush=True)
    if failed:
        print(f"[graphiti] ⚠️  {len(failed)}/{total} windows skipped after retries: "
              f"{failed}", flush=True)
    return ep_to_indices


# ---------------------------------------------------------------------------
# PUBLIC ENTRY — build the retrieve() closure the harness calls.
# ---------------------------------------------------------------------------
def build_graphiti_retriever(
    messages: List[dict],
    *,
    window: int = 10,
    group_id: str = "gmb",
    max_ingest_messages: int = 0,
    prefer_valid: bool = True,
    window_range: Tuple[int, int] | None = None,
    resume: bool = False,
    retrieve_only: bool = False,
) -> Callable[[str, int], List[int]]:
    """Ingest `messages` into a Graphiti temporal KG and return `retrieve(query, k)`.

    Args:
      window:              messages per episode (the swept knob; 0/1 = one per msg).
      group_id:            Neo4j namespace for this KG.
      max_ingest_messages: cap ingested messages for fast SMOKE tests (0 = all).
      prefer_valid:        rank currently-valid facts (invalid_at IS NULL) above
                           superseded ones — the temporal edge over BM25.
      window_range:        (start, end) 1-based INCLUSIVE-EXCLUSIVE slice of the
                           global window list to ingest. This is how one 30,000-
                           message corpus is split across N parallel SLURM jobs:
                           every job sees the FULL corpus (so message indices and
                           window numbers are global and identical everywhere) but
                           only pays the LLM cost for its own range. None = all.
      resume:              do NOT wipe the group; skip windows already present.
                           Required for parallel jobs (a wipe would delete a
                           sibling job's work) and for restarting after a walltime
                           kill. See _existing_episodes.
      retrieve_only:       ingest NOTHING — just map the existing graph's episodes
                           back to message indices and return retrieve(). This is
                           the mode for SCORING an already-built graph (e.g. the
                           merged full-corpus store): no LLM cost, no waiting for
                           windows that ingest skipped. Implies resume.
    """
    # retrieve_only is meaningless without resume — and forgetting resume here
    # would WIPE the graph we are trying to score. Fail loudly rather than
    # silently destroy a 40-hour ingest.
    if retrieve_only and not resume:
        resume = True
        print("[graphiti] retrieve-only implies --resume (refusing to wipe)",
              flush=True)
    # One event loop kept alive for the retriever's lifetime: ingest once here,
    # then each retrieve() call reuses it to run the async Graphiti search.
    loop = asyncio.new_event_loop()
    graphiti = _build_client()

    # For a smoke test, ingest only the first N messages (indices stay aligned
    # with the corpus `run_qa` shows the agent, since we slice the SAME list).
    ingest_msgs = messages
    if max_ingest_messages and max_ingest_messages > 0:
        ingest_msgs = messages[:max_ingest_messages]
        print(f"[graphiti] --max-ingest-messages={max_ingest_messages} "
              f"→ ingesting {len(ingest_msgs)} of {len(messages)} messages", flush=True)

    w = max(1, window)  # window<=1 means one message per episode
    # Number every window over the WHOLE corpus first, then slice. Numbering
    # before slicing is the whole trick: job 3 of 6 calls its windows 977..1464
    # exactly as a single serial run would, so the six partial graphs are
    # mergeable and no episode name is ever reused for different messages.
    all_windows: List[Tuple[int, List[int]]] = list(
        enumerate(_build_windows(ingest_msgs, w), start=1)
    )
    windows = all_windows
    if window_range is not None:
        lo, hi = window_range
        windows = [(n, idxs) for (n, idxs) in all_windows if lo <= n < hi]
        print(f"[graphiti] --window-range {lo}:{hi} → this job ingests "
              f"{len(windows)} of {len(all_windows)} global windows", flush=True)

    print(f"[graphiti] {len(ingest_msgs)} messages → {len(all_windows)} windows "
          f"(window={w}) → ingesting into group '{group_id}'", flush=True)

    # A wipe is destructive to SIBLING jobs writing the same group, and it throws
    # away exactly the work a resume exists to keep. Only wipe on a plain,
    # single-job, from-scratch run.
    if not resume:
        loop.run_until_complete(_wipe_group(graphiti, group_id))
    ep_to_indices = loop.run_until_complete(
        _ingest(graphiti, messages, windows, group_id, resume=resume,
                retrieve_only=retrieve_only)
    )
    print(f"[graphiti] KG built: {len(ep_to_indices)} episodes in group '{group_id}'",
          flush=True)

    # Close the Graphiti driver cleanly when the process exits.
    atexit.register(lambda: loop.run_until_complete(graphiti.close()))

    def _best_index_in_window(fact: str, query: str, idxs: List[int]) -> int:
        """Within a hit window, pick the single message most lexically similar to
        the fact (tie-broken by the query) — the ingest→return decoupling step."""
        fact_toks = set(_tokens(fact))
        query_toks = set(_tokens(query))
        best_i, best_score = idxs[0], -1.0
        for i in idxs:
            m_toks = set(_tokens(_msg_text(messages[i])))
            if not m_toks:
                continue
            # Primary: overlap with the fact; secondary: overlap with the query.
            score = len(m_toks & fact_toks) + 0.5 * len(m_toks & query_toks)
            if score > best_score:
                best_i, best_score = i, score
        return best_i

    # ── OPTIONAL: turn on the graph traversal the default recipe leaves off ────
    # Built once, outside retrieve(), because it never varies per query.
    bfs_config = _build_bfs_config()

    def retrieve(query: str, k: int) -> List[int]:
        # Ask Graphiti for more facts than we need — after temporal ranking and
        # collapsing multiple facts onto the same message we still want k distinct.
        num = max(k * 4, 20)
        if bfs_config is None:
            # Default path: EDGE_HYBRID_SEARCH_RRF (BM25 + cosine, no traversal).
            edges = loop.run_until_complete(graphiti.search(query, num_results=num))
        else:
            # BFS path: same two channels PLUS a graph walk seeded from their hits.
            # search_() is the lower-level entrypoint that accepts a SearchConfig;
            # it returns a SearchResults object rather than a bare edge list.
            bfs_config.limit = num
            results = loop.run_until_complete(
                graphiti.search_(query, config=bfs_config)
            )
            edges = results.edges

        # Temporal preference: currently-valid facts first (invalid_at is None),
        # superseded facts after — Graphiti already ranked within each bucket by
        # relevance, so a stable sort on validity preserves that order.
        if prefer_valid:
            edges = sorted(edges, key=lambda e: getattr(e, "invalid_at", None) is not None)

        # Map each fact -> its window(s) -> the single best message index, dedup,
        # and stop once we have k. This returns PRECISE indices from COARSE episodes.
        picked: List[int] = []
        seen = set()
        for e in edges:
            fact = getattr(e, "fact", "") or ""
            for ep_uuid in getattr(e, "episodes", []) or []:
                idxs = ep_to_indices.get(ep_uuid)
                if not idxs:
                    continue
                best = _best_index_in_window(fact, query, idxs)
                if best not in seen:
                    seen.add(best)
                    picked.append(best)
                    if len(picked) >= k:
                        return picked
        return picked

    return retrieve
