#!/usr/bin/env python3
"""Retrieve over the KG's FACT TEXT lexically — the extraction-vs-retrieval ablation.

THE QUESTION
------------
The 2026-08-05 answer audit (`analysis/mine_answers_in_kg.py`) found something
specific: the fact SENTENCES Graphiti extracts are often good — for several
questions the top fact is nearly the gold answer verbatim — but the ENTITIES they
are stored on are frequently sentence fragments (`User_5 -> user`,
`User_7 -> recovery states`, `-> completeness`, `-> drift`).

That matters because Graphiti's search walks entities and their embeddings. If the
facts are fine and the entity layer is the broken part, then indexing the exact
same facts by their TEXT should retrieve much better than Graphiti's own search
does — over an identical graph, with nothing re-extracted.

So this retriever is a deliberate ablation, not a proposed system:

    same graph  →  same 111,258 facts  →  BM25 over fact text  →  source messages

Reading the outcome:
  * clearly beats Graphiti's 28.1 %  → the facts are good and the ENTITY/embedding
    layer is what loses. Fix retrieval; the extraction prompt is not the problem.
  * roughly matches 28.1 %           → the facts themselves are the ceiling.
    Fix extraction; no retrieval work will help.
  * clearly below                    → Graphiti's semantic search is adding real
    value and the lexical view is the weaker one.

Each of those points at a different month of work, which is the whole reason to
spend one GPU-hour separating them.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, List

from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase word split. No stopword removal — BM25's IDF already discounts
    common words, and keeping them makes the ranking reproducible without a list."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def build_kgfacts_retriever(
    messages: List[dict],
    *,
    window: int,
    group_id: str,
    bolt_uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> Callable[[str, int], List[int]]:
    """Return `retrieve(query, k) -> message indices`, ranked by fact-text BM25.

    The mapping fact → message indices reuses the SAME windowing the ingest used
    (`_build_windows`), so a fact retrieved here resolves to exactly the messages
    Graphiti would have resolved it to. Anything else would make the comparison
    dishonest.
    """
    # Imported here (not at module scope) so this module stays importable without
    # graphiti-core installed — the ablation itself needs only neo4j + rank_bm25.
    from baselines.graphiti.graphiti_retriever import _build_windows

    bolt_uri = bolt_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.environ.get("NEO4J_USER", "neo4j")
    password = password or os.environ.get("NEO4J_PASSWORD", "graphiti123")

    # window number (1-based, global over the whole corpus) → message indices
    windows = _build_windows(messages, window)
    window_indices: Dict[int, List[int]] = {
        n: idxs for n, idxs in enumerate(windows, start=1)
    }
    print(f"[kgfacts] {len(window_indices)} windows over {len(messages)} messages",
          flush=True)

    driver = GraphDatabase.driver(bolt_uri, auth=(user, password))

    # episode uuid → window number, parsed back out of the "{group_id}_w{N}" name
    # the ingest assigned. Restricting to this group_id keeps a store holding
    # several namespaces from leaking across them.
    ep_window: Dict[str, int] = {}
    with driver.session() as s:
        for rec in s.run(
            "MATCH (e:Episodic) WHERE e.group_id = $gid "
            "RETURN e.uuid AS uuid, e.name AS name", gid=group_id
        ):
            m = re.search(r"_w(\d+)$", rec["name"] or "")
            if m:
                ep_window[rec["uuid"]] = int(m.group(1))
    print(f"[kgfacts] mapped {len(ep_window)} episodes", flush=True)

    # Every fact, with the episodes it was extracted from.
    fact_texts: List[str] = []
    fact_indices: List[List[int]] = []
    with driver.session() as s:
        for rec in s.run(
            "MATCH (:Entity)-[r:RELATES_TO]->(:Entity) "
            "RETURN r.fact AS fact, r.episodes AS episodes"
        ):
            fact = rec["fact"] or ""
            if not fact:
                continue
            idxs: List[int] = []
            for uuid in (rec["episodes"] or []):
                idxs.extend(window_indices.get(ep_window.get(uuid, -1), []))
            if not idxs:
                # A fact we cannot ground in a message is useless as a passage —
                # dropping it keeps the ranking honest rather than padding it.
                continue
            fact_texts.append(fact)
            fact_indices.append(idxs)

    driver.close()
    print(f"[kgfacts] indexed {len(fact_texts)} groundable facts", flush=True)

    bm25 = BM25Okapi([_tokenize(t) for t in fact_texts])

    def retrieve(query: str, k: int) -> List[int]:
        scores = bm25.get_scores(_tokenize(query))
        # Walk facts best-first and collect their source messages until we have k.
        # Dedup preserves rank order: a message backed by a better fact wins.
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        seen: Dict[int, None] = {}
        for i in order:
            if scores[i] <= 0:
                break
            for idx in fact_indices[i]:
                if idx not in seen:
                    seen[idx] = None
                    if len(seen) >= k:
                        return list(seen)
        return list(seen)[:k]

    return retrieve
