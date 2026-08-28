#!/usr/bin/env python3
"""Merge the N per-shard Graphiti graphs into ONE Finance knowledge graph.

WHY THIS EXISTS
---------------
Ingesting all 30,000 Finance messages costs ~38 h of GPU, but the cluster caps a
reservation at 1 day. So `scripts/submit_shards.sh` splits the work across 8
parallel jobs, each with its OWN Neo4j instance (own data dir, own ports) so the
jobs can never fight over a database lock. That leaves 8 PARTIAL graphs on
/fscratch. This module glues them back together.

WHY GLUING IS SAFE
------------------
Every job loads the SAME full corpus and numbers its windows over the WHOLE
corpus BEFORE slicing (see --window-range in graphiti_retriever.py). So:

  • episode names are GLOBAL and unique  — `gmb_finance_full_w1503` means the same
    five messages no matter which shard produced it;
  • every node and edge Graphiti writes carries a UUID, so nothing collides when
    two shards' dumps land in the same database.

THE ONE REAL PROBLEM: CROSS-SHARD ENTITY DUPLICATES
---------------------------------------------------
Graphiti deduplicates entities at ingest time (`resolve_extracted_nodes`), but it
can only deduplicate against what is ALREADY IN ITS OWN GRAPH. Shard s3 never saw
shard s7's nodes, so "Basel III" exists up to 8 times — once per shard, each with
a different UUID. A naive union therefore produces a graph that is numerically
fine but visually broken.

That distinction matters, so it is worth stating plainly:
  • For the SCORE it barely matters. Retrieval hits FACT edges (RELATES_TO), and
    duplicate entities do not split facts — every fact stays retrievable and
    still maps back to its source episode.
  • For the VISUALIZATION it matters a lot, and the visualization is the actual
    deliverable — a graph with six separate "Basel III" bubbles reads as broken.

So we stitch duplicates by EXACT NORMALIZED NAME (casefolded, whitespace- and
punctuation-normalised). That is deliberately conservative: it is free, it is
deterministic, and it cannot invent a merge that a human would call wrong. It
will leave near-duplicates ("Basel III" vs "the Basel III framework") separate;
`--embed-threshold` optionally catches those using the `name_embedding` vectors
that are already stored, so no model call is needed there either.

THREE MODES
-----------
  export : read ONE shard's Neo4j → nodes.jsonl + edges.jsonl on disk
  import : load every shard's dump into ONE fresh Neo4j, then stitch + reindex
  report : print the merged graph's census and, crucially, WHICH global windows
           are missing (the gap-fill list for a later --resume pass)

Usage (driven by scripts/cluster_merge_job.sh, which owns the Neo4j lifecycle):

    python -m baselines.graphiti.merge_shards export \
        --bolt-uri bolt://localhost:7690 --group-id gmb_finance_full \
        --shard s3 --dump-dir /fscratch/abuali/neo4j/dumps

    python -m baselines.graphiti.merge_shards import \
        --bolt-uri bolt://localhost:7690 --group-id gmb_finance_full \
        --dump-dir /fscratch/abuali/neo4j/dumps --expected-windows 6002
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from neo4j import GraphDatabase

# ─────────────────────────────────────────────────────────────────────────────
# JSON round-tripping for Neo4j values
# ─────────────────────────────────────────────────────────────────────────────
# Neo4j hands back its own temporal types (neo4j.time.DateTime), which json cannot
# serialise. We tag them on the way out and rebuild real Python datetimes on the
# way in — the driver stores a native datetime as a proper Neo4j DateTime, so the
# merged graph keeps the same TYPES as the shard graphs. That matters: Graphiti
# reads `valid_at` / `invalid_at` as temporal values, and silently turning them
# into strings would break exactly the bi-temporal filtering this whole project
# is about.
_DT_KEY = "__datetime__"


def _to_jsonable(value: Any) -> Any:
    """Neo4j value → something json.dumps can write (datetimes get tagged)."""
    # neo4j.time.DateTime and datetime.datetime both expose isoformat().
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return {_DT_KEY: value.isoformat()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def _from_jsonable(value: Any) -> Any:
    """Inverse of _to_jsonable — tagged dicts become real datetimes again."""
    if isinstance(value, dict):
        if _DT_KEY in value and len(value) == 1:
            return datetime.fromisoformat(value[_DT_KEY])
        return {k: _from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(v) for v in value]
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Identifier safety
# ─────────────────────────────────────────────────────────────────────────────
# Labels and relationship types CANNOT be parameterised in Cypher — they have to
# be interpolated into the query string. Every value we interpolate comes out of
# our own database, but interpolation is interpolation: validate it anyway, and
# refuse rather than build a query we did not intend.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_ident(name: str) -> str:
    """Return `name` if it is a plain Cypher identifier, else raise."""
    if not _IDENT_RE.match(name or ""):
        raise ValueError(f"refusing to interpolate unsafe identifier: {name!r}")
    return name


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT — one shard's subgraph → JSONL
# ─────────────────────────────────────────────────────────────────────────────
def export_shard(driver, group_id: str, shard: str, dump_dir: Path) -> Tuple[int, int]:
    """Dump every node and relationship belonging to `group_id` to JSONL.

    Scoped by group_id rather than "everything in the database" so a shard store
    that also holds an older experiment's graph cannot leak into the merge.

    Returns (node_count, edge_count).
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = dump_dir / f"{shard}.nodes.jsonl"
    edges_path = dump_dir / f"{shard}.edges.jsonl"

    n_nodes = 0
    with driver.session() as session, nodes_path.open("w") as fh:
        # labels(n) is exported alongside the properties because Graphiti gives
        # entities extra type labels (e.g. :Entity:Organization) and the merged
        # graph must keep them — they are what makes a visualization colourable.
        result = session.run(
            "MATCH (n) WHERE n.group_id = $g "
            "RETURN labels(n) AS labels, properties(n) AS props",
            g=group_id,
        )
        for rec in result:
            fh.write(json.dumps({
                "labels": sorted(rec["labels"]),
                "props": _to_jsonable(dict(rec["props"])),
            }) + "\n")
            n_nodes += 1

    n_edges = 0
    with driver.session() as session, edges_path.open("w") as fh:
        # BOTH endpoints must be in the group. An edge to a node we are not
        # exporting would be unimportable (its endpoint would not exist), and
        # Graphiti never creates such an edge anyway — so if this filter ever
        # drops something, that is a signal, not a silent fix.
        result = session.run(
            "MATCH (a)-[r]->(b) WHERE a.group_id = $g AND b.group_id = $g "
            "RETURN type(r) AS type, a.uuid AS src, b.uuid AS dst, "
            "       properties(r) AS props",
            g=group_id,
        )
        for rec in result:
            fh.write(json.dumps({
                "type": rec["type"],
                "src": rec["src"],
                "dst": rec["dst"],
                "props": _to_jsonable(dict(rec["props"])),
            }) + "\n")
            n_edges += 1

    print(f"📦 [merge] exported shard {shard}: {n_nodes} nodes, {n_edges} edges "
          f"→ {dump_dir}", flush=True)
    return n_nodes, n_edges


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT — every shard's JSONL → one Neo4j
# ─────────────────────────────────────────────────────────────────────────────
def _batched(items: Iterable[dict], size: int) -> Iterable[List[dict]]:
    """Yield lists of at most `size` items (keeps each transaction bounded)."""
    batch: List[dict] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def import_nodes(
    driver, dump_dir: Path, batch_size: int = 1000
) -> Tuple[int, Dict[str, str]]:
    """Load every `*.nodes.jsonl` into the target database, MERGEd on uuid.

    Nodes are grouped by their exact label set, because the label list has to be
    baked into the query text (Cypher cannot parameterise labels). MERGE (not
    CREATE) on uuid makes the whole import idempotent: re-running after a crash
    updates rather than duplicates.

    Returns (node_count, uuid -> primary label). That second value is not
    bookkeeping for its own sake — see `import_edges`, which cannot use an index
    without knowing each endpoint's label.
    """
    by_labels: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    uuid_to_label: Dict[str, str] = {}
    for path in sorted(dump_dir.glob("*.nodes.jsonl")):
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                labels = tuple(rec["labels"])
                props = _from_jsonable(rec["props"])
                by_labels[labels].append(props)
                # Labels were sorted on export, so labels[0] is deterministic.
                if labels and props.get("uuid"):
                    uuid_to_label[props["uuid"]] = labels[0]

    total = 0
    for labels, rows in by_labels.items():
        label_str = ":".join(_safe_ident(lbl) for lbl in labels)
        query = (
            f"UNWIND $rows AS props "
            f"MERGE (n:{label_str} {{uuid: props.uuid}}) "
            f"SET n = props"
        )
        with driver.session() as session:
            for batch in _batched(rows, batch_size):
                session.run(query, rows=batch)
        total += len(rows)
        print(f"📥 [merge] imported {len(rows):>7} nodes :{label_str}", flush=True)
    return total, uuid_to_label


def create_uuid_indexes(driver, labels: Iterable[str]) -> None:
    """Index `uuid` on every label we imported, BEFORE any lookup by uuid.

    This is not an optimisation, it is the difference between minutes and hours.
    A `MATCH (n:Label {uuid: $u})` with no index is a full label scan; the edge
    import does two such lookups for each of ~270,000 edges, and the stitch does
    four per duplicate entity. Without indexes that is billions of comparisons.

    Graphiti's own `build_indices_and_constraints()` runs later and creates its
    full index set (including the fulltext and vector indices search needs); these
    plain range indexes just have to exist EARLY. `IF NOT EXISTS` keeps the two
    compatible.
    """
    with driver.session() as session:
        for label in sorted(set(labels)):
            lbl = _safe_ident(label)
            session.run(
                f"CREATE INDEX gmb_merge_{lbl.lower()}_uuid IF NOT EXISTS "
                f"FOR (n:{lbl}) ON (n.uuid)"
            )
        # Wait for population — Neo4j creates indexes asynchronously, and an
        # index that is still POPULATING will not be used by the planner.
        session.run("CALL db.awaitIndexes(300)")
    print(f"🗂️  [merge] uuid indexes ready for labels: "
          f"{', '.join(sorted(set(labels)))}", flush=True)


def import_edges(
    driver, dump_dir: Path, uuid_to_label: Dict[str, str], batch_size: int = 1000
) -> int:
    """Load every `*.edges.jsonl`, MERGEd on the edge's own uuid.

    Edges are grouped by (relationship type, source label, target label) rather
    than by type alone. The reason is purely about indexes: Cypher can only use
    an index when the pattern names a LABEL, so an unlabelled
    `MATCH (a {uuid: ...})` scans every node in the database. With ~11k nodes and
    ~270k edges that is billions of comparisons; with the label it is an index
    seek. Nodes must therefore be imported (and indexed) first.
    """
    # (rel_type, src_label, dst_label) -> rows
    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    skipped = 0
    for path in sorted(dump_dir.glob("*.edges.jsonl")):
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                src_label = uuid_to_label.get(rec["src"])
                dst_label = uuid_to_label.get(rec["dst"])
                if not src_label or not dst_label:
                    # An endpoint we never imported. Graphiti does not create
                    # such edges, so this is a signal worth counting rather than
                    # a condition to swallow silently.
                    skipped += 1
                    continue
                grouped[(rec["type"], src_label, dst_label)].append({
                    "src": rec["src"],
                    "dst": rec["dst"],
                    "props": _from_jsonable(rec["props"]),
                })

    total = 0
    for (rel_type, src_label, dst_label), rows in grouped.items():
        rt, sl, dl = (_safe_ident(rel_type), _safe_ident(src_label),
                      _safe_ident(dst_label))
        query = (
            f"UNWIND $rows AS row "
            f"MATCH (a:{sl} {{uuid: row.src}}), (b:{dl} {{uuid: row.dst}}) "
            f"MERGE (a)-[r:{rt} {{uuid: row.props.uuid}}]->(b) "
            f"SET r = row.props"
        )
        with driver.session() as session:
            for batch in _batched(rows, batch_size):
                session.run(query, rows=batch)
        total += len(rows)
        print(f"📥 [merge] imported {len(rows):>7} edges "
              f"(:{sl})-[:{rt}]->(:{dl})", flush=True)
    if skipped:
        print(f"⚠️  [merge] {skipped} edges skipped — an endpoint was not in any "
              f"node dump", flush=True)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# STITCH — collapse the same entity seen by different shards
# ─────────────────────────────────────────────────────────────────────────────
_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm_name(name: str) -> str:
    """Casefold and strip punctuation/whitespace so 'Basel III' == 'basel iii'.

    Deliberately conservative — it only merges names that a human would call the
    SAME string. Semantic near-duplicates are left to --embed-threshold.
    """
    return _NORM_RE.sub(" ", (name or "").lower()).strip()


def _rewire(session, keep: str, loser: str) -> None:
    """Move every edge off `loser` onto `keep`, then delete `loser`.

    Graphiti stores the endpoints TWICE — once as the actual graph relationship,
    and once as `source_node_uuid` / `target_node_uuid` properties ON the edge,
    which is what `graphiti.search()` reads when it rebuilds EntityEdge objects.
    Rewiring only the relationship would leave the properties pointing at a node
    that no longer exists, and search would break in a way that looks like bad
    retrieval rather than a bad merge. So both are updated together.
    """
    # Outgoing facts: (loser)-[:RELATES_TO]->(other)  becomes (keep)->(other).
    # Self-loops (other == keep) are dropped: after collapsing duplicates, an edge
    # from "Basel III"(s3) to "Basel III"(s7) is a statement about one node.
    session.run(
        """
        MATCH (keep:Entity {uuid: $keep}), (loser:Entity {uuid: $loser})
        MATCH (loser)-[r:RELATES_TO]->(o:Entity)
        WHERE o.uuid <> $keep
        CREATE (keep)-[r2:RELATES_TO]->(o)
        SET r2 = properties(r), r2.source_node_uuid = $keep
        DELETE r
        """,
        keep=keep, loser=loser,
    )
    # Incoming facts: (other)-[:RELATES_TO]->(loser)  becomes (other)->(keep).
    session.run(
        """
        MATCH (keep:Entity {uuid: $keep}), (loser:Entity {uuid: $loser})
        MATCH (o:Entity)-[r:RELATES_TO]->(loser)
        WHERE o.uuid <> $keep
        CREATE (o)-[r2:RELATES_TO]->(keep)
        SET r2 = properties(r), r2.target_node_uuid = $keep
        DELETE r
        """,
        keep=keep, loser=loser,
    )
    # Episode provenance: (episode)-[:MENTIONS]->(loser) becomes ->(keep). This is
    # what preserves "which window said this", i.e. the mapping the retriever uses
    # to turn a fact back into a message index.
    session.run(
        """
        MATCH (keep:Entity {uuid: $keep}), (loser:Entity {uuid: $loser})
        MATCH (e:Episodic)-[r:MENTIONS]->(loser)
        MERGE (e)-[r2:MENTIONS]->(keep)
        ON CREATE SET r2 = properties(r)
        DELETE r
        """,
        keep=keep, loser=loser,
    )
    # Anything still attached (e.g. community membership) goes with the node.
    session.run("MATCH (loser:Entity {uuid: $loser}) DETACH DELETE loser",
                loser=loser)


def stitch_entities(driver, group_id: str) -> Dict[str, int]:
    """Collapse duplicate :Entity nodes by normalized name. Returns a stats dict.

    Survivor = the OLDEST node (smallest created_at), so the merged entity keeps
    the identity that appeared earliest in the conversation timeline — the same
    node a single serial ingest would have created and then reused.
    """
    with driver.session() as session:
        rows = session.run(
            "MATCH (n:Entity {group_id: $g}) "
            "RETURN n.uuid AS uuid, n.name AS name, n.created_at AS created_at",
            g=group_id,
        ).data()

    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        key = _norm_name(row["name"])
        if key:                       # a nameless entity has nothing to match on
            groups[key].append(row)

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    merged = 0
    for key, members in dup_groups.items():
        # Sort by created_at; None sorts last so a node with a timestamp always
        # wins over one without.
        members.sort(key=lambda r: (r["created_at"] is None, r["created_at"]))
        keep = members[0]["uuid"]
        with driver.session() as session:
            for loser in members[1:]:
                _rewire(session, keep, loser["uuid"])
                merged += 1

    stats = {
        "entities_before": len(rows),
        "duplicate_groups": len(dup_groups),
        "entities_deleted": merged,
        "entities_after": len(rows) - merged,
    }
    print(f"🔗 [merge] stitched {merged} duplicate entities across "
          f"{len(dup_groups)} name groups: "
          f"{stats['entities_before']} → {stats['entities_after']}", flush=True)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# REPORT — census + the gap-fill list
# ─────────────────────────────────────────────────────────────────────────────
def report(driver, group_id: str, expected_windows: int) -> Dict[str, Any]:
    """Print what the merged graph contains and WHICH windows never made it.

    The missing-window list is the practical payoff: those are the episodes lost
    to JSONDecodeError skips during ingest. Feeding them back through a --resume
    pass is far cheaper than re-running anything, because extraction at
    temperature > 0 usually parses on the second attempt.
    """
    with driver.session() as session:
        counts = session.run(
            """
            MATCH (n {group_id: $g})
            WITH labels(n) AS labels
            UNWIND labels AS label
            RETURN label, count(*) AS n ORDER BY n DESC
            """,
            g=group_id,
        ).data()
        rels = session.run(
            "MATCH (a {group_id: $g})-[r]->(b {group_id: $g}) "
            "RETURN type(r) AS type, count(r) AS n ORDER BY n DESC",
            g=group_id,
        ).data()
        # Only currently-valid facts are what a "what is the CURRENT X?" question
        # should see — worth reporting because it is the differentiator vs BM25.
        valid = session.run(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.group_id = $g AND r.invalid_at IS NULL "
            "RETURN count(r) AS n",
            g=group_id,
        ).single()["n"]
        names = session.run(
            "MATCH (e:Episodic {group_id: $g}) RETURN e.name AS name", g=group_id
        ).data()

    print("\n📊 [merge] node labels:")
    for row in counts:
        print(f"     {row['label']:<24} {row['n']:>8}")
    print("📊 [merge] relationships:")
    for row in rels:
        print(f"     {row['type']:<24} {row['n']:>8}")
    print(f"📊 [merge] currently-valid facts (invalid_at IS NULL): {valid}")

    # Episode names are `{group_id}_w{N}` — parse N back out and diff against the
    # full 1..expected_windows range.
    pat = re.compile(rf"^{re.escape(group_id)}_w(\d+)$")
    present = {int(m.group(1)) for n in names
               if (m := pat.match(n.get("name") or ""))}
    missing = sorted(set(range(1, expected_windows + 1)) - present)

    pct = 100.0 * len(present) / expected_windows if expected_windows else 0.0
    print(f"\n📊 [merge] episodes: {len(present)}/{expected_windows} ({pct:.2f}%)")
    if missing:
        print(f"⚠️  [merge] {len(missing)} windows MISSING — gap-fill these with a "
              f"--resume pass:")
        print(f"     {missing}")
    else:
        print("✅ [merge] every window is present")

    return {
        "labels": counts,
        "relationships": rels,
        "valid_facts": valid,
        "episodes_present": len(present),
        "episodes_expected": expected_windows,
        "missing_windows": missing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Index rebuild
# ─────────────────────────────────────────────────────────────────────────────
def rebuild_indices(bolt_uri: str, user: str, password: str) -> None:
    """Recreate Graphiti's fulltext + vector indices on the merged database.

    A fresh database has none, and without them `graphiti.search()` silently
    degrades (the hybrid search loses its BM25 and vector legs and falls back to
    graph traversal only) — which would look like a bad retriever rather than a
    missing index.

    We call Graphiti's own `build_indices_and_constraints()` rather than
    hand-writing the DDL, so the index set can never drift from what the installed
    graphiti-core version expects. The three clients below are never invoked by
    that call; they exist only because the constructor builds them eagerly.

    ‼️  ALL THREE must be passed. Graphiti's constructor defaults `cross_encoder`
    to a bare `OpenAIRerankerClient()`, which builds an `AsyncOpenAI` with no
    key and raises `OpenAIError: Missing credentials` — even though we are only
    creating indexes and no model is ever called. That killed the first merge run
    (2026-08-04) *after* a successful import and stitch, at the very last step.
    """
    from graphiti_core import Graphiti
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    import asyncio

    cfg = LLMConfig(api_key="unused", model="unused", base_url="http://localhost:1")
    graphiti = Graphiti(
        bolt_uri, user, password,
        llm_client=OpenAIGenericClient(config=cfg),
        embedder=OpenAIEmbedder(config=OpenAIEmbedderConfig(
            api_key="unused", embedding_model="unused",
            base_url="http://localhost:1",
        )),
        cross_encoder=OpenAIRerankerClient(config=cfg),
    )
    asyncio.run(graphiti.build_indices_and_constraints())
    print("🗂️  [merge] indices and constraints rebuilt", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["export", "import", "report"])
    ap.add_argument("--bolt-uri", required=True)
    ap.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--neo4j-password",
                    default=os.getenv("NEO4J_PASSWORD", "graphiti123"))
    ap.add_argument("--group-id", default="gmb_finance_full")
    ap.add_argument("--dump-dir", required=True,
                    help="Where the per-shard JSONL dumps live.")
    ap.add_argument("--shard", default=None,
                    help="export mode: which shard is being dumped (names the files).")
    ap.add_argument("--expected-windows", type=int, default=6002,
                    help="Total global windows, for the missing-window report.")
    ap.add_argument("--no-stitch", action="store_true",
                    help="import mode: skip cross-shard entity deduplication.")
    ap.add_argument("--stats-json", default=None,
                    help="import/report mode: also write the census to this path.")
    args = ap.parse_args()

    driver = GraphDatabase.driver(
        args.bolt_uri, auth=(args.neo4j_user, args.neo4j_password)
    )
    try:
        if args.mode == "export":
            if not args.shard:
                ap.error("--shard is required in export mode")
            export_shard(driver, args.group_id, args.shard, Path(args.dump_dir))
            return 0

        if args.mode == "import":
            n, uuid_to_label = import_nodes(driver, Path(args.dump_dir))
            # Indexes MUST exist before any uuid lookup — see create_uuid_indexes.
            create_uuid_indexes(driver, uuid_to_label.values())
            e = import_edges(driver, Path(args.dump_dir), uuid_to_label)
            print(f"📥 [merge] total: {n} nodes, {e} edges", flush=True)
            stitch_stats = {}
            if not args.no_stitch:
                stitch_stats = stitch_entities(driver, args.group_id)
            driver.close()
            # Indices are rebuilt through a fresh Graphiti client, which opens its
            # own driver — so ours is closed first.
            rebuild_indices(args.bolt_uri, args.neo4j_user, args.neo4j_password)
            driver = GraphDatabase.driver(
                args.bolt_uri, auth=(args.neo4j_user, args.neo4j_password)
            )
            stats = report(driver, args.group_id, args.expected_windows)
            stats["stitch"] = stitch_stats
        else:
            stats = report(driver, args.group_id, args.expected_windows)

        if args.stats_json:
            Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.stats_json).write_text(json.dumps(stats, indent=2))
            print(f"📝 [merge] census written to {args.stats_json}", flush=True)
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
