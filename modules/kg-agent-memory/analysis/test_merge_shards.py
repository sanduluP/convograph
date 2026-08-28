#!/usr/bin/env python3
"""End-to-end test of the shard merge, against a real Neo4j on a throwaway group.

WHY A TEST AT ALL
-----------------
`merge_shards.py` is the step that decides whether ~40 h of GPU ingest is usable.
If it drops an edge, mis-wires `source_node_uuid`, or turns a temporal value into
a string, the damage shows up much later as "Graphiti retrieves badly" — which is
the most expensive kind of bug to chase. So we prove the mechanics on a tiny
synthetic graph first, where every expected number is countable by hand.

WHAT IT SIMULATES
-----------------
Two shards that each saw part of the same conversation and therefore each created
their OWN copy of the shared entity "Basel III" (different UUIDs — exactly what
happens on the cluster, because a shard can only deduplicate against its own
graph):

    shard A : episodes w1, w2   entities  Basel III(A), Treasury Team
    shard B : episode  w3       entities  Basel III(B), Audit Committee

After merging we expect:
    episodes            3   (w1, w2, w3 — nothing lost, nothing duplicated)
    entities before     4
    entities after      3   (the two Basel III copies collapse into one)
    RELATES_TO          3   (all facts survive the rewire)
    missing windows   [4]   (we claim 4 expected windows; w4 was never ingested)

It runs against the LOCAL Docker Neo4j on a group_id nobody else uses, and
deletes that group on the way in and out, so it can never touch real data.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neo4j import GraphDatabase

from baselines.graphiti.merge_shards import (
    create_uuid_indexes,
    export_shard,
    import_edges,
    import_nodes,
    report,
    stitch_entities,
)
from baselines.rag_common.eval_lib import load_env_file

GROUP = "merge_selftest"          # a namespace no real run uses
NOW = datetime(2026, 8, 4, 9, 0, tzinfo=timezone.utc)

# Failures are collected rather than asserted one at a time, so a single run
# reports EVERY problem instead of only the first one.
FAILURES: list[str] = []


def check(label: str, got, want) -> None:
    """Record a comparison; print it either way so the log shows the evidence."""
    ok = got == want
    print(f"   {'✅' if ok else '❌'} {label}: got {got!r}, want {want!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def wipe(driver) -> None:
    """Remove the test group entirely (safe to call before and after)."""
    with driver.session() as s:
        s.run("MATCH (n {group_id: $g}) DETACH DELETE n", g=GROUP)


def build_shard_a(driver) -> None:
    """Episodes w1+w2, entities 'Basel III' (uuid a-basel) and 'Treasury Team'."""
    with driver.session() as s:
        s.run(
            """
            CREATE (e1:Episodic {uuid:'a-ep1', name:$n1, group_id:$g,
                                 created_at:$t, valid_at:$t, content:'w1 body'})
            CREATE (e2:Episodic {uuid:'a-ep2', name:$n2, group_id:$g,
                                 created_at:$t, valid_at:$t, content:'w2 body'})
            CREATE (b:Entity {uuid:'a-basel', name:'Basel III', group_id:$g,
                              created_at:$t, summary:'capital rules',
                              name_embedding:[0.1,0.2,0.3]})
            CREATE (t:Entity {uuid:'a-treas', name:'Treasury Team', group_id:$g,
                              created_at:$t, summary:'the team'})
            CREATE (e1)-[:MENTIONS {uuid:'a-m1', group_id:$g, created_at:$t}]->(b)
            CREATE (e2)-[:MENTIONS {uuid:'a-m2', group_id:$g, created_at:$t}]->(t)
            // A fact whose SOURCE is the duplicated entity — after stitching, its
            // source_node_uuid property must point at the survivor.
            CREATE (b)-[:RELATES_TO {uuid:'a-r1', group_id:$g, name:'APPLIES_TO',
                                     fact:'Basel III applies to the Treasury Team',
                                     fact_embedding:[0.4,0.5,0.6],
                                     source_node_uuid:'a-basel',
                                     target_node_uuid:'a-treas',
                                     episodes:['a-ep1'], created_at:$t,
                                     valid_at:$t, invalid_at:null}]->(t)
            // A fact whose TARGET is the duplicated entity (tests the other
            // rewiring branch), and which is SUPERSEDED — invalid_at set — so we
            // also prove a temporal value survives the JSON round-trip.
            CREATE (t)-[:RELATES_TO {uuid:'a-r2', group_id:$g, name:'TRACKS',
                                     fact:'Treasury Team tracks Basel III',
                                     fact_embedding:[0.7,0.8,0.9],
                                     source_node_uuid:'a-treas',
                                     target_node_uuid:'a-basel',
                                     episodes:['a-ep2'], created_at:$t,
                                     valid_at:$t, invalid_at:$t}]->(b)
            """,
            g=GROUP, t=NOW, n1=f"{GROUP}_w1", n2=f"{GROUP}_w2",
        )


def build_shard_b(driver) -> None:
    """Episode w3 with its OWN 'Basel III' copy — the cross-shard duplicate."""
    with driver.session() as s:
        s.run(
            """
            CREATE (e3:Episodic {uuid:'b-ep3', name:$n3, group_id:$g,
                                 created_at:$t, valid_at:$t, content:'w3 body'})
            // Same name, different uuid, created LATER — so the stitcher must
            // keep 'a-basel' (the older one) and delete this one.
            CREATE (b:Entity {uuid:'b-basel', name:'basel  III', group_id:$g,
                              created_at:$t2, summary:'capital rules again',
                              name_embedding:[0.1,0.2,0.31]})
            CREATE (c:Entity {uuid:'b-audit', name:'Audit Committee', group_id:$g,
                              created_at:$t2, summary:'the committee'})
            CREATE (e3)-[:MENTIONS {uuid:'b-m3', group_id:$g, created_at:$t2}]->(b)
            CREATE (e3)-[:MENTIONS {uuid:'b-m4', group_id:$g, created_at:$t2}]->(c)
            CREATE (b)-[:RELATES_TO {uuid:'b-r3', group_id:$g, name:'REVIEWED_BY',
                                     fact:'Basel III is reviewed by the Audit Committee',
                                     fact_embedding:[0.2,0.2,0.2],
                                     source_node_uuid:'b-basel',
                                     target_node_uuid:'b-audit',
                                     episodes:['b-ep3'], created_at:$t2,
                                     valid_at:$t2, invalid_at:null}]->(c)
            """,
            g=GROUP, t=NOW, t2=NOW.replace(hour=11), n3=f"{GROUP}_w3",
        )


def main() -> int:
    load_env_file(str(REPO_ROOT / ".env"))
    import os
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "graphiti123")
    print(f"🔌 connecting to {uri}")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    dump_dir = Path(tempfile.mkdtemp(prefix="merge_selftest_"))
    try:
        # ── 1. build + export shard A ────────────────────────────────────────
        print("\n🏗️  building shard A")
        wipe(driver)
        build_shard_a(driver)
        n_a, e_a = export_shard(driver, GROUP, "sA", dump_dir)
        check("shard A nodes exported", n_a, 4)
        check("shard A edges exported", e_a, 4)

        # ── 2. build + export shard B (in the same store, after wiping) ──────
        # Wiping between the two is what makes them independent shards: shard B's
        # export must not accidentally contain shard A's nodes.
        print("\n🏗️  building shard B")
        wipe(driver)
        build_shard_b(driver)
        n_b, e_b = export_shard(driver, GROUP, "sB", dump_dir)
        check("shard B nodes exported", n_b, 3)
        check("shard B edges exported", e_b, 3)

        # ── 3. import both into an empty store ───────────────────────────────
        print("\n📥 importing both dumps into an empty group")
        wipe(driver)
        n_imported, uuid_to_label = import_nodes(driver, dump_dir)
        # Indexes must exist before any uuid lookup — without them the edge
        # import falls back to a full scan per endpoint (minutes vs hours at
        # real scale). Asserting the map is built correctly is part of that.
        check("uuid->label map covers every node", len(uuid_to_label), 7)
        check("episode labelled correctly", uuid_to_label.get("a-ep1"), "Episodic")
        check("entity labelled correctly", uuid_to_label.get("a-basel"), "Entity")
        create_uuid_indexes(driver, uuid_to_label.values())
        e_imported = import_edges(driver, dump_dir, uuid_to_label)
        check("nodes imported", n_imported, 7)
        check("edges imported", e_imported, 7)

        # Temporal values must still be temporal, not strings — this is the whole
        # basis of the bi-temporal filtering the project depends on.
        with driver.session() as s:
            row = s.run(
                "MATCH ()-[r:RELATES_TO {uuid:'a-r2'}]->() "
                "RETURN r.invalid_at AS iv"
            ).single()
        check("invalid_at survived as a temporal value",
              type(row["iv"]).__name__, "DateTime")

        # ── 4. stitch the cross-shard duplicate ─────────────────────────────
        print("\n🔗 stitching duplicate entities")
        stats = stitch_entities(driver, GROUP)
        check("entities before stitch", stats["entities_before"], 4)
        check("duplicate name groups", stats["duplicate_groups"], 1)
        check("entities after stitch", stats["entities_after"], 3)

        with driver.session() as s:
            # The OLDER copy must be the survivor.
            survived = s.run(
                "MATCH (n:Entity {group_id:$g}) WHERE n.uuid IN ['a-basel','b-basel'] "
                "RETURN n.uuid AS uuid", g=GROUP
            ).data()
            check("survivor is the older copy",
                  sorted(r["uuid"] for r in survived), ["a-basel"])

            # No fact may be lost by the rewire.
            n_rel = s.run(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.group_id=$g RETURN count(r) AS n",
                g=GROUP
            ).single()["n"]
            check("facts survived the rewire", n_rel, 3)

            # The fact that shard B attached to its own copy must now hang off the
            # survivor, in BOTH the relationship and the stored property.
            moved = s.run(
                "MATCH (a:Entity)-[r:RELATES_TO {uuid:'b-r3'}]->(b:Entity) "
                "RETURN a.uuid AS src, r.source_node_uuid AS src_prop"
            ).single()
            check("rewired relationship endpoint", moved["src"], "a-basel")
            check("rewired source_node_uuid property", moved["src_prop"], "a-basel")

            # Episode provenance must follow too, or a fact can no longer be
            # traced back to a message index.
            mentions = s.run(
                "MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid:'a-basel'}) "
                "RETURN count(e) AS n"
            ).single()["n"]
            check("MENTIONS edges rewired onto the survivor", mentions, 2)

        # ── 5. report / missing-window detection ────────────────────────────
        print("\n📊 census")
        rep = report(driver, GROUP, expected_windows=4)
        check("episodes present", rep["episodes_present"], 3)
        check("missing windows detected", rep["missing_windows"], [4])
        check("currently-valid facts", rep["valid_facts"], 2)

    finally:
        wipe(driver)
        driver.close()
        shutil.rmtree(dump_dir, ignore_errors=True)

    print("\n" + "═" * 60)
    if FAILURES:
        print(f"❌ {len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"   • {f}")
        return 1
    print("✅ all merge checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
