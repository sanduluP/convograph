#!/usr/bin/env python3
"""
check_supersession.py — can we recover WHICH fact replaced a superseded one?

THE PROBLEM
-----------
Graphiti's bi-temporal layer records that a fact stopped being true
(`invalid_at`) but stores no pointer to the fact that replaced it. EntityEdge
has valid_at / invalid_at / expired_at and nothing else. So "what superseded
this?" has to be INFERRED.

THE JOIN, AND WHY IT IS EXACT
-----------------------------
graphiti_core/utils/maintenance/edge_operations.py:569 does, literally:

    edge.invalid_at = resolved_edge.valid_at

The superseding fact's valid_at is copied into the old fact's invalid_at. So the
successor is not approximated by "the next fact after this one" — it is joined
on EXACT timestamp equality.

WHY THAT IS NOT ENOUGH ON ITS OWN
---------------------------------
Two complications, which this script measures rather than assumes:

  1. Many edges can share one valid_at (episodes in this corpus routinely share
     a timestamp), so the equality alone can match a crowd.
  2. Invalidation candidates come from a SEMANTIC search around the edge, not
     from the same node pair, so the successor does not have to connect the same
     two entities — it only usually does.

So the query REQUIRES a shared endpoint and ranks what is left: same
(source,target) pair first, then merely sharing one entity. A candidate that
shares no entity is a coincidental timestamp collision, not a successor, and the
query returns null rather than guessing.

Measured on gmb_finance_full, 300 superseded facts (2026-09-07):
    52%  successor is the same entity pair
    39%  successor shares one entity
     9%  no successor found -> null
Without the shared-endpoint requirement, 95% of rows have several candidates and
the widest has 113.

Usage:
    python analysis/check_supersession.py --group-id gmb_finance_full --limit 200
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The same cypher the .cypher file hands to Neo4j Browser, so what we measure is
# what a human will paste. Kept as ONE string for exactly that reason.
SUCCESSOR_CYPHER = """
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.invalid_at IS NOT NULL
  AND ($group IS NULL OR r.group_id = $group)
WITH r, a, b LIMIT $limit
OPTIONAL CALL (r, a, b) {
    MATCH (x:Entity)-[s:RELATES_TO]->(y:Entity)
    WHERE s.valid_at = r.invalid_at
      AND s.uuid <> r.uuid
      AND s.group_id = r.group_id
      // Share at least one endpoint. Without this the join is 95% ambiguous
      // (up to 113 candidates), because many edges carry the same valid_at -
      // a coincidental timestamp collision is not a successor.
      AND (x.uuid IN [a.uuid, b.uuid] OR y.uuid IN [a.uuid, b.uuid])
    RETURN s,
           CASE WHEN x.uuid = a.uuid AND y.uuid = b.uuid THEN 0
                WHEN x.uuid = b.uuid AND y.uuid = a.uuid THEN 1
                ELSE 2 END AS tier
    ORDER BY tier, s.created_at
    LIMIT 1
}
RETURN r.uuid AS uuid, tier
"""

COVERAGE_CYPHER = """
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.invalid_at IS NOT NULL
  AND ($group IS NULL OR r.group_id = $group)
WITH r LIMIT $limit
OPTIONAL MATCH (x:Entity)-[s:RELATES_TO]->(y:Entity)
WHERE s.valid_at = r.invalid_at AND s.uuid <> r.uuid AND s.group_id = r.group_id
RETURN r.uuid AS uuid, count(s) AS candidates
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group-id", default=None,
                    help="restrict to one group (default: whole database)")
    ap.add_argument("--limit", type=int, default=200,
                    help="how many superseded facts to test (default 200)")
    ap.add_argument("--verify-file", action="store_true",
                    help="run section 3 of eyeball_kg.cypher AS WRITTEN and show "
                         "rows, so the file we hand over is known to execute")
    ap.add_argument("--env-file", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    args = ap.parse_args()

    import ui_ingest
    ui_ingest._load_env_file(args.env_file)
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    db = os.getenv("NEO4J_DATABASE", "neo4j")

    if args.verify_file:
        # Read the statements out of the .cypher file itself. Testing a copy
        # would prove nothing about the file someone actually pastes.
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "eyeball_kg.cypher")
        text = open(path).read()
        section = text[text.index("// 3. THE BI-TEMPORAL"):text.index("// 4. A PICTURE")]
        stmts = [q.strip() for q in section.split(";") if "MATCH" in q]
        with driver.session(database=db) as sess:
            for i, q in enumerate(stmts, 1):
                # Strip the leading comment block so the error, if any, points
                # at cypher rather than at a wall of prose.
                body = "\n".join(l for l in q.splitlines()
                                 if not l.strip().startswith("//"))
                try:
                    rows = list(sess.run(body))
                except Exception as exc:                       # noqa: BLE001
                    print(f"❌ statement {i} FAILED: {type(exc).__name__}: "
                          f"{str(exc)[:300]}")
                    continue
                print(f"✅ statement {i} ran — {len(rows)} row(s)")
                for row in rows[:2]:
                    for k, v in row.items():
                        sv = str(v)
                        print(f"      {k:<24} {sv[:96]}")
                    print()
        driver.close()
        return

    params = {"group": args.group_id, "limit": args.limit}
    with driver.session(database=db) as s:
        cov = list(s.run(COVERAGE_CYPHER, **params))
        tiers = list(s.run(SUCCESSOR_CYPHER, **params))
    driver.close()

    n = len(cov)
    if not n:
        raise SystemExit("no superseded facts found — check --group-id")

    none_ = sum(1 for r in cov if r["candidates"] == 0)
    one = sum(1 for r in cov if r["candidates"] == 1)
    many = sum(1 for r in cov if r["candidates"] > 1)
    widest = max((r["candidates"] for r in cov), default=0)

    print(f"🔍 superseded facts tested : {n}")
    print(f"   ✅ exactly one candidate : {one:>5}  ({100*one/n:.0f}%)")
    print(f"   ⚠️  several candidates    : {many:>5}  ({100*many/n:.0f}%)  "
          f"widest = {widest}")
    print(f"   ❌ no candidate at all    : {none_:>5}  ({100*none_/n:.0f}%)")
    print()

    names = {0: "same pair, same direction", 1: "same pair, reversed",
             2: "shares one entity", None: "no successor found"}
    counts: dict[int, int] = {}
    for r in tiers:
        counts[r["tier"]] = counts.get(r["tier"], 0) + 1
    print("── which tier the chosen successor came from ──")
    for t in sorted(counts, key=lambda v: (v is None, v)):
        print(f"   tier {t}  {names[t]:<26} {counts[t]:>5}  "
              f"({100*counts[t]/max(1,len(tiers)):.0f}%)")

    if none_:
        print(f"\n💡 {none_} superseded fact(s) have no edge whose valid_at equals "
              f"their invalid_at.\n   Expected: the successor can be pruned by "
              f"dedup, or live in another group.\n   The query returns them with a "
              f"null successor rather than dropping the row.")


if __name__ == "__main__":
    main()
