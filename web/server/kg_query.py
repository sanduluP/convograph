"""
Read one session's graph out of Neo4j: entities as nodes, facts as edges.

Runs inside MODULE 2's venv (which has the neo4j driver and the .env), invoked
as a subprocess by the web backend — module 2's code stays untouched, and the
web app's own venv never needs a bolt driver:

    modules/kg-agent-memory/.venv/bin/python web/server/kg_query.py <group_id>

Prints one JSON object:
    {"nodes": [{id, label}],
     "edges": [{id, source, target, fact, valid_at, invalid_at, episodes}]}

Graphiti's model is entities (:Entity) joined by RELATES_TO relationships that
carry the fact sentence plus bi-temporal validity — exactly the node/edge/
status split the design's graph panel wants. invalid_at set = the conversation
overturned this fact (drawn dashed + struck through).
"""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KG_MODULE = os.path.join(REPO_ROOT, "modules", "kg-agent-memory")


def load_env() -> None:
    path = os.path.join(KG_MODULE, ".env")
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    group_id = sys.argv[1]
    load_env()

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )
    db = os.environ.get("NEO4J_DATABASE") or None

    with driver.session(database=db) as neo:
        records = neo.run(
            "MATCH (a:Entity {group_id: $gid})-[r:RELATES_TO]->(b:Entity {group_id: $gid}) "
            "RETURN a.uuid AS a_id, a.name AS a_name, b.uuid AS b_id, b.name AS b_name, "
            "       r.uuid AS id, r.fact AS fact, r.valid_at AS valid_at, "
            "       r.invalid_at AS invalid_at, r.episodes AS episodes",
            gid=group_id,
        ).data()
    driver.close()

    nodes: dict[str, dict] = {}
    edges = []
    for r in records:
        nodes[r["a_id"]] = {"id": r["a_id"], "label": r["a_name"]}
        nodes[r["b_id"]] = {"id": r["b_id"], "label": r["b_name"]}
        edges.append({
            "id": r["id"],
            "source": r["a_id"],
            "target": r["b_id"],
            "fact": r["fact"],
            "valid_at": str(r["valid_at"]) if r["valid_at"] else None,
            "invalid_at": str(r["invalid_at"]) if r["invalid_at"] else None,
            "episodes": r["episodes"] or [],
        })

    print(json.dumps({"nodes": list(nodes.values()), "edges": edges},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
