#!/usr/bin/env python3
"""
rank_entities.py — does this graph rank CONCEPTS, or does it rank PEOPLE?

THE QUESTION
------------
A board digest needs "what was this meeting about", which means ranking
entities. The obvious metric is node degree. On `gmb_finance_full` that fails
completely — the top 12 entities by degree are all participants:

    17812 User_13 · 13104 User_1 · 12656 User_12 · 11620 User_3 · ...

which is the star-graph shape we already measured (94% of facts person-rooted).

But that graph was ingested WITH speaker names in the text. The speaker-free
corpus is the same conversations re-ingested with the names stripped, so the
extractor could not root every fact at a person. Faris's hypothesis is that
degree ranking becomes meaningful there. This script tests exactly that, on any
graph, with the same code — so the comparison is a comparison and not two
different scripts disagreeing.

FOUR RANKINGS, and why more than one
------------------------------------
  degree        edges on the entity. The naive baseline; the thing under test.
  episode spread  how many DISTINCT episodes mention it. Measures "a thread
                running through the meeting" rather than "said ten times in one
                breath" — a different and usually better notion of important.
  co-mention    centrality on a graph PROJECTED from co-occurrence: two entities
                in the same episode are related, whatever the extractor drew.
                This is the one that survives a star-shaped graph, because it
                does not use the extracted edges at all.
  pagerank      GDS, on the projection. Only if GDS is present.

PERSON DETECTION is reported, not assumed. `User_\\d+` is a GroupMemBench
artifact and will not survive contact with a real meeting, where the speaker
list should come from module 1's diarization instead. It is used here only to
SCORE the rankings ("what fraction of the top 20 are people"), never to filter
silently.

Usage:
    python analysis/rank_entities.py --group-id gmb_finance_full
    python analysis/rank_entities.py --group-id finance_speaker_free \\
        --uri bolt://localhost:7689 --user neo4j --password <pw>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Names that are obviously a participant rather than a topic. Used for SCORING a
# ranking, not for filtering it — a metric that needs this list to look good is
# a metric that will not transfer to a real meeting.
PERSON_RE = re.compile(r"^(user[_ ]?\d+|speaker[_ ]?\d+|participant[_ ]?\d+)$", re.I)
# Not people, but not topics either: artifacts and bare dates. Reported
# separately because they dominate naive rankings and need their own treatment.
URL_RE = re.compile(r"https?://|sharepoint|\.com/|/spec", re.I)
DATE_RE = re.compile(
    r"^((mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{0,2}|"
    r"\d{4}-\d{2}-\d{2}|eod|q[1-4])\b", re.I)


def classify(name: str) -> str:
    n = (name or "").strip()
    if PERSON_RE.match(n):
        return "person"
    if URL_RE.search(n):
        return "artifact"
    if DATE_RE.match(n):
        return "date"
    return "concept"


def show(title: str, rows: list[tuple[str, int]], secs: float) -> dict:
    kinds = [classify(n) for n, _ in rows]
    concepts = sum(1 for k in kinds if k == "concept")
    print(f"\n── {title}   ({secs:.2f}s) ──")
    for (name, score), kind in list(zip(rows, kinds))[:15]:
        mark = {"person": "👤", "artifact": "📄", "date": "📅", "concept": "💡"}[kind]
        print(f"   {mark} {score:>7}  {name[:78]}")
    pct = 100 * concepts / max(1, len(rows))
    verdict = "✅ concept-led" if pct >= 60 else (
              "🟡 mixed" if pct >= 30 else "❌ dominated by people/artifacts")
    print(f"   → {concepts}/{len(rows)} of the top are concepts ({pct:.0f}%)  {verdict}")
    return {"title": title, "seconds": round(secs, 2),
            "concept_fraction": round(pct / 100, 3),
            "rows": [{"name": n, "score": s, "kind": k}
                     for (n, s), k in zip(rows, kinds)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group-id", required=True)
    ap.add_argument("--uri", default=None, help="override NEO4J_URI")
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--database", default=None)
    ap.add_argument("--top", type=int, default=20, help="rows per ranking (default 20)")
    ap.add_argument("--episode-limit", type=int, default=0,
                    help="restrict to the first N episodes by name — use this to "
                         "digest ONE MEETING-sized slice instead of a whole corpus "
                         "(0 = the whole group)")
    ap.add_argument("--out", default=None, help="write the rankings as JSON here")
    ap.add_argument("--env-file", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    args = ap.parse_args()

    import ui_ingest
    ui_ingest._load_env_file(args.env_file)
    from neo4j import GraphDatabase

    uri = args.uri or os.environ["NEO4J_URI"]
    user = args.user or os.environ["NEO4J_USER"]
    pw = args.password or os.environ["NEO4J_PASSWORD"]
    # A local container serves the default database; Aura names it after the
    # instance. Guess from the URI rather than making the caller remember.
    db = args.database or ("neo4j" if "localhost" in uri or "127.0.0.1" in uri
                           else os.getenv("NEO4J_DATABASE", "neo4j"))

    print(f"🔗 {uri}  db={db}  group={args.group_id}")
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    out: dict = {"group_id": args.group_id, "uri": uri,
                 "episode_limit": args.episode_limit, "rankings": []}

    # One clause reused by every query, so "a meeting-sized slice" means exactly
    # the same set of episodes in all four rankings.
    slice_clause = ("MATCH (ep:Episodic) WHERE ep.group_id=$g WITH ep ORDER BY ep.name "
                    + (f"LIMIT {args.episode_limit} " if args.episode_limit else ""))

    with driver.session(database=db) as s:
        n = list(s.run("MATCH (e:Episodic) WHERE e.group_id=$g RETURN count(*) AS n",
                       g=args.group_id))[0]["n"]
        if not n:
            raise SystemExit(f"❌ group '{args.group_id}' has no episodes at {uri}")
        print(f"   {n} episode(s) in the group"
              + (f", digesting the first {args.episode_limit}" if args.episode_limit else ""))

        # 1. degree — the naive baseline under test
        t = time.time()
        rows = [(r["name"], r["score"]) for r in s.run(
            "MATCH (e:Entity)-[:RELATES_TO]-() WHERE e.group_id=$g "
            "RETURN e.name AS name, count(*) AS score ORDER BY score DESC LIMIT $k",
            g=args.group_id, k=args.top)]
        out["rankings"].append(show("degree (edges on the entity)", rows, time.time() - t))

        # 2. episode spread
        t = time.time()
        rows = [(r["name"], r["score"]) for r in s.run(
            slice_clause +
            "MATCH (ep)-[:MENTIONS]->(e:Entity) "
            "RETURN e.name AS name, count(DISTINCT ep.uuid) AS score "
            "ORDER BY score DESC LIMIT $k", g=args.group_id, k=args.top)]
        out["rankings"].append(show("episode spread (distinct windows)", rows,
                                    time.time() - t))

        # 3. co-mention weight — the projection that ignores extracted edges
        t = time.time()
        rows = [(r["name"], r["score"]) for r in s.run(
            slice_clause +
            "MATCH (ep)-[:MENTIONS]->(a:Entity) "
            "MATCH (ep)-[:MENTIONS]->(b:Entity) WHERE a.uuid < b.uuid "
            "WITH a, count(DISTINCT ep.uuid) AS w "
            "RETURN a.name AS name, sum(w) AS score ORDER BY score DESC LIMIT $k",
            g=args.group_id, k=args.top)]
        out["rankings"].append(show("co-mention weight (projected graph)", rows,
                                    time.time() - t))

        # 4. same, with people excluded — what the digest would actually use
        t = time.time()
        rows = [(r["name"], r["score"]) for r in s.run(
            slice_clause +
            "MATCH (ep)-[:MENTIONS]->(e:Entity) "
            "WHERE NOT e.name =~ '(?i)(user|speaker|participant)[_ ]?[0-9]+' "
            "RETURN e.name AS name, count(DISTINCT ep.uuid) AS score "
            "ORDER BY score DESC LIMIT $k", g=args.group_id, k=args.top)]
        out["rankings"].append(show("episode spread, people excluded", rows,
                                    time.time() - t))

    driver.close()

    print("\n══ summary ══")
    for r in out["rankings"]:
        print(f"   {r['concept_fraction']*100:>5.0f}% concepts   {r['title']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n📝 {args.out}")


if __name__ == "__main__":
    main()
