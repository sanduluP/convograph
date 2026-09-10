#!/usr/bin/env python3
"""
tkg_digest.py — distil a whole meeting out of the temporal KG, with cypher only.

THE PROBLEM
-----------
Module 3 plans a board from a WINDOW: 2 episodes, ~40 facts, a 9 KB prompt,
against a graph holding 82,165 facts. A window cannot tell the planner what the
meeting was about, so the board describes whichever five minutes it happened to
see. Feeding more windows does not fix it either — 500 near-duplicate facts bury
the model as surely as 19 do.

THE APPROACH
------------
Put a DETERMINISTIC step between the graph and the planner: a handful of cypher
queries that ask the graph what deserves to be on a board, and hand the planner
those answers instead of a slice of transcript.

Deterministic rather than an LLM summariser, for three reasons, and the third is
the one that matters for the paper:

  1. the same graph gives the same digest — no sampling temperature between the
     graph and the board;
  2. it is cheap — every query below runs in well under a second;
  3. it is the argument for the temporal KG existing at all. If the graph's
     structure cannot say what mattered in a meeting, the graph is not earning
     its keep against BM25.

THE QUERIES
-----------
  Q1  revision spine   what the meeting CHANGED ITS MIND about, old fact -> new
  Q2  topics           what it was ABOUT, by how many windows an idea spans
  Q3  participants     who was in the room                     (NOT from the graph)
  Q4  decisions        what was SETTLED, from Graphiti's typed edges
  Q5  open threads     what is still HANGING — asked for, never confirmed

WHY Q3 CANNOT COME FROM THIS GRAPH
----------------------------------
Measured on finance_speaker_free: FIVE speaker entities survive in the whole
graph, and 0.1% of facts touch one. That is not a defect — it is the point of
speaker-free ingestion, which is what took concept->concept facts from 6.6% to
68.9% and made a content map possible at all. But it means participation is not
a graph question here. It comes from module 1's diarization, or from the
transcript, and this module reads it from the episode text rather than pretending
the graph knows.

RUN IT AGAINST A MEETING, NOT A CORPUS
--------------------------------------
`finance_speaker_free` is 5,964 episodes spanning six weeks of several parallel
projects. A digest over all of it summarises a corpus and would read as
impressive nonsense. Use --episode-limit / --episode-start to take one
meeting-sized slice, or point --group-id at a graph that IS one meeting.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reused, not reimplemented: Graphiti emits a listing sentence as one fact per
# item AND the combined one, so any list of facts is ~2x redundant. Measured
# 2026-09-07 on the planner's input: 40 facts, 19 distinct ideas.
from analysis.fact_redundancy import dedupe as _dedupe_facts  # noqa: E402

# Edges that record something being SETTLED, as opposed to discussed. Taken from
# the relation names Graphiti actually emits on this corpus, not invented:
# CONFIRMS 3,934 · OWNS 1,226 · LOCKS 877 · STAMPS 807 · FREEZES 712 · VALIDATES 641.
DECISIVE = ["CONFIRMS", "LOCKS", "FREEZES", "OWNS", "STAMPS", "VALIDATES",
            "APPROVES", "ACCEPTS"]
# Edges that record something being ASKED FOR. An ask with no matching CONFIRMS
# is the "still open" signal.
ASKING = ["REQUIRES", "REQUESTS", "NEEDS", "REQUESTS_CONFIRMATION_FROM",
          "REQUIRES_CONFIRMATION", "REQUIRES_CONFIRMATION_FROM"]

# A speaker line is "Name: text" or "Name (Role): text" at the START of a line.
# The naive version of this matched any prose containing a colon — it reported
# "Please review the spec at http", "The spec is here" and "Two options on the
# table" as participants. Two constraints kill all of those:
#   * at most 3 words, since a name is not a sentence;
#   * the colon must not be a URL scheme (http:, https:), which is what caught
#     every line citing a SharePoint link.
SPEAKER_RE = re.compile(
    r"^(?!https?\b)"                    # not a URL scheme
    r"([A-Za-z][\w.'-]*(?:[ ][\w.'-]+){0,2})"   # 1-3 words
    r"\s*(?:\([^)]{0,60}\))?\s*:\s",  # optional (Role), then ": "
    re.M)


def _slice_clause(episode_limit: int, episode_start: int) -> str:
    """The episode set every query shares.

    One clause, used verbatim by all five, so "this meeting" means exactly the
    same episodes everywhere. Ordering is by the WINDOW NUMBER in the name, not
    by valid_at: many episodes in this corpus share a timestamp, and ordering by
    time interleaves window 5418 with window 3349.
    """
    skip = f"SKIP {episode_start} " if episode_start else ""
    take = f"LIMIT {episode_limit} " if episode_limit else ""
    return ("MATCH (ep:Episodic) WHERE ep.group_id=$g "
            "WITH ep, toInteger(split(ep.name,'_w')[-1]) AS wno ORDER BY wno "
            f"{skip}{take}")


# ── Q1 ───────────────────────────────────────────────────────────────────────
def q1_revisions(s, g, slice_clause, limit=25) -> list[dict]:
    """Facts the conversation later overturned, with the fact that replaced them.

    RANKING. A meeting can overturn hundreds of facts and a board has room for a
    handful, so the cap is a SELECTION and the ordering is the whole argument.
    Three signals, best first:

      1. EPISODE SPAN between the old fact and the new one. A decision revisited
         many windows later is a genuine change of mind; one corrected two
         minutes later is a typo. This is the strongest available signal and it
         is purely structural.
      2. ENDPOINT DEGREE — a revision about a hub entity touched more of the
         meeting than one about something mentioned once.
      3. Recency, as the tiebreak.

    Faris proposed degree as the primary signal; span beats it because degree
    ranks the same handful of hub entities to the top of every query, so Q1 and
    Q2 would return the same material. Degree still contributes as (2).

    The successor join is EXACT, not "the next fact afterwards": invalidation
    copies the new fact's valid_at into the old fact's invalid_at
    (graphiti_core edge_operations.py:569). Requiring a shared endpoint is what
    makes it usable — on the timestamp alone 95% of rows match several
    candidates, the widest 113.
    """
    return [dict(r) for r in s.run(slice_clause + """
        MATCH (ep)-[:MENTIONS]->(:Entity)-[r:RELATES_TO]->(:Entity)
        WHERE r.invalid_at IS NOT NULL AND r.fact IS NOT NULL
        WITH DISTINCT r
        MATCH (a:Entity)-[r]->(b:Entity)
        OPTIONAL CALL (r, a, b) {
            MATCH (x:Entity)-[s2:RELATES_TO]->(y:Entity)
            WHERE s2.valid_at = r.invalid_at AND s2.uuid <> r.uuid
              AND s2.group_id = r.group_id
              AND (x.uuid IN [a.uuid,b.uuid] OR y.uuid IN [a.uuid,b.uuid])
            RETURN s2.fact AS successor
            ORDER BY s2.created_at LIMIT 1
        }
        WITH r, a, b, successor,
             CASE WHEN r.valid_at IS NULL THEN -1
                  ELSE duration.between(r.valid_at, r.invalid_at).days END AS span_days,
             COUNT { (a)-[:RELATES_TO]-() } + COUNT { (b)-[:RELATES_TO]-() } AS degree
        RETURN a.name AS subject, b.name AS object,
               r.fact AS was, successor AS became,
               toString(r.valid_at) AS valid_from,
               toString(r.invalid_at) AS changed_at,
               span_days, degree
        ORDER BY span_days DESC, degree DESC, r.invalid_at DESC
        LIMIT $k
    """, g=g, k=limit)]


# ── Q2 ───────────────────────────────────────────────────────────────────────
def q2_topics(s, g, slice_clause, limit=8) -> list[dict]:
    """What the meeting was about, by EPISODE SPREAD.

    Spread, not raw mention count: an entity named in 40 different windows was a
    thread running through the meeting, while one named 40 times inside a single
    window was one person on a tangent. Same arithmetic, opposite meaning.

    Measured 2026-09-07: on the speaker-free graph this returns 100% concepts in
    0.26 s. The co-mention projection originally designed for this step is
    unnecessary — it was a fix for the speaker-rooted graph, where degree ranking
    returned the attendee list.

    Each topic carries its most-connected neighbours, which is what lets the
    planner say what the topic IS rather than just name it.
    """
    return [dict(r) for r in s.run(slice_clause + """
        MATCH (ep)-[:MENTIONS]->(e:Entity)
        WHERE NOT e.name =~ '(?i)(user|speaker|participant)[_ ]?[0-9]+'
          // Artifacts are not topics. A SharePoint URL ranked 4th among "what
          // this meeting was about" — it is the document the topic was argued
          // in. q2_artifacts returns those separately.
          AND NOT e.name =~ '(?i).*(https?://|sharepoint|_spec$).*'
        WITH e, count(DISTINCT ep.uuid) AS episodes
        ORDER BY episodes DESC LIMIT $k
        OPTIONAL CALL (e) {
            MATCH (e)-[r:RELATES_TO]-(n:Entity)
            WHERE n.uuid <> e.uuid
              AND NOT n.name =~ '(?i).*(https?://|sharepoint|_spec$).*'
              AND NOT n.name =~ '(?i)(user|speaker|participant)[_ ]?[0-9]+' 
            WITH n.name AS nb, count(*) AS w ORDER BY w DESC LIMIT 5
            RETURN collect(nb) AS neighbours
        }
        RETURN e.name AS topic, episodes, neighbours
    """, g=g, k=limit)]


def q2b_artifacts(s, g, slice_clause, limit=5) -> list[dict]:
    """The documents the meeting kept returning to.

    Split out of q2_topics rather than dropped: "the spec everyone argued about"
    is real board material, but it is not what the meeting was ABOUT, and mixing
    the two put a URL fourth in the topic list.
    """
    return [dict(r) for r in s.run(slice_clause + """
        MATCH (ep)-[:MENTIONS]->(e:Entity)
        WHERE e.name =~ '(?i).*(https?://|sharepoint|_spec$).*'
        RETURN e.name AS artifact, count(DISTINCT ep.uuid) AS episodes
        ORDER BY episodes DESC LIMIT $k
    """, g=g, k=limit)]


# ── Q3 ───────────────────────────────────────────────────────────────────────
def q3_participants(s, g, slice_clause, limit=12) -> list[dict]:
    """Who spoke, read from the EPISODE TEXT — not from the graph.

    On a speaker-free graph the speakers are deliberately not nodes: five survive
    out of 6,147 entities. So this parses the "Name: utterance" lines out of the
    raw episode content, which is the same information module 1's diarization
    provides directly and will replace this with.

    Counting UTTERANCES rather than facts, because a speaker-free graph attributes
    nothing: there is no "facts this person asserted" to count. Utterances is the
    honest measure of participation here, and it is not a proxy for influence.
    """
    rows = [dict(r) for r in s.run(slice_clause + "RETURN ep.content AS content", g=g)]
    counter: Counter = Counter()
    episodes_seen: dict[str, set] = {}
    for i, row in enumerate(rows):
        for who in SPEAKER_RE.findall(row.get("content") or ""):
            name = who.strip()
            if 1 < len(name) <= 40:
                counter[name] += 1
                episodes_seen.setdefault(name, set()).add(i)

    # A person speaks ACROSS the meeting; a false positive is one turn of phrase
    # in a couple of windows. Measured on an 80-window slice: real speakers span
    # 5 to 63 windows, the leftover prose matches span exactly 2. The gap is the
    # signal, so the bar is RELATIVE to the most active speaker rather than a
    # magic number — 5% of the top, floor of 2. An absolute threshold tuned on
    # this meeting would be wrong for a shorter one.
    if episodes_seen:
        top = max(len(v) for v in episodes_seen.values())
        floor = max(2, round(0.05 * top))
    else:
        floor = 2
    real = {n: c for n, c in counter.items() if len(episodes_seen[n]) >= floor}
    total = sum(real.values()) or 1
    return [{"speaker": n, "utterances": c,
             "episodes": len(episodes_seen[n]),
             "share": round(c / total, 3)}
            for n, c in Counter(real).most_common(limit)]


# ── Q4 ───────────────────────────────────────────────────────────────────────
def q4_decisions(s, g, slice_clause, limit=20) -> list[dict]:
    """What was settled — Graphiti's own typed edges, filtered to decisive verbs.

    Nearly free, because the typing already exists: the extractor emits
    CONFIRMS / LOCKS / FREEZES / OWNS / STAMPS / VALIDATES as relation names. No
    LLM, no keyword search over fact text.

    STILL-VALID ONLY (invalid_at IS NULL). A decision that was later overturned
    belongs to Q1's revision spine; showing it here as well would put the same
    material on the board twice, once as settled and once as changed.
    """
    return [dict(r) for r in s.run(slice_clause + """
        MATCH (ep)-[:MENTIONS]->(:Entity)-[r:RELATES_TO]->(:Entity)
        WHERE r.name IN $verbs AND r.invalid_at IS NULL AND r.fact IS NOT NULL
        WITH DISTINCT r
        MATCH (a:Entity)-[r]->(b:Entity)
        WITH a, b, r,
             COUNT { (a)-[:RELATES_TO]-() } + COUNT { (b)-[:RELATES_TO]-() } AS degree
        RETURN r.name AS verb, a.name AS subject, b.name AS object,
               r.fact AS fact, toString(r.valid_at) AS valid_from, degree
        ORDER BY degree DESC, r.valid_at DESC
        LIMIT $k
    """, g=g, verbs=DECISIVE, k=limit)]


# ── Q5 ───────────────────────────────────────────────────────────────────────
def q5_open_threads(s, g, slice_clause, limit=15) -> list[dict]:
    """Asked for, never confirmed, never overturned — the "still open" panel.

    A real graphic recording always has one and we have never produced one. It is
    also the only query here that surfaces something nobody said out loud: the
    absence of a confirmation is not in any single utterance.

    THE TEST IS DELIBERATELY STRICT. An ask counts as open only when the SAME
    subject-object pair has no decisive edge at all. Matching on the subject
    alone would silently close every ask involving a busy entity, which on this
    graph is most of them.

    Expect false positives regardless: CONFIRMS edges are sparser than REQUIRES
    ones (3,934 vs 5,225), so a confirmation the extractor missed reads here as
    an open thread. Treat the output as "worth checking", not "provably open" —
    which is also how a human graphic recorder uses that panel.
    """
    return [dict(r) for r in s.run(slice_clause + """
        MATCH (ep)-[:MENTIONS]->(:Entity)-[r:RELATES_TO]->(:Entity)
        WHERE r.name IN $asks AND r.invalid_at IS NULL AND r.fact IS NOT NULL
        WITH DISTINCT r
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE NOT EXISTS { (a)-[c:RELATES_TO]->(b) WHERE c.name IN $decisive }
          AND NOT EXISTS { (b)-[c2:RELATES_TO]->(a) WHERE c2.name IN $decisive }
        WITH a, b, r,
             COUNT { (a)-[:RELATES_TO]-() } + COUNT { (b)-[:RELATES_TO]-() } AS degree
        RETURN r.name AS verb, a.name AS subject, b.name AS object,
               r.fact AS fact, toString(r.valid_at) AS asked_at, degree
        ORDER BY degree DESC, r.valid_at DESC
        LIMIT $k
    """, g=g, asks=ASKING, decisive=DECISIVE, k=limit)]


# ── the digest ───────────────────────────────────────────────────────────────
def digest(uri: str, user: str, password: str, database: str, group_id: str,
           episode_limit: int = 0, episode_start: int = 0,
           caps: dict | None = None) -> dict:
    """Run all five queries and return one labelled result set per query.

    The sets stay SEPARATE. Merging them into one flat list would throw away
    exactly the distinction the five queries were run to compute — and they map
    almost one-to-one onto board regions: topics are the anchors, revisions are
    the arrows, decisions are badges, open threads are the corner panel.
    """
    from neo4j import GraphDatabase

    caps = {"revisions": 25, "topics": 8, "artifacts": 5, "participants": 12,
            "decisions": 20, "open_threads": 15, **(caps or {})}
    slice_clause = _slice_clause(episode_limit, episode_start)
    out: dict = {
        "group_id": group_id, "uri": uri,
        "episode_limit": episode_limit, "episode_start": episode_start,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "queries": {}, "timings": {},
    }

    drv = GraphDatabase.driver(uri, auth=(user, password))
    with drv.session(database=database) as s:
        n = list(s.run("MATCH (e:Episodic) WHERE e.group_id=$g RETURN count(*) AS n",
                       g=group_id))[0]["n"]
        if not n:
            drv.close()
            raise SystemExit(f"❌ group '{group_id}' has no episodes at {uri}")
        out["episodes_in_group"] = n
        out["episodes_digested"] = min(n - episode_start, episode_limit or n)

        for key, fn, cap in (
            ("revisions", q1_revisions, caps["revisions"]),
            ("topics", q2_topics, caps["topics"]),
            ("artifacts", q2b_artifacts, caps["artifacts"]),
            ("participants", q3_participants, caps["participants"]),
            ("decisions", q4_decisions, caps["decisions"]),
            ("open_threads", q5_open_threads, caps["open_threads"]),
        ):
            t0 = time.time()
            # Over-fetch, then collapse restatements, so a cap of 15 means 15
            # DISTINCT items rather than 15 rows that say five things. Q2 and Q3
            # are already one-row-per-thing and are left alone.
            over = cap * 3 if key in ("revisions", "decisions", "open_threads") else cap
            rows = fn(s, group_id, slice_clause, over)
            if key in ("revisions", "decisions", "open_threads") and rows:
                keyed = [{**r, "fact": r.get("fact") or r.get("was") or ""} for r in rows]
                kept, removed = _dedupe_facts(keyed)
                seen = {id(k) for k in kept}
                rows = [r for r, k in zip(rows, keyed) if id(k) in seen][:cap]
                out.setdefault("deduped", {})[key] = removed
            out["queries"][key] = rows[:cap]
            out["timings"][key] = round(time.time() - t0, 3)
    drv.close()
    return out


README_FIELDS = {
    "revisions": "Q1 — a fact the meeting overturned, plus the fact that replaced it. "
                 "was/became are the before and after; span_days is how long the old "
                 "fact stood (the ranking signal); degree is the endpoints' connectivity. "
                 "became may be null: 85.3% of successors are recoverable, not all.",
    "artifacts": "Q2b — the documents the meeting kept returning to. Separate from "
                 "topics on purpose: a spec is where a topic was argued, not the topic.",
    "topics": "Q2 — what the meeting was about. `episodes` counts DISTINCT windows "
              "mentioning the entity, so it measures a thread through the meeting "
              "rather than one loud moment. `neighbours` are its most-connected "
              "entities, which say what the topic IS.",
    "participants": "Q3 — who spoke, parsed from episode TEXT, not from the graph: on a "
                    "speaker-free graph speakers are deliberately not nodes. "
                    "`utterances` is participation, and is NOT a proxy for influence. "
                    "Module 1's diarization replaces this.",
    "decisions": "Q4 — what was settled. Graphiti's own typed edges filtered to "
                 "decisive verbs, still-valid only; anything later overturned is in "
                 "revisions instead, so the two never show the same material.",
    "open_threads": "Q5 — asked for, never confirmed, never overturned. Strict: the same "
                    "subject-object pair must carry no decisive edge. Expect false "
                    "positives — CONFIRMS edges are sparser than REQUIRES ones, so a "
                    "missed confirmation reads as an open thread. 'Worth checking', not "
                    "'provably open'.",
}


def write_digest(result: dict, out_dir: str) -> None:
    """One JSONL per query, a readme documenting the fields, and a digest.md.

    JSONL because these are lists of records: head, grep and wc -l work on them,
    and a single record is readable without a viewer. The readme is written in
    the same step that writes the data, never later.
    """
    os.makedirs(out_dir, exist_ok=True)
    for key, rows in result["queries"].items():
        with open(os.path.join(out_dir, f"{key}.jsonl"), "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
        with open(os.path.join(out_dir, f"{key}.readme.jsonl"), "w") as fh:
            fh.write(json.dumps({
                "file": f"{key}.jsonl",
                "what": README_FIELDS[key],
                "rows": len(rows),
                "seconds": result["timings"][key],
                "group_id": result["group_id"],
                "episodes_digested": result["episodes_digested"],
            }, indent=None) + "\n")

    with open(os.path.join(out_dir, "digest.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    with open(os.path.join(out_dir, "digest.md"), "w") as fh:
        fh.write(render_markdown(result))


def render_markdown(result: dict) -> str:
    """The human-readable form — and the exact material the planner will get.

    Kept as ONE function so what a person reads and what the model reads cannot
    drift apart.
    """
    q = result["queries"]
    L = [f"# Meeting digest — `{result['group_id']}`", "",
         f"{result['episodes_digested']} of {result['episodes_in_group']} episodes"
         + (f", starting at {result['episode_start']}" if result["episode_start"] else "")
         + f" · generated {result['generated_at']}", ""]

    L += ["## What this meeting was about", ""]
    for t in q["topics"]:
        nb = ", ".join((t.get("neighbours") or [])[:4])
        L.append(f"- **{t['topic']}** — in {t['episodes']} windows"
                 + (f" · with {nb}" if nb else ""))

    if q.get("artifacts"):
        L += ["", "## The documents it kept coming back to", ""]
        for a in q["artifacts"]:
            L.append(f"- {a['artifact']} — in {a['episodes']} windows")

    L += ["", "## What changed", ""]
    if not q["revisions"]:
        L.append("_Nothing was overturned in this slice._")
    for r in q["revisions"]:
        span = r.get("span_days")
        age = f" ({span}d)" if isinstance(span, int) and span >= 0 else ""
        L.append(f"- **{r['subject']} → {r['object']}**{age}")
        L.append(f"  - was: {r['was']}")
        L.append(f"  - became: {r['became'] or '_(successor not identified)_'}")

    L += ["", "## What was settled", ""]
    for d in q["decisions"]:
        L.append(f"- `{d['verb']}` {d['fact']}")

    L += ["", "## Still open", ""]
    if not q["open_threads"]:
        L.append("_Nothing outstanding found._")
    for o in q["open_threads"]:
        L.append(f"- `{o['verb']}` {o['fact']}")

    L += ["", "## Who was in the room", ""]
    for p in q["participants"]:
        L.append(f"- {p['speaker']} — {p['utterances']} utterances across "
                 f"{p['episodes']} windows ({100*p['share']:.0f}%)")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group-id", required=True)
    ap.add_argument("--uri", default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--database", default=None)
    ap.add_argument("--episode-limit", type=int, default=0,
                    help="digest only the first N episodes — use this to take ONE "
                         "meeting-sized slice out of a corpus (0 = everything)")
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--out", default=None, help="directory for the digest files")
    ap.add_argument("--env-file", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    args = ap.parse_args()

    import ui_ingest
    ui_ingest._load_env_file(args.env_file)

    uri = args.uri or os.environ["NEO4J_URI"]
    db = args.database or ("neo4j" if "localhost" in uri or "127.0.0.1" in uri
                           else os.getenv("NEO4J_DATABASE", "neo4j"))
    result = digest(uri, args.user or os.environ["NEO4J_USER"],
                    args.password or os.environ["NEO4J_PASSWORD"], db,
                    args.group_id, args.episode_limit, args.episode_start)

    print(render_markdown(result))
    total = sum(result["timings"].values())
    print("─" * 70)
    for k, v in result["timings"].items():
        print(f"   {k:<14} {v:>6.2f}s  ({len(result['queries'][k])} rows)")
    print(f"   {'TOTAL':<14} {total:>6.2f}s")

    if args.out:
        write_digest(result, args.out)
        print(f"\n📝 {args.out}/  (digest.md, digest.json, one jsonl per query)")


if __name__ == "__main__":
    main()
