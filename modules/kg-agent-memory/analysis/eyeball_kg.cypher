// ============================================================================
//  eyeball_kg.cypher — starter queries for LOOKING AT the merged Finance KG.
//
//  Paste these one at a time into Neo4j Browser (http://localhost:7475).
//  They are ordered from "is anything there at all" to "is the extraction any
//  good", which is the order you actually want to ask them in.
//
//  THE SCHEMA IN ONE PARAGRAPH
//  ---------------------------
//    (:Episodic)  one ingested WINDOW of 5 chat messages. `name` is
//                 "<group_id>_w<N>", `content` is the raw text, `valid_at` is
//                 the timestamp of the window.
//    (:Entity)    a thing the LLM pulled out (a person, a system, a policy).
//                 `name` + an LLM-written `summary`.
//    MENTIONS     (:Episodic)->(:Entity)  — this window talked about this thing.
//    RELATES_TO   (:Entity)->(:Entity)    — an extracted FACT. The English
//                 sentence lives on `fact`. `valid_at` = when it became true,
//                 `invalid_at` = when it was SUPERSEDED (null = still current).
//                 That invalid_at field is the whole bi-temporal story — and it
//                 is also the JOIN KEY back to the fact that replaced it, since
//                 Graphiti sets old.invalid_at = new.valid_at. See query 3.
// ============================================================================


// ─────────────────────────────────────────────────────────────────────────────
// 1. What is in here at all?  (sanity check — run this first)
// ─────────────────────────────────────────────────────────────────────────────
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC;

MATCH ()-[r]->() RETURN type(r) AS relationship, count(*) AS count ORDER BY count DESC;


// ─────────────────────────────────────────────────────────────────────────────
// 2. THE MONEY QUERY — read 25 extracted facts and judge them yourself.
//    This is the "is the extraction any good?" question, answered by reading.
// ─────────────────────────────────────────────────────────────────────────────
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
RETURN a.name AS subject, r.fact AS fact, b.name AS object, r.valid_at AS since
ORDER BY rand()
LIMIT 25;


// ─────────────────────────────────────────────────────────────────────────────
// 3. THE BI-TEMPORAL QUERY — facts that were later SUPERSEDED,
//    AND the fact that replaced each one.
//
//    These are the ~18k facts BM25 cannot represent at all: the graph knows
//    they USED to be true, when they stopped, and — with the join below — what
//    replaced them.
//
//    HOW THE SUCCESSOR IS FOUND, since Graphiti stores no pointer to it.
//    An EntityEdge has valid_at / invalid_at / expired_at and nothing naming its
//    replacement. But invalidation does, literally
//    (graphiti_core/utils/maintenance/edge_operations.py:569):
//
//        edge.invalid_at = resolved_edge.valid_at
//
//    The NEW fact's valid_at is copied into the OLD fact's invalid_at. So the
//    successor is joined on EXACT timestamp equality — not on "the next fact
//    afterwards", which would be a guess.
//
//    The shared-endpoint requirement is NOT optional. On the timestamp alone,
//    95% of rows match SEVERAL candidates and the widest matches 113, because
//    many edges carry the same valid_at. Requiring the successor to touch one of
//    the same two entities is what makes the answer meaningful.
//
//    MEASURED on the whole of gmb_finance_full (query 3c, 2026-09-07):
//        18,450 superseded facts
//        15,737 with a recoverable successor  =  85.3%
//         2,713 unknown                       ->  superseded_by IS NULL
//    Of a 300-row sample, 52% resolved to the same entity pair and 39% to a
//    fact sharing one entity.
//
//    Nulls are returned rather than guessed. A candidate sharing no entity is a
//    coincidental timestamp collision, and calling it a successor would put a
//    false revision history in front of a reader.
//
//    Needs Neo4j 5.23+ for OPTIONAL CALL. Re-check the numbers any time with
//        bash scripts/run_check_supersession.sh --group-id <group>
// ─────────────────────────────────────────────────────────────────────────────
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.invalid_at IS NOT NULL
OPTIONAL CALL (r, a, b) {
    MATCH (x:Entity)-[s:RELATES_TO]->(y:Entity)
    WHERE s.valid_at = r.invalid_at            // the exact copy described above
      AND s.uuid <> r.uuid                     // never itself
      AND s.group_id = r.group_id              // never across experiments
      AND (x.uuid IN [a.uuid, b.uuid]          // must touch one of the same
           OR y.uuid IN [a.uuid, b.uuid])      // two entities
    RETURN s,
           CASE WHEN x.uuid = a.uuid AND y.uuid = b.uuid THEN 'same-pair'
                WHEN x.uuid = b.uuid AND y.uuid = a.uuid THEN 'same-pair-reversed'
                ELSE 'shares-entity' END AS confidence
    ORDER BY confidence, s.created_at          // strongest match wins, then oldest
    LIMIT 1
}
RETURN a.name           AS subject,
       r.fact           AS fact_that_was_replaced,
       b.name           AS object,
       r.valid_at       AS became_true,
       r.invalid_at     AS superseded_at,
       s.fact           AS superseded_by,      // null = no successor identified
       confidence       AS successor_confidence
ORDER BY r.invalid_at DESC
LIMIT 25;


// 3b. Only the rows where we KNOW what replaced the fact — the clean
//     before/after pairs. This is the shape to put in a paper or a demo.
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.invalid_at IS NOT NULL
MATCH (x:Entity)-[s:RELATES_TO]->(y:Entity)
WHERE s.valid_at = r.invalid_at
  AND s.uuid <> r.uuid
  AND s.group_id = r.group_id
  AND x.uuid = a.uuid AND y.uuid = b.uuid      // same pair only: the strongest tier
RETURN a.name AS subject, b.name AS object,
       r.fact AS before, s.fact AS after,
       r.valid_at AS before_from, r.invalid_at AS changed_at
ORDER BY r.invalid_at DESC
LIMIT 25;


// 3c. How much revision history is actually recoverable, as one row.
//     Run this before quoting any number from 3 or 3b.
MATCH ()-[r:RELATES_TO]->()
WHERE r.invalid_at IS NOT NULL
WITH count(*) AS superseded
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.invalid_at IS NOT NULL
OPTIONAL CALL (r, a, b) {
    MATCH (x:Entity)-[s:RELATES_TO]->(y:Entity)
    WHERE s.valid_at = r.invalid_at AND s.uuid <> r.uuid AND s.group_id = r.group_id
      AND (x.uuid IN [a.uuid, b.uuid] OR y.uuid IN [a.uuid, b.uuid])
    RETURN s LIMIT 1
}
RETURN superseded,
       count(s)                                   AS with_known_successor,
       superseded - count(s)                      AS successor_unknown,
       round(100.0 * count(s) / superseded, 1)    AS pct_recoverable;


// ─────────────────────────────────────────────────────────────────────────────
// 3d. ONE MEETING — the phase graph on AuraDB.
//
//     Every query above scans whatever database you are connected to. AuraDB
//     holds SEVERAL graphs side by side, so on it you almost always want a
//     group filter:
//
//       treasury_prod_deploy_speaker_free   ONE meeting (76 episodes) — this one
//       gmb_finance_full                    6 weeks, speakers as nodes
//       ui_* / web_*                        scratch runs from the UI
//
//     FILTER ON THE EDGE, NOT ALL THREE. Both work — Entity nodes do carry
//     group_id — but Graphiti creates a SEPARATE node per group, so an edge in
//     this group can only ever join nodes in this group. Checking a.group_id and
//     b.group_id as well returns the identical 1,394 facts and just reads
//     heavier. (Verified 2026-09-11: edge-only 1394, all-three 1394.)
// ─────────────────────────────────────────────────────────────────────────────
:param g => 'treasury_prod_deploy_speaker_free';

// 3d-i. Is it there, and how big?
MATCH (e:Episodic) WHERE e.group_id = $g
WITH count(*) AS episodes
MATCH ()-[r:RELATES_TO]->() WHERE r.group_id = $g
RETURN episodes, count(r) AS facts,
       sum(CASE WHEN r.invalid_at IS NOT NULL THEN 1 ELSE 0 END) AS superseded;

// 3d-ii. 25 random facts — judge the extraction yourself.
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.group_id = $g
RETURN a.name AS subject, r.fact AS fact, b.name AS object, r.valid_at AS since
ORDER BY rand()
LIMIT 25;

// 3d-iii. What this meeting was ABOUT — the same ranking the digest's Q2 uses.
//     Episode SPREAD, not mention count: an entity named in 40 different windows
//     was a thread through the meeting; one named 40 times in a single window
//     was one person on a tangent.
MATCH (ep:Episodic)-[:MENTIONS]->(e:Entity)
WHERE ep.group_id = $g
RETURN e.name AS topic, count(DISTINCT ep.uuid) AS windows
ORDER BY windows DESC
LIMIT 15;

// 3d-iv. What CHANGED, with the fact that replaced it.
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.group_id = $g AND r.invalid_at IS NOT NULL
OPTIONAL CALL (r, a, b) {
    MATCH (x:Entity)-[s:RELATES_TO]->(y:Entity)
    WHERE s.valid_at = r.invalid_at AND s.uuid <> r.uuid AND s.group_id = r.group_id
      AND (x.uuid IN [a.uuid, b.uuid] OR y.uuid IN [a.uuid, b.uuid])
    RETURN s.fact AS successor ORDER BY s.created_at LIMIT 1
}
RETURN r.fact AS was, successor AS became, r.invalid_at AS changed_at
ORDER BY r.invalid_at DESC
LIMIT 25;

// 3d-v. The graph as a PICTURE. Neo4j Browser renders this one.
MATCH path = (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.group_id = $g
RETURN path LIMIT 200;


// ─────────────────────────────────────────────────────────────────────────────
// 4. A PICTURE — the neighbourhood of the busiest entity.
//    Neo4j Browser renders this as an actual graph, which is the closest thing
//    we currently have to "graphic recording".
// ─────────────────────────────────────────────────────────────────────────────
MATCH (e:Entity)-[:RELATES_TO]-()
WITH e, count(*) AS degree ORDER BY degree DESC LIMIT 1
MATCH path = (e)-[:RELATES_TO*1..2]-(other:Entity)
RETURN path LIMIT 150;

// …or pick a topic by name instead of by degree:
MATCH path = (e:Entity)-[:RELATES_TO*1..2]-(:Entity)
WHERE toLower(e.name) CONTAINS 'settlement'
RETURN path LIMIT 150;


// ─────────────────────────────────────────────────────────────────────────────
// 5. QUALITY SMELLS — where extraction typically goes wrong.
// ─────────────────────────────────────────────────────────────────────────────

// 5a. Duplicate-ish entities: the same concept extracted under near-identical
//     names. Lots of these = the entity-resolution step is under-merging.
MATCH (e:Entity)
WITH toLower(e.name) AS norm, collect(e.name) AS variants, count(*) AS n
WHERE n > 1
RETURN norm, variants, n ORDER BY n DESC LIMIT 30;

// 5b. Over-generic entities: "the team", "the process". High degree + vague
//     name = a hub node that pollutes every retrieval.
MATCH (e:Entity)-[:RELATES_TO]-()
RETURN e.name AS entity, count(*) AS degree
ORDER BY degree DESC LIMIT 30;

// 5c. Orphan entities: extracted but never connected to anything. Pure noise.
MATCH (e:Entity) WHERE NOT (e)-[:RELATES_TO]-()
RETURN count(*) AS orphan_entities;

// 5d. Facts that are just a restatement of the entity names (low information).
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE size(r.fact) < 40
RETURN a.name, r.fact, b.name LIMIT 25;


// ─────────────────────────────────────────────────────────────────────────────
// 6. TRACE ONE FACT BACK TO THE RAW CHAT that produced it.
//    Answers "did the model actually read this, or did it hallucinate?"
// ─────────────────────────────────────────────────────────────────────────────
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.fact CONTAINS 'approval'          // ← put any phrase here
WITH r, a, b LIMIT 1
MATCH (ep:Episodic)-[:MENTIONS]->(a)
RETURN r.fact AS extracted_fact, ep.name AS window, ep.valid_at AS at,
       ep.content AS raw_messages
LIMIT 5;


// ─────────────────────────────────────────────────────────────────────────────
// 7. CROSS-CHANNEL CONTAMINATION — the diagnosis from 2026-08-05, visible.
//    An entity that is MENTIONED by windows from several different channels is
//    exactly the node that lets a search about project A return project B.
// ─────────────────────────────────────────────────────────────────────────────
MATCH (ep:Episodic)-[:MENTIONS]->(e:Entity)
WITH e, count(DISTINCT ep.group_id) AS groups, count(*) AS mentions
WHERE mentions > 20
RETURN e.name AS entity, mentions ORDER BY mentions DESC LIMIT 30;
