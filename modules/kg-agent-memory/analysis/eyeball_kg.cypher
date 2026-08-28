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
//                 That invalid_at field is the whole bi-temporal story.
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
// 3. THE BI-TEMPORAL QUERY — facts that were later SUPERSEDED.
//    These are the ~18k facts BM25 cannot represent at all: the graph knows
//    they USED to be true and knows when they stopped being true.
// ─────────────────────────────────────────────────────────────────────────────
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.invalid_at IS NOT NULL
RETURN a.name AS subject, r.fact AS fact, b.name AS object,
       r.valid_at AS became_true, r.invalid_at AS superseded_at
ORDER BY r.invalid_at DESC
LIMIT 25;


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
