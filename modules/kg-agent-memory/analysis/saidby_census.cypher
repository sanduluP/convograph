// ═══════════════════════════════════════════════════════════════════════════════
//  saidby_census.cypher — does demoting the speaker out of subject position
//  change the SHAPE of the extracted graph?
//
//  WHY THIS WORKS AS A CONTROLLED A/B
//  ----------------------------------
//  A shard store happens to hold BOTH runs, over the SAME window range:
//     group_id = 'gmb_finance_full'    → episodes rendered "User_5: <text>"
//     group_id = 'gmb_finance_saidby'  → episodes rendered "[said by User_5] <text>"
//  Same messages, same model, same prompts, same code — only the framing differs.
//  So any difference below is attributable to the framing alone.
//
//  THE TWO NUMBERS THAT DECIDE IT
//    person-rooted %   : baseline 94.1 %  → we want this DOWN
//    concept→concept % : baseline  0.4 %  → we want this UP
//  A person here is a `User_N` node; anything else counts as a concept.
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. size of each run (sanity: are both actually present?) ──────────────────
MATCH (e:Episodic)
RETURN e.group_id AS run, count(*) AS episodes
ORDER BY run;

// ── 2. entities per run ───────────────────────────────────────────────────────
MATCH (n:Entity)
RETURN n.group_id AS run,
       count(*) AS entities,
       sum(CASE WHEN n.name =~ '^User_\\d+$' THEN 1 ELSE 0 END) AS person_nodes
ORDER BY run;

// ── 3. THE HEADLINE: what does a fact connect? ────────────────────────────────
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
WITH r.group_id AS run,
     (s.name =~ '^User_\\d+$') AS src_person,
     (t.name =~ '^User_\\d+$') AS tgt_person
WITH run,
     count(*)                                                   AS facts,
     sum(CASE WHEN src_person THEN 1 ELSE 0 END)                AS person_rooted,
     sum(CASE WHEN NOT src_person AND NOT tgt_person THEN 1 ELSE 0 END) AS concept_concept
RETURN run,
       facts,
       person_rooted,
       round(100.0 * person_rooted   / facts, 1) AS pct_person_rooted,
       concept_concept,
       round(100.0 * concept_concept / facts, 1) AS pct_concept_concept
ORDER BY run;

// ── 4. relation-vocabulary reuse (the other defect we found) ─────────────────
MATCH ()-[r:RELATES_TO]->()
WITH r.group_id AS run, r.name AS rel, count(*) AS uses
WITH run, count(*) AS distinct_rel_types, sum(uses) AS total_facts,
     sum(CASE WHEN uses = 1 THEN 1 ELSE 0 END) AS used_once
RETURN run, total_facts, distinct_rel_types, used_once,
       round(100.0 * used_once / distinct_rel_types, 1) AS pct_types_used_once
ORDER BY run;

// ── 5. the biggest hubs — are the users still the centre of the graph? ───────
MATCH (n:Entity)-[r:RELATES_TO]-()
WITH n.group_id AS run, n.name AS name, count(r) AS degree
ORDER BY run, degree DESC
WITH run, collect({name: name, degree: degree})[0..5] AS top5
RETURN run, top5
ORDER BY run;
