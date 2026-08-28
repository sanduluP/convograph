#!/usr/bin/env python3
"""Measure what the extracted ENTITIES actually are — the structure audit.

WHY
---
Reading the 2026-08-05 answer audit by hand suggested the fact SENTENCES are good
while the ENTITIES they hang on are sentence fragments (`User_5 -> user`,
`User_7 -> recovery states`, `-> completeness`, `-> drift`). That claim was based
on about a dozen rows, which is not enough to act on — and "the entities look bad
to me" is not something to put in front of a supervisor.

So this counts, over all 111k facts and 5.8k entities:

  1. WHAT KIND of thing each entity name is (person / role / URL / document /
     capitalised proper noun / lowercase phrase).
  2. THE SHAPE OF THE GRAPH: how many facts are person→person, person→concept,
     concept→concept. This is the number that matters. A knowledge graph whose
     edges nearly all run *from a person to a fragment* is not a knowledge graph
     of the domain — it is a star graph around the speakers, and it cannot
     support multi-hop reasoning between the things being discussed.
  3. HUBS and DUPLICATES: generic nodes everything attaches to, and the same
     concept extracted under several near-identical names.

Everything here is descriptive. No thresholds are tuned, and the classifier is a
handful of transparent regexes so any number can be checked by reading the
examples it prints alongside.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neo4j import GraphDatabase  # noqa: E402

BOLT_URI = os.environ.get("KG_BOLT_URI", "bolt://localhost:7688")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "graphiti123")

# Role words that mark a name as a PERSON even when it is not `User_N` — taken
# from the corpus's own `speaker_role` vocabulary, not invented here.
_ROLE_WORDS = {
    "lead", "analyst", "manager", "owner", "engineer", "officer", "director",
    "architect", "specialist", "coordinator", "consultant", "auditor", "head",
    "ops", "legal", "security", "compliance", "risk", "it", "bi", "cab",
}
_USER_RE = re.compile(r"^user_\d+$", re.I)
_URL_RE = re.compile(r"^https?://", re.I)
# A "document" is one of the corpus's SharePoint spec pages.
_DOC_RE = re.compile(r"(_spec|sharepoint)", re.I)


def classify(name: str) -> str:
    """Bucket one entity name. Deliberately coarse and readable."""
    n = (name or "").strip()
    if not n:
        return "empty"
    if _URL_RE.match(n) or _DOC_RE.search(n):
        return "url/document"
    if _USER_RE.match(n):
        return "person (User_N)"
    words = n.split()
    if len(words) <= 3 and any(w.lower().strip(":,.") in _ROLE_WORDS for w in words):
        return "person (role)"
    # A capitalised multi-word name is the shape a real named thing takes
    # ("UX Prototype Approval", "Treasury Management System").
    if n[0].isupper() and sum(1 for w in words if w[:1].isupper()) >= max(1, len(words) // 2):
        return "proper noun / named thing"
    if len(words) == 1:
        return "lowercase single word"
    return "lowercase phrase"


# The two buckets that represent an actual domain concept rather than a speaker.
_CONCEPT_KINDS = {"proper noun / named thing"}
_PERSON_KINDS = {"person (User_N)", "person (role)"}


def norm(name: str) -> str:
    """Casefold + strip punctuation — for near-duplicate detection only."""
    return re.sub(r"[^a-z0-9 ]", "", (name or "").lower()).strip()


def main() -> int:
    print(f"🔗 connecting to {BOLT_URI}", flush=True)
    driver = GraphDatabase.driver(BOLT_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as s:
        entities = [r["name"] for r in s.run("MATCH (e:Entity) RETURN e.name AS name")]
        edges: List[Tuple[str, str]] = [
            (r["a"], r["b"]) for r in s.run(
                "MATCH (a:Entity)-[:RELATES_TO]->(b:Entity) "
                "RETURN a.name AS a, b.name AS b")
        ]
    driver.close()
    print(f"   {len(entities)} entities, {len(edges)} facts\n", flush=True)

    # ---------------------------------------------------------------- 1. kinds
    kinds = Counter(classify(e) for e in entities)
    examples: Dict[str, List[str]] = defaultdict(list)
    for e in entities:
        k = classify(e)
        if len(examples[k]) < 4:
            examples[k].append(e)

    print("═" * 74)
    print("  1. WHAT KIND OF THING IS EACH ENTITY?")
    print("═" * 74)
    for kind, n in kinds.most_common():
        print(f"  {kind:28} {n:6}  {n/len(entities):6.1%}   "
              f"e.g. {', '.join(repr(x) for x in examples[kind][:3])}")

    concepts = sum(n for k, n in kinds.items() if k in _CONCEPT_KINDS)
    fragments = sum(n for k, n in kinds.items()
                    if k in {"lowercase phrase", "lowercase single word"})
    print(f"\n  → named things : {concepts:6} ({concepts/len(entities):.1%})")
    print(f"  → fragment-like: {fragments:6} ({fragments/len(entities):.1%})")

    # ------------------------------------------------------- 2. shape of graph
    def side(name: str) -> str:
        k = classify(name)
        if k in _PERSON_KINDS:
            return "person"
        if k in _CONCEPT_KINDS:
            return "concept"
        if k == "url/document":
            return "document"
        return "fragment"

    shape = Counter(f"{side(a)} → {side(b)}" for a, b in edges)
    print("\n" + "═" * 74)
    print("  2. SHAPE OF THE GRAPH — what does a fact actually connect?")
    print("═" * 74)
    for pat, n in shape.most_common(10):
        print(f"  {pat:26} {n:7}  {n/len(edges):6.1%}")

    concept_to_concept = sum(n for p, n in shape.items() if p == "concept → concept")
    from_person = sum(n for p, n in shape.items() if p.startswith("person →"))
    print(f"\n  → facts linking two domain CONCEPTS : {concept_to_concept:7} "
          f"({concept_to_concept/len(edges):.1%})")
    print(f"  → facts originating at a PERSON     : {from_person:7} "
          f"({from_person/len(edges):.1%})")
    print("    (a graph that is mostly person→something is a star around the")
    print("     speakers: it cannot support multi-hop reasoning between topics)")

    # ------------------------------------------------------ 3. hubs, duplicates
    degree = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    print("\n" + "═" * 74)
    print("  3. HUB NODES — everything attaches to these")
    print("═" * 74)
    for name, d in degree.most_common(12):
        print(f"  {d:6}  {classify(name):28} {name[:40]!r}")

    groups: Dict[str, List[str]] = defaultdict(list)
    for e in entities:
        groups[norm(e)].append(e)
    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    print("\n" + "═" * 74)
    print("  4. NEAR-DUPLICATE ENTITIES (same name after casefold/punctuation)")
    print("═" * 74)
    print(f"  {len(dupes)} groups covering {sum(len(v) for v in dupes.values())} entities")
    for k, v in sorted(dupes.items(), key=lambda kv: -len(kv[1]))[:8]:
        print(f"    {len(v)}× {v[:4]}")
    print("═" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
