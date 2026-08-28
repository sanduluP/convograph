"""EXPERIMENT 3a — override Graphiti's edge-contradiction prompt for MEETINGS.

WHY THIS FILE EXISTS
--------------------
Graphiti decides whether a NEW fact invalidates an OLD one through ONE LLM prompt:
`graphiti_core/prompts/dedupe_edges.py::resolve_edge`. That stock prompt is a
GENERIC *deduplication* prompt. Its own worked example literally teaches the model:

    EXISTING: "Bob ran 5 miles on Tuesday"
    NEW:      "Bob ran 3 miles on Wednesday"
    => contradicted_facts=[]   (different events on different days — no contradiction)

That rule is fine for encyclopedic text, but it is exactly WRONG for multi-party
human MEETINGS, where people revise earlier decisions IMPLICITLY — they almost
never say "I retract X". A deadline moves ("end of June" -> "mid-July"), a task is
reassigned (Islam -> Priyabanta), a chosen corpus is swapped (AMI -> ICSI), an
agreed step is dropped (drop the clustering step). Our windowing experiment proved
Graphiti now EXTRACTS both sides of these revisions, yet still invalidates only the
one EXPLICIT retraction — because this prompt tells it to ignore the rest.

WHAT WE CHANGE
--------------
We rewrite `resolve_edge` so the model flags an older fact as `contradicted` when a
newer fact updates a MUTABLE attribute of the SAME underlying commitment
(deadline / owner / chosen option / agreed step / decision). We keep a genuine
NON-contradiction example so we don't start over-flagging independent events.

WHAT WE DO NOT CHANGE (on purpose — keep the mechanics identical)
----------------------------------------------------------------
- The response schema stays `EdgeDuplicate` (duplicate_facts + contradicted_facts).
- The idx-numbering constraints across the two lists stay verbatim.
- The three context blocks (existing_edges / edge_invalidation_candidates /
  new_edge) stay verbatim.
- This is schema-free: no ontology, no custom edge types — Rahul's constraint.

HOW IT IS APPLIED (monkeypatch, no fork, no venv surgery)
---------------------------------------------------------
`graphiti_core.prompts.lib.VersionWrapper.__call__` calls `self.func(context)`, so
replacing `.func` on the shared `prompt_library` singleton swaps ONLY the prompt
text while preserving Graphiti's unicode-escape post-processing. Call
`apply_overrides()` once, BEFORE building the Graphiti client.
"""

from typing import Any

# The shared singleton every call site imports (edge_operations.py line 36). Patch
# THIS object's `.func` and the change is live everywhere Graphiti resolves edges.
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.models import Message


def resolve_edge_meetings(context: dict[str, Any]) -> list[Message]:
    """Meeting-aware replacement for dedupe_edges.resolve_edge.

    Same inputs (context dict), same output contract (list[Message] that make the
    LLM emit EdgeDuplicate JSON: duplicate_facts + contradicted_facts). Only the
    guidance and examples differ — recalibrated for implicit decision revisions.
    """
    return [
        # ---- SYSTEM MESSAGE -------------------------------------------------
        # Reframes the model's job from pure dedup to *reconciliation* of an
        # evolving meeting record: still never MERGE facts with key differences,
        # but DO mark a superseded earlier fact as contradicted.
        Message(
            role='system',
            content=(
                'You are a knowledge-graph fact reconciliation assistant for '
                'MULTI-PARTY MEETING transcripts. In meetings, people revise earlier '
                'decisions IMPLICITLY — they rarely say "I retract X"; they simply '
                'state the new deadline, the new owner, or the new plan. Your job: '
                '(a) detect exact duplicates, and (b) detect when a NEW FACT '
                'SUPERSEDES an earlier fact about the SAME underlying commitment. '
                'Never mark facts with key differences as DUPLICATES — but DO mark a '
                'superseded earlier fact as CONTRADICTED.'
            ),
        ),
        # ---- USER MESSAGE ---------------------------------------------------
        Message(
            role='user',
            content=f"""
Never mark facts as DUPLICATES if they have key differences (numeric values, dates,
owners, or key qualifiers). Duplicates must be the SAME fact; a changed value is a
CONTRADICTION (supersession), not a duplicate.

IMPORTANT constraints (unchanged — these govern the idx bookkeeping):
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
{context['existing_edges']}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{context['edge_invalidation_candidates']}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{context['new_edge']}
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT states IDENTICAL factual information as any fact in EXISTING FACTS,
     return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION / SUPERSESSION DETECTION (RECALL — catch implicit revisions):
   Mark an existing fact as contradicted ONLY when the NEW FACT asserts a
   DIFFERENT, MUTUALLY-EXCLUSIVE VALUE for the SAME slot of the SAME underlying
   thing (task, deliverable, decision, assignment, or plan) — such that the old and
   new values CANNOT both be true at the same time. This holds even when phrased
   implicitly, with NO explicit "retract" / "cancel" / "instead" wording. Typical
   meeting revisions to catch:
     - DEADLINE / DATE / TIME changed for the SAME deliverable
       ("submit by end of June" -> "submit by mid-July").
     - OWNER / ASSIGNEE changed for the SAME task — responsibility MOVES to a
       DIFFERENT person ("Islam will compute the metrics" -> "Priyabanta will
       compute the metrics").
     - CHOSEN OPTION / TOOL / DATASET / APPROACH changed to a DIFFERENT one for the
       SAME purpose ("benchmark on the AMI corpus" -> "benchmark on the ICSI corpus").
     - AGREED STEP added, dropped, or reversed, or a DECISION flipped
       ("include a FULL clustering step" -> "drop the clustering step").
   Return all contradicted idx values in contradicted_facts (empty list if none).
   A fact from EXISTING FACTS can be BOTH a duplicate AND contradicted.

3. WHAT IS **NOT** A CONTRADICTION (PRECISION — do NOT over-fire):
   Do NOT mark a fact contradicted just because the NEW FACT shares the same topic,
   person, or decision. In meetings, most later utterances CONFIRM or BUILD ON an
   earlier decision rather than reverse it. Return the candidate in NEITHER list when
   the NEW FACT merely:
     - RESTATES, APPROVES, CONFIRMS, or AGREES WITH the SAME value the old fact
       already holds (same corpus, same owner, same deadline). The SAME value is
       NEVER a contradiction — not even when a different speaker says it.
     - reports PROGRESS, STATUS, or a RESULT of the SAME task/plan (e.g. "X is
       assigned to Priyabanta" is NOT contradicted by "Priyabanta shared results on
       X" — the assignment still stands).
     - ELABORATES, adds DETAIL, or describes a NEXT STEP of the same decision.
     - is a SEPARATE, independent event, or a different slot that can coexist.
   The test is strict MUTUAL EXCLUSIVITY of the SAME slot with a CHANGED value —
   NOT mere topical similarity, and NOT "a later fact mentions the same thing".

<EXAMPLES>
-- these ARE contradictions (a value on the same slot genuinely CHANGED) --
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[]  (identical information)

EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: duplicate_facts=[], contradicted_facts=[1]  (same role slot, CHANGED title — supersession)

EXISTING FACT: idx=2, "The team will submit the report by the end of June"
NEW FACT: "The team will submit the report by mid-July"
Result: duplicate_facts=[], contradicted_facts=[2]  (same deliverable, deadline CHANGED — old superseded)

EXISTING FACT: idx=3, "Islam is responsible for computing the evaluation metrics"
NEW FACT: "Priyabanta is responsible for computing the evaluation metrics"
Result: duplicate_facts=[], contradicted_facts=[3]  (same task, DIFFERENT owner — reassignment)

EXISTING FACT: idx=4, "The team will benchmark KGGen on the AMI corpus"
NEW FACT: "The team will benchmark KGGen on the ICSI corpus"
Result: duplicate_facts=[], contradicted_facts=[4]  (same purpose, DIFFERENT corpus)

EXISTING FACT: idx=5, "The pipeline will include a FULL clustering step"
NEW FACT: "The team decided to drop the clustering step from the pipeline"
Result: duplicate_facts=[], contradicted_facts=[5]  (agreed step REVERSED)

-- these are NOT contradictions (same value / progress / elaboration — do NOT fire) --
EXISTING FACT: idx=6, "Rahul approves switching the benchmark to the ICSI corpus"
NEW FACT: "Faris shared preliminary results from the ICSI corpus"
Result: duplicate_facts=[], contradicted_facts=[]  (SAME choice, ICSI — a progress report CONFIRMS the decision, it does not reverse it)

EXISTING FACT: idx=7, "Priyabanta is assigned the KG-metrics survey task"
NEW FACT: "Priyabanta proposed semantic-similarity measures for the KG-metrics survey"
Result: duplicate_facts=[], contradicted_facts=[]  (SAME owner doing the SAME task — elaboration, NOT a reassignment)

EXISTING FACT: idx=8, "Bob presented the roadmap on Tuesday"
NEW FACT: "Bob presented the budget on Wednesday"
Result: duplicate_facts=[], contradicted_facts=[]  (two independent events that can both be true)
</EXAMPLES>
""",
        ),
    ]


def apply_overrides() -> None:
    """Install the meeting-aware resolve_edge prompt onto the shared singleton.

    Must run BEFORE the Graphiti client resolves any edge. Because every call site
    imports the same `prompt_library` object and `VersionWrapper.__call__` reads
    `self.func` at call time, reassigning `.func` here is enough — no re-import, no
    fork, no reinstall. Idempotent: calling twice just sets the same function again.
    """
    prompt_library.dedupe_edges.resolve_edge.func = resolve_edge_meetings
    print("🩹 prompt override ACTIVE: dedupe_edges.resolve_edge -> meeting-aware "
          "(implicit-revision) contradiction detection")
