#!/usr/bin/env python3
"""For each benchmark question, find the closest FACTS in our knowledge graph.

WHY THIS EXISTS
---------------
By 2026-08-05 we had tested three retrieval-side explanations for Graphiti's
28.1 % — entity stitching, `group_id` namespacing, oracle channel scoping — and
all three came back null. A fourth score would not tell us anything new.

The question that is actually open is upstream of all of them:

    IS THE ANSWER EVEN IN THE GRAPH?

No judge LLM can settle that, because the judge only ever sees what retrieval
happened to surface. So this script deliberately CHEATS in the one way that makes
the question answerable: it uses the GOLD ANSWER as the search query, mines the
whole graph for the facts closest to it, and lays question / gold answer / top
facts side by side for a human to read.

If the right fact is sitting there and retrieval never found it, that is a
retrieval problem. If the right fact was never extracted at all, no retriever can
ever win and the extraction prompt is the thing to fix. Those two conclusions
point at completely different months of work, which is why it is worth an hour to
tell them apart by reading.

WHY BM25 AND NOT EMBEDDINGS
---------------------------
The graph stores a `fact_embedding` per edge, but embedding the gold answer would
need the same embedding model the ingest used — which lives on the cluster behind
a GPU job. Lexical scoring runs locally in seconds, is deterministic, and is
fully explainable when Faris reads a row and asks "why did this fact surface?".
The scores here are a retrieval aid for a human reader, not a metric, so the
usual objections to lexical matching do not bite.

THE knowledge_update ANGLE
--------------------------
Every question in this set is one where an earlier decision is later REVISED.
So the interesting property is not just "is the fact present" but "does the graph
hold the OLD fact and the NEW one, with `invalid_at` marking the switch". That is
the one thing BM25 structurally cannot represent, so it is where the temporal KG
has to earn its keep. Each candidate fact is therefore tagged CURRENT or
SUPERSEDED, and the per-question block reports whether any superseded fact showed
up at all.

Outputs (both written in the same run, per house rule 11):
  * an Obsidian note  — the human artefact, meant to be read row by row
  * a JSONL + its .readme.jsonl — the machine-readable record of the same audit
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neo4j import GraphDatabase          # noqa: E402
from rank_bm25 import BM25Okapi          # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — all overridable from the wrapper script, none hardcoded twice.
# ---------------------------------------------------------------------------
BOLT_URI = os.environ.get("KG_BOLT_URI", "bolt://localhost:7688")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "graphiti123")

CORPUS_JSON = REPO_ROOT / "data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json"
QUESTIONS_JSONL = REPO_ROOT / "questions/Finance/knowledge_update.jsonl"
WINDOW_SIZE = int(os.environ.get("WINDOW", "5"))
GROUP_ID = os.environ.get("GROUP_ID", "gmb_finance_full")
TOP_N = int(os.environ.get("TOP_N", "5"))

VAULT = Path(os.environ.get(
    "VAULT_DIR", "/home/faris/Documents/Obsidian Vault/🧮  DSA"))
NOTE_OUT = VAULT / "🔬 KG answer audit — Finance knowledge_update.md"
JSONL_OUT = REPO_ROOT / "results" / "kg_answer_audit_finance_knowledge_update.jsonl"

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
# Same grammar-word list the other analyses use, so "overlap" means one thing
# across the whole repo.
_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "by", "from",
    "as", "that", "this", "these", "those", "it", "its", "will", "would",
    "should", "can", "could", "has", "have", "had", "do", "does", "did", "not",
    "no", "yes", "we", "they", "he", "she", "you", "i", "our", "their", "team",
    "current", "currently", "approach", "now", "using", "use", "used", "which",
    "what", "how", "when", "who", "there", "then", "than", "so", "if", "all",
    "any", "each", "more", "most", "other", "some", "such", "only", "own",
    "same", "also", "into", "over", "after", "before", "during", "while",
}


def content_words(text: str) -> set:
    """Lowercased content tokens — grammar and 1-char tokens dropped."""
    return {t.lower() for t in _TOKEN_RE.findall(text or "")
            if len(t) > 1 and t.lower() not in _STOP}


def tokenize(text: str) -> List[str]:
    """BM25 needs a token LIST (duplicates carry term-frequency signal)."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")
            if len(t) > 1 and t.lower() not in _STOP]


# ---------------------------------------------------------------------------
# 1. Corpus → which channel does each ingest window belong to?
# ---------------------------------------------------------------------------
def build_window_channels() -> Dict[int, str]:
    """Map 1-based window number → channel name.

    We reuse the retriever's own `_build_windows` rather than re-deriving the
    windowing here: if the two ever disagreed, every channel label in this audit
    would be silently wrong, and a silently-wrong label is worse than none.
    """
    from baselines.rag_common.eval_lib import load_conversation_messages
    from baselines.graphiti.graphiti_retriever import _build_windows

    messages = load_conversation_messages(str(CORPUS_JSON))
    windows = _build_windows(messages, WINDOW_SIZE)
    # Window numbers are 1-based and global across the whole corpus — that is
    # what makes the 8 shard graphs mergeable, and what makes this map valid.
    return {n: messages[idxs[0]].get("_channel", "")
            for n, idxs in enumerate(windows, start=1)}


# ---------------------------------------------------------------------------
# 2. Graph → every extracted fact, with its temporal state and source channels
# ---------------------------------------------------------------------------
def load_facts(driver, window_channel: Dict[int, str]) -> List[dict]:
    """Pull all RELATES_TO edges plus enough context to judge them by eye."""
    # episode uuid → window number, so a fact can be traced to its channel.
    ep_window: Dict[str, int] = {}
    with driver.session() as s:
        for rec in s.run("MATCH (e:Episodic) RETURN e.uuid AS uuid, e.name AS name"):
            m = re.search(r"_w(\d+)$", rec["name"] or "")
            if m:
                ep_window[rec["uuid"]] = int(m.group(1))

    facts: List[dict] = []
    with driver.session() as s:
        result = s.run(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
            "RETURN a.name AS subject, b.name AS object, r.fact AS fact, "
            "       r.valid_at AS valid_at, r.invalid_at AS invalid_at, "
            "       r.episodes AS episodes"
        )
        for rec in result:
            eps = rec["episodes"] or []
            chans = sorted({window_channel.get(ep_window.get(u, -1), "")
                            for u in eps} - {""})
            facts.append({
                "subject": rec["subject"],
                "object": rec["object"],
                "fact": rec["fact"] or "",
                # Neo4j temporals → plain ISO strings for JSON round-tripping.
                "valid_at": str(rec["valid_at"]) if rec["valid_at"] else None,
                "invalid_at": str(rec["invalid_at"]) if rec["invalid_at"] else None,
                "channels": chans,
            })
    return facts


# ---------------------------------------------------------------------------
# 3. Score every fact against one question's GOLD ANSWER
# ---------------------------------------------------------------------------
def rank_facts(bm25: BM25Okapi, facts: List[dict], question: str, gold: str,
               top_n: int) -> List[dict]:
    """Return the `top_n` facts closest to this question's gold answer.

    The BM25 query is gold answer + question: the gold answer supplies the
    specific artefacts we are hunting for, the question supplies the topic so a
    fact about the right subject is not buried by one that merely shares a word.
    """
    scores = bm25.get_scores(tokenize(f"{gold} {question}"))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

    gold_words = content_words(gold)
    out = []
    for i in order:
        f = dict(facts[i])
        f["bm25_score"] = round(float(scores[i]), 2)
        # Reported separately from the BM25 score because it answers a different
        # question: how much of the gold answer's actual WORDING this fact carries.
        f["gold_coverage"] = (round(len(gold_words & content_words(f["fact"]))
                                    / len(gold_words), 3) if gold_words else 0.0)
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# 4. Render the Obsidian note — the artefact a human actually reads
# ---------------------------------------------------------------------------
def render_note(rows: List[dict], n_facts: int) -> str:
    n = len(rows)
    any_superseded = sum(1 for r in rows
                         if any(c["invalid_at"] for c in r["candidates"]))
    mean_cov = sum(max((c["gold_coverage"] for c in r["candidates"]), default=0.0)
                   for r in rows) / max(n, 1)

    out: List[str] = []
    out.append("---")
    out.append("tags: [dsa, kg, graphiti, audit, groupmembench]")
    out.append("---\n")
    out.append("# 🔬 Is the answer even IN the graph? — manual audit of all "
               f"{n} `knowledge_update` questions\n")
    out.append("> [!info] TL;DR — what this note is for")
    out.append("> Three retrieval-side fixes tested on 2026-08-05 all came back null,")
    out.append("> so the open question moved upstream: **was the answer ever extracted")
    out.append("> into the graph at all?** Here every question is paired with its gold")
    out.append(f"> answer and the **{n_facts} closest facts in the merged KG**, found by")
    out.append("> searching the graph *with the gold answer itself*. No judge LLM — the")
    out.append("> point is for us to read it and decide.")
    out.append(">")
    out.append(f"> Graph searched: {len(rows)} questions against the merged full-corpus")
    out.append("> KG (111,258 facts). Mean gold-answer word coverage of the single best")
    out.append(f"> fact per question: **{mean_cov:.1%}**. Questions where at least one")
    out.append(f"> candidate is a SUPERSEDED fact: **{any_superseded}/{n}**.\n")
    out.append("> [!question] How to read a row")
    out.append("> For each question ask yourself two things:")
    out.append("> 1. **Is the answer here at all?** If yes → this is a RETRIEVAL problem.")
    out.append(">    If no → extraction never captured it, and no retriever can win.")
    out.append("> 2. **Is the revision represented?** These are all *knowledge_update*")
    out.append(">    questions, so the graph should hold the old fact AND the new one,")
    out.append(">    with `invalid_at` set on the old. `SUPERSEDED` marks that.")
    out.append("> Tick the checkbox under each question with your verdict.\n")
    out.append("---\n")

    # Summary table first, so the shape of the result is visible before the detail.
    out.append("## Summary — scan this, then read the rows that look bad\n")
    out.append("| # | question (short) | best gold-coverage | any superseded fact? |")
    out.append("|---|---|---:|:---:|")
    for i, r in enumerate(rows, 1):
        best = max((c["gold_coverage"] for c in r["candidates"]), default=0.0)
        sup = "✅" if any(c["invalid_at"] for c in r["candidates"]) else "—"
        q = r["question"][:64] + ("…" if len(r["question"]) > 64 else "")
        out.append(f"| {i} | {q} | {best:.0%} | {sup} |")
    out.append("")
    out.append("---\n")

    for i, r in enumerate(rows, 1):
        out.append(f"## Q{i}. {r['question']}\n")
        out.append(f"**Asked by:** `{r['asking_user_id']}`\n")
        out.append("> [!success] Gold answer")
        for line in r["answer"].split("\n"):
            out.append(f"> {line}")
        out.append("")
        out.append(f"**Closest {len(r['candidates'])} facts in our KG:**\n")
        out.append("| # | state | fact | channel(s) | gold-cov |")
        out.append("|---|---|---|---|---:|")
        for j, c in enumerate(r["candidates"], 1):
            state = "🕘 SUPERSEDED" if c["invalid_at"] else "🟢 current"
            # Pipes inside a fact would break the Markdown table.
            fact = c["fact"].replace("|", "\\|")
            chans = ", ".join(ch[:28] for ch in c["channels"]) or "—"
            out.append(f"| {j} | {state} | {fact} | {chans} | {c['gold_coverage']:.0%} |")
        out.append("")
        out.append("- [ ] **answer is present in the graph**")
        out.append("- [ ] **the revision (old → new) is represented**")
        out.append("")
        out.append("---\n")

    return "\n".join(out)


def write_readme(path: Path) -> None:
    """One line per field — the house format (rule 11), written in the same step."""
    fields = [
        ("question_id", "str", "question", "GroupMemBench question id", "knowledge_update_1", "32", "false", "0", "Stable id, lets a row be traced back to questions/Finance/knowledge_update.jsonl"),
        ("question", "str", "question", "the question text as asked", "What is the current approval scope…", "32", "false", "0", "What the benchmark asks; also half of the BM25 query used to mine the graph"),
        ("asking_user_id", "str", "question", "user who asks it", "User_2", "12", "false", "0", "Does NOT identify a channel — 10 of 12 Finance users are in several channels"),
        ("answer", "str", "question", "gold answer", "Expand the approval scope now to…", "32", "false", "0", "Used AS THE SEARCH QUERY here: this audit deliberately cheats to ask whether the fact exists at all"),
        ("candidates", "list[obj]", "graph", "closest facts found", "[{fact,…}]", "32", "false", "0", "Top-N RELATES_TO edges by BM25 against (gold answer + question)"),
        ("candidates[].fact", "str", "graph", "the extracted fact sentence", "User_2 requests Security to name one approver by Friday.", "-", "false", "0", "The LLM-written English fact stored on the edge — this is what a human judges"),
        ("candidates[].subject", "str", "graph", "source Entity name", "User_2", "-", "false", "0", "Head of the RELATES_TO edge; weak/abstract subjects are the suspected root cause"),
        ("candidates[].object", "str", "graph", "target Entity name", "approver", "-", "false", "0", "Tail of the edge; often a sentence fragment rather than a thing"),
        ("candidates[].valid_at", "str|null", "graph", "when the fact became true", "2025-07-09T00:18:41", "-", "false", "some", "Bi-temporal lower bound as ISO text"),
        ("candidates[].invalid_at", "str|null", "graph", "when it was superseded", "2025-08-02T11:00:00", "-", "false", "many", "NULL = still current. Non-null is the whole point of a temporal KG: BM25 cannot represent this"),
        ("candidates[].channels", "list[str]", "graph", "channels of the source episodes", "['AML (Anti-Money Laundering) Project']", "-", "false", "0", "Derived episode → window number → channel; >1 means the fact was built from several projects"),
        ("candidates[].bm25_score", "float", "derived", "lexical match strength", "18.42", "-", "false", "0", "Ranking aid only, not comparable across questions"),
        ("candidates[].gold_coverage", "float", "derived", "share of gold-answer content words present in the fact", "0.35", "-", "false", "0", "The readable number: 0 means this fact shares no wording with the gold answer"),
    ]
    header = ("field, type, layer, layer_note, example, distinct_values, "
              "constant_across_file, nulls, description")
    lines = [header] + [", ".join(f) for f in fields]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print(f"🔗 connecting to {BOLT_URI}", flush=True)
    driver = GraphDatabase.driver(BOLT_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("🗺️  mapping windows → channels from the corpus", flush=True)
    window_channel = build_window_channels()
    print(f"   {len(window_channel)} windows, "
          f"{len(set(window_channel.values()))} channels", flush=True)

    print("📥 loading every RELATES_TO fact from the graph …", flush=True)
    facts = load_facts(driver, window_channel)
    driver.close()
    print(f"   {len(facts)} facts loaded", flush=True)
    if not facts:
        print("❌ no facts found — is the right Neo4j running on that port?")
        return 1

    print("🔎 building the BM25 index over fact text …", flush=True)
    bm25 = BM25Okapi([tokenize(f["fact"]) for f in facts])

    questions = [json.loads(l) for l in QUESTIONS_JSONL.open()]
    print(f"❓ mining {len(questions)} questions …", flush=True)

    rows = []
    for i, q in enumerate(questions, 1):
        cands = rank_facts(bm25, facts, q["question"], q.get("answer", ""), TOP_N)
        rows.append({
            "question_id": q.get("id", f"q{i}"),
            "question": q["question"],
            "asking_user_id": q.get("asking_user_id", ""),
            "answer": q.get("answer", ""),
            "candidates": cands,
        })
        best = max((c["gold_coverage"] for c in cands), default=0.0)
        print(f"   Q{i:<3} best gold-coverage {best:.0%}", flush=True)

    JSONL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_readme(JSONL_OUT.with_suffix(".readme.jsonl"))
    print(f"📝 wrote {JSONL_OUT} (+ .readme.jsonl)", flush=True)

    NOTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    # NEVER clobber an existing audit: the whole point of the note is that Faris
    # ticks the per-question verdict checkboxes by hand, and a re-run would throw
    # that reading away. Archive first, always — same discipline as never
    # overwriting a figure in place.
    if NOTE_OUT.exists():
        from datetime import datetime
        archive_dir = NOTE_OUT.parent / "archive"
        archive_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        archived = archive_dir / f"{NOTE_OUT.stem}_{stamp}{NOTE_OUT.suffix}"
        archived.write_text(NOTE_OUT.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"🗄️  archived the previous audit (and any ticks) → {archived}",
              flush=True)

    NOTE_OUT.write_text(render_note(rows, TOP_N), encoding="utf-8")
    print(f"📓 wrote {NOTE_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
