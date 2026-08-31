#!/usr/bin/env python3
"""
kg_to_caption.py — TemporalKG subgraph -> LLM-written caption -> ImageRequest.

Test 2 of the "flux -> excalidraw" idea: the caption is no longer a hardcoded
default (see generate_image.py:DEFAULT_CAPTION, which Test 1 used and this
script does not touch). Instead it's grounded in a decision node from a
temporal KG (schemas/temporal-kg.schema.json), same shape module 2 produces.
No live Neo4j / module-2 KG export (nodes+edges) was reachable while wiring
this, but module 2's own eval corpus was: samples/real_kg_finance_aml.json is
built from three real messages (Msg_2084, Msg_3274, Msg_3495 — a decision and
the one that later superseded it) in
modules/kg-agent-memory/data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json,
reshaped into schema-shaped nodes/edges. samples/sample_kg.json (fully
hand-written) is kept as an offline-safe fallback. Point --kg at a real
Neo4j-sourced export once one exists; nothing downstream changes.

BACKEND
-------
Local ollama, OpenAI-compatible-shaped native /api/chat, stdlib only (no
`openai` package needed for this one call). Model is qwen2.5:3b-instruct —
the same short-output model modules/kg-agent-memory/analysis/compress_facts.py
uses, and for the same documented reason (see that module's CLAUDE.md): a
REASONING/hybrid build (this laptop's other local tag, qwen3.6, has a
`thinking` capability per `ollama show qwen3.6`) narrates its deliberation and
exhausts the token budget instead of answering a short-output prompt. An
INSTRUCT model answers directly.

Output is one schemas/image-request.schema.json object: {caption, subgraph_ref}.
Reusing generate_image.py / image_to_excalidraw.py after this is a caller
change only, same as the module's caption -> FLUX -> Excalidraw scaffold
already documents — see scripts/run_kg_to_image.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("CAPTION_MODEL", "qwen2.5:3b-instruct")
PROMPT_FILE = os.getenv(
    "CAPTION_PROMPT",
    os.path.join(os.path.dirname(__file__), "prompts", "image_caption_system.txt"),
)
STYLE_SUFFIX = ", hand-drawn sticky-note icon style, marker on white paper"
MAX_WORDS = 30


def load_kg(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def pick_decision_subgraph(kg: dict) -> tuple[str, list[str]]:
    """Return (fact text, grounding node ids) for the first decision node found.

    Prefers an explicit type=="decision" node; falls back to any message with
    is_decision_point=true, matching how modules/kg-agent-memory flags facts
    that matter enough to put on the board.
    """
    nodes = {n["id"]: n for n in kg["nodes"]}
    decision = next((n for n in kg["nodes"] if n.get("type") == "decision"), None)
    if decision is None:
        decision = next((n for n in kg["nodes"] if n.get("is_decision_point")), None)
    if decision is None:
        raise ValueError("no decision node (type='decision' or is_decision_point=true) in this KG")

    grounding = {decision["id"]}
    for edge in kg.get("edges", []):
        if edge["target"] == decision["id"] or edge["source"] == decision["id"]:
            grounding.add(edge["source"])
            grounding.add(edge["target"])
    node_ids = [nid for nid in nodes if nid in grounding]
    return decision["content"], node_ids


def _tidy(text: str) -> str:
    t = text.strip()
    t = re.sub(r"(?s)<think>.*?</think>", "", t).strip()  # hybrid models leak this even off
    t = t.strip().strip('"').strip("'")
    t = t.split("\n")[0].strip()
    words = t.split()
    if len(words) > MAX_WORDS:
        words = words[:MAX_WORDS]
    t = " ".join(words).rstrip(".,;: ")
    if not t.endswith(STYLE_SUFFIX):
        t += STYLE_SUFFIX
    return t


def _fallback(fact: str) -> str:
    """Deterministic caption used when ollama is unreachable — keeps the
    pipeline runnable offline, same rationale as compress_facts.py's fallback."""
    t = re.sub(r"^(Integration Team|Finance Ops|User_\d+)\s+", "", fact).strip()
    words = t.split()[:12]
    return " ".join(words).rstrip(".,;:") + STYLE_SUFFIX


def caption_from_fact(fact: str) -> str:
    system = open(PROMPT_FILE).read()
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": fact},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        return _tidy(body["message"]["content"])
    except (urllib.error.URLError, KeyError, TimeoutError) as exc:
        print(f"⚠️  {type(exc).__name__} calling ollama at {OLLAMA_HOST} — using fallback caption")
        return _fallback(fact)


def build_image_request(kg_path: str) -> dict:
    kg = load_kg(kg_path)
    fact, node_ids = pick_decision_subgraph(kg)
    caption = caption_from_fact(fact)
    return {
        "caption": caption,
        "subgraph_ref": {
            "source_conversation_id": kg["source_conversation_id"],
            "node_ids": node_ids,
        },
        "edit_target": None,
        "style_refs": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kg",
        default=os.path.join(os.path.dirname(__file__), "samples", "real_kg_finance_aml.json"),
        help="Path to a TemporalKG JSON file (schemas/temporal-kg.schema.json).",
    )
    parser.add_argument("--out", default=None, help="Write the ImageRequest JSON here too.")
    args = parser.parse_args()

    request = build_image_request(args.kg)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(request, fh, indent=2)

    print(f"caption : {request['caption']}")
    print(f"grounded: {', '.join(request['subgraph_ref']['node_ids'])}")


if __name__ == "__main__":
    main()
