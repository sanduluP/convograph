#!/usr/bin/env python3
"""
trace_fact_to_image.py — follow ONE real fact from the knowledge graph all the
way to a PNG, printing the exact string at every hop.

WHY THIS EXISTS
---------------
The pipeline crosses three processes, two machines and two models, so it is easy
to understand the ARCHITECTURE and still have no idea what the DATA looks like.
This script answers "what actually flows through here?" with real values rather
than a diagram:

    cypher query        -> the fact sentence Graphiti stored
    qwen2.5:3b-instruct -> the FLUX prompt that sentence becomes
    FLUX.1-schnell      -> the PNG that prompt becomes

Nothing here is a mock: it queries the live Neo4j, calls the real caption model,
and hits the warm FLUX server on unicorn.

OUTPUT
------
  <out-dir>/trace.jsonl        one record per hop (see trace.readme.jsonl)
  <out-dir>/image.png          what FLUX returned

REQUIREMENTS
------------
  - .env with NEO4J_* (this module's, git-ignored)
  - ollama running locally with the caption model
  - a tunnel to the FLUX server:  ssh -N -L 8500:localhost:8500 unicorn
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request

# kg_to_caption lives in module 3 but is stdlib-only, so importing it here costs
# nothing and guarantees we trace the SAME function the orchestrator calls -
# a re-implementation would drift and quietly stop being a true trace.
_MODULE3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "..", "graphic-generation")
sys.path.insert(0, os.path.abspath(_MODULE3))

# kg_to_caption reads these at IMPORT time, so they must be set before it is
# imported further down. Same defaults as ui/orchestrator.py, deliberately: a
# trace is only worth writing down if it used the models the pipeline actually
# uses. Left on the laptop this hop took 33 s on CPU instead of well under a
# second on the H100 - and reported a different model than the real run.
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11435")
os.environ.setdefault("CAPTION_MODEL", "qwen3:4b-instruct")


def load_env(path: str) -> None:
    """Read the git-ignored .env the same way ui_ingest.py does."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def fetch_facts(group_id: str) -> list[dict]:
    """Run the EXACT cypher ui_ingest.py runs, so the trace starts where the
    real pipeline starts rather than at a query invented for this script."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )
    cypher = (
        "MATCH ()-[r:RELATES_TO {group_id: $gid}]->() WHERE r.fact IS NOT NULL "
        "RETURN r.fact AS fact, r.valid_at AS valid_at, r.invalid_at AS invalid_at "
        "ORDER BY r.valid_at ASC"
    )
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
        rows = [dict(r) for r in s.run(cypher, gid=group_id)]
    driver.close()
    return rows, cypher


def pick(facts: list[dict], index: int | None) -> tuple[int, dict]:
    """Prefer a SUPERSEDED fact when the caller did not choose one.

    A fact with invalid_at set is one the conversation later overturned - the
    only kind that shows the temporal layer doing something a flat summary
    could not. That is also what the orchestrator's _select_facts() prioritises.
    """
    if index is not None:
        return index, facts[index]
    for i, f in enumerate(facts):
        if f.get("invalid_at"):
            return i, f
    return 0, facts[0]


def flux(caption: str, server: str, out_png: str) -> dict:
    """Ask the warm FLUX server for one image. Returns timing + the saved path."""
    payload = json.dumps({"captions": [caption], "seed": 11}).encode()
    req = urllib.request.Request(f"{server.rstrip('/')}/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        result = json.loads(r.read())
    img = result["images"][0]
    if "png_b64" not in img:
        raise SystemExit(f"FLUX returned no image: {img.get('error')}")
    with open(out_png, "wb") as fh:
        fh.write(base64.b64decode(img["png_b64"]))
    return {"seconds": img.get("seconds"), "wall_seconds": round(time.time() - t0, 2),
            "path": out_png, "bytes": os.path.getsize(out_png)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group-id", required=True, help="Graphiti group_id to read.")
    ap.add_argument("--fact-index", type=int, default=None,
                    help="Which fact to trace. Default: the first superseded one.")
    ap.add_argument("--flux-server", default=os.getenv("FLUX_SERVER_URL", "http://localhost:8500"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--env-file", default=os.path.join(os.path.dirname(__file__), "..", ".env"))
    args = ap.parse_args()

    load_env(os.path.abspath(args.env_file))
    os.makedirs(args.out_dir, exist_ok=True)
    trace_path = os.path.join(args.out_dir, "trace.jsonl")
    records = []

    # ── HOP 1: the knowledge graph ──────────────────────────────────────────
    print("═" * 74)
    print("HOP 1  cypher  →  the fact Graphiti stored")
    print("═" * 74)
    facts, cypher = fetch_facts(args.group_id)
    print(f"  group_id : {args.group_id}")
    print(f"  cypher   : {cypher}")
    print(f"  returned : {len(facts)} fact(s), "
          f"{sum(1 for f in facts if f.get('invalid_at'))} superseded")
    if not facts:
        raise SystemExit(f"no facts in group '{args.group_id}' — nothing to trace")

    idx, fact = pick(facts, args.fact_index)
    fact_text = fact["fact"]
    print(f"\n  tracing fact #{idx}"
          f"{'  [SUPERSEDED]' if fact.get('invalid_at') else ''}:")
    print(f"    {fact_text!r}")
    print(f"    valid_at   = {fact.get('valid_at')}")
    print(f"    invalid_at = {fact.get('invalid_at')}")
    records.append({"hop": 1, "stage": "cypher", "group_id": args.group_id,
                    "cypher": cypher, "facts_returned": len(facts),
                    "fact_index": idx, "value": fact_text,
                    "valid_at": str(fact.get("valid_at")),
                    "invalid_at": str(fact.get("invalid_at"))})

    # ── HOP 2: the caption model ────────────────────────────────────────────
    import kg_to_caption  # noqa: E402  (path set at the top of this file)

    print()
    print("═" * 74)
    print(f"HOP 2  {kg_to_caption.MODEL}  →  the FLUX prompt")
    print("═" * 74)
    print(f"  system prompt: {kg_to_caption.PROMPT_FILE}")
    t0 = time.time()
    caption = kg_to_caption.caption_from_fact(fact_text)
    dt = round(time.time() - t0, 2)
    print(f"  took {dt}s\n")
    print(f"    IN  : {fact_text!r}")
    print(f"    OUT : {caption!r}")
    records.append({"hop": 2, "stage": "caption", "model": kg_to_caption.MODEL,
                    "prompt_file": kg_to_caption.PROMPT_FILE,
                    "input": fact_text, "value": caption, "seconds": dt})

    # ── HOP 3: FLUX ─────────────────────────────────────────────────────────
    print()
    print("═" * 74)
    print("HOP 3  FLUX.1-schnell  →  the PNG")
    print("═" * 74)
    print(f"  server: {args.flux_server}")
    out_png = os.path.join(args.out_dir, "image.png")
    res = flux(caption, args.flux_server, out_png)
    print(f"  render {res['seconds']}s (wall {res['wall_seconds']}s) → "
          f"{res['path']}  ({res['bytes'] // 1024} KB)")
    records.append({"hop": 3, "stage": "flux", "model": "FLUX.1-schnell",
                    "input": caption, "value": res["path"],
                    "seconds": res["seconds"], "bytes": res["bytes"]})

    with open(trace_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\n📄 trace: {trace_path}")


main()
