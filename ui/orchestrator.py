#!/usr/bin/env python3
"""
orchestrator.py — text transcript -> KG -> captions -> images -> ONE Excalidraw
canvas. The glue between modules 2 and 3, called by app.py (the Streamlit UI).

Deliberately a subprocess-per-stage pipeline, not a shared import graph: module
2 (graphiti-core, neo4j) and module 3 (diffusers/torch, remote-only) each keep
their own venv for a reason (see their requirements.txt headers — conflicting
heavy pins). This script's own venv (ui/requirements.txt) only needs
streamlit + Pillow; it shells out to the other two venvs/hosts instead of
importing across them.

    transcript text
        │  (module 2's own venv, subprocess)
        ▼
    modules/kg-agent-memory/ui_ingest.py --text-file ... --group-id ...
        │  JSON: {facts: [{fact, valid_at, invalid_at}, ...]}
        ▼
    kg_to_caption.caption_from_fact(fact)      (stdlib-only, imported directly)
        │  caption per fact
        ▼
    modules/graphic-generation/scripts/run_on_unicorn.sh --caption "..."
        │  (subprocess per fact, sequential — see NOT BUILT YET in that
        │  module's README about batching this into one remote call)
        ▼
    compose_board.compose(...)                  (imported directly, needs PIL)
        │
        ▼
    one .excalidraw file, ui/output/board_<stamp>.excalidraw
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Callable, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_MODULE = os.path.join(REPO_ROOT, "modules", "kg-agent-memory")
IMG_MODULE = os.path.join(REPO_ROOT, "modules", "graphic-generation")
KG_PYTHON = os.path.join(KG_MODULE, ".venv", "bin", "python")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

sys.path.insert(0, IMG_MODULE)
import kg_to_caption  # noqa: E402  (stdlib-only, safe to import cross-module)
import compose_board  # noqa: E402  (needs PIL, ui/requirements.txt provides it)

ProgressCB = Optional[Callable[[str], None]]


class PipelineError(RuntimeError):
    pass


def _log(cb: ProgressCB, msg: str) -> None:
    if cb:
        cb(msg)


def _ingest(text: str, group_id: str, cb: ProgressCB) -> dict:
    if not os.path.exists(KG_PYTHON):
        raise PipelineError(
            f"module 2's venv not found at {KG_PYTHON} — run "
            f"`python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt` "
            f"inside {KG_MODULE}"
        )
    _log(cb, f"🧠 ingesting transcript into the KG (group_id={group_id}) — "
             f"this is LLM-extraction-bound, expect low minutes...")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(text)
        text_path = fh.name
    try:
        proc = subprocess.run(
            [KG_PYTHON, os.path.join(KG_MODULE, "ui_ingest.py"),
             "--text-file", text_path, "--group-id", group_id],
            cwd=KG_MODULE, capture_output=True, text=True, timeout=1800,
        )
    finally:
        os.unlink(text_path)

    if proc.returncode != 0:
        raise PipelineError(f"ingestion failed:\n{proc.stderr[-4000:]}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise PipelineError(
            f"ingestion produced no parseable JSON — stdout tail:\n"
            f"{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
        ) from exc


def _select_facts(facts: list[dict], max_facts: int) -> list[dict]:
    """Prefer superseded/superseding pairs (the interesting revision narrative),
    then fill remaining slots chronologically. Mirrors build_board.py's rationale
    (a fact with invalid_at set is one the conversation later overturned) without
    that script's per-topic column grouping, since a UI-submitted transcript is
    one short conversation, not a multi-topic corpus."""
    superseded = [f for f in facts if f.get("invalid_at")]
    current = [f for f in facts if not f.get("invalid_at")]
    picked = (superseded + current)[:max_facts]
    picked.sort(key=lambda f: f.get("valid_at") or "")
    return picked


def _generate_image(caption: str, cb: ProgressCB) -> str:
    before = set(os.listdir(os.path.join(IMG_MODULE, "output"))) \
        if os.path.isdir(os.path.join(IMG_MODULE, "output")) else set()
    _log(cb, f"🖼️  FLUX (unicorn) — \"{caption[:70]}...\"" if len(caption) > 70
             else f"🖼️  FLUX (unicorn) — \"{caption}\"")
    proc = subprocess.run(
        ["bash", os.path.join(IMG_MODULE, "scripts", "run_on_unicorn.sh"),
         "--caption", caption],
        cwd=IMG_MODULE, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise PipelineError(f"FLUX generation failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    after = set(os.listdir(os.path.join(IMG_MODULE, "output")))
    new_pngs = [f for f in (after - before) if f.endswith(".png")]
    if not new_pngs:
        raise PipelineError(f"no new PNG appeared in output/ after generation:\n{proc.stdout[-1000:]}")
    return os.path.join(IMG_MODULE, "output", sorted(new_pngs)[-1])


def run_pipeline(
    text: str,
    group_id: Optional[str] = None,
    max_facts: int = 6,
    columns: int = 3,
    progress_cb: ProgressCB = None,
) -> dict:
    """Returns {"board_path": ..., "entries": [...]} — entries is what actually
    went on the board (image path, caption, timestamp, label), for a UI that
    wants to show more than just the file path."""
    if not text or not text.strip():
        raise PipelineError("no transcript text provided")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    group_id = group_id or f"ui_{stamp}"

    ingest_result = _ingest(text, group_id, progress_cb)
    facts = ingest_result.get("facts", [])
    _log(progress_cb, f"   {ingest_result.get('episodes', '?')} episode(s) → {len(facts)} fact(s) extracted")
    if not facts:
        raise PipelineError(
            "the KG extraction produced zero facts — the transcript may be too "
            "short/vague for the extraction model, or the model missed it "
            "entirely (see README's note on 3B-model extraction quality)"
        )

    selected = _select_facts(facts, max_facts)
    _log(progress_cb, f"   using {len(selected)} of {len(facts)} facts "
                       f"({sum(1 for f in selected if f.get('invalid_at'))} superseded)")

    entries = []
    for i, fact in enumerate(selected):
        caption = kg_to_caption.caption_from_fact(fact["fact"])
        image_path = _generate_image(caption, progress_cb)
        entries.append({
            "image": image_path,
            "caption": caption,
            "timestamp": fact.get("valid_at"),
            "label": f"{i + 1}" + (" · superseded" if fact.get("invalid_at") else ""),
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"board_{stamp}.excalidraw")
    _log(progress_cb, f"🧩 composing {len(entries)} images onto one canvas...")
    scene = compose_board.compose(entries, out_path, columns=columns)
    debug = scene.pop("_layout_debug")
    with open(out_path, "w") as fh:
        json.dump(scene, fh, indent=2)
    _log(progress_cb, f"✅ wrote {out_path} — canvas {debug['width']:.0f}x{debug['height']:.0f}")

    return {"board_path": out_path, "entries": entries}
