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
    modules/graphic-generation/scripts/run_on_unicorn.sh --captions-file ...
        │  ONE subprocess for the whole board: the ~31 GB FLUX pipeline is
        │  constructed once, not once per fact (~6 min -> ~70 s for 6 facts)
        ▼
    compose_board.compose(...)                  (imported directly, needs PIL)
        │
        ▼
    one .excalidraw file, ui/output/board_<stamp>.excalidraw
"""
from __future__ import annotations

import base64
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


def _generate_images_via_server(captions: list[str], cb: ProgressCB) -> list[str]:
    """Render via the warm FLUX server, if FLUX_SERVER_URL points at one.

    The batch subprocess path below still pays one ~40-90 s model load per
    board. A server that already holds the pipeline in VRAM removes even that:
    ~10 s for a 6-image board instead of ~70 s.

    Returns None for this whole path if no server is reachable, so the caller
    can fall back rather than failing - the server is an optimisation, never a
    requirement.
    """
    base = os.getenv("FLUX_SERVER_URL", "").rstrip("/")
    if not base:
        return None  # type: ignore[return-value]

    import urllib.error
    import urllib.request

    # Probe first: a stale FLUX_SERVER_URL (tunnel dropped, server restarted)
    # should cost a 2 s timeout and a fallback, not a hung board.
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=2) as r:
            health = json.loads(r.read())
    except Exception:                                     # noqa: BLE001
        _log(cb, f"   ℹ️  no FLUX server at {base} — falling back to a batched "
                 f"remote call (one model load)")
        return None  # type: ignore[return-value]

    _log(cb, f"🖼️  FLUX server ({health.get('device', '?')}, "
             f"{health.get('status')}) — {len(captions)} caption(s), no model load")

    payload = json.dumps({"captions": [" ".join(c.split()) for c in captions]}).encode()
    req = urllib.request.Request(f"{base}/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(
            req, timeout=int(os.getenv("FLUX_BATCH_TIMEOUT", "3600"))) as r:
        result = json.loads(r.read())

    # The server returns PNG bytes inline (base64) because it lives on another
    # host with no shared filesystem. Land them next to every other board asset.
    out_dir = os.path.join(OUT_DIR, f"flux_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    paths: list[str] = []
    by_index = {img["index"]: img for img in result.get("images", [])}
    for i in range(len(captions)):
        img = by_index.get(i)
        if img is None or "png_b64" not in img:
            err = (img or {}).get("error", "no response entry")
            _log(cb, f"   ⚠️  caption {i + 1} produced no image ({err}) — skipping it")
            paths.append(None)  # type: ignore[arg-type]
            continue
        path = os.path.join(out_dir, f"image_{i:03d}.png")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(img["png_b64"]))
        paths.append(path)
    return paths


def _generate_images(captions: list[str], cb: ProgressCB) -> list[str]:
    """Render EVERY caption in ONE remote call. Returns paths, aligned to input.

    This used to be _generate_image(), invoked once per fact. That meant each
    image paid the full round trip: rsync up, ssh, ~40-90 s of FLUX pipeline
    construction (~31 GB of weights), render, rsync down. A 6-fact board spent
    ~6 minutes almost entirely on loading the same model six times, for ~12 s of
    actual rendering.

    Sending all captions at once makes it one load: ~70 s for the same board.

    Result paths come from the batch's manifest.jsonl, NOT from diffing the
    output directory. The old approach listed output/ before and after and took
    the newest PNG, which mis-assigns caption to image the moment two runs
    overlap or anything else writes there.

    A caption FLUX failed on yields None in its slot rather than aborting the
    board - losing one pictogram should not cost the other five.
    """
    if not captions:
        return []

    # Warm server first (no model load at all); the batched subprocess below is
    # the fallback when no server is configured or reachable.
    via_server = _generate_images_via_server(captions, cb)
    if via_server is not None:
        return via_server

    # One caption per line is the contract generate_image.py --captions-file
    # reads. Newlines inside a caption would silently split it into two prompts,
    # so flatten any that appear.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        for c in captions:
            fh.write(" ".join(c.split()) + "\n")
        captions_path = fh.name

    _log(cb, f"🖼️  FLUX — {len(captions)} caption(s) in ONE remote call "
             f"(model loads once, not {len(captions)}x)")
    try:
        proc = subprocess.run(
            ["bash", os.path.join(IMG_MODULE, "scripts", "run_on_unicorn.sh"),
             "--captions-file", captions_path],
            cwd=IMG_MODULE, capture_output=True, text=True,
            # One load plus N renders, plus two rsyncs of a large tree. The old
            # per-image timeout of 300 s is far too tight for the whole batch.
            timeout=int(os.getenv("FLUX_BATCH_TIMEOUT", "3600")),
        )
    finally:
        os.unlink(captions_path)

    if proc.returncode != 0:
        raise PipelineError(
            f"FLUX generation failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}")

    # run_on_unicorn.sh prints this as its last line.
    batch_dir = None
    for line in proc.stdout.splitlines():
        if line.startswith("local-batch-dir: "):
            batch_dir = line.split("local-batch-dir: ", 1)[1].strip()
    if not batch_dir:
        raise PipelineError(
            f"no batch directory reported by run_on_unicorn.sh — stdout tail:\n"
            f"{proc.stdout[-2000:]}")

    manifest = os.path.join(batch_dir, "manifest.jsonl")
    by_index: dict[int, dict] = {}
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                by_index[rec["index"]] = rec

    paths: list[str] = []
    for i in range(len(captions)):
        rec = by_index.get(i)
        if rec is None or not rec.get("path"):
            err = (rec or {}).get("error", "no manifest entry")
            _log(cb, f"   ⚠️  caption {i + 1} produced no image ({err}) — skipping it")
            paths.append(None)  # type: ignore[arg-type]
        else:
            paths.append(rec["path"])
    return paths


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

    # Captions first, for ALL facts, then a single batched render. Interleaving
    # them (caption -> image -> caption -> image) is what forced one model load
    # per fact; the captions are stdlib-only and cost milliseconds, so there is
    # no reason to spread them across the expensive calls.
    captions = [kg_to_caption.caption_from_fact(f["fact"]) for f in selected]
    image_paths = _generate_images(captions, progress_cb)

    entries = []
    for i, (fact, caption, image_path) in enumerate(
            zip(selected, captions, image_paths)):
        if image_path is None:
            continue          # FLUX failed on this one; the board omits it
        entries.append({
            "image": image_path,
            "caption": caption,
            "timestamp": fact.get("valid_at"),
            "label": f"{i + 1}" + (" · superseded" if fact.get("invalid_at") else ""),
        })
    if not entries:
        raise PipelineError("every caption failed to render — nothing to compose")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"board_{stamp}.excalidraw")
    _log(progress_cb, f"🧩 composing {len(entries)} images onto one canvas...")
    scene = compose_board.compose(entries, out_path, columns=columns)
    debug = scene.pop("_layout_debug")
    with open(out_path, "w") as fh:
        json.dump(scene, fh, indent=2)
    _log(progress_cb, f"✅ wrote {out_path} — canvas {debug['width']:.0f}x{debug['height']:.0f}")

    return {"board_path": out_path, "entries": entries}
