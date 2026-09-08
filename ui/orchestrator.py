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

# kg_to_caption reads OLLAMA_HOST / CAPTION_MODEL at IMPORT time, so these have
# to be set before the import below. Default to the hosted model on unicorn's
# H100 rather than a laptop ollama: this machine has no NVIDIA GPU, and one
# 4B-model call on its CPU costs seconds where the H100 costs 0.28 s. A real
# environment variable still wins, so the laptop fallback is one export away.
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11435")
os.environ.setdefault("CAPTION_MODEL", "qwen3:4b-instruct")

# Prefer the warm FLUX server too. Without this the code silently falls back to
# the rsync-and-ssh path, which still carries the ORIGINAL author's account as
# its default — so the failure surfaces as "Permission denied for sandulu",
# which says nothing about the real cause (no FLUX_SERVER_URL set).
# A real environment variable still wins; _generate_images_via_server() probes
# /health and falls back cleanly if nothing is listening.
os.environ.setdefault("FLUX_SERVER_URL", "http://localhost:8500")

import kg_to_caption  # noqa: E402  (stdlib-only, safe to import cross-module)
import compose_board  # noqa: E402  (needs PIL, ui/requirements.txt provides it)

ProgressCB = Optional[Callable[[str], None]]


class PipelineError(RuntimeError):
    pass


def _log(cb: ProgressCB, msg: str) -> None:
    if cb:
        cb(msg)


def _ensure_tunnels(cb: ProgressCB = None) -> None:
    """Open the two SSH tunnels if they are not already answering.

    Called at the top of every entry point so nobody has to remember a -L line —
    including someone driving the Streamlit UI, who has no terminal in front of
    them at all. tunnels.sh is idempotent and probes the SERVICES rather than the
    ports (a dead tunnel keeps its port bound, so a port check reports healthy
    while every request hangs), so calling this always is cheap and safe.

    Never fatal: FLUX has an rsync+ssh fallback, and a run that only reads an
    existing graph needs no tunnel at all. Set CONVOGRAPH_NO_TUNNEL=1 to skip —
    for anyone running ON unicorn, where the services are already local.
    """
    if os.getenv("CONVOGRAPH_NO_TUNNEL"):
        return
    script = os.path.join(REPO_ROOT, "scripts", "tunnels.sh")
    if not os.path.exists(script):
        return
    try:
        proc = subprocess.run(["bash", script, "start"], capture_output=True,
                              text=True, timeout=60)
    except subprocess.TimeoutExpired:
        _log(cb, "   ⚠️  tunnel setup timed out — continuing")
        return
    for line in proc.stdout.splitlines():
        if line.strip():
            _log(cb, f"   {line.rstrip()}")


def list_groups(cb: ProgressCB = None) -> list[dict]:
    """Every group_id module 2 has in the store, most superseded facts first.

    The UI builds its group dropdown from this and auto-runs on the first entry,
    so the ORDER is the default choice. It is decided in ui_ingest.list_groups
    (one place, in the database), not here. Each entry is
    {group_id, episodes, facts, superseded}.
    """
    if not os.path.exists(KG_PYTHON):
        raise PipelineError(
            f"module 2's venv not found at {KG_PYTHON} — run "
            f"`python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt` "
            f"inside {KG_MODULE}"
        )
    _log(cb, "🗄️  listing graphs in Neo4j")
    proc = subprocess.run(
        [KG_PYTHON, os.path.join(KG_MODULE, "ui_ingest.py"), "--list-groups"],
        cwd=KG_MODULE, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise PipelineError(f"listing graphs failed:\n{proc.stderr[-4000:]}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])["groups"]
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        raise PipelineError(
            f"group listing produced no parseable JSON — stdout tail:\n"
            f"{proc.stdout[-2000:]}"
        ) from exc


def _facts_from_existing_graph(group_id: str, limit: int, cb: ProgressCB) -> dict:
    """Read facts from a graph module 2 ALREADY built. Extracts nothing.

    Module 2's extraction is the slow, expensive stage — minutes per transcript,
    and a benchmark-quality graph needs the 30B model on the cluster rather than
    the 4B we serve for the UI. Module 3 does not need to re-derive any of it: a
    graph full of facts already exists, so treat it as FIXED INPUT and go
    straight to captions and images.

    This is also the shape module 3 was always meant to have — a cypher query
    against the temporal KG, not a transcript box. Re-running extraction later
    is just choosing the other branch in run_pipeline().
    """
    _log(cb, f"🗄️  reading existing graph (group_id={group_id}) — no extraction")
    proc = subprocess.run(
        [KG_PYTHON, os.path.join(KG_MODULE, "ui_ingest.py"),
         "--query-only", "--group-id", group_id, "--limit", str(limit)],
        cwd=KG_MODULE, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise PipelineError(f"reading the graph failed:\n{proc.stderr[-4000:]}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise PipelineError(
            f"query produced no parseable JSON — stdout tail:\n{proc.stdout[-2000:]}"
        ) from exc


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
    text: str = "",
    group_id: Optional[str] = None,
    max_facts: int = 6,
    columns: int = 3,
    progress_cb: ProgressCB = None,
    existing_group_id: Optional[str] = None,
) -> dict:
    """Returns {"board_path": ..., "entries": [...]} — entries is what actually
    went on the board (image path, caption, timestamp, label), for a UI that
    wants to show more than just the file path."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    _ensure_tunnels(progress_cb)

    if existing_group_id:
        # Module 2's output is treated as FIXED INPUT. Nothing is extracted.
        group_id = existing_group_id
        # Ask for exactly what the board needs: the query ranks facts the
        # conversation later overturned first, so a limit is a selection, not a
        # truncation.
        ingest_result = _facts_from_existing_graph(
            existing_group_id, max_facts, progress_cb)
        facts = ingest_result.get("facts", [])
        _log(progress_cb, f"   {len(facts)} fact(s) read from the existing graph")
        if not facts:
            raise PipelineError(
                f"group '{existing_group_id}' has no facts. Check the group_id — "
                f"a typo here looks exactly like an empty graph."
            )
        # Already ranked and limited by the query; re-selecting would undo that.
        selected = facts
    else:
        if not text or not text.strip():
            raise PipelineError("no transcript text provided")
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


# ─── the CONTENT MAP path (hops 2b + 4) ─────────────────────────────────────
# run_pipeline() above is the ORIGINAL shape: one fact -> one caption -> one
# image, laid out as a grid. It stays because it is what produced every board
# before 2026-09-06 and is the thing the redesign is measured against.
#
# What follows is the redesign. The differences are the whole point:
#
#   run_pipeline()                     run_content_map()
#   ------------------------------     ---------------------------------------
#   one fact -> one caption            a WINDOW of messages + facts -> one PLAN
#   caption does words AND drawing     plan splits them: labels/notes = text,
#                                        glyph = the only thing FLUX sees
#   N images in a grid (a gallery)     2-4 nodes + labelled arrows (a MAP)
#   no model choice                    provider/model is an experiment variable
#   output flat in ui/output/          one folder per run, holding the plan,
#                                        the images, the board and a preview

def _run_digest(group_id: str, episode_limit: int, cb: ProgressCB) -> dict:
    """Run module 2's five digest queries in ITS venv and read back the JSON.

    A subprocess rather than an import, for the reason this whole file is built
    that way: module 2 owns graphiti-core and the neo4j driver, the UI venv has
    streamlit and Pillow, and their heavy pins conflict. Importing tkg_digest
    here raised ModuleNotFoundError for neo4j — the rule stated at the top of
    this file, broken and then caught by the first run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [KG_PYTHON, os.path.join(KG_MODULE, "analysis", "tkg_digest.py"),
               "--group-id", group_id, "--out", tmp]
        if episode_limit:
            cmd += ["--episode-limit", str(episode_limit)]
        # The connection is passed explicitly so the digest reads the SAME
        # database the rest of the run does, rather than whatever module 2's
        # .env happens to point at.
        for flag, env in (("--uri", "NEO4J_URI"), ("--user", "NEO4J_USER"),
                          ("--password", "NEO4J_PASSWORD"),
                          ("--database", "NEO4J_DATABASE")):
            if os.getenv(env):
                cmd += [flag, os.environ[env]]
        proc = subprocess.run(cmd, cwd=KG_MODULE, capture_output=True,
                              text=True, timeout=900)
        if proc.returncode != 0:
            raise PipelineError(f"the digest failed:\n{proc.stderr[-4000:]}")
        with open(os.path.join(tmp, "digest.json")) as fh:
            return json.load(fh)


def _window_run(group_id: str, windows: int, max_facts: int,
                cb: ProgressCB) -> dict:
    """A CONTIGUOUS run of `windows` episodes: their raw messages AND facts.

    The raw messages are the reason this exists. A lone fact carries no context
    ("Waiting one week too long on a setup-question spike caused the fix to
    become a training scramble" is not interpretable alone), so the planner is
    given the conversation the facts came from, not just the facts.
    """
    _log(cb, f"🗄️  reading {windows} window(s) from group '{group_id}' "
             f"(≤{max_facts} facts) — module 2 output is FIXED INPUT, "
             f"nothing is extracted")
    proc = subprocess.run(
        [KG_PYTHON, os.path.join(KG_MODULE, "ui_ingest.py"),
         "--query-only", "--group-id", group_id,
         "--windows", str(windows), "--max-facts", str(max_facts)],
        cwd=KG_MODULE, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise PipelineError(f"reading the graph failed:\n{proc.stderr[-4000:]}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise PipelineError(
            f"query produced no parseable JSON — stdout tail:\n{proc.stdout[-2000:]}"
        ) from exc


# Every field of plan.json, documented next to it. Written in the SAME step
# that writes the plan, never "later": in three months the difference between
# _n_facts_in and len(_facts) is the difference between reading a result and
# reverse-engineering one.
PLAN_README = {
    "title": "str — the board headline, drawn as canvas text. LLM-authored, <=5 words.",
    "anchors": "list — the nodes. Each is {label, glyph, from_facts}.",
    "anchors[].label": "str — the words drawn under the node's pictogram. Canvas TEXT, never sent to FLUX.",
    "anchors[].glyph": "str — a wordless physical object. The ONLY field FLUX sees, via board_plan.glyph_to_prompt().",
    "anchors[].from_facts": "list[int] — indices into _facts. This is the traceability link back to the temporal KG.",
    "links": "list — {from, to, label}: from/to are ANCHOR indices, label is drawn on the arrow. These make it a map.",
    "notes": "list — {text, anchor}: <=10 words drawn inside that anchor's card.",
    "dropped": "list[int] — fact indices the planner judged not worth drawing. Not a failure; being selective is the job.",
    "_provider": "str — 'saia' or 'ollama'. See board_plan.PROVIDERS.",
    "_model": "str — the exact planner model. The ablation variable.",
    "_n_facts_in": "int — how many facts the planner was given. Equals len(_facts).",
    "_windows": "int — contiguous 5-message episodes fed to the planner.",
    "_group_id": "str — the Neo4j group the facts came from.",
    "_plan_seconds": "float — wall-clock for the single planning call.",
    "_validation": "list[str] — problems board_plan.validate() found. NON-BLOCKING: a flawed plan is still rendered, on purpose.",
    "_facts": "list — the exact facts handed to the planner, in index order, each {fact, valid_at, invalid_at}. invalid_at non-null = the conversation later overturned it.",
}


def _write_plan_readme(run_dir: str) -> None:
    with open(os.path.join(run_dir, "plan.readme.json"), "w") as fh:
        json.dump(PLAN_README, fh, indent=2)


def _write_prompt(run_dir: str, messages: list[dict], raw_reply: str) -> None:
    """Persist the EXACT prompt the planner received, plus its raw reply.

    Written as plain .txt in a prompt/ subfolder rather than embedded in
    plan.json, because the point is to READ it — count the facts the model saw,
    check whether they are intelligible, see whether the conversation text
    survived. A 40 KB string inside a JSON field is not readable.

    full.txt is what actually goes over the wire, in order, with the role
    banners the API applies — that is the artefact to reason about when a board
    comes out wrong.
    """
    d = os.path.join(run_dir, "prompt")
    os.makedirs(d, exist_ok=True)
    for msg in messages:
        with open(os.path.join(d, f"{msg['role']}.txt"), "w") as fh:
            fh.write(msg["content"])
    with open(os.path.join(d, "full.txt"), "w") as fh:
        for msg in messages:
            fh.write(f"{'=' * 78}\n=== {msg['role'].upper()}\n{'=' * 78}\n")
            fh.write(msg["content"].rstrip() + "\n\n")
    if raw_reply:
        with open(os.path.join(d, "reply_raw.txt"), "w") as fh:
            fh.write(raw_reply)

    chars = sum(len(m["content"]) for m in messages)
    with open(os.path.join(d, "README.md"), "w") as fh:
        fh.write(
            "# The prompt this run sent to the board planner\n\n"
            "| file | what it is |\n|---|---|\n"
            "| `full.txt` | every message in order — what actually went over the wire |\n"
            "| `system.txt` | the standing instructions (`prompts/board_plan_system.txt`) |\n"
            "| `user.txt` | this run's input: the meeting messages, then the indexed facts |\n"
            "| `reply_raw.txt` | the model's reply before JSON extraction |\n\n"
            f"Total prompt: **{chars:,} characters** (~{chars // 4:,} tokens).\n\n"
            "The fact indices in `user.txt` are the same ones `plan.json` refers to "
            "in `anchors[].from_facts`, `notes[].fact` and `dropped`, so a board "
            "element can be traced to the exact line the planner read.\n")


def run_content_map(
    group_id: str,
    text: str = "",
    windows: int = 2,
    max_facts: int = 40,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    dedupe_facts: bool = True,
    use_digest: bool = False,
    episode_limit: int = 0,
    progress_cb: ProgressCB = None,
) -> dict:
    """Window -> board plan -> pictograms -> ONE content map. Returns a dict
    with the run directory and everything in it.

    Pass `text` to extract first (module 1's transcript -> a new graph), or
    leave it empty to read a group module 2 already built.

    EVERY run gets its OWN directory, named for the model and window count:

        ui/output/<provider>-<model>_w<N>_<stamp>/
            plan.json          what the LLM decided, with its own provenance
            board.excalidraw   the canvas
            preview.png        an approximate raster, so the board can be
                               judged without opening the app
            images/000.png     one pictogram per anchor

    Flat output was actively costing us: yesterday's three-model ablation left
    nothing on disk but images whose model of origin could not be recovered, so
    this morning there was no plan to look at. A run that cannot be told apart
    from another run is not a measurement.
    """
    # Imported here, not at module import: these live in module 3's directory
    # (already on sys.path above) and only the content-map path needs them.
    import board_plan          # noqa: PLC0415
    import render_board        # noqa: PLC0415
    sys.path.insert(0, os.path.join(IMG_MODULE, "analysis"))
    import preview_board       # noqa: PLC0415
    # Module 2 owns the redundancy measure, and there is exactly one
    # implementation of it. It is stdlib-only, so importing it here costs
    # nothing and cannot drag graphiti-core into the UI's venv.
    sys.path.insert(0, os.path.join(KG_MODULE, "analysis"))
    import fact_redundancy     # noqa: PLC0415

    stamp = time.strftime("%Y%m%d_%H%M%S")
    _ensure_tunnels(progress_cb)
    provider = provider or board_plan.DEFAULT_PROVIDER
    model = model or board_plan.PROVIDERS[provider]["default_model"]

    # ── hop 0 (optional): extraction ────────────────────────────────────────
    # A transcript means the graph does not exist yet, so module 2 has to run
    # before there is anything to window over. This is the module 1 -> board
    # path: ASR/diarization -> pipeline/transcript_to_ui_text.py -> this box.
    # It is NOT the default, because re-deriving module 2's output on every
    # module-3 iteration is exactly the cost the --query-only path removed.
    if text and text.strip():
        group_id = group_id or f"ui_{stamp}"
        ingested = _ingest(text, group_id, progress_cb)
        _log(progress_cb, f"   {ingested.get('episodes', '?')} episode(s) → "
                          f"{len(ingested.get('facts', []))} fact(s) extracted")
        # Ask for no more windows than were actually created, or the query
        # silently returns fewer and the run dir's name lies about its input.
        windows = min(windows, max(1, int(ingested.get("episodes") or 1)))

    # ── hop 1: the input ────────────────────────────────────────────────────
    # Two shapes, and the difference is the point of the digest work:
    #
    #   window   2 episodes, ~40 facts, chosen by POSITION. The planner sees
    #            whichever five minutes it was handed.
    #   digest   five cypher queries over the whole meeting, chosen by what the
    #            graph says MATTERED. Same prompt size, far more coverage.
    if use_digest:
        _log(progress_cb, f"🧪 digesting the graph — five queries over "
                          f"{'the whole group' if not episode_limit else f'{episode_limit} episodes'}")
        t_dig = time.time()
        dig = _run_digest(group_id, episode_limit, progress_cb)
        for k, v in dig["timings"].items():
            _log(progress_cb, f"   {k:<14} {v:>5.2f}s  ({len(dig['queries'][k])} rows)")
        _log(progress_cb, f"   digest took {time.time() - t_dig:.1f}s over "
                          f"{dig['episodes_digested']} episode(s)")

        _log(progress_cb, f"🧭 planning the board — {provider}/{model}")
        t0 = time.time()
        plan, items = board_plan.plan_board_from_digest(dig, provider=provider,
                                                        model=model)
        plan_secs = time.time() - t0
        messages = board_plan.build_messages_from_digest(dig)
        facts = [{"fact": it.get("fact") or it.get("was") or "",
                  "invalid_at": it.get("changed_at"),
                  "valid_at": it.get("valid_from") or it.get("asked_at")}
                 for it in items]
        problems = board_plan.validate(plan, len(items),
                                       plan.get("_anchor_range", (2, 4)))
        run = {"episode_texts": [], "facts": facts, "window_start": None}
        raw_n = len(facts)
        digest_result = dig
    else:
        run = _window_run(group_id, windows, max_facts, progress_cb)
        digest_result = None
    episode_texts = run.get("episode_texts", [])
    facts = run.get("facts", [])
    if not use_digest and not facts:
        raise PipelineError(
            f"group '{group_id}' returned no facts. Check the group_id — a typo "
            f"looks exactly like an empty graph."
        )
    if not use_digest:
        superseded = sum(1 for f in facts if f.get("invalid_at"))
        _log(progress_cb, f"   {len(episode_texts)} episode(s) from "
                          f"w{run.get('window_start')}, {len(facts)} fact(s), "
                          f"{superseded} superseded")

    # ── hop 1b: collapse restatements ───────────────────────────────────────
    # Graphiti decomposes a listing sentence into one fact per item AND keeps
    # the combined one, so a narrow window arrives ~2x redundant (measured
    # 2026-09-07: 40 facts, 19 distinct ideas at 2 windows). The planner was
    # already discarding these — its "dropped" list was mostly duplicates — but
    # it was spending its attention, and its 2-4 anchor budget, choosing between
    # restatements of the same sentence. Giving it distinct ideas instead makes
    # "dropped" mean "judged not worth drawing" rather than "was a duplicate".
    raw_n = len(facts)
    if dedupe_facts and not use_digest:
        facts, removed = fact_redundancy.dedupe(facts)
        if removed:
            _log(progress_cb, f"   ♻️  collapsed {removed} restatement(s) → "
                              f"{len(facts)} distinct idea(s) "
                              f"({raw_n / len(facts):.2f}x redundant)")

    # ── hop 2: the plan ─────────────────────────────────────────────────────
    if not use_digest:
        _log(progress_cb, f"🧭 planning the board — {provider}/{model}")
        t0 = time.time()
        plan = board_plan.plan_board(episode_texts, facts, provider=provider,
                                     model=model)
        plan_secs = time.time() - t0
        problems = board_plan.validate(plan, len(facts))
    # Built a second time rather than returned from plan_board, so persisting the
    # prompt can never change what was actually sent. build_messages is pure.
        messages = board_plan.build_messages(episode_texts, facts)

    # A flawed plan is still rendered. The point of this stage is looking at
    # results, and a board that is 90% right teaches more than an exception.
    if problems:
        _log(progress_cb, f"   ⚠️  plan has {len(problems)} validation problem(s) "
                          f"— rendering anyway:")
        for p in problems:
            _log(progress_cb, f"      - {p}")
    else:
        _log(progress_cb, f"   ✅ plan validates clean in {plan_secs:.1f}s")

    anchors = plan.get("anchors", [])
    _log(progress_cb, f"   \"{plan.get('title', '')}\" — {len(anchors)} anchor(s), "
                      f"{len(plan.get('links', []))} link(s), "
                      f"{len(plan.get('notes', []))} note(s), "
                      f"{len(plan.get('dropped', []))} fact(s) dropped")

    tag = "digest" if use_digest else f"w{windows}"
    run_dir = os.path.join(
        OUT_DIR, f"{provider}-{model}_{tag}_{stamp}".replace("/", "-"))
    os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)

    plan["_windows"] = windows
    plan["_group_id"] = group_id
    plan["_plan_seconds"] = round(plan_secs, 2)
    plan["_validation"] = problems
    plan["_facts"] = facts            # the exact input, so a plan's fact indices
                                      # stay resolvable months from now
    plan["_facts_before_dedupe"] = raw_n
    plan["_deduped"] = bool(dedupe_facts and not use_digest)
    if digest_result is not None:
        # The digest is written next to the plan: a board planned from it is only
        # auditable if the exact input survives beside the output.
        d = os.path.join(run_dir, "digest")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "digest.json"), "w") as fh:
            json.dump(digest_result, fh, indent=2, default=str)
    with open(os.path.join(run_dir, "plan.json"), "w") as fh:
        json.dump(plan, fh, indent=2)
    _write_plan_readme(run_dir)
    _write_prompt(run_dir, messages, plan.get("_raw_reply", ""))

    # ── hop 3: one pictogram per anchor ─────────────────────────────────────
    # The glyph, NOT the label: FLUX draws wordless objects and letters as
    # gibberish, so only the glyph is ever allowed near it.
    prompts = []
    for a in anchors:
        drawn, replaced = board_plan.safe_glyph(a.get("glyph", ""))
        if replaced:
            _log(progress_cb, f"   🔁 glyph {a.get('glyph')!r} → {drawn!r} "
                              f"(a {replaced} comes back covered in invented letters)")
            a["_glyph_drawn"] = drawn
            a["_glyph_replaced"] = replaced
        prompts.append(board_plan.glyph_to_prompt(a.get("glyph", "")))
    image_paths = _generate_images(prompts, progress_cb) if prompts else []

    kept = []
    for i, path in enumerate(image_paths):
        if not path:
            kept.append(None)
            continue
        dest = os.path.join(run_dir, "images", f"{i:03d}.png")
        with open(path, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())      # copied INTO the run dir so the folder is
                                       # self-contained and survives a cleanup
        kept.append(dest)

    # ── hop 4: the canvas ───────────────────────────────────────────────────
    _log(progress_cb, f"🧩 drawing the content map — {len(anchors)} node(s), "
                      f"{sum(1 for k in kept if k)} pictogram(s)")
    board_path = os.path.join(run_dir, "board.excalidraw")
    scene = render_board.build_scene(plan, kept)
    debug = scene.pop("_layout_debug")
    with open(board_path, "w") as fh:
        json.dump(scene, fh, indent=2)

    # A preview and a structural check, so a broken board is caught here rather
    # than when someone opens it in the app.
    preview_path = os.path.join(run_dir, "preview.png")
    preview_board.render(scene, preview_path)
    issues = preview_board.check(scene)
    for issue in issues:
        _log(progress_cb, f"   ⚠️  layout: {issue}")

    _log(progress_cb, f"✅ {board_path} — canvas {debug['canvas'][0]}x{debug['canvas'][1]}")
    return {
        "run_dir": run_dir, "board_path": board_path,
        "preview_path": preview_path, "plan": plan,
        "provider": provider, "model": model, "windows": windows,
        "used_digest": use_digest,
        "plan_seconds": round(plan_secs, 2),
        "validation": problems, "layout_issues": issues, "debug": debug,
    }
