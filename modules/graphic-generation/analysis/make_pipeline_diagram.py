#!/usr/bin/env python3
"""
make_pipeline_diagram.py — the module 3 pipeline, as one landscape diagram.

WHAT IT DRAWS
-------------
The whole path from the temporal KG to the Excalidraw board, with the thing that
is easiest to get wrong made explicit: WORDS AND DRAWINGS SPLIT at the plan and
never meet again until the canvas. FLUX only ever sees a wordless noun phrase;
every actual word on the board is Excalidraw text.

Counts in the boxes come from a real run (the treasury phase, 2026-09-11) rather
than being invented, so the diagram is a record of something that happened.

STYLE — CLAUDE.md rule 12
-------------------------
Serpentine rows, alternating direction, so the shape is ~2:1 landscape and fits
a LaTeX \\textwidth slot instead of eating a page. roughness 0 and fontFamily 6
(Nunito), because this is a diagram to be READ — the hand-drawn preset belongs
to the board itself, which is a poster.

Colour says WHERE A STEP RUNS, which is the question people actually ask of this
pipeline:
    yellow   DATA        a file or a graph that exists on disk
    purple   LOCAL       runs on the laptop
    blue     REMOTE API  SAIA
    green    GPU         FLUX on unicorn's H100

ONE-SHOT. Per rule 12 this generates the FIRST version and then the file is
Faris's: he aligns it by hand in the app and re-running would destroy that.
Changes after that go to him as instructions, not as a re-run.

Usage:
    python analysis/make_pipeline_diagram.py
"""
from __future__ import annotations

import json
import os
import random
import textwrap
import time
import uuid

# ── geometry, matching module 2's serpentine diagrams ───────────────────────
BOX_W, BOX_H = 292, 104
GAP_X, GAP_Y = 52, 132          # a little more vertical room than module 2: these
                               # boxes all carry a note underneath
NOTE_GAP = 12
MARGIN_X, TOP = 58, 176
NOTE_FS, NOTE_LH = 11.5, 1.25
TITLE_FS, SUB_FS, LABEL_FS = 26, 13, 14
CHAR_W = 0.52                  # Nunito advance width per font size

INK, MUTED = "#1e1e1e", "#6b7280"
# Background / stroke per kind. Muted, never saturated (rule 6).
KIND = {
    "data":   ("#fff9db", "#e0b400"),   # yellow — a file or graph on disk
    "local":  ("#f3f0ff", "#7048e8"),   # purple — runs on the laptop
    "remote": ("#e7f5ff", "#1971c2"),   # blue   — SAIA
    "gpu":    ("#ebfbee", "#2f9e44"),   # green  — FLUX on the H100
}


def _id() -> str:
    return uuid.uuid4().hex[:16]


def _base(kind: str, x: float, y: float, w: float, h: float) -> dict:
    return {
        "type": kind, "id": _id(), "x": round(x, 2), "y": round(y, 2),
        "width": round(w, 2), "height": round(h, 2), "angle": 0,
        "strokeColor": INK, "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": None,
        "seed": random.randint(1, 2**31 - 1),
        "versionNonce": random.randint(1, 2**31 - 1), "version": 1,
        "isDeleted": False, "boundElements": None,
        "updated": round(time.time() * 1000), "link": None, "locked": False,
    }


def _text(x, y, s, size, color=INK, w=None, align="left") -> dict:
    lines = s.split("\n")
    width = w if w is not None else max(len(l) for l in lines) * size * CHAR_W
    el = _base("text", x, y, width, len(lines) * size * NOTE_LH)
    el.update({"strokeColor": color, "text": s, "originalText": s,
               "fontSize": size, "fontFamily": 6, "textAlign": align,
               "verticalAlign": "top", "containerId": None,
               "lineHeight": NOTE_LH, "baseline": round(size * 0.8)})
    return el


def _box(x, y, kind, label, note) -> list[dict]:
    """A step: the rounded rectangle, its bound label, and a grey note under it.

    The label is BOUND (containerId) so it stays centred when the box is dragged;
    the note is free text, placed under the box because a serpentine leaves no
    room to the right and a note there would sit on the next box.
    """
    bg, stroke = KIND[kind]
    rect = _base("rectangle", x, y, BOX_W, BOX_H)
    rect.update({"backgroundColor": bg, "strokeColor": stroke,
                 "roundness": {"type": 3}, "boundElements": []})

    lab = _text(x, y, label, LABEL_FS, INK, BOX_W, "center")
    lab.update({"containerId": rect["id"], "verticalAlign": "middle"})
    rect["boundElements"].append({"id": lab["id"], "type": "text"})

    out = [rect, lab]
    if note:
        # Wrap each authored line SEPARATELY. These notes are IN:/OUT:/WHERE:
        # records where the line breaks carry meaning; flattening them into one
        # paragraph and re-wrapping would run the output path into the input.
        cols = max(20, int(BOX_W / (NOTE_FS * CHAR_W)) - 1)
        lines = []
        for raw in note.split("\n"):
            lines += textwrap.wrap(raw, cols) or [""]
        out.append(_text(x, y + BOX_H + NOTE_GAP, "\n".join(lines),
                         NOTE_FS, MUTED, BOX_W))
    return out


def _arrow(a: dict, b: dict, label: str = "", dashed: bool = False) -> list[dict]:
    """Bound arrow between two boxes, routed by their relative position."""
    ax, ay, bx, by = a["x"], a["y"], b["x"], b["y"]
    if abs(ay - by) < 1:                       # same row
        if bx > ax:                            # left to right
            x0, y0 = ax + BOX_W, ay + BOX_H / 2
            x1, y1 = bx, by + BOX_H / 2
        else:                                  # right to left
            x0, y0 = ax, ay + BOX_H / 2
            x1, y1 = bx + BOX_W, by + BOX_H / 2
    else:                                      # the row wrap: a short drop
        x0, y0 = ax + BOX_W / 2, ay + BOX_H
        x1, y1 = bx + BOX_W / 2, by

    el = _base("arrow", x0, y0, x1 - x0, y1 - y0)
    el.update({
        "strokeColor": "#343a40", "strokeWidth": 1.5,
        "strokeStyle": "dashed" if dashed else "solid",
        "roundness": {"type": 2},
        "points": [[0, 0], [round(x1 - x0, 2), round(y1 - y0, 2)]],
        "lastCommittedPoint": None,
        "startBinding": {"elementId": a["id"], "focus": 0, "gap": 6},
        "endBinding": {"elementId": b["id"], "focus": 0, "gap": 6},
        "startArrowhead": None, "endArrowhead": "arrow", "elbowed": False,
        "boundElements": [],
    })
    for box in (a, b):
        box.setdefault("boundElements", []).append({"id": el["id"], "type": "arrow"})

    out = [el]
    if label:
        lab = _text(0, 0, label, 12, "#1971c2")
        lab.update({"containerId": el["id"], "textAlign": "center",
                    "verticalAlign": "middle"})
        el["boundElements"].append({"id": lab["id"], "type": "text"})
        out.append(lab)
    return out


# ── the pipeline ────────────────────────────────────────────────────────────
# Every box earns its place (rule 12): no box merely restates the one before it.
# Rows are chosen so a stage never straddles a wrap.
ROWS = [
    # row 0 — the DIGEST: graph to five answers
    [
        ("data", "Temporal KG\n(AuraDB)",
         "IN: nothing — it is the source.\n"
         "OUT: group_id treasury_prod_deploy_speaker_free\n"
         "76 episodes · 1,394 facts · 157 superseded\n"
         "WHERE: AuraDB, neo4j+s://fa42f976…"),
        ("local", "5 cypher queries\ntkg_digest.py",
         "IN: group_id (+ optional episode cap)\n"
         "OUT: 5 result sets · 3.9 s · no LLM\n"
         "CODE: modules/kg-agent-memory/analysis/tkg_digest.py\n"
         "RUN: scripts/run_tkg_digest.sh"),
        ("data", "digest.json\n+ 6 .jsonl",
         "ALL FIVE result sets combined in ONE json.\n"
         "OUT: ui/output/<run>/digest/digest.json\n"
         "     …/digest/{revisions,topics,artifacts,\n"
         "     participants,decisions,open_threads}.jsonl\n"
         "     …/digest/readme/*.readme.jsonl"),
    ],
    # row 1 — the PLAN: answers to a JSON board plan
    [
        ("local", "flatten_digest()\nboard_plan.py",
         "IN: digest.json\n"
         "OUT: one user message, sections numbered [0],[1]…\n"
         "CODE: modules/graphic-generation/board_plan.py"),
        ("data", "the prompt\nsystem + user",
         "IN: flatten_digest output + the system file\n"
         "OUT: ui/output/<run>/prompt/full.txt (as sent)\n"
         "     …/prompt/{system,user,reply_raw}.txt\n"
         "SYSTEM: prompts/board_plan_digest_system.txt\n"
         "SIZE: ~10.9 KB ≈ 2.7k tokens"),
        ("remote", "Board Planner LLM\nSAIA qwen3-30b",
         "IN: the two messages\n"
         "OUT: JSON only · one call, 7-49 s\n"
         "WHERE: chat-ai.academiccloud.de (needs SAIA_API_KEY)\n"
         "Retries 4x on 5xx — SAIA hiccups."),
        ("data", "plan.json",
         "OUT: ui/output/<run>/plan.json\n"
         "title · anchors[label + glyph + from_facts]\n"
         "links · notes · dropped\n"
         "THIS is where words and drawings part."),
    ],
    # row 2 — the SPLIT and the canvas
    [
        ("gpu", "FLUX.1-schnell\nunicorn H100",
         "IN: the GLYPH ONLY — 'a padlock', never a sentence\n"
         "OUT: one PNG per anchor · ~12 s for six\n"
         "WHERE: localhost:8500 via the unicorn tunnel\n"
         "SERVER: modules/graphic-generation/serve_flux.py"),
        ("data", "one PNG per anchor",
         "OUT: ui/output/<run>/images/000.png …\n"
         "Wordless pictograms. FLUX renders letters as\n"
         "gibberish, which is why no word is sent to it."),
        ("local", "render_board.py",
         "IN: plan.json + the PNGs\n"
         "OUT: anchors→nodes, links→labelled arrows,\n"
         "     notes→text inside cards\n"
         "CODE: modules/graphic-generation/render_board.py"),
        ("data", "board.excalidraw\n+ preview.png",
         "OUT: ui/output/<run>/board.excalidraw\n"
         "     ui/output/<run>/preview.png\n"
         "The content map. Yours from here — open it in\n"
         "the app and drag things."),
    ],
]

# Where the words go. This is the arrow the diagram exists to make obvious.
WORDS_NOTE = (
    "THE SPLIT — plan.json carries two kinds of thing and they never meet again "
    "until the canvas.   GLYPH → FLUX, wordless, one noun phrase per anchor.   "
    "LABEL · NOTE · TITLE → drawn straight onto Excalidraw as TEXT, crisp and "
    "editable.        "
    "NOTE ON digest.md: it is the digest rendered FOR A HUMAN. Nothing "
    "downstream reads it — the planner reads digest.json, and FLUX never sees "
    "the digest at all.        "
    "EVERY <run> IS ui/output/<provider>-<model>_digest_<timestamp>/")


def build() -> dict:
    elements: list[dict] = []
    boxes: list[list[dict]] = []          # row-major, the rectangle of each step

    for r, row in enumerate(ROWS):
        y = TOP + r * (BOX_H + GAP_Y + 64)
        row_boxes = []
        for c, (kind, label, note) in enumerate(row):
            # Serpentine: odd rows are drawn right-to-left, so the wrap between
            # rows is a short drop rather than a long arrow back across the page.
            col = c if r % 2 == 0 else (len(row) - 1 - c)
            x = MARGIN_X + col * (BOX_W + GAP_X)
            parts = _box(x, y, kind, label, note)
            elements += parts
            row_boxes.append(parts[0])
        boxes.append(row_boxes)

    # arrows within each row, then the wrap to the next
    for r, row_boxes in enumerate(boxes):
        for i in range(len(row_boxes) - 1):
            elements += _arrow(row_boxes[i], row_boxes[i + 1])
        if r + 1 < len(boxes):
            elements += _arrow(row_boxes[-1], boxes[r + 1][0], "glyph only"
                               if r == 1 else "")

    # The words path: plan.json also feeds render_board directly, as TEXT.
    elements += _arrow(boxes[1][3], boxes[2][2], "labels · notes · title", dashed=True)

    # ── titles ──────────────────────────────────────────────────────────────
    width = MARGIN_X * 2 + 4 * BOX_W + 3 * GAP_X
    elements.insert(0, _text(MARGIN_X, 44,
                             "Module 3 — temporal knowledge graph to graphic recording",
                             TITLE_FS, INK, width - 2 * MARGIN_X))
    sub = ("Five cypher queries distil one meeting; an LLM plans the board; FLUX draws "
           "only wordless glyphs. Counts from the treasury phase, 2026-09-11.")
    cols = int((width - 2 * MARGIN_X) / (SUB_FS * CHAR_W))
    elements.insert(1, _text(MARGIN_X, 44 + TITLE_FS * 1.6,
                             "\n".join(textwrap.wrap(sub, cols)), SUB_FS, MUTED,
                             width - 2 * MARGIN_X))

    # legend, as real coloured swatches — a grey dot keys nothing
    lx = MARGIN_X
    ly = 44 + TITLE_FS * 1.6 + SUB_FS * 1.25 * 2 + 10
    for kind, name in (("data", "data on disk"), ("local", "runs locally"),
                       ("remote", "SAIA API"), ("gpu", "FLUX on the H100")):
        bg, stroke = KIND[kind]
        sw = _base("rectangle", lx, ly, 13, 13)
        sw.update({"backgroundColor": bg, "strokeColor": stroke,
                   "roundness": {"type": 3}})
        elements += [sw, _text(lx + 19, ly - 1, name, 12, MUTED)]
        lx += 19 + len(name) * 12 * CHAR_W + 26

    bottom = TOP + len(ROWS) * (BOX_H + GAP_Y + 64)
    cols = int((width - 2 * MARGIN_X) / (12 * CHAR_W))
    elements.append(_text(MARGIN_X, bottom,
                          "\n".join(textwrap.wrap(WORDS_NOTE, cols)),
                          12, "#7048e8", width - 2 * MARGIN_X))

    return {"type": "excalidraw", "version": 2,
            "source": "convograph/modules/graphic-generation/analysis/make_pipeline_diagram.py",
            "elements": elements,
            "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
            "files": {}}


def main() -> None:
    random.seed(7)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "excalidraw", "pipeline")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "module3-pipeline.excalidraw")

    # Rule 12: never regenerate over a file that exists — it has been edited by
    # hand and none of that is visible from here.
    if os.path.exists(out):
        raise SystemExit(
            f"❌ {out} already exists.\n"
            f"   This generator is ONE-SHOT by design (CLAUDE.md rule 12): the "
            f"file is yours once it exists,\n   and re-running would destroy "
            f"alignment work that cannot be recovered.\n"
            f"   To rebuild from scratch, move the old one aside first.")

    scene = build()
    with open(out, "w") as fh:
        json.dump(scene, fh, indent=2)
    print(f"✅ {out}")
    print(f"   {len(scene['elements'])} elements, "
          f"{sum(len(r) for r in ROWS)} steps in {len(ROWS)} rows")


if __name__ == "__main__":
    main()
