#!/usr/bin/env python3
"""
serpentine_layout.py — re-lay an existing journey diagram into a landscape,
serpentine layout WITHOUT changing a single word of its content.

WHY
---
The v1 diagrams are a single tall column: 0.67:1 and 0.47:1. That shape is the
worst possible one for a paper or a slide — it eats a whole page and renders
unreadably small inside a README. The reference layout in the thesis repo
(stage1_part_area_census) is 1.96:1, which drops straight into a LaTeX
\\textwidth figure slot or a 16:9 slide.

WHAT IT DOES
------------
Reads the v1 file, keeps every box, every label and every grey side-note exactly
as written, and re-places them into rows that snake:

      row 0   ▸ ▸ ▸ ▸        (left to right)
      row 1   ◂ ◂ ◂ ◂        (right to left)
      row 2   ▸ ▸ ▸ ▸

The wrap between rows becomes a short vertical drop instead of a long arrow
flying back across the page — which is the entire point of a serpentine.

Side notes move from the RIGHT of each box to UNDERNEATH it, because in a
serpentine there is no room to the right, and a note under its box also keeps it
clear of the descending wrap arrows.

Nothing here invents content. If a box says something, it still says it.
"""
from __future__ import annotations

import copy
import json
import sys
import textwrap

# ── geometry ─────────────────────────────────────────────────────────────────
BOX_W, BOX_H = 292, 104          # every box identical: a grid, not a collage
GAP_X, GAP_Y = 52, 74            # generous, per the "relaxed, never crowded" rule
NOTE_GAP = 12                    # box bottom → note top
NOTE_BLOCK = 104                 # vertical room reserved for a note under a box
MARGIN_X, TOP = 58, 172
NOTE_FS, NOTE_LH = 11.5, 1.25    # note font size and line height
# Nunito is a touch narrower than 0.55em average; measured against the rendered
# PNGs, 0.52 * fontSize per character predicts line width within a few pixels.
CHAR_W = 0.52          # TOP leaves room for title + subtitle + legend

def _rewrap(text: str, width_px: float, font_size: float) -> str:
    """Re-flow note text to fit `width_px`, keeping blank-line paragraph breaks.

    Existing single newlines inside a paragraph are treated as soft (they were
    layout for a different width) and are re-flowed. Blank lines are meaningful
    and are preserved. Bullet lines starting with a bullet char keep their own
    line so lists do not run together.
    """
    cols = max(20, int(width_px / (font_size * CHAR_W)) - 1)
    out_paras = []
    for para in text.split("\n\n"):
        lines = [l.rstrip() for l in para.split("\n")]
        # a paragraph whose lines are bullets/arrows stays line-per-line
        if any(l.lstrip().startswith(("•", "→", "-", "1.", "2.")) for l in lines):
            wrapped = []
            for l in lines:
                wrapped += textwrap.wrap(l, cols, subsequent_indent="   ") or [""]
            out_paras.append("\n".join(wrapped))
        else:
            joined = " ".join(l.strip() for l in lines if l.strip())
            out_paras.append("\n".join(textwrap.wrap(joined, cols)) if joined else "")
    return "\n\n".join(out_paras)


def col_x(i: int) -> float:
    return MARGIN_X + i * (BOX_W + GAP_X)

def row_y(r: int) -> float:
    return TOP + r * (BOX_H + NOTE_GAP + NOTE_BLOCK + GAP_Y)


def relayout(src: str, dst: str, rows: list[list[str]], note_of: dict[str, str],
             group: dict | None = None, title: str | None = None) -> None:
    """Re-place `rows` of box ids in serpentine order; drop old arrows, redraw."""
    doc = json.load(open(src))
    by_id = {e["id"]: e for e in doc["elements"] if not e.get("isDeleted")}

    out: list[dict] = []
    placed: dict[str, tuple[float, float]] = {}      # box id -> (x, y)

    # Every row is right-aligned to the widest row, so the vertical drop between
    # rows is always straight down rather than diagonal.
    width_cols = max(len(r) for r in rows)

    for r, row in enumerate(rows):
        # Serpentine: odd rows read right-to-left, so reverse their placement.
        order = row if r % 2 == 0 else list(reversed(row))
        # Right-align short rows into the same column grid as the widest row.
        offset = width_cols - len(row)
        for i, bid in enumerate(order):
            box = copy.deepcopy(by_id[bid])
            slot = i + (offset if r % 2 == 0 else 0)
            x, y = col_x(slot), row_y(r)
            box.update(x=x, y=y, width=BOX_W, height=BOX_H)
            box.pop("index", None)
            out.append(box)
            placed[bid] = (x, y)

            # the bound label rides with its box, re-centred
            for b in (box.get("boundElements") or []):
                if b.get("type") != "text":
                    continue
                t = copy.deepcopy(by_id[b["id"]])
                th = t.get("height", 40)
                t.update(x=x + 6, y=y + (BOX_H - th) / 2, width=BOX_W - 12)
                t.pop("index", None)
                out.append(t)

            # the grey note moves from the right of the box to underneath it
            nid = note_of.get(bid)
            if nid and nid in by_id:
                n = copy.deepcopy(by_id[nid])
                # RE-WRAP. The v1 notes carry hard newlines chosen for a ~560 px
                # column; dropped into a 292 px box they would run past its right
                # edge and collide with the next column. A hand-placed \n is
                # correct for exactly one width, so re-wrap to the actual box
                # width instead of trusting the old breaks.
                txt = _rewrap(n["text"], BOX_W, NOTE_FS)
                n.update(x=x, y=y + BOX_H + NOTE_GAP, width=BOX_W,
                         height=NOTE_LH * NOTE_FS * (txt.count("\n") + 1),
                         fontSize=NOTE_FS, textAlign="left",
                         text=txt, originalText=txt)
                n.pop("index", None)
                out.append(n)

    # ── group band: one dashed rectangle behind whichever row it covers ───────
    if group:
        r = group["row"]
        gx0 = col_x(0) - 26
        gx1 = col_x(width_cols - 1) + BOX_W + 26
        gy0 = row_y(r) - 46
        gy1 = row_y(r) + BOX_H + NOTE_GAP + NOTE_BLOCK - 8
        out.append({
            "id": "group-band", "type": "rectangle", "x": gx0, "y": gy0,
            "width": gx1 - gx0, "height": gy1 - gy0, "angle": 0,
            "strokeColor": "#9c36b5", "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "dashed",
            "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
            "roundness": None, "seed": 91001, "version": 1, "versionNonce": 1,
            "isDeleted": False, "boundElements": None, "updated": 1,
            "link": None, "locked": False,
        })
        out.append(_text("group-band-label", gx0 + 14, gy0 + 10, gx1 - gx0 - 28,
                         22, group["label"], 16, "#9c36b5", "left"))

    # ── arrows: within a row, then a vertical drop at the wrap ───────────────
    seq = [b for row in rows for b in row]
    for k in range(len(seq) - 1):
        a, b = seq[k], seq[k + 1]
        (ax, ay), (bx, by) = placed[a], placed[b]
        same_row = abs(ay - by) < 1
        if same_row:
            # horizontal, pointing whichever way this row runs
            if bx > ax:
                x0, y0, dx = ax + BOX_W, ay + BOX_H / 2, bx - (ax + BOX_W)
            else:
                x0, y0, dx = ax, ay + BOX_H / 2, bx + BOX_W - ax
            out.append(_arrow(f"ar-{k}", x0, y0, [[0, 0], [dx, 0]], abs(dx), 0))
        else:
            # the wrap: straight down the same column
            x0 = ax + BOX_W / 2
            y0 = ay + BOX_H + NOTE_GAP + NOTE_BLOCK - 4
            out.append(_arrow(f"ar-{k}", x0, y0, [[0, 0], [0, by - y0]],
                              0, by - y0, dashed=True, color="#1971c2"))

    # ── title block ──────────────────────────────────────────────────────────
    # Carry over EVERY standalone text that sits above the first box in v1, not
    # just the ones literally named "title"/"subtitle". Faris retitles these
    # diagrams by hand in Excalidraw, which creates fresh elements with random
    # ids and deletes the originals — keying on the name would silently discard
    # his edits. Anything above the topmost box is title-block material.
    boxes_v1 = [e for e in by_id.values()
                if e["type"] == "rectangle" and e.get("backgroundColor", "transparent") != "transparent"]
    first_box_y = min((e["y"] for e in boxes_v1), default=10**9)
    heads = [e for e in by_id.values()
             if e["type"] == "text" and not e.get("containerId")
             and not e["id"].startswith(("note", "group", "loop"))
             and e["y"] < first_box_y]
    heads.sort(key=lambda e: e["y"])
    y_cursor = 40
    for t in heads:
        t = copy.deepcopy(t)
        size = t.get("fontSize", 16)
        t.update(x=MARGIN_X, y=y_cursor)
        t.pop("index", None)
        if title and t is heads[0]:
            t["text"] = t["originalText"] = title
        out.append(t)
        # gap in POINTS, derived from this element's own size — not a magic offset
        y_cursor += size * 1.25 + 10

    doc["elements"] = out
    json.dump(doc, open(dst, "w"), indent=2, ensure_ascii=False)

    xs = [e["x"] for e in out]; ys = [e["y"] for e in out]
    X = [e["x"] + e.get("width", 0) for e in out]
    Y = [e["y"] + e.get("height", 0) for e in out]
    w, h = max(X) - min(xs), max(Y) - min(ys)
    print(f"✅ {dst}\n   {w:.0f} x {h:.0f}  → aspect {w/h:.2f}:1   ({len(out)} elements)")


def _text(i, x, y, w, h, txt, size, color, align="left"):
    return {"id": i, "type": "text", "x": x, "y": y, "width": w, "height": h,
            "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
            "roundness": None, "seed": abs(hash(i)) % 99999, "version": 1,
            "versionNonce": 1, "isDeleted": False, "boundElements": None,
            "updated": 1, "link": None, "locked": False, "text": txt,
            "fontSize": size, "fontFamily": 6, "textAlign": align,
            "verticalAlign": "top", "containerId": None, "originalText": txt,
            "lineHeight": 1.25}


def _arrow(i, x, y, pts, w, h, dashed=False, color="#495057"):
    return {"id": i, "type": "arrow", "x": x, "y": y, "width": abs(w),
            "height": abs(h), "angle": 0, "strokeColor": color,
            "backgroundColor": "transparent", "fillStyle": "solid",
            "strokeWidth": 2, "strokeStyle": "dashed" if dashed else "solid",
            "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
            "roundness": None, "seed": abs(hash(i)) % 99999, "version": 1,
            "versionNonce": 1, "isDeleted": False, "boundElements": None,
            "updated": 1, "link": None, "locked": False, "points": pts,
            "lastCommittedPoint": None, "startBinding": None, "endBinding": None,
            "startArrowhead": None, "endArrowhead": "arrow", "elbowed": True}
