#!/usr/bin/env python3
"""
render_board.py — a board plan (+ one pictogram per anchor) -> a CONTENT MAP
on an Excalidraw canvas.

WHAT MAKES THIS A MAP AND NOT A GALLERY
---------------------------------------
compose_board.py places N (image, caption) pairs in a wrapping grid. That is a
gallery: the reader learns what each picture is about, but nothing about how
the ideas relate — and "concepts as nodes, relations as labelled arrows" is the
definition of the content map that 94% of the CHI 2021 sketchnote corpus uses.

So this draws the plan's four parts as four different things:

    title    -> one headline across the top
    anchor   -> a NODE: a card holding its wordless pictogram and its label
    link     -> a LABELLED ARROW between two nodes    <- the part that maps
    note     -> small recessive text INSIDE its anchor's card

Words and drawings stay strictly separated, which is the finding this whole
redesign rests on (2026-09-06): FLUX renders wordless objects well and letters
as gibberish, so every actual WORD on this canvas is a crisp Excalidraw text
element and every IMAGE is wordless.

LAYOUT
------
Nodes sit on an ELLIPSE, not a grid. A grid implies reading order (row 1, then
row 2) which a set of linked concepts does not have; a ring gives every node an
unobstructed straight line to every other, so a 0->2 arrow never has to cross
node 1. The ellipse is wider than tall (RATIO below) because a board is read on
a screen and printed in a paper column — both landscape.

    n = 2  ->  angles 180, 0            (a pair, side by side)
    n = 3  ->  angles -90, 30, 150      (a triangle, point up)
    n = 4  ->  angles -90, 0, 90, 180   (a diamond)

The radius is DERIVED from the card size and the node count rather than fixed,
so three big cards do not overlap and four small ones do not float apart:
adjacent nodes on a circle are 2*R*sin(pi/n) apart, so R follows from the
closest approach we are willing to accept.

WHY ARROWS ARE *BOUND*
----------------------
Each arrow carries startBinding/endBinding to the two cards, and each card
lists the arrow in boundElements. That is what makes the arrow follow the card
when Faris drags it in the app — and he will drag it, because no generated
layout survives a human eye. An unbound arrow is a line that detaches the
moment the board is touched.

Usage:
    python render_board.py --plan plan.json --images img0.png,img1.png \
        --out board.excalidraw
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import textwrap
import time
import uuid

from PIL import Image

# ── geometry ────────────────────────────────────────────────────────────────
IMG_W = 200            # displayed pictogram width; height follows aspect ratio
CARD_PAD = 18          # white space between the card's edge and its contents
CARD_W = IMG_W + 2 * CARD_PAD
MIN_NODE_GAP = 120     # closest two adjacent cards may come, edge to edge.
                       # Was 230, which was compensating for the radius bug
                       # below rather than reflecting what a label needs: once
                       # separation is MEASURED instead of assumed, even 80
                       # keeps every link label off every card at n=2..6. 120
                       # leaves margin without the airy 2113x1451 canvas that
                       # 230 produced for five anchors.
RATIO = 1.9            # ellipse width / height. >1 = landscape (see LAYOUT).
                       # 1.45 gave a 1.06:1 canvas for 4 anchors - almost
                       # square, which wastes a paper column and a screen.
MIN_RX = 300
PAD = 70               # canvas margin
TITLE_BAND = 110       # vertical space reserved above the nodes for the title

# ── type ────────────────────────────────────────────────────────────────────
TITLE_FS = 34
LABEL_FS = 20
LINK_FS = 15
NOTE_FS = 13
LINE_H = 1.25
CHAR_W = 0.52          # Excalifont advance width as a fraction of font size,
                       # the same estimate compose_board.py uses
FONT_HAND = 5          # Excalifont — the hand-drawn face, matching the board's
                       # graphic-recording preset

# ── colour ──────────────────────────────────────────────────────────────────
# One ink, one accent, one recessive gray — the discipline a real graphic
# recorder works with. The accent is reserved for the RELATION layer (arrows and
# their labels) so the eye can separate "the things" from "how they connect".
INK = "#1e1e1e"
ARROW = "#343a40"
ACCENT = "#1971c2"
NOTE_GRAY = "#5c5f66"
CARD_BG = "#ffffff"

ROUGHNESS = 1          # 1 = hand-drawn. This board is a sketchnote, not an
                       # architecture diagram (those use roughness 0).

# ── appending one meeting after another ─────────────────────────────────────
APPEND_GAP = 160       # white space between two MEETING blocks on one canvas,
                       # either direction. Larger than TITLE_BAND so the next
                       # block's title reads as a new board, not as a subtitle
                       # of the one above it. See append_scene().
CAPTION_FS = NOTE_FS   # a block caption is metadata: recessive gray, small
CAPTION_KEY = "convograph_caption"
                       # customData marker on caption elements. customData is
                       # the ONE per-element slot Excalidraw preserves through a
                       # hand edit and re-save; a top-level scene key is dropped
                       # on export, after which a chained append could no longer
                       # tell which blocks are already captioned.


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _now_ms() -> int:
    return round(time.time() * 1000)


def _rand() -> int:
    return random.randint(1, 2 ** 31 - 1)


def _wrap(text: str, width_px: float, fs: float) -> list[str]:
    """Wrap to the pixel width it will be drawn in, not to a guessed column
    count — the same rule as the plotting standards: a hand-placed newline is
    correct at exactly one size."""
    cols = max(8, int(width_px / (fs * CHAR_W)))
    return textwrap.wrap(" ".join(str(text).split()), cols,
                         break_on_hyphens=False) or [""]


def _text_size(lines: list[str], fs: float) -> tuple[float, float]:
    w = max((len(l) for l in lines), default=0) * fs * CHAR_W
    return w, len(lines) * fs * LINE_H


# ── element constructors ────────────────────────────────────────────────────
# Every Excalidraw element shares a long block of required keys. Building them
# through one _base() keeps the differences between element types visible
# instead of buried in forty lines of identical boilerplate.
def _base(kind: str, x: float, y: float, w: float, h: float,
          group_ids: list[str] | None = None) -> dict:
    return {
        "type": kind, "id": _new_id(),
        "x": round(x, 2), "y": round(y, 2),
        "width": round(w, 2), "height": round(h, 2),
        "angle": 0,
        "strokeColor": INK, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid",
        "roughness": ROUGHNESS, "opacity": 100,
        "groupIds": group_ids or [], "frameId": None, "roundness": None,
        "seed": _rand(), "versionNonce": _rand(), "version": 1,
        "isDeleted": False, "boundElements": None,
        "updated": _now_ms(), "link": None, "locked": False,
    }


def _card(x: float, y: float, w: float, h: float, gid: str) -> dict:
    """The node frame. It exists to be BINDABLE — arrows attach to a shape, not
    to an image — and to give the pictogram and its label one visual body."""
    el = _base("rectangle", x, y, w, h, [gid])
    el.update({
        "backgroundColor": CARD_BG,
        "roundness": {"type": 3},     # 3 = adaptive radius, Excalidraw's default
        "boundElements": [],          # arrows are appended here as they are made
    })
    return el


def _image(x: float, y: float, w: float, h: float, file_id: str,
           gid: str) -> dict:
    el = _base("image", x, y, w, h, [gid])
    el.update({
        "strokeColor": "transparent",
        "backgroundColor": "transparent",
        "status": "saved", "fileId": file_id, "scale": [1, 1],
    })
    return el


def _text(x: float, y: float, lines: list[str], fs: float, color: str,
          gid: str | None = None, align: str = "left",
          box_w: float | None = None) -> dict:
    """A free-standing text element (not bound to a container)."""
    txt = "\n".join(lines)
    w, h = _text_size(lines, fs)
    el = _base("text", x, y, box_w if box_w is not None else w, h,
               [gid] if gid else [])
    el.update({
        "strokeColor": color,
        "text": txt, "originalText": txt,
        "fontSize": fs, "fontFamily": FONT_HAND,
        "textAlign": align, "verticalAlign": "top",
        "containerId": None, "lineHeight": LINE_H, "baseline": round(fs * 0.8),
    })
    return el


def _edge_point(cx: float, cy: float, w: float, h: float,
                tx: float, ty: float, gap: float) -> tuple[float, float]:
    """Where a line from (cx,cy) towards (tx,ty) leaves the card, plus `gap`.

    Excalidraw recomputes this itself from the bindings when the file is
    opened, but writing correct geometry means the raw JSON is already right —
    so the board looks correct in any viewer, and a binding that fails to
    resolve degrades to a sensible line instead of a spike to (0,0).
    """
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    # Scale the direction vector until it hits whichever edge comes first.
    sx = (w / 2 + gap) / abs(dx) if dx else math.inf
    sy = (h / 2 + gap) / abs(dy) if dy else math.inf
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def _arrow(src: dict, dst: dict, label: str) -> list[dict]:
    """One labelled arrow between two cards, bound at both ends.

    Returns [arrow] or [arrow, label_text]. The label is a CONTAINED text
    (containerId = the arrow), which is how Excalidraw keeps a label centred on
    an arrow as the arrow moves; a free text element beside the arrow would
    stay behind the moment anything is dragged.
    """
    scx, scy = src["x"] + src["width"] / 2, src["y"] + src["height"] / 2
    dcx, dcy = dst["x"] + dst["width"] / 2, dst["y"] + dst["height"] / 2

    x0, y0 = _edge_point(scx, scy, src["width"], src["height"], dcx, dcy, 8)
    x1, y1 = _edge_point(dcx, dcy, dst["width"], dst["height"], scx, scy, 8)

    el = _base("arrow", x0, y0, x1 - x0, y1 - y0)
    el.update({
        "strokeColor": ARROW, "strokeWidth": 2,
        "roundness": {"type": 2},          # 2 = the curved multi-point style
        "points": [[0, 0], [round(x1 - x0, 2), round(y1 - y0, 2)]],
        "lastCommittedPoint": None,
        # focus 0 = aim at the shape's centre; gap = how far short of the
        # outline the arrowhead stops.
        "startBinding": {"elementId": src["id"], "focus": 0, "gap": 8},
        "endBinding": {"elementId": dst["id"], "focus": 0, "gap": 8},
        "startArrowhead": None, "endArrowhead": "arrow",
        "elbowed": False,
        "boundElements": [],
    })
    # Each card must ALSO name the arrow, or dragging the card leaves it behind.
    for card in (src, dst):
        card.setdefault("boundElements", [])
        card["boundElements"].append({"id": el["id"], "type": "arrow"})

    out = [el]
    label = " ".join(str(label or "").split())
    if label:
        lines = _wrap(label, 150, LINK_FS)
        w, h = _text_size(lines, LINK_FS)
        txt = "\n".join(lines)
        lab = _base("text", (x0 + x1) / 2 - w / 2, (y0 + y1) / 2 - h / 2, w, h)
        lab.update({
            "strokeColor": ACCENT,
            "text": txt, "originalText": txt,
            "fontSize": LINK_FS, "fontFamily": FONT_HAND,
            "textAlign": "center", "verticalAlign": "middle",
            "containerId": el["id"], "lineHeight": LINE_H,
            "baseline": round(LINK_FS * 0.8),
        })
        el["boundElements"].append({"id": lab["id"], "type": "text"})
        out.append(lab)
    return out


# ── the board ───────────────────────────────────────────────────────────────
def _node_positions(n: int, card_h: float) -> list[tuple[float, float]]:
    """Centre of each node, on an ellipse around the origin.

    The radius is derived, not fixed: adjacent points on a circle of radius R
    are 2*R*sin(pi/n) apart, so requiring that separation to clear one card
    plus MIN_NODE_GAP gives the R that guarantees no overlap for ANY n.
    """
    if n == 1:
        return [(0.0, 0.0)]
    need = max(CARD_W, card_h) + MIN_NODE_GAP

    # 2*R*sin(pi/n) is the adjacent separation on a CIRCLE, and we draw an
    # ELLIPSE. Dividing the vertical axis by RATIO shrinks every separation that
    # is not purely horizontal, by up to RATIO itself — so the derived radius
    # promised a clearance the layout did not deliver. At n=4 the slack absorbed
    # it; at n=5 two link labels landed on cards (caught by preview_board's
    # label/card check, 55x2 and 62x8 px).
    #
    # Rather than patch the formula with a fudge factor, place the nodes and
    # MEASURE the closest pair, then scale until it clears. One pass, exact for
    # any n and any RATIO, and it cannot drift out of agreement with the drawing.
    rx = max(MIN_RX, need / (2 * math.sin(math.pi / n)))
    if n == 2:
        # NOT 180/0. A dead-horizontal pair gives a 3.18:1 canvas — two cards in
        # a flat line, outside the readable range and a poor shape for a page.
        # A slight diagonal keeps the pair reading left-to-right while giving the
        # board some height.
        angles = [200.0, 20.0]
    else:
        angles = [-90.0 + i * 360.0 / n for i in range(n)]

    for _ in range(12):                 # bounded: each pass scales up, so it
                                        # terminates; the cap guards a bad RATIO
        ry = rx / RATIO
        pts = [(rx * math.cos(math.radians(a)), ry * math.sin(math.radians(a)))
               for a in angles]
        closest = min(math.dist(p, q)
                      for i, p in enumerate(pts) for q in pts[i + 1:])
        if closest >= need:
            return pts
        rx *= need / closest * 1.02     # 2% over, so float error cannot leave it
                                        # a hair short and loop again
    return pts


def build_scene(plan: dict, images: list[str | None]) -> dict:
    """plan + one image path per anchor (None where FLUX failed) -> a scene.

    A missing image is NOT fatal: the node is drawn with its label and its glyph
    written out in gray, so the board still shows what belongs there. Losing one
    pictogram must not cost the other three.
    """
    anchors = plan.get("anchors", [])
    if not anchors:
        raise ValueError("the plan has no anchors — nothing to draw")
    images = list(images) + [None] * (len(anchors) - len(images))

    # Notes are keyed by the anchor they annotate, so gather them per node
    # before laying anything out: a node's height depends on its notes.
    notes_by_anchor: dict[int, list[str]] = {}
    for note in plan.get("notes", []):
        idx = note.get("anchor")
        if isinstance(idx, int) and 0 <= idx < len(anchors):
            notes_by_anchor.setdefault(idx, []).append(note.get("text", ""))

    # ── pass 1: measure every node, so the ellipse can be sized ─────────────
    measured = []
    files: dict[str, dict] = {}
    for i, anchor in enumerate(anchors):
        path = images[i] if i < len(images) else None
        img_h = 0.0
        file_id = None
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                raw = fh.read()
            with Image.open(path) as im:
                iw, ih = im.size
            img_h = round(IMG_W * ih / iw)
            file_id = _new_id()
            files[file_id] = {
                "mimeType": "image/png", "id": file_id,
                "dataURL": "data:image/png;base64,"
                           + base64.b64encode(raw).decode("ascii"),
                "created": _now_ms(), "lastRetrieved": _now_ms(),
            }
        else:
            # No pictogram: reserve a shallow band and write the glyph there, so
            # the gap is self-explanatory rather than a mysterious empty card.
            img_h = 46.0

        label_lines = _wrap(anchor.get("label", ""), IMG_W, LABEL_FS)
        _, label_h = _text_size(label_lines, LABEL_FS)
        card_h = CARD_PAD + img_h + 12 + label_h + CARD_PAD

        note_lines: list[str] = []
        for text in notes_by_anchor.get(i, []):
            note_lines += _wrap(text, IMG_W, NOTE_FS)
        notes_h = _text_size(note_lines, NOTE_FS)[1] if note_lines else 0.0
        if note_lines:
            card_h += 10 + notes_h

        measured.append({
            "file_id": file_id, "img_h": img_h,
            "label_lines": label_lines, "label_h": label_h,
            "card_h": card_h, "note_lines": note_lines, "notes_h": notes_h,
            "glyph": anchor.get("glyph", ""),
        })

    tallest = max(m["card_h"] for m in measured)   # notes are inside the card
    centres = _node_positions(len(anchors), tallest)

    # ── pass 2: emit the elements ───────────────────────────────────────────
    elements: list[dict] = []
    cards: list[dict] = []
    for (cx, cy), m in zip(centres, measured):
        gid = _new_id()                       # groups card+image+label+notes so
                                              # one drag moves the whole node
        x = cx - CARD_W / 2
        y = cy - m["card_h"] / 2

        card = _card(x, y, CARD_W, m["card_h"], gid)
        cards.append(card)
        elements.append(card)

        inner_x = x + CARD_PAD
        if m["file_id"]:
            elements.append(_image(inner_x, y + CARD_PAD, IMG_W, m["img_h"],
                                   m["file_id"], gid))
        else:
            elements.append(_text(inner_x, y + CARD_PAD,
                                  _wrap(f"[{m['glyph']}]", IMG_W, NOTE_FS),
                                  NOTE_FS, NOTE_GRAY, gid, box_w=IMG_W))

        elements.append(_text(inner_x, y + CARD_PAD + m["img_h"] + 12,
                              m["label_lines"], LABEL_FS, INK, gid,
                              align="center", box_w=IMG_W))

        if m["note_lines"]:
            elements.append(_text(inner_x,
                                  y + m["card_h"] - CARD_PAD - m["notes_h"],
                                  m["note_lines"], NOTE_FS, NOTE_GRAY, gid,
                                  align="center", box_w=IMG_W))

    for link in plan.get("links", []):
        a, b = link.get("from"), link.get("to")
        if not (isinstance(a, int) and isinstance(b, int)):
            continue
        if not (0 <= a < len(cards) and 0 <= b < len(cards)) or a == b:
            continue                          # a bad index is a plan bug; drop
                                              # the arrow, keep the board
        elements += _arrow(cards[a], cards[b], link.get("label", ""))

    # ── normalise: shift everything to (PAD, PAD + TITLE_BAND) ──────────────
    # Done after the fact because the ellipse is built around the origin, which
    # puts half the board at negative coordinates.
    min_x = min(e["x"] for e in elements)
    min_y = min(e["y"] for e in elements)
    max_x = max(e["x"] + e["width"] for e in elements)
    max_y = max(e["y"] + e["height"] for e in elements)
    off_x, off_y = PAD - min_x, PAD + TITLE_BAND - min_y
    for e in elements:
        e["x"] = round(e["x"] + off_x, 2)
        e["y"] = round(e["y"] + off_y, 2)

    board_w = max_x - min_x
    title = " ".join(str(plan.get("title") or "").split())
    if title:
        # Wrapped to the width of the content it sits over, and centred on it —
        # never emitted as one long line that silently widens the canvas.
        lines = _wrap(title, board_w, TITLE_FS)
        _, h = _text_size(lines, TITLE_FS)
        elements.insert(0, _text(PAD, PAD + (TITLE_BAND - h) / 2, lines,
                                 TITLE_FS, INK, align="center", box_w=board_w))

    return {
        "type": "excalidraw", "version": 2,
        "source": "convograph/modules/graphic-generation/render_board.py",
        "elements": elements,
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": files,
        "_layout_debug": {
            "canvas": [round(board_w + 2 * PAD), round(max_y - min_y + 2 * PAD + TITLE_BAND)],
            "anchors": len(anchors),
            "images": sum(1 for m in measured if m["file_id"]),
            "links": sum(1 for e in elements if e["type"] == "arrow"),
            "notes": sum(len(v) for v in notes_by_anchor.values()),
        },
    }


# ── appending: one canvas, many meetings ────────────────────────────────────
def _scene_bbox(elements: list[dict]) -> tuple[float, float, float, float]:
    """Min/max over BOTH corners of every live element.

    Not build_scene's own formula. That one takes min(x) and max(x + width),
    which is wrong for arrows: _arrow() emits width = x1 - x0, so an arrow
    pointing left or up has a NEGATIVE extent and its far end falls outside
    the box. Harmless inside build_scene (arrow ends sit on cards, which are
    in the box anyway), but an append offset computed from a wrong edge puts
    the next meeting on top of this one. Same form as preview_board._bbox.
    """
    live = [e for e in elements if not e.get("isDeleted")]
    xs = [e["x"] for e in live] + [e["x"] + e.get("width", 0) for e in live]
    ys = [e["y"] for e in live] + [e["y"] + e.get("height", 0) for e in live]
    return min(xs), min(ys), max(xs), max(ys)


def _caption(x: float, y: float, text: str) -> dict:
    """The small gray line above a meeting block saying which meeting it is.

    append_scene() also writes the block's extent into this element's
    customData as `bbox` (caption included) the moment the block is placed.
    Coordinates never move afterwards — append_scene only ever shifts the NEW
    block — so a bbox recorded on an earlier append stays true for the life
    of the file. That record is what lets a later append centre on its
    neighbour or continue a grid row without re-deriving which elements
    belong to which meeting.
    """
    el = _text(x, y, [" ".join(str(text).split())], CAPTION_FS, NOTE_GRAY)
    el["customData"] = {CAPTION_KEY: {"text": el["text"]}}
    return el


def _has_caption(elements: list[dict]) -> bool:
    return any((e.get("customData") or {}).get(CAPTION_KEY) for e in elements)


def block_bounds(elements: list[dict]) -> list[list[float]]:
    """Per-block extents [x0, y0, x1, y1], from what each caption recorded.

    Falls back to ONE block covering everything when there are no captions (a
    board never appended to) or when any caption predates this bookkeeping
    (boards generated before it existed carry text only). Degrading to a single
    block keeps every placement rule well-defined on old files; it just means
    "the neighbour" is the whole canvas there.
    """
    recs = [(e.get("customData") or {}).get(CAPTION_KEY) for e in elements]
    recs = [m for m in recs if m]
    if not recs or any("bbox" not in m for m in recs):
        return [list(_scene_bbox(elements))]
    return [list(m["bbox"]) for m in recs]


def _rows(bounds: list[list[float]]) -> list[list[list[float]]]:
    """Group blocks into rows by vertical overlap, top row first, left to right.

    Derived from geometry rather than stored, so a canvas built with mixed
    directions still yields sensible rows for the grid rule.
    """
    rows: list[list[list[float]]] = []
    for b in sorted(bounds, key=lambda t: (t[1], t[0])):
        for row in rows:
            ry0, ry1 = min(r[1] for r in row), max(r[3] for r in row)
            overlap = min(b[3], ry1) - max(b[1], ry0)
            if overlap > 0.5 * min(b[3] - b[1], ry1 - ry0):
                row.append(b)
                break
        else:
            rows.append([b])
    for row in rows:
        row.sort(key=lambda t: t[0])
    return rows


def _target(bounds: list[list[float]], canvas: tuple[float, float, float, float],
            nw: float, nh: float, direction: str, gap: float, align: str,
            columns: int) -> tuple[float, float]:
    """Top-left corner for the new block's OCCUPIED box (caption included).

    Two rules, kept separate on purpose:

      placement   clears the WHOLE canvas in the chosen direction, so nothing
                  already drawn is ever overlapped;
      alignment   follows the NEIGHBOUR — the block nearest to where the new
                  one lands — so a column of meetings shares an edge or a
                  spine, rather than the far edge of some block placed elsewhere.

    Grid fills a row left to right, top-aligned to that row, then wraps to a
    new row below everything, back at the first row's left margin.
    """
    cx0, cy0, cx1, cy1 = canvas
    if direction == "grid":
        rows = _rows(bounds)
        last = rows[-1]
        if len(last) < columns:
            ref = max(last, key=lambda b: b[2])
            return ref[2] + gap, min(b[1] for b in last)
        return min(b[0] for b in rows[0]), cy1 + gap
    if direction in ("below", "above"):
        ref = (max(bounds, key=lambda b: b[3]) if direction == "below"
               else min(bounds, key=lambda b: b[1]))
        x = (ref[0] + ref[2]) / 2 - nw / 2 if align == "center" else ref[0]
        y = cy1 + gap if direction == "below" else cy0 - gap - nh
        return x, y
    ref = (max(bounds, key=lambda b: b[2]) if direction == "right"
           else min(bounds, key=lambda b: b[0]))
    y = (ref[1] + ref[3]) / 2 - nh / 2 if align == "center" else ref[1]
    x = cx1 + gap if direction == "right" else cx0 - gap - nw
    return x, y


DIRECTIONS = ("below", "above", "right", "left", "grid")
ALIGNS = ("start", "center")


def append_scene(base: dict, new: dict, direction: str = "below",
                 gap: float = APPEND_GAP,
                 base_caption: str | None = None,
                 new_caption: str | None = None,
                 align: str = "start", columns: int = 3) -> dict:
    """Place `new` (one meeting's board) onto `base` (everything so far).

    WHY THIS EXISTS
    ---------------
    build_scene() lays a board out around the origin and then normalises it to
    a fixed top-left corner, so every board lands in exactly the same place.
    A graphic recording is a canvas that GROWS as meetings happen; the next
    meeting has to be placed relative to what is already drawn.

    WHERE IT GOES
    -------------
    `direction` is below / above / right / left, or "grid": fill a row left to
    right up to `columns` blocks, then wrap to a new row. Placement always
    clears the whole canvas; `align` ("start" or "center") decides how the
    block lines up with its NEIGHBOUR on the other axis. Centring is on the
    neighbour, not on the canvas, so a column keeps a spine as it widens.
    Grid ignores `align`: a row is top-aligned, a new row starts at the first
    row's left margin.

    WHAT IT DOES NOT DO
    -------------------
    No arrows between meetings, and no re-layout of anything already on the
    canvas. Each meeting stays a separate block with its own title, plus a
    gray caption naming the meeting and recording the block's extent, so
    blocks stay attributable and placeable once there are ten of them.
    Cross-meeting supersession links are a later step, once fact identity is
    carried through to the cards.

    `base` may itself be the output of an earlier append (a chain). It is
    detected as already captioned via CAPTION_KEY and left alone; only the
    very first block ever gets a caption retroactively.

    Neither input is mutated, and no file is touched: the caller writes the
    merged scene into the NEW run's folder, never over the base.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if align not in ALIGNS:
        raise ValueError(f"align must be one of {ALIGNS}, got {align!r}")
    if direction == "grid" and columns < 1:
        raise ValueError(f"columns must be >= 1, got {columns}")

    # JSON round-trip as the deep copy: a scene must be JSON anyway, and the
    # orchestrator still holds `new` for its own debug numbers.
    base = json.loads(json.dumps(base))
    new = json.loads(json.dumps(new))

    # Deleted elements are tombstones Excalidraw keeps for undo; carrying them
    # would drag phantom boxes into the bbox. "index" is a fractional ordering
    # key a hand-saved file carries; mixing indexed and unindexed elements
    # makes Excalidraw fight the list order (the serpentine relayout hit this).
    base_els = [e for e in base.get("elements", []) if not e.get("isDeleted")]
    new_els = [e for e in new.get("elements", []) if not e.get("isDeleted")]
    for e in base_els + new_els:
        e.pop("index", None)
    if not base_els:
        raise ValueError("base scene has no live elements — nothing to append to")
    if not new_els:
        raise ValueError("new scene has no live elements — nothing to append")

    cap_h = _text_size(["x"], CAPTION_FS)[1]
    cap_gap = 6                          # caption bottom to block top

    bx0, by0, bx1, by1 = _scene_bbox(base_els)
    if base_caption and not _has_caption(base_els):
        cap = _caption(bx0, by0 - cap_h - cap_gap, base_caption)
        base_els.insert(0, cap)
        # Recorded AFTER insertion, so the first block's extent includes its
        # own caption the same way every later block's does.
        cap["customData"][CAPTION_KEY]["bbox"] = [round(v, 2) for v in _scene_bbox(base_els)]
    canvas = _scene_bbox(base_els)
    bounds = block_bounds(base_els)

    nx0, ny0, nx1, ny1 = _scene_bbox(new_els)
    reserve = (cap_h + cap_gap) if new_caption else 0.0
    # The new block is placed as its OCCUPIED box — content plus the caption
    # line above it — so the gap and the alignment hold for what a reader sees.
    ox, oy = _target(bounds, canvas, nx1 - nx0, (ny1 - ny0) + reserve,
                     direction, gap, align, columns)
    dx, dy = ox - nx0, (oy + reserve) - ny0
    for e in new_els:
        # Only x/y move. Arrow `points` are relative to the arrow's own x/y,
        # and bound labels are elements in this list, so they move with it.
        e["x"] = round(e["x"] + dx, 2)
        e["y"] = round(e["y"] + dy, 2)
    if new_caption:
        cap = _caption(ox, oy, new_caption)
        new_els.insert(0, cap)
        cap["customData"][CAPTION_KEY]["bbox"] = [round(v, 2) for v in _scene_bbox(new_els)]

    elements = base_els + new_els
    ids = [e["id"] for e in elements]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        # Ids are uuid4 here, so this only fires on a foreign or hand-copied
        # base. Refusing beats a canvas where dragging one card moves two.
        raise ValueError(f"duplicate element ids across base and new: {dupes}")

    files = {**(base.get("files") or {}), **(new.get("files") or {})}
    mx0, my0, mx1, my1 = _scene_bbox(elements)
    return {
        "type": "excalidraw", "version": 2,
        "source": new.get("source") or "convograph/modules/graphic-generation/render_board.py",
        "elements": elements,
        "appState": base.get("appState") or new.get("appState")
                    or {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": files,
        "_layout_debug": {
            "canvas": [round(mx1 - mx0 + 2 * PAD), round(my1 - my0 + 2 * PAD)],
            "blocks": sum(1 for e in elements
                          if (e.get("customData") or {}).get(CAPTION_KEY)),
            "direction": direction,
            "align": align if direction != "grid" else None,
            "columns": columns if direction == "grid" else None,
            "elements": len(elements),
            "files": len(files),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="board plan JSON")
    parser.add_argument("--images", default="",
                        help="comma-separated PNG per anchor, '' for a missing one")
    parser.add_argument("--out", default="board.excalidraw")
    args = parser.parse_args()

    with open(args.plan) as fh:
        plan = json.load(fh)
    images = [p.strip() or None for p in args.images.split(",")] if args.images else []

    scene = build_scene(plan, images)
    debug = scene.pop("_layout_debug")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(scene, fh, indent=2)
    print(f"wrote {args.out} — {debug['anchors']} anchors "
          f"({debug['images']} with a pictogram), {debug['links']} links, "
          f"{debug['notes']} notes, canvas {debug['canvas'][0]}x{debug['canvas'][1]}")


if __name__ == "__main__":
    main()
