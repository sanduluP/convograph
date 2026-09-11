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
import re
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


# ── side panels ─────────────────────────────────────────────────────────────
# The planner compresses: given 60 citable items it returns 6 anchors and 8
# notes and drops the rest. That is right for the MAP — a content map with
# twenty nodes is unreadable — but wrong for the board, because the other 46
# items are exactly the material a real graphic recording carries in its
# margins: what was decided, what is still open, who was in the room.
#
# So those come STRAIGHT FROM THE DIGEST, not through the model. No selection
# step means nothing is lost to compression, and the LLM is left doing the one
# job only it can do: deciding what relates to what.
PANEL_W = 300
PANEL_TITLE_FS = 16
PANEL_ITEM_FS = 12
PANEL_GAP = 26


# A SharePoint path is ~90 characters of which the last word is the only part
# that identifies anything. textwrap cannot break it — there are no spaces — so
# the line runs straight past the panel edge. Shortened to the leaf, which is
# what a reader would say aloud anyway.
_URL_RE = re.compile(r"https?://\S+")


def _shorten_urls(text: str) -> str:
    def leaf(m: re.Match) -> str:
        tail = m.group(0).rstrip("/").rsplit("/", 1)[-1]
        return tail[:48] if tail else m.group(0)[:48]
    return _URL_RE.sub(leaf, text)


def _panel(x: float, y: float, title: str, items: list[str],
           accent: str, max_items: int = 10) -> tuple[list[dict], float]:
    """One margin panel. Returns (elements, height consumed)."""
    els: list[dict] = []
    gid = _new_id()
    head = _text(x, y, [title], PANEL_TITLE_FS, accent, gid)
    els.append(head)
    cursor = y + PANEL_TITLE_FS * LINE_H + 8

    shown = [_shorten_urls(i) for i in items[:max_items]]
    for item in shown:
        lines = _wrap(f"• {item}", PANEL_W, PANEL_ITEM_FS)
        h = _text_size(lines, PANEL_ITEM_FS)[1]
        els.append(_text(x, cursor, lines, PANEL_ITEM_FS, NOTE_GRAY, gid,
                         box_w=PANEL_W))
        cursor += h + 6
    if len(items) > max_items:
        # Say what was cut. A panel that silently shows 10 of 15 is a panel that
        # lies about how much is outstanding.
        more = [f"+ {len(items) - max_items} more in digest/"]
        els.append(_text(x, cursor, more, PANEL_ITEM_FS, accent, gid,
                         box_w=PANEL_W))
        cursor += PANEL_ITEM_FS * LINE_H + 6
    return els, cursor - y


def build_scene(plan: dict, images: list[str | None],
                digest: dict | None = None) -> dict:
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

    # ── margin panels, straight from the digest ─────────────────────────────
    # Placed to the RIGHT of the ring, after the nodes are measured, so they
    # never collide with it: the ring's width is known by now and the panels
    # start past it.
    panel_els: list[dict] = []
    if digest:
        q = digest.get("queries", {})
        ring_right = max(e["x"] + e["width"] for e in elements)
        px = ring_right + PANEL_GAP * 2
        py = min(e["y"] for e in elements)

        for title, rows, key, accent in (
            ("Still open", q.get("open_threads", []), "fact", "#e8590c"),
            ("Decided", q.get("decisions", []), "fact", "#2f9e44"),
        ):
            if not rows:
                continue
            items = [r.get(key, "") for r in rows if r.get(key)]
            els, h = _panel(px, py, f"{title}  ({len(items)})", items, accent)
            panel_els += els
            py += h + PANEL_GAP

        people = q.get("participants", [])
        if people:
            items = [f"{p['speaker']} — {100 * p['share']:.0f}%" for p in people]
            els, h = _panel(px, py, f"In the room  ({len(items)})", items,
                            "#1971c2", max_items=12)
            panel_els += els
    elements += panel_els

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
            "panels": len(panel_els),
            "anchors": len(anchors),
            "images": sum(1 for m in measured if m["file_id"]),
            "links": sum(1 for e in elements if e["type"] == "arrow"),
            "notes": sum(len(v) for v in notes_by_anchor.values()),
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
