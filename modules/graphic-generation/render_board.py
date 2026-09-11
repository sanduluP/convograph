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
import io
import json
import math
import os
import random
import re
import textwrap
import time
import uuid

from PIL import Image

# The concept-icon vocabulary lives next door. Imported defensively: the
# renderer is also used by a pure-geometry test that has no icons on disk, and
# a board without icons is still a board.
try:
    from icons import concept_for as icon_concept
except Exception:                                              # noqa: BLE001
    def icon_concept(_text: str) -> str | None:                # type: ignore
        return None

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


def _register_icon(path: str, files: dict, cache: dict[str, str],
                   max_px: int = 128) -> str | None:
    """Embed a concept icon once and hand back its fileId.

    Two things happen here that do NOT happen for the anchor pictograms:

    * DEDUPLICATION. Thirty margin items share maybe eight concepts, and
      Excalidraw keys images by fileId — so the same icon must be embedded once
      and referenced thirty times. Embedding per item would put the same 300 KB
      into the scene eight times over.
    * DOWNSCALING. An icon is drawn at ICON_PX (~34 px) but FLUX returns 1024 px.
      Shipping the full-size PNG would inflate the .excalidraw file by an order
      of magnitude for pixels no one will ever see. The anchor glyphs keep their
      resolution because they ARE displayed large.
    """
    if path in cache:
        return cache[path]
    try:
        with Image.open(path) as im:
            im = im.convert("RGBA")
            im.thumbnail((max_px, max_px))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        raw = buf.getvalue()
    except Exception:                                          # noqa: BLE001
        return None        # a missing or broken icon costs an icon, not a board
    file_id = _new_id()
    files[file_id] = {
        "mimeType": "image/png", "id": file_id,
        "dataURL": "data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
        "created": _now_ms(), "lastRetrieved": _now_ms(),
    }
    cache[path] = file_id
    return file_id


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


# How far each successive arrow bows out of the straight line, as a fraction of
# its own length. Nodes sit on a ring, so several links are DIAMETERS and every
# one of them passes through the same centre point — where Excalidraw puts a
# contained label. Three labels landed on top of each other (observed
# 2026-09-11). Bowing each arrow by a different amount moves its midpoint, and
# therefore its label, off that shared crossing. A gentle curve also reads as
# hand-drawn, which is the look this board is after.
BOWS = (0.0, 0.07, -0.07, 0.12, -0.12, 0.17, -0.17)
                       # Halved once the ring grew to the panel column's height:
                       # bow is a fraction of LENGTH, so the same fraction on a
                       # longer arrow became a visible dogleg rather than a curve.


def _arrow(src: dict, dst: dict, label: str, bow: float = 0.0) -> list[dict]:
    """One labelled arrow between two cards, bound at both ends.

    Returns [arrow] or [arrow, label_text]. The label is a CONTAINED text
    (containerId = the arrow), which is how Excalidraw keeps a label centred on
    an arrow as the arrow moves; a free text element beside the arrow would
    stay behind the moment anything is dragged.

    `bow` curves the arrow sideways by that fraction of its length — see BOWS.
    """
    scx, scy = src["x"] + src["width"] / 2, src["y"] + src["height"] / 2
    dcx, dcy = dst["x"] + dst["width"] / 2, dst["y"] + dst["height"] / 2

    x0, y0 = _edge_point(scx, scy, src["width"], src["height"], dcx, dcy, 8)
    x1, y1 = _edge_point(dcx, dcy, dst["width"], dst["height"], scx, scy, 8)

    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1.0
    # Unit vector at right angles to the line, so the bow is sideways rather
    # than a change of length.
    perp_x, perp_y = -dy / length, dx / length
    mid_x = dx / 2 + perp_x * bow * length
    mid_y = dy / 2 + perp_y * bow * length
    points = ([[0, 0], [round(mid_x, 2), round(mid_y, 2)],
               [round(dx, 2), round(dy, 2)]] if bow
              else [[0, 0], [round(dx, 2), round(dy, 2)]])

    el = _base("arrow", x0, y0, x1 - x0, y1 - y0)
    el.update({
        "strokeColor": ARROW, "strokeWidth": 2,
        "roundness": {"type": 2},          # 2 = the curved multi-point style
        "points": points,
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
        # On the BOWED midpoint, not the straight one — otherwise the label sits
        # off its own arrow, and back on the pile it was bowed away from.
        lab = _base("text", x0 + mid_x - w / 2, y0 + mid_y - h / 2, w, h)
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
LAYOUTS = ("ring", "grid")
LAYOUT = os.getenv("BOARD_LAYOUT", "grid")
GRID_GAP = 90           # between two clusters in the grid layout


GRID_TARGET = 1.3       # width / height the column count is chosen to hit


def _grid_positions(heights: list[float], cell_w: float) -> list[tuple[float, float]]:
    """Cluster centres on a GRID, as the alternative to the ring.

    The ring's argument is that a set of linked concepts has no reading order, so
    do not imply one. The grid's argument is that it wastes no space: clusters
    are rectangles, and rectangles tile.

    Once every digest item hangs off its own anchor the clusters are TALL and
    UNEVEN, and that is what decides it. On a ring, spacing is set by the tallest
    cluster in every direction at once: one anchor carrying fourteen decisions
    pushed the canvas to 4008x3509 with voids across the middle. The same content
    on a grid is 1220x2568 — the grid simply does not care how tall its
    neighbour is.

    Both are kept so the question stays settled by looking rather than by
    arguing:

        BOARD_LAYOUT=grid bash scripts/run_content_map.sh …   (default)
        BOARD_LAYOUT=ring bash scripts/run_content_map.sh …

    Rows are TOP-ALIGNED, not centred: every anchor card in a row starts at the
    same y, so the pictograms read as a row and the stickies hang below them at
    whatever length each one needs. Centring them made the cards bob up and down
    by half the difference in their cluster heights.
    """
    n = len(heights)

    def shape(cols: int) -> tuple[float, float, list[float]]:
        """(width, height, row heights) for a given column count."""
        row_h = [max(heights[r * cols:(r + 1) * cols])
                 for r in range(math.ceil(n / cols))]
        return (cols * cell_w + (cols - 1) * GRID_GAP,
                sum(row_h) + (len(row_h) - 1) * GRID_GAP,
                row_h)

    # The column count is CHOSEN, not guessed at with sqrt(n): it is the one
    # whose resulting block comes closest to a landscape page. Cluster heights
    # vary by a factor of four here, so the arithmetic guess is routinely wrong.
    cols = min(range(1, n + 1),
               key=lambda c: abs(math.log((shape(c)[0] / shape(c)[1]) / GRID_TARGET)))
    width, height, row_h = shape(cols)

    # Where each row's cards start, as an offset from the block's top.
    row_top, acc = [], 0.0
    for h in row_h:
        row_top.append(acc)
        acc += h + GRID_GAP

    out = []
    for i, h in enumerate(heights):
        c, r = i % cols, i // cols
        # SERPENTINE: every other row runs right-to-left, the house rule for
        # pipeline diagrams and for the same reason here — the last cluster of a
        # row and the first of the next end up ABOVE AND BELOW each other, so the
        # link between them is a short drop instead of a long diagonal back
        # across the whole board.
        if r % 2:
            c = cols - 1 - c
        # Centred on the origin, the same convention the ring uses, so nothing
        # downstream has to know which layout produced the points. The y is the
        # row's top plus half this cluster's own height, because the caller
        # places a cluster by its centre.
        out.append(((c - (cols - 1) / 2) * (cell_w + GRID_GAP),
                    row_top[r] + h / 2 - height / 2))
    return out


def _node_positions(n: int, card_h: float, card_w: float = CARD_W,
                    min_height: float = 0.0) -> list[tuple[float, float]]:
    """Centre of each node, on an ellipse around the origin.

    The radius is derived, not fixed: adjacent points on a circle of radius R
    are 2*R*sin(pi/n) apart, so requiring that separation to clear one card
    plus MIN_NODE_GAP gives the R that guarantees no overlap for ANY n.

    `min_height` is how tall the ring should be ALLOWED to grow to. The margin
    panels are a tall column, and the ring is sized only by its node count — so
    six anchors sat in the top half of the canvas with a quarter of the board
    empty below them. Stretching the ring to the column's height fills that
    space with the content map instead of with nothing. It is a floor, not a
    target: a ring that is already taller is left alone.
    """
    if n == 1:
        return [(0.0, 0.0)]
    # card_w is passed now that a node is a CLUSTER — its card plus the stack of
    # digest stickies beneath it — and a cluster is wider than CARD_W.
    need = max(card_w, card_h) + MIN_NODE_GAP   # the starting radius only

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
    # Start SMALL and let the clearance test below grow it. The old start,
    # need / (2*sin(pi/n)), assumed clusters were circles of diameter `need`;
    # once `need` became a tall cluster's height that opening guess was already
    # past what the per-axis test would have asked for, so the loop exited on the
    # first pass and the ring never shrank back.
    rx = float(MIN_RX)
    if n == 2:
        # NOT 180/0. A dead-horizontal pair gives a 3.18:1 canvas — two cards in
        # a flat line, outside the readable range and a poor shape for a page.
        # A slight diagonal keeps the pair reading left-to-right while giving the
        # board some height.
        angles = [200.0, 20.0]
    else:
        angles = [-90.0 + i * 360.0 / n for i in range(n)]

    def place(rx: float, ry: float) -> list[tuple[float, float]]:
        return [(rx * math.cos(math.radians(a)), ry * math.sin(math.radians(a)))
                for a in angles]

    # Clearance is tested per AXIS, not as a straight-line distance. Two clusters
    # are rectangles: they miss each other as soon as they are far enough apart
    # HORIZONTALLY or far enough apart VERTICALLY, and requiring the diagonal to
    # exceed the larger of the two dimensions demands far more room than that.
    # With a tall cluster (an anchor carrying eight stickies) the difference is
    # not academic — it blew the canvas out to 3768x3339 with voids in it.
    def clears(pts: list[tuple[float, float]]) -> bool:
        for i, p in enumerate(pts):
            for q in pts[i + 1:]:
                if (abs(p[0] - q[0]) < card_w + MIN_NODE_GAP
                        and abs(p[1] - q[1]) < card_h + MIN_NODE_GAP):
                    return False
        return True

    for _ in range(24):                 # bounded: each pass scales up, so it
                                        # terminates; the cap guards a bad RATIO
        pts = place(rx, rx / RATIO)
        if clears(pts):
            break
        rx *= 1.12

    # The height stretch is applied AFTER rx has converged, never inside the
    # loop. Inside, a tall ry separates the vertical neighbours immediately, the
    # loop exits on the first pass, and rx stays at its floor — which turned a
    # 2391x2062 board into a 1605x2062 one: the ring filled the height by going
    # narrow, not by spreading out. Applied after, rx keeps the width it earned
    # and ry only grows, so every separation the loop guaranteed still holds.
    want_ry = max(0.0, (min_height - card_h) / 2)
    return place(rx, max(rx / RATIO, want_ry))


# ── satellites: the digest, attached to the map ─────────────────────────────
# The planner compresses: given 60 citable items it returns 6 anchors and 8
# notes and drops the rest. That is right for the MAP — a content map with
# twenty nodes is unreadable — but wrong for the board, because the other 46
# items are exactly the material a real graphic recording carries.
#
# They come STRAIGHT FROM THE DIGEST, not through the model. They used to be
# three panels down the right-hand margin, and Faris's verdict was exact: "a
# football field where only 11 players are playing and the others are sitting on
# the bench". A margin IS a bench.
#
# So every item is now a SATELLITE of the anchor it is about — drawn in the map,
# beneath the thing it concerns — and only the genuinely cross-cutting ones sit
# in a band along the bottom. One field, everybody on it.
SAT_W = 300             # one satellite sticky
SAT_FS = 12
SAT_PAD = 7             # sticky edge → its contents
SAT_GAP_Y = 8           # between two satellites in a stack
SAT_GAP_X = 22          # between two columns of satellites in a band
CLUSTER_GAP = 16        # anchor card → its first satellite
SAT_MAX_STACK_H = 820   # a stack taller than this wraps into another column
BAND_GAP = 50           # the map → the bands above and below it
BAND_TITLE_FS = 15

PANEL_TITLE_FS = 17
PANEL_PAD = 14          # container edge → its contents
ICON_PX = 44            # the concept icon on a sticky note. Big enough to read
                        # the object at a glance, small enough that the note is
                        # still mostly its sentence — the icon is a bookmark for
                        # the eye, not the content.
ICON_GAP = 9            # icon → its text

# "Still open" against "decided" is the distinction Faris asked to be able to see
# WITHOUT reading — so it is carried by the sticky's own colour, not by which
# list it was filed under. (stroke, fill)
SAT_STYLES = {
    "open":    ("#e8590c", "#fff4e6"),
    "decided": ("#2f9e44", "#ebfbee"),
}


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


def assign_to_anchors(items: list[tuple[str, str]],
                      labels: list[str]) -> tuple[dict[int, list], list]:
    """Which anchor each digest item belongs to. -> ({anchor index: items}, rest).

    Two rules, tried in order of how much they prove:

      1. the anchor's LABEL appears in the item — "Runbooks require monitoring"
         belongs under Runbook, and nothing else needs to be said;
      2. one of the anchor's CONCEPT keywords appears in it. "Ops" is the concept
         `ops`, whose keywords include "operations", so "Operations is to freeze
         the access-control change" lands on Ops even though the word "Ops" is
         nowhere in the sentence. Rule 1 alone left a third of the board homeless.

    The LONGEST match wins, so "…sign-off from Ops, Security, and Support" goes to
    Security rather than to Ops: a longer keyword is a more specific claim.

    An item that matches nothing is returned as a leftover, NOT forced onto the
    nearest anchor. It really is about the meeting as a whole, and the band along
    the bottom is where such an item is true.
    """
    try:
        from icons import VOCABULARY                            # noqa: PLC0415
    except Exception:                                           # noqa: BLE001
        VOCABULARY = {}

    # What each anchor answers to: its own label, plus the keywords of whichever
    # concept its label maps to.
    terms: list[list[str]] = []
    for label in labels:
        words = {" ".join(str(label).lower().split())}
        concept = icon_concept(label)
        if concept and concept in VOCABULARY:
            words |= {k.strip().lower() for k in VOCABULARY[concept][1]}
        terms.append(sorted((w for w in words if len(w) >= 3), key=len, reverse=True))

    by_anchor: dict[int, list] = {}
    leftovers: list = []
    for item in items:
        low = " ".join(str(item[0]).lower().split())
        # (score, index) for the best match that is not the sentence's subject,
        # and separately for the subject — see why below.
        best_i, best_len = None, 0
        subj_i = None
        for i, words in enumerate(terms):
            for word in words:
                if len(word) <= best_len:
                    break        # sorted longest-first: nothing here can win now
                # The same plural-tolerant word match icons.concept_for uses, so
                # "Runbooks" finds Runbook.
                m = re.search(rf"\b{re.escape(word)}s?\b", low)
                if not m:
                    continue
                # A match at position 0 is the sentence's SUBJECT — who did it —
                # and a graphic recording files a note under what it is ABOUT.
                # Every one of twenty decisions starts "ProjectManager …", so
                # taking the subject gave one anchor a column 1850 px tall and
                # left the others nearly empty. "ProjectManager freezes primary,
                # backup, and approver in the runbook" belongs under Runbook.
                if m.start() <= 1:
                    subj_i = i if subj_i is None else subj_i
                    continue
                best_i, best_len = i, len(word)
                break
        if best_i is None:
            best_i = subj_i          # nothing but the actor — then the actor it is
        if best_i is None:
            leftovers.append(item)
        else:
            by_anchor.setdefault(best_i, []).append(item)

    # Deliberately UNCAPPED. A cap sent the overflow to the band, where a sticky
    # that says "ProjectManager must lock primary/backup contacts" sat under a
    # heading claiming it belonged to no anchor — the heading was then a lie. The
    # band holds items that matched NOTHING, and only those; a talkative anchor
    # is allowed to be tall, which the grid absorbs in its row height.
    return by_anchor, leftovers


def measure_satellites(items: list[tuple[str, str]], icon_paths: dict[str, str],
                       max_h: float = SAT_MAX_STACK_H
                       ) -> tuple[list[list], float, float]:
    """Lay out a cluster's stickies without drawing them. -> (columns, w, h).

    Measured separately from drawing because the layout is sized around the
    CLUSTERS — anchor card plus its stickies — and that size is not known until
    the text has been wrapped.

    A stack taller than `max_h` WRAPS into a second column, and a third if it has
    to. One anchor can legitimately collect twenty items, and a single 1850 px
    column of them made the board a cliff with a void beside it; two columns of
    ten are the same information at half the height.
    """
    blocks = []
    for text, kind in items:
        # The icon is chosen from the ORIGINAL item text, not the URL-shortened
        # one: shortening throws away words the keyword match may need.
        concept = icon_concept(text)
        path = icon_paths.get(concept) if concept else None
        text_w = SAT_W - 2 * SAT_PAD - ((ICON_PX + ICON_GAP) if path else 0)
        lines = _wrap(_shorten_urls(text), text_w, SAT_FS)
        text_h = _text_size(lines, SAT_FS)[1]
        # A one-line sticky beside a 44 px icon has to grow to the icon's height,
        # or the icon overflows the sticky it sits in.
        h = max(text_h, ICON_PX if path else 0) + 2 * SAT_PAD
        blocks.append((lines, h, path, text_w, kind))

    if not blocks:
        return [], 0.0, 0.0

    # How many columns this stack needs, then an EVEN split across them — not
    # "fill the first column to max_h, then start the next", which leaves the
    # last column a stub.
    total = sum(b[1] + SAT_GAP_Y for b in blocks) - SAT_GAP_Y
    n_cols = max(1, math.ceil(total / max_h)) if max_h > 0 else 1
    per = math.ceil(len(blocks) / n_cols)
    columns = [blocks[i:i + per] for i in range(0, len(blocks), per)]

    width = len(columns) * SAT_W + (len(columns) - 1) * SAT_GAP_X
    height = max(sum(b[1] + SAT_GAP_Y for b in col) - SAT_GAP_Y
                 for col in columns)
    return columns, width, height


def draw_satellites(x: float, y: float, columns: list[list], files: dict,
                    icon_cache: dict, gid: str | None = None) -> list[dict]:
    """Draw measured columns with their top-left at (x, y). `gid` groups them
    with the anchor card, so dragging the node takes its stickies with it."""
    els: list[dict] = []
    for c, column in enumerate(columns):
        cx = x + c * (SAT_W + SAT_GAP_X)
        cursor = y
        for lines, h, path, text_w, kind in column:
            stroke, fill = SAT_STYLES.get(kind, SAT_STYLES["open"])
            note = _base("rectangle", cx, cursor, SAT_W, h, [gid] if gid else [])
            note.update({"backgroundColor": fill, "strokeColor": stroke,
                         "strokeWidth": 1, "roundness": {"type": 3},
                         "boundElements": []})
            els.append(note)

            text_x = cx + SAT_PAD
            if path:
                file_id = _register_icon(path, files, icon_cache)
                if file_id:
                    # Centred on the sticky's height so a three-line item does
                    # not leave its icon stranded at the top.
                    els.append(_image(text_x, cursor + (h - ICON_PX) / 2,
                                      ICON_PX, ICON_PX, file_id, gid))
                    text_x += ICON_PX + ICON_GAP
            text_h = _text_size(lines, SAT_FS)[1]
            els.append(_text(text_x, cursor + (h - text_h) / 2, lines,
                             SAT_FS, INK, gid, box_w=text_w))
            cursor += h + SAT_GAP_Y
    return els


def band(x: float, y: float, width: float, title: str,
         items: list[tuple[str, str]], icon_paths: dict[str, str],
         files: dict, icon_cache: dict) -> tuple[list[dict], float]:
    """The cross-cutting items, in COLUMNS along the full width of the board.

    Not a margin panel: a margin is a bench. These are the items that belong to
    the meeting rather than to any one anchor — "the July 19 cutover requires a
    confirmed staffed bridge roster" names nothing on the map — so they run the
    width of the field underneath it, in the same stickies and the same colours
    as the ones clipped to the nodes.
    """
    if not items:
        return [], 0.0
    els: list[dict] = []
    gid = _new_id()

    # One column per SAT_W the board is wide: the band runs the width of the
    # field, so it should use it.
    cols = max(1, int((width + SAT_GAP_X) // (SAT_W + SAT_GAP_X)))
    blocks, _, _ = measure_satellites(items, icon_paths, max_h=0)
    total = sum(b[1] + SAT_GAP_Y for b in blocks[0]) if blocks else 0.0
    columns, _, band_h = measure_satellites(
        items, icon_paths, max_h=max(1.0, total / cols))

    head_h = BAND_TITLE_FS * LINE_H + 8
    els.append(_text(x, y, [title], BAND_TITLE_FS, NOTE_GRAY, gid, box_w=width))
    els += draw_satellites(x, y + head_h, columns, files, icon_cache, gid)
    return els, head_h + band_h


# ── who was in the room ─────────────────────────────────────────────────────
# The participants used to be a list of "User_1 — 26%" strings on sticky notes,
# which is a table wearing a board's clothes. A graphic recorder draws the room:
# a face per person, sized and coloured, with how much they talked shown as a
# bar rather than spelled out. Nothing is lost — the percentage is still written
# — but the shape of the meeting is visible before a word is read.
#
# Drawn directly, never generated: FLUX is banned from drawing people (it
# produces uncanny faces and invented lettering), and an initialled disc is both
# cheaper and more legible than any portrait would be.
AVATAR_D = 52           # circle diameter
AVATAR_CELL_W = 78      # the column one avatar and its name occupy
AVATAR_NAME_FS = 11
AVATAR_BAR_H = 6        # the share bar under each name

# Seaborn "deep", in a FIXED order — the house palette (global rule 6). Fixed
# rather than cycled means the same person keeps their colour across a re-run,
# and muted rather than primary means five of them side by side do not shout.
AVATAR_COLORS = ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
                 "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd")


def _initials(name: str) -> str:
    """Up to two characters that identify a speaker on a 52 px disc.

    "User_1" -> "U1", "Priya Banta" -> "PB", "ProjectManager" -> "PM". The
    underscore/space split is what these transcripts actually use; a name that
    splits into nothing falls back to its first two characters rather than to an
    empty circle.
    """
    parts = [p for p in re.split(r"[\s_\-.]+", str(name).strip()) if p]
    if len(parts) >= 2:
        # A NUMERIC second part is kept whole. Taking its first character turned
        # User_1, User_12 and User_13 into three discs all reading "U1" — three
        # different people wearing the same badge, which is worse than no badge.
        if parts[1].isdigit():
            return (parts[0][0] + parts[1][:3]).upper()
        return (parts[0][0] + parts[1][0]).upper()
    if parts:
        word = parts[0]
        # A CamelCase single token still carries two capitals worth of identity.
        caps = re.findall(r"[A-Z]", word)
        return (caps[0] + caps[1]).upper() if len(caps) >= 2 else word[:2].upper()
    return "?"


def people_strip(x: float, y: float, people: list[dict], accent: str = "#1971c2",
                 tint: str = "#e7f5ff",
                 cols: int | None = None) -> tuple[list[dict], float, float]:
    """The room, as a strip of avatars. -> (elements, width, height).

    It used to be a panel down the right margin, four avatars per row. It is now
    ONE ROW across the top of the board, directly under the title: the people are
    not a footnote to the map, they are the room the map came out of. `cols`
    still wraps if a meeting ever has more faces than fit.
    """
    els: list[dict] = []
    gid = _new_id()
    cols = cols or len(people)
    rows = math.ceil(len(people) / cols)
    cell_w = AVATAR_CELL_W
    inner_w = cols * cell_w
    panel_w = inner_w + 2 * PANEL_PAD

    name_h = AVATAR_NAME_FS * LINE_H
    cell_h = AVATAR_D + 6 + name_h + 5 + AVATAR_BAR_H + 4 + name_h
    head_h = PANEL_TITLE_FS * LINE_H + 10
    total_h = PANEL_PAD + head_h + rows * cell_h + PANEL_PAD

    container = _base("rectangle", x, y, panel_w, total_h, [gid])
    container.update({"backgroundColor": tint, "strokeColor": accent,
                      "roundness": {"type": 3}, "boundElements": []})
    els.append(container)
    els.append(_text(x + PANEL_PAD, y + PANEL_PAD,
                     [f"◆ In the room  ({len(people)})"], PANEL_TITLE_FS,
                     accent, gid, box_w=inner_w))

    # Bars are scaled to the LOUDEST speaker, not to 100%: five people sharing a
    # meeting evenly all sit near 20%, and bars drawn against 100% would all be
    # stubs that show nothing. Against the maximum, the shape of the room reads.
    top = max((p.get("share") or 0) for p in people) or 1.0

    for i, person in enumerate(people):
        col, row = i % cols, i // cols
        cx = x + PANEL_PAD + col * cell_w + cell_w / 2
        cy = y + PANEL_PAD + head_h + row * cell_h
        colour = AVATAR_COLORS[i % len(AVATAR_COLORS)]

        disc = _base("ellipse", cx - AVATAR_D / 2, cy, AVATAR_D, AVATAR_D, [gid])
        disc.update({"backgroundColor": colour, "strokeColor": colour,
                     "fillStyle": "solid", "boundElements": []})
        els.append(disc)

        initials = _initials(person.get("speaker", ""))
        iw, ih = _text_size([initials], 20)
        els.append(_text(cx - AVATAR_D / 2, cy + (AVATAR_D - ih) / 2, [initials],
                         20, "#ffffff", gid, align="center", box_w=AVATAR_D))

        below = cy + AVATAR_D + 6
        els.append(_text(cx - cell_w / 2, below,
                         [" ".join(str(person.get("speaker", "")).split())[:12]],
                         AVATAR_NAME_FS, INK, gid, align="center", box_w=cell_w))

        share = person.get("share") or 0
        bar_w = cell_w - 18
        track = _base("rectangle", cx - bar_w / 2, below + name_h + 5,
                      bar_w, AVATAR_BAR_H, [gid])
        track.update({"backgroundColor": "#ffffff", "strokeColor": "#ced4da",
                      "strokeWidth": 1, "roundness": {"type": 3},
                      "boundElements": []})
        els.append(track)
        fill = _base("rectangle", cx - bar_w / 2, below + name_h + 5,
                     max(3.0, bar_w * share / top), AVATAR_BAR_H, [gid])
        fill.update({"backgroundColor": colour, "strokeColor": colour,
                     "fillStyle": "solid", "roundness": {"type": 3},
                     "boundElements": []})
        els.append(fill)

        els.append(_text(cx - cell_w / 2, below + name_h + 5 + AVATAR_BAR_H + 4,
                         [f"{100 * share:.0f}%"], AVATAR_NAME_FS, NOTE_GRAY,
                         gid, align="center", box_w=cell_w))
    return els, panel_w, total_h


def digest_items(digest: dict | None) -> list[tuple[str, str]]:
    """Every citable digest row as one (text, kind) pair, in ONE flat list.

    Open threads and decisions used to be two separate panels, which is how they
    came out of the cypher and not how a reader thinks: "what is still open about
    the runbook" and "what was decided about the runbook" belong side by side,
    under the runbook. They are one list now, and the SAME sticky colour that
    used to be the panel's colour still says which is which.

    Nothing is capped here. The old panels showed 10 of 15 and wrote "+ 5 more in
    digest/" — a board that admits it is hiding things is still hiding them.
    """
    if not digest:
        return []
    q = digest.get("queries", {})
    out: list[tuple[str, str]] = []
    for kind, rows in (("open", q.get("open_threads", [])),
                       ("decided", q.get("decisions", []))):
        for row in rows:
            fact = row.get("fact")
            if fact:
                out.append((" ".join(str(fact).split()), kind))
    return out


def resolve_icons(items: list[tuple[str, str]], log=None) -> dict[str, str]:
    """The cached icon path for every concept these items need, in ONE batch.

    Resolved before any drawing: the concepts are known from the text, so the
    (rare) generation of a missing one happens once for the whole board rather
    than once per cluster.
    """
    wanted = [c for c in (icon_concept(t) for t, _ in items) if c]
    if not wanted:
        return {}
    try:
        import icons as icon_vocab                              # noqa: PLC0415
        return icon_vocab.ensure(wanted, log=log)
    except Exception as exc:                                    # noqa: BLE001
        if log:
            log(f"   ⚠️  concept icons unavailable ({type(exc).__name__})"
                f" — the stickies will be text only")
        return {}


def build_scene(plan: dict, images: list[str | None],
                digest: dict | None = None, log=None) -> dict:
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

    # ── the digest, attached to the anchors ─────────────────────────────────
    # This is the change Faris asked for in as many words: the open threads and
    # the decisions used to be panels down the right-hand margin, and a margin is
    # a bench. Each item is routed to the anchor it is ABOUT and becomes a sticky
    # under that anchor's card; only the genuinely cross-cutting ones are left,
    # and they go in a band under the whole map. One field, everybody on it.
    items = digest_items(digest)
    icon_paths = resolve_icons(items, log=log)
    icon_cache: dict[str, str] = {}
    by_anchor, leftovers = assign_to_anchors(
        items, [a.get("label", "") for a in anchors])

    # Measured before anything is placed: a cluster is its card plus its stack,
    # and the layout is sized around clusters, not around cards.
    for i, m in enumerate(measured):
        cols, stack_w, stack_h = measure_satellites(by_anchor.get(i, []), icon_paths)
        m["sat_cols"], m["stack_w"] = cols, stack_w
        m["cluster_w"] = max(CARD_W, stack_w)
        m["cluster_h"] = m["card_h"] + ((CLUSTER_GAP + stack_h) if cols else 0)
    cluster_w = max(m["cluster_w"] for m in measured)
    tallest = max(m["cluster_h"] for m in measured)

    if LAYOUT == "grid":
        centres = _grid_positions([m["cluster_h"] for m in measured], cluster_w)
    else:
        centres = _node_positions(len(anchors), tallest, card_w=cluster_w)

    # ── pass 2: emit the elements ───────────────────────────────────────────
    elements: list[dict] = []
    cards: list[dict] = []
    for (cx, cy), m in zip(centres, measured):
        gid = _new_id()                       # groups card+image+label+notes AND
                                              # the satellites, so one drag moves
                                              # the whole cluster
        x = cx - CARD_W / 2
        # The CARD sits at the top of its cluster and the stickies hang below it,
        # so a cluster with eight satellites does not push its own card up out of
        # line with its neighbours' — the row of pictograms stays a row.
        y = cy - m["cluster_h"] / 2

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

        if m["sat_cols"]:
            elements += draw_satellites(cx - m["stack_w"] / 2,
                                        y + m["card_h"] + CLUSTER_GAP,
                                        m["sat_cols"], files, icon_cache, gid)

    drawn_links = 0                           # counts arrows ACTUALLY drawn, so
                                              # a dropped one does not consume a
                                              # bow and leave a gap in the fan
    for link in plan.get("links", []):
        a, b = link.get("from"), link.get("to")
        if not (isinstance(a, int) and isinstance(b, int)):
            continue
        if not (0 <= a < len(cards) and 0 <= b < len(cards)) or a == b:
            continue                          # a bad index is a plan bug; drop
                                              # the arrow, keep the board
        elements += _arrow(cards[a], cards[b], link.get("label", ""),
                           bow=BOWS[drawn_links % len(BOWS)])
        drawn_links += 1

    # ── the two bands, above and below the map ──────────────────────────────
    # Both span the map rather than sitting beside it. Placed now, because both
    # want the map's real bounding box and neither changes it.
    map_left = min(e["x"] for e in elements)
    map_right = max(e["x"] + e["width"] for e in elements)
    map_top = min(e["y"] for e in elements)
    map_bottom = max(e["y"] + e["height"] for e in elements)
    map_w = map_right - map_left

    band_els: list[dict] = []

    # The room, as one row directly under the title — the people are not a
    # footnote to the map, they are the room the map came out of.
    people = (digest or {}).get("queries", {}).get("participants", [])
    if people:
        # Wrapped only if a meeting ever brings more faces than the map is wide.
        cols = max(1, min(len(people), int(map_w // AVATAR_CELL_W)))
        strip, strip_w, strip_h = people_strip(0, 0, people, cols=cols)
        dx = map_left + (map_w - strip_w) / 2
        dy = map_top - BAND_GAP - strip_h
        for e in strip:
            e["x"], e["y"] = round(e["x"] + dx, 2), round(e["y"] + dy, 2)
        band_els += strip

    # What belongs to the meeting rather than to any one anchor.
    if leftovers:
        rest, rest_h = band(
            map_left, map_bottom + BAND_GAP, map_w,
            f"◆ Across the board  ({len(leftovers)})",
            leftovers, icon_paths, files, icon_cache)
        band_els += rest

    elements += band_els

    if log:
        placed = sum(len(v) for v in by_anchor.values())
        log(f"   🧲 {placed}/{len(items)} digest item(s) clipped to an anchor, "
            f"{len(leftovers)} in the band below — "
            f"{len(icon_cache)} distinct icon(s), {LAYOUT} layout")

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
            "satellites": sum(len(v) for v in by_anchor.values()),
            "band": len(leftovers),
            "layout": LAYOUT,
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
