#!/usr/bin/env python3
"""
build_board.py — RUNG 0 of Module 3: turn the knowledge graph into one board.

WHAT THIS IS
------------
The smallest thing that answers "can a temporal KG be drawn as a graphic
recording?". It produces a single static `.excalidraw` board. No incrementality,
no icons, no live input — those are later rungs. The point is to have a REAL
artifact to argue about instead of a hypothetical one.

WHY IT DRAWS FACTS, NOT THE ENTITY GRAPH
----------------------------------------
The obvious move is to render entities as boxes and RELATES_TO as arrows. On this
graph that produces a bad board, for a measured reason: 94 % of facts start at a
person and the targets are frequently bare dates ("2025-07-27", "EOD"). You would
get twelve person-blobs wired to a scatter of dates — an entity-relationship
diagram, and not a readable one.

The *fact sentences*, though, are good: "User_13 locks the 2025-07-27 baseline."
So the board's unit is a **statement**, not a node. That is also closer to what a
real graphic recording is — decisions and commitments written on a wall, not an
ER diagram.

WHAT GOES ON THE BOARD (the selection rule)
-------------------------------------------
A board holds 20-40 elements; the graph holds 111,258 facts. We do not summarise
with an LLM. We filter on something the graph already knows:

    a fact with `invalid_at` set is one the conversation LATER OVERTURNED.

Those are, by construction, the things people cared enough about to revisit. We
group them by the entity they are about, so each column reads as "what was said
about X, and how it changed".

THE POINT OF THE WHOLE DEMO
---------------------------
A superseded statement is drawn **struck through**, with the date it stopped
being true, and the statement that replaced it sits below in solid ink. That
single visual is the bi-temporal layer made visible. BM25 cannot represent it and
a diffusion model cannot show it — it is the one thing only this pipeline can do.

Writes: excalidraw/board/board.excalidraw
"""
from __future__ import annotations

import json
import os
import textwrap
from collections import defaultdict

from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://localhost:7688")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD", "graphiti123")
OUT = os.getenv("BOARD_OUT", "excalidraw/board/board.excalidraw")

# ── presentation comes from a config file, never from code ───────────────────
# How a board LOOKS gets re-tuned constantly, by people who are not editing
# Python. Hardcoding it here would mean a code change (and a review) for every
# colour tweak, and would make it impossible to compare two looks side by side.
STYLE_FILE = os.getenv("BOARD_STYLE_FILE", "styles/board_styles.json")
STYLE_NAME = os.getenv("BOARD_STYLE", "graphic-recording")

def load_style(name: str) -> dict:
    with open(STYLE_FILE) as fh:
        all_styles = json.load(fh)
    if name not in all_styles:
        avail = [k for k in all_styles if not k.startswith("_")]
        raise SystemExit(f"unknown style {name!r}; available: {avail}")
    return all_styles[name]

STYLE = load_style(STYLE_NAME)

# Geometry that does not vary by style.
COL_TOP, PAD = 250, 60
LH = 1.28
CHAR_W = 0.54                       # Nunito/Excalifont ≈ 0.54 em per character

INK, MUTED, STRIKE = "#2f3337", "#7a7f87", STYLE["strike_color"]
ROUGH = STYLE["roughness"]
FONT = STYLE["fontFamily"]
GAP_X, CARD_GAP = STYLE["gap_x"], STYLE["card_gap"]
SHAPE = STYLE["shape"]
ICONS = STYLE.get("icons", {})
HEAD = STYLE["column_header"]
CUR, SUP = STYLE["card"]["current"], STYLE["card"]["superseded"]
# The widest card decides the column, so a size hierarchy never overlaps a neighbour.
CARD_W = max(CUR["width"], SUP["width"])

PERSON_RE = r"(?i)^(user_\\d+|ops|compliance|risk|qa|finance|it|security|legal|product|ux|support|comms)\\s?(lead|owner|analyst|team)?$"

# A NAMED THING: starts capitalised and has at least two words. Only 13 % of our
# entities look like this — the other 80 % are sentence fragments ("owner",
# "drift", "data-engineering"), which make meaningless column headings. Requiring
# the named-thing shape is the cheapest available proxy for "a real topic".
NAMED_RE = r"^[A-Z][A-Za-z0-9]*( [A-Za-z0-9&/-]+)+$"

# Dates got extracted as entities too ("July 29", "July 18 target"), and they pass
# the named-thing test while being useless as a heading — a board column titled
# "July 19" tells the reader nothing about what was decided.
DATE_RE = r"(?i).*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\\d{4}-\\d{2}).*"

TOPICS = """
MATCH (a:Entity)-[r:RELATES_TO]-(:Entity)
WHERE r.fact IS NOT NULL
  AND a.name =~ $named_re
  AND NOT a.name =~ $person_re
  AND NOT a.name =~ $date_re
WITH a, count(r) AS total,
     sum(CASE WHEN r.invalid_at IS NOT NULL THEN 1 ELSE 0 END) AS superseded
WHERE superseded >= 2 AND total - superseded >= 2
RETURN a.name AS topic, total, superseded
ORDER BY superseded DESC, total DESC
LIMIT $n
"""

FACTS_SUPERSEDED = """
MATCH (a:Entity {name: $topic})-[r:RELATES_TO]-(:Entity)
WHERE r.fact IS NOT NULL AND r.invalid_at IS NOT NULL
RETURN r.fact AS fact, r.valid_at AS valid_at, r.invalid_at AS invalid_at,
       true AS superseded
ORDER BY r.invalid_at DESC LIMIT $k
"""

FACTS_CURRENT = """
MATCH (a:Entity {name: $topic})-[r:RELATES_TO]-(:Entity)
WHERE r.fact IS NOT NULL AND r.invalid_at IS NULL
RETURN r.fact AS fact, r.valid_at AS valid_at, r.invalid_at AS invalid_at,
       false AS superseded
ORDER BY r.valid_at DESC LIMIT $k
"""


def _wrap(text: str, width_px: float, fs: float) -> list[str]:
    """Wrap to the card's real width. A hand-placed break is right at one size only."""
    cols = max(16, int(width_px / (fs * CHAR_W)) - 1)
    # break_on_hyphens=False: the corpus is full of hyphenated domain terms
    # ("go/no-go", "full-path", "lineage-impacting") and splitting them mid-word
    # makes the card look like it contains typos.
    return textwrap.wrap(" ".join(text.split()), cols,
                         break_on_hyphens=False) or [""]


def _short_date(v) -> str:
    return str(v)[:10] if v else ""


def fetch(n_topics: int, per_topic: int):
    """Return [(topic, [facts])], each column mixing overturned and surviving."""
    n_sup = max(1, per_topic // 2)          # roughly half the column shows change
    n_cur = per_topic - n_sup
    drv = GraphDatabase.driver(URI, auth=(USER, PWD))
    with drv.session() as s:
        topics = [r.data() for r in s.run(TOPICS, n=n_topics, person_re=PERSON_RE,
                                    named_re=NAMED_RE, date_re=DATE_RE)]
        board = []
        for t in topics:
            sup = [r.data() for r in s.run(FACTS_SUPERSEDED, topic=t["topic"], k=n_sup)]
            cur = [r.data() for r in s.run(FACTS_CURRENT, topic=t["topic"], k=n_cur)]
            # Superseded first, then what replaced them: the column reads top-down
            # as "this was decided … and then this is what stands now".
            board.append((t["topic"], sup + cur))
    drv.close()
    return board


# ── excalidraw element helpers ───────────────────────────────────────────────
_seed = [1000]


def _base(kind, x, y, w, h, stroke, bg, **kw):
    _seed[0] += 1
    e = {"id": kw.pop("id", f"el{_seed[0]}"), "type": kind, "x": x, "y": y,
         "width": w, "height": h, "angle": 0, "strokeColor": stroke,
         "backgroundColor": bg, "fillStyle": "solid", "strokeWidth": 2,
         "strokeStyle": "solid", "roughness": ROUGH, "opacity": 100, "groupIds": [],
         "frameId": None, "roundness": {"type": 3} if kind == "rectangle" else None,
         "seed": _seed[0], "version": 1, "versionNonce": 1, "isDeleted": False,
         "boundElements": None, "updated": 1, "link": None, "locked": False}
    e.update(kw)
    return e


def _text(x, y, w, h, txt, fs, color, align="left", **kw):
    e = _base("text", x, y, w, h, color, "transparent", **kw)
    e.update(text=txt, originalText=txt, fontSize=fs, fontFamily=FONT,
             textAlign=align, verticalAlign="top", containerId=None, lineHeight=LH)
    return e


def build(board, out_path: str) -> None:
    els: list[dict] = []
    x = PAD
    col_bottoms = []

    for topic, facts in board:
        # ── column header: a HEADLINE word, hand-lettered and large ─────────
        # On a real board five to eight big words carry the structure. The header
        # is deliberately the biggest text on the column.
        els.append(_base(SHAPE, x, COL_TOP, CARD_W, HEAD["height"],
                         HEAD["stroke"], HEAD["bg"]))
        head_lines = _wrap(topic, CARD_W - 30, HEAD["size"])[:2]
        hh = len(head_lines) * HEAD["size"] * LH
        els.append(_text(x + 15, COL_TOP + (HEAD["height"] - hh) / 2, CARD_W - 30,
                         hh, "\n".join(head_lines), HEAD["size"], INK, "center"))

        y = COL_TOP + HEAD["height"] + CARD_GAP
        for f in facts:
            sup = f["superseded"]
            spec = SUP if sup else CUR
            # SIZE HIERARCHY — the main signal a graphic recording uses. A live
            # decision is drawn big; something the meeting moved past is small and
            # faded. Uniform cards are what made the first board read as a Kanban
            # wall rather than a poster.
            w, fs = spec["width"], spec["size"]
            cx = x + (CARD_W - w) / 2          # narrow cards stay centred in the column

            icon = ICONS.get("superseded" if sup else "current", "")
            body = f"{icon} {f['fact']}".strip() if icon else f["fact"]
            # 0.68 for an ellipse: the inscribed rectangle of an ellipse is about
            # 0.71 of its width, and a little less once several lines are stacked.
            inner_w = (w - 26) * (1.0 if SHAPE == "rectangle" else 0.68)
            lines = _wrap(body, inner_w, fs)
            body_h = len(lines) * fs * LH

            since = _short_date(f["valid_at"])
            stamp = (f"superseded {_short_date(f['invalid_at'])}" if sup
                     else (f"still true · since {since}" if since else "still true"))
            # An ellipse's usable area is NOT its bounding box. At the vertical
            # centre the full width is available, but a text block spanning most
            # of the height only has ~70 % of it — which is why the first attempt
            # spilled words out through the curved sides. So for an ellipse the
            # text is wrapped to 68 % of the width and the box is inflated around
            # the text, rather than the text being fitted to the box.
            pad_v = 34 if SHAPE == "rectangle" else int(body_h * 0.9) + 46
            card_h = body_h + pad_v

            els.append(_base(SHAPE, cx, y, w, card_h, spec["stroke"], spec["bg"],
                             strokeStyle=spec["style"]))
            tx = cx + (w - inner_w) / 2
            ty = y + (card_h - body_h - 18) / 2
            els.append(_text(tx, ty, inner_w, body_h, "\n".join(lines), fs,
                             spec["text"], "center"))
            els.append(_text(tx, ty + body_h + 4, inner_w, 14, stamp,
                             max(9.5, fs * 0.72), spec["stroke"], "center"))

            # THE money shot: a line actually drawn through an overturned statement.
            if sup:
                mid = ty + body_h / 2
                els.append(_base("line", tx, mid, inner_w, 0, STRIKE,
                                 "transparent", strokeWidth=2,
                                 points=[[0, 0], [inner_w, 0]],
                                 lastCommittedPoint=None, startBinding=None,
                                 endBinding=None, startArrowhead=None,
                                 endArrowhead=None))
            y += card_h + CARD_GAP
        col_bottoms.append(y)
        x += CARD_W + GAP_X

    width = x - GAP_X + PAD

    # ── title block; gaps derived from the type sizes, not magic numbers ────
    t, sub = STYLE["title"], STYLE["subtitle"]
    els.append(_text(PAD, 50, width - 2 * PAD, t["size"] * LH,
                     "What this meeting decided — and changed its mind about",
                     t["size"], t["color"]))
    y2 = 50 + t["size"] * LH + 16
    els.append(_text(PAD, y2, width - 2 * PAD, sub["size"] * LH,
                     "Generated from the temporal knowledge graph. "
                     "One column per topic; one card per statement.",
                     sub["size"], sub["color"]))
    els.append(_text(PAD, y2 + sub["size"] * LH + 10, width - 2 * PAD,
                     sub["size"] * LH,
                     "big + solid = still true          "
                     "small + faded + struck through = the meeting later overturned it",
                     sub["size"], sub["color"]))

    doc = {"type": "excalidraw", "version": 2,
           "source": "convograph/module-3 build_board.py",
           "elements": els,
           "appState": {"gridSize": 20, "viewBackgroundColor": "#ffffff"},
           "files": {}}

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    # A board on disk may have been hand-edited since it was generated, and that
    # edit is invisible from here. MOVE the old one into an archive beside it
    # before writing — never overwrite in place, never assume the previous file
    # was ours to discard.
    if os.path.exists(out_path):
        import shutil, datetime
        archive = os.path.join(out_dir, "archive")
        os.makedirs(archive, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base = os.path.basename(out_path).replace(".excalidraw", "")
        dest = os.path.join(archive, f"{base}_{stamp}_{STYLE_NAME}.excalidraw")
        shutil.move(out_path, dest)
        print(f"🗄️  archived previous board → {dest}")
    json.dump(doc, open(out_path, "w"), indent=2, ensure_ascii=False)

    height = max(col_bottoms) + PAD
    n_sup = sum(1 for _, fs in board for f in fs if f["superseded"])
    n_cur = sum(1 for _, fs in board for f in fs if not f["superseded"])
    print(f"✅ wrote {out_path}   [style: {STYLE_NAME}]")
    print(f"   {len(board)} topics · {n_cur} still-true · {n_sup} struck through")
    print(f"   canvas {width:.0f} x {height:.0f}  → aspect {width/height:.2f}:1")
    print(f"   roughness={ROUGH} font={FONT} shape={SHAPE} · elements {len(els)}")
    for topic, fs in board:
        sc = sum(1 for f in fs if f["superseded"])
        print(f"     · {topic[:44]:<46} {len(fs)} cards ({sc} superseded)")


if __name__ == "__main__":
    n = int(os.getenv("BOARD_TOPICS", "4"))
    k = int(os.getenv("BOARD_FACTS_PER_TOPIC", "5"))
    build(fetch(n, k), OUT)
