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

# ── board geometry ───────────────────────────────────────────────────────────
CARD_W, GAP_X = 330, 46
COL_TOP, HEADER_H = 210, 46
CARD_GAP, PAD = 18, 60
FS_CARD, FS_HEAD, LH = 13, 16, 1.28
CHAR_W = 0.54                       # Nunito ≈ 0.54 em per char, measured

# ── palette: muted, and status colours are RESERVED ─────────────────────────
INK        = "#2f3337"
MUTED      = "#7a7f87"
CUR_BG, CUR_ST   = "#e7f5ff", "#1971c2"   # still true
SUP_BG, SUP_ST   = "#f1f3f5", "#adb5bd"   # superseded — deliberately drained
HEAD_BG, HEAD_ST = "#fff3bf", "#f08c00"   # the topic this column is about
STRIKE     = "#e03131"                     # reserved status colour: struck through

# A column heading must be a TOPIC, not a person. Left to itself the query picks
# User_13, User_2 … because 94 % of facts start at a speaker — the star graph
# leaking straight onto the board. So people are excluded from the heading role
# here (they remain visible inside the sentences, where they belong).
#
# A topic also has to have CHANGED: a column with nothing superseded has no story
# to tell, and the whole point of the board is showing the meeting change its mind.
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
         "strokeStyle": "solid", "roughness": 0, "opacity": 100, "groupIds": [],
         "frameId": None, "roundness": {"type": 3} if kind == "rectangle" else None,
         "seed": _seed[0], "version": 1, "versionNonce": 1, "isDeleted": False,
         "boundElements": None, "updated": 1, "link": None, "locked": False}
    e.update(kw)
    return e


def _text(x, y, w, h, txt, fs, color, align="left", **kw):
    e = _base("text", x, y, w, h, color, "transparent", **kw)
    e.update(text=txt, originalText=txt, fontSize=fs, fontFamily=6,
             textAlign=align, verticalAlign="top", containerId=None, lineHeight=LH)
    return e


def build(board, out_path: str) -> None:
    els: list[dict] = []
    x = PAD
    col_bottoms = []

    for topic, facts in board:
        # ── column header: the thing this column is about ───────────────────
        els.append(_base("rectangle", x, COL_TOP, CARD_W, HEADER_H, HEAD_ST, HEAD_BG))
        head = _wrap(topic, CARD_W - 24, FS_HEAD)[:1]
        els.append(_text(x + 12, COL_TOP + 13, CARD_W - 24, 20, head[0],
                         FS_HEAD, INK, "center"))

        y = COL_TOP + HEADER_H + CARD_GAP
        for f in facts:
            sup = f["superseded"]
            lines = _wrap(f["fact"], CARD_W - 26, FS_CARD)
            body_h = len(lines) * FS_CARD * LH
            # Not every fact carries valid_at; "still true · since " with nothing
            # after it reads as a rendering bug rather than missing data.
            since = _short_date(f["valid_at"])
            stamp = (f"superseded {_short_date(f['invalid_at'])}" if sup
                     else (f"still true · since {since}" if since else "still true"))
            card_h = body_h + 34

            els.append(_base("rectangle", x, y, CARD_W, card_h,
                             SUP_ST if sup else CUR_ST,
                             SUP_BG if sup else CUR_BG,
                             strokeStyle="dashed" if sup else "solid",
                             strokeWidth=2 if sup else 2))
            els.append(_text(x + 13, y + 10, CARD_W - 26, body_h, "\n".join(lines),
                             FS_CARD, MUTED if sup else INK))
            els.append(_text(x + 13, y + 10 + body_h + 4, CARD_W - 26, 14, stamp,
                             10.5, SUP_ST if sup else MUTED))

            # THE money shot: a real line drawn through an overturned statement.
            if sup:
                mid = y + 10 + body_h / 2
                els.append(_base("line", x + 10, mid, CARD_W - 20, 0, STRIKE,
                                 "transparent", strokeWidth=2,
                                 points=[[0, 0], [CARD_W - 20, 0]],
                                 lastCommittedPoint=None, startBinding=None,
                                 endBinding=None, startArrowhead=None,
                                 endArrowhead=None))
            y += card_h + CARD_GAP
        col_bottoms.append(y)
        x += CARD_W + GAP_X

    width = x - GAP_X + PAD

    # ── title block: bold headline, recessive grey subtitle, legend ─────────
    els.append(_text(PAD, 46, width - 2 * PAD, 34,
                     "What this meeting decided — and changed its mind about",
                     26, INK))
    els.append(_text(PAD, 92, width - 2 * PAD, 20,
                     "Generated from the temporal knowledge graph. "
                     "One column per topic; one card per statement.", 13.5, MUTED))
    els.append(_text(PAD, 132, width - 2 * PAD, 20,
                     "solid = still true          "
                     "greyed + struck through = the meeting later overturned it",
                     13.5, MUTED))

    doc = {"type": "excalidraw", "version": 2,
           "source": "convograph/module-3 build_board.py",
           "elements": els,
           "appState": {"gridSize": 20, "viewBackgroundColor": "#ffffff"},
           "files": {}}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):                 # never overwrite a board in place
        import shutil, datetime
        os.makedirs("excalidraw/archive", exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        shutil.copy(out_path, f"excalidraw/archive/board_{stamp}.excalidraw")
    json.dump(doc, open(out_path, "w"), indent=2, ensure_ascii=False)

    height = max(col_bottoms) + PAD
    n_sup = sum(1 for _, fs in board for f in fs if f["superseded"])
    n_cur = sum(1 for _, fs in board for f in fs if not f["superseded"])
    print(f"✅ wrote {out_path}")
    print(f"   {len(board)} topics · {n_cur} still-true cards · {n_sup} struck through")
    print(f"   canvas {width:.0f} x {height:.0f}  → aspect {width/height:.2f}:1")
    print(f"   elements {len(els)}")
    for topic, fs in board:
        s = sum(1 for f in fs if f["superseded"])
        print(f"     · {topic[:44]:<46} {len(fs)} cards ({s} superseded)")


if __name__ == "__main__":
    n = int(os.getenv("BOARD_TOPICS", "4"))
    k = int(os.getenv("BOARD_FACTS_PER_TOPIC", "5"))
    build(fetch(n, k), OUT)
