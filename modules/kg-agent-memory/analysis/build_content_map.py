#!/usr/bin/env python3
"""
build_content_map.py — draw the knowledge graph as a CONTENT MAP.

WHAT A CONTENT MAP IS
---------------------
CHI 2021 (Zheng et al.) classifies real sketchnotes into four types. One of them:

    Content maps illustrate the different topics and points made in the talk akin
    to a visual non-linear table of content. Key concepts are often illustrated
    with icons or diagrams, and visually connected with lines and arrows.

That is a knowledge graph. So rather than inventing a layout, we target a named
type from a published taxonomy — which also gives the paper a design rationale.

WHY THIS WAS IMPOSSIBLE UNTIL YESTERDAY
---------------------------------------
A content map is *defined* by lines between concepts. The original extraction
produced 433 concept→concept edges out of 111,258 (0.4 %) — everything else ran
through a speaker. Drawing it gave twelve person-blobs wired to bare dates, which
is why the first board fell back to rendering fact SENTENCES in columns.

Excluding speakers from the node set changed that: 60 windows of the fixed
extraction yield **462** concept→concept edges — more than the entire 30,000
message corpus produced before. The map became drawable, so we draw it.

HOW IT DIFFERS FROM build_board.py
----------------------------------
    build_board.py       node = a fact sentence, grouped in columns  → a list
    build_content_map.py node = a CONCEPT, edge = the fact between   → a map

The fact moves from being the card's text to being the label on the arrow. That
is the whole difference between a list of statements and a map of a domain.

Input:  tmp/concept_edges_speaker_free.csv  (scripts/export_concept_edges.sh)
Output: excalidraw/board/content_map.excalidraw
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
import shutil
import sys
import textwrap
from collections import defaultdict

CSV_IN = os.getenv("EDGES_CSV", "../../tmp/concept_edges_speaker_free.csv")
OUT = os.getenv("MAP_OUT", "excalidraw/board/content_map.excalidraw")
STYLE_FILE = os.getenv("BOARD_STYLE_FILE", "styles/board_styles.json")
STYLE_NAME = os.getenv("BOARD_STYLE", "graphic-recording")
N_CONCEPTS = int(os.getenv("MAP_CONCEPTS", "14"))

STYLE = json.load(open(STYLE_FILE))[STYLE_NAME]
ROUGH, FONT = STYLE["roughness"], STYLE["fontFamily"]
INK, MUTED = "#2f3337", "#7a7f87"
CUR, SUP = STYLE["card"]["current"], STYLE["card"]["superseded"]
HEAD = STYLE["column_header"]
CHAR_W, LH = 0.54, 1.25

_seed = [7000]


def _el(kind, x, y, w, h, stroke, bg, **kw):
    _seed[0] += 1
    e = {"id": kw.pop("id", f"m{_seed[0]}"), "type": kind, "x": x, "y": y,
         "width": w, "height": h, "angle": 0, "strokeColor": stroke,
         "backgroundColor": bg, "fillStyle": "solid", "strokeWidth": 2,
         "strokeStyle": "solid", "roughness": ROUGH, "opacity": 100,
         "groupIds": [], "frameId": None,
         "roundness": {"type": 3} if kind == "rectangle" else None,
         "seed": _seed[0], "version": 1, "versionNonce": 1, "isDeleted": False,
         "boundElements": None, "updated": 1, "link": None, "locked": False}
    e.update(kw)
    return e


def _text(x, y, w, h, txt, fs, color, align="center"):
    e = _el("text", x, y, w, h, color, "transparent")
    e.update(text=txt, originalText=txt, fontSize=fs, fontFamily=FONT,
             textAlign=align, verticalAlign="top", containerId=None, lineHeight=LH)
    return e


def _wrap(t, w_px, fs):
    return textwrap.wrap(" ".join(t.split()), max(8, int(w_px / (fs * CHAR_W))),
                         break_on_hyphens=False) or [""]


def load_edges(path):
    """Read the exported edges.

    cypher-shell's `--format plain` writes ", " after each comma, so a plain
    DictReader yields keys like " target" with a leading space and every lookup
    silently returns None. skipinitialspace fixes the values; the keys are
    stripped explicitly because DictReader builds them before that applies.
    """
    rows = []
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh, skipinitialspace=True)
        rdr.fieldnames = [f.strip() for f in (rdr.fieldnames or [])]
        for r in rdr:
            src = (r.get("source") or "").strip().strip('"')
            tgt = (r.get("target") or "").strip().strip('"')
            rel = (r.get("relation") or "").strip().strip('"')
            if not src or not tgt or src == tgt:
                continue
            rows.append({"src": src, "tgt": tgt, "rel": rel,
                         "superseded": "TRUE" in (r.get("superseded") or "").upper()})
    return rows


def pick_subgraph(edges, n):
    """Keep the n best-connected concepts and the edges among them.

    A map of 5,000 nodes is a hairball. Degree is the cheapest defensible way to
    say "these are the topics the conversation kept returning to", and it needs no
    domain knowledge — unlike the name-shaped filters the list-board needed.
    """
    deg = defaultdict(int)
    for e in edges:
        deg[e["src"]] += 1
        deg[e["tgt"]] += 1
    keep = {k for k, _ in sorted(deg.items(), key=lambda kv: -kv[1])[:n]}
    sub, seen = [], set()
    for e in edges:
        if e["src"] in keep and e["tgt"] in keep:
            k = (e["src"], e["tgt"])
            if k in seen:            # one arrow per pair; parallel edges are noise
                continue
            seen.add(k)
            sub.append(e)
    return keep, sub, deg


def build():
    edges = load_edges(CSV_IN)
    keep, sub, deg = pick_subgraph(edges, N_CONCEPTS)
    if not sub:
        sys.exit("no edges survived the subgraph selection")

    # RADIAL placement: the most-connected concept sits at the centre, the rest
    # ring it, ordered by degree. Non-linear (the taxonomy's word) but fully
    # deterministic — a node keeps its position across renders, which a
    # force-directed layout could not promise.
    ranked = sorted(keep, key=lambda k: -deg[k])
    centre, ring = ranked[0], ranked[1:]
    CX, CY, R = 900.0, 760.0, 520.0
    pos = {centre: (CX, CY)}
    for i, name in enumerate(ring):
        a = 2 * math.pi * i / max(1, len(ring)) - math.pi / 2
        # Slight radius alternation stops long labels in adjacent slots colliding.
        r = R * (1.0 if i % 2 == 0 else 0.74)
        pos[name] = (CX + r * math.cos(a), CY + r * math.sin(a))

    els = []

    # ── edges first, so nodes paint over the arrow ends ─────────────────────
    for e in sub:
        (x1, y1), (x2, y2) = pos[e["src"]], pos[e["tgt"]]
        spec = SUP if e["superseded"] else CUR
        els.append(_el("arrow", x1, y1, x2 - x1, y2 - y1,
                       spec["stroke"], "transparent",
                       strokeWidth=2 if e["superseded"] else 2,
                       strokeStyle="dashed" if e["superseded"] else "solid",
                       points=[[0, 0], [x2 - x1, y2 - y1]],
                       lastCommittedPoint=None, startBinding=None,
                       endBinding=None, startArrowhead=None, endArrowhead="arrow"))
        # The relation reads as the map's connective tissue: "X —CONTRIBUTES_TO→ Y".
        label = e["rel"].replace("_", " ").lower()[:26]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        els.append(_text(mx - 80, my - 9, 160, 15, label, 11,
                         spec["stroke"] if e["superseded"] else MUTED))

    # ── concept nodes, sized by how connected they are ──────────────────────
    for name in ranked:
        x, y = pos[name]
        d = deg[name]
        fs = 17 if name == centre else 13
        w = 250 if name == centre else 190
        lines = _wrap(name, w - 34, fs)[:3]
        h = max(70, len(lines) * fs * LH + 46)
        els.append(_el("ellipse", x - w / 2, y - h / 2, w, h,
                       HEAD["stroke"] if name == centre else CUR["stroke"],
                       HEAD["bg"] if name == centre else CUR["bg"],
                       strokeWidth=3 if name == centre else 2))
        th = len(lines) * fs * LH
        els.append(_text(x - w / 2 + 17, y - th / 2 - 5, w - 34, th,
                         "\n".join(lines), fs, INK))
        els.append(_text(x - w / 2 + 17, y + th / 2 + 1, w - 34, 13,
                         f"{d} links", 10, MUTED))

    # ── title block ─────────────────────────────────────────────────────────
    t, sub_s = STYLE["title"], STYLE["subtitle"]
    els.append(_text(60, 40, 1300, t["size"] * LH,
                     "What this project is actually about", t["size"], t["color"], "left"))
    els.append(_text(60, 40 + t["size"] * LH + 14, 1300, sub_s["size"] * LH,
                     f"A content map of the temporal knowledge graph — "
                     f"{len(keep)} concepts, {len(sub)} relationships between them.",
                     sub_s["size"], sub_s["color"], "left"))
    els.append(_text(60, 40 + t["size"] * LH + sub_s["size"] * LH + 26, 1300,
                     sub_s["size"] * LH,
                     "solid = still true        dashed = the project later moved on from it",
                     sub_s["size"], sub_s["color"], "left"))

    doc = {"type": "excalidraw", "version": 2,
           "source": "convograph/module-3 build_content_map.py",
           "elements": els,
           "appState": {"gridSize": 20, "viewBackgroundColor": "#ffffff"},
           "files": {}}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    if os.path.exists(OUT):
        arch = os.path.join(os.path.dirname(OUT), "archive")
        os.makedirs(arch, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        shutil.move(OUT, os.path.join(arch, f"content_map_{stamp}.excalidraw"))
        print(f"🗄️  archived previous map")
    json.dump(doc, open(OUT, "w"), indent=2, ensure_ascii=False)

    xs = [e["x"] for e in els]; ys = [e["y"] for e in els]
    X = [e["x"] + e.get("width", 0) for e in els]
    Y = [e["y"] + e.get("height", 0) for e in els]
    w_, h_ = max(X) - min(xs), max(Y) - min(ys)
    print(f"✅ wrote {OUT}")
    print(f"   {len(keep)} concepts · {len(sub)} relationships · {len(els)} elements")
    print(f"   canvas {w_:.0f} x {h_:.0f} → aspect {w_/h_:.2f}:1")
    print(f"   centre: {centre}  ({deg[centre]} links)")
    print("   ring:")
    for n in ranked[1:7]:
        print(f"     · {n[:52]:<54} {deg[n]} links")


if __name__ == "__main__":
    build()
