#!/usr/bin/env python3
"""
compose_board.py — place N generated (image, caption) pairs on ONE Excalidraw
canvas, positioned by a grid cursor so nothing overlaps and reading order
follows decision time — not on top of each other at (0,0), which is what you'd
get from calling image_to_excalidraw.py once per image into the same file.

LAYOUT ALGORITHM
----------------
Adapted from modules/kg-agent-memory/analysis/build_board.py's column layout:
that script places fact-cards with two additive cursors (x accumulates one
column's width per topic, y accumulates one card's height per fact) so
non-overlap falls out of arithmetic, not manual coordinates. This does the same
thing for images, but as a wrapping GRID instead of topic columns — a fact-card
board groups by topic because the unit is "everything said about X"; an image
board's unit is one decision-caption-image triple, so ordering by TIME (the
order decisions were made) is the more legible read.

    x cursor : advances one CELL_W + GAP_X per image, wraps after N_COLS
    y cursor : advances by the TALLEST cell in the row just finished + GAP_Y
               (image heights vary by aspect ratio, so this can't be a fixed
               row height — same reasoning build_board.py uses for card height
               varying with wrapped-text line count)

Once elements are in one scene, Excalidraw's own editor is the "let the user
reposition things" mechanism — nothing else is needed for that part.

Usage:
    python compose_board.py --manifest board_manifest.json --out board.excalidraw

Where the manifest is a JSON list of
    {"image": "path/to.png", "caption": "...", "timestamp": "2025-07-21T17:22:43", "label": "optional"}
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import textwrap
import time
import uuid

from PIL import Image

CELL_W = 340           # fixed display width per image; height follows aspect ratio
N_COLS = 3
GAP_X = 40
GAP_Y = 70              # room for the caption block under the tallest image in a row
PAD = 60
CAPTION_FONT = 14
CAPTION_LH = 1.25
CHAR_W = 0.52
LABEL_FONT = 13


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _wrap(text: str, width_px: float, fs: float) -> list[str]:
    cols = max(12, int(width_px / (fs * CHAR_W)))
    return textwrap.wrap(" ".join(text.split()), cols, break_on_hyphens=False) or [""]


def _image_element(x: float, y: float, w: float, h: float, file_id: str) -> dict:
    return {
        "type": "image", "id": _new_id(), "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": "transparent", "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0,
        "opacity": 100, "groupIds": [], "frameId": None, "roundness": None,
        "seed": random.randint(1, 2**31 - 1), "versionNonce": random.randint(1, 2**31 - 1),
        "isDeleted": False, "boundElements": None, "updated": round(time.time() * 1000),
        "link": None, "locked": False, "status": "saved", "fileId": file_id, "scale": [1, 1],
    }


def _text_element(x: float, y: float, w: float, h: float, text: str, fs: float,
                   color: str = "#2f3337") -> dict:
    return {
        "type": "text", "id": _new_id(), "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0,
        "opacity": 100, "groupIds": [], "frameId": None, "roundness": None,
        "seed": random.randint(1, 2**31 - 1), "versionNonce": random.randint(1, 2**31 - 1),
        "isDeleted": False, "boundElements": None, "updated": round(time.time() * 1000),
        "link": None, "locked": False, "text": text, "originalText": text, "fontSize": fs,
        "fontFamily": 5, "textAlign": "left", "verticalAlign": "top", "baseline": 18,
        "containerId": None, "lineHeight": CAPTION_LH,
    }


def compose(entries: list[dict], out_path: str, columns: int = N_COLS) -> dict:
    """entries sorted by timestamp (if present) -> one Excalidraw scene, grid layout."""
    ordered = sorted(
        enumerate(entries),
        key=lambda pair: (pair[1].get("timestamp") or "", pair[0]),
    )

    elements: list[dict] = []
    files: dict = {}
    x, y = PAD, PAD
    col = 0
    row_max_h = 0.0

    for order, entry in ordered:
        with open(entry["image"], "rb") as fh:
            raw = fh.read()
        data_url = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
        with Image.open(entry["image"]) as im:
            iw, ih = im.size
        display_h = round(CELL_W * ih / iw)

        file_id = _new_id()
        files[file_id] = {
            "mimeType": "image/png", "id": file_id, "dataURL": data_url,
            "created": round(time.time() * 1000), "lastRetrieved": round(time.time() * 1000),
        }
        elements.append(_image_element(x, y, CELL_W, display_h, file_id))

        label = entry.get("label") or f"{order + 1}"
        elements.append(_text_element(x, y - LABEL_FONT * CAPTION_LH - 4, CELL_W,
                                       LABEL_FONT * CAPTION_LH, label, LABEL_FONT, "#7a7f87"))

        caption_lines = _wrap(entry.get("caption", ""), CELL_W, CAPTION_FONT)
        caption_h = len(caption_lines) * CAPTION_FONT * CAPTION_LH
        elements.append(_text_element(x, y + display_h + 10, CELL_W, caption_h,
                                       "\n".join(caption_lines), CAPTION_FONT))

        cell_h = display_h + 10 + caption_h
        row_max_h = max(row_max_h, cell_h)

        col += 1
        if col >= columns:
            col = 0
            x = PAD
            y += row_max_h + GAP_Y
            row_max_h = 0.0
        else:
            x += CELL_W + GAP_X

    width = PAD * 2 + columns * CELL_W + (columns - 1) * GAP_X
    height = y + row_max_h + PAD if row_max_h else y + PAD

    return {
        "type": "excalidraw", "version": 2,
        "source": "convograph/modules/graphic-generation/compose_board.py",
        "elements": elements,
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": files,
        "_layout_debug": {"width": width, "height": height, "n_images": len(entries)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                         help="JSON list of {image, caption, timestamp?, label?}.")
    parser.add_argument("--out", default="board.excalidraw")
    parser.add_argument("--columns", type=int, default=N_COLS)
    args = parser.parse_args()

    with open(args.manifest) as fh:
        entries = json.load(fh)

    scene = compose(entries, args.out, columns=args.columns)
    debug = scene.pop("_layout_debug")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(scene, fh, indent=2)
    print(f"wrote {args.out} — {debug['n_images']} images, "
          f"canvas {debug['width']:.0f}x{debug['height']:.0f}, "
          f"{len(scene['elements'])} elements")


if __name__ == "__main__":
    main()
