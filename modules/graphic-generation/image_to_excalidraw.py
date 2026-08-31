#!/usr/bin/env python3
"""
image_to_excalidraw.py — wrap one PNG (+ its caption) into a valid .excalidraw
scene: an image element with the caption as a text element beneath it.

This is the "...and then that goes to Excalidraw" half of the pipeline:
    caption -> generate_image.py -> PNG -> image_to_excalidraw.py -> .excalidraw

Rung 0 on purpose: one image, one board. Placing several generated images
alongside the fact-card columns that analysis/build_board.py draws (i.e.
merging into one shared canvas instead of a separate file per image) is later
work, once there's more than one image to place.

Usage:
    python image_to_excalidraw.py --image out.png --caption "..." --out image.excalidraw
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import time
import uuid

from PIL import Image


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def build_scene(image_path: str, caption: str, display_width: int = 360) -> dict:
    with open(image_path, "rb") as fh:
        raw = fh.read()
    data_url = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"

    with Image.open(image_path) as im:
        w, h = im.size
    display_height = round(display_width * h / w)

    now_ms = round(time.time() * 1000)
    file_id = _new_id()
    image_el_id = _new_id()
    text_el_id = _new_id()

    image_element = {
        "type": "image",
        "id": image_el_id,
        "x": 0,
        "y": 0,
        "width": display_width,
        "height": display_height,
        "angle": 0,
        "strokeColor": "transparent",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": random.randint(1, 2**31 - 1),
        "versionNonce": random.randint(1, 2**31 - 1),
        "isDeleted": False,
        "boundElements": None,
        "updated": now_ms,
        "link": None,
        "locked": False,
        "status": "saved",
        "fileId": file_id,
        "scale": [1, 1],
    }

    text_element = {
        "type": "text",
        "id": text_el_id,
        "x": 0,
        "y": display_height + 12,
        "width": display_width,
        "height": 25,
        "angle": 0,
        "strokeColor": "#2f3337",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": random.randint(1, 2**31 - 1),
        "versionNonce": random.randint(1, 2**31 - 1),
        "isDeleted": False,
        "boundElements": None,
        "updated": now_ms,
        "link": None,
        "locked": False,
        "text": caption,
        "fontSize": 14,
        "fontFamily": 5,  # Excalifont — matches the board's graphic-recording preset
        "textAlign": "left",
        "verticalAlign": "top",
        "baseline": 18,
        "containerId": None,
        "originalText": caption,
        "lineHeight": 1.25,
    }

    return {
        "type": "excalidraw",
        "version": 2,
        "source": "convograph/modules/graphic-generation/image_to_excalidraw.py",
        "elements": [image_element, text_element],
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {
            file_id: {
                "mimeType": "image/png",
                "id": file_id,
                "dataURL": data_url,
                "created": now_ms,
                "lastRetrieved": now_ms,
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the PNG to embed.")
    parser.add_argument("--caption", required=True, help="Caption shown under the image.")
    parser.add_argument("--out", default="image.excalidraw", help="Output .excalidraw path.")
    args = parser.parse_args()

    scene = build_scene(args.image, args.caption)
    with open(args.out, "w") as fh:
        json.dump(scene, fh, indent=2)
    print(f"wrote {args.out} ({len(scene['elements'])} elements, 1 embedded image)")


if __name__ == "__main__":
    main()
