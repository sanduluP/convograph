#!/usr/bin/env python3
"""
preview_board.py — rasterise an .excalidraw scene to a PNG, approximately.

WHY THIS EXISTS
---------------
CLAUDE.md rule 12 is right that a human has to own the final look of a diagram:
Claude cannot see how it renders, so every generated layout needs manual
alignment fixes only a person can judge. But "cannot see it AT ALL" makes the
first version a blind guess, and a blind guess wastes the human's time on
problems a glance would have caught — a label sitting on an arrow, a card
overlapping its neighbour, a canvas that is 3:1 and unreadable.

So this is a PROOFING tool, not a renderer. It answers "is anything colliding,
clipped, or absurdly proportioned?" before the file is handed over. It is
deliberately NOT faithful:

  * DejaVu Sans stands in for Excalifont, so text metrics are close but not
    exact — good enough for collision checking, useless for judging typography;
  * roughness is ignored (everything is drawn with clean strokes);
  * bound arrow labels are drawn at the arrow's midpoint, which is where
    Excalidraw puts them, but without its collision avoidance.

Anything this tool says looks fine may still need the human's eye. Anything it
says is BROKEN, is broken.

Usage:
    python analysis/preview_board.py board.excalidraw --out preview.png
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

# DejaVu ships with matplotlib/PIL on every machine we use, so the preview never
# depends on a font being installed. Excalifont is wider; the preview therefore
# slightly UNDER-estimates text width, which is the safe direction for a
# collision check only if we keep that in mind when reading it.
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

SCALE = 1.0            # device pixels per Excalidraw unit
MARGIN = 20


def _font(size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REG
    try:
        return ImageFont.truetype(path, max(6, int(round(size))))
    except OSError:
        return ImageFont.load_default()


def _bbox(elements: list[dict]) -> tuple[float, float, float, float]:
    xs = [e["x"] for e in elements] + [e["x"] + e.get("width", 0) for e in elements]
    ys = [e["y"] for e in elements] + [e["y"] + e.get("height", 0) for e in elements]
    return min(xs), min(ys), max(xs), max(ys)


def _arrowhead(draw: ImageDraw.ImageDraw, x0: float, y0: float,
               x1: float, y1: float, color: str, size: float = 13) -> None:
    ang = math.atan2(y1 - y0, x1 - x0)
    for spread in (2.6, -2.6):
        draw.line([(x1, y1),
                   (x1 + size * math.cos(ang + spread),
                    y1 + size * math.sin(ang + spread))], fill=color, width=2)


def render(scene: dict, out_path: str) -> dict:
    elements = [e for e in scene.get("elements", []) if not e.get("isDeleted")]
    if not elements:
        raise ValueError("scene has no elements")
    files = scene.get("files", {})

    min_x, min_y, max_x, max_y = _bbox(elements)
    W = int((max_x - min_x) * SCALE) + 2 * MARGIN
    H = int((max_y - min_y) * SCALE) + 2 * MARGIN

    img = Image.new("RGB", (W, H), scene.get("appState", {})
                    .get("viewBackgroundColor", "#ffffff"))
    draw = ImageDraw.Draw(img)

    def sx(v: float) -> float:
        return (v - min_x) * SCALE + MARGIN

    def sy(v: float) -> float:
        return (v - min_y) * SCALE + MARGIN

    by_id = {e["id"]: e for e in elements}

    # Shapes and images first, then text and arrows on top — the same painting
    # order Excalidraw uses for its default z-ordering here.
    for e in elements:
        kind = e["type"]
        x, y = sx(e["x"]), sy(e["y"])
        w, h = e.get("width", 0) * SCALE, e.get("height", 0) * SCALE

        if kind == "rectangle":
            bg = e.get("backgroundColor", "transparent")
            draw.rounded_rectangle([x, y, x + w, y + h], radius=12,
                                   fill=None if bg == "transparent" else bg,
                                   outline=e.get("strokeColor", "#000"),
                                   width=max(1, int(e.get("strokeWidth", 1))))
        elif kind == "ellipse":
            draw.ellipse([x, y, x + w, y + h], outline=e.get("strokeColor", "#000"))
        elif kind == "image":
            rec = files.get(e.get("fileId"))
            if rec and rec.get("dataURL", "").startswith("data:image"):
                raw = base64.b64decode(rec["dataURL"].split(",", 1)[1])
                with Image.open(io.BytesIO(raw)) as src:
                    img.paste(src.convert("RGB").resize(
                        (max(1, int(w)), max(1, int(h)))), (int(x), int(y)))
            else:
                # A missing file entry is a real bug — make it loud, not blank.
                draw.rectangle([x, y, x + w, y + h], outline="#e03131", width=2)
                draw.line([x, y, x + w, y + h], fill="#e03131", width=2)

    for e in elements:
        kind = e["type"]
        if kind == "arrow" or kind == "line":
            pts = e.get("points") or [[0, 0], [e.get("width", 0), e.get("height", 0)]]
            abs_pts = [(sx(e["x"] + p[0]), sy(e["y"] + p[1])) for p in pts]
            col = e.get("strokeColor", "#000")
            draw.line(abs_pts, fill=col, width=max(1, int(e.get("strokeWidth", 1))))
            if kind == "arrow" and e.get("endArrowhead") and len(abs_pts) >= 2:
                _arrowhead(draw, *abs_pts[-2], *abs_pts[-1], color=col)

        elif kind == "text":
            fs = e.get("fontSize", 16) * SCALE
            font = _font(fs, bold=fs >= 30)
            lines = e.get("text", "").split("\n")
            col = e.get("strokeColor", "#000")
            container = by_id.get(e.get("containerId"))
            if container is not None and container.get("type") != "arrow":
                # A label bound to a SHAPE is centred inside it, both axes.
                # Without this branch the arrow path below ran for rectangles
                # too, read a `points` key they do not have, and drew every box
                # label at the box's top-left corner — which looked like a
                # layout bug in the diagram rather than one in this previewer.
                cx = sx(container["x"] + container.get("width", 0) / 2)
                cy = sy(container["y"] + container.get("height", 0) / 2)
                th = len(lines) * fs * 1.25
                for i, line in enumerate(lines):
                    lw = draw.textlength(line, font=font)
                    draw.text((cx - lw / 2, cy - th / 2 + i * fs * 1.25), line,
                              font=font, fill=col)
                continue
            if container is not None:
                # A bound label: Excalidraw centres it on its container. For an
                # arrow that is the midpoint of the polyline.
                pts = container.get("points") or [[0, 0]]
                if len(pts) >= 2:
                    # Midpoint of the whole polyline. Indexing pts[len//2] is
                    # wrong for the 2-point arrows we emit: it returns the END.
                    a, b = pts[len(pts) // 2 - 1], pts[len(pts) // 2]
                    mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
                else:
                    mid = pts[0]
                cx = sx(container["x"] + mid[0])
                cy = sy(container["y"] + mid[1])
                th = len(lines) * fs * 1.25
                # Excalidraw knocks a hole in the arrow behind its label; mimic
                # it so the preview shows what the reader will actually see.
                tw = max(draw.textlength(l, font=font) for l in lines)
                draw.rectangle([cx - tw / 2 - 4, cy - th / 2 - 2,
                                cx + tw / 2 + 4, cy + th / 2 + 2], fill="#ffffff")
                for i, line in enumerate(lines):
                    lw = draw.textlength(line, font=font)
                    draw.text((cx - lw / 2, cy - th / 2 + i * fs * 1.25), line,
                              font=font, fill=col)
                continue

            x, y = sx(e["x"]), sy(e["y"])
            box_w = e.get("width", 0) * SCALE
            align = e.get("textAlign", "left")
            for i, line in enumerate(lines):
                lw = draw.textlength(line, font=font)
                lx = x
                if align == "center":
                    lx = x + (box_w - lw) / 2
                elif align == "right":
                    lx = x + box_w - lw
                draw.text((lx, y + i * fs * 1.25), line, font=font, fill=col)

    img.save(out_path)
    return {"size": [W, H], "elements": len(elements)}


# ── structural + geometric checks ───────────────────────────────────────────
def check(scene: dict, aspect: tuple[float, float] | None = (0.4, 3.0)) -> list[str]:
    """Problems a rendering cannot show but a reader would hit. Empty = clean.

    `aspect` is the readable width:height range for the WHOLE canvas. It is a
    property of one board: a strip of three meetings appended side by side is
    legitimately 4:1, and each block already passed this test on its own run.
    Pass None to skip it — the structural checks (ids, bindings, file refs,
    overlaps) still run across the union, which is exactly what an append needs.
    """
    problems: list[str] = []
    elements = [e for e in scene.get("elements", []) if not e.get("isDeleted")]
    ids = [e["id"] for e in elements]
    by_id = {e["id"]: e for e in elements}

    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        problems.append(f"duplicate element ids: {sorted(dupes)}")

    for e in elements:
        # Every reference must resolve, or Excalidraw silently drops the link
        # and the arrow detaches on the first drag.
        for key in ("containerId", "frameId"):
            ref = e.get(key)
            if ref and ref not in by_id:
                problems.append(f"{e['type']} {e['id']}: {key} -> missing {ref}")
        for side in ("startBinding", "endBinding"):
            b = e.get(side)
            if b and b.get("elementId") not in by_id:
                problems.append(f"arrow {e['id']}: {side} -> missing {b.get('elementId')}")
        for b in (e.get("boundElements") or []):
            if b.get("id") not in by_id:
                problems.append(f"{e['type']} {e['id']}: boundElements -> missing {b.get('id')}")
        if e["type"] == "image":
            if e.get("fileId") not in scene.get("files", {}):
                problems.append(f"image {e['id']}: fileId not in files{{}} — renders blank")

    # A bound text must be listed by its container, not just point at it.
    for e in elements:
        cid = e.get("containerId")
        if cid and cid in by_id:
            listed = {b.get("id") for b in (by_id[cid].get("boundElements") or [])}
            if e["id"] not in listed:
                problems.append(f"text {e['id']} names container {cid}, "
                                f"but the container does not list it back")

    # Cards must not overlap. Only rectangles are checked: text and images live
    # inside a card by construction, so a rectangle collision is the real signal.
    rects = [e for e in elements if e["type"] == "rectangle"]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            ox = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
            oy = min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"])
            if ox > 0 and oy > 0:
                problems.append(f"cards {a['id']} and {b['id']} overlap "
                                f"by {ox:.0f}x{oy:.0f}px")

    # A bound arrow label sitting on a card is unreadable, and it is the defect a
    # generated ring layout produces most often: the label goes at the arrow's
    # midpoint, which for a link between distant nodes can land on a third card.
    # Checked here rather than left to the eye, because "I looked and it seemed
    # fine" does not survive a re-run with different anchor counts.
    cards = [e for e in elements if e["type"] == "rectangle"]
    for e in elements:
        if e["type"] != "text" or not e.get("containerId"):
            continue
        container = by_id.get(e["containerId"])
        if not container or container.get("type") != "arrow":
            continue
        pts = container.get("points") or [[0, 0]]
        if len(pts) >= 2:
            a, b = pts[len(pts) // 2 - 1], pts[len(pts) // 2]
            mid = (container["x"] + (a[0] + b[0]) / 2,
                   container["y"] + (a[1] + b[1]) / 2)
        else:
            mid = (container["x"], container["y"])
        # The label's own box, centred on that midpoint.
        lw, lh = e.get("width", 0), e.get("height", 0)
        lx0, ly0 = mid[0] - lw / 2, mid[1] - lh / 2
        for c in cards:
            ox = min(lx0 + lw, c["x"] + c["width"]) - max(lx0, c["x"])
            oy = min(ly0 + lh, c["y"] + c["height"]) - max(ly0, c["y"])
            if ox > 0 and oy > 0:
                problems.append(
                    f"link label {e.get('text','')!r} overlaps a card "
                    f"by {ox:.0f}x{oy:.0f}px — unreadable")
                break

    if aspect is not None:
        lo, hi = aspect
        min_x, min_y, max_x, max_y = _bbox(elements)
        w, h = max_x - min_x, max_y - min_y
        if h and not lo <= w / h <= hi:
            problems.append(f"canvas aspect {w / h:.2f}:1 ({w:.0f}x{h:.0f}) — "
                            f"outside the readable {lo}-{hi} range")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", help="path to an .excalidraw file")
    ap.add_argument("--out", default=None, help="PNG path (default: <scene>.preview.png)")
    args = ap.parse_args()

    with open(args.scene) as fh:
        scene = json.load(fh)

    out = args.out or os.path.splitext(args.scene)[0] + ".preview.png"
    info = render(scene, out)
    problems = check(scene)

    print(f"🖼️  preview  : {out}  ({info['size'][0]}x{info['size'][1]}px, "
          f"{info['elements']} elements)")
    if problems:
        print(f"⚠️  {len(problems)} problem(s):")
        for p in problems:
            print(f"   - {p}")
    else:
        print("✅ structural + geometric checks clean")


if __name__ == "__main__":
    main()
