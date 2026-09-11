#!/usr/bin/env python3
"""Pure test of render_board.append_scene — no GPU, no LLM, no network.

WHY A TEST AT ALL
-----------------
append_scene is geometry over a JSON document: shift one block clear of
another, carry the image files, keep ids unique. Every expectation is
countable by hand on two tiny synthetic boards and takes seconds, whereas the
end-to-end proof costs a digest, a planner call and several FLUX images per
block. So the mechanics are proven here first, and the expensive run only has
to show that the wiring passed the right things in.

Runs with the UI venv (needs Pillow, which render_board imports):

    ui/.venv/bin/python modules/graphic-generation/analysis/test_append_scene.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))       # render_board
sys.path.insert(0, HERE)                        # preview_board

from PIL import Image  # noqa: E402

import preview_board as pb  # noqa: E402
import render_board as rb  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)
    print(("   ✅ " if cond else "   ❌ ") + msg)


def _plan(n: int, title: str) -> dict:
    anchors = [{"label": f"anchor {i}", "glyph": "a padlock", "from_facts": [i]}
               for i in range(n)]
    return {"title": title, "anchors": anchors,
            "links": [{"from": 0, "to": 1, "label": "leads to"}],
            "notes": [{"text": "a short note", "anchor": 0}],
            "dropped": []}


def _png(path: str) -> str:
    Image.new("RGB", (64, 48), "white").save(path)
    return path


def _captions(scene: dict) -> list[dict]:
    return [e for e in scene["elements"]
            if (e.get("customData") or {}).get(rb.CAPTION_KEY)]


def _split(merged: dict, n_base_live: int) -> tuple[list[dict], list[dict]]:
    """(base part incl. its caption, new part incl. its caption)."""
    els = merged["elements"]
    return els[:n_base_live + 1], els[n_base_live + 1:]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        img = _png(os.path.join(tmp, "a.png"))
        A = rb.build_scene(_plan(2, "Meeting one"), [img, None]); A.pop("_layout_debug")
        B = rb.build_scene(_plan(3, "Meeting two"), []);          B.pop("_layout_debug")
        C = rb.build_scene(_plan(2, "Meeting three"), []);        C.pop("_layout_debug")
        D = rb.build_scene(_plan(2, "Meeting four"), []);         D.pop("_layout_debug")
        a_before = json.dumps(A, sort_keys=True)
        nA, nB, nC = len(A["elements"]), len(B["elements"]), len(C["elements"])
        check(len(A["files"]) == 1 and len(B["files"]) == 0,
              "fixture: A carries one image, B none")

        print("\n🧷 below")
        M = rb.append_scene(A, B, "below", base_caption="m1", new_caption="m2")
        dbg = M.pop("_layout_debug")
        ids = [e["id"] for e in M["elements"]]
        check(len(M["elements"]) == nA + nB + 2, "element count = A + B + 2 captions")
        check(len(set(ids)) == len(ids), "ids unique across the union")
        check(M["files"] == {**A["files"], **B["files"]}, "files dict is the union")
        check(not any("index" in e for e in M["elements"]), "no 'index' keys carried")
        check(len(_captions(M)) == 2 and dbg["blocks"] == 2,
              "exactly two captions, and debug agrees")
        base_part, new_part = _split(M, nA)
        bx0, by0, bx1, by1 = rb._scene_bbox(base_part)
        nx0, ny0, nx1, ny1 = rb._scene_bbox(new_part)
        check(ny0 >= by1 + rb.APPEND_GAP - 0.01, "new block starts >= APPEND_GAP below base")
        check(abs(nx0 - bx0) < 0.01, "new block is left-aligned to base")
        titles = [e["text"] for e in M["elements"] if e["type"] == "text"]
        check("Meeting one" in titles and "Meeting two" in titles, "both titles present")
        check(json.dumps(A, sort_keys=True) == a_before, "base input not mutated")
        struct = pb.check(M, aspect=None)
        check(struct == [], f"structural check clean: {struct}")

        print("\n🧷 right")
        R = rb.append_scene(A, B, "right", base_caption="m1", new_caption="m2")
        R.pop("_layout_debug")
        base_part, new_part = _split(R, nA)
        bx0, by0, bx1, by1 = rb._scene_bbox(base_part)
        nx0, ny0, nx1, ny1 = rb._scene_bbox(new_part)
        check(nx0 >= bx1 + rb.APPEND_GAP - 0.01, "new block starts >= APPEND_GAP right of base")
        check(abs(ny0 - by0) < 0.01, "new block is top-aligned to base")

        print("\n🧷 chain: (A below B) then C right")
        M2 = rb.append_scene(M, C, "right", base_caption="ignored", new_caption="m3")
        d2 = M2.pop("_layout_debug")
        check(len(M2["elements"]) == nA + nB + nC + 3, "chain adds exactly one more caption")
        check(d2["blocks"] == 3, "three blocks on the canvas")
        struct = pb.check(M2, aspect=None)
        check(struct == [], f"chain structural check clean: {struct}")
        narrow = pb.check(M2, aspect=(0.99, 1.01))
        check(any("aspect" in p for p in narrow),
              "aspect parameter is honoured (a near-square range flags this canvas)")
        check(not any("aspect" in p for p in pb.check(M2, aspect=None)),
              "aspect=None suppresses only the aspect message")

        print("\n🧷 hand-edited base: index keys and a tombstone")
        H = json.loads(json.dumps(A))
        for e in H["elements"]:
            e["index"] = "a3"
        H["elements"].append({**H["elements"][-1], "id": "dead0000dead0000",
                              "isDeleted": True})
        MH = rb.append_scene(H, B, "below", base_caption="m1", new_caption="m2")
        MH.pop("_layout_debug")
        check(not any("index" in e for e in MH["elements"]), "index keys stripped")
        check(not any(e.get("isDeleted") for e in MH["elements"]), "tombstone not carried")
        check(len(MH["elements"]) == nA + nB + 2, "tombstone excluded from the count")

        print("\n🧷 block bounds recorded in the captions")
        bb = rb.block_bounds(M["elements"])
        base_part, new_part = _split(M, nA)
        check(len(bb) == 2, "two blocks recorded on the below-append")
        check(all(abs(a - b) < 0.01 for a, b in zip(bb[0], rb._scene_bbox(base_part))),
              "block 1 bbox matches its elements, caption included")
        check(all(abs(a - b) < 0.01 for a, b in zip(bb[1], rb._scene_bbox(new_part))),
              "block 2 bbox matches its elements, caption included")
        old = json.loads(json.dumps(M))
        for e in old["elements"]:
            meta = (e.get("customData") or {}).get(rb.CAPTION_KEY)
            if meta:
                meta.pop("bbox", None)
        check(len(rb.block_bounds(old["elements"])) == 1,
              "captions without a bbox (older boards) fall back to one block")
        check(len(rb.block_bounds(A["elements"])) == 1,
              "an uncaptioned board is one block")

        print("\n🧷 above and left")
        Ab = rb.append_scene(A, B, "above", base_caption="m1", new_caption="m2")
        Ab.pop("_layout_debug")
        bb = rb.block_bounds(Ab["elements"])          # document order: base, new
        check(bb[1][3] <= bb[0][1] - rb.APPEND_GAP + 0.01, "new block ends >= APPEND_GAP above base")
        check(abs(bb[1][0] - bb[0][0]) < 0.01, "above: left-aligned to base")
        check(pb.check(Ab, aspect=None) == [], "above: structural check clean")
        Lf = rb.append_scene(A, B, "left", base_caption="m1", new_caption="m2")
        Lf.pop("_layout_debug")
        bb = rb.block_bounds(Lf["elements"])
        check(bb[1][2] <= bb[0][0] - rb.APPEND_GAP + 0.01, "new block ends >= APPEND_GAP left of base")
        check(abs(bb[1][1] - bb[0][1]) < 0.01, "left: top-aligned to base")
        check(pb.check(Lf, aspect=None) == [], "left: structural check clean")

        print("\n🧷 centre on the neighbour")
        mid = lambda b: (b[0] + b[2]) / 2                      # noqa: E731
        midy = lambda b: (b[1] + b[3]) / 2                     # noqa: E731
        Cc = rb.append_scene(A, B, "below", align="center", base_caption="m1", new_caption="m2")
        Cc.pop("_layout_debug")
        bb = rb.block_bounds(Cc["elements"])
        check(abs(mid(bb[1]) - mid(bb[0])) < 1.0,
              f"below+center: x-centres agree ({mid(bb[0]):.1f} vs {mid(bb[1]):.1f})")
        Cr = rb.append_scene(A, B, "right", align="center", base_caption="m1", new_caption="m2")
        Cr.pop("_layout_debug")
        bb = rb.block_bounds(Cr["elements"])
        check(abs(midy(bb[1]) - midy(bb[0])) < 1.0,
              f"right+center: y-centres agree ({midy(bb[0]):.1f} vs {midy(bb[1]):.1f})")
        # Centring is on the NEIGHBOUR, not the canvas: after A-below-B, a third
        # block below must centre on B (the bottom-most), not on the union.
        Cn = rb.append_scene(Cc, C, "below", align="center", new_caption="m3")
        Cn.pop("_layout_debug")
        bb = rb.block_bounds(Cn["elements"])
        check(abs(mid(bb[2]) - mid(bb[1])) < 1.0, "third block centres on the block above it")

        print("\n🧷 grid, 2 per row")
        G1 = rb.append_scene(A, B, "grid", columns=2, base_caption="m1", new_caption="m2")
        G1.pop("_layout_debug")
        bb = rb.block_bounds(G1["elements"])
        check(bb[1][0] >= bb[0][2] + rb.APPEND_GAP - 0.01, "block 2 continues the row to the right")
        check(abs(bb[1][1] - bb[0][1]) < 0.01, "block 2 top-aligned to the row")
        G2 = rb.append_scene(G1, C, "grid", columns=2, new_caption="m3")
        d3 = G2.pop("_layout_debug")
        bb = rb.block_bounds(G2["elements"])
        check(len(bb) == 3 and d3["blocks"] == 3 and d3["columns"] == 2, "three blocks, grid debug recorded")
        row_bottom = max(bb[0][3], bb[1][3])
        check(bb[2][1] >= row_bottom + rb.APPEND_GAP - 0.01, "block 3 wraps below the full first row")
        check(abs(bb[2][0] - bb[0][0]) < 0.01, "block 3 returns to the first row's left margin")
        check(pb.check(G2, aspect=None) == [], f"grid structural check clean: {pb.check(G2, aspect=None)}")
        # A distinct fourth scene: re-appending B here would collide with B's
        # own ids already on the canvas, which the guard rightly refuses.
        G3 = rb.append_scene(G2, D, "grid", columns=2, new_caption="m4")
        G3.pop("_layout_debug")
        bb = rb.block_bounds(G3["elements"])
        check(bb[3][0] >= bb[2][2] + rb.APPEND_GAP - 0.01 and abs(bb[3][1] - bb[2][1]) < 0.01,
              "block 4 continues the second row")

        print("\n🧷 errors")
        try:
            rb.append_scene(A, B, "diagonal")
            check(False, "bad direction raises ValueError")
        except ValueError:
            check(True, "bad direction raises ValueError")
        try:
            rb.append_scene(A, B, "below", align="middle")
            check(False, "bad align raises ValueError")
        except ValueError:
            check(True, "bad align raises ValueError")
        dup = json.loads(json.dumps(B))
        dup["elements"][0]["id"] = A["elements"][0]["id"]
        try:
            rb.append_scene(A, dup, "below")
            check(False, "duplicate id raises ValueError")
        except ValueError:
            check(True, "duplicate id raises ValueError")

    print("\n" + "═" * 60)
    if FAILURES:
        print(f"❌ {len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"   • {f}")
        return 1
    print("✅ all append_scene checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
