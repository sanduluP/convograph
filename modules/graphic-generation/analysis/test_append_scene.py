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

        print("\n🧷 errors")
        try:
            rb.append_scene(A, B, "left")
            check(False, "bad direction raises ValueError")
        except ValueError:
            check(True, "bad direction raises ValueError")
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
