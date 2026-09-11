#!/usr/bin/env python3
"""
icons.py — a small, CACHED vocabulary of concept icons for the board margins.

THE PROBLEM THIS SOLVES
-----------------------
A board carries ~45 items and had 6 drawings. Everything in the margins — what
is still open, what was decided — was plain text, because it was never given
pictures. A real graphic recording puts a pictogram next to nearly everything.

WHY A VOCABULARY RATHER THAN ONE IMAGE PER ITEM
-----------------------------------------------
Generating a unique glyph per item would work and would be wrong:

  * it costs 1.67 s per image, every run, forever (measured 2026-09-11);
  * it makes the board INCONSISTENT — "confirm the owner by Friday" and "confirm
    the roster by Friday" would get two unrelated pictures for the same idea.

A shared vocabulary fixes both. The same concept always gets the same icon, so
the board reads as one visual language, and each icon is generated ONCE and
cached to disk. First board pays ~35 s; every board after it pays nothing.

WHY KEYWORDS AND NOT AN LLM
---------------------------
Tagging an item with a concept is a lookup, not a judgement. A keyword match is
instant, free, deterministic and debuggable — three properties an extra LLM call
would each cost. When it cannot tell, it says so and the item simply gets no
icon, which is better than a confident wrong one.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.request

ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
FLUX_URL = os.getenv("FLUX_SERVER_URL", "http://localhost:8500")

# ── the vocabulary ──────────────────────────────────────────────────────────
# concept -> (glyph to draw, keywords that mean it)
#
# Every glyph is a WORDLESS PHYSICAL OBJECT, the same rule the anchors follow:
# FLUX renders letters as gibberish, so nothing whose identity is its markings
# can be used. No calendar for a deadline — an hourglass.
#
# Keyword order matters: the first concept whose keywords match wins, so the
# specific ones are listed before the general ones. "confirm ... by Friday" is
# a deadline first and an approval second, because the deadline is the thing a
# reader needs to see on a board.
VOCABULARY: dict[str, tuple[str, tuple[str, ...]]] = {
    "deadline":   ("an hourglass running out",
                   ("by eod", "deadline", "due ", "by friday", "by thursday",
                    "by monday", "by tuesday", "by wednesday", "today", "tomorrow",
                    "this week", "cutoff")),
    "lock":       ("a closed padlock",
                   ("freeze", "frozen", "lock", "locked", "locks", "seal")),
    "approval":   ("a wax seal with a ribbon",
                   ("approve", "approval", "sign-off", "sign off", "bless",
                    "confirm", "confirms", "confirmation", "stamp")),
    "owner":      ("a single chess king piece",
                   ("owner", "ownership", "owns", "accountable", "responsible",
                    "projectmanager", "project manager")),
    "evidence":   ("a magnifying glass",
                   ("evidence", "audit", "trail", "proof", "log", "record")),
    # WHY THESE FOUR WORDINGS CHANGED (2026-09-11, after looking at the sheet):
    #   handoff  "two hands passing a baton" -> FLUX drew a lettered sports
    #            banner, because "baton"+"hands" reads as a team-sport poster.
    #            A bare relay baton has no such association.
    #   decision "a fork in a road" -> a literal dinner fork. The idiom is
    #            invisible to an image model; a balance scale is the picture.
    #   rollback "carved in stone" pulled a stone slab that swallowed the arrow.
    #   data     "a stack of coins" says MONEY, not data.
    "rollback":   ("a curved arrow bending back on itself",
                   ("rollback", "roll back", "fallback", "revert", "undo",
                    "contingency")),
    "handoff":    ("a relay race baton",
                   ("handoff", "hand off", "handover", "transfer", "escalation",
                    "escalate")),
    "coverage":   ("an umbrella",
                   ("coverage", "roster", "staffing", "staffed", "on-call",
                    "hours", "bridge")),
    "monitoring": ("a lighthouse",
                   ("monitor", "monitoring", "alert", "watch", "observability")),
    "risk":       ("a warning triangle without text",
                   ("risk", "gap", "blocker", "blocked", "issue", "defect",
                    "failure", "exception")),
    "validation": ("a shield with a checkmark",
                   ("validate", "validation", "verify", "test", "qa", "check")),
    "scope":      ("a compass rose",
                   ("scope", "requirement", "spec", "definition", "criteria")),
    "schedule":   ("a ship's anchor",
                   ("cutover", "go-live", "golive", "launch", "release",
                    "deployment", "phase")),
    "people":     ("three simple wooden figures in a row",
                   ("team", "stakeholder", "attendee", "participant", "roster")),
    "data":       ("a cylindrical database drum of stacked discs",
                   ("data", "dataset", "feed", "pipeline", "extract")),
    "decision":   ("a two-pan balance scale",
                   ("decide", "decision", "choose", "option", "agreed",
                    "alignment")),

    # ── the ENTITY tier ─────────────────────────────────────────────────────
    # Everything above is a PROCESS concept and is what a margin item is about.
    # These are the THINGS a meeting is about — the nouns that become the
    # anchors of the content map.
    #
    # They exist because the planner, asked to invent a glyph for an anchor,
    # free-associates: on 2026-09-11 it returned Security -> "anchor",
    # Finance -> "split road", Ops -> "key". Those are not wrong so much as
    # arbitrary, and FLUX drew two of them covered in invented lettering
    # ("ANCHER", "Tat Road") because a nautical anchor and a road are things
    # the training data almost always labels.
    #
    # Selection beats generation: a fixed picture per entity is semantically
    # right, already verified wordless, and free (cached). ORDER MATTERS — the
    # entity tier is LAST, so a margin sentence still matches on what it is
    # about ("Operations is to freeze the chain" -> lock) and only a bare
    # anchor label falls through to here.
    "runbook":    ("an open book",
                   ("runbook", "playbook", "procedure", "manual")),
    "security":   ("a fingerprint",
                   ("security", "access control", "permission", "credential")),
    "finance":    ("a stack of gold coins",
                   ("finance", "budget", "invoice", "payment", "treasury")),
    "ops":        ("a mechanical gear wheel",
                   ("ops", "operations", "infrastructure", "runtime")),
    "support":    ("a headset with a microphone",
                   ("support", "helpdesk", "service desk")),
}

def entity_for(label: str) -> str | None:
    """Which vocabulary icon an ANCHOR LABEL should use, or None to let FLUX draw it.

    This first restricted itself to the entity tier, on the theory that an
    anchor is a thing and should not pick up the "deadline" hourglass because
    its label happens to contain "today". The restriction was wrong in practice:
    the planner names anchors after PROCESSES as often as after entities, and on
    the very next run it produced "Handoff Trigger" and "Rollback Path" — two
    labels we have exact icons for — which fell through to FLUX and came back as
    a road and a nautical anchor, both covered in invented lettering.

    An anchor label is one to three words, so a keyword match is a match on
    nearly the whole label rather than on an incidental word in a sentence.
    The whole vocabulary is therefore eligible; only a label that matches
    nothing is drawn from scratch.
    """
    return concept_for(label)

# The style suffix comes from board_plan so there is ONE definition of what an
# icon looks like. Imported lazily to keep this module importable on its own.
def _style() -> str:
    from board_plan import GLYPH_STYLE
    return GLYPH_STYLE


def concept_for(text: str) -> str | None:
    """Which concept an item is about, or None when it is not clear.

    None is a real answer. An item with no icon reads as a plain note, which is
    honest; an item with a confidently wrong icon teaches the reader a false
    association, and they will trust it because a picture looks deliberate.
    """
    low = " ".join((text or "").lower().split())
    for concept, (_, keywords) in VOCABULARY.items():
        for kw in keywords:
            # WORD BOUNDARIES, not substrings. Plain `in` tagged "blocked
            # approvals" as `lock`, because "blocked" contains "lock" — a wrong
            # icon that looked deliberate. Multi-word keywords keep their spaces
            # and are matched the same way.
            #
            # A trailing `s?` covers the plural, which word boundaries alone do
            # not: "ProjectManager STAMPS the validator" fell past `approval`
            # ("stamp") and landed on `handoff`, because "stamps" is a different
            # word. Listing every plural by hand is the same fix done worse.
            if re.search(rf"\b{re.escape(kw.strip())}s?\b", low):
                return concept
    return None


def cached_path(concept: str) -> str:
    return os.path.join(ICON_DIR, f"{concept}.png")


def missing(concepts: list[str]) -> list[str]:
    return [c for c in dict.fromkeys(concepts)
            if c in VOCABULARY and not os.path.exists(cached_path(c))]


def ensure(concepts: list[str], log=None) -> dict[str, str]:
    """Make sure each concept has a cached PNG. Returns {concept: path}.

    Only the MISSING ones are generated, in ONE batched call — the warm FLUX
    server holds the model, so a batch of twelve costs 20 s while twelve separate
    calls would cost far more. If the server is unreachable the function returns
    whatever is already cached rather than raising: a board with fewer icons is
    still a board, and the alternative is no board at all.
    """
    os.makedirs(ICON_DIR, exist_ok=True)
    want = [c for c in dict.fromkeys(concepts) if c in VOCABULARY]
    have = {c: cached_path(c) for c in want if os.path.exists(cached_path(c))}
    todo = [c for c in want if c not in have]
    if not todo:
        return have

    if log:
        log(f"   🎨 generating {len(todo)} new icon(s), {len(have)} from cache: "
            f"{', '.join(todo)}")
    captions = [f"{VOCABULARY[c][0]}, {_style()}" for c in todo]
    try:
        req = urllib.request.Request(
            f"{FLUX_URL.rstrip('/')}/generate",
            data=json.dumps({"captions": captions}).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=1800) as r:
            res = json.loads(r.read())
    except Exception as exc:                                   # noqa: BLE001
        if log:
            log(f"   ⚠️  icon generation unavailable ({type(exc).__name__}) — "
                f"the board will use the {len(have)} cached icon(s)")
        return have

    by_index = {img["index"]: img for img in res.get("images", [])}
    for i, concept in enumerate(todo):
        img = by_index.get(i)
        if not img or "png_b64" not in img:
            continue
        path = cached_path(concept)
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(img["png_b64"]))
        have[concept] = path
    if log:
        log(f"   🎨 {len(todo)} icon(s) in {time.time() - t0:.1f}s "
            f"— cached, so later boards pay nothing")
    return have


def main() -> None:
    """Pre-warm the whole vocabulary, so the first real board is instant."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="generate every concept, not just the missing ones")
    ap.add_argument("--only", default="",
                    help="comma-separated concepts to REDRAW (drops their cache "
                         "first). Use this after changing one glyph's wording, "
                         "so the fifteen icons that are already right are not "
                         "re-rolled into fifteen different pictures.")
    ap.add_argument("--list", action="store_true", help="show the vocabulary")
    args = ap.parse_args()

    if args.list:
        for c, (glyph, kws) in VOCABULARY.items():
            state = "cached" if os.path.exists(cached_path(c)) else "—"
            print(f"  {c:<12} {state:<7} {glyph}")
            print(f"  {'':<12} {'':<7} keywords: {', '.join(kws[:6])}…")
        return

    concepts = list(VOCABULARY)
    # --only redraws a named few; --all redraws everything. Both work by
    # DELETING the cache entry, because ensure() generates exactly what is
    # missing — there is no second code path to keep in sync.
    redraw = concepts if args.all else [c.strip() for c in args.only.split(",")
                                        if c.strip()]
    unknown = [c for c in redraw if c not in VOCABULARY]
    if unknown:
        raise SystemExit(f"❌ not in the vocabulary: {', '.join(unknown)}")
    for c in redraw:
        p = cached_path(c)
        if os.path.exists(p):
            os.remove(p)
    got = ensure(concepts, log=print)
    print(f"\n✅ {len(got)}/{len(concepts)} icons available in {ICON_DIR}")


if __name__ == "__main__":
    main()
