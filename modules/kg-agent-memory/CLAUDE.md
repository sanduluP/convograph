# Module 2 / Module 3 — repo-local conventions

These apply **only inside this module**. They do not change anything in the global
`~/.claude/CLAUDE.md`, and they must not be generalised out of this folder.

## Two Excalidraw style regimes — do not mix them

This repo produces two very different kinds of Excalidraw artifact, and the house
rules for one are wrong for the other.

| | Pipeline diagrams | The board (Module 3) |
|---|---|---|
| Files | `excalidraw/extraction/`, `excalidraw/retrieval/` | `excalidraw/board/` |
| Purpose | explain a system in a paper or slide | **look like a graphic recording** |
| `roughness` | **0** — straight, clean edges | **1** — sketchy, marker-drawn |
| `fontFamily` | **6** — Nunito ("Normal") | **5** — Excalifont (hand-drawn) |
| Layout | uniform grid, serpentine | size hierarchy, varied shapes |

**Why the difference.** The global rule (`roughness: 0`, Nunito) exists because
the hand-drawn style is hard to read in a figure — that judgement stands and is
unchanged for every diagram meant to be *read*. A graphic recording is not read
that way; it is a **poster**, and the marker look is the artifact rather than a
defect. Same tool, opposite goal.

**So:** when generating a pipeline diagram anywhere, keep the global rule. Only
`excalidraw/board/` uses the sketchy style, and only via the preset below.

## Board styling lives in a config file, never in code or in CLAUDE.md

`styles/board_styles.json` holds two named presets, `clean` and
`graphic-recording`. The renderer reads one; it hardcodes nothing.

```bash
BOARD_STYLE=graphic-recording bash scripts/run_build_board.sh
BOARD_STYLE=clean             bash scripts/run_build_board.sh
```

How a board looks will be re-tuned many times, by people who are not editing
Python. Keeping it as data means the look changes without touching the renderer,
two looks can be compared by swapping one argument, and the chosen preset **ships
with the product** rather than living in someone's editor instructions.

If the hand-drawn font turns out to be unreadable at presentation size, change
`fontFamily` to `6` in the preset. That is exactly the kind of edit this file
exists to make cheap.
