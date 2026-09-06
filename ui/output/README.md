# `ui/output/` — generated boards

`output/` is git-ignored: every run writes a new board and most are scratch.
Boards worth keeping are added deliberately with `git add -f`.

## Why these files are large

An `.excalidraw` board embeds every image as base64 inside the JSON, so a
6-image board is ~7 MB. That is fine for a handful of committed examples and
bad as a habit — do not add every run.

## What is committed

| file | what it shows |
|---|---|
| `board_20260831_*` | Priyabanta's original runs (transcript in, per-fact captions) |
| `board_20260906_155111` | first board driven by a cypher query over an existing KG instead of re-extracting a transcript — 3 facts, 11.2 s end to end |
| `board_20260906_182556` | 6 superseded facts from the full corpus |

⚠️ **Both 2026-09-06 boards still use the OLD per-fact captioning.** Their canvas
text ends with "hand-drawn sticky-note icon style, marker on white paper" —
the FLUX style suffix leaking onto the board, which is exactly the flaw
`modules/graphic-generation/board_plan.py` exists to fix. That planner is
written and measured but not yet wired into `run_pipeline`; see the four hops in
the 2026-09-06 daily note.
