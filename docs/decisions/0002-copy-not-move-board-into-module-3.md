# 0002: Copy (not move) the board/graphic-recording content into module 3

## Status

Accepted (2026-08-31).

## Context

The Excalidraw fact-card board (`build_board.py`, `compress_facts.py`,
`excalidraw/board/`, `styles/board_styles.json`, `prompts/board_phrase_system.txt`,
`scripts/run_build_board.sh`) was built inside `modules/kg-agent-memory` and is
conceptually Module 3's work — its own code comments already call it "Module 3,
rung 0". Module 2's author was actively iterating on it (same-day archived
board revisions) when the question of relocating it came up.

## Decision

Copy the board-related files into `modules/graphic-generation`, leaving
`modules/kg-agent-memory`'s copies untouched. Do not `git mv` or delete
anything from module 2.

## Rationale

- Avoids colliding with in-flight edits — a `git mv` mid-iteration risks merge
  conflicts or silently orphaning the author's next commit.
- Module 2's author still knows that codebase best and is the right person to
  decide what should stay there (e.g. as an internal audit/debugging tool) vs.
  what's purely Module 3's concern, once they have a moment to look.
- Costs a small amount of duplication now (~2 small Python files, a prompt, a
  style JSON, one small board file) in exchange for zero risk to someone else's
  active work.

## Revisit if

Once module 2's author has reviewed, the copies in `modules/kg-agent-memory`
should be deleted (or kept only if there's a standalone reason for them to
live there) so there's a single source of truth. Track this as a follow-up,
not a blocker for Module 3's own progress.
