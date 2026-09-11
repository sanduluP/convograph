# Boards from before the UI

Everything in here predates the Streamlit UI. **Boards are now written in exactly
one place:**

```
ui/output/<provider>-<model>_<digest|wN>_<timestamp>/board.excalidraw
```

There used to be three folders holding boards, which meant three places to look
and no way to tell which was current:

| was | now |
| --- | --- |
| `modules/kg-agent-memory/excalidraw/board/` | `archive/boards-pre-ui/board/` |
| `modules/graphic-generation/excalidraw/board/` | deleted — it was a byte-identical copy of the above |
| `modules/kg-agent-memory/excalidraw/archive/` | `archive/boards-pre-ui/early-diagram-drafts/` |

`early-diagram-drafts/` holds the v1 extraction-phase diagrams, superseded by the
serpentine v2 versions that module 2's README embeds.

## What was NOT moved, and why

These are **pipeline diagrams**, not boards, and they are embedded in READMEs:

- `modules/kg-agent-memory/excalidraw/extraction/` — how a window becomes an episode
- `modules/kg-agent-memory/excalidraw/retrieval/` — how a question becomes a verdict
- `modules/graphic-generation/excalidraw/pipeline/` — the module 3 pipeline

## Safe to delete?

Yes, once nobody wants the old attempts. Nothing in the codebase reads this
directory. Two generators still *write* to the old path if run —
`scripts/run_build_board.sh` and `analysis/build_content_map.py` in module 2 —
but both are superseded by `run_content_map.sh`, which writes to `ui/output/`.
