# graphic-generation

Turns a temporal-KG subgraph into a visual. Two pipelines live here, both "rung 0":

## 1. Fact-card board (copied from modules/kg-agent-memory)

`analysis/build_board.py` queries a temporal KG in Neo4j directly for facts that
were later superseded (`invalid_at` set), compresses each into a 3-6 word
headline (`analysis/compress_facts.py`), and renders them as an `.excalidraw`
board — struck-through superseded cards, replacement below in solid ink. See
[`modules/kg-agent-memory/CLAUDE.md`](../kg-agent-memory/CLAUDE.md) for the two
Excalidraw style regimes (`clean` vs `graphic-recording`) and why they're kept
separate.

> This content was **copied**, not moved, from `modules/kg-agent-memory` on
> 2026-08-31, to unblock Module 3 work without disturbing that module's active
> development. It still needs deduplication/cleanup — Module 2's author should
> decide what stays there vs. here.

Run: `bash scripts/run_build_board.sh` (needs this module's own venv + a
reachable Neo4j holding the merged KG — see the script header).

## 2. Caption -> image -> Excalidraw

The other half of the "flux → excalidraw" idea: an LLM-written caption becomes
a raster image via FLUX.1-schnell, then gets embedded as an image element in a
`.excalidraw` scene, with the caption as a text element underneath.

```
caption ──▶ generate_image.py ──▶ PNG ──▶ image_to_excalidraw.py ──▶ .excalidraw
(dummy for now, KG-derived later)   (FLUX.1-schnell, needs a CUDA GPU)
```

- `generate_image.py` — caption in, PNG out. Caption is `DEFAULT_CAPTION` unless
  `--caption` is passed; wiring a real caption from the KG (e.g. a compressed
  fact headline, or a caption written against
  [`schemas/image-request.schema.json`](../../schemas/image-request.schema.json))
  is a caller change, not a change to this file.
- `image_to_excalidraw.py` — wraps one PNG + its caption into a minimal, valid
  `.excalidraw` scene (base64-embedded image element + text element).
- `scripts/run_generate_image.sh` — runs both steps back to back, timestamped
  output under `output/`.

Requires a CUDA GPU (FLUX.1-schnell is a 12B-parameter model) — run on the
department GPU cluster, not a laptop. Cluster submission wiring (partition,
paths, venv setup) is not done yet; see `modules/kg-agent-memory/scripts/` for
the existing `srun_submit.sh` pattern this module will likely reuse.

## Not built yet

- Placing a generated image *inside* the fact-card board (same canvas) rather
  than in its own separate `.excalidraw` file.
- Wiring a real caption from the KG instead of the dummy default.
- The MCP bridge from an LLM client (GPT/Claude) to Excalidraw for interactive,
  incremental edits — everything above is a batch script, not an agent loop.
- A reusable style/image library for reproducible visuals across renders.
- Real-time, partial-region updates (the actual "editable graphic recording"
  end goal) — today everything is a one-shot full render.

## Status

Board rendering: copied from module 2, functionally working there. Caption→image:
scripted, untested end-to-end (no local GPU to run FLUX.1-schnell against —
needs the department cluster). Image→Excalidraw: implemented and tested with a
synthetic image.
