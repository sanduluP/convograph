# graphic-generation

Turns a temporal-KG subgraph into a visual: LLM generates a caption, Flux Schnell renders the image.

- Input: [`schemas/image-request.schema.json`](../../schemas/image-request.schema.json) (caption + subgraph reference).
- Ideas in flight:
  - MCP bridge from an LLM client (GPT/Claude) to Excalidraw (open source) for a canvas-based, inspectable/editable output instead of a flat raster image.
  - A reusable style/image library so repeated generations stay visually reproducible.
  - End goal: a real-time, *editable* graphic recording — not one static image regenerated wholesale on every update, but targeted edits to the affected region/subgraph (important because an hour of conversation isn't fully agreed-upon as it happens, so early drawn content must stay revisable).

## Status

Scaffold only — caption -> Flux Schnell generation is solved in prototype form; canvas/editability work (Excalidraw MCP) is upcoming.

## Adding this module's code

Drop the module in directly under this folder (own README, own dependency file, own run scripts), or use [`scripts/import_module.sh`](../../scripts/import_module.sh) if importing from a zip or existing git repo.
