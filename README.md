# convograph

A pipeline for turning multi-party spoken conversation into a temporal knowledge graph and an agent memory system (AMS), with an optional path to real-time visual output (graphic recording).

## Research question

How well can domain adaptation from agent memory be done from multi-party human conversation? Concretely: given what person X said over N weeks of conversation, filter the relevant subgraph and use it to adapt/ground an agent's memory — and quantify how well that adaptation works.

## Pipeline

```
audio (4 speakers)
  -> [1] ASR + diarization  ---------------->  speaker-tagged transcript
  -> [2] temporal KG + agent memory system --> temporal KG, quantitative ablation results
  -> [3] graphic generation ----------------->  caption -> image -> editable canvas (future)
```

Modules are independent and communicate through the data contracts in [`schemas/`](schemas/). See [`docs/architecture.md`](docs/architecture.md) for the full picture.

## Modules

| Module | Path | Status | Description |
|---|---|---|---|
| ASR + diarization | [`modules/asr-diarization`](modules/asr-diarization) | scaffold | Streaming 4-speaker diarization (NVIDIA Sortformer, DGX GB10) -> speaker-tagged script -> batch KG generation input. |
| Temporal KG + agent memory | [`modules/kg-agent-memory`](modules/kg-agent-memory) | **working, measured** | Transcript → temporal KG (Graphiti + Neo4j) with a bi-temporal `invalid_at` layer, evaluated on GroupMemBench. Full Finance domain ingested: 5,810 episodes / 111,258 facts / 18,450 superseded. Browsable live — see the module README. |
| Graphic generation | [`modules/graphic-generation`](modules/graphic-generation) | scaffold | LLM caption -> Flux Schnell image; planned MCP bridge to Excalidraw for an editable, real-time graphic-recording canvas. |

## Adding a new module

See [`scripts/import_module.sh`](scripts/import_module.sh) — handles both "here's a zip" and "here's a git repo" cases. Each module should stay self-contained (own README, own dependency file, own run scripts) and document how its inputs/outputs map to [`schemas/`](schemas/).

## Status

Early stage, modules land independently before full end-to-end assembly. A unifying UI is planned once the pipeline stabilizes (see [`pipeline/`](pipeline)).

## License

MIT — see [`LICENSE`](LICENSE).
