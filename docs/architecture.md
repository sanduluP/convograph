# Architecture

## Pipeline

```
                 ┌─────────────────────────┐
  audio (4 spk)  │ 1. asr-diarization       │  diarized-transcript.schema.json
  ─────────────▶ │ Sortformer, DGX GB10     │ ─────────────────────────────────┐
                 │ streaming                │                                  │
                 └─────────────────────────┘                                  ▼
                                                                ┌─────────────────────────┐
                                                                │ 2. kg-agent-memory       │  temporal-kg.schema.json
                                                                │ temporal KG + AMS        │ ──────────────────────┐
                                                                │ (GroupMemBench harness)  │                        │
                                                                └─────────────────────────┘                        ▼
                                                                                                     ┌─────────────────────────┐
                                                                                                     │ 3. graphic-generation    │
                                                                                                     │ caption -> Flux Schnell  │
                                                                                                     │ -> editable canvas (MCP  │
                                                                                                     │    to Excalidraw)        │
                                                                                                     └─────────────────────────┘
```

## Module boundaries

Modules do not call into each other's internals. Each module reads/writes JSON conforming to a schema in [`schemas/`](../schemas), so modules can be developed, tested, and swapped independently:

- `schemas/diarized-transcript.schema.json` — output of module 1, input to module 2.
- `schemas/temporal-kg.schema.json` — output of module 2 (the temporal KG itself, plus per-person/time-window subgraph filters used for the domain-adaptation RQ).
- `schemas/image-request.schema.json` — input to module 3 (an LLM-generated caption plus a reference to the KG subgraph it was derived from).

## Module 2 detail (kg-agent-memory)

Landed from an existing benchmark repo (GroupMemBench). It evaluates group-conversation memory systems on synthetic multi-channel conversations across four domains (Finance, Technology, Healthcare, Manufacturing), comparing a lexical retriever (BM25), a dense retriever (`text-embedding-3-large`), and a Graphiti-based temporal-KG retriever, all feeding the same QA agent + judge. This produces the quantitative ablation results for the RQ: how well domain adaptation from agent memory holds up across question types (multi-hop, temporal, knowledge-update, user-implicit, term-ambiguity, abstention).

Large per-domain data files (`modules/kg-agent-memory/data/final/**/*.json`, ~150MB total) are tracked with Git LFS — see that module's `.gitattributes`.

## Future: pipeline / UI

[`pipeline/`](../pipeline) is the planned home for the glue code that runs modules 1 -> 2 -> 3 end-to-end and for the backend of an eventual UI that assembles/edits the graphic recording in real time (not regenerating the whole image on every update — only the affected subgraph/region).

## Decisions

See [`docs/decisions/`](decisions) for lightweight architecture decision records, starting with why this is a monorepo rather than git submodules.
