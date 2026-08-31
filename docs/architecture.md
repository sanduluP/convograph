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

- `schemas/diarized-transcript.schema.json` — output of module 1 (via `scripts/07_export_transcript.py`). Module 2's loaders predate this schema and read the GroupMemBench corpus shape instead, so the pipeline converts at the hop: `pipeline/transcript_to_kg_input.py`, verified end-to-end by `pipeline/check_handoff.py` (see ADR 0003).
- `schemas/temporal-kg.schema.json` — output of module 2 (the temporal KG itself, plus per-person/time-window subgraph filters used for the domain-adaptation RQ).
- `schemas/image-request.schema.json` — input to module 3 (an LLM-generated caption plus a reference to the KG subgraph it was derived from).

## Module 2 detail (kg-agent-memory)

Landed from an existing benchmark repo (GroupMemBench). It evaluates group-conversation memory systems on synthetic multi-channel conversations across four domains (Finance, Technology, Healthcare, Manufacturing), comparing a lexical retriever (BM25), a dense retriever (`text-embedding-3-large`), and a Graphiti-based temporal-KG retriever, all feeding the same QA agent + judge. This produces the quantitative ablation results for the RQ: how well domain adaptation from agent memory holds up across question types (multi-hop, temporal, knowledge-update, user-implicit, term-ambiguity, abstention).

Large per-domain data files (`modules/kg-agent-memory/data/final/**/*.json`, ~150MB total) are tracked with Git LFS — see that module's `.gitattributes`.

## Module 1 detail (asr-diarization)

Landed from the standalone sortformer-spark work: Dockerized streaming diarization (NVIDIA Streaming Sortformer, `diar_streaming_sortformer_4spk-v2.1`) on the DGX Spark GB10, offline and live-microphone, with DER scoring against AMI-derived 2/3/4-speaker fixtures. The ASR half runs `nvidia/multitalker-parakeet-streaming-0.6b-v1` alongside the same diarizer (one recognizer instance per active speaker, so overlapped speech transcribes correctly; RTF ~0.06 on the GB10). Transcript text can alternatively come from any external ASR via the module's `--asr-json` interface — relevant for German/mixed-language audio, where the English-trained multitalker model may underperform. See that module's README.

## Pipeline / UI

[`pipeline/`](../pipeline) holds the glue between modules. The 1 -> 2 hop exists (`transcript_to_kg_input.py` + `check_handoff.py`); the 2 -> 3 hop, the end-to-end runner, and the backend of an eventual UI that assembles/edits the graphic recording in real time (not regenerating the whole image on every update — only the affected subgraph/region) are still to come.

## Decisions

See [`docs/decisions/`](decisions) for lightweight architecture decision records, starting with why this is a monorepo rather than git submodules.
