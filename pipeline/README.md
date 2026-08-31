# pipeline

Glue code between modules, validating each hop against the schemas in
[`schemas/`](../schemas). Also the planned home for the backend of an eventual
unifying UI.

## What exists

The module 1 -> module 2 hop:

| File | What it does |
|---|---|
| `transcript_to_kg_input.py` | Converts a schema-valid `DiarizedTranscript` (module 1's output) into the GroupMemBench conversation shape that module 2's loaders actually read (`{channel: [{author, content, timestamp, msg_node}, ...]}`). Merges diarization segments into speaker turns on the way, and refuses transcripts with no ASR text. See ADR 0003 for why this conversion lives here and not in either module. |
| `check_handoff.py` | End-to-end proof of the hop: runs a tracked diarization fixture through module 1's exporter, then this converter, then loads the result with module 2's **own** `eval_lib` loader and passage formatter. No GPU, no LLM, no cost. Run it after touching anything on either side of the hop. |

```bash
python3 pipeline/check_handoff.py
```

## What doesn't yet

The 2 -> 3 hop (temporal KG -> image requests) and the end-to-end runner.
