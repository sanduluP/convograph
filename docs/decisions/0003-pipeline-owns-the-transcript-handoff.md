# 0003: The pipeline, not either module, converts transcripts for module 2

## Status

Accepted (2026-08-31).

## Context

Module 1 (asr-diarization, landed from the standalone sortformer-spark work)
emits `schemas/diarized-transcript.schema.json`: `{conversation_id, speakers,
segments[{speaker, text, start_ts, end_ts}]}`. Module 2's loaders
(`baselines/rag_common/eval_lib.py::load_conversation_messages`, shared by the
mem0/hipporag/graphiti baselines) read something else: the GroupMemBench corpus
shape, `{channel: [{author, content, timestamp, msg_node, ...}]}`. The
architecture docs said "module 1's output is module 2's input", but the two
formats never actually matched — someone had to own the translation.

There is also a granularity mismatch: diarization emits a segment every few
seconds of continuous speech, while GroupMemBench messages are paragraph-sized
(Finance alone is 5,810 episodes). Feeding raw segments to Graphiti would
multiply LLM ingest cost for nothing.

## Decision

Neither module changes. `pipeline/transcript_to_kg_input.py` converts at the
hop: schema-valid transcript in, module 2's native conversation shape out,
merging consecutive same-speaker segments into turns on the way.
`pipeline/check_handoff.py` locks the contract by loading the converter's
output with module 2's own code.

## Rationale

- The schema stays the inter-module contract — it is the right shape for a
  diarized transcript, and module 3 or a future UI may read it too.
- Module 2's loader is measured, working benchmark code owned by its author;
  changing its input format would invalidate the "byte-for-byte identical to
  the baselines" property its comments promise (same reasoning as ADR 0002:
  don't rework another author's live code for an integration convenience).
- Format translation between modules is exactly what `pipeline/` was reserved
  for, and this makes the 1 -> 2 hop its first real (and testable) content.

## Revisit if

Module 2 grows a native `DiarizedTranscript` ingest path, or real diarized
meetings (rather than GroupMemBench synthetic corpora) become module 2's
primary input — then the converter's turn-merging heuristics (merge gap,
empty-turn dropping) should be re-tuned against real ASR output, and the
speaker-attribution caveat in module 1's `--asr-json` revisited.
