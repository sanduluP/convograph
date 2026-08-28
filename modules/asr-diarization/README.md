# asr-diarization

Streaming speech-to-text with 4-speaker diarization.

- Package: NVIDIA Sortformer (DGX GB10).
- Mode: streaming.
- Output: speaker-tagged script (speaker 1-4), conforming to [`schemas/diarized-transcript.schema.json`](../../schemas/diarized-transcript.schema.json).
- Also feeds batch KG generation.

## Status

Scaffold only — implementation to be added.

## Adding this module's code

Drop the module in directly under this folder (own README, own dependency file, own run scripts), or use [`scripts/import_module.sh`](../../scripts/import_module.sh) if importing from a zip or existing git repo. Update the "Status" line above once real code lands, and note the audio input corpus used (e.g. a speech-to-translate / speaker-diarization dataset) here.
