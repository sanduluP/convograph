# asr-diarization

Streaming speaker diarization — **who spoke when**, in real time, entirely on
your own machine — using NVIDIA's Streaming Sortformer model on a DGX Spark
(GB10). This is convograph's module 1: its job is to turn multi-party audio
into the speaker-tagged transcript that module 2 (`kg-agent-memory`) ingests.

- Output contract: [`schemas/diarized-transcript.schema.json`](../../schemas/diarized-transcript.schema.json),
  produced by `scripts/07_export_transcript.py` (step 7 below).
- Handoff to module 2: [`pipeline/transcript_to_kg_input.py`](../../pipeline/transcript_to_kg_input.py);
  prove the whole hop with `python3 pipeline/check_handoff.py` from the repo root.

## Status

Working end to end. Diarization (offline and live) with scored accuracy on
2/3/4-speaker mixes, plus speaker-attributed transcription — who said *what* —
via `make transcribe` (step 8), which pairs the same Sortformer diarizer with
NVIDIA's multitalker Parakeet ASR. Measured on the 5-minute AMI 2-speaker mix:
RTF 0.064 on the GB10, comfortably real-time capable. Main caveat: both models
are English-mostly — measure on German/mixed audio before relying on it.

## What Docker is doing here, in one paragraph

Your DGX Spark has an unusual combination: an ARM processor and a very new
Blackwell GPU. Most Python audio packages assume an Intel/AMD processor, so if
you install them normally they either fail outright or quietly fall back to
running on the CPU, which is roughly 30x slower. Docker lets you run inside a
prepared environment that NVIDIA built specifically for this hardware. You
build it once and then forget about it.

## Prerequisites

You almost certainly already have these on a Spark. Check with:

```bash
docker --version
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

The second command should print a table showing your GB10. If it errors, the
NVIDIA Container Toolkit is missing and nothing below will work.

You also need a free Hugging Face account, because the model is downloaded from
there. Get a token at https://huggingface.co/settings/tokens and set it:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

Add that line to your `~/.bashrc` so you don't have to repeat it.

## Setup

```bash
cd modules/asr-diarization
chmod +x scripts/02_prep_audio.sh
mkdir -p audio out
```

(`audio/` and `out/` are gitignored — recordings and generated output never go
into the repo; the small `.rttm` fixtures in `examples/` do, because
`pipeline/check_handoff.py` runs against them.)

### Step 1 — Build the image (once, roughly 15–30 minutes)

```bash
make build
```

Most of that time is downloading NVIDIA's base image, which is large. The build
deliberately fails at the last step if the GPU is not reachable, so if it
finishes, you are in good shape.

### Step 2 — Check the GPU

```bash
make verify
```

You want to see `GPU visible : True` and `GPU compute test: OK`.

If it says the GPU is not visible, the container could not reach the hardware —
that is a host setup problem, not a problem with this code.

### Step 3 — Prepare an audio file

Put a recording in the `audio/` folder, then convert it:

```bash
./scripts/02_prep_audio.sh audio/my_meeting.m4a
```

This creates `audio/my_meeting_16k.wav`. The model *only* accepts 16 kHz mono,
and gives unhelpful errors otherwise, so never skip this.

### Step 4 — Run diarization

```bash
make diarize FILE=audio/my_meeting_16k.wav
```

First run downloads the model (~500 MB). Output looks like:

```
      0.64s ->     12.30s   speaker_0
     12.30s ->     18.94s   speaker_1
     19.10s ->     31.02s   speaker_0
```

Results are also saved to `out/my_meeting_16k.rttm`.

### Step 5 — Try the real-time setting

Step 4 used the cheap `offline` preset. Once you are happy with the quality,
compare against the live one:

```bash
make diarize FILE=audio/my_meeting_16k.wav PRESET=realtime
```

Watch the **real-time factor** line at the bottom. Below 1.0 means the machine
processes audio faster than it plays, which is the requirement for live use.

### Step 6 — Live from a microphone

```bash
make mics                     # list microphones
make live                     # default mic; Ctrl-C to stop
make live MIC=plughw:1,0      # a specific one
```

The host records and pipes raw audio into the container; every ~1.5 s the model
re-runs over a rolling window and prints newly settled segments with stable
speaker names. Finalized segments are also appended to `out/live.rttm`
(override with `LIVE_RTTM=...`), so a live session can be exported in step 7
exactly like an offline run.

### Step 8 — Who said WHAT: speaker-attributed transcription

(Numbered 8 after the script name, but you will usually run it before step 7 —
it produces the files step 7 consumes.)

```bash
make transcribe FILE=audio/my_meeting_16k.wav
```

This runs the diarizer together with `nvidia/multitalker-parakeet-streaming-0.6b-v1`
(first run downloads ~2.5 GB), an ASR built to pair with Sortformer: it runs
one recognizer instance per active speaker, so it keeps transcribing correctly
even while people talk over each other. One run writes three files to `out/`:

- `<stem>.seglst.json` — sentence-level speaker-tagged transcript (SegLST format),
- `<stem>.words.json` — the `ASR=` input for step 7,
- `<stem>_mt.rttm` — diarization intervals from the *same* run, so speaker
  labels agree with the words file by construction (a separately-run step 4
  RTTM may number speakers differently — don't mix the two).

Two honest caveats. Sentences only break when a speaker goes quiet for
`--sent-break-sec` (default 4 s), so an uninterrupted monologue comes out as
one long block with one start timestamp — acceptable for the knowledge graph,
but don't expect sentence-level timing inside it. And it is English-trained,
like the diarizer.

### Step 7 — Export for the pipeline

This is the module's output boundary: RTTM in, schema-valid
`DiarizedTranscript` JSON out. Runs on the host — plain `python3`, no
container, no GPU.

```bash
make transcript RTTM=out/my_meeting_16k.rttm START=2026-08-31T14:00:00
# with transcribed words from some ASR source:
make transcript RTTM=out/my_meeting_16k.rttm START=2026-08-31T14:00:00 ASR=out/words.json
```

Three translations happen here (see the script's docstring for detail):
model speaker labels (`speaker_0..3` offline, `A/B/C/D` live) become canonical
`speaker_1..4`; float seconds become wall-clock ISO timestamps anchored at
`START` (module 2 orders and supersedes facts by absolute time — pass the real
recording start); and per-segment text is attached from `ASR=` — normally
the `.words.json` from step 8, or any other source producing
`{"start", "end", "text", "speaker"?}` items (see "Other ASR sources"
below). Without `ASR=` the transcript is schema-valid but empty of words,
and the pipeline will refuse to feed it to module 2.

From the repo root, the resulting JSON goes to module 2 via:

```bash
python3 pipeline/transcript_to_kg_input.py out/my_meeting_16k.transcript.json \
    --speaker-map "speaker_1=Alice,speaker_2=Bob"
```

## Testing with 2, 3 and 4 speakers

Rather than downloading a whole benchmark corpus (whose published numbers you
can already read on the model card), this uses **one real meeting, split three
ways**.

AMI recorded each of four participants on their own headset microphone. We
download those four channels and add them together in different combinations:

- `2spk.wav` = participants 0 + 1
- `3spk.wav` = participants 0 + 1 + 2
- `4spk.wav` = all four

Same room, same voices, same recording conditions. The *only* thing that
changes is how many people are talking, which is exactly the variable you care
about. A comparison across different corpora would also change the acoustics,
the microphones and the speaking style, and you would not know which of those
caused any difference you saw.

```bash
make getexamples      # ~200 MB, host-side, a few minutes
make build            # rebuild once, to add the scoring library
make examples         # builds the three mixes + ground-truth labels
make all-examples     # runs and scores all three
```

Output per file looks like:

```
--- Accuracy (collar 0.25s) ---
  Missed speech    :   3.11%
  False alarm      :   1.902%
  Wrong speaker    :   6.44%
  TOTAL ERROR (DER):  11.45%
  Reference had 3 speakers, model found 3
```

**DER** is the share of audio time labelled wrong; lower is better. The three
components tell you *what kind* of wrong: missed speech means it heard silence
where someone was talking, false alarm the reverse, and wrong speaker means it
heard the speech but credited the wrong person. On a 4-speaker mix, a jump in
**wrong speaker** is the signature of running out of capacity.

Then compare against the live-latency setting:

```bash
make all-examples PRESET=realtime
```

A small gap between the two runs means real-time costs you little. A large gap
means you should consider trading a few seconds of delay for accuracy.

Use `make getexamples MINUTES=10` if five minutes feels too short to judge.

### Where the ground truth comes from

When person A talks, their own headset mic is loud and everyone else's picks up
a faint distant version. So for each 20 ms slice we ask which channel is
loudest, and that tells us who was speaking. This is standard practice for
close-talking meeting corpora.

It is very good but not perfect — it can miss very quiet speech and sometimes
clips the start of a word. So treat these numbers as a reliable guide for
comparing conditions against each other, not as a publishable benchmark figure.

## The two things most likely to disappoint you

**1. Four speakers, maximum.** This model tracks at most 4 people. With 5 or
more, it merges some of them together and accuracy drops sharply. This is a
hard architectural limit, not a setting you can raise (and the reason the
transcript schema caps `speakers` at 4). Test this before you build anything
on top.

**2. It is trained mostly on English.** It works on other languages but less
well. If your recordings are German, or mixed German and English, measure it
rather than assuming.

Both of these are why Step 4 comes before any live-streaming work. Find out
whether the model handles *your* audio before investing in the plumbing.

## Common errors

| What you see | What it means |
|---|---|
| `CUDA is not available` | You dropped `--gpus all`, or you are running outside the container |
| `Input shape mismatch` / tensor shape error | Audio is stereo or not 16 kHz — re-run step 3 |
| `object.__init__() takes exactly one argument` | PyTorch/Lhotse version clash — you changed the base image away from `25.10` |
| `401` or `gated repo` when downloading | `HF_TOKEN` is not set, or you have not accepted the model licence on Hugging Face |
| Build hangs on `pip install` | Normal. NeMo has many dependencies; give it 10+ minutes |
| `CUDA error: out of memory` while `free -g` shows plenty *available* | GB10 unified-memory quirk: CUDA won't reclaim page cache (e.g. another model server's mmap'd weights). Fix: `sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`. Module 1 itself needs ~1 GB (diarize) / ~5.5 GB (transcribe) — measured |

## Other ASR sources

Step 8 is the built-in way to get words, but step 7's `--asr-json` interface
deliberately accepts ANY transcription source: a JSON list of
`{"start", "end", "text", "speaker"?}` items. That matters for one concrete
reason — the multitalker model is English-trained, so for German or mixed
German/English meetings you may get better words from e.g. Whisper, converted
into that shape, while keeping Sortformer's diarization. Include the `speaker`
field when the source is speaker-attributed — without it, words are assigned
to segments by time overlap alone, which guesses wrong wherever two people
talk at once.

Implementation note: step 8 is adapted from NVIDIA's official example
(`examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py`
in the NeMo repo). NeMo also has a "serial" single-stream strategy there with
finer sentence timing; at our pinned NeMo version it collapsed all speech onto
one speaker when tested, so step 8 uses the parallel strategy only.
