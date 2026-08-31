"""
Hop 1 -> 2: convert a DiarizedTranscript into module 2's conversation format.

Why this file exists, and why it lives here
-------------------------------------------
Module 1 emits schemas/diarized-transcript.schema.json. Module 2's loaders
(baselines/rag_common/eval_lib.py::load_conversation_messages, and the
mem0/hipporag/graphiti baselines that share it) read something different:

    { "<channel name>": [ {"author": ..., "content": ..., "timestamp": ...,
                           "msg_node": ..., ...}, ... ] }

— the GroupMemBench corpus shape. Neither side should absorb the other's
format: the schema is the inter-module contract, and module 2's loader is
measured, working benchmark code owned by its author (see ADR 0002 for the
precedent, ADR 0003 for this decision). So the pipeline translates at the
hop, which is exactly what pipeline/ is for.

Two things are not a straight rename:

  * Granularity. Diarization emits a segment every few seconds of continuous
    speech; GroupMemBench messages are paragraph-sized contributions. Feeding
    raw segments to Graphiti would multiply LLM ingest cost for no benefit.
    So consecutive segments by the same speaker, up to --merge-gap seconds
    apart, are merged into one message — a "turn".
  * Empty text. A transcript without ASR text validates against the schema
    but would make module 2 burn LLM calls extracting facts from nothing.
    That is refused by default (--allow-empty-text to override, e.g. for
    plumbing tests).

Usage:

    python3 pipeline/transcript_to_kg_input.py meeting.transcript.json \
        --out meeting.kg_input.json \
        --speaker-map "speaker_1=Alice,speaker_2=Bob"

Verify the whole hop end-to-end (module 1 fixture -> module 2's own loader):

    python3 pipeline/check_handoff.py
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(REPO_ROOT, "schemas", "diarized-transcript.schema.json")


def validate_transcript(transcript):
    """Validate against the schema — with jsonschema if installed, otherwise
    with a hand check of the same constraints, so the pipeline has no hard
    third-party dependency."""
    try:
        import jsonschema
    except ImportError:
        jsonschema = None

    if jsonschema is not None:
        with open(SCHEMA_PATH) as fh:
            jsonschema.validate(transcript, json.load(fh))
    else:
        for key in ("conversation_id", "speakers", "segments"):
            if key not in transcript:
                raise SystemExit(f"ERROR: transcript is missing required key {key!r}")
        if not 1 <= len(transcript["speakers"]) <= 4:
            raise SystemExit("ERROR: schema allows 1-4 speakers, "
                             f"got {len(transcript['speakers'])}")
        for i, seg in enumerate(transcript["segments"]):
            for key in ("speaker", "text", "start_ts", "end_ts"):
                if key not in seg:
                    raise SystemExit(f"ERROR: segment {i} is missing {key!r}")

    # The schema itself cannot express this cross-field rule; check it here.
    speakers = set(transcript["speakers"])
    for i, seg in enumerate(transcript["segments"]):
        if seg["speaker"] not in speakers:
            raise SystemExit(f"ERROR: segment {i} speaker {seg['speaker']!r} "
                             f"is not in the transcript's speakers list")


def parse_speaker_map(arg):
    """'speaker_1=Alice,speaker_2=Bob' -> {'speaker_1': 'Alice', ...}"""
    mapping = {}
    if arg:
        for pair in arg.split(","):
            label, _, name = pair.partition("=")
            if not name:
                raise SystemExit(f"ERROR: --speaker-map entry {pair!r} is not "
                                 f"of the form label=Name")
            mapping[label.strip()] = name.strip()
    return mapping


def ts_gap_seconds(prev_end, next_start):
    """Gap between two ISO timestamps, in seconds."""
    from datetime import datetime
    return (datetime.fromisoformat(next_start)
            - datetime.fromisoformat(prev_end)).total_seconds()


def merge_into_turns(segments, merge_gap):
    """Consecutive same-speaker segments <= merge_gap seconds apart become one
    message. Text pieces join with a space; empty pieces vanish."""
    turns = []
    for seg in segments:
        if (turns
                and turns[-1]["speaker"] == seg["speaker"]
                and ts_gap_seconds(turns[-1]["end_ts"], seg["start_ts"]) <= merge_gap):
            last = turns[-1]
            last["end_ts"] = max(last["end_ts"], seg["end_ts"])
            last["texts"].append(seg["text"])
        else:
            turns.append({
                "speaker": seg["speaker"],
                "start_ts": seg["start_ts"],
                "end_ts": seg["end_ts"],
                "texts": [seg["text"]],
            })
    for turn in turns:
        turn["text"] = " ".join(t.strip() for t in turn["texts"] if t.strip())
        del turn["texts"]
    return turns


def convert(transcript, channel=None, speaker_map=None, merge_gap=15.0,
            allow_empty_text=False):
    """DiarizedTranscript dict -> {channel: [message, ...]} dict."""
    validate_transcript(transcript)

    channel = channel or transcript["conversation_id"]
    speaker_map = speaker_map or {}
    turns = merge_into_turns(transcript["segments"], merge_gap)

    if not any(t["text"] for t in turns) and not allow_empty_text:
        raise SystemExit(
            "ERROR: every segment has empty text — this transcript has diarization "
            "but no ASR words, so there is nothing for module 2 to ingest.\n"
            "Attach text in module 1 (07_export_transcript.py --asr-json) or pass "
            "--allow-empty-text if you only want to test the plumbing."
        )

    # Turns that stayed textless (untranscribed backchannels, or words the ASR
    # attributed to a concurrent speaker) would be dead weight in module 2: the
    # retrievers skip empty content and ingest would still pay for the episode.
    # Drop them — unless the caller asked to keep an all-empty plumbing test.
    if any(t["text"] for t in turns):
        kept = [t for t in turns if t["text"]]
        if len(kept) < len(turns):
            print(f"NOTE: dropped {len(turns) - len(kept)} turn(s) with no ASR text "
                  f"(nothing for module 2 to index there).", file=sys.stderr)
        turns = kept

    messages = []
    for i, turn in enumerate(turns):
        messages.append({
            # msg_node is zero-padded because module 2 sorts on it as a string.
            "msg_node": f"m{i:05d}",
            "author": speaker_map.get(turn["speaker"], turn["speaker"]),
            "content": turn["text"],
            "timestamp": turn["start_ts"],
            "end_timestamp": turn["end_ts"],
        })
    return {channel: messages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", help="DiarizedTranscript JSON from modules/asr-diarization")
    ap.add_argument("--out", default=None,
                    help="Output path (default: <transcript-stem>.kg_input.json)")
    ap.add_argument("--channel", default=None,
                    help="Channel name for module 2 (default: the conversation_id)")
    ap.add_argument("--speaker-map", default=None,
                    help="Rename speakers to real people, e.g. "
                         "'speaker_1=Alice,speaker_2=Bob'. Unmapped labels pass through.")
    ap.add_argument("--merge-gap", type=float, default=15.0,
                    help="Merge same-speaker segments up to this many seconds apart "
                         "into one message (default 15)")
    ap.add_argument("--allow-empty-text", action="store_true",
                    help="Accept a transcript whose segments have no ASR text")
    args = ap.parse_args()

    with open(args.transcript) as fh:
        transcript = json.load(fh)

    result = convert(
        transcript,
        channel=args.channel,
        speaker_map=parse_speaker_map(args.speaker_map),
        merge_gap=args.merge_gap,
        allow_empty_text=args.allow_empty_text,
    )

    out_path = args.out
    if not out_path:
        stem = args.transcript
        for suffix in (".transcript.json", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        out_path = stem + ".kg_input.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    (channel, messages), = result.items()
    n_segments = len(transcript["segments"])
    print(f"Channel  : {channel}")
    print(f"Messages : {n_segments} segments -> {len(messages)} speaker turns "
          f"(merge gap {args.merge_gap}s)")
    print(f"Authors  : {', '.join(sorted({m['author'] for m in messages}))}")
    print(f"Saved    : {out_path}")


if __name__ == "__main__":
    main()
