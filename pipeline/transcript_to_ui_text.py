"""
Hop 1 -> 2, second door: render a DiarizedTranscript as utterance lines.

Module 2 now has two entry points, and they eat different formats:

  * the benchmark loaders (baselines/rag_common/eval_lib.py) read the
    GroupMemBench JSON shape — that hop is transcript_to_kg_input.py;
  * ui_ingest.py (added with the Streamlit UI) reads PLAIN TEXT, one
    utterance per line ("Alice: we should lock the spec today"), and windows
    N lines into each Graphiti episode. The UI's text box is the same format.

This script renders the second one, reusing the same validation and
turn-merging as transcript_to_kg_input.py so both doors see identical turns.
To include roles, put them in the --speaker-map value:
    --speaker-map "speaker_1=Sarah (Project Manager),speaker_2=David"

Usage:

    python3 pipeline/transcript_to_ui_text.py meeting.transcript.json \
        --out meeting.ui_text.txt --speaker-map "speaker_1=Sarah"
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcript_to_kg_input import (  # noqa: E402
    merge_into_turns, parse_speaker_map, validate_transcript,
)


def render_lines(transcript, speaker_map=None, merge_gap=15.0):
    validate_transcript(transcript)
    speaker_map = speaker_map or {}
    turns = merge_into_turns(transcript["segments"], merge_gap)
    lines = []
    for turn in turns:
        if not turn["text"]:
            continue  # nothing said (or nothing transcribed) — no line
        who = speaker_map.get(turn["speaker"], turn["speaker"])
        lines.append(f"{who}: {turn['text']}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", help="DiarizedTranscript JSON from modules/asr-diarization")
    ap.add_argument("--out", default=None,
                    help="Output path (default: <transcript-stem>.ui_text.txt)")
    ap.add_argument("--speaker-map", default=None,
                    help="Rename speakers, roles allowed in the value: "
                         "'speaker_1=Sarah (Project Manager),speaker_2=David'")
    ap.add_argument("--merge-gap", type=float, default=15.0,
                    help="Same-speaker merge gap in seconds (default 15, "
                         "matching transcript_to_kg_input.py)")
    args = ap.parse_args()

    with open(args.transcript) as fh:
        transcript = json.load(fh)

    lines = render_lines(transcript, parse_speaker_map(args.speaker_map), args.merge_gap)
    if not lines:
        raise SystemExit("ERROR: no turns with text — run module 1's step 8 "
                         "(make transcribe) to get words first.")

    out_path = args.out
    if not out_path:
        stem = args.transcript
        for suffix in (".transcript.json", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        out_path = stem + ".ui_text.txt"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Lines : {len(lines)} utterances "
          f"(-> {max(1, -(-len(lines) // 12))} episode(s) at ui_ingest's default "
          f"12-line window)")
    print(f"Saved : {out_path}")


if __name__ == "__main__":
    main()
