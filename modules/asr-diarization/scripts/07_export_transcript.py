"""
Step 7: Export diarization output as a DiarizedTranscript JSON.

This is the module's OUTPUT boundary. Everything before this step speaks RTTM
(the diarization community's format: float seconds from the start of the
recording, model-invented speaker labels). Everything after this step —
convograph's pipeline/ and modules/kg-agent-memory — speaks
schemas/diarized-transcript.schema.json. This script is the one place where
that translation happens, so the other scripts never need to know the schema
exists, and the pipeline never needs to know what RTTM is.

Three translations happen here:

  1. Speaker labels become canonical. The offline script emits speaker_0..3,
     the live script emits A/B/C/D — the schema wants speaker_1..4. Labels are
     renamed in order of first appearance.
  2. Float seconds become wall-clock timestamps. RTTM says "12.30s into the
     file"; the knowledge graph needs "2026-08-31T14:00:12Z", because module 2
     orders and supersedes facts by absolute time. You anchor this with
     --session-start (when the recording actually began).
  3. Words are attached, if you have them. Sortformer only labels WHO speaks;
     it produces no words. Normally they come from step 8
     (scripts/08_transcribe_multitalker.py), whose .words.json plugs straight
     into --asr-json — but --asr-json deliberately accepts any transcription
     source: a JSON list of
     {"start": sec, "end": sec, "text": "...", "speaker": optional} items,
     each assigned to the diarization segment it overlaps most — restricted
     to that speaker's segments when the item names one, which a
     speaker-attributed ASR should, or words land on the wrong person
     wherever two people talk at once. Without it, segments carry empty
     text and this script warns you — the transcript is structurally valid but
     useless for knowledge-graph ingestion.

Runs on the host, stdlib only — no Docker, no GPU.

Usage:

    python3 scripts/07_export_transcript.py out/my_meeting_16k.rttm \
        --session-start 2026-08-31T14:00:00 \
        --asr-json out/my_meeting_words.json \
        --out out/my_meeting.transcript.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

MAX_SPEAKERS = 4  # hard model limit, and the schema's maxItems


def read_rttm(path):
    """Parse RTTM into a time-sorted list of (start, end, label)."""
    segments = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start, dur, label = float(parts[3]), float(parts[4]), parts[7]
            if dur > 0:
                segments.append((start, start + dur, label))
    segments.sort()
    return segments


def merge_gaps(segments, max_gap):
    """Join consecutive segments of the same speaker separated by a short
    pause. The model tends to split one sentence across several RTTM lines at
    every breath; downstream nobody cares about breaths."""
    merged = []
    for start, end, label in segments:
        if merged and merged[-1][2] == label and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]), label)
        else:
            merged.append((start, end, label))
    return merged


def canonical_labels(segments):
    """Rename model labels to speaker_1..speaker_4 in order of first
    appearance, so downstream never sees speaker_0 vs 'A' inconsistencies."""
    mapping = {}
    for _, _, label in segments:
        if label not in mapping:
            mapping[label] = f"speaker_{len(mapping) + 1}"
    if len(mapping) > MAX_SPEAKERS:
        raise SystemExit(
            f"\nERROR: found {len(mapping)} speaker labels ({', '.join(mapping)}).\n"
            f"The model tracks at most {MAX_SPEAKERS} speakers and the schema allows no more.\n"
            f"More than {MAX_SPEAKERS} usually means the live label anchor lost track "
            f"(names like S4, S5) — re-run diarization on the recording with "
            f"scripts/03_diarize.py instead.\n"
        )
    return mapping


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def attach_text(segments, asr_items):
    """Give each ASR item's text to the diarization segment it overlaps most.

    If an item carries a "speaker" field (the model's own label, as a
    speaker-attributed ASR like multitalker-parakeet emits), only segments of
    that speaker are candidates. Without it, overlap alone decides — which is
    ambiguous exactly when two people talk at once, so expect some words on
    the wrong speaker in overlapped regions. Items that match no segment are
    reported, not silently dropped."""
    texts = [[] for _ in segments]
    orphans = 0
    for item in asr_items:
        i_start, i_end = float(item["start"]), float(item["end"])
        want_label = item.get("speaker")
        best, best_ov = None, 0.0
        for idx, (s, e, label) in enumerate(segments):
            if want_label is not None and label != want_label:
                continue
            ov = overlap(i_start, i_end, s, e)
            if ov > best_ov:
                best, best_ov = idx, ov
        if best is None:
            orphans += 1
        else:
            texts[best].append((i_start, str(item["text"])))
    if orphans:
        print(f"WARNING: {orphans} ASR item(s) matched no diarization segment "
              f"and were dropped.", file=sys.stderr)
    # Within a segment, keep the words in spoken order.
    return [" ".join(t for _, t in sorted(chunks)) for chunks in texts]


def resolve_session_start(arg, rttm_path):
    if arg:
        try:
            dt = datetime.fromisoformat(arg)
        except ValueError:
            raise SystemExit(f"ERROR: --session-start {arg!r} is not ISO 8601 "
                             f"(expected e.g. 2026-08-31T14:00:00)")
        # A naive time means "my local clock"; make it explicit, then use UTC.
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return dt.astimezone(timezone.utc)
    # Fall back to the RTTM file's mtime. That is when diarization RAN, which
    # for a live session is roughly right and for an old recording is wrong —
    # hence the warning. Absolute times only need to be *consistent* for
    # module 2's ordering to work, but supersession across sessions cares.
    dt = datetime.fromtimestamp(os.path.getmtime(rttm_path), tz=timezone.utc)
    print(f"WARNING: no --session-start given; anchoring wall-clock times to the "
          f"RTTM file's mtime ({dt.isoformat(timespec='seconds')}). Pass the real "
          f"recording start time if you have it.", file=sys.stderr)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rttm", help="RTTM from scripts/03_diarize.py or scripts/06_live.py")
    ap.add_argument("--out", default=None,
                    help="Output path (default: <rttm-stem>.transcript.json next to the RTTM)")
    ap.add_argument("--session-start", default=None,
                    help="ISO 8601 wall-clock time the recording started, e.g. "
                         "2026-08-31T14:00:00 (naive = local time). Default: RTTM mtime.")
    ap.add_argument("--asr-json", default=None,
                    help="JSON list of {start, end, text} transcription items to "
                         "attach to segments by time overlap")
    ap.add_argument("--conversation-id", default=None,
                    help="Stable id for this session (default: RTTM filename stem)")
    ap.add_argument("--merge-gap", type=float, default=1.0,
                    help="Join same-speaker segments separated by pauses up to this "
                         "many seconds (default 1.0)")
    args = ap.parse_args()

    segments = read_rttm(args.rttm)
    if not segments:
        raise SystemExit(f"ERROR: no SPEAKER lines found in {args.rttm}")
    raw_count = len(segments)
    segments = merge_gaps(segments, args.merge_gap)
    mapping = canonical_labels(segments)

    if args.asr_json:
        with open(args.asr_json) as fh:
            texts = attach_text(segments, json.load(fh))
    else:
        texts = [""] * len(segments)
        print("WARNING: no --asr-json given. Segments will have empty text — valid "
              "against the schema, but module 2 has nothing to ingest. Sortformer "
              "does not transcribe; get words from scripts/08_transcribe_multitalker.py "
              "(make transcribe).", file=sys.stderr)

    t0 = resolve_session_start(args.session_start, args.rttm)

    def ts(seconds):
        return (t0 + timedelta(seconds=seconds)).isoformat(timespec="milliseconds")

    conversation_id = args.conversation_id or os.path.splitext(os.path.basename(args.rttm))[0]
    transcript = {
        "conversation_id": conversation_id,
        "speakers": sorted(set(mapping.values()), key=lambda s: int(s.rsplit("_", 1)[1])),
        "segments": [
            {
                "speaker": mapping[label],
                "text": text,
                "start_ts": ts(start),
                "end_ts": ts(end),
            }
            for (start, end, label), text in zip(segments, texts)
        ],
    }

    out_path = args.out or os.path.splitext(args.rttm)[0] + ".transcript.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(transcript, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    renames = ", ".join(f"{old} -> {new}" for old, new in mapping.items())
    print(f"Speakers  : {renames}")
    print(f"Segments  : {raw_count} RTTM lines -> {len(segments)} after merging "
          f"pauses <= {args.merge_gap}s")
    print(f"Anchored  : {transcript['segments'][0]['start_ts']} .. "
          f"{transcript['segments'][-1]['end_ts']}")
    print(f"Saved     : {out_path}")


if __name__ == "__main__":
    main()
