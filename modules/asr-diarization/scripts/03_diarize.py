"""
Step 3: Run streaming speaker diarization ("who spoke when") on an audio file.

This downloads NVIDIA's Streaming Sortformer model the first time you run it
(about 500 MB), then labels every stretch of speech with a speaker number.

Usage inside the container:

    python scripts/03_diarize.py audio/my_recording_16k.wav
    python scripts/03_diarize.py audio/my_recording_16k.wav --preset offline

The two presets matter a lot, so they are explained below.
"""

import argparse
import os
import time
import wave

from nemo.collections.asr.models import SortformerEncLabelModel

# ---------------------------------------------------------------------------
# The presets come straight from NVIDIA's model card.
#
# All numbers are counted in "frames", where 1 frame = 80 milliseconds.
#
#   realtime : ~1.04 seconds of delay before you get a speaker label.
#              This is what you want for a LIVE system.
#              It costs ~46x more compute than the offline preset.
#
#   offline  : ~30.4 seconds of delay. Useless live, but very cheap and
#              slightly more accurate. Use this when you are diarizing
#              recordings you already have, or when you just want to
#              test whether the model handles your audio at all.
#
# Start with 'offline' to check quality. Switch to 'realtime' only once
# you know the model works on your recordings.
# ---------------------------------------------------------------------------
PRESETS = {
    "realtime": dict(
        chunk_len=6,
        chunk_right_context=7,
        fifo_len=188,
        spkcache_update_period=144,
        spkcache_len=188,
    ),
    "offline": dict(
        chunk_len=340,
        chunk_right_context=40,
        fifo_len=40,
        spkcache_update_period=300,
        spkcache_len=188,
    ),
}

MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2.1"


def wav_duration_seconds(path: str) -> float:
    """Read the length of a WAV file, and sanity-check its format."""
    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        rate = w.getframerate()
        seconds = w.getnframes() / float(rate)

    if channels != 1 or rate != 16000:
        raise SystemExit(
            f"\nERROR: {path} is {channels}-channel at {rate} Hz.\n"
            f"Sortformer needs 1-channel (mono) at 16000 Hz.\n"
            f"Fix it with:  ./scripts/02_prep_audio.sh {path}\n"
        )
    return seconds


def parse_segment(seg):
    """
    Turn one result from the model into (start, end, speaker).

    NeMo has returned these as plain strings ("0.64 3.12 speaker_0") in some
    versions and as tuples in others, so we handle both rather than crashing.
    """
    if isinstance(seg, str):
        parts = seg.split()
        return float(parts[0]), float(parts[1]), parts[2]
    start, end, spk = seg[0], seg[1], seg[2]
    return float(start), float(end), str(spk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to a 16 kHz mono WAV file")
    ap.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="offline",
        help="'offline' to test quality cheaply, 'realtime' for live use",
    )
    ap.add_argument(
        "--rttm",
        default=None,
        help="Optional path to save results in RTTM format (for scoring later)",
    )
    ap.add_argument(
        "--ref",
        default=None,
        help="Optional ground-truth RTTM. If given, prints how accurate the "
             "result was (Diarization Error Rate).",
    )
    ap.add_argument(
        "--collar",
        type=float,
        default=0.25,
        help="Seconds of forgiveness around each speaker change. 0.25 is the "
             "usual convention; use 0.0 for a strict score.",
    )
    args = ap.parse_args()

    duration = wav_duration_seconds(args.audio)
    print(f"Audio     : {args.audio}  ({duration:.1f} seconds)")
    print(f"Preset    : {args.preset}")
    print("Loading model (first run downloads ~500 MB)...")

    model = SortformerEncLabelModel.from_pretrained(MODEL_NAME)
    model.eval()
    model = model.to("cuda")

    # Apply the chosen latency preset.
    for key, value in PRESETS[args.preset].items():
        setattr(model.sortformer_modules, key, value)
    # Ask NeMo to confirm the numbers are internally consistent.
    model.sortformer_modules._check_streaming_parameters()

    print("Running diarization...\n")
    started = time.perf_counter()
    results = model.diarize(audio=[args.audio], batch_size=1)
    elapsed = time.perf_counter() - started

    segments = [parse_segment(s) for s in results[0]]

    speakers = sorted({spk for _, _, spk in segments})
    for start, end, spk in segments:
        print(f"  {start:8.2f}s -> {end:8.2f}s   {spk}")

    print(f"\nSpeakers found : {len(speakers)}  ({', '.join(speakers)})")
    print(f"Segments       : {len(segments)}")
    print(f"Processing time: {elapsed:.1f}s for {duration:.1f}s of audio")

    # RTF = "real time factor". Below 1.0 means it processes faster than
    # the audio plays, which is the minimum requirement for live use.
    rtf = elapsed / duration if duration else float("nan")
    print(f"Real-time factor: {rtf:.3f}  ({'OK for live' if rtf < 1 else 'TOO SLOW for live'})")

    if len(speakers) >= 4:
        print(
            "\nNOTE: this model tops out at 4 speakers. You hit the ceiling, "
            "so if your recording really had 5+ people, some of them were merged."
        )

    if args.ref:
        # Only imported when needed, so the script still runs without
        # the scoring library installed.
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        def to_annotation(items, from_rttm):
            ann = Annotation()
            for item in items:
                if from_rttm:
                    parts = item.split()
                    if not parts or parts[0] != "SPEAKER":
                        continue
                    s, dur, spk = float(parts[3]), float(parts[4]), parts[7]
                    e = s + dur
                else:
                    s, e, spk = item
                if e > s:
                    ann[Segment(s, e)] = spk
            return ann

        with open(args.ref) as fh:
            reference = to_annotation(fh.readlines(), from_rttm=True)
        hypothesis = to_annotation(segments, from_rttm=False)

        metric = DiarizationErrorRate(collar=args.collar, skip_overlap=False)
        der = metric(reference, hypothesis, detailed=True)

        total = der["total"] or 1.0
        print(f"\n--- Accuracy (collar {args.collar}s) ---")
        print(f"  Missed speech    : {100 * der['missed detection'] / total:6.2f}%")
        print(f"  False alarm      : {100 * der['false alarm'] / total:6.2f}%")
        print(f"  Wrong speaker    : {100 * der['confusion'] / total:6.2f}%")
        print(f"  TOTAL ERROR (DER): {100 * der['diarization error rate']:6.2f}%")
        print(f"  Reference had {len(reference.labels())} speakers, "
              f"model found {len(hypothesis.labels())}")

    if args.rttm:
        os.makedirs(os.path.dirname(args.rttm) or ".", exist_ok=True)
        file_id = os.path.splitext(os.path.basename(args.audio))[0]
        with open(args.rttm, "w") as fh:
            for start, end, spk in segments:
                fh.write(
                    f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} "
                    f"<NA> <NA> {spk} <NA> <NA>\n"
                )
        print(f"\nSaved RTTM: {args.rttm}")


if __name__ == "__main__":
    main()