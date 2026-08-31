"""
Step 6: Live speaker diarization from a microphone.

How this works
--------------
Audio arrives on standard input as raw 16 kHz mono PCM (the host records it
and pipes it in, which avoids fighting with sound drivers inside Docker).

We keep a rolling window of the last WINDOW_S seconds. Every HOP_S seconds we
re-run the model over that whole window and print who was speaking in the
newly-arrived slice.

Re-running over a window rather than feeding a continuous stream is the
pragmatic choice: it uses only the documented API, and because the cheap
preset processes 30 seconds of audio in well under a second on this hardware,
your delay comes from waiting for audio to arrive, not from the model.

The one thing this must handle carefully is that the model numbers speakers
per call -- "speaker_0" in one window is not necessarily "speaker_0" in the
next. So after each run we match the new speakers against the previous ones by
how much they overlap in time, and keep stable names (A, B, C, D).

Usage (run this from the HOST, it pipes into the container):

    arecord -f S16_LE -r 16000 -c 1 -t raw | \
      docker run --rm -i --gpus all --ipc=host \
        -v $HOME/.cache/huggingface:/root/.cache/huggingface \
        -v $PWD:/work -w /work sortformer-spark \
        python scripts/06_live.py

Or simply:  make live

Stop with Ctrl-C.
"""

import argparse
import sys
import time

import numpy as np

from nemo.collections.asr.models import SortformerEncLabelModel

SAMPLE_RATE = 16000
MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2.1"

PRESETS = {
    "realtime": dict(
        chunk_len=6, chunk_right_context=7, fifo_len=188,
        spkcache_update_period=144, spkcache_len=188,
    ),
    "offline": dict(
        chunk_len=340, chunk_right_context=40, fifo_len=40,
        spkcache_update_period=300, spkcache_len=188,
    ),
}

NAMES = ["A", "B", "C", "D", "E", "F"]


def parse_segments(raw):
    out = []
    for seg in raw:
        if isinstance(seg, str):
            parts = seg.split()
            start, end, spk = float(parts[0]), float(parts[1]), parts[2]
        else:
            start, end, spk = float(seg[0]), float(seg[1]), str(seg[2])
        if end > start:
            out.append((start, end, spk))
    return out


def overlap(a, b):
    """Seconds of shared time between two (start, end) intervals."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


class LabelAnchor:
    """
    Keeps speaker names stable across windows.

    Each time the model returns a fresh set of local labels, we compare them
    against what we saw last time in the region the two windows share. The
    local label that overlaps most with a known speaker inherits that name.
    """

    def __init__(self):
        self.known = {}          # stable name -> list of (start, end) absolute
        self.next_index = 0

    def _new_name(self):
        name = NAMES[self.next_index] if self.next_index < len(NAMES) \
            else f"S{self.next_index}"
        self.next_index += 1
        return name

    def assign(self, segments, shared_from, shared_to):
        """segments: list of (start, end, local_label) in absolute seconds."""
        local_groups = {}
        for start, end, local in segments:
            local_groups.setdefault(local, []).append((start, end))

        # Score every (local, known) pair by shared speaking time.
        scores = []
        for local, local_segs in local_groups.items():
            for name, known_segs in self.known.items():
                total = 0.0
                for ls in local_segs:
                    if ls[1] <= shared_from or ls[0] >= shared_to:
                        continue
                    for ks in known_segs:
                        total += overlap(ls, ks)
                if total > 0:
                    scores.append((total, local, name))

        scores.sort(reverse=True)
        mapping = {}
        used_names = set()
        for _, local, name in scores:
            if local in mapping or name in used_names:
                continue
            mapping[local] = name
            used_names.add(name)

        for local in local_groups:
            if local not in mapping:
                mapping[local] = self._new_name()

        # Remember recent activity per stable name, for the next comparison.
        fresh = {}
        for local, segs in local_groups.items():
            fresh.setdefault(mapping[local], []).extend(segs)
        for name, segs in fresh.items():
            combined = self.known.get(name, []) + segs
            # Only the recent past is useful for matching; drop the rest.
            self.known[name] = [s for s in combined if s[1] > shared_from][-200:]

        return [(s, e, mapping[l]) for s, e, l in segments]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="offline",
                    help="'offline' is cheaper and works well here because "
                         "latency comes from the window, not the model")
    ap.add_argument("--window", type=float, default=30.0,
                    help="Seconds of context the model sees each time")
    ap.add_argument("--hop", type=float, default=1.5,
                    help="How often to produce output, in seconds")
    ap.add_argument("--warmup", type=float, default=5.0,
                    help="Seconds to collect before the first result")
    ap.add_argument("--rttm", default=None,
                    help="Also append finalized segments to this RTTM file, so a "
                         "live session can be exported with "
                         "scripts/07_export_transcript.py afterwards")
    args = ap.parse_args()

    rttm_fh = None
    if args.rttm:
        import os
        os.makedirs(os.path.dirname(args.rttm) or ".", exist_ok=True)
        rttm_fh = open(args.rttm, "w")

    print(f"Loading model ({args.preset} preset)...", file=sys.stderr)
    model = SortformerEncLabelModel.from_pretrained(MODEL_NAME).eval().to("cuda")
    for key, value in PRESETS[args.preset].items():
        setattr(model.sortformer_modules, key, value)
    model.sortformer_modules._check_streaming_parameters()

    window_samples = int(args.window * SAMPLE_RATE)
    hop_samples = int(args.hop * SAMPLE_RATE)
    warmup_samples = int(args.warmup * SAMPLE_RATE)
    read_size = hop_samples * 2          # int16 = 2 bytes per sample

    buffer = np.zeros(0, dtype=np.float32)
    total_consumed = 0                   # samples that have scrolled off the front
    reported_until = 0.0                 # absolute time already printed
    anchor = LabelAnchor()

    print("Listening. Speak now. Ctrl-C to stop.\n", file=sys.stderr)
    stream = sys.stdin.buffer

    try:
        while True:
            raw = stream.read(read_size)
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.concatenate([buffer, chunk])

            if len(buffer) + total_consumed < warmup_samples:
                continue

            if len(buffer) > window_samples:
                dropped = len(buffer) - window_samples
                buffer = buffer[dropped:]
                total_consumed += dropped

            window_start = total_consumed / SAMPLE_RATE
            window_end = (total_consumed + len(buffer)) / SAMPLE_RATE

            started = time.perf_counter()
            result = model.diarize(
                audio=[buffer.copy()], batch_size=1, sample_rate=SAMPLE_RATE
            )
            compute_s = time.perf_counter() - started

            segments = [
                (s + window_start, e + window_start, spk)
                for s, e, spk in parse_segments(result[0])
            ]
            segments = anchor.assign(segments, window_start, reported_until or window_start)

            # Print only what is newly settled: anything that ends before the
            # most recent audio, and that we have not already printed.
            for start, end, name in sorted(segments):
                if end <= reported_until or start >= window_end:
                    continue
                shown_start = max(start, reported_until)
                if end - shown_start < 0.15:
                    continue
                print(f"[{shown_start:7.2f}s - {end:7.2f}s]  Speaker {name}", flush=True)
                if rttm_fh:
                    # Same shape 03_diarize.py writes, so 07_export_transcript.py
                    # can consume the output of either script.
                    rttm_fh.write(
                        f"SPEAKER live 1 {shown_start:.3f} {end - shown_start:.3f} "
                        f"<NA> <NA> {name} <NA> <NA>\n"
                    )
                    rttm_fh.flush()

            reported_until = window_end
            if compute_s > args.hop:
                print(
                    f"  (falling behind: {compute_s:.2f}s to process, "
                    f"hop is {args.hop:.2f}s -- try --window 20)",
                    file=sys.stderr, flush=True,
                )

    except KeyboardInterrupt:
        pass
    finally:
        if rttm_fh:
            rttm_fh.close()
            print(f"\nSaved RTTM: {args.rttm}", file=sys.stderr)

    print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    main()