"""
Step 5: Build 2-speaker, 3-speaker and 4-speaker versions of one meeting,
each with ground-truth labels.

How the ground truth is derived
-------------------------------
Each participant wore their own headset microphone. When person A talks, their
own mic is loud and everyone else's mic picks up only a faint, distant version.
So for every short slice of time we can ask: which channel is loudest? That
channel's owner was the one speaking.

This is called cross-channel speaker activity detection, and it is roughly how
close-talking meeting corpora get labelled in the first place. It is very good
but not perfect -- it can miss very quiet speech and occasionally clips the
first moments of a word. Treat the resulting numbers as a solid guide rather
than a published benchmark figure.

Usage:  python scripts/05_make_examples.py
"""

import os

import numpy as np
import soundfile as sf

RAW = "examples/raw"
OUT = "examples"

FRAME_MS = 20          # resolution of the speech/silence decision
# How far below the loudest channel a mic can be while still counting as
# "this person is talking". Too small and you miss overlapping speech, because
# only the louder of two simultaneous speakers gets credited. Too large and
# you start labelling bleed-through as real speech.
#
# On synthetic four-channel audio with -18 dB crosstalk, 6 dB recovered only
# 2.6% overlap against a true 4.2%, while 10-12 dB recovered nearly all of it
# with no loss of accuracy. Real headset bleed sits around -10 to -15 dB, so
# 10 dB is the compromise: most of the overlap, safely under the bleed floor.
DOMINANCE_DB = 10.0
MIN_SEGMENT_S = 0.20   # discard blips shorter than this
MAX_GAP_S = 0.30       # join segments separated by less than this


def load_channels():
    paths = [os.path.join(RAW, f"ch{i}.wav") for i in range(4)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Missing channel files:\n  " + "\n  ".join(missing) +
            "\n\nRun ./scripts/04_get_examples.sh first."
        )

    signals, rate = [], None
    for p in paths:
        data, sr = sf.read(p, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        rate = sr if rate is None else rate
        if sr != rate:
            raise SystemExit(f"Sample rate mismatch in {p}")
        signals.append(data)

    n = min(len(s) for s in signals)
    return np.stack([s[:n] for s in signals]), rate


def frame_energies(signals, rate):
    """Return per-channel loudness in dB, one value per 20 ms frame."""
    hop = int(rate * FRAME_MS / 1000)
    n_frames = signals.shape[1] // hop
    trimmed = signals[:, : n_frames * hop]
    frames = trimmed.reshape(signals.shape[0], n_frames, hop)
    rms = np.sqrt((frames ** 2).mean(axis=2) + 1e-12)
    return 20 * np.log10(rms), hop, n_frames


def active_speakers(energy_db, channels):
    """
    Decide, per frame, which of the given channels were active.

    A channel counts as active if it is within DOMINANCE_DB of the loudest
    channel at that moment AND is clearly above its own noise floor. The
    first condition handles who is speaking; the second stops us labelling
    silence, where the loudest channel is still just room tone.
    """
    subset = energy_db[channels]

    # Per-channel noise floor: the 30th percentile is safely inside silence
    # for meeting audio, where people are quiet more often than they talk.
    floor = np.percentile(subset, 30, axis=1, keepdims=True)
    above_floor = subset > (floor + 10.0)

    loudest = subset.max(axis=0, keepdims=True)
    dominant = subset > (loudest - DOMINANCE_DB)

    return above_floor & dominant


def frames_to_segments(mask, frame_s):
    """Turn a boolean per-frame array into a list of (start, end) times."""
    segments = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            segments.append((start * frame_s, i * frame_s))
            start = None
    if start is not None:
        segments.append((start * frame_s, len(mask) * frame_s))

    merged = []
    for s, e in segments:
        if merged and s - merged[-1][1] <= MAX_GAP_S:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return [(s, e) for s, e in merged if e - s >= MIN_SEGMENT_S]


def write_rttm(path, file_id, per_speaker):
    with open(path, "w") as fh:
        for spk, segments in sorted(per_speaker.items()):
            for start, end in segments:
                fh.write(
                    f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} "
                    f"<NA> <NA> {spk} <NA> <NA>\n"
                )


def main():
    signals, rate = load_channels()
    energy_db, hop, n_frames = frame_energies(signals, rate)
    frame_s = FRAME_MS / 1000.0

    os.makedirs(OUT, exist_ok=True)
    print(f"Loaded 4 channels, {n_frames * frame_s:.0f} seconds each\n")

    for n_speakers in (2, 3, 4):
        channels = list(range(n_speakers))
        file_id = f"{n_speakers}spk"

        # Mix: simply add the chosen channels, then scale down so the result
        # never clips. Dividing by the count keeps loudness roughly constant
        # across the three mixes, so the model is not handed a quieter signal
        # in the 4-speaker case.
        mix = signals[channels].sum(axis=0) / n_speakers
        peak = np.abs(mix).max()
        if peak > 0:
            mix = mix * (0.89 / peak)   # about -1 dBFS

        wav_path = os.path.join(OUT, f"{file_id}.wav")
        sf.write(wav_path, mix, rate, subtype="PCM_16")

        mask = active_speakers(energy_db, channels)
        per_speaker = {}
        total_speech = 0.0
        for idx, ch in enumerate(channels):
            segments = frames_to_segments(mask[idx], frame_s)
            per_speaker[f"spk{ch}"] = segments
            total_speech += sum(e - s for s, e in segments)

        rttm_path = os.path.join(OUT, f"{file_id}.rttm")
        write_rttm(rttm_path, file_id, per_speaker)

        # How much of the time is more than one person talking? This is the
        # single best predictor of whether a diarizer will struggle.
        overlap_frames = (mask.sum(axis=0) >= 2).sum()
        any_frames = (mask.sum(axis=0) >= 1).sum()
        overlap_pct = 100 * overlap_frames / any_frames if any_frames else 0

        print(
            f"{file_id}.wav   {n_frames * frame_s:6.0f}s   "
            f"speech {total_speech:6.0f}s   overlap {overlap_pct:5.1f}%"
        )

    print("\nBuilt 3 examples in examples/. Try them with:")
    print("  make example N=2")
    print("  make example N=3")
    print("  make example N=4")


if __name__ == "__main__":
    main()