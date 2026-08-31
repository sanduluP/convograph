#!/usr/bin/env bash
#
# Step 2: Convert an audio file into the exact format Sortformer needs.
#
# Sortformer ONLY accepts 16,000 Hz, single-channel (mono) WAV.
# If you feed it a stereo file or a 44.1 kHz file, you get a confusing
# tensor-shape error, not a helpful message. So always run this first.
#
# Usage:
#   ./scripts/02_prep_audio.sh audio/my_recording.m4a
#
# Produces:
#   audio/my_recording_16k.wav

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input-audio-file>"
    exit 1
fi

IN="$1"
if [ ! -f "$IN" ]; then
    echo "ERROR: file not found: $IN"
    exit 1
fi

DIR="$(dirname "$IN")"
BASE="$(basename "${IN%.*}")"
OUT="${DIR}/${BASE}_16k.wav"

# -ac 1  => mix down to 1 channel (mono)
# -ar 16000 => resample to 16 kHz
# -y     => overwrite without asking
ffmpeg -hide_banner -loglevel error -y -i "$IN" -ac 1 -ar 16000 "$OUT"

echo "Wrote: $OUT"
ffprobe -hide_banner -loglevel error -show_entries \
    stream=sample_rate,channels,duration -of default=noprint_wrappers=1 "$OUT"