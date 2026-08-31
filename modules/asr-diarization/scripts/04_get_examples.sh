#!/usr/bin/env bash
#
# Step 4: Get example audio -- one real 4-person meeting, four separate mics.
#
# AMI recorded each participant on their own close-talking headset microphone.
# By downloading those four channels separately, we can build a 2-speaker,
# 3-speaker and 4-speaker version of the SAME conversation. Same room, same
# voices, same recording conditions -- the only thing that changes is how many
# people are in the mix. That tells you far more than comparing across
# different corpora, where acoustics and speaking style vary too.
#
# AMI is Creative Commons licensed. Download is roughly 200 MB.
#
# Usage:  ./scripts/04_get_examples.sh          # first 5 minutes (default)
#         ./scripts/04_get_examples.sh 10       # first 10 minutes

set -euo pipefail

MINUTES="${1:-5}"
MEETING="ES2004a"
BASE="http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/${MEETING}/audio"

mkdir -p examples/raw

echo ">> Downloading 4 headset channels from meeting ${MEETING}..."
for i in 0 1 2 3; do
    OUT="examples/raw/${MEETING}.Headset-${i}.wav"
    if [ -f "$OUT" ]; then
        echo "   channel $i already present"
    else
        wget -c -q --show-progress -O "$OUT" "${BASE}/${MEETING}.Headset-${i}.wav"
    fi
done

echo ">> Trimming to first ${MINUTES} minutes and converting to 16 kHz mono..."
SECONDS_LIMIT=$((MINUTES * 60))
for i in 0 1 2 3; do
    ffmpeg -hide_banner -loglevel error -y \
        -i "examples/raw/${MEETING}.Headset-${i}.wav" \
        -t "$SECONDS_LIMIT" -ac 1 -ar 16000 \
        "examples/raw/ch${i}.wav"
done

echo ""
echo "Done. Now build the mixes:"
echo "  make examples"