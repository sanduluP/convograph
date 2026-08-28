#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run_build_serpentine.sh — regenerate the LANDSCAPE (v2) journey diagrams.
#
#  The v1 diagrams are a single tall column (0.67:1 and 0.47:1). That is the worst
#  shape for a README, a slide, or a LaTeX figure. This re-lays the SAME content —
#  same boxes, same wording, same grey notes — into a serpentine that snakes
#  right, left, right, giving ~1.9:1.
#
#  ⚠️  IT REGENERATES v2 FROM v1. Per the standing rule that a .excalidraw file
#  belongs to Faris once it exists, this never touches the v1 files, and you
#  should not re-run it after hand-editing a v2 — your edits would be lost.
#  Hand-edit v1, or stop running this and edit v2 directly.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs excalidraw/archive

PY="./.venv/bin/python"; [[ -x "$PY" ]] || PY=python3

{
  echo "📐 rebuilding serpentine diagrams — $(date '+%Y-%m-%d %H:%M:%S')"
  # Never overwrite a diagram in place without keeping the previous one.
  for f in excalidraw/*-v2.excalidraw; do
    [[ -e "$f" ]] || continue
    cp "$f" "excalidraw/archive/$(basename "${f%.excalidraw}")_$(date +%Y-%m-%d-%H%M%S).excalidraw"
    echo "🗄️  archived $(basename "$f")"
  done
  echo ""
  PYTHONUNBUFFERED=1 PYTHONPATH=. "$PY" -u analysis/build_serpentine_diagrams.py
  echo ""
  echo "✅ done — open the v2 files and export PNGs from Excalidraw"
} 2>&1 | tee logs/build_serpentine.log
