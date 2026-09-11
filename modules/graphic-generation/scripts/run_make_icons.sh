#!/usr/bin/env bash
# =============================================================================
#  run_make_icons.sh — pre-warm the concept-icon vocabulary.
#
#  The board margins carry ~35 items. Each gets an icon from a SHARED
#  vocabulary rather than its own generated image: the same concept always
#  looks the same, and each icon is generated once and cached to icons/.
#
#  Run this once. Boards afterwards pay nothing for icons.
#
#  USAGE
#      bash scripts/run_make_icons.sh            # fill in what is missing
#      bash scripts/run_make_icons.sh --all      # regenerate everything
#      bash scripts/run_make_icons.sh --only decision,handoff   # redraw a few
#      bash scripts/run_make_icons.sh --list     # show the vocabulary
# =============================================================================
set -euo pipefail
MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MODULE_ROOT}/logs"; mkdir -p "${LOG_DIR}"
PY="${MODULE_ROOT}/../../ui/.venv/bin/python"
{
  echo "═══════════════════════════════════════════════════════════════════"
  echo "🎨 CONCEPT ICONS — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "═══════════════════════════════════════════════════════════════════"
  PYTHONUNBUFFERED=1 "${PY}" -u "${MODULE_ROOT}/icons.py" "$@"
  echo "🧹 done"
} 2>&1 | tee -a "${LOG_DIR}/make_icons.log"
echo; echo "📝 log: ${LOG_DIR}/make_icons.log"
