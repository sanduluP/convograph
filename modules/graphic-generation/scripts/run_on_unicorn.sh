#!/usr/bin/env bash
# run_on_unicorn.sh — one command from the laptop: sync, run generate_image.py
# on unicorn (it has the GPU this laptop doesn't), copy the result back.
#
# Usage:
#   bash scripts/run_on_unicorn.sh                          # dummy caption
#   bash scripts/run_on_unicorn.sh --caption "Finance Ops locks go/no-go"
#   bash scripts/run_on_unicorn.sh --captions-file /tmp/captions.txt   # BATCH
#
# BATCH MODE is what the board pipeline uses: every caption for a board is sent
# in ONE call so the ~31 GB model is loaded once instead of once per fact. It
# prints, as its LAST line, the LOCAL directory the results landed in:
#
#     local-batch-dir: /path/to/output/batch_20260903_101500
#
# containing image_000.png... and manifest.jsonl. Callers should parse that line
# rather than guessing which PNGs are new.
#
# Assumes scripts/setup_remote_env.sh has already been run once on the host.
#
# CLUSTER_USER / CLUSTER_ROOT are overridable because this module is checked out
# under more than one account: it was written against sandulu's tree, and a
# second person's checkout lives somewhere else entirely.
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_USER="${CLUSTER_USER:-sandulu}"
CLUSTER_HOST="${CLUSTER_HOST:-serv-7101.kl.dfki.de}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/scratch/mpatil/sandulu/convograph-graphic-generation}"

# ── If a captions file was passed, stage it INSIDE the module tree ───────────
# The caller writes it to a temp path (e.g. /tmp/...), which the rsync below
# would never carry across. Copying it into the tree first means the existing
# sync ships it, and we then rewrite the argument to its path on the remote.
STAGED_CAPTIONS="${LOCAL_ROOT}/.captions_batch.txt"
ARGS=()
BATCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --captions-file)
      [[ -r "${2:-}" ]] || { echo "❌ captions file not readable: ${2:-<missing>}"; exit 1; }
      cp "$2" "$STAGED_CAPTIONS"
      ARGS+=(--captions-file "${CLUSTER_ROOT}/.captions_batch.txt")
      BATCH=1
      shift 2
      ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

echo "📤 syncing to ${CLUSTER_USER}@${CLUSTER_HOST}..."
CLUSTER_USER="$CLUSTER_USER" CLUSTER_HOST="$CLUSTER_HOST" CLUSTER_ROOT="$CLUSTER_ROOT" \
  bash "${LOCAL_ROOT}/scripts/sync_to_cluster.sh"

echo "🚀 running remotely..."
# $* would re-join args with spaces and let the remote shell word-split them
# again, silently breaking any --caption with spaces in it (observed: a
# multi-word caption arrived at generate_image.py as many separate argv
# entries). printf %q re-quotes each arg so it survives the second shell hop
# as one token.
REMOTE_ARGS=""
for arg in "${ARGS[@]+"${ARGS[@]}"}"; do
  REMOTE_ARGS+=" $(printf '%q' "$arg")"
done

# Capture the remote output while still streaming it, so we can read back the
# batch directory it created. PIPESTATUS keeps a remote failure fatal despite
# the pipe into tee.
REMOTE_LOG="$(mktemp)"
# shellcheck disable=SC2029
ssh "${CLUSTER_USER}@${CLUSTER_HOST}" \
  "cd '${CLUSTER_ROOT}' && bash scripts/run_generate_image.sh${REMOTE_ARGS}" \
  2>&1 | tee "$REMOTE_LOG"
[[ "${PIPESTATUS[0]}" -eq 0 ]] || { rm -f "$REMOTE_LOG" "$STAGED_CAPTIONS"; exit 1; }

echo "📥 copying results back..."
mkdir -p "${LOCAL_ROOT}/output"
rsync --archive --compress --human-readable \
  "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_ROOT}/output/" \
  "${LOCAL_ROOT}/output/"

if [[ "$BATCH" -eq 1 ]]; then
  # Translate the REMOTE batch dir to its LOCAL path. Only the basename is
  # meaningful after the rsync — the remote prefix does not exist here.
  REMOTE_BATCH="$(sed -n 's/^✅ batch-dir : //p' "$REMOTE_LOG" | tail -1)"
  [[ -n "$REMOTE_BATCH" ]] || { echo "❌ remote did not report a batch dir"; rm -f "$REMOTE_LOG" "$STAGED_CAPTIONS"; exit 1; }
  LOCAL_BATCH="${LOCAL_ROOT}/output/$(basename "$REMOTE_BATCH")"
  [[ -f "${LOCAL_BATCH}/manifest.jsonl" ]] || { echo "❌ no manifest at ${LOCAL_BATCH}"; rm -f "$REMOTE_LOG" "$STAGED_CAPTIONS"; exit 1; }
  echo "✅ done"
  # LAST line, machine-readable, parsed by ui/orchestrator.py.
  echo "local-batch-dir: ${LOCAL_BATCH}"
else
  echo "✅ done — see ${LOCAL_ROOT}/output/"
fi

rm -f "$REMOTE_LOG" "$STAGED_CAPTIONS"
