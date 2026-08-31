#!/usr/bin/env bash
# run_on_unicorn.sh — one command from the laptop: sync, run generate_image.py
# on unicorn (it has the GPU this laptop doesn't), copy the result back.
#
# Usage:
#   bash scripts/run_on_unicorn.sh                        # dummy caption
#   bash scripts/run_on_unicorn.sh --caption "Finance Ops locks go/no-go"
#
# Assumes scripts/setup_remote_env.sh has already been run once on unicorn.
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_USER="${CLUSTER_USER:-sandulu}"
CLUSTER_HOST="${CLUSTER_HOST:-serv-7101.kl.dfki.de}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/scratch/mpatil/sandulu/convograph-graphic-generation}"

echo "📤 syncing to unicorn..."
CLUSTER_USER="$CLUSTER_USER" CLUSTER_HOST="$CLUSTER_HOST" CLUSTER_ROOT="$CLUSTER_ROOT" \
  bash "${LOCAL_ROOT}/scripts/sync_to_cluster.sh"

echo "🚀 running on unicorn..."
# $* would re-join args with spaces and let the remote shell word-split them
# again, silently breaking any --caption with spaces in it (observed: a
# multi-word caption arrived at generate_image.py as many separate argv
# entries). printf %q re-quotes each arg so it survives the second shell hop
# as one token.
REMOTE_ARGS=""
for arg in "$@"; do
  REMOTE_ARGS+=" $(printf '%q' "$arg")"
done
# shellcheck disable=SC2029
ssh "${CLUSTER_USER}@${CLUSTER_HOST}" \
  "cd '${CLUSTER_ROOT}' && bash scripts/run_generate_image.sh${REMOTE_ARGS}"

echo "📥 copying results back..."
mkdir -p "${LOCAL_ROOT}/output"
rsync --archive --compress --human-readable \
  "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_ROOT}/output/" \
  "${LOCAL_ROOT}/output/"

echo "✅ done — see ${LOCAL_ROOT}/output/"
