#!/usr/bin/env bash
# sync_to_cluster.sh — push local graphic-generation changes to the "unicorn"
# GPU host WITHOUT committing. The local clone is the source of truth; the
# remote copy is an execution mirror (this is where generate_image.py actually
# runs — it needs a CUDA GPU this laptop doesn't have).
#
# Mirrors modules/kg-agent-memory/scripts/sync_to_cluster.sh's pattern; unicorn
# is a plain SSH host (serv-7101.kl.dfki.de), not behind SLURM like Pegasus, so
# there's no srun_submit.sh equivalent here — sync, ssh in, run directly.
#
# Usage:
#   bash scripts/sync_to_cluster.sh            # sync
#   bash scripts/sync_to_cluster.sh --dry-run  # preview only
#
# Add this to ~/.ssh/config once, and CLUSTER_HOST below can just say "unicorn":
#   Host unicorn
#     HostName serv-7101.kl.dfki.de
#     User sandulu
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_USER="${CLUSTER_USER:-sandulu}"
CLUSTER_HOST="${CLUSTER_HOST:-serv-7101.kl.dfki.de}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/scratch/mpatil/sandulu/convograph-graphic-generation}"

DRY_RUN_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN_FLAG="--dry-run"; echo "[dry-run] nothing will transfer";;
  esac
done

# shellcheck disable=SC2029
ssh "$CLUSTER_USER@$CLUSTER_HOST" "mkdir -p '$CLUSTER_ROOT'"

rsync \
    --archive --verbose --compress --human-readable --progress --delete \
    --exclude-from="$LOCAL_ROOT/.gitignore" \
    --exclude=".git/" \
    --exclude=".venv/" \
    $DRY_RUN_FLAG \
    "$LOCAL_ROOT/" \
    "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_ROOT/"

echo ""
echo "✅ Done. Remote path: $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_ROOT"
