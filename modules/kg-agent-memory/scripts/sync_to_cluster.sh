#!/bin/bash
# sync_to_cluster.sh — Push local GroupMemBench changes to the Pegasus cluster
# WITHOUT committing. The local clone is the source of truth; the cluster copy is
# an execution mirror (we run vLLM-backed jobs there).
#
# Usage:
#   bash scripts/sync_to_cluster.sh            # sync
#   bash scripts/sync_to_cluster.sh --dry-run  # preview only
#
# NOT synced (cluster keeps its own):
#   - everything in .gitignore  → incl. .env (local serv-3306 config), .venv/,
#     logs/, results/  (the cluster has its own .env.cluster + venv + outputs)
#   - .git/      → no history needed on the mirror
#   - data/      → 152 MB of conversations; push ONCE, then skip re-transfer
# rsync's --exclude also protects from --delete, so a cluster-side .env.cluster
# is preserved across syncs.
set -euo pipefail

LOCAL_ROOT="/home/faris/code/DSA_HiWi/GroupMemBench-upstream"
CLUSTER_USER="abuali"
CLUSTER_HOST="login1.pegasus.kl.dfki.de"
CLUSTER_ROOT="/home/abuali/projects/GroupMemBench"

# By default we DO ship data/ the first time; pass --no-data on later syncs to
# skip the 152 MB re-scan once it's already on the cluster.
DATA_EXCLUDE=""
DRY_RUN_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN_FLAG="--dry-run"; echo "[dry-run] nothing will transfer";;
    --no-data) DATA_EXCLUDE="--exclude=data/"; echo "[no-data] skipping data/ (assumed already on cluster)";;
  esac
done

rsync \
    --archive --verbose --compress --human-readable --progress --delete \
    --exclude-from="$LOCAL_ROOT/.gitignore" \
    --exclude=".git/" \
    --exclude=".venv/" \
    $DATA_EXCLUDE \
    $DRY_RUN_FLAG \
    "$LOCAL_ROOT/" \
    "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_ROOT/"

echo ""
echo "✅ Done. Cluster path: $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_ROOT"
