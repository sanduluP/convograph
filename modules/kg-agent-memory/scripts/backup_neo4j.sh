#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# backup_neo4j.sh — snapshot the local Graphiti Neo4j so months of experiment
# KGs can never be lost to a stray `docker rm` / `docker volume prune`.
#
# WHY THIS EXISTS: the `neo4j-graphiti` container stores /data on an ANONYMOUS
# docker volume (no human-readable name). Stop/start and reboots are safe, but
# removing the container orphans the volume, and `docker volume prune` deletes
# it outright — taking every group_id with it (the June A/B experiment KGs, the
# ATOM side-by-side, and now the GroupMemBench slices).
#
# It makes TWO backups, because they fail differently:
#   1) LOGICAL export (CSV, no downtime)  — human-readable, greppable, survives
#      any Neo4j version change. Good for eyeballing + for the thesis appendix.
#   2) RAW volume tarball (brief downtime) — a byte-exact restore of the whole
#      database, including indexes. This is the real disaster-recovery artifact.
#
# The raw step STOPS the container (Neo4j Community has no online backup), so do
# NOT run it while an ingest is in flight — the script refuses if it sees one.
#
# Usage:
#   bash scripts/backup_neo4j.sh              # both backups
#   LOGICAL_ONLY=1 bash scripts/backup_neo4j.sh   # CSV only, zero downtime
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONTAINER=${CONTAINER:-neo4j-graphiti}
NEO4J_USER=${NEO4J_USER:-neo4j}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-graphiti123}
# Backups live OUTSIDE the repo — this is data, not code, and must never be
# committed (it would also be huge).
BACKUP_DIR=${BACKUP_DIR:-"${HOME}/neo4j-backups"}
STAMP="$(date +%Y-%m-%d-%H%M%S)"

mkdir -p logs "${BACKUP_DIR}"
LOG="${ROOT_DIR}/logs/backup_neo4j.log"

{
echo "🗄️  [backup] container=${CONTAINER}  dest=${BACKUP_DIR}  stamp=${STAMP}"

# --- preflight ---------------------------------------------------------------
docker ps --filter "name=${CONTAINER}" --format '{{.Names}}' | grep -q "${CONTAINER}" \
  || { echo "❌ ${CONTAINER} is not running — start it first: docker start ${CONTAINER}"; exit 1; }
echo "✅ [backup] container is up"

# Refuse the disruptive raw step if an ingest is mid-flight (it would kill it).
INGEST_RUNNING=0
if pgrep -f "eval_benchmark.py" >/dev/null 2>&1; then
  INGEST_RUNNING=1
  echo "⚠️  [backup] an eval_benchmark.py run is IN FLIGHT"
fi

# --- 1) LOGICAL export — read-only, safe at any time -------------------------
echo "📝 [backup] 1/2 logical CSV export (no downtime) …"
LOGICAL_DIR="${BACKUP_DIR}/logical_${STAMP}"
mkdir -p "${LOGICAL_DIR}"

# Helper: run a Cypher query and write plain CSV-ish output to a file.
run_cypher () {  # $1=query  $2=outfile
  docker exec -i "${CONTAINER}" cypher-shell \
      -u "${NEO4J_USER}" -p "${NEO4J_PASSWORD}" --format plain "$1" > "$2"
}

# a) Inventory: what group_ids exist and how big each is. This is the index you
#    actually want months later when you've forgotten the group names.
run_cypher "MATCH (n) WHERE n.group_id IS NOT NULL
            RETURN n.group_id AS group_id, labels(n)[0] AS label, count(*) AS n
            ORDER BY group_id, label" \
           "${LOGICAL_DIR}/inventory_nodes.csv"

# b) Every fact edge WITH its bi-temporal fields — the scientifically important
#    part (valid_at/invalid_at is our contribution; losing it loses the result).
run_cypher "MATCH (a)-[r:RELATES_TO]->(b)
            RETURN r.group_id AS group_id, a.name AS subject, r.fact AS fact,
                   b.name AS object, r.valid_at AS valid_at,
                   r.invalid_at AS invalid_at, r.created_at AS created_at
            ORDER BY group_id, valid_at" \
           "${LOGICAL_DIR}/fact_edges.csv"

# c) Per-group summary of how many facts got SUPERSEDED — the headline metric.
run_cypher "MATCH ()-[r:RELATES_TO]->()
            RETURN r.group_id AS group_id, count(r) AS facts,
                   count(r.invalid_at) AS invalidated
            ORDER BY group_id" \
           "${LOGICAL_DIR}/invalidation_summary.csv"

echo "✅ [backup] logical export → ${LOGICAL_DIR}"
wc -l "${LOGICAL_DIR}"/*.csv | sed 's/^/   /'

# --- 2) RAW volume tarball — needs the DB stopped ----------------------------
if [[ "${LOGICAL_ONLY:-0}" == "1" ]]; then
  echo "⏭️  [backup] LOGICAL_ONLY=1 → skipping the raw volume snapshot"
  exit 0
fi
if [[ "${INGEST_RUNNING}" == "1" ]]; then
  echo "⏭️  [backup] SKIPPING raw snapshot — an ingest is running and stopping"
  echo "   Neo4j would kill it. Re-run this script (without LOGICAL_ONLY) after it finishes."
  exit 0
fi

# Resolve the anonymous volume backing /data so we can mount it read-only into a
# throwaway container and tar it up.
VOLUME="$(docker inspect "${CONTAINER}" \
          --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}')"
[ -n "${VOLUME}" ] || { echo "❌ could not resolve the /data volume"; exit 1; }
echo "💾 [backup] 2/2 raw snapshot of volume ${VOLUME:0:12}… (brief downtime)"

TARBALL="${BACKUP_DIR}/neo4j-data_${STAMP}.tar.gz"

echo "🛑 [backup] stopping ${CONTAINER} (so the store files are quiescent) …"
docker stop "${CONTAINER}" >/dev/null

# Tar from a throwaway alpine that mounts the SAME volume. Raw file copy = no
# dependency on the Neo4j version, unlike neo4j-admin dump.
docker run --rm \
  -v "${VOLUME}":/data:ro \
  -v "${BACKUP_DIR}":/backup \
  alpine:3 tar czf "/backup/$(basename "${TARBALL}")" -C /data . \
  || { echo "❌ tar failed — restarting container anyway"; docker start "${CONTAINER}" >/dev/null; exit 1; }

echo "▶️  [backup] restarting ${CONTAINER} …"
docker start "${CONTAINER}" >/dev/null

echo "✅ [backup] raw snapshot → ${TARBALL}"
ls -lh "${TARBALL}" | sed 's/^/   /'
echo
echo "♻️  TO RESTORE into a fresh container:"
echo "   docker run -d --name neo4j-restored -p 7475:7474 -p 7688:7687 \\"
echo "     -e NEO4J_AUTH=${NEO4J_USER}/${NEO4J_PASSWORD} -v neo4j-restored-data:/data neo4j:5.26"
echo "   docker stop neo4j-restored"
echo "   docker run --rm -v neo4j-restored-data:/data -v ${BACKUP_DIR}:/backup \\"
echo "     alpine:3 sh -c 'rm -rf /data/* && tar xzf /backup/$(basename "${TARBALL}") -C /data'"
echo "   docker start neo4j-restored"
echo "🎉 [backup] done"
} 2>&1 | tee "${LOG}"
