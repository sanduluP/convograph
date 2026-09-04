#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  export_concept_edges.sh — pull concept→concept edges out of a cluster store.
#
#  WHY: a CONTENT MAP (CHI 2021's term) is concepts connected by labelled lines.
#  Until the speaker-exclusion fix, this graph had 433 such edges out of 111,258
#  (0.4 %) — not enough to draw one. The speaker-free store has ~60 %. This dumps
#  those edges as JSON so the board renderer can draw a map instead of a list,
#  without needing the whole store copied down.
#
#  Usage (on the Pegasus login node):
#     bash scripts/export_concept_edges.sh                    # smoke store
#     STORE=/fscratch/abuali/neo4j/shards_speaker_free/s1 bash scripts/export_concept_edges.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
FS_ROOT="${FS_ROOT:-/fscratch/abuali}"
STORE="${STORE:-${FS_ROOT}/neo4j/smoke/speaker_excluded}"
NEO4J_HOME="${FS_ROOT}/neo4j/neo4j-community-5.26.0"
export JAVA_HOME="${FS_ROOT}/conda_envs/java21"; export PATH="${JAVA_HOME}/bin:${PATH}"
PASS="${NEO4J_PASSWORD:-graphiti123}"
OUT="${OUT:-/tmp/concept_edges.json}"

[[ -d "${STORE}/data" ]] || { echo "❌ no store at ${STORE}/data"; exit 1; }
CONF="${STORE}/conf_export"; mkdir -p "${CONF}" "${STORE}/logs_export" "${STORE}/run_export"
BOLT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
HTTP=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
cat > "${CONF}/neo4j.conf" <<CONF_EOF
server.directories.data=${STORE}/data
server.directories.logs=${STORE}/logs_export
server.directories.run=${STORE}/run_export
server.bolt.listen_address=:${BOLT}
server.http.listen_address=:${HTTP}
server.memory.heap.max_size=2G
CONF_EOF
export NEO4J_CONF="${CONF}"
"${NEO4J_HOME}/bin/neo4j" start >/dev/null 2>&1 || true
for _ in $(seq 1 50); do
  "${NEO4J_HOME}/bin/cypher-shell" -a "bolt://localhost:${BOLT}" -u neo4j -p "${PASS}" "RETURN 1;" >/dev/null 2>&1 && break
  sleep 3
done

# Concept→concept only: both endpoints must be real named things, not people,
# not bare dates, not sentence fragments. This is the material a content map is
# drawn from, and nothing else on the board is worth an arrow.
"${NEO4J_HOME}/bin/cypher-shell" -a "bolt://localhost:${BOLT}" -u neo4j -p "${PASS}" --format plain <<'CYPHER' > "${OUT}.csv"
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.fact IS NOT NULL
  AND NOT a.name =~ '(?i)^(user_\d+|ops|compliance|risk|qa|finance|it|security|legal|product|ux|support|comms)\s?(lead|owner|analyst|team)?$'
  AND NOT b.name =~ '(?i)^(user_\d+|ops|compliance|risk|qa|finance|it|security|legal|product|ux|support|comms)\s?(lead|owner|analyst|team)?$'
  AND NOT a.name =~ '(?i).*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}-\d{2}).*'
  AND NOT b.name =~ '(?i).*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}-\d{2}).*'
  AND size(a.name) > 4 AND size(b.name) > 4
RETURN a.name AS source, r.name AS relation, b.name AS target,
       r.fact AS fact,
       (r.invalid_at IS NOT NULL) AS superseded
LIMIT 4000;
CYPHER
"${NEO4J_HOME}/bin/neo4j" stop >/dev/null 2>&1 || true
echo "✅ $(($(wc -l < "${OUT}.csv") - 1)) concept→concept edges → ${OUT}.csv"
