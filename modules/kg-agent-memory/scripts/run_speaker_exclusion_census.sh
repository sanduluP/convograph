#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run_speaker_exclusion_census.sh — read the A/B result off the two smoke stores.
#
#  Reports, per variant:
#    person-rooted %      94.1 % on the full graph today  → want DOWN
#    concept -> concept %  0.4 % on the full graph today  → want UP
#    facts per episode                                    → want ROUGHLY HELD
#
#  The third number decides whether a shape win is real. If speakers vanish and
#  facts-per-episode collapses with them, the extractor simply gave up rather than
#  finding two concepts — a star traded for a sparse graph, which helps nobody.
#
#  Usage (on the Pegasus login node, after both smoke jobs finish):
#     bash scripts/run_speaker_exclusion_census.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="$(pwd)"
FS_ROOT="${FS_ROOT:-/fscratch/abuali}"
NEO4J_HOME="${FS_ROOT}/neo4j/neo4j-community-5.26.0"
export JAVA_HOME="${FS_ROOT}/conda_envs/java21"
export PATH="${JAVA_HOME}/bin:${PATH}"
PASS="${NEO4J_PASSWORD:-graphiti123}"
mkdir -p logs

census() {
  local variant="$1" store="${FS_ROOT}/neo4j/smoke/$1"
  [[ -d "${store}/data" ]] || { echo "  ⚠️  ${variant}: no store yet"; return; }

  local conf="${store}/conf_census"
  mkdir -p "${conf}" "${store}/logs_census" "${store}/run_census"
  local bolt http
  bolt=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
  http=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
  cat > "${conf}/neo4j.conf" <<CONF
server.directories.data=${store}/data
server.directories.logs=${store}/logs_census
server.directories.run=${store}/run_census
server.bolt.listen_address=:${bolt}
server.http.listen_address=:${http}
server.memory.heap.max_size=2G
server.memory.pagecache.size=512m
CONF
  NEO4J_CONF="${conf}" "${NEO4J_HOME}/bin/neo4j" start >/dev/null 2>&1 || true
  for _ in $(seq 1 50); do
    "${NEO4J_HOME}/bin/cypher-shell" -a "bolt://localhost:${bolt}" -u neo4j -p "${PASS}" \
      "RETURN 1;" >/dev/null 2>&1 && break
    sleep 3
  done

  echo "═══ ${variant} ═══"
  "${NEO4J_HOME}/bin/cypher-shell" -a "bolt://localhost:${bolt}" -u neo4j -p "${PASS}" --format plain <<'CYPHER'
MATCH (e:Episodic) WITH count(e) AS episodes
MATCH (n:Entity)   WITH episodes, count(n) AS entities
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
WITH episodes, entities,
     (s.name =~ '^User_\d+$' OR s.name =~ '(?i)^(ops|compliance|risk|qa|finance|it|security|legal|product|ux)\s?(lead|owner|analyst|team)?$') AS sp,
     (t.name =~ '^User_\d+$' OR t.name =~ '(?i)^(ops|compliance|risk|qa|finance|it|security|legal|product|ux)\s?(lead|owner|analyst|team)?$') AS tp
WITH episodes, entities, count(*) AS facts,
     sum(CASE WHEN sp THEN 1 ELSE 0 END) AS person_rooted,
     sum(CASE WHEN NOT sp AND NOT tp THEN 1 ELSE 0 END) AS non_person_pairs
RETURN episodes, entities, facts,
       round(1.0*facts/episodes, 1)            AS facts_per_episode,
       round(100.0*person_rooted/facts, 1)     AS pct_person_rooted,
       round(100.0*non_person_pairs/facts, 1)  AS pct_thing_to_thing;
CYPHER
  echo "--- do any speaker nodes survive? ---"
  "${NEO4J_HOME}/bin/cypher-shell" -a "bolt://localhost:${bolt}" -u neo4j -p "${PASS}" --format plain \
    "MATCH (n:Entity) WHERE n.name =~ '^User_\\\\d+$' RETURN count(n) AS user_nodes;"
  echo "--- biggest hubs ---"
  "${NEO4J_HOME}/bin/cypher-shell" -a "bolt://localhost:${bolt}" -u neo4j -p "${PASS}" --format plain \
    "MATCH (n:Entity)-[r:RELATES_TO]-() RETURN n.name AS name, count(r) AS degree ORDER BY degree DESC LIMIT 6;"
  echo ""
  NEO4J_CONF="${conf}" "${NEO4J_HOME}/bin/neo4j" stop >/dev/null 2>&1 || true
}

{
  echo "🔬 speaker-exclusion A/B — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   reference (full graph, today): 94.1 % person-rooted · 0.4 % concept→concept"
  echo ""
  census baseline
  census speaker_excluded
  echo "✅ done"
} 2>&1 | tee logs/speaker_exclusion_census.log
