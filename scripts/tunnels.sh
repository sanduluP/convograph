#!/usr/bin/env bash
# =============================================================================
#  tunnels.sh — open (and keep honest) the two SSH tunnels Convograph needs.
#
#  Nobody should have to remember a -L line. Run this, or run nothing at all:
#  scripts/run_content_map.sh calls `tunnels.sh start` itself, so the normal
#  path is that you never think about tunnels.
#
#      bash scripts/tunnels.sh start     # idempotent — safe to run any time
#      bash scripts/tunnels.sh status
#      bash scripts/tunnels.sh stop
#
#  WHAT IS TUNNELLED, AND WHY ONLY THESE TWO
#      :8500   FLUX.1-schnell, warm on unicorn's H100
#      :11435  the bge-m3 embedder
#  Everything else needs no tunnel: all LLM work is SAIA (an HTTPS API) and
#  Neo4j is AuraDB. Both unicorn servers bind to 127.0.0.1 because that box is
#  shared and an unauthenticated GPU endpoint has no business on its network
#  interface — so the tunnel is the front door, one per laptop.
#
#  WHY LIVENESS IS AN HTTP PROBE, NOT `ss`
#      A dead tunnel keeps its local port BOUND. `ss -ltn` shows it listening
#      and `curl` fails — so a port check reports healthy while every request
#      hangs. This has cost real debugging time, so `start` probes the actual
#      endpoints and tears down a stale forward rather than trusting the port.
#
#  FIRST TIME ON A NEW MACHINE
#      Needs a `unicorn` host in ~/.ssh/config. Add (with YOUR username):
#          Host unicorn
#            HostName serv-7101.kl.dfki.de
#            User <your-dfki-username>
#            IdentityFile ~/.ssh/id_ed25519
#      DFKI VPN must be up.
# =============================================================================
set -uo pipefail

SSH_HOST="${CONVOGRAPH_SSH_HOST:-unicorn}"
FLUX_PORT="${FLUX_PORT:-8500}"
EMBED_PORT="${EMBED_PORT:-11435}"
PID_FILE="${TMPDIR:-/tmp}/convograph-tunnels-${SSH_HOST}.pid"

# ── liveness: ask the SERVICE, never the socket ──────────────────────────────
flux_alive()  { curl -sf --max-time 5 "http://localhost:${FLUX_PORT}/health"   >/dev/null 2>&1; }
embed_alive() { curl -sf --max-time 5 "http://localhost:${EMBED_PORT}/api/tags" >/dev/null 2>&1; }
both_alive()  { flux_alive && embed_alive; }

port_bound() { # $1 = port. Bound but not answering == the stale case.
  ss -ltn 2>/dev/null | grep -q ":$1 "
}

kill_ours() {
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    kill "$(cat "${PID_FILE}")" 2>/dev/null
  fi
  rm -f "${PID_FILE}"
  # Also catch a forward started by hand or by an earlier shell. Matched on the
  # forward SPEC, not on a bare hostname, so this never kills an unrelated
  # interactive ssh session to the same host.
  pkill -f "ssh.*-L ${FLUX_PORT}:localhost:${FLUX_PORT}.*${SSH_HOST}" 2>/dev/null
  pkill -f "ssh.*-L ${EMBED_PORT}:localhost:${EMBED_PORT}.*${SSH_HOST}" 2>/dev/null
  sleep 1
}

check_ssh_config() {
  if ! ssh -G "${SSH_HOST}" 2>/dev/null | grep -q "^hostname serv-7101"; then
    echo "❌ No usable '${SSH_HOST}' host in ~/.ssh/config."
    echo "   Add this block (with YOUR DFKI username) and make sure the VPN is up:"
    echo
    echo "       Host unicorn"
    echo "         HostName serv-7101.kl.dfki.de"
    echo "         User <your-dfki-username>"
    echo "         IdentityFile ~/.ssh/id_ed25519"
    return 1
  fi
  return 0
}

cmd_status() {
  echo "🔎 Convograph tunnels via '${SSH_HOST}'"
  local rc=0
  if flux_alive; then
    local dev; dev=$(curl -s --max-time 5 "http://localhost:${FLUX_PORT}/health" \
                     | sed -n 's/.*"device":"\([^"]*\)".*/\1/p')
    echo "   ✅ FLUX      localhost:${FLUX_PORT}   (${dev:-warm})"
  elif port_bound "${FLUX_PORT}"; then
    echo "   💀 FLUX      localhost:${FLUX_PORT}   PORT BOUND BUT DEAD — stale tunnel"; rc=1
  else
    echo "   ❌ FLUX      localhost:${FLUX_PORT}   not forwarded"; rc=1
  fi
  if embed_alive; then
    echo "   ✅ embedder  localhost:${EMBED_PORT}  (bge-m3)"
  elif port_bound "${EMBED_PORT}"; then
    echo "   💀 embedder  localhost:${EMBED_PORT}  PORT BOUND BUT DEAD — stale tunnel"; rc=1
  else
    echo "   ❌ embedder  localhost:${EMBED_PORT}  not forwarded"; rc=1
  fi
  echo "   ℹ️  SAIA (all LLM work) and AuraDB need no tunnel."
  return ${rc}
}

cmd_start() {
  if both_alive; then
    echo "✅ tunnels already up and answering — nothing to do"
    return 0
  fi

  # Bound-but-dead, or half up. Either way the existing forward is not usable.
  if port_bound "${FLUX_PORT}" || port_bound "${EMBED_PORT}"; then
    echo "💀 a forward is bound but not answering — tearing it down first"
    kill_ours
  fi

  check_ssh_config || return 1

  echo "🔌 opening tunnels to '${SSH_HOST}'..."
  # -f background, -N no remote command, ServerAliveInterval so a dropped link
  # is noticed rather than hanging forever. ExitOnForwardFailure makes ssh FAIL
  # when a port is already taken, instead of connecting with no forward at all
  # and leaving us to discover it later as a mystery timeout.
  ssh -f -N \
      -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes \
      -o ConnectTimeout=10 \
      -L "${FLUX_PORT}:localhost:${FLUX_PORT}" \
      -L "${EMBED_PORT}:localhost:${EMBED_PORT}" \
      "${SSH_HOST}"
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "❌ ssh failed (exit ${rc}). Usual causes, in order:"
    echo "   1. DFKI VPN is down"
    echo "   2. ports ${FLUX_PORT}/${EMBED_PORT} already taken by another process"
    echo "   3. no key on ${SSH_HOST} for your user"
    return 1
  fi

  # Record the PID we just made, matched on the forward spec so we only ever
  # stop our own tunnel later.
  pgrep -f "ssh.*-L ${FLUX_PORT}:localhost:${FLUX_PORT}.*${SSH_HOST}" \
    | head -1 > "${PID_FILE}" 2>/dev/null

  # ssh -f returns as soon as it is backgrounded, which is BEFORE the forward
  # is usable. Wait for the service to actually answer.
  for _ in $(seq 1 15); do
    both_alive && break
    sleep 1
  done

  if both_alive; then
    echo "✅ tunnels up"
    cmd_status
    return 0
  fi

  echo "⚠️  ssh connected but a service is not answering — the tunnel is fine,"
  echo "    the server on ${SSH_HOST} is probably not running:"
  cmd_status
  echo
  echo "    start FLUX:      ssh ${SSH_HOST} 'cd ~/projects/convograph/modules/graphic-generation && bash scripts/serve_flux.sh --daemon'"
  echo "    start embedder:  ssh ${SSH_HOST} 'cd ~/projects/convograph/modules/kg-agent-memory && bash scripts/serve_llm_on_unicorn.sh start'"
  return 1
}

cmd_stop() {
  kill_ours
  echo "🧹 tunnels closed"
}

case "${1:-status}" in
  start)  cmd_start  ;;
  stop)   cmd_stop   ;;
  status) cmd_status ;;
  *) echo "usage: bash scripts/tunnels.sh {start|stop|status}"; exit 2 ;;
esac
