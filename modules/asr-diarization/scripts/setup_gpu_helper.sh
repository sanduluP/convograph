#!/usr/bin/env bash
# One-time (per machine, needs sudo): install a root helper that frees
# CUDA-allocatable memory on GB10 unified-memory hosts.
#
# WHY: when a large model server (e.g. an 80 GB llama.cpp) runs alongside
# module 1, CUDA allocations start failing with "out of memory" even though
# tens of GB are free — page cache and physical-memory fragmentation build up
# and the CUDA allocator won't reclaim/compact on its own (observed on DGX
# Spark; small allocations keep working, model loads fail). Dropping caches
# and compacting memory fixes it, but both need root.
#
# This installs /usr/local/sbin/convograph-free-gpu (does ONLY those two
# things) plus a sudoers rule so the UI / scripts can run it without a
# password. Review the 10 lines below before running; remove with:
#   sudo rm /usr/local/sbin/convograph-free-gpu /etc/sudoers.d/convograph-free-gpu
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run with sudo: sudo $0"; exit 1; }
invoking_user="${SUDO_USER:?run via sudo, not as a root login}"

cat > /usr/local/sbin/convograph-free-gpu <<'EOF'
#!/bin/sh
# Free reclaimable memory and defragment, so CUDA can allocate again.
# Safe: evicts caches (things reload from disk) and compacts free pages.
sync
echo 3 > /proc/sys/vm/drop_caches
echo 1 > /proc/sys/vm/compact_memory
EOF
chmod 755 /usr/local/sbin/convograph-free-gpu

echo "$invoking_user ALL=(root) NOPASSWD: /usr/local/sbin/convograph-free-gpu" \
    > /etc/sudoers.d/convograph-free-gpu
chmod 440 /etc/sudoers.d/convograph-free-gpu
visudo -c -f /etc/sudoers.d/convograph-free-gpu >/dev/null

echo "Installed. Test with: sudo -n /usr/local/sbin/convograph-free-gpu"
