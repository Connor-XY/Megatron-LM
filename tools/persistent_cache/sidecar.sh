#!/bin/bash
# tools/persistent_cache/sidecar.sh — long-running per-node writeback loop.
# Spawn from rank_entrypoint with `&`. One per node (gated by SLURM_LOCALID==0).

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/lib.sh"

is_local_rank_zero || exit 0
[[ "${MCORE_CACHE_SIDECAR_ENABLED:-1}" == "1" ]] || exit 0

WB="$(dirname "$0")/writeback.sh"

_final() {
  # Per-node exit jitter scaled by cluster size. Avoids N-node thundering herd
  # at job teardown.
  local nnodes=${SLURM_JOB_NUM_NODES:-1}
  local window=${MCORE_CACHE_SYNC_EXIT_JITTER_SECONDS:-}
  if [[ -z "$window" ]]; then
    window=$(( nnodes / 20 ))
    (( window < 15 )) && window=15
    (( window > 45 )) && window=45
  fi
  sleep "$(( RANDOM % (window > 0 ? window : 1) ))"
  bash "$WB" --final || true
}
trap '_final' EXIT

# Initial per-node jitter so 64 nodes don't first-rsync at the same wall clock.
sleep "$(( RANDOM % MCORE_CACHE_SYNC_JITTER_SECONDS ))"

while sleep "$MCORE_CACHE_SYNC_FREQUENCY"; do
  bash "$WB" || true
done
