#!/bin/bash
# tools/persistent_cache/promote_tarballs.sh — refresh CACHE_READ_DIR/<scope>.tar.zst from
# CACHE_WRITE_DIR/<scope>/. Called from sbatch wrapper as a separate
# `srun -p cpu`, never from a GPU job. Heavy tar work belongs on a CPU node.
#
# Args: scope names to promote. Defaults to all SCOPES.

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/lib.sh"

mkdir -p "${CACHE_READ_DIR}"

# shellcheck disable=SC2206
todo=("${@:-${SCOPES[@]}}")

for scope in "${todo[@]}"; do
  src="$(scope_lustre_write_dir "$scope")"
  tarball="$(scope_lustre_read_tar "$scope")"
  [[ -d "$src" ]] || continue
  [[ -n "$(ls -A "$src" 2>/dev/null)" ]] || continue

  needs=0
  if [[ ! -f "$tarball" ]]; then
    needs=1
  elif find "$src" -type f -newer "$tarball" -print -quit 2>/dev/null | grep -q .; then
    needs=1
  fi

  if (( needs )); then
    tar_safe "$src" "$tarball" "$scope" || true
  else
    echo "[CACHE PROMOTE] ${scope}: tarball up to date"
  fi
done
