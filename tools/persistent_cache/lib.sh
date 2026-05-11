#!/bin/bash
# tools/persistent_cache/lib.sh — shared functions and config. Source me; don't run me.
#
# Required env (caller must set):
#   PERSISTENT_CACHE  Lustre dir holding cache_read/ and cache_write/ subdirs.
#
# Derived (set by this lib if not already):
#   CACHE_READ_DIR, CACHE_WRITE_DIR, NODE_CACHE_BASE,
#   MCORE_CACHE_SYNC_DELAY_SECONDS, MCORE_CACHE_SYNC_FREQUENCY,
#   MCORE_CACHE_SYNC_JITTER_SECONDS, MCORE_CACHE_SYNC_EXIT_JITTER_SECONDS,
#   MCORE_JOB_START_EPOCH, SCOPES.

: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is unset (set in launcher before sourcing)}"
: "${CACHE_READ_DIR:=${PERSISTENT_CACHE}/cache_read}"
: "${CACHE_WRITE_DIR:=${PERSISTENT_CACHE}/cache_write}"
: "${NODE_CACHE_BASE:=/dev/shm/mcore_cache_${SLURM_JOB_ID:-manual}}"

: "${MCORE_CACHE_SYNC_DELAY_SECONDS:=600}"
: "${MCORE_CACHE_SYNC_FREQUENCY:=300}"
: "${MCORE_CACHE_SYNC_JITTER_SECONDS:=120}"
: "${MCORE_CACHE_SYNC_EXIT_JITTER_SECONDS:=}"
: "${MCORE_JOB_START_EPOCH:=$(date +%s)}"
export MCORE_JOB_START_EPOCH

# shellcheck disable=SC2206
SCOPES=(${MCORE_CACHE_SCOPES:-triton inductor cuda_ptx hybrid_ep cudnn_fe nccl_topo dataset_idx})

# Multi-threaded zstd by default (decompress 4-8x faster on multi-core nodes).
# Can be overridden by user; tar --zstd reads ZSTD_NBTHREADS for both compress/decompress.
: "${ZSTD_NBTHREADS:=0}"   # 0 = use all available cores
export ZSTD_NBTHREADS

# Probe rsync availability once per process. Sidecar / writeback uses cp -an fallback if absent.
if command -v rsync >/dev/null 2>&1; then
  MCORE_CACHE_RSYNC_AVAILABLE=1
else
  MCORE_CACHE_RSYNC_AVAILABLE=0
fi
export MCORE_CACHE_RSYNC_AVAILABLE

scope_local_dir()        { echo "${NODE_CACHE_BASE}/$1"; }
scope_lustre_write_dir() { echo "${CACHE_WRITE_DIR}/$1"; }
scope_lustre_read_tar()  { echo "${CACHE_READ_DIR}/$1.tar.zst"; }

is_local_rank_zero() { [[ "${SLURM_LOCALID:-0}" == "0" ]]; }

# Copy src/ -> dst/ with two backends: rsync (if available) or cp -anR fallback.
# rsync rc=23/24 (vanished files during live compile) treated as success.
# cp -an = archive + no-clobber, so it won't trample lustre files from other jobs.
rsync_safe() {
  local src="$1" dst="$2" name="$3"
  [[ -d "$src" ]] || return 1
  [[ -n "$(ls -A "$src" 2>/dev/null)" ]] || return 1
  mkdir -p "$dst"
  local _err _rc=0
  if command -v rsync >/dev/null 2>&1; then
    _err=$(rsync -a --exclude='tmp*' --exclude='.tmp_*' --exclude='.*' \
      "$src/" "$dst/" 2>&1) || _rc=$?
    if (( _rc == 0 || _rc == 23 || _rc == 24 )); then
      echo "[CACHE] ${name}: rsynced ($(du -sh "$dst" 2>/dev/null | cut -f1))"
      return 0
    fi
    echo "[CACHE] ${name}: rsync failed (rc=${_rc}: ${_err})" >&2
    return 1
  fi
  # Fallback: cp -an (archive, no-clobber). Iterate top-level entries so we can
  # exclude the tmp* / .* patterns ourselves. cp -a implies recursion.
  shopt -s nullglob dotglob 2>/dev/null
  local entry _bn
  for entry in "$src"/*; do
    _bn="$(basename "$entry")"
    case "$_bn" in
      tmp*|.tmp_*|.*) continue ;;
    esac
    cp -an "$entry" "$dst/" 2>>/dev/null || _rc=$?
  done
  if (( _rc == 0 )); then
    echo "[CACHE] ${name}: cp-copied ($(du -sh "$dst" 2>/dev/null | cut -f1))"
    return 0
  fi
  echo "[CACHE] ${name}: cp failed (rc=${_rc})" >&2
  return 1
}

# Atomic tar+zstd. tmp+rename so readers never see a partial tarball.
# NOTE: don't use --exclude='.*' — that matches the '.' source arg and emits an empty tar.
tar_safe() {
  local src="$1" tarball="$2" name="$3"
  [[ -d "$src" ]] || return 1
  [[ -n "$(ls -A "$src" 2>/dev/null)" ]] || return 1
  local tmp="${tarball}.tmp.$$"
  if tar --zstd -cf "$tmp" --blocking-factor=8192 \
       -C "$src" --exclude='tmp*' --exclude='.tmp_*' . ; then
    mv "$tmp" "$tarball"
    echo "[CACHE] ${name}: tarball ($(du -sh "$tarball" 2>/dev/null | cut -f1))"
    return 0
  fi
  rm -f "$tmp"
  echo "[CACHE] ${name}: tar failed" >&2
  return 1
}
