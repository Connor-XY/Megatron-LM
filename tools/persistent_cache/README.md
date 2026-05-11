# Persistent Cache

First-iteration acceleration across job restarts. Caches Triton autotune results,
TorchInductor FX graphs, CUDA PTX→SASS, cuDNN frontend heuristics, NCCL topology,
hybrid_ep NVCC JIT, and dataset index across job boundaries.

Measured win on a 64-card Mamba+MoE+hybrid_ep run: iter 1 drops from 140 s to
96 s (−31 %) on warm restart, with bootstrap extract overhead < 1 s.

## Architecture (three storage tiers)

```
${PERSISTENT_CACHE}/                         ◄── Lustre (durable)
├── cache_read/<scope>.tar.zst                  read-only at job time
└── cache_write/<scope>/<files>                 sidecar's rsync/cp sink

/tmp/mcore_cache_${SLURM_JOB_ID}/<scope>/    ◄── per-node, ephemeral
                                                live working cache
```

**Read = tarball, write = directory tree.** Splitting them avoids Lustre MDT
invalidation storms from concurrent reader/writer access on the same dir, and
lets every job-start be one sequential tarball read instead of thousands of
small-file opens.

## Scripts

| File                  | When                              | What                                              |
| --------------------- | --------------------------------- | ------------------------------------------------- |
| `lib.sh`              | sourced by all                    | shared functions + env defaults                   |
| `bootstrap.sh`        | sourced per rank in rank entrypoint | per-node seed of `/tmp` + per-rank env export    |
| `sidecar.sh`          | spawned `&` once per node         | long-running periodic writeback loop + EXIT trap |
| `writeback.sh`        | invoked by sidecar / save-hook / atexit | one-shot rsync `/tmp/<scope>/` → `cache_write/<scope>/` |
| `promote_tarballs.sh` | launcher's `srun -p cpu` between jobs | tar `cache_write/<scope>/` → `cache_read/<scope>.tar.zst` |

## Required env

Set by the launcher before sourcing `bootstrap.sh`:

```bash
export PERSISTENT_CACHE=/lustre/.../<user>/.cache/mcore_persistent
# Derived (optional override):
export CACHE_READ_DIR="${PERSISTENT_CACHE}/cache_read"
export CACHE_WRITE_DIR="${PERSISTENT_CACHE}/cache_write"
```

Tunables (defaults shown):

```bash
export MCORE_CACHE_SYNC_DELAY_SECONDS=600     # don't rsync before compile is likely done
export MCORE_CACHE_SYNC_FREQUENCY=300         # sidecar loop interval
export MCORE_CACHE_SYNC_JITTER_SECONDS=120    # spread initial rsync across 16 nodes
export MCORE_CACHE_SIDECAR_ENABLED=1          # set 0 to disable sidecar entirely
export MCORE_CACHE_SCOPES="triton inductor cuda_ptx hybrid_ep cudnn_fe nccl_topo dataset_idx"
```

## How to use (launcher integration)

```bash
# 1. In sbatch wrapper, before srun:
export PERSISTENT_CACHE=/lustre/.../<user>/.cache/mcore_persistent
mkdir -p "${PERSISTENT_CACHE}/cache_read" "${PERSISTENT_CACHE}/cache_write"

# 2. Refresh stale tarballs out-of-band on a CPU partition:
srun -N1 -n1 -p cpu -q cpu-short -A "${SLURM_ACCOUNT}" -t 00:15:00 --quiet \
    bash "${MEGATRON_LM_DIR}/tools/persistent_cache/promote_tarballs.sh" \
    || echo "[CACHE] promote step failed (first job will compile cold)"

# 3. Add the new CLI flags to your training command:
OPTIONS+=" --persistent-cache-read-dir ${PERSISTENT_CACHE}/cache_read"
OPTIONS+=" --persistent-cache-write-dir ${PERSISTENT_CACHE}/cache_write"

# 4. In your rank entrypoint, source bootstrap and spawn sidecar:
if [[ -n "${PERSISTENT_CACHE:-}" ]]; then
    # shellcheck disable=SC1091
    source "${MEGATRON_LM_DIR}/tools/persistent_cache/bootstrap.sh"
    if [[ "${MCORE_CACHE_SIDECAR_ENABLED:-1}" == "1" ]]; then
        bash "${MEGATRON_LM_DIR}/tools/persistent_cache/sidecar.sh" </dev/null &
    fi
fi

# 5. Exec the training command — env vars from bootstrap stick:
exec python pretrain_mamba.py …
```

## Cache scopes

| Scope        | Env var                       | What's cached                                         |
| ------------ | ----------------------------- | ----------------------------------------------------- |
| `triton`     | `TRITON_CACHE_DIR`            | Triton compiled cubins + autotune choices             |
| `inductor`   | `TORCHINDUCTOR_CACHE_DIR`     | torch._dynamo FX graphs + autograd cache              |
| `cuda_ptx`   | `CUDA_CACHE_PATH`             | CUDA driver PTX→SASS                                  |
| `hybrid_ep`  | `HYBRID_EP_JIT_DIR`           | NVCC-JIT compiled .so for DeepEP hybrid_ep dispatcher |
| `cudnn_fe`   | `CUDNN_FRONTEND_CACHE_DIR`    | cuDNN heuristic results                               |
| `nccl_topo`  | `NCCL_TOPO_DUMP_FILE`         | NCCL topology dump (informational; read-side broken on NCCL 2.28 at >64 GPUs) |
| `dataset_idx`| `--data-cache-path`           | Megatron dataset blend / index files                  |

## Failure modes

| Failure | What happens | Recovery |
| ------- | ------------ | -------- |
| Tarball missing | `[CACHE SEED] <scope>: no tarball` | Cold compile, sidecar populates `cache_write/`, next promote builds tarball |
| Tarball corrupt | `tar --zstd -xf` fails non-fatally | Cold compile; bash logs warning |
| `/dev/shm` noexec | exec-probe falls back to `/tmp` | Automatic |
| `rsync` not in container image | sidecar uses `cp -an` fallback | Automatic, logged once |
| Job crashes before writeback | `cache_write/` may be empty for this job | No corruption; next job compiles cold |
| Sidecar hangs | atexit terminates it after 60 s wait | Bounded exit delay |
| Lustre temporarily unreachable | Tar/rsync fails | Logged non-fatal; training unaffected |

## How to opt out

Don't pass `--persistent-cache-read-dir` or `--persistent-cache-write-dir`. The
Python side becomes a no-op; the bash side never sources without `PERSISTENT_CACHE`.

## Reference

- Architecture inspired by `nemo-rl-internal!241` (75f48d75b).
- Design notes: `~/Projects/mcore_persistent_cache_design.md` (out of tree).
