# Determinism (developer docs)

Developer reference for bit-exact deterministic training in Megatron-Core: the
current status, the per-op catalog, and the training-step branch map. This is the
foundation artifact for the determinism roadmap (validation, performance, tooling).

> User-facing "how do I turn it on" guide: `docs/user-guide/deterministic-training.md`.

## Contents

1. **[`status.md`](./status.md)** — start here. Definition of bitwise determinism,
   why it's hard, the perf targets (~15% → <10% → 5%), the control plane
   (`--deterministic-mode`, `determinism.py`, env vars), enforced limitations, the
   determinism branch surface, validation status, and known gaps.
2. **[`training-path.md`](./training-path.md)** — a forward→backward→optimizer walk
   that flags every point where determinism enters or is decided, with file:line
   refs and a 🟢/🔵/🟡/🔴 status for each.
3. **[`op-catalog.md`](./op-catalog.md)** — the per-operation catalog table
   (det? / det path / non-det path / how selected / evidence / perf Δ / gap), plus
   the perf hotspot priority list and the verification backlog.

## Locate the first difference in existing training dumps

Megatron can already save named activations, parameters, wgrads, and dgrads from
selected iterations. Run the same job twice with the relevant save interval,
then compare matching iteration directories:

```bash
python tools/determinism/compare_dumps.py \
  /path/to/run-a/activations/iter_0000100 \
  /path/to/run-b/activations/iter_0000100
```

The comparator matches relative rank-shard paths and semantic state-dict keys,
sorts layer numbers numerically, and checks the underlying tensor bytes. The
first reported difference includes tensor shape/dtype, SHA-256 hashes, mismatch
count, first mismatching index and values, and maximum absolute/relative error.
Use `--json` for automation and `--max-details N` to bound the report. Exit code
0 means bit-exact, 1 means a difference, and 2 means invalid input.

This avoids treating PP/VPP hook arrival order as execution order, but it only
localizes surfaces that the existing dump hooks capture. Use the structured
trace below for recompute and optimizer boundaries; collective payloads and
selected kernel identities remain follow-up surfaces. Only compare trusted
`.pth` files because PyTorch dump loading uses pickle serialization.

## Trace recompute and optimizer boundaries during training

The structured runtime tracer complements tensor dumps with semantic JSONL
events. Select one or a few iterations; every rank writes independently, so the
tool adds no collectives or cross-rank ordering constraints:

```bash
pretrain_gpt.py ... \
  --determinism-trace-dir /path/to/run-a \
  --determinism-trace-interval 100 \
  --determinism-trace-tensor-hashes
```

The trace records determinism-relevant runtime configuration, forward/backward
and optimizer boundaries, Megatron activation-checkpoint forward/recompute
identities, and semantic MoE expert-parallel all-to-all boundaries. With
`--determinism-trace-tensor-hashes`, it also records exact wgrad, updated
parameter, checkpoint, all-to-all, pipeline P2P, DP gradient-reduction, and
distributed-optimizer parameter-gather hashes. Async outputs are recorded only
after their existing completion boundary; the tracer does not add a collective
or distributed wait. Exact hashing copies device tensors to the CPU and
synchronizes execution; use it only for targeted debug iterations because those
synchronizations can perturb communication overlap. Without that flag, phase,
checkpoint, and collective tensor metadata remain available without the byte
copies; optimizer boundary tensors are omitted.

An `iteration.end` event reports `pending_collectives`. A nonzero value means a
collective launched inside the selected window but completed after it; the late
completion is intentionally not written into the closed iteration trace.

Run the same launch into a second directory, then align events by semantic
identity rather than arrival order:

```bash
python tools/determinism/compare_traces.py \
  /path/to/run-a /path/to/run-b
```

Exit codes match `compare_dumps.py`: 0 is a match, 1 is a divergence, and 2 is
invalid input. Use `--json` for automation. The current integration covers the
Megatron tensor-parallel activation-checkpoint implementation, optimizer
inputs/outputs, the standard MoE expert-parallel all-to-all dispatcher, pipeline
P2P sends/receives, DP all-reduce/reduce-scatter, and distributed-optimizer
parameter all-gather. TP reduction/gather payloads, TE FP8/FP4 recompute,
optimizer moment state, and actual selected kernel identities remain follow-up
instrumentation surfaces.

For a scaled model run, use the stricter certifier instead of relying on a trace
comparison alone:

```bash
python tools/determinism/certify_traces.py \
  /path/to/run-a /path/to/run-b \
  --expected-ranks 32 \
  --expected-iterations 2 \
  --require-collective-prefix moe.ep_ \
  --require-collective-prefix data_parallel. \
  --require-dp-fp32-accumulation \
  --require-dp-hierarchical-fp32-accumulation
```

`certify_traces.py` checks both trees independently before comparing them. It
requires deterministic runtime state, the requested rank/iteration/file counts,
matching activation recomputes, completed collective output hashes, zero pending
collectives, the requested semantic collective surfaces, and (optionally) the
ordered fp32 and hierarchical multi-rank data-parallel reduction paths.
Single-rank reductions are excluded from the hierarchy requirement because they
perform no inter-rank reduction. Exit code 0 is a certificate, 1 is a failed
invariant or cross-run divergence, and 2 is invalid input. Use `--json` to retain
the complete evidence report.

## Profile one distributed rank with Nsight Systems

Profiling every rank produces redundant reports and can exhaust host resources.
Run `profile_rank.py` as the Python entrypoint under `torchrun` to wrap exactly
one rank with Nsight Systems while the other ranks execute the training script
directly:

```bash
python -m torch.distributed.run \
  --nnodes "$NNODES" --nproc-per-node "$GPUS_PER_NODE" \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 29500 \
  tools/determinism/profile_rank.py \
    --profile-rank 0 --output /shared/profiles/nemotron-rank0 -- \
  pretrain_hybrid.py ... \
    --profile --nvtx-ranges \
    --profile-step-start 5 --profile-step-end 7
```

`nsys` must be on `PATH`, and the output path must be visible to the profiled
rank. The wrapper intentionally uses `--capture-range-end=stop`: Megatron's
CUDA-profiler range stops collection but the wrapped rank keeps running to the
same distributed completion point as its peers. Using `stop-shutdown` here can
terminate the profiled rank early and strand the rest of the worker group.

When an aggregate operator still has multiple possible call sites, export the
report to SQLite and print the same-thread NVTX containment chain for each
matching range:

```bash
nsys stats --force-export=true /shared/profiles/nemotron-rank0.nsys-rep
python tools/determinism/attribute_nsys_ranges.py \
  /shared/profiles/nemotron-rank0.sqlite 'aten::index_put_'
```

Parents are listed nearest-first. Containing ranges overlap by definition, so
their durations are attribution context and must not be added together. Use
`--json` to retain the complete machine-readable report.

Broad operators such as allocator fills can match hundreds of ranges. Aggregate
them by a canonicalized enclosing range, select the useful ancestor level, and
print only the largest groups:

```bash
python tools/determinism/attribute_nsys_ranges.py \
  /shared/profiles/nemotron-rank0.sqlite 'aten::fill_' \
  --summary --parent-depth 2 --top 20
```

`--parent-depth 1` is the nearest enclosing range. Sequence and operator IDs
are removed before grouping, so repeated invocations share one row. Every
matched range contributes to exactly one group; unlike nested parent durations,
the grouped matched durations can be compared and summed.

## Benchmark the deterministic data-parallel reduction

Use the distributed microbenchmark to compare the native NCCL reduce-scatter
with the rank-ordered fp32 implementation at the actual gradient-bucket size:

```bash
torchrun \
  --nnodes 8 --nproc-per-node 4 \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 29500 \
  tools/determinism/benchmark_reduce_scatter.py \
  --numel 41943040 --warmup 3 --iterations 10 \
  --hierarchical-group-size 4
```

Run both `--order native-first` and `--order ordered-first` to expose ordering
or cache effects, and repeat on the allocation topology used by the target
recipe: single-domain and cross-domain results can differ substantially. Rank 0
prints JSON with min/median/max latency, every sample, the ordered/native median
ratio, numerical-difference bounds, rank-to-host placement, and exact output
SHA-256 hashes by rank. The benchmark synchronizes each sample and reports the
slowest rank, so it measures isolated collective latency rather than
communication/computation overlap in a training step.

`--hierarchical-group-size` evaluates the same two-level deterministic path
available to production DDP through
`--ddp-reduce-scatter-hierarchical-group-size`. It first sums fixed contiguous
logical-rank groups, then exchanges fp32 partials and sums them in fixed group
order; it does not choose a tree from measured timing. The group size must
divide the world size and should map the target recipe's logical DP ranks onto
fast local communication domains. Keep that logical size fixed across runs.

## Maintenance

Keep the catalog **evidence-based**: classify each op via PyTorch/TE/NCCL docs, an
explicit code branch, or a `BitExactRunner` result — and fill perf deltas from the
nsys det-vs-nondet leaderboard, never by estimate. See "How this catalog is
maintained" in [`op-catalog.md`](./op-catalog.md).
