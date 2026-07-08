---
orphan: true
---

# Determinism developer reference

> **Audience:** Megatron developers and reviewers. This reference explains the
> implementation and debugging tools; it is not the setup guide for people
> launching training.
>
> Start with the [Deterministic Training user guide](../../user-guide/deterministic-training.md)
> for supported setup and constraints. The tracked documentation contains
> conclusion-level evidence only; raw run provenance is deliberately local and
> excluded from version control.

## Contents

1. **[`status.md`](./status.md)** — start here. Supported behavior, limitations,
   evidence policy, and the upstream state without internal run provenance.
2. **[`training-path.md`](./training-path.md)** — a forward→backward→optimizer walk
   that flags every point where determinism enters or is decided, with file:line
   refs and a ✔/◆/▲/✖ status for each (shape-coded legend defined there).
3. **[`op-catalog.md`](./op-catalog.md)** — the per-operation catalog table
   (det? / det path / non-det path / how selected / evidence / perf Δ / gap), plus
   the perf hotspot priority list and the verification backlog.
4. **[`glossary.md`](./glossary.md)** — expansion of the parallelism and kernel
   abbreviations used by the developer references.
5. **[`validation-evidence.md`](./validation-evidence.md)** — findings,
   conclusions, limits, and the evidence standard. Raw provenance is retained
   only in an ignored local record.
6. **[`dsv4-assessment.md`](./dsv4-assessment.md)** — DeepSeek-V4 determinism
   scope, DSA op verdicts, coverage added, effectiveness/performance, and the
   gaps (mHC, CSA, suite-on-dev) still open for a full DSV4 claim.

## Audit source candidates for the operation catalog

Run the zero-dependency AST audit with the repository Python after changing
collective, routing, indexing, or sorting code:

```bash
uv run python tools/determinism/audit_sensitive_ops.py megatron --git-tracked-only
uv run python tools/determinism/audit_sensitive_ops.py megatron \
  --git-tracked-only --category floating_collective_reduction --json
uv run python tools/determinism/audit_sensitive_ops.py megatron \
  --git-tracked-only \
  --verify-catalog docs/developer/determinism/op-catalog.md
```

The report is stable by file, line, column, and enclosing class/function. It
resolves common `torch.distributed` import aliases and separates floating
collective reductions, rank-indexed permutation collectives, indexed
reductions, indexed writes/gathers, ordering operations, and explicit
determinism-control calls. Common control-plane collisions such as queue
`put`, `asyncio.gather`, and arbitrary module `embedding` calls are filtered;
tensor `put_`, tensor/`torch.gather`, and PyTorch embedding calls remain in the
review queue. `--git-tracked-only` excludes local scratch files so the report
represents the branch under review. Repeat `--category` to select multiple
classes and `--exclude` to omit a source-root-relative glob.

This is a candidate inventory, not a determinism verdict: whether a call is on
the training path, has unique indices, reduces integer-valued data, or is
protected by a deterministic branch still requires code/runtime analysis. Use
the audit to find missing catalog rows, then record the classification and
evidence in [`op-catalog.md`](./op-catalog.md). The catalog-verification mode
requires an explicit disposition for every audited source file and matches a
line-number-independent fingerprint of every path, symbol, category, and call;
new or changed sensitive operations therefore require a deliberate catalog
refresh.

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
trace below for recompute, optimizer, and collective boundaries; async TP
userbuffer payloads and in-process kernel selection remain follow-up surfaces.
Captured kernel identities can be recovered offline with the Nsight attribution
tool described below.
Only compare trusted `.pth` files because PyTorch dump loading uses pickle
serialization.

## Gate external training recipes from their console logs

Some launchers, including Megatron Bridge performance recipes, do not yet enter
the MCore structured-trace context. Use the strict console-log comparator as an
integration gate for two otherwise identical runs:

```bash
python tools/determinism/compare_training_logs.py \
  /path/to/run-a.log /path/to/run-b.log \
  --expected-iterations 50 --json
```

The comparator reads every logged iteration and compares the exact serialized
value of every pipe-delimited metric. By default it requires `lm loss` and
`grad norm`, and excludes only elapsed time, throughput, energy, and power.
Only fields closed by a `|` delimiter and following Megatron's leading-space
field convention belong to the iteration record. After the first metric, an
adjacent rank prefix appended directly by Slurm terminates the record, so that
rank's own pipe-delimited timing or memory fields are also ignored.
Identical duplicate rank-zero records are accepted for aggregated distributed
logs; conflicting duplicates, missing iterations, missing required metrics, or
any other metric difference fail the gate. Additional required or volatile
fields can be selected with repeatable `--require-metric` and
`--ignore-metric` options. Exit code 0 is an exact serialized match, 1 is a
complete but divergent comparison, and 2 is invalid or ambiguous input.

This is stronger than sampling one loss value, but it is not a bitwise tensor
certificate: equal printed metrics can hide an earlier tensor difference below
the logging precision. Use it to gate an external training loop, then use
`certify_traces.py` or `compare_dumps.py` when the loop exposes the corresponding
MCore instrumentation.

## Trace recompute and optimizer boundaries during training

The structured runtime tracer complements tensor dumps with semantic JSONL
events. Select one or a few iterations; every rank writes independently, so the
tool adds no collectives or cross-rank ordering constraints:

```bash
pretrain_gpt.py ... \
  --determinism-trace-dir /path/to/run-a \
  --determinism-trace-interval 100 \
  --determinism-trace-tensor-hashes \
  --determinism-trace-optimizer-state
```

The trace records determinism-relevant runtime configuration, forward/backward
and optimizer boundaries, Megatron and Transformer Engine activation-checkpoint
forward/recompute identities, output-discarding checkpoint recomputation, and
semantic MoE expert-parallel all-to-all boundaries. Runtime metadata includes
PyTorch/CUDA/cuDNN, device and allocator properties, installed
Transformer-Engine/Triton/Mamba/FlashAttention package versions, and the
cuBLAS, NCCL, TE-attention, Mamba, and Triton environment controls that govern
deterministic or autotuned kernel selection. Package discovery uses installed
metadata and does not import optional kernel libraries. After every traced TE
attention forward, the wrapper also records the backend TE actually selected,
its sub-backend, the configured selector, forward/recompute phase, layer, mask,
QKV layout, and tensor shapes/dtypes. If the installed TE version no longer
exposes the diagnostic state, it emits a separate
`te.attention.backend.unavailable` event instead of guessing. With
`--determinism-trace-tensor-hashes`, it also records exact wgrad, updated
parameter, checkpoint, all-to-all, pipeline P2P, DP gradient-reduction, and
distributed-optimizer parameter-gather hashes. End-of-backward finalization
additionally fingerprints deterministic TP SUM/AVG, PP embedding/replicated-
parameter all-reduces, token-count broadcast/reduction boundaries, and the exact
integer SUM that drives router expert-bias updates. It also
fingerprints the standard synchronous TP all-reduce, first/last-dimension
all-gather, and
first/last-dimension reduce-scatter paths in original forward, activation
recompute, and backward. Core TP linear tracing additionally covers the
synchronous sequence-parallel forward gather and the backward async gather,
dgrad all-reduce, and dgrad reduce-scatter. Async outputs are recorded only when
the existing work-handle wait completes; the tracer does not add a collective
or distributed wait. Exact hashing copies device tensors to the CPU and
synchronizes execution; use it only for targeted debug iterations because those
synchronizations can perturb communication overlap. Without that flag, phase,
checkpoint, and collective tensor metadata remain available without the byte
copies; optimizer boundary tensors are omitted. The additional
`--determinism-trace-optimizer-state` flag records local main parameters and
direct tensor/scalar optimizer state entries before and after the step, keyed by
stable optimizer, parameter-group, and parameter ordinals. It requires exact
tensor hashes and is deliberately separate because hashing Adam moments roughly
triples the optimizer-state bytes copied to the CPU.

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
invalid input. Use `--json` for automation. Matching remains independent of
cross-kind arrival order, but differences within each rank trace are reported
by the earliest local event sequence present on either side. Human and JSON
reports include the left/right sequence numbers, so the first displayed
difference is a causal boundary rather than the lexically first event name.
The current integration covers the
Megatron tensor-parallel and Transformer Engine activation-checkpoint
implementations, output-discarding checkpoint recomputation, optimizer
inputs/outputs, the standard MoE expert-parallel all-to-all dispatcher, pipeline
P2P sends/receives, DP all-reduce/reduce-scatter, and distributed-optimizer
parameter all-gather, synchronous TP mapping all-reduce/all-gather/
reduce-scatter, core TP linear synchronous/async collectives, final TP/PP
gradient and token-count synchronization, plus opt-in local
optimizer main parameters and moment state. TE FP8/FP4 checkpoint boundaries
are covered, but internal quantizer state is not independently fingerprinted.
TE attention selection is covered at runtime; TP userbuffer payloads and other
auto-dispatching libraries remain follow-up instrumentation surfaces. Actual
kernel identities from a captured iteration
can be recovered from an Nsight Systems SQLite export with
`attribute_nsys_ranges.py --kernels`. Final TP/PP gradient SUM and AVG use a
topology-independent deterministic implementation, while floating reductions
inside TP mappings, TP linears, and vocab-parallel cross-entropy remain open.
Megatron-FSDP gradient reductions are a separate uninstrumented
native-collective gap; its existing FSDP8 cells are same-topology tests, not
cross-allocation certificates.

For a scaled model run, use the stricter certifier instead of relying on a trace
comparison alone:

```bash
python tools/determinism/certify_traces.py \
  /path/to/run-a /path/to/run-b \
  --expected-ranks 32 \
  --expected-iterations 2 \
  --require-event-prefix te.attention.backend.selected \
  --require-collective-prefix moe.ep_ \
  --require-collective-prefix moe.router_expert_bias. \
  --require-collective-prefix data_parallel. \
  --require-dp-fp32-accumulation \
  --require-dp-hierarchical-fp32-accumulation
```

`certify_traces.py` checks both trees independently before comparing them. It
requires deterministic runtime state, the requested rank/iteration/file counts,
matching activation recomputes, completed collective output hashes, zero pending
collectives, requested semantic event/collective surfaces, and (optionally) the
ordered fp32 and hierarchical data-parallel reduction paths. A traced
`te.attention.backend.unavailable` event fails closed instead of allowing a
backend-blind certificate.
It also rejects unsupported or incomplete event schemas, mixed rank/iteration
identities within one file, sequence gaps, duplicate or missing runtime and
iteration boundary markers, explicit `iteration.error` events, and collective
begins/ends that are not balanced within the trace window.
Groups of at most two are excluded from the hierarchy requirement: one rank has
no reduction and two ranks have exactly one floating-point addition per output,
so there is no reduction-tree order for physical topology to change. They still
must use fp32 accumulation when that invariant is requested. Exit code 0 is a
certificate, 1 is a failed invariant or cross-run divergence, and 2 is invalid
input. Use `--json` to retain the complete evidence report.

## Run the checked-in EP32 model certificates

The multi-node runner launches and certifies two independent deterministic runs
of either the DSV3-style or Nemotron-3-Ultra-style MCore proxy. Invoke it once
per node in an 8-node, 4-GPU-per-node allocation; it derives node rank and node
count from Slurm by default and writes the retained JSON report under
`$OUTPUT_PATH/determinism-certification/`:

```bash
bash tests/functional_tests/shell_test_utils/determinism/run_model_certification.sh dsv3
bash tests/functional_tests/shell_test_utils/determinism/run_model_certification.sh nemotron
```

`tests/test_utils/recipes/gb200/determinism-certification.yaml` registers both
32-GPU certificates as weekly L3 GB200 workloads. Each workload fails unless
all 32 ranks and both iterations are present, recomputes are exact, MoE and DP
collective surfaces include expert-bias token counts, no collective remains
pending, and every multi-rank DP reduction used hierarchical fp32 accumulation.

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

To identify the CUDA kernels actually launched by a matching NVTX range, join
the contained same-thread CUDA runtime calls to GPU activities by Nsight process
and correlation ID:

```bash
python tools/determinism/attribute_nsys_ranges.py \
  /shared/profiles/nemotron-rank0.sqlite \
  '_VocabParallelCrossEntropyBackward' \
  --kernels --top 20
```

This reports exact demangled kernel identities, launch counts, stream IDs, and
the matching canonical NVTX ranges. A launch covered by nested matching ranges
is counted once. Reported kernel durations are summed GPU activity time; kernels
on different streams can overlap, so the total is attribution evidence rather
than wall-clock step time. Use `--json` to retain the machine-readable report.

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
