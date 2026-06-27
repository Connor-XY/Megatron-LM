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
identities, and (with `--determinism-trace-tensor-hashes`) exact wgrad, updated
parameter, checkpoint-input, and checkpoint-output hashes. Exact hashing copies
device tensors to the CPU and synchronizes execution; use it only for targeted
debug iterations. Without that flag, phase events and checkpoint tensor metadata
remain available without the byte copies; optimizer boundary tensors are omitted.

Run the same launch into a second directory, then align events by semantic
identity rather than arrival order:

```bash
python tools/determinism/compare_traces.py \
  /path/to/run-a /path/to/run-b
```

Exit codes match `compare_dumps.py`: 0 is a match, 1 is a divergence, and 2 is
invalid input. Use `--json` for automation. The current integration covers the
Megatron tensor-parallel activation-checkpoint implementation and optimizer
inputs/outputs. Collective payloads, TE FP8/FP4 recompute, optimizer moment
state, and actual selected kernel identities remain follow-up instrumentation
surfaces.

## Maintenance

Keep the catalog **evidence-based**: classify each op via PyTorch/TE/NCCL docs, an
explicit code branch, or a `BitExactRunner` result — and fill perf deltas from the
nsys det-vs-nondet leaderboard, never by estimate. See "How this catalog is
maintained" in [`op-catalog.md`](./op-catalog.md).
