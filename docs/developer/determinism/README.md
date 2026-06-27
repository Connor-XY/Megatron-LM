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
localizes surfaces that the existing dump hooks capture. Collectives, optimizer
internals, recompute identities, and allocator decisions still require the typed
instrumentation API described in [`status.md`](./status.md). Only compare trusted
`.pth` files because PyTorch dump loading uses pickle serialization.

## Maintenance

Keep the catalog **evidence-based**: classify each op via PyTorch/TE/NCCL docs, an
explicit code branch, or a `BitExactRunner` result — and fill perf deltas from the
nsys det-vs-nondet leaderboard, never by estimate. See "How this catalog is
maintained" in [`op-catalog.md`](./op-catalog.md).
