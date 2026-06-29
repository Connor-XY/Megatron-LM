---
orphan: true
---

# Deterministic training status

> **Audience:** developers evaluating or extending deterministic training. If you
> are launching a model, start with the [Deterministic Training user
> guide](../../user-guide/deterministic-training.md).
>
> This page intentionally describes supported behavior, limitations, and the
> evidence required for a claim. It does not contain cluster names, job IDs, or
> artifact hashes. The tracked [validation evidence](./validation-evidence.md)
> page reports findings and conclusions; detailed run provenance remains only in
> an ignored local record.

## Documentation boundaries

| Audience | Start here | Purpose |
| --- | --- | --- |
| Training user | [Deterministic Training](../../user-guide/deterministic-training.md) | Enable the mode, understand its constraints, and validate a run. |
| Developer | [training-path.md](./training-path.md) and [op-catalog.md](./op-catalog.md) | Follow the control flow and review deterministic versus ordinary operation paths. |
| Maintainer / reviewer | [validation-evidence.md](./validation-evidence.md) | Review findings, limits, and the standard required for a determinism claim. |

Acronyms used in the developer references are defined in the
[glossary](./glossary.md).

## Upstream state

[PR #5041](https://github.com/NVIDIA/Megatron-LM/pull/5041) is open against
`NVIDIA/main`. This branch is stacked on that PR and contains additional
determinism implementation, tests, diagnostics, and evidence. Those additional
commits have no separate upstream PR yet and are not an upstream guarantee.

## Behavioral contract

With fixed model configuration, input order, seed, software versions, hardware
class, and parallel layout, deterministic mode aims to reproduce model outputs,
gradients, and training metrics exactly. It is not enough to choose the same
collective algorithm: floating-point reductions must also use a stable
accumulation order.

The public switch is `--deterministic-mode`. It configures safe runtime
defaults, enables PyTorch deterministic algorithms, and rejects or disables
features without a certified deterministic path.

## Current capabilities

| Surface | Current behavior |
| --- | --- |
| Model-level coverage | DeepSeek-V3-style and Nemotron-3-Ultra-style proxy models have exact-output and exact-gradient certification coverage. |
| Mixture-of-experts routing | Certified standard all-to-all dispatch is used in deterministic mode; unsupported fused dispatchers fail closed. |
| Gradient reduction | One-rank reductions use an identity path, two-rank supported dtypes use a native single-addition path, and larger groups use fixed-order FP32 accumulation. |
| Transformer Engine attention | The trace records the backend selected by an actual attention forward and fails certification if that selection cannot be observed. |
| Divergence diagnosis | Tensor-dump and structured-trace tools locate the first observable mismatch without adding training collectives. |
| Performance safety | Fast paths are accepted only when the deterministic contract is unchanged; larger floating-point reductions retain the fixed-order implementation. |

## Current limitations

- Fused cross-entropy loss, tensor-parallel communication overlap, multiple
  distributed-optimizer instances, collective averaging, and Megatron-FSDP do
  not currently provide the required cross-allocation deterministic contract.
- Some attention kernels and mixture-of-experts operations can be slower in
  deterministic mode. Do not globally force an attention backend; eligibility
  depends on the model input and installed kernel library.
- A matching loss alone is not a certificate. Use the trace or dump tools for
  a targeted investigation, and use paired independent launches for scaled
  qualification.
- Full production-scale Bridge qualification remains a maintainer validation
  gate, not a user-facing promise.

## Performance guidance

Performance varies with model shape, parallel layout, and kernel availability.
The implementation avoids deterministic overhead where arithmetic proves it is
unnecessary, including one-rank identity reductions and safe two-rank native
reductions. For larger floating-point reductions, reproducibility takes
priority over the lowest-latency native collective.

Use the profiler workflow in the developer reference to measure a representative
workload before changing a deterministic branch. Record the workload, software
versions, and comparison method; detailed run provenance is intentionally not
tracked in this repository.

## Evidence policy

A determinism claim needs all of the following:

1. A code-path review that identifies the deterministic branch.
2. A focused test for the relevant operation or model component.
3. Two independent launches with the required outputs or trace events compared
   exactly at the intended scale.
4. A recorded performance comparison when the branch changes runtime cost.

The [operation catalog](./op-catalog.md) records the required evidence by
operation. [Validation evidence](./validation-evidence.md) summarizes the
findings and conclusions without retaining environment-specific provenance.
