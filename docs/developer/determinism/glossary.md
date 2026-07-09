---
orphan: true
---

# Determinism glossary

This glossary is for the developer references. The user guide avoids these
abbreviations where possible. Terms are grouped: modes and guarantees first,
then measurement terms, parallelism, and model/kernel abbreviations.

## Modes and guarantees

| Term | Meaning |
| --- | --- |
| Deterministic mode | Execution with `--deterministic-mode`: `apply_determinism_to_args` validates/overrides unsupported features, sets the determinism env vars, and enables torch deterministic algorithms. Library code selects deterministic branches via `config.deterministic_mode` or `torch.are_deterministic_algorithms_enabled()`. |
| Default mode | Execution without `--deterministic-mode`. Also called "normal" or "ordinary" mode. |
| Torch deterministic algorithms | The global `torch.use_deterministic_algorithms(True)` state. It makes listed PyTorch ops select deterministic implementations and raises on ops with none; it is one ingredient of deterministic mode, not the whole contract. |
| Bit-exact / bitwise identical | Two runs produce byte-identical values for the compared tensors or serialized metrics. The strongest claim used here; "identical loss curve" alone is weaker because logging precision can hide low-bit drift. |
| Reproducible | Same result under the *same* conditions (allocation, caches, environment). Weaker than deterministic-across-allocations: a run can be reproducible within one allocation yet diverge on a different physical topology. |
| Cross-allocation | Comparison of runs that use independently assigned resources (fresh scheduler allocation, different physical nodes/rings), rather than repeated work in one allocation. The determinism target is cross-allocation bit-exactness. |
| Same-allocation | Comparison of repeated runs inside one allocation/process-group topology. Useful evidence, but it cannot certify topology-independence. |
| Certificate / certified | A checked evidence artifact: an exact comparison (trace, dump, or serialized-metric) between independent runs that passed the required invariants (e.g. via `certify_traces.py`). "Certified" claims in these docs always name their scope. |
| Fail closed | When deterministic mode meets a feature without a certificate, it rejects the configuration (assert/override) instead of silently running the uncertified path. |
| Collision-free (unique-index) write | An indexed write (`scatter`, `scatter_`, `index_put_(accumulate=False)`) whose indices are unique, so no accumulation happens and floating-point ordering cannot change the result. Deterministic without a special kernel. |
| Target workloads | The scoped claim of this work: DeepSeek-V3-style MoE/MLA and Nemotron-3-Ultra-style hybrid configurations. Audit dispositions use "target deterministic / outside target / conditional / forbidden" relative to this scope. |

## Measurement terms

| Term | Meaning |
| --- | --- |
| ABBA measurement | Paired benchmark run in alternating order (A, B, then B, A) so allocator, cache, and warm-up drift cancel instead of biasing one side. "Production ABBA" = ABBA on the production recipe. |
| Isolated (kernel/op) measurement | Microbenchmark of one op outside a training step. Establishes kernel-level speedup only; a training-step (end-to-end) win must be shown separately because overlap can hide or invert isolated gains. |
| Dispatch time vs wall-clock | NVTX/CPU op ranges measure launch/dispatch time, which overlaps GPU execution. Large per-op deltas there are *not* step-time deltas; only paired end-to-end step times are. |
| NVTX | NVIDIA Tools Extension ranges used by the nsys leaderboard and trace attribution. Range totals may overlap and must not be summed into step latency. |
| Same-topology test | A test that repeats work within one initialized process-group topology (e.g. the FSDP8 proxy cells). Valid for regression detection; not a cross-allocation certificate. |

## Parallelism and infrastructure

| Term | Meaning |
| --- | --- |
| MCore | Megatron Core, the model-parallel training library in this repository. |
| DP | Data parallelism: replicas process different batches and synchronize gradients. |
| TP | Tensor parallelism: one layer is partitioned across devices. |
| PP | Pipeline parallelism: consecutive layer ranges run on different devices. |
| VPP | Virtual pipeline parallelism: interleaved pipeline chunks used to reduce pipeline idle time. |
| EP | Expert parallelism: experts are partitioned across devices. |
| CP | Context parallelism: one sequence is partitioned across devices. |
| DDP | Distributed data parallel wrapper and its gradient synchronization. |
| A2A | All-to-all collective (MoE token dispatch/combine). Rank-indexed permutation, not a floating-point reduction. |
| TE | Transformer Engine, NVIDIA's transformer-kernel library. |
| FP32 / BF16 / FP16 | Floating-point formats. FP32 has higher precision than BF16 and FP16. |
| wgrad / dgrad | Weight gradient / data (input) gradient of a linear layer's backward pass. |

## Model and kernel abbreviations

| Term | Meaning |
| --- | --- |
| DSV3 / DSV4 | DeepSeek-V3- and DeepSeek-V4-style model configurations. DSV3 combines MLA with fine-grained MoE (plus MTP). DSV4 additionally uses DSA sparse attention. |
| Nemotron | Nemotron-3-Ultra-style hybrid model configuration used by the certification proxies (Mamba + attention + MoE). |
| MoE | Mixture of Experts: a layer routes tokens to one or more expert networks. |
| MLA | Multi-Latent Attention: low-rank latent q/kv projections (DeepSeek family). |
| DSA | DeepSeek Sparse Attention: a lightning indexer scores tokens and top-k selection sparsifies core attention. |
| MTP | Multi-Token Prediction: auxiliary layers predicting additional future tokens. |
| SSM | State-space model layers (Mamba family). |
| GDN | Gated delta net, an SSM variant (`ssm/gated_delta_net.py`). |
| FLA | flash-linear-attention: the external fused-kernel library providing the non-deterministic fast paths for GDN. |
| FAG | FlashAttention gradient (backward) kernel. "Deterministic FAG" means independent accumulation buffers plus a global deterministic sum (Longcat/DSV4 reports). |
