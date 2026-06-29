---
orphan: true
---

# Determinism glossary

This glossary is for the developer references. The user guide avoids these
abbreviations where possible.

| Term | Meaning |
| --- | --- |
| MCore | Megatron Core, the model-parallel training library in this repository. |
| DSV3 | DeepSeek-V3-style model configuration used by the certification proxies. |
| Nemotron | Nemotron-3-Ultra-style hybrid model configuration used by the certification proxies. |
| MoE | Mixture of Experts: a layer routes tokens to one or more expert networks. |
| DP | Data parallelism: replicas process different batches and synchronize gradients. |
| TP | Tensor parallelism: one layer is partitioned across devices. |
| PP | Pipeline parallelism: consecutive layer ranges run on different devices. |
| VPP | Virtual pipeline parallelism: interleaved pipeline chunks used to reduce pipeline idle time. |
| EP | Expert parallelism: experts are partitioned across devices. |
| CP | Context parallelism: one sequence is partitioned across devices. |
| DDP | Distributed data parallel wrapper and its gradient synchronization. |
| TE | Transformer Engine, NVIDIA's transformer-kernel library. |
| FP32 / BF16 / FP16 | Floating-point formats. FP32 has higher precision than BF16 and FP16. |
| Cross-allocation | Comparison of runs that use independently assigned resources, rather than repeated work in one allocation. |
