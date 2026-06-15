# Hybrid (Mamba+attn+MoE) 1F1B EP-A2A-overlap x CUDA-graph - repro & findings

Reproduction scripts + code fixes for running HybridModel with MoE expert-parallel A2A overlap
(`--overlap-moe-expert-parallel-comm`) under TE per-layer CUDA graphs on GB300 (cmh). FSDP OFF
(DDP + distributed optimizer); EP and FSDP A2A-overlap do not compose.

## Code changes (megatron/core)
- `transformer/cuda_graphs.py`: guard `chunk.mtp` in TECudaGraphHelper static-input assert (HybridModel
  has no `.mtp` on non-last PP stages) -> broad-scope TE-CG captures all layers on hybrid. [LANDED]
- `models/common/utils.py` `_BackwardDWWrapper`: MambaLayer-aware (was TransformerLayer-only) -> clears the
  "Pop empty queue" in Mamba backward_dw under overlap. [WIP - necessary but NOT sufficient]
- `models/hybrid/fine_grained_callables.py`: arm set_te_cuda_graph_backward_dw_wrapper() for graphed Mamba +
  route Mamba backward_dw through the wrapper. [WIP]

## Scripts (paths are user-specific - edit BASE/MEGATRON_LM_DIR/IMAGE/deepep_jit dir)
- `lyris_proxy_train.sh` : single-run launcher. Knobs: MODEL=gpt|hybrid, OVERLAP=0|1, EP_OVERRIDE, PP_OVERRIDE,
  NLPVPS (GPT VPP) / HYBRID_PATTERN (hybrid VPP via |-segments), CUDA_GRAPH=te|full_iteration, TE_CG_SCOPE,
  RANKS_PER_NVL_DOMAIN, DELAY_WGRAD, LOG_INTERVAL, NAN_OFF.
- `sweep_*_ab.sh` : sbatch wrappers, baseline+overlap A/B in one allocation. `mon_job.sh <jobid>` : log monitor.

## Matched results (EP32 / PP2 / VPP2 / 16 nodes, hybridEP, mxfp8, recompute)
| config | baseline | overlap |
|---|---|---|
| GPT                         | ~895 TF | ~897 (parity) |
| Hybrid, Mamba EAGER (nomamba scope) | ~790 | ~603 (-24%, numerically correct) |
| Hybrid, Mamba CAPTURED (broad scope) | ~907 (CG parity w/ GPT) | NaN (see below) |

## Key findings
- Hybrid TE-CG **baseline** parity with GPT achieved (907 >= 895), via the two crash fixes above.
- **Mamba-captured + overlap NaNs on the FIRST graph replay**: iter1 (eager capture) is bit-identical-correct
  (grad norm 50.261 == eager & captured-no-overlap); iter2 (first graph engagement) grad norm AND params norm = nan.
  Ruled out: delayed-wgrad (DELAY_WGRAD=0 still NaN), buffer-reuse (`_reuse_graph_input_output_buffers=False` still
  NaN), per-microbatch instances, the wrapper. => the fused mamba_ssm selective-scan does not compose with TE
  make_graphed_callables under the EP-overlap reordering (reads stale/garbage memory on replay). Fix is
  vendored-scan / TE-capture CG-replay-safety - kernel-owner territory.
- **+7-10% overlap win is NOT reproducible on this proxy** (intra-NVLink EP32 A2A too cheap; GPT itself = parity).
  Needs the real DSv3-671B + cuteDSL + mxfp8 + full_cg regime.
