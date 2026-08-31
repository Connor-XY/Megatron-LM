# Reproducing the H100 cuDNN Fused-Attention Divergence

Transformer Engine's cuDNN fused-attention **forward** pass is not bitwise
reproducible across processes on H100. In one 8-GPU job, every rank feeds the
call bit-identical Q, K and V and every keyword argument, selects the identical
cuDNN `engineId` and `knobChoices`, and launches the same kernel with the same
grid and block. Two or three ranks still return a different result: 25,285 of
64,520,192 output elements differ, always the same elements, always by one
bfloat16 ULP.

NVFP4 PTQ turns that into a visible failure because amax is a max-reduction, so
one perturbed element in 64 million moves a quantizer scale. 296 of 3,036
quantizers shift, MoE routing changes, and training diverges. A loss-averaged
functional test averages the same perturbation away, which is why ordinary tests
pass and PTQ does not.

`--moe-permute-fusion` is not the cause. It is an incidence amplifier: the rate
drops from 9 of 10 repeats to 2 of 10 without it, while the two output values are
unchanged.

## Environment

The divergence has been observed on:

| component | version |
| --- | --- |
| GPU | H100 80GB HBM3, sm90, 8 per node |
| Transformer Engine | 2.16.0+b9d690e0 |
| cuDNN | 9.23 |
| CUDA | 13.3, driver 13030 |
| container | `pytorch-26.06-py3` |
| parallelism | TP=1, EP=8 |

## The Recipe

The test case is `hybrid_nemotron_v3_pico_7b_a1b_tp1_ep8_QAD_dgx_h100_1N8G`. It
is not on `main`. Retrieve it from the branch that introduced it:

```bash
git fetch origin pr4798-2-hybrid-stack-grouped
git show origin/pr4798-2-hybrid-stack-grouped:\
tests/functional_tests/test_cases/hybrid/hybrid_nemotron_v3_pico_7b_a1b_tp1_ep8_QAD_dgx_h100_1N8G/model_config.yaml
```

It was added in `e97ac8360`. Observations here were taken at `167ccbb2`.

Note the recipe already sets `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `NCCL_ALGO=Ring`, `TRITON_CACHE_AUTOTUNING=0`
and `MAMBA_DETERMINISTIC=1`. The divergence occurs anyway. See "Why the usual
determinism switches do not help" below.

## Running It

`run_split_repeat.sbatch` is a template. Fill in the site-specific values at the
top, then:

```bash
REPEAT_COUNT=10 sbatch run_split_repeat.sbatch
```

Repeats matter more than they usually would. See "Sample size" below.

To collect the evidence, the entry point needs the digest trace installed after
the model is built:

```python
from tools.determinism.cudnn_attention_repro.all_rank_digest_trace import (
    install_all_rank_digest_trace,
)

handles = install_all_rank_digest_trace(model)
```

and these set per repeat:

```bash
export MCORE_DIGEST_TRACE="$WORK/repeat-$N/digest-trace/rank{rank}.jsonl"
export MCORE_DIGEST_TRACE_MAX_LAYER=5
export MCORE_DIGEST_TRACE_MAX_CALLS=2
```

Then:

```bash
python tools/determinism/cudnn_attention_repro/analyze_split.py <run-root>
```

## What You Should See

```
                         run     repeat  ranks  splits  first diverging module
   nt3-qad-attn-full-trace    repeat-2      8      28  ...core_attention.fused_attention [output] minority=[4, 5]
   nt3-qad-attn-full-trace    repeat-3      8      28  ...core_attention.fused_attention [output] minority=[0]

repeats with a minority split: 9 of 10
  first diverging module decoder.layers.5.self_attention.core_attention.fused_attention: 9 of 9
  ranks ever in the minority: [0, 1, 2, 3, 4, 5, 6, 7]
  more than one rank appears, so this is not a single faulty GPU
```

Across 20 repeats spanning two independent trace jobs, 15 carried a split and
all 15 first diverged at the same module, on the output side. The minority set
differs every time and every rank appears in it at some point. It is not always a
small minority: 4-versus-4 partitions occur, which is why this tool counts any
two-group partition rather than requiring a small odd group.

## Sample Size

The per-repeat rate is a property of the configuration, not of the bug:

| configuration | rate |
| --- | --- |
| baseline, across six jobs | 0.40, 0.60, 0.83, 0.83, 0.90, 1.00 |
| warm CUDA JIT cache | 0.90 |
| fresh CUDA JIT cache | 0.40 |
| `--moe-permute-fusion` removed | 0.20 |

The baseline rate varies by a factor of two across nominally identical jobs, so a
five-repeat cell tells you very little on its own and the cell does not reveal
which regime it was in. Four conclusions in
this investigation were reversed when rerun at ten or more repeats, including
one that had already been circulated.

Two rules follow. Run intervention cells at ten repeats or more. And judge any
rate against a control that runs the *same* script with only the one variable
changed, verified by diffing the two scripts, rather than against a remembered
number.

A third rule, learned by getting it wrong: check what your classifier counts. An
earlier version of this analysis flagged a split only when 3 or fewer of the 8
ranks disagreed, which silently discarded even 4-versus-4 partitions and
undercut every rate. The matched control moved from 6 of 10 to 9 of 10 when that
was fixed.

Where possible, prefer the within-run cross-rank comparison this tool
implements. All 8 ranks are one sample, so it sidesteps the problem entirely.

## Why The Usual Determinism Switches Do Not Help

`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` cannot reach the forward pass. In the
cudnn-frontend bundled with this build, `set_deterministic_algorithm` is declared
only on `SDPA_backward_attributes` (`graph_properties.h:2181`) and
`SDPA_fp8_backward_attributes` (:2378). The forward class `SDPA_attributes`
(:1628) has no such setter, and Transformer Engine's
`fused_attn_arbitrary_seqlen_fwd_impl` takes no `deterministic` parameter, while
its backward counterpart does and calls it at
`fused_attn_f16_arbitrary_seqlen.cu:825`. There is no API through which to
express the constraint.

`CUDA_LAUNCH_BLOCKING=1` does not help either: 10 of 12 repeats still split,
against 9 of 10 in a matched control. That is informative rather than merely
negative. Launch blocking removes concurrency *between* kernels but not the order
in which CTAs *within* one kernel claim tickets from a global atomic counter, and
the sm90 fprop kernel assigns tiles that way.

## Mitigation

| configuration | repeats with a diverging rank | verdict |
| --- | --- | --- |
| baseline | 9 of 10 | — |
| `--moe-permute-fusion` removed | 2 of 10 | reduces, does not fix |
| fresh or per-rank CUDA JIT cache | 0.40 versus 0.90 | reduces, does not fix |
| `--attention-backend flash` | 0 of 11 | fixes |

Only the backend switch eliminates it. The others lower the rate enough to pass a
handful of CI runs and then fail later.

## No Minimal Reproducer

There is no standalone reproducer, and that is itself informative. Capturing the
real Q, K, V and all 18 non-null keyword arguments from a run where rank 5
diverged, then replaying them through `FusedAttention(softmax_scale)` on eight
concurrent ranks for five iterations, produced the identical digest on all eight.
The divergence needs full-model execution context and does not survive
extraction. Reproducing therefore requires the full recipe.

## Things That Have Been Ruled Out

Measured per rank inside single split runs, so none of these depend on
reproducing the failure twice:

- inputs, including every tensor and non-tensor keyword argument
- cuDNN plan selection: identical `engineId` multiset and `knobChoices`
- the entire cuDNN API boundary: an 825,641-line level-3 log diff between a
  diverging and a non-diverging rank differs only in `cudaDeviceId`, the `GPU=`
  header and timestamps, which is exactly the residue a normal-versus-normal
  control leaves
- launch geometry: same kernel, grid, block, shared memory, registers
- the generated kernel image, which is bit-identical across ranks
- buffer addresses, alignment, and free device memory
- memory contents, via poisoning every driver allocation `0x00` against `0xFF`
- ambient device state at plan finalization: clocks, temperature, power,
  utilization, perf state, throttle reasons

A note on method: run the control arm. A 40 GiB poison arena raised the minority
rate sharply, which one arm alone would have read as confirming an uninitialized
read. Running `0x00` against `0xFF` showed the shift was identical in both, so it
came from allocation churn rather than the bytes.

## Follow-Up Experiments (nvbug 6489647)

The bug review asked for four things: cuDNN 9.25, `CUDA_LAUNCH_BLOCKING=1`,
`PYTORCH_NO_CUDA_MEMORY_CACHING=1`, and compute-sanitizer.
`run_intervention_repeat.sbatch` runs each as a named `INTERVENTION` so every
arm is diff-identical to its control except for one variable.

`CUDA_LAUNCH_BLOCKING=1` is already answered above: 10 of 12 repeats split
against 9 of 10 in a matched control. It does not help, and surviving launch
blocking is what an intra-kernel ordering dependence predicts, since blocking
serializes between kernels but not the order CTAs within one kernel claim
tickets from a global atomic counter.

For the remaining three:

| arm | what it tests | how to read it |
| --- | --- | --- |
| `cudnn925` | whether the newer cuDNN changes the fprop kernel or its tile-assignment scheme | 0 of 12 with a same-window control near its usual rate is a real fix signal; a nonzero rate at 12 repeats means not fixed |
| `nocache` | whether caching-allocator block reuse participates | expect a rate *shift* either way; the 0x00-vs-0xFF poison arena already showed allocation churn moves the rate without being the cause, so only 0 of 12 splits would be surprising |
| `memcheck` / `racecheck` / `initcheck` / `synccheck` | memory corruption, shared-memory races, uninitialized reads, invalid sync | any finding inside TE/cuDNN/Megatron MoE kernels is signal; NCCL noise under memcheck is expected and not signal |

Two cautions carried over from the main investigation. First, judge every rate
arm against a control submitted in the same window with the same script —
`INTERVENTION=control` exists for exactly that. Second, a clean sanitizer run
is weak evidence of absence here: the divergence is one bf16 ULP from two
valid accumulation orders, which is not a memory error, so the sanitizer arms
test the *alternative* hypothesis (corruption / missing sync), not the primary
one.

The `cudnn925` arm swaps only `libcudnn` via the `nvidia-cudnn-cu13` wheel
(`LD_LIBRARY_PATH` + `LD_PRELOAD`), keeping torch, TE, and the container
fixed, and aborts unless both `ctypes` and `torch.backends.cudnn.version()`
report the expected version. Swapping the whole container would change
torch/TE/cuDNN at once and say nothing about which moved the rate.

The sanitizer arms cut `--calib-size` to 16 (the divergence is on the first
calibration batch, so coverage survives) and disable the caching allocator for
`memcheck`/`initcheck`, since suballocation from cached blocks hides
out-of-bounds and uninitialized reads from those tools.
