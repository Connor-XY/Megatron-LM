# GTP + 1F1B MoE Overlap: Measured Findings

Nemotron-3 a55b, GTP64 + HybridEP EP64 + cuteDSL + mxfp8, 32 nodes / 128 GB300 GPUs on CMH2.
All numbers are 1F1B relative to GTP alone on the same allocation.

## From a 10% Regression to Break-Even

Enabling 1F1B on this recipe used to cost 10-18% throughput. It now costs nothing. Four separate
things closed that gap, and only the last two are configuration.

**Correct gradients under delay-wgrad.** The overlap needs `--delay-wgrad-compute`, which was
banned with EGTP + cuteDSL because it produced wrong gradients: grad norm 3.4 against a reference
9.7. The deferred weight gradient for the dense and shared GTP linears was computed into a full
unsharded scratch buffer and then dropped, because TE's `backward_dw` assumed fused accumulation
had already placed it in `main_grad`. Reduce-scattering that buffer into the sharded `main_grad`
before it is discarded makes every parameter class bit-identical to the eager reference. Routed
experts were never affected; the cuteDSL grouped MLP already did this explicitly.

**Migration to current MCore and TransformerEngine.** Moving from the prototype checkout to
MCore `gtp_release` 269949e and TE 2.19.0.dev0 removed most of the remaining deficit. TE's fused
MXFP8 grouped MLP now passes caller buffers to the cuDNN frontend kernel as `d_tensor`, which
eliminates device-to-device copies but requires cuDNN frontend 1.26.0. The version TE declares,
1.25.0, does not carry that signature. On current sources the deficit measured about 2%.

**Honest measurement.** Arms had been run A-then-B with one sample each, which aliases allocation
drift onto the arm effect. One such run reported 1F1B 6.488% *faster*. Running A-B-B-A instead,
with a 2% per-arm CV gate, put the real figure at -2.033%.

**Two settings, worth about 2 points together.** Detailed below.

| stage | 1F1B against GTP alone |
|---|---|
| prototype, wrong gradients | -10% to -18% |
| current sources, fixed gradients, fixed-order measurement | about -2% |
| A-B-B-A measurement, correct baseline | -2.033% |
| non-cuteDSL wgrad kernel | -0.910% |
| `CUDA_DEVICE_MAX_CONNECTIONS` left at default | -0.065% |

## The Two Settings

- **`NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP=1`** — worth about 1.7 points.

  The fine-grained schedule reserves two slots for delay-wgrad work, `b.mlp.backward_dw` and
  `b.pre_dispatch.backward_dw` (`model_chunk_schedule_plan.py:236`). Those slots exist to cover
  the dispatch and combine all-to-all. Running without `--delay-wgrad-compute` leaves them empty.
  Running with it fills them, but the cuteDSL wgrad kernel then contends with GTP's all-gather:
  all-gather kernel time rises 7.5%, from 829.9 ms to 891.9 ms, which alone accounts for the whole
  2% deficit. The non-cuteDSL wgrad kernel keeps the slots covered without the contention.
  All-gather time returns to 823.7 ms against a GTP-alone mean of 838.7 ms.

- **Leave `CUDA_DEVICE_MAX_CONNECTIONS` at the driver default** — worth 0.33 points.

  The profiling launcher exports `CUDA_DEVICE_MAX_CONNECTIONS=32` for every non-`no1f1b` variant,
  so the 1F1B arms ran at 32 channels while the GTP-alone arms ran at the driver default. A third
  of the apparent 1F1B penalty was that asymmetry, not 1F1B.

Recommended configuration:

```
--overlap-moe-expert-parallel-comm --delay-wgrad-compute
NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP=1
CUDA_DEVICE_MAX_CONNECTIONS      leave unset
--moe-hybridep-num-sms 32
NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
```

Result: two accepted runs at -0.291% and +0.162%, mean -0.065%. 1F1B is throughput-neutral, so it
can be enabled for its memory and scheduling properties without paying for it.

## Configuration Sweep

Both HybridEP dimensions are bracketed on each side and both defaults are optimal.

| HybridEP SMs | result | | combine chunk | result |
|---|---|---|---|---|
| 16 | -0.494% | | 64 | -0.108% |
| 32 (default) | -0.065% | | 128 (default) | -0.065% |
| 64 | -2.304% | | 512 | -1.157% |

## Larger Global Batch: 1F1B Is Faster by 0.70%

1F1B interleaves micro-batch N's forward with micro-batch N-1's backward, so it needs at least two
micro-batches per step. The exact recipe's `--global-batch-size 128` with `--micro-batch-size 1`
and DP 128 gives one, so the mechanism is inert there. `GBS=256` gives two.

Absolute throughput rises for both arms at `GBS=256`, to about 1205-1231 TFLOP/s/GPU against about
1165, so the comparison uses a `GBS=256` baseline.

Two arms run per allocation, with the order reversed in a companion job, because the four-arm
design overflowed the paged stash on its third arm. Thirteen accepted runs:

| ordering | 1F1B position | mean | SD | n |
|---|---|---|---|---|
| A then B | second | -0.120% | 0.947 | 8 |
| B then A | first | +1.518% | 0.943 | 5 |

Arm effect **+0.699%**, SE 0.269, t 2.60, p about 0.029.

## Position Penalty: Balance Ordering or the Measurement Is Noise

Whichever arm runs second in an allocation is **0.82% slower**. That penalty is larger than the
effect under test, so the two orderings measure `T - P` and `T + P`, and only their average
recovers the arm effect.

The naive pooled mean over the same thirteen runs is +0.510%, biased low because eight of them
place 1F1B in the penalised second position.

This explains results that previously looked like failed replications. A run measuring +1.692%
and a later run of the identical configuration measuring -0.472% are the same effect seen through
opposite position bias, not a contradiction. The two orderings have near-identical spread, 0.947
and 0.943, which is what a constant position offset separating two equivalent distributions looks
like.

Any A/B throughput comparison on this cluster must balance ordering. Unbalanced comparisons
produce differences indistinguishable from the position artefact.

## Refuted

- Dense-AG stream priority. The mechanism is inert on this stack: all all-gather streams stay at
  CUDA priority 0, confirmed by a separate ProcessGroupNCCL probe.
- Combine-only and all-operation dense-AG arbitration. Both target roughly a third of the deficit,
  and the hook is ignored when delayed wgrad is inactive.
- Capping concurrent in-flight all-gathers. The captured graph already runs at time-weighted
  concurrency 1.05 with a peak of 2, so there is nothing to cap.
- GTP CTA throttling. Dropping 32 to 8 CTAs makes isolated GTP 3.2x slower to buy 1-6% of
  concurrent span.
- Disabling the cuteDSL fused grouped MLP entirely. `ScaledSReLU(activation_recompute_in_mlp=True)`
  requires that path.

## Measurement Method

Arms run A-B-B-A on one allocation so the mean time position of each arm is identical and linear
drift cancels. Throughput is averaged over iterations 14-20, after the profiler window and its
contaminated exit iteration. Any arm whose CV exceeds 2% rejects the run.

Six favourable results failed verification: +6.488% (fixed A-then-B ordering), +1.655% and
+0.957% (degraded baselines), +0.428% (reversed to -1.039% on replication), and the larger-batch
+1.692% (reversed to -0.472% and -0.928% on the same ordering). The +0.957% case re-ran clean at
-1.157%, a 2.1-point swing. Every over-estimate favoured 1F1B, because allocation degradation
lands on ABBA position 4, which is a baseline arm.

Between-run SD is about 1%, larger than per-arm CV within a run. Single allocations cannot resolve
sub-1% differences, so candidates need replication.

## Where This Leaves 1F1B

At `GBS=128`, the production operating point, 1F1B is throughput-neutral. One micro-batch per step
means the cross-micro-batch mechanism cannot engage, so parity is the structural ceiling.

At `GBS=256` the mechanism engages and 1F1B is faster by 0.699%, p about 0.029 over thirteen
accepted runs. Three of the last six allocations failed outright, so the sample is smaller than
intended and independent confirmation would strengthen it.
