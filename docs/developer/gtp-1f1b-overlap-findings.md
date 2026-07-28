# GTP + 1F1B MoE Overlap: Measured Findings

Nemotron-3 a55b, GTP64 + HybridEP EP64 + cuteDSL + mxfp8, 32 nodes / 128 GB300 GPUs on CMH2.
All numbers are 1F1B relative to GTP alone on the same allocation.

## Two Settings Worth Changing Today

Together these take 1F1B from a 2% throughput tax to break-even, so it can be enabled for its
memory and scheduling properties without paying for it.

- **`NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP=1`** — worth about 1.7 points.

  The fine-grained schedule reserves two slots for delay-wgrad work, `b.mlp.backward_dw` and
  `b.pre_dispatch.backward_dw` (`model_chunk_schedule_plan.py:236`). Those slots exist to cover
  the dispatch and combine all-to-all. Running without `--delay-wgrad-compute` leaves them empty.
  Running with it fills them, but the cuteDSL wgrad kernel then contends with GTP's all-gather:
  all-gather kernel time rises 7.5% (891.9 ms against 829.9 ms), which alone accounts for the
  whole 2% deficit. Switching to the non-cuteDSL wgrad kernel keeps the slots covered and removes
  the contention. All-gather time returns to 823.7 ms against a GTP-alone mean of 838.7 ms.

- **Leave `CUDA_DEVICE_MAX_CONNECTIONS` at the driver default** — worth 0.33 points.

  The profiling launcher exports `CUDA_DEVICE_MAX_CONNECTIONS=32` for every non-`no1f1b` variant,
  so the 1F1B arms ran at 32 channels while the GTP-alone arms ran at the driver default. A third
  of the apparent 1F1B penalty was this asymmetry, not 1F1B.

Recommended configuration:

```
--overlap-moe-expert-parallel-comm --delay-wgrad-compute
NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP=1
CUDA_DEVICE_MAX_CONNECTIONS      leave unset
--moe-hybridep-num-sms 32
NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
```

## The Exact Recipe Runs One Micro-Batch Per Step

`--global-batch-size 128` with `--micro-batch-size 1` and DP 128 gives M = 1. 1F1B interleaves
micro-batch N's forward with micro-batch N-1's backward, which needs M >= 2. At M = 1 that
mechanism cannot engage at all, so every M = 1 measurement describes only the intra-micro-batch
overlap that `--overlap-moe-expert-parallel-comm` performs inside a layer.

At M = 1 the best configuration reaches parity: two accepted runs at -0.291% and +0.162%, mean
-0.065%. Every configuration dimension is bracketed with the defaults optimal.

| HybridEP SMs | result | | combine chunk | result |
|---|---|---|---|---|
| 16 | -0.494% | | 64 | -0.108% |
| 32 (default) | -0.065% | | 128 (default) | -0.065% |
| 64 | -2.304% | | 512 | -1.157% |

## M = 2 Is Feasible and the First Comparison Favours 1F1B

`GBS=256` gives M = 2. The paged stash must grow, because 1F1B holds two micro-batches of expert
activations; CUDA factor 1.03 overflows and 2.0 exhausts GPU memory. At 1.03 for the baseline and
1.2 for 1F1B, one rep of each arm completed:

| arm | TFLOP/s/GPU | CV |
|---|---|---|
| GTP alone | 1220.60 | 0.55% |
| GTP + 1F1B | 1227.33 | 0.44% |

That is +0.551% for 1F1B. Treat it as preliminary. The third rep overflowed the stash, so the
four-arm design did not complete and allocation drift is not controlled. A full ABBA at stash 1.3
is running.

Absolute throughput at M = 2 is higher for both arms, 1220-1227 against about 1165 at M = 1, so
an M = 2 comparison must use an M = 2 baseline.

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

Four favourable results failed verification: +6.488% (fixed A-then-B ordering), +1.655% and
+0.957% (degraded baselines), and +0.428% (reversed to -1.039% on replication). The +0.957% case
re-ran clean at -1.157%, a 2.1-point swing. Every over-estimate favoured 1F1B, because allocation
degradation lands on ABBA position 4, which is a baseline arm.

Between-run SD is about 1%, larger than per-arm CV within a run. Single allocations cannot resolve
sub-1% differences, so candidates need replication.

## Open

A complete four-arm ABBA at M = 2, replicated. That is the first test of 1F1B's actual mechanism,
and the preliminary sign is positive.
