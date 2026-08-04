# The M≥3 Stall — GTP + 1F1B EP Overlap on HybridEP

Investigation record, 2026-08-03. Nemotron-3 a55b: GTP64 + HybridEP EP64 + cuteDSL + mxfp8,
32 nodes / 128 GB300, cluster `cmh`.

---

## Bottom Line

1F1B EP-comm overlap is clean at M ≤ 2 and hangs from M = 3 up.

**The hang is inside HybridEP, in a device-side barrier — not in mcore, and not in the allgather.**

The current best candidate is `csrc/hybrid_ep/backend/hybrid_ep_backend.cuh`, `device_sync_kernel`:
an exact-equality spin (`while (flag_data != expected)`) with **no timeout**, over a **persistent
ping-pong parity**, launched around *every* dispatch and combine. Any desync spins forever and
silently. See [The Real Candidate](#the-real-candidate--device_sync_kernel). Under test as job
2806745.

**A prior claim on this page has been refuted.** An earlier version of this document named the
custom NVLink allgather (`ag_nvl_kernel`, `iter_id` launch-overlap race) as the root cause and
proposed routing the allgather through NCCL as the fix. **That fix was tested and failed** — see
[Attempt 1 Refuted](#attempt-1-refuted--the-allgather-was-the-messenger). The allgather was merely
the one handshake instrumented with a `TIMEOUT` and a `printf`; it complained, so it looked guilty.
Silence it and the same desync simply hangs mutely elsewhere.

Everything else was refuted too. Note that the one change which does make the hang disappear —
`CUDA_LAUNCH_BLOCKING=1` — is blanket serialization costing ~100× throughput. It *masks* the race
rather than fixing it, but it is a real clue: serialization is exactly what defeats a
launch-overlap or interleaving bug.

---

## The Symptom

Device-side waits of the form `HYBRID-EP ALLGATHER TIMEOUT:SM n [0]:expecting X got Y`.

| M | GBS | overlapped steps/iter | mismatch lines | verdict |
|---|---|---|---|---|
| 1 | 128 | 0 | 0 | clean |
| 2 | 256 | 1 | 0 | clean |
| 3 | 384 | 2 | 125 | hangs |
| 4 | 512 | 3 | 299 | hangs |
| 8 | 1024 | 7 | 63 | hangs |
| 64 | 8192 | 63 | 2071 | hangs |
| 240 | 30720 | 239 | 954 | hangs |

> **Read this column as zero vs non-zero only.** The line counts are accumulation artifacts of how
> long each job hung before being killed, not severity. See *The mismatch counts are NOT a severity
> measure* below.

`M = GBS / (MBS × DP)`; DP = 128, so `M = GBS / 128`.

Shortfalls are 1–7 against expected counts of 100480 (M=64) and 370560 (M=240) — roughly **five
parts per million**, non-growing, and spread across all 32 SMs. Capacity overflow or drift would
scale with transfer size or accumulate over time; neither happens. Routing is force-load-balanced,
so expected counts are static: `expecting 104320 got 104315` means five flag writes that never
landed, not five dropped tokens.

## The Cliff Is Exact

`megatron/core/pipeline_parallel/combined_1f1b.py` —
`combined_1f1b_schedule_for_no_pipelining` loops `for i in range(num_microbatches - 1)`, so the
number of overlapped steps is `M − 1`. **The break is exactly at two consecutive overlapped
steps.** It is a cliff, not a gradual degradation, and the line count does not scale with M.

This rules out a large class of explanation. The per-layer body inside an overlapped step does not
depend on M, and in `TransformerLayerSchedulePlan.run` the `f_layer` and `b_layer` are always
different layer objects (54 layers; `i == N−1−i` has no integer solution). **Any mechanism that is
structurally identical at M=2 cannot be the cause**, because M=2 is clean.

## Minimal Repro

**M=3, GBS=384, 32 nodes, single 1F1B arm, 5 iterations, profiler off** — 80× less compute than
the M=240 case it replaces, same signature. `gtp-bisect-m3.sbatch`, inner sha `aa66737f`.

## Calibration — Read Every Arm Against Iteration 2

Stock M=3 completes **exactly one iteration** (368,926 ms) and the first mismatch appears
immediately after that iteration line. **The failure occurs during iteration 2.**

An arm that clears iteration 1 has shown nothing; stock does that too. This single fact
invalidates the naive reading of three separate runs, so it is stated up front.

---

## Confirmed Mechanism

`megatron/core/transformer/moe/fused_a2a.py`, `HybridEPDispatch.forward`:

```python
# If we provide the num_permuted_tokens, we do not need to use sync to
non_blocking = num_permuted_tokens is not None
...
non_blocking=non_blocking,
```

mcore derives `non_blocking` **directly** from whether a precomputed count was supplied. The chain:

1. Our recipe sets `--moe-expert-rank-capacity-factor 2` and `--moe-permute-fusion`.
2. `token_dispatcher.py setup_metadata` therefore computes the count analytically:
   `budget = int(padded_num_tokens × topk × capacity_factor)`, rounded to `pad_multiple`.
3. A count is supplied → `non_blocking=True`.
4. `deep_ep/hybrid_ep_buffer.py` then skips its synchronization — *"If non_blocking is True, no
   stream synchronization will be used"*. The blocking path exists to guarantee
   *"the data in the pinned_memory_buffer: num_dispatched_tokens_tensor is ready"*.

**The sync-free permute path is active in every run we have ever made.** This is read off the
source, not inferred from behaviour.

Corroborating: with capacity `None` the code instead takes
`self.num_permuted_tokens = self.tokens_per_expert.sum()`, which the source comments mark
**"(CPU sync)"**.

---

## The Sync Cannot Be Isolated

Both available knobs couple synchronization to other semantics, so neither can move it alone.

### Overlay v1 — override `num_permuted_tokens`

Also sets the permuted output **size**. Forward went actual-sized while backward stayed
budget-sized:

```
HybridEPCombineBackward returned an invalid gradient at index 0 -
got [360448, 2048] but expected shape compatible with [181248, 2048]
```

### Overlay v2 — override `non_blocking`

Also moves metadata to pinned/CPU memory (*"the metadata outputs are on the GPU… Otherwise
`tokens_per_expert` is copied through pinned memory"*). Downstream permute-fusion Triton kernels
then receive a CPU pointer:

```
ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)
  triton/runtime/jit.py run → backends/nvidia/driver.py __call__
```

The overlay was demonstrably **live**, not inert: iteration 1 ran 417,380 ms vs stock 368,926 ms,
**+13.1%** — the signature of a per-dispatch `cudaStreamSynchronize`. This check existed precisely
because an inert overlay and a real fix both produce "clean" logs.

### The flag route is impossible

Dropping `--moe-expert-rank-capacity-factor` to reach the sync path dies in argument validation:

```
ValueError: moe_paged_stash requires moe_expert_rank_capacity_factor to be set;
            there is no need to use paged stashing without it.
```

and `--moe-paged-stash` cannot be dropped alongside it — paged-stash-off already fails earlier for
unrelated reasons. The gate itself worked (config dump showed `moe_expert_rank_capacity_factor
None`, backend still `hybridep`), so this is a genuine dependency chain.

---

## Key Deduction — The Counts Already Agree

The v1 shape error exposed real per-rank numbers, and that run had **padding off**, so
`padded_num_tokens == num_tokens`:

- **budget = 360448, uniform on every rank**
- actual permuted counts differ: 180480, 180736, 181248, 181760, 182016 (~0.8% spread, despite
  force load balancing)

Since `budget = int(padded_num_tokens × topk × capacity_factor)`, a uniform budget implies
**`num_tokens` is already uniform across ranks**. Two consequences:

1. Every rank hands HybridEP the **same** count, so `expecting N got N−k` is **not** a
   count-disagreement problem. It is a missing synchronization.
2. **Padding to the group max is a numerical no-op.** Its entire effect is the
   `torch.distributed.all_reduce(op=MAX)` barrier it inserts before each dispatch — so padding was
   never a targeted fix, only a cheap blanket one.

This retires an experiment: an overlay to log per-rank `num_tokens` was built
(`mcore-overlay-tokcount-log-v1.tar.gz`, sha `f472359c…`) to settle count-vs-sync. The uniform
budget answers it directly. **Do not spend a 32-node slot on it.**

---

## Every Hypothesis And Its Fate

| # | hypothesis | test | result |
|---|---|---|---|
| 1 | staging-buffer race between steps | `comm.wait_stream(comp)`, CG on (2795066) | **refuted** — 61 lines |
| 2 | same, graph-replay confound | as above, CG off (2797933) | **refuted** — 63 lines |
| 3 | ranks disagree on token counts | `pad_uneven=1`, CG on (2798200) | mismatches 125→0, then **IMA** |
| 4 | shared per-dispatch dispatcher state | analysis | **refuted** — identical at M=2 |
| 5 | insufficient work queues | `CDMC=32` (2802283) | **refuted** — still hangs (mismatches non-zero) |
| 6 | padding is a real fix | pad, CG **off**, no launch-blocking (2802548) | **refuted** — **IMA**, CG-independent, control-confirmed |
| 7 | force the sync via count | overlay v1 (2803640) | **refuted as instrument** — shape error |
| 8 | force the sync via `non_blocking` | overlay v2 (2804079) | **refuted as instrument** — Triton CPU pointer |

### Why the stream barrier could never have worked

The count that mismatches is read through **pinned host memory**, and HybridEP is
**device-initiated** (GDAKI). A host-issued `wait_stream` orders *kernel launches*; it does not
establish that the remote side observed the data. Attempts 1 and 2 aimed at the right boundary and
the wrong resource.

### Why padding failed

Job 2802548 arm 1, with CG **off** and **no** launch-blocking, config verified in-log
(`PAD_UNEVEN=1 LAUNCH_BLOCKING=unset`, dump shows `moe_hybridep_pad_uneven_dispatch_inputs …
True`). Iteration 1 clean at 362,251 ms (−1.8% vs stock — the `all_reduce` is genuinely cheap),
then:

```
[PG GUID 3(GTP_WEIGHT_REMAT_GROUP) Rank 6] Process group watchdog thread terminated
with exception: CUDA error: an illegal memory access was encountered
```

followed by SIGTERM via `--distributed-timeout-minutes 15`.

This is the **same IMA** as job 2798200, which had CG **on** — therefore **the IMA is not a
CUDA-graph interaction**. Padding trades the mismatch hang for an illegal memory access. The
earlier probe that looked clean (2799733, 5/5 iterations) was confounded: `CUDA_LAUNCH_BLOCKING=1`
serialized everything and masked it.

### The CDMC deviation (found, and separately important)

The inner recipe applies `CUDA_DEVICE_MAX_CONNECTIONS=32` to every 1F1B variant — but only when
`GTP_SKIP_CDMC != 1`, and every driver we ran passes `GTP_SKIP_CDMC=1`. CDMC fell back to the CUDA
default of 8. Verified: `EFFECTIVE_CDMC=unset` in every log carrying the marker.

- **Not a confound for the cliff** — constant across the entire sweep, clean M=2 runs included.
- **Not a fix** — CDMC=32 still hung (mismatches non-zero at iteration 2), refuting work-queue
  concurrency as the mechanism.
- **But a real caveat for the throughput numbers**, tracked separately: every perf comparison,
  including the headline M=2 `+0.699%`, ran the 1F1B arm on 8 work queues instead of the 32 the
  recipe intends. mcore's MoE README lists `CUDA_DEVICE_MAX_CONNECTIONS > 1` as an EP-overlap
  requirement; 8 satisfies it, so nothing asserted and the deviation was silent.

---

## What Only "Works"

`CUDA_LAUNCH_BLOCKING=1` (job 2799733): 5/5 iterations clean, iteration 1 401,136 ms /
iteration 2 341,228 ms — steady state, ~100× off the ~1266 TFLOP/s baseline. Blanket
serialization. Useless as a fix, and it establishes that **any** sufficiently broad synchronization
hides the bug — which is why "a sync fixes it" carries little diagnostic weight.

---

## Artifacts

| item | sha256 / id |
|---|---|
| mcore archive | `93cf1ded2c95f694196d562d797874d49408bd8fd193492b6108a7e2343489fb` |
| TE wheel | `dcf3d308e6aa9116daf9dd01b5b599d150be8ed6b64085afbc26cd187499b630` |
| inner (stash) | `aa66737f6e20c1b64ed4026cf2baf0377b7ee25425c4a8b464242ea9d82675de` |
| inner (rootcause) | `1d63155eadeac0ae6016b32b113eb3bd158395d1093da599b54e2990eaeca6c6` |
| overlay: pad-uneven gate | `2c9e4feaeb7cca347e0ff35fd0d6074c73b6fd9fa825b7bbb2bcfe1360a1d2a9` |
| overlay v1: force count | `7dfd429a95c324093ca9bb06407260337628fb62791ce0066b9c1c0bde854283` |
| overlay v2: force non_blocking | `246729f257a8148444af9d7c56ab42e355610358eaf50765bb0aaa5c60de5c65` |
| overlay: tokcount log (retired) | `f472359c24a32b6f4c4133ae23f69932bbaf81110b4d7179420b6a5d946da70a` |

Overlay builders live beside the archives in `gtp-profile-staging/make_*_overlay.py`. Every overlay
is env-gated and defaults to off, so an unset environment reproduces stock behaviour exactly; the
pinned archive is never modified.

Jobs: 2795066 / 2797933 (step barrier), 2798200 (pad, CG on), 2799733 (pad + launch-blocking),
2799960 (capgate, blocked), 2802283 (CDMC=32), 2802548 (pad, CG off, no LB), 2803640 (overlay v1),
2804079 (overlay v2).

## The Control Landed — Padding Result Confirmed

Job 2802548 ran both arms in **one allocation**, which makes this a controlled A/B rather than a
comparison across jobs:

| arm | config | mismatch lines | outcome |
|---|---|---|---|
| 1 | `pad_uneven=1` | **0** | **illegal memory access** on `GTP_WEIGHT_REMAT_GROUP` |
| 2 | `pad_uneven=0` (stock) | many, still climbing | `expecting 8384 got 8382` — the classic hang |

The stock arm reproduces the hang in this exact allocation, so arm 1's IMA is **not** an
environmental artifact. **Padding converts the mismatch hang into an illegal memory access.**
Both arms' iteration 1 came in near-identical (362,251 vs 361,995 ms) and close to historical
stock (368,926 ms), so the allocation behaved normally.

### The mismatch counts are NOT a severity measure

While watching this control I recorded 62, then 125, then 187 lines from the same arm. A hung job
emits `HYBRID-EP ALLGATHER TIMEOUT` lines **continuously until something kills it**, so the count
measures *how long the job sat hanging before termination* — wall-clock, watchdog, or manual
cancel — not how badly it failed.

Consequences, and they reach backwards through this document:

- **The count column in the M table is not a magnitude.** M=64 showing 2071 and M=8 showing 63
  reflects termination timing, not that M=64 fails 33× harder. Read the table as
  **zero vs non-zero**.
- **Do not compare counts across arms.** The CDMC arm was noted as "187→243 vs stock 125, if
  anything worse" — that comparison is meaningless. What CDMC=32 actually showed is
  **non-zero, i.e. it still hangs**, which is all that was needed to refute it.
- A run that reaches 125 and one that reaches 2071 are the same result: **it hung**.

The signal to trust is binary — mismatches present or absent — plus the shortfall *size*, which is
genuinely stable (1–7, here 63×1 and 62×2 across 30 SMs) and is what supports the
parts-per-million missing-flag-write reading.

Note the expected count here is 8384, versus 100480 at M=64 — the absolute expected counts scale
with M while the shortfall does not.

---

## Where The HybridEP Code Actually Is

The full DeepEP source **ships inside the training image** — this was the unlock that turned a
black-box investigation into a source-level one.

| what | path |
|---|---|
| **Full source tree** | `/workspace/DeepEP/csrc/hybrid_ep/` |
| the allgather in question | `csrc/hybrid_ep/extension/allgather.cu` (280 lines) |
| dispatch/executor | `csrc/hybrid_ep/{hybrid_ep.cu,executor/executor.cu,buffer/intranode.cu}` |
| device headers (also installed) | `dist-packages/deep_ep/backend/hybrid_ep_backend.cuh` (375 KB) |
| Python layer | `dist-packages/deep_ep/{hybrid_ep_buffer,buffer,utils}.py` |
| compiled ext | `dist-packages/{hybrid_ep_cpp,deep_ep_cpp}.cpython-312-aarch64-linux-gnu.so` |
| version | `deep_ep 1.2.1+4214430` (matches image tag `hybridep42144303`) |

A copy of `csrc` + headers + Python is on Lustre at `gtp-profile-staging/hybridep-src/`, so it can
be read without allocating a node. The `.so` files also carry their build paths
(`/workspace/DeepEP/csrc/...`), which is how the tree was located.

## Root Cause — The `iter_id` Launch-Overlap Race

`csrc/hybrid_ep/extension/allgather.cu`, `ag_nvl_kernel`:

```cuda
auto iter_id = *iter_id_ptr;   // EVERY block reads this at kernel entry
iter_id++;
...
if (threadIdx.x == 0) {
    unsigned long long value_to_add = blockIdx.x == 0 ? MAX_BLOCKS - gridDim.x + 1 : 1;
    auto old_val_sm_sync = atomicAdd(flag_sm_ptr, value_to_add);
    is_last_SM = (gridDim.x == 1 || old_val_sm_sync + value_to_add == iter_id * MAX_BLOCKS);
}                                                    // ↑ EXACT equality
if (is_last_SM) {
    red.relaxed.sys.global.add.u64 [flag_nvl_ptr], 1;   // the ONE cross-rank increment
    *iter_id_ptr = iter_id;                            // written ONLY by the last SM
    auto expected = iter_id * rank_num;
    do { ld.relaxed... } while (flag_data < expected); // ← emits our timeout message, line 84
}
```

`flag_sm_ptr` accumulates exactly `MAX_BLOCKS` (256) per launch — block 0 adds
`256 − gridDim.x + 1`, the other blocks add 1, totalling 256 with our
`--moe-hybridep-num-sms 32`. "Last SM" is then detected by an **exact equality** against
`iter_id * MAX_BLOCKS`.

**The design assumes two launches never overlap on the GPU.** When launch *k+1* begins before
launch *k*'s last SM has written `*iter_id_ptr`:

1. both launches read the **same** `iter_id`;
2. `flag_sm` advances by 512, but only one block ever matches `iter_id * 256`;
3. so **two launches produce one** `red.add` on the cross-rank counter;
4. every peer spins until timeout reporting `expecting N got N−k`, where *k* is the number of
   ranks that collapsed a pair;
5. and because `*iter_id_ptr` never advances, the equality can never match again — **the desync is
   permanent**.

`executor.cu:46` launches it on `at::cuda::getCurrentCUDAStream()` — whatever PyTorch stream is
current — and there is exactly one `allgather_obj` per buffer (`hybrid_ep.cuh:83`), backed by the
single process-global `_buffer` in mcore's `fused_a2a.py`. One counter set, launched under
whatever stream context the caller happens to be in.

### Why we are on this path at all

`executor.cu` only uses the custom kernel when `config.num_of_nodes == 1`. That holds for us
because EP64 fits inside a single NVL72 domain (hosts are `nvl72dNNN-TNN`). Confirmed
arithmetically: the observed `expecting 8384` divided by `rank_num = 64` is exactly 131.

### It explains every observation

| observation | explanation |
|---|---|
| clean at M ≤ 2, hangs from M = 3 | M=2 runs one overlapped step — never two allgathers resident at once |
| shortfall tiny (1–7) and non-growing | *k* = ranks that collapsed a pair; not a quantity that scales |
| once broken, stays broken | `*iter_id_ptr` is never updated again, so the equality can never re-match |
| `CUDA_LAUNCH_BLOCKING=1` "fixes" it | serializes launches → no overlap → no collapse |
| the schedule-level stream barrier did nothing | it fenced *schedule steps*; the overlap is between *kernel launches* |
| padding "fixed" the mismatches | its `all_reduce` barrier also separated the launches |
| counts are not severity | it is a spin-until-killed loop |

## Attempt 1 Refuted — The Allgather Was The Messenger

The NCCL-allgather fix below was **tested and failed.** Jobs 2805681 (30 min) and 2805631 (75 min)
on cmh, overlay verified live via its marker on **127 and 126 of 128 ranks**, so this was a real
test and not an inert no-op. Both completed **exactly one iteration** and then hung in iteration 2 —
the same place stock fails — and were killed only by wall clock, with no IMA, no NCCL watchdog and
no exception.

| job | iter 1 done | killed | silence after iter 1 |
|---|---|---|---|
| 2805681 | 10:56:30 | 11:13:51 | **17m21s** |
| 2805631 | 10:56:22 | cancelled ~11:34 | **37m** |

Iteration 1 itself takes ~6.4 min *including graph capture*, so a healthy iteration 2 should be
faster, not 3–6× longer with no output. This is a hang.

### The interpretation trap — do not judge by mismatch count

**`mismatch=0` does not indicate success once the custom allgather is bypassed.**
`expecting X got Y` is emitted *only* by `ag_nvl_kernel`. Remove that kernel and the message can
never appear, whether or not the underlying bug persists — so a silent hang and a genuine fix look
**identical** on that metric. Judge by **iteration count** (must clear iteration 2). This trap
nearly produced a false positive here.

### And upstream #682 is not corroboration

DeepEP #682 really does delete `allgather.cu` and flip `enable_custom_allgather` to `False`
(verified in the post-#682 image). An earlier version of this page read that as independent
confirmation of the diagnosis. **It is not.** Upstream removed that kernel for their own reasons;
two things being true does not make one the cause of our hang.

## The Real Candidate — `device_sync_kernel`

`csrc/hybrid_ep/backend/hybrid_ep_backend.cuh`:

```cuda
uint32_t flag_parity = *parity;                                  // persistent state, read
uint32_t expected = expected_flag_value[flag_parity] + NUM_OF_RANKS_PER_NODE;
red.relaxed.sys.global.add.u32 [flags + flag_parity], 1;         // bump rank 0's counter
do {
    ld.relaxed.sys.global.u32 flag_data, [flags + flag_parity];
} while (flag_data != expected);                                 // EXACT inequality, NO timeout
expected_flag_value[flag_parity] = expected;                     // persistent state, written
*parity = flag_parity ^ 1;                                       // ping-pong parity
```

Launched `<<<1,1,0,stream>>>` around **every dispatch and every combine**
(~5700–5719 and ~5821–5840; with permute fusion enabled it runs after only, otherwise before *and*
after). So all 64 ranks in the NVLink domain must hit it the same number of times, in the same
order, without interleaving.

Why it fits better than the allgather did:

| property | allgather | `device_sync_kernel` |
|---|---|---|
| spin condition | `flag_data < expected` — terminates on overshoot | **`!= expected` — spins forever** |
| timeout | `TIMEOUT` + `printf` | **none — hangs mutely** |
| cross-call state | `iter_id` | **`parity` + `expected_flag_value`** — desync unrecoverable |
| touched by #682 | deleted | **untouched** |

It explains the current symptom exactly: no message, GPUs at 100%, and **no NCCL watchdog in
37 minutes** — because the ranks are not in an NCCL collective at all, they are spinning in a
HybridEP kernel.

### The lever, and why it needed a monkey-patch

`DEVICE_SIDE_SYNC` is a compile-time template parameter, but it is fed from **runtime** config
bools that *are* exposed to Python on `HybridEpConfigInstance`:
`device_side_sync_dispatch_api`, `device_side_sync_combine_api`. Both are hardcoded `true` in
`config.cuh` (lines 422/431) while every neighbouring knob is `get_env_int()`-overridable — so
there is no env switch. However `update_template_config` ends with

```python
for key, value in kwargs.items():
    setattr(config, key, value)
```

so a kwarg reaches them, and the JIT then compiles the `DEVICE_SIDE_SYNC=false` variant. **No
rebuild required.** Overlay `mcore-overlay-no-device-sync-v1.tar.gz`, sha
`692cdbef2872e0dc88e50837288594a97d2fd24bfd4a93210c54f78bd863c869`, monkey-patches
`HybridEPBuffer.update_template_config` from mcore so it works regardless of call path, and prints
`GTP_NO_DEVICE_SYNC: ...` for in-log verification.

**Status: job 2806745.** Clearing iteration 2 identifies the device-side barrier as the hang site.
A crash or wrong results is also informative — that barrier exists for a reason, so this is a
diagnostic, not a proposed production fix.

## Superseded — Route The Allgather Through NCCL (refuted, see above)

`executor.cu` already contains the escape hatch:

```cpp
if (config.num_of_nodes > 1 || !enable_custom_allgather) {
    torch_distributed.attr("all_gather_into_tensor")(...);   // NCCL — no iter_id counter
} else {
    allgather_obj.launch(...);                                // the buggy kernel
}
```

`enable_custom_allgather` is a plain Python constructor kwarg
(`deep_ep/hybrid_ep_buffer.py:52`, default `True`), so **no rebuild is needed** — mcore can pass
`False` and take the NCCL branch. This is the same change **DeepEP PR #682** makes upstream, which
independently suggests the custom path was known to be problematic.

Overlay `mcore-overlay-nccl-allgather-v1.tar.gz`, sha
`6a2c33fcced0b24dd77d0b24dd43c4f3c19d9968cb36db16b4c9f2bb9d99a46b`, injects it into mcore's
existing `kwargs` dict in `fused_a2a.py`, signature-guarded so it degrades safely on a `deep_ep`
lacking the kwarg, and prints `GTP_NCCL_ALLGATHER: enable_custom_allgather=False` so the change can
be **verified in-log rather than inferred from a clean run**.

**Status: under test as jobs 2805681 and 2805631.** The gate marker prints on 127/126 of 128 ranks,
so the overlay is confirmed **live**, not inert. Success criterion: clears iteration 2 with zero
mismatches. (Iteration 1 proves nothing — stock clears that too.)

### Upstream Made The Same Fix — Independent Confirmation

Inspecting shiqingf's post-#682 image on aga
(`.../nemotron_n4_pre/users/shiqingf/nt4/images/pyt26.04-temain4adad4c2-hybridep94a9f8f6-arm.sqsh`,
`deep_ep-1.2.1+94a9f8f`) shows that **DeepEP PR #682 fixed exactly this bug, in exactly this way**:

| check | post-#682 result |
|---|---|
| `csrc/hybrid_ep/extension/allgather.cu` / `.cuh` | **absent — the kernel is deleted** |
| `ALLGATHER TIMEOUT` string anywhere in `csrc` | **not found** — the symptom cannot occur |
| `deep_ep/hybrid_ep_buffer.py` `enable_custom_allgather` | **default flipped `True` → `False`** |

Two consequences:

1. **Our root-cause localization is independently confirmed.** Upstream deleted precisely the file
   we identified.
2. **Our overlay is not a workaround** — setting `enable_custom_allgather=False` is upstream's new
   default. The proper long-term fix is simply to move to a post-#682 DeepEP.

It also definitively closes the old, twice-mishandled claim that "the post-#682 image does not fix
the hang". Post-#682 **cannot emit the symptom at all**: no kernel, no message. That claim was
wrong, and both of the reasons previously given for setting it aside were also wrong (see the aga
correction above).

**Caveat on adopting those images wholesale:** they are on aga only (`nemotron_n4_pre` is not
mounted on cmh) and carry TE `2.19.0.dev0+4adad4c2`, in which a (weak) probe found no GTP symbols
— so our custom GTP wheel is likely still required, and that wheel is sm103a/GB300 while aga
dispatches `sm100`.

**Expected trade-off:** NCCL replaces a hand-tuned NVLink kernel, so the routing-map gather may
cost more. That is a throughput question to measure once correctness holds — and it is only one
allgather of the routing map, not the token payload.

## Handoff

This needs an ordering fix inside HybridEP's device-initiated dispatch, or inside mcore's 1F1B
schedule. Owners: Pingtian Li (1F1B overlap); Nan Zheng described an adjacent staging-buffer race
in `#swdl-nccl-ep` (`1780443120.171239`).

What to give them:

1. The M=3 minimal repro — 80× cheaper than the M=240 case originally reported.
2. The confirmed `non_blocking = num_permuted_tokens is not None` mechanism, with the note that
   `capacity_factor` silently selects the sync-free path.
3. The uniform-budget finding: ranks **agree** on the dispatched count, so this is a missing sync,
   not a count disagreement.
4. That the IMA under padding is **CG-independent**.
5. That the API couples sync to output shape (v1) and to metadata device placement (v2), so there
   is no user-side way to add only the synchronization.

## Open Questions

- Where exactly is the ordering guarantee missing — between consecutive dispatches on the comm
  stream, or between a dispatch and the consumer of the previous dispatch's staging buffer?
- Why does padding produce an IMA on `GTP_WEIGHT_REMAT_GROUP` specifically? That points at an
  interaction with GTP weight rematerialisation rather than at the MoE path alone.
- Does DeepEP PR #682 (`94a9f8f6b146c07d97ec58f67cd6d303296d6098`) change this? Still not properly
  evaluated — but **for a worse reason than previously recorded**. See the correction below.

## Correction — The aga Disqualification Was Wrong

An earlier version of this investigation set aside all `aga` results, and withdrew a
"post-#682 does not fix the hang" finding, on the grounds that aga "lacks GDAKI" and therefore
could not exercise the code path. **That reasoning was faulty.**

The observation itself is real and was re-verified live: 512 instances of

```
NCCL WARN gin_plugin.cc:396 Call to ncclGinGdakiCreateContext(cComm, config->nSignals,
config->nCounters, config->nContexts, config->queueDepth, config->trafficClass, ginCtx...
```

across 35 aga run logs, with zero such warnings on cmh. GIN/GDAKI context creation does fail on
every rank there.

But **GDAKI is NIC-initiated networking**, and the kernel that stalls is `ag_nvl_kernel` — the
*intra-node NVLink* allgather, selected only when `num_of_nodes == 1`, moving data by direct peer
stores and signalling with `red.relaxed.sys.global.add` over NVLink. **No NIC is involved**, so a
failing GIN plugin does not prevent that path from running. Corroborating: aga run 332135 showed
all four GPUs at 100% with every CPU thread blocked — our exact stall signature, i.e. aga was
*reproducing* the bug, not failing to reach it.

Two consequences:

1. The withdrawn #682 result may have been valid. It was discarded rather than examined, and the
   logs are still on disk.
2. "aga lacks GDAKI" overstates the measurement. What is established is that **GDAKI context
   creation fails** there; whether that is missing hardware capability or a driver / IB / plugin
   configuration problem was never determined.

aga is consequently back in use — it is ~90% staged and is running the NCCL fix test as job
347141, in parallel with cmh. It needs its own inner script (`gtp_hybridep_stash_aga.sbatch`,
which sets `NCCL_SHM_DISABLE=0`; the cmh value of 1 fails there with
`ncclIbIsErrorRecoverable(r, wc, i) failed: 3`). Note its overlay tarball sha differs from cmh's
purely because gzip embeds an mtime — the extracted `fused_a2a.py` is byte-identical
(`a3364fc6…`, verified on both).
