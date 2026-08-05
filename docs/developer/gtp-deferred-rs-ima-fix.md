# GTP deferred reduce-scatter: illegal memory access under `--overlap-grad-reduce`

**Status:** root cause identified from the failing stack; one-line fix proposed and under test.
**Component:** `megatron/core/tensor_parallel/generalized_tensor_parallelism.py`
**Reported by:** Yan Xu (yxu1) — 2026-08-05
**Affects:** GTP + 1F1B MoE overlap runs with `--delay-wgrad-compute` and `--overlap-grad-reduce`

---

## Summary

`_finalize_one_deferred_rs` fires DDP's gradient-ready hook **inside** a
`with torch.cuda.stream(record.rs_stream)` block. Under `--overlap-grad-reduce`, that hook
launches an asynchronous DDP reduce-scatter which allocates a fresh tensor via
`torch.empty_like`. The allocation is therefore stream-associated to the GTP RS stream, while
its real consumers run on the caller stream. Nothing calls `record_stream()` to correct the
association, so once the caching allocator is under pressure it can recycle that block while it
is still in use — surfacing as `cudaErrorIllegalAddress`.

The failure needs three conditions at once, which is why it looked intermittent:

| # | condition | why it matters |
|---|-----------|----------------|
| 1 | `--delay-wgrad-compute` | both deferred RS queues are gated on it; without it `finalize_deferred_gtp_rs` is a no-op |
| 2 | `--overlap-grad-reduce` | makes the DDP hook launch an *async* reduce-scatter, with its own allocation, inside the RS-stream context |
| 3 | allocator pressure | with free blocks available the mis-associated memory sits idle and the race never fires |

---

## The failing path

```
finalize_deferred_gtp_rs                    generalized_tensor_parallelism.py:2101
 └─ _finalize_one_deferred_rs                                             :2069
     └─ _handle_megatron_grad_accum                                       :2087  ← inside rs_stream
         └─ start_grad_sync                 (DDP, async only when overlap_grad_reduce=True)
             └─ reduce_scatter_with_fp32_accumulation
                 └─ torch.empty_like        reduce_scatter_with_fp32_accumulation.py:77
                     └─ cudaErrorIllegalAddress
```

Observed frame counts from one crashed rank set: `reduce_scatter_with_fp32_accumulation` ×29,
`start_grad_sync` ×4, `_handle_megatron_grad_accum` ×4, `_finalize_one_deferred_rs` ×4,
`finalize_deferred_gtp_rs` ×4, `empty_like` ×2. Error text:
`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`.

`record_stream()` appears **zero** times in either `generalized_tensor_parallelism.py` or
`reduce_scatter_with_fp32_accumulation.py` — cross-stream ordering is done entirely with manual
events, so a missing one is unguarded.

### Current code (lines 2076–2098)

```python
    with torch.cuda.stream(record.rs_stream):
        record.rs_stream.wait_event(finalize_outer_event)
        record.handle.wait()
        reduced_wgrads = [cache.get(ticket) for ticket in record.output_tickets]
        if len(record.weights) == 1:
            record.weights[0].main_grad.add_(reduced_wgrads[0])
        else:
            torch._foreach_add_([weight.main_grad for weight in record.weights], reduced_wgrads)
        for weight in record.weights:
            record.leader._handle_megatron_grad_accum(weight)   # <-- allocates on rs_stream
        reuse_event = torch.cuda.Event()
        reuse_event.record(record.rs_stream)

    caller_stream.wait_event(reuse_event)
```

---

## Proposed fix

Move the grad-ready loop out of the stream context, to after the caller stream has synchronised:

```diff
-        for weight in record.weights:
-            record.leader._handle_megatron_grad_accum(weight)
         reuse_event = torch.cuda.Event()
         reuse_event.record(record.rs_stream)

     caller_stream.wait_event(reuse_event)
+    # Grad-ready fires on the CALLER stream, not rs_stream. Under --overlap-grad-reduce this hook
+    # runs DDP start_grad_sync -> reduce_scatter_with_fp32_accumulation, which allocates through
+    # torch.empty_like. Inside the rs_stream context that allocation is stream-associated to
+    # rs_stream while its real consumers run on the caller stream, and since that file makes no
+    # record_stream() call the caching allocator may recycle the block while it is still in use.
+    # The main_grad.add_ above is already ordered before this by the wait_event on reuse_event.
+    for weight in record.weights:
+        record.leader._handle_megatron_grad_accum(weight)
     for ticket in record.output_tickets:
```

**Ordering is preserved.** `caller_stream.wait_event(reuse_event)` already sequences the caller
stream after everything recorded on `rs_stream`, including the `main_grad.add_`. The hook
therefore still observes fully-accumulated gradients.

**Please sanity-check the intent behind the original placement.** The comment at lines 2077–2078
("A bucket-completing DDP hook may consume non-GTP grads produced since this RS was issued, so
refresh the caller dependency immediately before finalization") shows the position was chosen
deliberately. That concern is about `finalize_outer_event` making `rs_stream` wait on the caller,
which this change leaves untouched — but the GTP owner should confirm there is no second reason
the hook needs to be on `rs_stream`.

### Alternative

Add `record_stream()` for the buffers allocated in `reduce_scatter_with_fp32_accumulation.py`.
This fixes the same race lower down and would also cover any other caller that invokes DDP hooks
from a non-default stream. It is the broader fix; the one above is the narrower and cheaper one.

---

## Evidence

### A/B at identical configuration

Same model, same variant, same tight stash, same M=8, same NVLS/recompute settings — differing
only in the DDP async flags:

| run | `overlap_grad_reduce` | result |
|-----|----------------------|--------|
| job 372266 arm 1 | **on** (default) | **8 IMAs**, died after 1 iteration |
| job 372121 arm 1 | **off** | **0 IMAs**, 10/10 iterations, 1237.3 TFLOP/s |

*Caveat:* these are two different jobs on different nodes, not a within-job A/B. The intended
within-job control (`ogr=1` arm of job 372121) never ran — it died three times on an unrelated
DeepEP JIT compile failure. The `overlap_param_gather` flag was also off in the `ogr=0` arm,
because it asserts `overlap_grad_reduce`; so strictly the evidence implicates
"`overlap_grad_reduce` and/or `overlap_param_gather`". `start_grad_sync` is specifically the
gradient-reduce path, which makes `overlap_grad_reduce` much the likelier of the two.

### Pressure dependence

Stash tightness × microbatch count separates cleanly, across two clusters:

| runs | M | paged stash (cuda/cpu/cap) | IMA |
|------|---|---------------------------|-----|
| aga, 5 jobs, 2026-08-05 | 8 | **−1.1 / 0 / 2** (tight) | 3–25 |
| aga, DSv3, same day | 4, 8 | 1.5 / 1 / 4 (roomy) | 0 |
| cmh, `msweep2`, 2026-08-04 | 2 | −1.1 / 0 / 2 (tight) | 0 |
| cmh, `m8-perf`, 2026-08-04 | 8 | 1.5 / 1 / 4 (roomy) | 0 |

No run had combined a tight stash *with* M=8 before 2026-08-05, which is why the bug appeared to
be new. Peak memory in the failing arm was 207008 MiB reserved (of 284208 MiB).

### Why `--delay-wgrad-compute` is required

`finalize_deferred_gtp_rs` (line 2101) drains only `_delayed_rs_queue` and `_regular_rs_queues`.
`_regular_rs_queues` is populated behind `if GTP_CONFIG.delay_wgrad_compute:` (line 1636), and
`_delayed_rs_queue` is fed from TE's `backward_dw` → `finalize_group_grads(delayed_wgrad=True)` →
`wgrad_reduce_scatter_delayed`, which only runs under delay-wgrad. Without the flag this function
has nothing to drain and the path cannot be reached.

---

## Reproduction

Nemotron-3 a55b hybrid, 32 nodes / 128 GB300 GPUs:

```
--hybrid-layer-pattern MEMEMEM*E (×6, 54 layers)   seq 8192   EP 64   PP 1   TP 1   CP 1
GBS 1024   MBS 1   -> M = 8        --cuda-graph-impl full_iteration
--overlap-moe-expert-parallel-comm --delay-wgrad-compute --overlap-grad-reduce
--moe-paged-stash --moe-paged-stash-buffer-size-factor-cuda -1.1
--moe-paged-stash-buffer-size-factor-cpu 0 --moe-expert-rank-capacity-factor 2
--recompute-modules moe_act        NCCL_NVLS_ENABLE=0
```

Fails within 1–2 iterations. Relaxing the stash to `1.5 / 1 / 4` avoids it, as does dropping
`--overlap-grad-reduce` — both are workarounds, not fixes.

---

## Open items

- [ ] Fix validation in progress (job 372580: a55b, M=8, roomy stash + patch, ABBA). A stronger
      test would be **tight** stash + patch, since roomy alone already suppresses the IMA.
- [ ] Within-job `overlap_grad_reduce` A/B still owed — the earlier attempt was lost to an
      unrelated DeepEP JIT failure.
- [ ] Separate `overlap_grad_reduce` from `overlap_param_gather` (`OGR=1, OPG=0`).
- [ ] Decide between the narrow fix (move the hook) and the broad one (`record_stream()` in
      `reduce_scatter_with_fp32_accumulation.py`).

## Files

| path | lines | role |
|------|-------|------|
| `megatron/core/tensor_parallel/generalized_tensor_parallelism.py` | 2069–2098 | `_finalize_one_deferred_rs` — the fix site |
| `megatron/core/tensor_parallel/generalized_tensor_parallelism.py` | 1636 | `_regular_rs_queues` gated on `delay_wgrad_compute` |
| `megatron/core/tensor_parallel/generalized_tensor_parallelism.py` | 2101 | `finalize_deferred_gtp_rs` — queue drain |
| `megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py` | 77 | `torch.empty_like` — the faulting allocation |
