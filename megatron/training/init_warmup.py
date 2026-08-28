"""One-shot collective warmup to absorb first-call NCCL costs into init.

Iter 1 of a fresh job hits a 1.25 s outlier on the first NCCL AllReduce because
NCCL lazily selects algorithm/protocol and allocates staging buffers on the
first call. NCCL_RUNTIME_CONNECT=0 only handles channel setup, not algo/proto
selection or buffer registration.

This module runs synthetic collectives at end of init using the production
tensor sizes, forcing NCCL to populate its internal caches during init slack
(when the GPU is otherwise idle).
"""

from __future__ import annotations

import time
from typing import Iterable

import torch
import torch.distributed as dist

_WARMUP_DTYPES = (torch.float32, torch.bfloat16)
# 4 KB, 4 MB, 256 MB cover the per-channel ring-buffer size classes.
# 1 GB and 2 GB are added because production iter-1 collectives at TP=4
# tensor-parallel boundaries reach >1 GB on Nemotron-scale models — those
# trigger a different `cuMemAddressReserve` size class (Goal 4 VMM proposal,
# docs/VMM_PREINIT_PROPOSAL.md). NSYS on hsg 64c job 2807416 measured ~21 s
# in cuMemCreate/cuMemSetAccess/cuMemMap/cuMemImportFromShareableHandle in
# iter 1 — warmup at production sizes shifts those VMM ops into init slack.
# Peak per-rank GPU memory during warmup ≈ 2 × max_size (buf + shard),
# safe on GB200 (192 GB HBM3e) and GB300 (288 GB).
_WARMUP_SIZES_BYTES = (4 << 10, 4 << 20, 256 << 20, 1 << 30, 2 << 30)


def _gather_groups() -> list:
    """Collect all distinct NCCL process groups from Megatron parallel_state.

    NSYS attribution on hsg 64c job 2807416 showed ~29 s of iter-1 wall spent in
    `ncclCommInitRankConfig` across 7 distinct comm groups. The original warmup
    covered only 4 (DP+CP, TP, EP-DP, EP-TP). Expand to every getter that
    returns a process group, including pipeline, expert-model, hierarchical-CP,
    embedding, and combined groups. Each distinct (by id) group becomes a
    candidate for the synthetic AllReduce/AllGather/ReduceScatter warmup.

    Probes each getter defensively — many groups don't exist for a given
    parallel config (e.g. no PP if `pipeline_model_parallel_size == 1`). The
    getters' `check_initialized=False` argument is honored where supported.
    """
    from megatron.core import parallel_state as ps

    candidates: list = []

    def _try(getter_name: str, **kwargs):
        getter = getattr(ps, getter_name, None)
        if getter is None:
            return
        try:
            result = getter(**kwargs)
        except TypeError:
            # Getter doesn't accept the kwargs we passed (some don't take
            # check_initialized) — retry with no kwargs.
            try:
                result = getter()
            except Exception:
                return
        except Exception:
            return
        if result is None:
            return
        # `get_hierarchical_context_parallel_groups` returns a list of groups.
        if isinstance(result, (list, tuple)):
            candidates.extend(result)
        else:
            candidates.append(result)

    # Core groups (always present in a non-trivial parallel config).
    _try("get_data_parallel_group")
    _try("get_data_parallel_group", with_context_parallel=True)
    _try("get_context_parallel_group", check_initialized=False)
    _try("get_hierarchical_context_parallel_groups", check_initialized=False)
    _try("get_tensor_model_parallel_group")
    _try("get_pipeline_model_parallel_group", check_initialized=False)
    # Combined groups — Megatron sometimes creates a distinct NCCL comm for these.
    _try("get_model_parallel_group", check_initialized=False)
    _try("get_embedding_group", check_initialized=False)
    _try("get_position_embedding_group", check_initialized=False)
    _try("get_tensor_and_data_parallel_group", check_initialized=False)
    _try("get_tensor_and_context_parallel_group", check_initialized=False)
    # Expert groups (MoE).
    _try("get_expert_model_parallel_group", check_initialized=False)
    _try("get_expert_data_parallel_group", check_initialized=False)
    _try("get_expert_tensor_parallel_group", check_initialized=False)
    _try("get_expert_tensor_and_model_parallel_group", check_initialized=False)
    _try("get_expert_tensor_model_pipeline_parallel_group", check_initialized=False)
    _try("get_data_modulo_expert_parallel_group")

    out = []
    seen = set()
    for g in candidates:
        if g is None:
            continue
        gid = id(g)
        if gid in seen:
            continue
        seen.add(gid)
        out.append(g)
    return out


def warmup_collectives(verbose: bool = True) -> float:
    """Run one synthetic AllReduce, AllGather, ReduceScatter on every group/size/dtype.

    Returns the elapsed wall-clock time in seconds.
    """
    if not dist.is_initialized():
        return 0.0

    groups = _gather_groups()
    if not groups:
        return 0.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for group in groups:
        ws = dist.get_world_size(group=group)
        for dtype in _WARMUP_DTYPES:
            element_bytes = torch.finfo(dtype).bits // 8
            for size_bytes in _WARMUP_SIZES_BYTES:
                n_elem = max(size_bytes // element_bytes, ws)
                n_elem = (n_elem // ws) * ws  # divisible by world-size for AllGather/ReduceScatter
                if n_elem <= 0:
                    continue
                buf = torch.zeros(n_elem, dtype=dtype, device="cuda")
                dist.all_reduce(buf, group=group)

                shard = torch.zeros(n_elem // ws, dtype=dtype, device="cuda")
                dist.all_gather_into_tensor(buf, shard, group=group)

                dist.reduce_scatter_tensor(shard, buf, group=group)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
        print(
            f"[INIT WARMUP] collectives done in {elapsed:.2f}s "
            f"across {len(groups)} group(s), "
            f"{len(_WARMUP_DTYPES)} dtype(s), {len(_WARMUP_SIZES_BYTES)} size(s)",
            flush=True,
        )
    return elapsed


def warmup_hybrid_ep(model: Iterable, verbose: bool = True) -> float:
    """Run one synthetic dispatch+combine on each MoE layer to warm DeepEP IPC.

    The first hybrid_ep dispatch in iter 1 costs ~5 s for cuMemImportFromShareableHandle
    (importing peers' symmetric buffer handles). One synthetic dispatch in init
    triggers the same handshake and shifts the cost out of iter 1.
    """
    try:
        from megatron.core.transformer.moe.moe_layer import MoELayer
    except ImportError:
        return 0.0

    if not dist.is_initialized():
        return 0.0

    # Find any MoE layer with hybrid_ep dispatcher
    moe_layers = []
    for module in model if isinstance(model, list) else [model]:
        for m in module.modules():
            if isinstance(m, MoELayer) and getattr(m, "token_dispatcher", None) is not None:
                dispatcher = m.token_dispatcher
                if (
                    "hybrid" in type(dispatcher).__name__.lower()
                    or "ep" in type(dispatcher).__name__.lower()
                ):
                    moe_layers.append(m)
                    break  # one per top-level module is enough
        if moe_layers:
            break

    if not moe_layers:
        if verbose and dist.get_rank() == 0:
            print(
                "[INIT WARMUP] no hybrid_ep MoE layer found; skipping dispatch warmup", flush=True
            )
        return 0.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    moe = moe_layers[0]
    config = moe.config
    hidden = config.hidden_size
    num_experts = config.num_moe_experts
    topk = getattr(config, "moe_router_topk", 1) or 1

    n_tokens = max(num_experts, 64)
    try:
        with torch.no_grad():
            hidden_states = torch.zeros(n_tokens, 1, hidden, dtype=torch.bfloat16, device="cuda")
            probs = torch.zeros(n_tokens, num_experts, dtype=torch.float32, device="cuda")
            probs[:, 0] = 1.0  # all tokens to expert 0

            # Try calling the dispatcher's lower-level dispatch path.
            # Different dispatchers have different signatures; just call the layer.
            _ = moe(hidden_states)
    except Exception as e:
        if verbose and dist.get_rank() == 0:
            print(f"[INIT WARMUP] hybrid_ep dispatch warmup skipped: {e}", flush=True)
        return time.perf_counter() - t0

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if verbose and dist.get_rank() == 0:
        print(f"[INIT WARMUP] hybrid_ep dispatch done in {elapsed:.2f}s", flush=True)
    return elapsed
