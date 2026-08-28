# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Kernel hunt: localize the non-deterministic BACKWARD kernel in a
Nemotron-3-Ultra-style hybrid model at small scale under scheduling stress.

Context (the large-scale finding this reproduces): on a real 24-node
Nemotron-3-Ultra deterministic run, the semantic tracer found the first
divergent collective is ``data_parallel.grad_reduce`` and the reduce-scatter
*input* (the local weight-gradient bucket) already differs between two
independent allocations. Local backward compute is topology-independent, so a
diverging local gradient means a NON-DETERMINISTIC LOCAL BACKWARD KERNEL
despite deterministic mode (Class A). Class-A non-determinism reproduces at
small scale under scheduling stress; this test reproduces it and NAMES the op.

Method, per config variant:

* Phase CONTROL  -- plain A/A repeat, no stress. Must be bit-exact (sanity that
  the model itself is deterministic when the scheduler happens to replay the
  same order).
* Phase DETECT   -- A/A under ``RacingStreams`` (and optionally ``CudaSleepJitter``)
  with NO op-trace. This is the ground-truth repro signal: a bit-exact FAILURE
  here proves a Class-A kernel. Op-tracing is deliberately absent because its
  per-op device->host readback can serialize enough to mask an atomic race.
* Phase NAME     -- A/A under the same stress WITH ``op_trace_mode`` +
  ``trace_iteration`` on both runs, then ``compare_trace_paths`` names the first
  divergent ATen op. Extension kernels (TE grouped GEMM / fused attention / TE
  fused permute) bypass the ATen dispatcher, so a divergence born inside one
  surfaces at the first ATen op that consumes its output.

The variants toggle the two MoE surfaces NOT governed by the determinism env:
``moe_permute_fusion`` (TE fused permute/unpermute scatter-add backward) and,
independently, MTP depth (more of the same kernels -> more collision pressure).

Global knobs via env so the cluster run can be retuned without re-copying:
``HUNT_SEQ_LEN`` (default 2048), ``HUNT_MB`` (default 2), ``HUNT_TOPK``
(default 4), ``HUNT_EP`` (default 2), ``HUNT_JITTER`` (default 0),
``HUNT_LAYERS`` repeats of the ``M*E`` block (default 1).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.determinism_op_trace import op_trace_mode
from megatron.core.determinism_trace import trace_iteration
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.configs import nemotron_hybrid_base
from tests.unit_tests.determinism.utils import (
    CudaSleepJitter,
    RacingStreams,
    assert_bit_exact,
    capture_rng_state,
    collect_grads,
    reset_quantizer_state,
    restore_rng_state,
    zero_grads,
)
from tests.unit_tests.test_utilities import Utils
from tools.determinism.compare_traces import compare_trace_paths

_SEQ_LEN = int(os.environ.get("HUNT_SEQ_LEN", "2048"))
_MICRO_BATCH = int(os.environ.get("HUNT_MB", "2"))
_TOPK = int(os.environ.get("HUNT_TOPK", "4"))
_EP = int(os.environ.get("HUNT_EP", "2"))
_JITTER = os.environ.get("HUNT_JITTER", "0") == "1"
_BLOCK_REPS = int(os.environ.get("HUNT_LAYERS", "1"))
_STREAMS = int(os.environ.get("HUNT_STREAMS", "4"))
# Stressor duration is load-bearing: RacingStreams enqueues all its noise up
# front, so the noise must OUTLAST the model's fwd+bwd for the BACKWARD to run
# contended. The default 200 iters (~12ms) overlaps only a microsecond-scale
# kernel (as in the permute-fusion unit test); a full-model fwd+bwd is far
# longer, so a short stressor leaves the backward uncontended and hides atomic
# non-determinism. Crank this so the noise spans the backward pass too.
_NOISE_ITERS = int(os.environ.get("HUNT_NOISE_ITERS", "6000"))
# Model width knobs. The default proxy (hidden=256, ffn=1024) yields tiny
# grouped-GEMM/wgrad shapes that map to single-tile deterministic kernels; the
# real recipe's wide layers are what would select split-K / atomic-accumulation
# GEMM backends. Set these to the wide dims to probe that surface.
_HIDDEN = int(os.environ.get("HUNT_HIDDEN", "0"))
_FFN = int(os.environ.get("HUNT_FFN", "0"))
_EXPERTS = int(os.environ.get("HUNT_EXPERTS", "0"))
_VOCAB_SIZE = 256


def _inputs() -> dict:
    return {
        "input_ids": torch.randint(
            0, _VOCAB_SIZE, (_MICRO_BATCH, _SEQ_LEN), device="cuda", dtype=torch.long
        ),
        "position_ids": (
            torch.arange(_SEQ_LEN, device="cuda", dtype=torch.long)
            .unsqueeze(0)
            .repeat(_MICRO_BATCH, 1)
        ),
        "attention_mask": torch.ones(
            _MICRO_BATCH, 1, _SEQ_LEN, _SEQ_LEN, dtype=torch.bool, device="cuda"
        ),
    }


# ``M*E`` = Mamba + attention + MoE, the three Nemotron-3-Ultra layer types.
_MAIN_BLOCK = "M*E"


def _variant_config(*, permute_fusion: bool, with_mtp: bool) -> tuple[dict, str]:
    """Return (config-overrides, hybrid_layer_pattern) for one variant."""
    main = _MAIN_BLOCK * _BLOCK_REPS
    cfg = dict(nemotron_hybrid_base())
    cfg["num_layers"] = len(main)
    # Alltoall dispatcher + higher top-k => more colliding scatter-adds in the
    # MoE permute/unpermute backward (topk>2 makes the adds non-commutative, so
    # a lucky same-order replay cannot mask a real divergence).
    cfg["moe_token_dispatcher_type"] = "alltoall"
    cfg["moe_router_topk"] = _TOPK
    if _HIDDEN:
        cfg["hidden_size"] = _HIDDEN
    if _FFN:
        cfg["ffn_hidden_size"] = _FFN
    if _EXPERTS:
        cfg["num_moe_experts"] = _EXPERTS
    if permute_fusion:
        cfg["moe_permute_fusion"] = True
    if _EP > 1:
        cfg["expert_model_parallel_size"] = _EP

    pattern = main
    if with_mtp:
        cfg["mtp_num_layers"] = 1
        pattern = f"{main}/{main}"  # one MTP depth mirroring the main block
    return cfg, pattern


def _build(cfg_overrides: dict, layer_pattern: str) -> HybridModel:
    cfg = TransformerConfig(**(nemotron_hybrid_base() | cfg_overrides))
    return HybridModel(
        config=cfg,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_SEQ_LEN,
        hybrid_layer_pattern=layer_pattern,
        pre_process=True,
        post_process=True,
    ).cuda()


def _fwd_bwd(model, inputs) -> tuple:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs)
    tensor = out[0] if isinstance(out, tuple) else out
    tensor.float().pow(2).mean().backward()
    return tensor.detach().clone(), collect_grads([model])


@contextlib.contextmanager
def _stress(model, enabled: bool):
    """Scheduling stressor: RacingStreams (SM contention through backward) plus
    optional CudaSleepJitter. RacingStreams is entered OUTSIDE any op-trace so
    its side-stream GEMMs are not fingerprinted, yet still contend on the GPU
    during the traced model backward."""
    if not enabled:
        yield
        return
    with RacingStreams(num_streams=_STREAMS, num_iters=_NOISE_ITERS):
        if _JITTER:
            with CudaSleepJitter(model):
                yield
        else:
            yield


def _reset_moe_state(model) -> None:
    """Zero the persistent MoE load-balancing counters and clear the aux-loss
    tracker so two consecutive traced forwards are not flagged divergent purely
    because a per-forward accumulator advanced. These buffers feed logging and
    the optimizer-step bias update, never the forward compute, so zeroing them
    does not change outputs or gradients — it only de-noises the op-trace."""
    try:
        from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker

        clear_aux_losses_tracker()
    except Exception:
        pass
    for sub in model.modules():
        for buf_name in ("local_tokens_per_expert", "global_tokens_per_expert"):
            buf = getattr(sub, buf_name, None)
            if isinstance(buf, torch.Tensor):
                buf.zero_()


def _run(model, inputs, *, stress: bool, trace_dir: str | None) -> tuple:
    zero_grads(model)
    reset_quantizer_state([model])
    _reset_moe_state(model)
    with _stress(model, stress):
        if trace_dir is None:
            return _fwd_bwd(model, inputs)
        with op_trace_mode(True):
            with trace_iteration(trace_dir, 1):
                return _fwd_bwd(model, inputs)


def _barrier():
    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])


def _bit_exact(out_a, g_a, out_b, g_b) -> tuple[bool, str]:
    try:
        assert_bit_exact(out_a, g_a, out_b, g_b)
        return True, ""
    except AssertionError as err:
        return False, str(err)


def _first_divergences(report: dict, limit: int = 6) -> list[str]:
    lines = []
    for d in report.get("divergences", [])[:limit]:
        loc = d.get("event", d.get("file", "?"))
        reason = d.get("reason", "?")
        detail = ""
        if reason == "event_values":
            # Surface the op name and the divergent output signatures.
            le = d.get("left_event", {})
            re = d.get("right_event", {})
            lp = le.get("payload", {})
            rp = re.get("payload", {})
            lo = lp.get("outputs")
            ro = rp.get("outputs")
            detail = (
                f" | name={le.get('name')} phase={lp.get('phase')} left_out={lo} right_out={ro}"
            )
        lines.append(f"[{reason}] {loc}{detail}")
    return lines


VARIANTS = [
    pytest.param(dict(permute_fusion=False, with_mtp=False), id="base"),
    pytest.param(dict(permute_fusion=True, with_mtp=False), id="permfusion"),
    pytest.param(dict(permute_fusion=False, with_mtp=True), id="mtp"),
    pytest.param(dict(permute_fusion=True, with_mtp=True), id="mtp-permfusion"),
]


class TestNemotronUltraKernelHunt:

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        torch.use_deterministic_algorithms(True, warn_only=True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        torch.cuda.empty_cache()

    @pytest.mark.internal
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_kernel_hunt(self, variant, request):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        vid = request.node.callspec.id
        tag = f"KERNEL-HUNT[{vid} rank{rank}]"

        try:
            cfg_overrides, pattern = _variant_config(**variant)
        except Exception as exc:  # pragma: no cover - config guard
            print(f"{tag} CONFIG-ERROR {type(exc).__name__}: {exc}", flush=True)
            raise

        Utils.destroy_model_parallel()
        init_kwargs = {"expert_model_parallel_size": _EP} if _EP > 1 else {}
        Utils.initialize_model_parallel(**init_kwargs)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(123)

        try:
            model = _build(cfg_overrides, pattern)
        except Exception as exc:
            print(f"{tag} BUILD-ERROR {type(exc).__name__}: {exc}", flush=True)
            pytest.skip(f"{vid}: build failed: {exc}")

        inputs = _inputs()
        print(
            f"{tag} config seq={_SEQ_LEN} mb={_MICRO_BATCH} topk={_TOPK} ep={_EP} "
            f"hidden={cfg_overrides.get('hidden_size', '(base)')} "
            f"ffn={cfg_overrides.get('ffn_hidden_size', '(base)')} "
            f"experts={cfg_overrides.get('num_moe_experts', '(base)')} "
            f"streams={_STREAMS} noise_iters={_NOISE_ITERS} jitter={_JITTER} "
            f"pattern={pattern} "
            f"permute_fusion={cfg_overrides.get('moe_permute_fusion', False)} "
            f"mtp={cfg_overrides.get('mtp_num_layers', 0)}",
            flush=True,
        )

        tmp_root = tempfile.mkdtemp(prefix=f"hunt_{vid}_")
        try:
            # -------- Phase CONTROL: no stress, must be bit-exact. --------
            state = capture_rng_state()
            out_a, g_a = _run(model, inputs, stress=False, trace_dir=None)
            _barrier()
            restore_rng_state(state)
            out_b, g_b = _run(model, inputs, stress=False, trace_dir=None)
            control_ok, control_err = _bit_exact(out_a, g_a, out_b, g_b)
            print(f"{tag} CONTROL bit_exact={control_ok} {control_err}", flush=True)

            # -------- Phase DETECT: stress, no trace (ground-truth repro). --------
            state = capture_rng_state()
            out_a, g_a = _run(model, inputs, stress=True, trace_dir=None)
            _barrier()
            restore_rng_state(state)
            out_b, g_b = _run(model, inputs, stress=True, trace_dir=None)
            detect_ok, detect_err = _bit_exact(out_a, g_a, out_b, g_b)
            print(
                f"{tag} DETECT bit_exact={detect_ok} "
                f"{'(determinism holds under stress)' if detect_ok else 'DIVERGED: ' + detect_err}",
                flush=True,
            )

            # -------- Phase NAME: stress + op-trace, compare traces. --------
            dir_a = os.path.join(tmp_root, "runA")
            dir_b = os.path.join(tmp_root, "runB")
            state = capture_rng_state()
            out_a, g_a = _run(model, inputs, stress=True, trace_dir=dir_a)
            _barrier()
            restore_rng_state(state)
            out_b, g_b = _run(model, inputs, stress=True, trace_dir=dir_b)
            name_ok, name_err = _bit_exact(out_a, g_a, out_b, g_b)
            report = compare_trace_paths(dir_a, dir_b, max_details=20)
            print(
                f"{tag} NAME traced_bit_exact={name_ok} trace_equal={report['equal']} "
                f"events_compared={report['events_compared']} "
                f"divergences={report['divergence_count']}",
                flush=True,
            )
            for line in _first_divergences(report):
                print(f"{tag} FIRST-DIVERGENCE {line}", flush=True)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        # The model must be deterministic when the scheduler replays the same
        # order; only the stress phases are diagnostic (printed, not asserted).
        assert control_ok, f"{vid}: CONTROL (no-stress) A/A diverged: {control_err}"
