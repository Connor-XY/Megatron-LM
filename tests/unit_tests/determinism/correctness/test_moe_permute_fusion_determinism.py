# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact coverage for the fused MoE permute / unpermute kernels.

``--moe-permute-fusion`` routes token permutation and combination through
Transformer Engine's fused kernels: ``moe_utils.permute(..., fused=True)`` and
``unpermute(..., fused=True)`` early-return into ``fused_permute`` /
``fused_unpermute`` *before* the ``torch.are_deterministic_algorithms_enabled()``
branch that otherwise selects the deterministic ``index_add_`` combine. So on
the fused path, determinism rests entirely on the TE kernels' own backward,
which the standard determinism env/config settings do not govern.

This isolates both atomic surfaces of that path — permute's scatter-add
backward and unpermute's scatter-add forward — with a full forward+backward and
asserts the output and input gradient are bit-identical across two runs.

``fused=False`` is the control: it takes the eager deterministic branch and
must always pass. ``fused=True`` is the path under question. A plain A/A repeat
can match by luck when the scheduler happens to replay the same atomic order,
so the meaningful check runs under ``RacingStreams`` scheduling pressure.
"""

import os

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import permute, unpermute
from tests.unit_tests.determinism.utils import (
    RacingStreams,
    assert_bit_exact,
    capture_rng_state,
    restore_rng_state,
)
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.extensions.transformer_engine import fused_permute, fused_unpermute

    HAVE_TE_FUSED = fused_permute is not None and fused_unpermute is not None
except Exception:
    HAVE_TE_FUSED = False

# Collision pressure: many tokens routed into few experts so the backward
# accumulation has many colliding adds, and topk > 2 so those adds are not
# pairwise-commutative (a commutative pair can match regardless of order).
_NUM_TOKENS = 4096
_HIDDEN = 256
_NUM_EXPERTS = 8
_TOPK = 4

_FUSED_PARAMS = [
    False,
    pytest.param(
        True,
        marks=pytest.mark.skipif(
            not HAVE_TE_FUSED, reason="TE fused permute/unpermute kernels not available"
        ),
    ),
]


def _build_routing_map(seed: int) -> torch.Tensor:
    """A dropless routing map with exactly ``_TOPK`` experts per token."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(_NUM_TOKENS, _NUM_EXPERTS, device="cuda", generator=gen)
    topk_idx = logits.topk(_TOPK, dim=-1).indices
    routing_map = torch.zeros(_NUM_TOKENS, _NUM_EXPERTS, dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, topk_idx, True)
    return routing_map


def _fwd_bwd(hidden_base, routing_map, target, fused):
    """permute -> differentiable expert stand-in -> unpermute, then backward.

    Returns the combined output and the input gradient. The gradient flows back
    through both the unpermute forward (scatter-add reduction) and the permute
    backward (scatter-add), so a mismatch in either fused kernel shows up here.
    """
    hidden = hidden_base.clone().detach().requires_grad_(True)
    permuted, _, sorted_indices, _, _ = permute(
        hidden,
        routing_map,
        num_out_tokens=_TOPK * _NUM_TOKENS,
        fused=fused,
        dropless_topk=_TOPK,
    )
    expert_out = permuted * 1.5
    combined = unpermute(
        expert_out,
        sorted_indices,
        restore_shape=hidden.shape,
        routing_map=routing_map,
        fused=fused,
        dropless_topk=_TOPK,
    )
    loss = (combined * target).sum()
    loss.backward()
    return combined.detach().clone(), {"hidden": hidden.grad.detach().clone()}


class TestMoEPermuteFusionDeterminism:

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(1234)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        torch.cuda.empty_cache()

    def _inputs(self):
        routing_map = _build_routing_map(seed=0)
        hidden_base = torch.randn(_NUM_TOKENS, _HIDDEN, device="cuda")
        target = torch.randn(_NUM_TOKENS, _HIDDEN, device="cuda")
        return hidden_base, routing_map, target

    @pytest.mark.internal
    @pytest.mark.parametrize("fused", _FUSED_PARAMS)
    def test_bit_exact_plain(self, fused):
        """A/A repeat on identical inputs. A match here is necessary but not
        sufficient for the fused path — see the racing-streams test."""
        hidden_base, routing_map, target = self._inputs()
        out_a, g_a = _fwd_bwd(hidden_base, routing_map, target, fused)
        torch.cuda.synchronize()
        out_b, g_b = _fwd_bwd(hidden_base, routing_map, target, fused)
        assert_bit_exact(out_a, g_a, out_b, g_b)

    @pytest.mark.internal
    @pytest.mark.skipif(
        os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1") == "1",
        reason=(
            "RacingStreams is a no-op when CUDA_DEVICE_MAX_CONNECTIONS=1 — side "
            "streams serialise through one hardware queue and cannot run "
            "concurrently with the kernel under test. Effective on Blackwell "
            "where the =1 requirement is dropped and the launcher leaves the "
            "multi-queue default."
        ),
    )
    @pytest.mark.parametrize("fused", _FUSED_PARAMS)
    def test_bit_exact_under_racing_streams(self, fused):
        """Perturb scheduling so atomic-order nondeterminism in the fused
        kernels surfaces as a bit-exact failure."""
        hidden_base, routing_map, target = self._inputs()
        state = capture_rng_state()
        with RacingStreams(num_streams=4):
            out_a, g_a = _fwd_bwd(hidden_base, routing_map, target, fused)
        restore_rng_state(state)
        with RacingStreams(num_streams=4):
            out_b, g_b = _fwd_bwd(hidden_base, routing_map, target, fused)
        assert_bit_exact(out_a, g_a, out_b, g_b)
