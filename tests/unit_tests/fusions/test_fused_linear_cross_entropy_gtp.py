# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import types

import torch

import megatron.core.fusions.fused_linear_cross_entropy as fused_linear_cross_entropy


def test_distributed_weight_uses_materialize_and_finalize_protocol(monkeypatch):
    calls = []
    hidden = torch.randn(2, 3, requires_grad=True)
    sharded_weight = torch.nn.Parameter(torch.randn(2, 3))
    materialized_weight = torch.randn(4, 3)
    labels = torch.tensor([0, 1])

    sharded_weight.is_distributed_weight = True

    def materialize_group_for_forward(self):
        calls.append("materialize_forward")
        return materialized_weight

    def materialize_group_for_backward(self, nvtx_label=None):
        calls.append(("materialize_backward", nvtx_label))
        return materialized_weight

    def finalize_group_grads(self, grad, nvtx_label=None):
        calls.append(("finalize", nvtx_label, tuple(grad.shape)))
        return grad[: self.shape[0]].contiguous()

    sharded_weight.materialize_group_for_forward = types.MethodType(
        materialize_group_for_forward, sharded_weight
    )
    sharded_weight.materialize_group_for_backward = types.MethodType(
        materialize_group_for_backward, sharded_weight
    )
    sharded_weight.finalize_group_grads = types.MethodType(
        finalize_group_grads, sharded_weight
    )

    class FakePlatform:
        @staticmethod
        def forward_func(hidden, weight, labels, *args):
            del args
            assert weight is materialized_weight
            num_tokens = labels.numel()
            stats = torch.zeros(num_tokens)
            return torch.ones(num_tokens), stats, stats, torch.tensor(num_tokens), 0, 1, hidden

        @staticmethod
        def backward_func(dloss, global_hidden, weight, labels, *args):
            del dloss, labels, args
            assert weight is materialized_weight
            return torch.full_like(global_hidden, 2.0), torch.full_like(weight, 3.0)

    monkeypatch.setattr(fused_linear_cross_entropy, "_get_platform", lambda: FakePlatform())
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _: contextlib.nullcontext())

    loss = fused_linear_cross_entropy.linear_cross_entropy(
        hidden, sharded_weight, labels, reduction="none"
    )
    loss.sum().backward()

    assert calls == [
        "materialize_forward",
        ("materialize_backward", "linear_cross_entropy"),
        ("finalize", "linear_cross_entropy", (4, 3)),
    ]
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 2.0))
    torch.testing.assert_close(sharded_weight.grad, torch.full_like(sharded_weight, 3.0))
