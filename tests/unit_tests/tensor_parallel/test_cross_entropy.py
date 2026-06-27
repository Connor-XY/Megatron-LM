# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.models.common.language_module import language_module as language_module_module
from megatron.core.tensor_parallel import cross_entropy as cross_entropy_module
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from tests.unit_tests.test_utilities import Utils


class _FakeTPGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


def test_vocab_parallel_cross_entropy_uses_explicit_tp_group(monkeypatch):
    tp_group = _FakeTPGroup()
    all_reduce_groups = []

    def fake_all_reduce(tensor, op=None, group=None):
        all_reduce_groups.append(group)
        return tensor

    def fail_parallel_state_call(*args, **kwargs):
        raise AssertionError("explicit tp_group should avoid parallel_state")

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        cross_entropy_module, "get_tensor_model_parallel_group", fail_parallel_state_call
    )

    vocab_parallel_logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]])
    target = torch.tensor([2, 0])
    expected_output = torch.nn.functional.cross_entropy(
        vocab_parallel_logits.clone(), target, reduction="none"
    )

    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=tp_group)

    torch.testing.assert_close(output, expected_output)
    assert all_reduce_groups == [tp_group, tp_group, tp_group]


def test_language_module_unfused_loss_passes_tp_group(monkeypatch):
    tp_group = _FakeTPGroup()
    captured = {}

    def fake_vocab_parallel_cross_entropy(logits, labels, label_smoothing=0.0, tp_group=None):
        captured["logits"] = logits
        captured["labels"] = labels
        captured["label_smoothing"] = label_smoothing
        captured["tp_group"] = tp_group
        return torch.zeros_like(labels, dtype=logits.dtype)

    monkeypatch.setattr(
        language_module_module.tensor_parallel,
        "vocab_parallel_cross_entropy",
        fake_vocab_parallel_cross_entropy,
    )

    module = SimpleNamespace(
        config=SimpleNamespace(cross_entropy_loss_fusion=False), tp_group=tp_group
    )
    labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
    logits = torch.randn(3, 2, 4)

    loss = language_module_module.LanguageModule.compute_language_model_loss(
        module, labels=labels, logits=logits
    )

    assert captured["logits"] is logits
    assert captured["tp_group"] is tp_group
    assert captured["label_smoothing"] == 0.0
    torch.testing.assert_close(captured["labels"], labels.transpose(0, 1).contiguous())
    assert loss.shape == labels.shape


def test_vocab_parallel_cross_entropy():
    Utils.initialize_model_parallel(4, 2)
    vocab_parallel_logits = torch.range(0, 7).repeat(16, 4).cuda()
    target = torch.arange(0, 32, 2).cuda()
    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target)
    expected_output = torch.tensor(
        [
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
        ]
    ).cuda()
    assert torch.equal(torch.round(expected_output), torch.round(output))
    Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("vocab_size", [128, 257])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_deterministic_vocab_parallel_cross_entropy_matches_index_put(
    monkeypatch, dtype, label_smoothing, vocab_size, batch_size
):
    tp_group = _FakeTPGroup()
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None, group=None: tensor)
    torch.manual_seed(1234)
    candidate_logits = torch.randn(
        16, batch_size, vocab_size, device="cuda", dtype=dtype, requires_grad=True
    )
    reference_logits = candidate_logits.detach().clone().requires_grad_(True)
    target = torch.randint(vocab_size, (16, batch_size), device="cuda")

    previous_deterministic_mode = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        candidate_loss = vocab_parallel_cross_entropy(
            candidate_logits, target, label_smoothing=label_smoothing, tp_group=tp_group
        )
        loss_gradient = torch.randn(
            batch_size, 16, device="cuda", dtype=candidate_loss.dtype
        ).transpose(0, 1)
        if batch_size > 1:
            assert not loss_gradient.is_contiguous()
        candidate_grad = torch.autograd.grad(
            candidate_loss, candidate_logits, grad_outputs=loss_gradient
        )[0]

        torch.use_deterministic_algorithms(False)
        reference_loss = vocab_parallel_cross_entropy(
            reference_logits, target, label_smoothing=label_smoothing, tp_group=tp_group
        )
        reference_grad = torch.autograd.grad(
            reference_loss, reference_logits, grad_outputs=loss_gradient
        )[0]
    finally:
        torch.use_deterministic_algorithms(previous_deterministic_mode)

    assert torch.equal(candidate_loss, reference_loss)
    assert torch.equal(candidate_grad, reference_grad)
