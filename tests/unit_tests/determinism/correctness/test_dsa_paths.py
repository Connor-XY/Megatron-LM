# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact coverage for DeepSeek Sparse Attention mask construction.

The DSA implementation builds sparse masks with ``scatter_`` in three places:
indexer-loss forward, its recomputed manual backward, and unfused attention.
These tests exercise all three with real downstream reductions and gradients.
"""

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa import (
    FusedDSAIndexerLoss,
    unfused_dsa_fn,
)
from tests.unit_tests.determinism.utils import assert_bit_exact
from tests.unit_tests.test_utilities import Utils

_SEQ_LEN = 32
_BATCH_SIZE = 2


class TestDSAPathDeterminism:

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(1234)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        torch.cuda.empty_cache()

    @pytest.mark.internal
    @pytest.mark.parametrize("sparse_loss", [False, True])
    def test_indexer_loss_forward_and_backward_are_bit_exact(self, sparse_loss):
        """Cover the mask scatter in both indexer-loss forward and manual backward."""
        index_heads = 4
        index_head_dim = 32
        attention_heads = 4
        attention_head_dim = 32
        topk = 8

        q_base = torch.randn(
            _SEQ_LEN, _BATCH_SIZE, index_heads, index_head_dim, device="cuda", dtype=torch.bfloat16
        )
        weights_base = torch.randn(
            _SEQ_LEN, _BATCH_SIZE, index_heads, device="cuda", dtype=torch.bfloat16
        )
        k_base = torch.randn(
            _SEQ_LEN, _BATCH_SIZE, index_head_dim, device="cuda", dtype=torch.bfloat16
        )
        query = torch.randn(
            _SEQ_LEN,
            _BATCH_SIZE,
            attention_heads,
            attention_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.randn_like(query)
        mask = torch.triu(
            torch.full((_SEQ_LEN, _SEQ_LEN), float("-inf"), device="cuda"), diagonal=1
        )

        def run_once():
            q = q_base.detach().clone().requires_grad_(True)
            weights = weights_base.detach().clone().requires_grad_(True)
            k = k_base.detach().clone().requires_grad_(True)
            indices, loss = FusedDSAIndexerLoss.apply(
                q,
                weights,
                k,
                query,
                key,
                attention_head_dim**-0.5,
                topk,
                1.0,
                mask,
                sparse_loss,
                self.pg_collection,
            )
            loss.backward()
            grads = {
                "q": q.grad.detach().clone(),
                "weights": weights.grad.detach().clone(),
                "k": k.grad.detach().clone(),
            }
            return indices.detach().clone(), loss.detach().clone(), grads

        indices_a, loss_a, grads_a = run_once()
        torch.cuda.synchronize()
        indices_b, loss_b, grads_b = run_once()
        if not torch.equal(indices_a, indices_b):
            raise AssertionError("DSA indexer top-k indices differ between deterministic runs")
        assert_bit_exact(loss_a, grads_a, loss_b, grads_b)

    @pytest.mark.internal
    def test_unfused_sparse_attention_is_bit_exact(self):
        """Cover the attention mask scatter and its downstream input gradients."""
        num_heads = 4
        head_dim = 32
        query_base = torch.randn(
            _SEQ_LEN, _BATCH_SIZE, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        key_base = torch.randn_like(query_base)
        value_base = torch.randn_like(query_base)
        # Unique indices along the scatter dimension. The causal mask removes
        # future positions for early query rows while always retaining key 0.
        topk_indices = (
            torch.arange(8, device="cuda", dtype=torch.long)
            .view(1, 1, -1)
            .expand(_BATCH_SIZE, _SEQ_LEN, -1)
        )

        def run_once():
            query = query_base.detach().clone().requires_grad_(True)
            key = key_base.detach().clone().requires_grad_(True)
            value = value_base.detach().clone().requires_grad_(True)
            output = unfused_dsa_fn(query, key, value, topk_indices, softmax_scale=head_dim**-0.5)
            output.float().square().mean().backward()
            grads = {
                "query": query.grad.detach().clone(),
                "key": key.grad.detach().clone(),
                "value": value.grad.detach().clone(),
            }
            return output.detach().clone(), grads

        output_a, grads_a = run_once()
        torch.cuda.synchronize()
        output_b, grads_b = run_once()
        assert_bit_exact(output_a, grads_a, output_b, grads_b)
