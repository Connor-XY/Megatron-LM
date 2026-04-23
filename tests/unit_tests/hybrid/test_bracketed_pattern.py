# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the bracketed combined-layer pattern parser.

Exercises the grammar added to support Option 1 of the hybrid 1F1B overlap
design: ``[M-][M*-][ME]``-style patterns that explicitly group symbols into
combined Mamba + optional-attention + MLP decoder layers.
"""

import pytest

from megatron.core.models.hybrid.hybrid_layer_allocation import (
    ATTENTION_TYPE_DS_ATTENTION,
    ATTENTION_TYPE_GATED_DELTA_NET,
    ATTENTION_TYPE_NONE,
    ATTENTION_TYPE_SELF_ATTENTION,
    MLP_KIND_DENSE,
    MLP_KIND_MOE,
    CombinedLayerGroup,
    is_bracketed_pattern,
    parse_bracketed_pattern,
)


@pytest.mark.internal
class TestIsBracketedPattern:

    def test_none_is_not_bracketed(self):
        assert not is_bracketed_pattern(None)

    def test_empty_is_not_bracketed(self):
        assert not is_bracketed_pattern("")

    def test_legacy_patterns_not_bracketed(self):
        for p in ["M*M-", "MMMM", "M-M*-|M-M-", "M*M*/MM/MM"]:
            assert not is_bracketed_pattern(p), p

    def test_bracketed_patterns_detected(self):
        for p in ["[M-]", "[M*-]", "[M-][M*-]", "[M-][M*-]|[M-][ME]"]:
            assert is_bracketed_pattern(p), p


@pytest.mark.internal
class TestParseBracketedPattern:

    def test_empty(self):
        assert parse_bracketed_pattern("") == []
        assert parse_bracketed_pattern(None) == []

    def test_single_dense_mamba(self):
        groups = parse_bracketed_pattern("[M-]")
        assert len(groups) == 1
        g = groups[0]
        assert g.attention_type == ATTENTION_TYPE_NONE
        assert g.mlp_kind == MLP_KIND_DENSE
        # M gets standalone layer_number 1; '-' gets 2 in the legacy flat pattern.
        assert g.submodule_standalone_indices == {"mamba": 1, "mlp": 2}
        assert "attention" not in g.submodule_standalone_indices

    def test_single_self_attn(self):
        groups = parse_bracketed_pattern("[M*-]")
        assert len(groups) == 1
        g = groups[0]
        assert g.attention_type == ATTENTION_TYPE_SELF_ATTENTION
        assert g.mlp_kind == MLP_KIND_DENSE
        assert g.submodule_standalone_indices == {"mamba": 1, "attention": 2, "mlp": 3}

    def test_single_gdn(self):
        groups = parse_bracketed_pattern("[MG-]")
        assert groups[0].attention_type == ATTENTION_TYPE_GATED_DELTA_NET
        assert groups[0].submodule_standalone_indices == {
            "mamba": 1,
            "attention": 2,
            "mlp": 3,
        }

    def test_single_moe(self):
        groups = parse_bracketed_pattern("[ME]")
        assert groups[0].attention_type == ATTENTION_TYPE_NONE
        assert groups[0].mlp_kind == MLP_KIND_MOE

    def test_multiple_groups_indices_accumulate(self):
        groups = parse_bracketed_pattern("[M-][M*-][ME]")
        assert len(groups) == 3

        # Group 1: M(1), -(2)
        assert groups[0].attention_type == ATTENTION_TYPE_NONE
        assert groups[0].mlp_kind == MLP_KIND_DENSE
        assert groups[0].submodule_standalone_indices == {"mamba": 1, "mlp": 2}

        # Group 2: M(3), *(4), -(5)
        assert groups[1].attention_type == ATTENTION_TYPE_SELF_ATTENTION
        assert groups[1].mlp_kind == MLP_KIND_DENSE
        assert groups[1].submodule_standalone_indices == {
            "mamba": 3,
            "attention": 4,
            "mlp": 5,
        }

        # Group 3: M(6), E(7)
        assert groups[2].attention_type == ATTENTION_TYPE_NONE
        assert groups[2].mlp_kind == MLP_KIND_MOE
        assert groups[2].submodule_standalone_indices == {"mamba": 6, "mlp": 7}

    def test_ds_attention_allowed_at_parse_time(self):
        """DSA parses as attention_type='ds_attention'. HybridStack rejects at build-time."""
        groups = parse_bracketed_pattern("[MD-]")
        assert groups[0].attention_type == ATTENTION_TYPE_DS_ATTENTION

    def test_raw_symbols_recorded(self):
        groups = parse_bracketed_pattern("[M*E]")
        assert groups[0].raw_symbols == ["M", "*", "E"]

    # ----- Negative cases -----

    def test_unclosed_bracket(self):
        with pytest.raises(ValueError, match="Unclosed bracket"):
            parse_bracketed_pattern("[M-")

    def test_empty_bracket(self):
        with pytest.raises(ValueError, match="Empty bracket"):
            parse_bracketed_pattern("[]")

    def test_two_mambas(self):
        with pytest.raises(ValueError, match="exactly one Mamba"):
            parse_bracketed_pattern("[MM-]")

    def test_two_attentions(self):
        with pytest.raises(ValueError, match="at most one attention"):
            parse_bracketed_pattern("[M*G-]")

    def test_two_mlps(self):
        with pytest.raises(ValueError, match="exactly one MLP"):
            parse_bracketed_pattern("[M--]")

    def test_no_mlp(self):
        with pytest.raises(ValueError, match="exactly one MLP"):
            parse_bracketed_pattern("[M]")

    def test_no_mamba(self):
        with pytest.raises(ValueError, match="exactly one Mamba"):
            parse_bracketed_pattern("[*-]")

    def test_pipe_inside_bracket(self):
        with pytest.raises(ValueError, match="Pipe '\\|'"):
            parse_bracketed_pattern("[M|*-]")

    def test_slash_inside_bracket(self):
        with pytest.raises(ValueError, match="MTP separator"):
            parse_bracketed_pattern("[M/-]")

    def test_reversed_order(self):
        """MLP must be last; ``[-M]`` flips the canonical order."""
        with pytest.raises(ValueError, match="Mamba first"):
            parse_bracketed_pattern("[-M]")

    def test_attn_before_mamba(self):
        with pytest.raises(ValueError, match="Mamba first"):
            parse_bracketed_pattern("[*M-]")

    def test_mlp_not_last(self):
        with pytest.raises(ValueError, match="MLP symbol last"):
            parse_bracketed_pattern("[M-*]")

    def test_invalid_symbol_in_bracket(self):
        with pytest.raises(ValueError, match="not a valid layer symbol"):
            parse_bracketed_pattern("[MX-]")

    def test_text_outside_bracket(self):
        with pytest.raises(ValueError, match="unexpected"):
            parse_bracketed_pattern("M[*-]")

    def test_nested_brackets(self):
        with pytest.raises(ValueError, match="Nested brackets|not a valid"):
            parse_bracketed_pattern("[[M-]]")

    def test_returns_combined_layer_group_instances(self):
        groups = parse_bracketed_pattern("[M-]")
        assert all(isinstance(g, CombinedLayerGroup) for g in groups)
