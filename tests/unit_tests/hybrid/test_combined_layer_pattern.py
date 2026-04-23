# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the bracket-notation combined-layer pattern parser.

Grammar is set-based: order inside a bracket is ignored, at most one mamba
(M/G), one attention (*/D), and one MLP (-/E); each component may appear at
most once.
"""

import pytest

from megatron.core.models.hybrid.combined_layer_pattern import (
    CombinedLayerSpec,
    is_bracket_pattern,
    parse_bracket_pattern,
    parse_bracket_segment,
)


@pytest.mark.internal
class TestIsBracketPattern:

    def test_none_is_not_bracket(self):
        assert not is_bracket_pattern(None)

    def test_empty_is_not_bracket(self):
        assert not is_bracket_pattern("")

    def test_legacy_patterns_not_bracket(self):
        for p in ["M*M-", "MMMM", "M-M*-|M-M-", "M*M*/MM/MM"]:
            assert not is_bracket_pattern(p), p

    def test_bracket_patterns_detected(self):
        for p in ["[M-]", "[M*-]", "[M-][M*-]", "[M-][M*-]|[M-][ME]"]:
            assert is_bracket_pattern(p), p


@pytest.mark.internal
class TestParseBracketSegment:

    def test_empty_returns_empty_list(self):
        assert parse_bracket_segment("") == []

    def test_mamba_mlp(self):
        [g] = parse_bracket_segment("[M-]")
        assert g.mamba_type == "mamba"
        assert g.attention_type == "none"
        assert g.mlp_type == "mlp"
        assert g.kwargs == {"mamba_type": "mamba", "attention_type": "none", "mlp_type": "mlp"}
        assert g.canonical == "[M-]"

    def test_mamba_self_attn_mlp(self):
        [g] = parse_bracket_segment("[M*-]")
        assert g.attention_type == "attention"
        assert g.canonical == "[M*-]"

    def test_gdn_mlp(self):
        """Pure GDN + MLP: GDN is the attention-slot choice, no Mamba."""
        [g] = parse_bracket_segment("[G-]")
        assert g.mamba_type == "none"
        assert g.attention_type == "gated_delta_net"
        assert g.mlp_type == "mlp"

    def test_mamba_plus_gdn_stacked(self):
        """``[MG-]`` stacks Mamba + GDN (GDN in the attention slot)."""
        [g] = parse_bracket_segment("[MG-]")
        assert g.mamba_type == "mamba"
        assert g.attention_type == "gated_delta_net"

    def test_mamba_moe(self):
        [g] = parse_bracket_segment("[ME]")
        assert g.mamba_type == "mamba"
        assert g.mlp_type == "moe"
        assert g.is_moe
        assert g.canonical == "[ME]"

    def test_pure_attention_moe(self):
        """No-mamba group: pure Attn + MoE layer (GPT-style inside hybrid)."""
        [g] = parse_bracket_segment("[*E]")
        assert g.mamba_type == "none"
        assert g.attention_type == "attention"
        assert g.mlp_type == "moe"

    def test_mla_accepted(self):
        [g] = parse_bracket_segment("[MD-]")
        assert g.attention_type == "mla"

    def test_order_inside_bracket_ignored(self):
        a = parse_bracket_segment("[M*-]")
        b = parse_bracket_segment("[-*M]")
        c = parse_bracket_segment("[*-M]")
        assert a[0].canonical == b[0].canonical == c[0].canonical == "[M*-]"

    def test_multiple_groups(self):
        specs = parse_bracket_segment("[M-][M*-][ME]")
        assert len(specs) == 3
        assert [s.canonical for s in specs] == ["[M-]", "[M*-]", "[ME]"]

    # ----- Negative cases -----

    def test_empty_bracket(self):
        with pytest.raises(ValueError, match="Empty bracket"):
            parse_bracket_segment("[]")

    def test_unclosed_bracket(self):
        with pytest.raises(ValueError, match="outside bracket"):
            parse_bracket_segment("[M-")

    def test_duplicate_symbol(self):
        with pytest.raises(ValueError, match="repeated"):
            parse_bracket_segment("[MM-]")

    def test_two_attention_kinds(self):
        # SelfAttention + MLA not allowed.
        with pytest.raises(ValueError, match="multiple attention-kind"):
            parse_bracket_segment("[M*D-]")

    def test_self_attn_and_gdn_both_in_attn_slot(self):
        # * and G both occupy the attention slot -- rejected.
        with pytest.raises(ValueError, match="multiple attention-kind"):
            parse_bracket_segment("[M*G-]")

    def test_mla_and_gdn_both_in_attn_slot(self):
        with pytest.raises(ValueError, match="multiple attention-kind"):
            parse_bracket_segment("[MDG-]")

    def test_two_mlp_kinds(self):
        with pytest.raises(ValueError, match="both MLP"):
            parse_bracket_segment("[M-E]")

    def test_invalid_symbol(self):
        with pytest.raises(ValueError, match="not a valid"):
            parse_bracket_segment("[MX-]")

    def test_text_outside_brackets(self):
        with pytest.raises(ValueError, match="outside bracket"):
            parse_bracket_segment("M[*-]")


@pytest.mark.internal
class TestParseBracketPattern:

    def test_empty(self):
        assert parse_bracket_pattern("") == [[]]

    def test_single_segment(self):
        segs = parse_bracket_pattern("[M-][M*-]")
        assert len(segs) == 1
        assert [g.canonical for g in segs[0]] == ["[M-]", "[M*-]"]

    def test_two_pp_segments(self):
        segs = parse_bracket_pattern("[M-][M*-]|[M-][ME]")
        assert len(segs) == 2
        assert [g.canonical for g in segs[0]] == ["[M-]", "[M*-]"]
        assert [g.canonical for g in segs[1]] == ["[M-]", "[ME]"]

    def test_returns_combined_layer_spec_instances(self):
        segs = parse_bracket_pattern("[M-]")
        assert all(isinstance(g, CombinedLayerSpec) for g in segs[0])

    def test_canonical_roundtrip(self):
        """``canonical`` renders in fixed order regardless of input ordering."""
        specs = parse_bracket_segment("[-M*]")
        assert specs[0].canonical == "[M*-]"
        # Re-parsing the canonical form yields the same spec.
        specs2 = parse_bracket_segment(specs[0].canonical)
        assert specs2[0].kwargs == specs[0].kwargs
