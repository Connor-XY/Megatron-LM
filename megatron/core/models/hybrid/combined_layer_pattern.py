# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Bracket-notation pattern parser for :class:`CombinedHybridLayer`.

The legacy single-character pattern (e.g. ``"M*M-"``) assigns one decoder
module per character. The *combined* hybrid layer bundles multiple functional
blocks into one module, so bracketed patterns introduce explicit grouping:
each ``[...]`` token corresponds to one entry in ``decoder.layers``.

Grammar
-------
::

    pattern     := segment ('|' segment)*
    segment     := group+
    group       := '[' component+ ']'
    component   := 'M' | 'G' | '*' | 'D' | '-' | 'E'

Composition rules per bracket group (``_validate_group``):

* at most one sequence mixer (``M`` or ``G``)
* at most one attention kind (``*`` or ``D``)
* at most one MLP kind (``-`` or ``E``)
* at least one component overall
* each component may appear at most once; **ordering inside the bracket is
  ignored** -- only the *set* of components matters

Examples
--------
* ``[M-]``          -> one combined layer: Mamba + MLP
* ``[M*-][M-][ME]`` -> Mamba+Attn+MLP, Mamba+MLP, Mamba+MoE
* ``[GE]``          -> GatedDeltaNet + MoE (GDN replaces Mamba in the mixer)
* ``[G*E]``         -> GatedDeltaNet + Attention + MoE
* ``[*E][*E]``      -> pure Attention+MoE layers (GPT-style, no mamba mixer)
* ``[M-][M*-]|[M-][ME]`` -> two PP stages with distinct combined layouts

The ``|`` separator marks pipeline-stage boundaries (same semantics as the
legacy grammar). ``/`` (MTP) is parsed upstream, before this module is called.

:func:`is_bracket_pattern` lets callers detect bracket grammar and route to
the combined-layer code path; otherwise the flat legacy grammar applies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

# Role-to-selector lookups shared with the CombinedHybridLayer constructor.
_MAMBA_SYMBOLS = {Symbols.MAMBA, Symbols.GDN}
_ATTN_SYMBOLS = {Symbols.ATTENTION, Symbols.DS_ATTENTION}
_MLP_SYMBOLS = {Symbols.MLP, Symbols.MOE}

_MAMBA_TAG = {Symbols.MAMBA: "mamba", Symbols.GDN: "gdn"}
_ATTN_TAG = {Symbols.ATTENTION: "attention", Symbols.DS_ATTENTION: "mla"}
_MLP_TAG = {Symbols.MLP: "mlp", Symbols.MOE: "moe"}

# Canonical display order inside a bracket group, used by ``CombinedLayerSpec.canonical``.
_CANONICAL_ORDER = (
    Symbols.MAMBA,
    Symbols.GDN,
    Symbols.ATTENTION,
    Symbols.DS_ATTENTION,
    Symbols.MLP,
    Symbols.MOE,
)

_BRACKET_TOKEN_RE = re.compile(r"\[([^\[\]]*)\]")


@dataclass(frozen=True)
class CombinedLayerSpec:
    """Descriptor for one combined decoder layer parsed from a bracket group.

    Maps directly onto :class:`CombinedHybridLayer` constructor selectors:
    ``spec.kwargs`` is a ``{'mamba_type': ..., 'attention_type': ...,
    'mlp_type': ...}`` dict that can be unpacked into ``**spec.kwargs``.
    """

    mamba_type: str  # 'mamba' | 'gdn' | 'none'
    attention_type: str  # 'attention' | 'mla' | 'none'
    mlp_type: str  # 'mlp' | 'moe' | 'none'

    @property
    def kwargs(self) -> dict:
        """The three selectors as a kwargs dict for ``CombinedHybridLayer``."""
        return {
            "mamba_type": self.mamba_type,
            "attention_type": self.attention_type,
            "mlp_type": self.mlp_type,
        }

    @property
    def canonical(self) -> str:
        """Canonical bracket rendering with components in fixed order.

        ``CombinedLayerSpec('mamba', 'attention', 'moe').canonical == '[M*E]'``.
        """
        present = []
        if self.mamba_type == "mamba":
            present.append(Symbols.MAMBA)
        elif self.mamba_type == "gdn":
            present.append(Symbols.GDN)
        if self.attention_type == "attention":
            present.append(Symbols.ATTENTION)
        elif self.attention_type == "mla":
            present.append(Symbols.DS_ATTENTION)
        if self.mlp_type == "mlp":
            present.append(Symbols.MLP)
        elif self.mlp_type == "moe":
            present.append(Symbols.MOE)
        ordered = [c for c in _CANONICAL_ORDER if c in present]
        return "[" + "".join(ordered) + "]"

    @property
    def is_moe(self) -> bool:
        return self.mlp_type == "moe"


def is_bracket_pattern(pattern: Optional[str]) -> bool:
    """True iff ``pattern`` uses the bracket grammar.

    Detection is by presence of ``'['``. ``None`` / empty returns False.
    """
    return pattern is not None and Symbols.LBRACKET in pattern


def _validate_group(body: str) -> Set[str]:
    """Validate one bracket body and return its component set.

    Raises ``ValueError`` with a clear message on any rule violation.
    """
    if not body:
        raise ValueError("Empty bracket group '[]' is not allowed.")

    if Symbols.PIPE in body:
        raise ValueError(f"Pipe '|' is not allowed inside a bracket group; got '[{body}]'.")
    if Symbols.MTP_SEPARATOR in body:
        raise ValueError(
            f"MTP separator '/' is not allowed inside a bracket group; got '[{body}]'."
        )
    if Symbols.LBRACKET in body or Symbols.RBRACKET in body:
        raise ValueError(f"Nested brackets are not allowed; got '[{body}]'.")

    seen: Set[str] = set()
    for ch in body:
        if ch not in Symbols.VALID_LAYERS:
            raise ValueError(
                f"In bracket group '[{body}]', '{ch}' is not a valid layer "
                f"symbol. Valid symbols: {sorted(Symbols.VALID_LAYERS)}."
            )
        if ch in seen:
            raise ValueError(
                f"Component '{ch}' is repeated in bracket group '[{body}]'; "
                "each component may appear at most once."
            )
        seen.add(ch)

    # At most one sequence mixer, one attention kind, one MLP kind.
    mamba_hits = seen & _MAMBA_SYMBOLS
    if len(mamba_hits) > 1:
        raise ValueError(
            f"Bracket group '[{body}]' has both Mamba (M) and GDN (G); "
            "pick at most one sequence mixer."
        )
    attn_hits = seen & _ATTN_SYMBOLS
    if len(attn_hits) > 1:
        raise ValueError(
            f"Bracket group '[{body}]' has both Attention (*) and DS_ATTENTION (D); "
            "pick at most one attention kind."
        )
    mlp_hits = seen & _MLP_SYMBOLS
    if len(mlp_hits) > 1:
        raise ValueError(
            f"Bracket group '[{body}]' has both MLP (-) and MoE (E); "
            "pick at most one MLP kind."
        )

    return seen


def _spec_from_components(components: Set[str]) -> CombinedLayerSpec:
    """Build a :class:`CombinedLayerSpec` from a set of bracket components."""
    mamba_type = "none"
    for sym, tag in _MAMBA_TAG.items():
        if sym in components:
            mamba_type = tag
            break
    attention_type = "none"
    for sym, tag in _ATTN_TAG.items():
        if sym in components:
            attention_type = tag
            break
    mlp_type = "none"
    for sym, tag in _MLP_TAG.items():
        if sym in components:
            mlp_type = tag
            break
    return CombinedLayerSpec(
        mamba_type=mamba_type, attention_type=attention_type, mlp_type=mlp_type
    )


def parse_bracket_segment(segment: str) -> List[CombinedLayerSpec]:
    """Parse one pipeline segment (no ``|`` / no ``/``) into combined-layer specs.

    Args:
        segment: e.g. ``"[M*-][M-][ME]"``. Whitespace is ignored.

    Returns:
        List of :class:`CombinedLayerSpec`, one per bracket group.
    """
    s = segment.replace(" ", "")
    if not s:
        return []

    tokens = _BRACKET_TOKEN_RE.findall(s)
    # Verify nothing extraneous sits between bracket groups.
    reconstructed = "".join(f"[{t}]" for t in tokens)
    if reconstructed != s:
        raise ValueError(
            f"Pattern segment '{segment}' contains characters outside bracket groups. "
            "Bracket mode requires every component to live inside a [...] group."
        )
    return [_spec_from_components(_validate_group(t)) for t in tokens]


def parse_bracket_pattern(pattern: str) -> List[List[CombinedLayerSpec]]:
    """Parse a full bracket pattern into per-pipeline-segment lists of specs.

    ``|`` marks pipeline-stage boundaries and cannot appear inside a bracket.
    ``/`` (MTP separator) is handled upstream, not here.

    Returns:
        Nested list ``[[spec, ...], ...]`` -- one inner list per pipeline
        segment in left-to-right order.
    """
    if not pattern:
        return [[]]
    segments = pattern.split(Symbols.PIPE)
    return [parse_bracket_segment(seg) for seg in segments]


__all__ = [
    "CombinedLayerSpec",
    "is_bracket_pattern",
    "parse_bracket_segment",
    "parse_bracket_pattern",
]
