# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bracket-notation pattern parser for combined hybrid layers.

The legacy single-character pattern (e.g. ``M*M-``) describes one separate
module per character. The *combined* hybrid layer bundles multiple functional
blocks into a single module, so we introduce a new bracket grammar where each
``[...]`` token describes the composition of one combined layer.

Grammar
-------
::

    pattern     := token+
    token       := '[' component+ ']'
    component   := 'M' | 'G' | '*' | 'D' | '-' | 'E'

Every bracket group corresponds to one entry in ``decoder.layers``. The ``|``
(pipeline boundary) and ``/`` (MTP separator) symbols used by the legacy parser
are still recognised around bracket groups.

Composition rules (per bracket group):

* At most one Mamba-family mixer (``M`` or ``G``)
* At most one attention kind (``*`` or ``D``)
* At most one MLP kind (``-`` or ``E``)
* Must contain at least one component
* Ordering inside brackets is ignored — only the set of characters matters

Examples
--------
* ``[M-]``          → one combined layer: Mamba + MLP
* ``[M*-][M-][ME]`` → three combined layers: Mamba+Attn+MLP → Mamba+MLP → Mamba+MoE
* ``[*E][*E]|[*E][*E]`` → four combined layers split across two PP stages
* ``[M-][M-]/[M]``  → two-layer main pattern with one MTP depth

Helpers in this module expose :func:`is_bracket_pattern` so callers can decide
whether to dispatch to the combined-layer pipeline or the legacy separate-layer
pipeline.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols


# ---------------------------------------------------------------------------
# Component selectors used to configure CombinedHybridLayer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombinedLayerSpec:
    """Describes the composition of a single combined layer.

    The three string selectors map directly to :class:`CombinedHybridLayer`
    constructor arguments (``mamba_type``, ``attention_type``, ``mlp_type``),
    so an allocator can build layers with ``CombinedHybridLayer(**spec.kwargs)``.
    """

    mamba_type: str       # 'mamba' | 'gdn' | 'none'
    attention_type: str   # 'attention' | 'mla' | 'none'
    mlp_type: str         # 'mlp' | 'moe' | 'none'

    @property
    def kwargs(self) -> dict:
        return {
            'mamba_type': self.mamba_type,
            'attention_type': self.attention_type,
            'mlp_type': self.mlp_type,
        }

    @property
    def canonical(self) -> str:
        """Canonical bracket form (components in fixed order, e.g. ``[M*-]``)."""
        order = [
            ('M', self.mamba_type == 'mamba'),
            ('G', self.mamba_type == 'gdn'),
            ('*', self.attention_type == 'attention'),
            ('D', self.attention_type == 'mla'),
            ('-', self.mlp_type == 'mlp'),
            ('E', self.mlp_type == 'moe'),
        ]
        return '[' + ''.join(c for c, enabled in order if enabled) + ']'

    @property
    def is_moe(self) -> bool:
        return self.mlp_type == 'moe'


_BRACKET_TOKEN_RE = re.compile(r'\[([^\[\]]+)\]')


def is_bracket_pattern(pattern: Optional[str]) -> bool:
    """True if ``pattern`` uses bracket notation (vs legacy single-char notation).

    An empty or ``None`` pattern returns False. A pattern that mixes bracket and
    non-bracket tokens raises — callers should reject such patterns upstream.
    """
    if not pattern:
        return False
    return '[' in pattern


def _components_from_group(group: str) -> set:
    """Validate a bracket group body and return its component set."""
    seen = set()
    for ch in group:
        if ch not in Symbols.VALID_LAYERS:
            raise ValueError(
                f"Invalid component '{ch}' in bracket group '[{group}]'. "
                f"Valid components: {sorted(Symbols.VALID_LAYERS)}"
            )
        if ch in seen:
            raise ValueError(f"Component '{ch}' repeated in bracket group '[{group}]'")
        seen.add(ch)

    if not seen:
        raise ValueError(f"Empty bracket group '[{group}]'")

    # At most one mamba kind
    if Symbols.MAMBA in seen and Symbols.GDN in seen:
        raise ValueError(
            f"Bracket group '[{group}]' has both Mamba (M) and GDN (G); pick one"
        )
    # At most one attention kind
    if Symbols.ATTENTION in seen and Symbols.DS_ATTENTION in seen:
        raise ValueError(
            f"Bracket group '[{group}]' has both Attention (*) and DS_ATTENTION (D); pick one"
        )
    # At most one MLP kind
    if Symbols.MLP in seen and Symbols.MOE in seen:
        raise ValueError(
            f"Bracket group '[{group}]' has both MLP (-) and MoE (E); pick one"
        )

    return seen


def _spec_from_components(components: set) -> CombinedLayerSpec:
    if Symbols.MAMBA in components:
        mamba_type = 'mamba'
    elif Symbols.GDN in components:
        mamba_type = 'gdn'
    else:
        mamba_type = 'none'

    if Symbols.ATTENTION in components:
        attention_type = 'attention'
    elif Symbols.DS_ATTENTION in components:
        attention_type = 'mla'
    else:
        attention_type = 'none'

    if Symbols.MLP in components:
        mlp_type = 'mlp'
    elif Symbols.MOE in components:
        mlp_type = 'moe'
    else:
        mlp_type = 'none'

    return CombinedLayerSpec(mamba_type=mamba_type, attention_type=attention_type, mlp_type=mlp_type)


def parse_bracket_segment(segment: str) -> List[CombinedLayerSpec]:
    """Parse one pipeline segment (no ``|`` or ``/``) and return combined-layer specs.

    The segment must consist entirely of bracket groups with no extraneous
    characters between them. Whitespace is allowed and ignored.

    Args:
        segment: e.g. ``"[M*-][M-][ME]"``

    Returns:
        List of :class:`CombinedLayerSpec`, one per bracket group.
    """
    s = segment.replace(' ', '')
    if not s:
        return []

    # Find all bracket groups
    tokens = _BRACKET_TOKEN_RE.findall(s)

    # Reconstruct with brackets and verify nothing else is present
    reconstructed = ''.join(f'[{t}]' for t in tokens)
    if reconstructed != s:
        raise ValueError(
            f"Pattern segment '{segment}' contains characters outside bracket groups. "
            "Bracket mode requires every component to live inside a [...] group."
        )

    specs = [_spec_from_components(_components_from_group(t)) for t in tokens]
    return specs


def parse_bracket_pattern(pattern: str) -> List[List[CombinedLayerSpec]]:
    """Parse a full bracket pattern into per-pipeline-segment lists of specs.

    Handles ``|`` (PP segment boundary). Does not handle ``/`` (MTP); callers
    should split on ``/`` first and pass only the main-pattern portion.

    Args:
        pattern: e.g. ``"[M*-][M-]|[M-][M*-]"``

    Returns:
        Nested list: ``[[spec, ...], [spec, ...], ...]``, one inner list per
        pipeline segment in order.
    """
    if not pattern:
        return [[]]
    segments = pattern.split(Symbols.PIPE)
    return [parse_bracket_segment(seg) for seg in segments]


__all__ = [
    'CombinedLayerSpec',
    'is_bracket_pattern',
    'parse_bracket_segment',
    'parse_bracket_pattern',
]
