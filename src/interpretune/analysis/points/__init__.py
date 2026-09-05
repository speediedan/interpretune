"""The activation-point vocabulary: one way to name a tensor across TransformerLens hooks, HF module paths and
the other vocabularies that address the same activations.

- :mod:`vocabulary`: :func:`parse` any accepted spelling into an :class:`ActivationPoint` (semantic names such
  as ``hook_resid_pre`` and component names such as ``blocks.5.ln2.hook_out`` are the same record).
- :mod:`component_map`: the per-architecture DATA (component -> module path + kind), bundled as YAML.
- :mod:`resolve`: point + map -> :class:`TensorRef` (a PyTorch tensor position, possibly derived) or
  :class:`Unresolvable` (with the reason, as a value).
"""

from __future__ import annotations

from interpretune.analysis.points.component_map import (
    ComponentEntry,
    ComponentMap,
    component_map_for,
    from_transformer_lens,
    known_architectures,
    load_component_map_file,
    register,
)
from interpretune.analysis.points.resolve import Resolution, TensorRef, Unresolvable, describe_unresolvable, resolve
from interpretune.analysis.points.vocabulary import (
    ActivationPoint,
    Contribution,
    Slot,
    UnknownPointError,
    parse,
    semantic_names,
)

__all__ = [
    "ActivationPoint",
    "ComponentEntry",
    "ComponentMap",
    "Contribution",
    "Resolution",
    "Slot",
    "TensorRef",
    "UnknownPointError",
    "Unresolvable",
    "component_map_for",
    "from_transformer_lens",
    "describe_unresolvable",
    "known_architectures",
    "load_component_map_file",
    "parse",
    "register",
    "resolve",
    "semantic_names",
]
