"""The activation-point vocabulary: one way to name a tensor across TransformerLens hooks, HF module paths and
the other vocabularies that address the same activations.

- :mod:`vocabulary`: :func:`parse` any accepted spelling into an :class:`ActivationPoint` (semantic names such
  as ``hook_resid_pre`` and component names such as ``blocks.5.ln2.hook_out`` are the same record).
- :mod:`component_map`: the per-architecture DATA (component -> module path + kind), bundled as YAML.
- :mod:`resolution`: point + map -> :class:`TensorRef` (a PyTorch tensor position, possibly derived) or
  :class:`Unresolvable` (with the reason, as a value).
"""

from __future__ import annotations

from interpretune.analysis.points.component_map import (
    KindSpec,
    LayerSet,
    ComponentEntry,
    ComponentMap,
    component_map_for,
    from_transformer_lens,
    known_architectures,
    load_component_map_file,
    register,
)

# the resolver module is `resolution`, not `resolve`: a submodule sharing the name of a re-exported function
# gets rebound over it whenever the submodule is (re)imported, which the test suite's module reloads do
from interpretune.analysis.points.resolution import Resolution, TensorRef, Unresolvable, describe_unresolvable, resolve
from interpretune.analysis.points.validation import (
    ComponentMapModelMismatch,
    MapProblem,
    check_map_against_model,
    derive_rmsnorm_offset,
    validate_map_against_model,
)
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
    "ComponentMapModelMismatch",
    "KindSpec",
    "LayerSet",
    "MapProblem",
    "Contribution",
    "Resolution",
    "Slot",
    "TensorRef",
    "UnknownPointError",
    "Unresolvable",
    "component_map_for",
    "from_transformer_lens",
    "describe_unresolvable",
    "check_map_against_model",
    "derive_rmsnorm_offset",
    "known_architectures",
    "load_component_map_file",
    "parse",
    "register",
    "resolve",
    "semantic_names",
    "validate_map_against_model",
]
