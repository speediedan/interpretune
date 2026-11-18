"""The activation-point vocabulary: one way to name a tensor across TransformerLens hooks, HF module paths and
the other vocabularies that address the same activations.

- :mod:`vocabulary`: :func:`parse` any accepted spelling into an :class:`ActivationPoint` (semantic names such
  as ``hook_resid_pre`` and component names such as ``blocks.5.ln2.hook_out`` are the same record).
- :mod:`component_map`: the per-architecture DATA (component -> module path + kind), read from venngram's component
  maps and bound to a model where one exists.
- :mod:`resolution`: point + map -> :class:`TensorRef` (a PyTorch tensor position, possibly derived) or
  :class:`Unresolvable` (with the reason, as a value).
"""

from __future__ import annotations

from venngram.maps import ComponentMapModelMismatch

from interpretune.analysis.points.component_map import (
    ComponentMap,
    bind_component_map,
    component_map_for,
    from_venngram,
    known_architectures,
    register,
)

# the resolver module is `resolution`, not `resolve`: a submodule sharing the name of a re-exported function
# gets rebound over it whenever the submodule is (re)imported, which the test suite's module reloads do
from interpretune.analysis.points.resolution import Resolution, TensorRef, Unresolvable, describe_unresolvable, resolve
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
    "ComponentMap",
    "ComponentMapModelMismatch",
    "Contribution",
    "Resolution",
    "Slot",
    "TensorRef",
    "UnknownPointError",
    "Unresolvable",
    "bind_component_map",
    "component_map_for",
    "describe_unresolvable",
    "from_venngram",
    "known_architectures",
    "parse",
    "register",
    "resolve",
    "semantic_names",
]
