"""Resolution: an :class:`ActivationPoint` plus a :class:`ComponentMap` gives a tensor position, or a reason.

"Does not translate" is a value here, not an exception. Two independent implementations grew a sentinel for
it (a third-party engine mapper's unmapped-hook marker, and interpretune's own unmappable-hook reasons), which
is the signal that it belongs in the schema: a consumer enumerating support renders it, and a consumer that
needs the tensor raises on it with the reason attached.
"""

from __future__ import annotations

from dataclasses import dataclass

from interpretune.analysis.points.component_map import KIND_TUPLE_OUTPUT, ComponentMap
from interpretune.analysis.points.vocabulary import ActivationPoint, Contribution, Slot


@dataclass(frozen=True)
class TensorRef:
    """A tensor position in a PyTorch module tree."""

    module_path: str
    io: str  # "input" | "output"
    tuple_output: bool = False
    """Whether the module returns a tuple whose element 0 is the tensor (static default per kind; a backend may
    measure the real value for its model)."""
    derivation: str | None = None
    """Set when the tensor is not emitted by any module and must be COMPUTED from the referenced one:
    ``"pre_gain_normalized"`` (a norm's ``x / scale``, from its input and its parameters) or ``"norm_scale"``
    (the per-token denominator)."""

    @property
    def derived(self) -> bool:
        """Whether the tensor must be computed rather than read off a module."""
        return self.derivation is not None


@dataclass(frozen=True)
class Unresolvable:
    """The point has no tensor position in this architecture, and why.

    Never a silent substitute.
    """

    reason: str
    alternatives: tuple[str, ...] = ()


Resolution = TensorRef | Unresolvable


def _contribution_component(point: ActivationPoint, cmap: ComponentMap) -> str:
    """Where a sublayer's residual contribution is read: the post-norm when the block has one, else the module."""
    if point.contribution is Contribution.ATTN:
        return "ln1_post" if cmap.sandwich_norms and cmap.kind_of("ln1_post", 0) == "norm" else "attn"
    if point.contribution is Contribution.MLP:
        return "ln2_post" if cmap.sandwich_norms and cmap.kind_of("ln2_post", 0) == "norm" else "mlp"
    return point.component


def resolve(point: ActivationPoint, cmap: ComponentMap) -> Resolution:
    """Resolve a parsed point against an architecture's component map."""
    layer = point.layer
    component = _contribution_component(point, cmap) if point.contribution is not None else point.component
    lookup_layer = layer if not point.is_global else None
    if point.is_global and not component:
        return Unresolvable("a global point needs a component (e.g. 'unembed.hook_out')")
    module = cmap.module_for(component, lookup_layer)
    kind = cmap.kind_of(component, lookup_layer)
    if module is None or kind is None:
        where = f"blocks.{layer}" if layer is not None else "the model"
        known = sorted(c for c in (cmap.block_components() if layer is not None else cmap.global_components()) if c)
        return Unresolvable(
            f"{cmap.architecture} has no component {component or '<block>'!r} under {where}; known: {known}",
            alternatives=tuple(f"{c}.hook_{point.slot.value}" for c in known[:6]),
        )

    tuple_output = KIND_TUPLE_OUTPUT[kind]
    if point.slot is Slot.IN:
        return TensorRef(module, "input", tuple_output=False)
    if point.slot is Slot.OUT:
        return TensorRef(module, "output", tuple_output=tuple_output)
    if kind != "norm":
        return Unresolvable(
            f"hook_{point.slot.value} is a norm's intermediate and {component!r} is a {kind}; "
            f"ask for {component}.hook_in or {component}.hook_out"
        )
    if point.slot is Slot.NORMALIZED:
        return TensorRef(
            module,
            "input",
            tuple_output=False,
            derivation="pre_gain_normalized",
        )
    return TensorRef(module, "input", tuple_output=False, derivation="norm_scale")


def describe_unresolvable(point: ActivationPoint, res: Unresolvable) -> str:
    """The refusal message a tensor-needing consumer raises with."""
    alt = f" Alternatives: {', '.join(res.alternatives)}." if res.alternatives else ""
    return f"{point.canonical} does not resolve: {res.reason}.{alt}"
