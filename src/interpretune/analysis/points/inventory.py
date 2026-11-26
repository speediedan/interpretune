"""The vocabulary's points for one architecture, enumerated from its component map.

A backend's capture declaration is a subset of this inventory, so the inventory is what makes "captures 181 of 298"
a fraction of something rather than a count. Base spellings are layer-free (``ln2.hook_out``, ``hook_resid_pre``,
``unembed.hook_in``); :func:`named_inventory` expands the block-relative ones over the model's layers.
"""

from __future__ import annotations

from collections.abc import Iterable

from interpretune.analysis.points.component_map import ComponentMap
from interpretune.analysis.points.resolution import _contribution_component
from interpretune.analysis.points.vocabulary import Slot, declaration_key, parse, semantic_names

#: Slots a norm component exposes beyond its input and output: the derived tensors the vocabulary names.
_NORM_SLOTS = (Slot.NORMALIZED, Slot.SCALE)


def inventory(cmap: ComponentMap, stack: str | None = None) -> tuple[str, ...]:
    """Every base spelling the vocabulary can address on ``cmap``'s primary stack: block-relative first, then
    global.

    Block-relative bases carry no ``blocks.{i}.`` prefix; the block itself contributes ``hook_in`` / ``hook_out`` and
    the semantic names that survive an architecture change. A component of kind ``norm`` also contributes its two
    derived slots, which the reference cannot capture but a bridge can. Sorted, so a declaration built from it
    is stable across runs.

    Layer-free by construction: a base belongs to the inventory when the map has a row for it, whichever layers that
    row covers, so a hybrid stack's attention (restricted or optional) is in the inventory of an unbound map too, and
    resolving it is where an unbound optional row is refused.
    """
    bases: set[str] = {"hook_in", "hook_out"}
    for component in cmap.block_components(stack):
        if not component:
            continue  # the block row itself: already `hook_in` / `hook_out`
        for slot in (Slot.IN, Slot.OUT):
            bases.add(f"{component}.hook_{slot.value}")
        row = cmap.row(component, stack)
        if row is not None and row.kind == "norm":
            for slot in _NORM_SLOTS:
                bases.add(f"{component}.hook_{slot.value}")
    for name in semantic_names():
        # a semantic name the architecture cannot host (a cross-attention point on a decoder) parses but does not
        # resolve; keep the inventory to what the component map can back so a declaration is not measured against
        # points no backend on this architecture could ever capture. Filed under the declaration key, so a semantic
        # name for the same tensor as a component spelling collapses into it and a contribution keeps its own.
        point = parse(f"blocks.0.{name}")
        if point.component == "" or cmap.row(point.component, stack) is not None:
            bases.add(declaration_key(f"blocks.0.{name}"))
    for component in cmap.global_components():
        for slot in (Slot.IN, Slot.OUT):
            bases.add(f"{component}.hook_{slot.value}")
    return tuple(sorted(bases))


def is_global(base: str, cmap: ComponentMap) -> bool:
    """Whether a base spelling names a global point (no layer) rather than a block-relative one."""
    head = base.split(".hook_")[0]
    return head in cmap.global_components()


def named_inventory(cmap: ComponentMap, n_layers: int, stack: str | None = None) -> tuple[str, ...]:
    """The inventory with every block-relative base expanded over ``n_layers`` layers, globals once."""
    names: list[str] = []
    for base in inventory(cmap, stack):
        if is_global(base, cmap):
            names.append(base)
        else:
            names.extend(f"blocks.{layer}.{base}" for layer in range(n_layers))
    return tuple(names)


def first_layer(base: str, cmap: ComponentMap) -> int:
    """The lowest layer a block-relative base exists on: that of the row it resolves through (a contribution
    through its post-norm where the block has one), ``0`` for a row that covers every layer.

    Raises:
        ValueError: If that row is optional and ``cmap`` is unbound, naming the reason.
    """
    point = parse(f"blocks.0.{base}")
    component = _contribution_component(point, cmap) if point.contribution is not None else point.component
    return cmap.first_layer(component)


def spelled_at(base: str, cmap: ComponentMap, layer: int | None = None) -> str:
    """A base as a full name at ``layer`` (or bare, for a global point), for probing a model with it.

    Without ``layer``, a block-relative base is spelled at the first layer it exists on (:func:`first_layer`), never
    at layer 0 by assumption: on a hybrid stack layer 0 may not have the module.
    """
    if is_global(base, cmap):
        return base
    return f"blocks.{first_layer(base, cmap) if layer is None else layer}.{base}"


def bases_of(names: Iterable[str], cmap: ComponentMap) -> dict[str, str]:
    """``{name: base}`` for full names, so a declaration keyed by base can answer for a caller's spelled names."""
    out: dict[str, str] = {}
    for name in names:
        point = parse(name)
        out[name] = point.base if not point.is_global else point.base
    return out
