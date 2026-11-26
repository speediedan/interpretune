"""``ComponentMap``: where each bridge component lives in an architecture's module tree, resolved for the
vocabulary.

The documents themselves are venngram's: one component map per architecture, read through venngram's reader and found
through venngram's lookup (a map registered in this process, else the configured registries, served from the cache
when the registry cannot be reached). Interpretune bundles no map and writes none; the schema, its reader and its
evolution policy are documented in venngram (``venngram.maps.schema``).

What this module adds is the RESOLVED form the vocabulary consumes. A venngram map may mark a block-relative row
``optional``: the row applies wherever its module exists, which on a hybrid stack depends on the model's size rather
than its architecture (Qwen3.5's full-attention layers are every fourth layer of whatever depth the checkpoint has).
Such a row has no layer coverage until it is read against a model, so:

- :func:`component_map_for` returns the map UNBOUND. Every row without ``optional`` resolves exactly as written; an
  optional row is refused by name ("resolves only against a model"), never assumed to cover layer 0.
- :func:`bind_component_map` checks the map against a loaded model and probes each optional row per layer, giving
  the concrete layer coverage the resolver already understands. The backends bind where they hold the model.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from venngram.maps import ComponentMap as VenngramComponentMap
from venngram.maps.check import stack_length, submodule, validate_map_against_model
from venngram.maps.schema import PRIMARY_STACK, ComponentEntry, RowLayers

#: Row names the registry spells differently from the maps Interpretune once bundled, old -> new. An address that uses
#: the old name is refused naming the new one; there is no alias, because a second spelling for one row is a second
#: place for the two to drift.
RENAMED_COMPONENTS: dict[str, str] = {"linear_attn.o": "linear_attn.out"}


def first_covered(layers: RowLayers) -> int:
    """The lowest layer a row's ``layers`` predicate covers, where a consumer that probes one layer must look."""
    if isinstance(layers.spec, str):  # a parity
        return 1 if layers.spec == "odd" else 0
    return min(layers.spec)  # a range's lower bound, or a list's least member


@dataclass(frozen=True)
class ComponentMap(VenngramComponentMap):
    """A venngram component map in the form the vocabulary resolves through, bound to a model or not.

    Unbound, an ``optional`` row maps nothing and :meth:`unmapped_reason` says it resolves only against a model;
    bound (:func:`bind_component_map`), no row is optional and each former one carries the layers it was found on.
    """

    bound: bool = False
    """Whether the optional rows were resolved against a model (:func:`bind_component_map`)."""

    def row(self, component: str, stack: str | None = None, *, in_block: bool = True) -> ComponentEntry | None:
        """The row naming ``component`` whatever layers it covers, or ``None``: a layer-free question, such as a
        component's kind, needs no layer to ask it at."""
        return self.components.get(self._key(component, 0 if in_block else None, stack))

    def _entry(self, component: str, layer: int | None, stack: str | None) -> ComponentEntry | None:
        entry = self.components.get(self._key(component, layer, stack))
        if entry is not None and entry.optional:
            return None
        return super()._entry(component, layer, stack)

    def unmapped_reason(self, component: str, layer: int | None, stack: str | None = None) -> str | None:
        """Why a component is unmapped when a row for it EXISTS: it does not cover ``layer``, or it is an optional row
        of an unbound map; else ``None``."""
        entry = self.components.get(self._key(component, layer, stack))
        if entry is not None and entry.optional:
            return optional_row_refusal(component)
        return super().unmapped_reason(component, layer, stack)

    def first_layer(self, component: str, stack: str | None = None) -> int:
        """The lowest layer a block-relative component's row covers (``0`` for an unrestricted or absent row).

        Raises:
            ValueError: If the row is optional and the map is unbound, naming the reason.
        """
        entry = self.row(component, stack)
        if entry is None or entry.layers is None and not entry.optional:
            return 0
        if entry.optional:
            raise ValueError(f"{self.architecture}: {optional_row_refusal(component)}")
        assert entry.layers is not None
        return first_covered(entry.layers)

    def cache_key(self) -> tuple:
        """Everything a table derived from this map depends on, hashable, so a derived table is cached per map
        rather than per architecture name (two bindings of one architecture to different depths differ here)."""
        return (
            self.architecture,
            tuple(sorted(self.components.items())),
            tuple(sorted(self.kinds.items())),
            tuple(sorted(self.stacks.items())),
            repr(sorted(self.properties.items())),
        )


def optional_row_refusal(component: str) -> str:
    """The refusal for an optional row read without a model."""
    return (
        f"the {component or 'block'!r} row is optional (it applies on the layers whose module exists, which vary with "
        "the model's size), so it resolves only against a model; bind the map to one "
        "(interpretune.analysis.points.bind_component_map)"
    )


def renamed_component_refusal(component: str, cmap: VenngramComponentMap, stack: str | None = None) -> str | None:
    """The refusal for an address that spells a renamed row by its old name, naming the new one; else ``None``."""
    new = RENAMED_COMPONENTS.get(component)
    if new is None or component in cmap.block_components(stack) or new not in cmap.block_components(stack):
        return None
    return f"the {component!r} row is named {new!r} (the registry's row name); address it as {new!r}"


def from_venngram(vmap: VenngramComponentMap) -> ComponentMap:
    """An unbound :class:`ComponentMap` carrying ``vmap``'s rows, kinds, stacks and properties unchanged."""
    if isinstance(vmap, ComponentMap):
        return vmap
    return ComponentMap(**{f.name: getattr(vmap, f.name) for f in fields(VenngramComponentMap)})


def bind_component_map(cmap: VenngramComponentMap, model: Any) -> ComponentMap:
    """``cmap`` checked against ``model`` and resolved for it: each optional row becomes the layers it was found
    on.

    The check is venngram's (every row's module exists where the row says, an optional row's on at least one layer,
    a derived property agrees with the model), so a map that is wrong about its model is refused here, listing every
    problem, rather than surfacing later as a plausible module path that does not exist. The probe is the one
    venngram's map-backed structure provider places tensors by: a covered layer is one whose module resolves.

    Raises:
        venngram.maps.ComponentMapModelMismatch: If the map does not describe ``model``.
    """
    validate_map_against_model(cmap, model)
    depths: dict[str, int] = {}
    for stack in cmap.stack_names():
        template = cmap.block_template(stack)
        if template is not None:
            depths[stack] = stack_length(model, template)
    components: dict[str, ComponentEntry] = {}
    for key, entry in cmap.components.items():
        if not entry.optional:
            components[key] = entry
            continue
        depth = depths.get(cmap.key(key).stack or PRIMARY_STACK, 0)
        found = tuple(
            layer
            for layer in range(depth)
            if (entry.layers is None or entry.layers.covers(layer))
            and submodule(model, entry.module.replace("{i}", str(layer))) is not None
        )
        # the check above refuses an optional row found on no layer, so `found` is never empty here
        components[key] = replace(entry, layers=RowLayers(found, "list"), optional=False)
    return replace(from_venngram(cmap), components=components, bound=True)


def component_map_for(architecture: str, *, strict: bool = False) -> ComponentMap:
    """The unbound map for an HF architecture class name, from venngram's lookup.

    A map registered in this process (:func:`register`, or a pulled ``hookmaps`` component) wins over the
    registries; ``strict`` refuses a retired map and takes only a ``verified`` registry row.

    Raises:
        KeyError: If no map is registered or indexed for ``architecture``, naming what is known.
        venngram.registry.RegistryError: If a registry cannot be read and was never fetched before.
    """
    from venngram.maps import component_map_for as lookup

    return from_venngram(lookup(architecture, strict=strict))


def known_architectures() -> list[str]:
    """Every architecture with a map registered in this process or indexed by a configured registry."""
    from venngram.maps import known_architectures as known

    return known()


def register(cmap: VenngramComponentMap) -> None:
    """Register (or replace) the map for an architecture in this process, ahead of the registries.

    Raises:
        ValueError: If ``cmap`` is bound: a bound map describes one model's depth, not the architecture.
    """
    if isinstance(cmap, ComponentMap) and cmap.bound:
        raise ValueError(
            f"the {cmap.architecture} map is bound to a model; register the document it was read from, since a bound "
            "map's layer coverage describes one model's depth rather than the architecture"
        )
    from venngram.maps import register as venngram_register

    venngram_register(cmap)


__all__ = [
    "ComponentMap",
    "RENAMED_COMPONENTS",
    "bind_component_map",
    "component_map_for",
    "first_covered",
    "from_venngram",
    "known_architectures",
    "optional_row_refusal",
    "register",
    "renamed_component_refusal",
]
