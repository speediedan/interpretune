"""Check a component map against the model it describes, once, at load.

A map that is wrong about its model must fail here, named, rather than at bind time as a plausible module path
that does not exist. Three assertions, all cheap and all on the module tree without a forward pass:

- every row's module exists for every layer it covers, and does NOT exist for a layer its ``layers`` predicate
  excludes (a row declared "all layers" on a heterogeneous stack fails here on the first layer that lacks it);
- every derived property agrees with the model (a document may state one as a cross-check; the model decides);
- every declared stack's template addresses at least one layer.

The output rule of a kind (tensor versus tuple) needs a forward and stays the backend's measurement.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from interpretune.analysis.points.component_map import PRIMARY_STACK, ComponentMap


@dataclass(frozen=True)
class MapProblem:
    """One disagreement between a map and its model."""

    row: str
    layer: int | None
    reason: str

    def __str__(self) -> str:
        where = f"{self.row} at layer {self.layer}" if self.layer is not None else self.row
        return f"{where}: {self.reason}"


class ComponentMapModelMismatch(ValueError):
    """The map does not describe this model; every problem is listed."""


def _submodule(model: Any, path: str) -> Any | None:
    node = model
    for part in path.split("."):
        if part.isdigit():
            try:
                node = node[int(part)]
            except (TypeError, IndexError, KeyError):
                return None
            continue
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def _stack_length(model: Any, template: str) -> int:
    """How many layers a ``{i}`` template addresses on this model (the first index that does not resolve)."""
    n = 0
    while _submodule(model, template.replace("{i}", str(n))) is not None:
        n += 1
        if n > 4096:  # a template that always resolves is not a stack
            break
    return n


def derive_rmsnorm_offset(model: Any, cmap: ComponentMap) -> bool | None:
    """Whether the final norm scales by ``(1 + weight)`` (gemma, gemma2, gemma3) or by ``weight`` (everything else,
    gemma3n and gemma4 included), read from the module's own forward rather than from a family name.

    ``None`` when the map has no final norm or its forward cannot be read (a compiled or scripted module).
    """
    path = cmap.module_for("ln_final", None)
    norm = _submodule(model, path) if path else None
    if norm is None:
        return None
    try:
        source = inspect.getsource(type(norm).forward)
    except (OSError, TypeError):
        return None
    return "1.0 + self.weight" in source or "1 + self.weight" in source or "(1.0 + self.weight" in source


def check_map_against_model(cmap: ComponentMap, model: Any) -> list[MapProblem]:
    """Every disagreement between the map and the model, empty when the map describes it."""
    problems: list[MapProblem] = []
    templates = {PRIMARY_STACK: None, **cmap.stacks}
    lengths: dict[str, int] = {}
    for stack, template in templates.items():
        if template is None:
            block = cmap.components.get("blocks.{i}")
            template = block.module if block is not None else None
        if template is None:
            continue
        lengths[stack] = _stack_length(model, template)
        if lengths[stack] == 0:
            problems.append(
                MapProblem(f"stack {stack!r}", None, f"template {template!r} addresses no layer on this model")
            )
    for name, entry in cmap.components.items():
        if "{i}" not in entry.module:
            if _submodule(model, entry.module) is None:
                problems.append(MapProblem(name, None, f"module {entry.module!r} does not exist"))
            continue
        stack = name.split(".blocks.{i}", 1)[0] if ".blocks.{i}" in name else PRIMARY_STACK
        layers = entry.layers
        for layer in range(lengths.get(stack, 0)):
            exists = _submodule(model, entry.module.replace("{i}", str(layer))) is not None
            covered = layers is None or layers.covers(layer)
            if covered and not exists:
                problems.append(
                    MapProblem(name, layer, f"row covers this layer but {entry.module!r} does not exist here")
                )
            elif exists and layers is not None and not covered:
                problems.append(
                    MapProblem(name, layer, f"row excludes this layer ({layers.describe()}) but the module exists")
                )
    declared = cmap.properties.get("rmsnorm_offset")
    if declared is not None:
        derived = derive_rmsnorm_offset(model, cmap)
        if derived is not None and derived != declared:
            problems.append(
                MapProblem(
                    "properties.rmsnorm_offset", None, f"document says {declared}, the model's norm says {derived}"
                )
            )
    return problems


def validate_map_against_model(cmap: ComponentMap, model: Any) -> None:
    """Raise :class:`ComponentMapModelMismatch` listing every problem, or return when the map describes the
    model."""
    problems = check_map_against_model(cmap, model)
    if problems:
        listing = "\n  ".join(str(p) for p in problems)
        raise ComponentMapModelMismatch(
            f"the component map for {cmap.architecture!r} does not describe this model ({len(problems)} problems):\n  "
            f"{listing}"
        )
