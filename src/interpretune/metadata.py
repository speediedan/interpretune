from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type


@dataclass(frozen=True)
class ITClassMetadata:
    """Structured, typed container for class-level Interpretune metadata.

    This is a small, dependency-free dataclass used by multiple core modules. Placing it
    in `interpretune.metadata` keeps imports lightweight and avoids importing the
    `interpretune.base` package during import-time of modules like `interpretune.session`.
    """

    base_attrs: dict[Any, tuple[str, ...]] = field(default_factory=dict)
    ready_attrs: tuple[str, ...] = field(default_factory=tuple)
    composition_target_attrs: tuple[str, ...] = field(default_factory=tuple)
    ready_protocols: tuple[Type, ...] = field(default_factory=tuple)

    # Generic extension points used by other components
    core_to_framework_attrs_map: dict[str, Any] = field(default_factory=dict)
    property_composition: dict[str, Any] = field(default_factory=dict)
    gen_prepares_inputs_sigs: tuple[str, ...] = field(default_factory=tuple)


__all__ = ["ITClassMetadata"]
