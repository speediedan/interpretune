from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Type


@dataclass(frozen=True)
class ITClassMetadata:
    """Structured, typed container for class-level Interpretune metadata.

    Use instances of this dataclass as a single protected class attribute
    (e.g. `_it_cls_metadata`) to reduce attribute clutter on public classes.
    """

    base_attrs: Dict[Any, Tuple[str, ...]] = field(default_factory=dict)
    ready_attrs: Tuple[str, ...] = field(default_factory=tuple)
    composition_target_attrs: Tuple[str, ...] = field(default_factory=tuple)
    ready_protocols: Tuple[Type, ...] = field(default_factory=tuple)

    # Generic extension points used by other components
    core_to_framework_attrs_map: Dict[str, Any] = field(default_factory=dict)
    property_composition: Dict[str, Any] = field(default_factory=dict)
    gen_prepares_inputs_sigs: Tuple[str, ...] = field(default_factory=tuple)
