"""The ``collection:`` header an op-definitions YAML may declare, and its compatibility window.

An op collection versions a *contract set*, not a package: the names, schemas and traits its ops present to
callers. That is why the version lives here rather than being inferred from the installed distribution -- a hub
collection has no distribution, and a bundled family's contract can change without interpretune's version
changing.

Compatibility is deliberately one window per collection against the installed interpretune, checked with the same
``requires:`` grammar and the same :func:`~interpretune.hub.components.enforce_component_requires` machinery that
component manifests use. There is no cross-collection dependency resolution and no solver (D8): an incompatible
collection is skipped with a warning, or raises under ``IT_STRICT_OP_LOAD=1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

COLLECTION_HEADER_KEY = "collection"


@dataclass(frozen=True)
class CollectionSpec:
    """A collection's declared identity and compatibility window.

    Args:
        name: the collection handle (``concept``, ``speediedan/concept_direction_ops``, ...). Free-form; it names
            the contract set for diagnostics and precedence, and is not required to match the repo or directory.
        version: PEP 440 version of the op *contract set*.
        requires: the same mapping component manifests use (``interpretune``, ``adapters``, ``pip``).
    """

    name: str
    version: str
    requires: dict[str, Any]

    RECOGNIZED_KEYS: ClassVar[frozenset[str]] = frozenset({"name", "version", "requires"})

    def __post_init__(self) -> None:
        from packaging.version import InvalidVersion, Version

        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"collection.name must be a non-empty string, got {self.name!r}")
        try:
            Version(str(self.version))
        except InvalidVersion as invalid:
            raise ValueError(f"collection.version must be a PEP 440 version, got {self.version!r}") from invalid
        if not isinstance(self.requires, dict):
            raise ValueError(
                f"collection.requires must be a mapping using the component `requires:` grammar "
                f"(interpretune/adapters/pip), got {type(self.requires).__name__}. A bare list is the shape used "
                "in early design notes; the mapping is what the shared enforcement machinery reads."
            )

    @classmethod
    def from_raw(cls, raw: Any) -> "CollectionSpec | None":
        """Build a spec from a raw YAML mapping (``None`` when a collection declares no header)."""
        if raw is None:
            return None
        if isinstance(raw, CollectionSpec):
            return raw
        if not isinstance(raw, dict):
            raise ValueError(f"`{COLLECTION_HEADER_KEY}` must be a mapping, got {type(raw).__name__}")
        unrecognized = sorted(set(raw) - cls.RECOGNIZED_KEYS)
        if unrecognized:
            raise ValueError(
                f"Unrecognized `{COLLECTION_HEADER_KEY}` keys: {unrecognized} "
                f"(recognized: {sorted(cls.RECOGNIZED_KEYS)})"
            )
        for required in ("name", "version"):
            if required not in raw:
                raise ValueError(f"`{COLLECTION_HEADER_KEY}` requires a `{required}`")
        requires = raw.get("requires") or {}
        if not isinstance(requires, dict):
            # Checked before coercion: `dict(["interpretune>=0.1"])` raises its own opaque error, and a bare
            # list is the shape early design notes used, so it is worth naming explicitly.
            raise ValueError(
                f"`{COLLECTION_HEADER_KEY}.requires` must be a mapping using the component `requires:` grammar "
                f"(interpretune/adapters/pip), got {type(requires).__name__}. A bare list is the shape used in "
                "early design notes; the mapping is what the shared enforcement machinery reads."
            )
        return cls(name=str(raw["name"]), version=str(raw["version"]), requires=dict(requires))

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to the raw YAML/cache mapping form."""
        return {"name": self.name, "version": self.version, "requires": dict(self.requires)}

    def incompatibility(self) -> str | None:
        """Return why this collection is incompatible with the installed environment, or None if it is fine.

        Returns a message rather than raising so the caller decides the policy: op loading warns and skips by
        default (one bad collection must not take down a session) and raises under strict loading.
        """
        if not self.requires:
            return None
        from interpretune.hub.components import ComponentRequirementError, enforce_component_requires

        try:
            enforce_component_requires({"requires": self.requires}, source=f"op collection {self.name!r}")
        except ComponentRequirementError as incompatible:
            return str(incompatible)
        return None


__all__ = ["COLLECTION_HEADER_KEY", "CollectionSpec"]
