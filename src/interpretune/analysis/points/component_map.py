"""``ComponentMap``: where each bridge component lives in an architecture's module tree.

This is the architecture-varying DATA the vocabulary resolves through, and the schema that mapping
artifacts are written in. One document per architecture, a handful of rows each, over a fixed schema: a
component path in the bridge grammar, the PyTorch module path it wraps (with ``{i}`` for the layer), and
the component's KIND, from which the slot rules follow. Per-row io flags are not needed: ``hook_in`` is
always the module's first input and ``hook_out`` its output, and the two derived norm tensors are a property
of every norm rather than of any row.

Sources, in precedence order: a bundled YAML under ``data/`` (five architectures today), a registration at
runtime (``register``), and, later, a map derived from TransformerLens' own per-architecture component
mapping when it is installed. Two sources for one architecture must agree, and a test says so.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DATA_DIR = Path(__file__).parent / "data"

#: The component-map document schema this code WRITES and reads by default. 1: `architecture`, `components`
#: (component path -> {module, kind}), optional `facts`, optional `deprecated_since` / `replacement`.
#:
#: A single integer, deliberately: an additive DESCRIPTIVE key (a fact, an annotation) never bumps it, because
#: readers ignore unknown keys of that class; an APPLICABILITY key (one that changes which rows apply or how a
#: row resolves, such as a per-row layer predicate or a second block stack) always bumps it, because a reader
#: that skipped one would resolve the wrong rows silently. The version is therefore the only thing that keeps
#: "ignore what you do not understand" safe. The policy is published in `docs/activation_point_vocabulary.md`.
COMPONENT_MAP_SCHEMA_VERSION = 1

#: The oldest schema this code READS. Published maps outlive the code that wrote them, so the reader accepts a
#: window, never a single version; raising this floor retires published maps and is a documented, deliberate act.
COMPONENT_MAP_SCHEMA_MIN_READABLE = 1

#: The declared `facts` vocabulary: name -> type. A known fact of the wrong type is refused; an unknown fact is
#: ignored, because facts describe the architecture and never select rows (an older reader loses nothing it
#: would have used). New facts are added here with their type.
KNOWN_FACTS: dict[str, type] = {"sandwich_norms": bool}

#: Component kinds and whether their module returns a tuple whose element 0 is the tensor (the static
#: default; the nnsight backend measures this per model because transformers 5.x changed decoder blocks).
KIND_TUPLE_OUTPUT: dict[str, bool] = {
    "block": True,
    "attn": True,
    "mlp": False,
    "norm": False,
    "linear": False,
    "embed": False,
    "unembed": False,
}


@dataclass(frozen=True)
class ComponentEntry:
    """One row: a component path (``blocks.{i}.ln2``, ``unembed``) -> module path template + kind."""

    module: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in KIND_TUPLE_OUTPUT:
            raise ValueError(
                f"unknown component kind {self.kind!r}; expected one of {sorted(KIND_TUPLE_OUTPUT)}. A kind decides "
                "the slot rule and whether the module's output is a tuple, so it is refused rather than defaulted: a "
                "guess would resolve to a plausible wrong tensor."
            )


@dataclass(frozen=True)
class ComponentMap:
    """The component -> module table for one architecture, plus the structural facts the resolver needs."""

    architecture: str
    components: dict[str, ComponentEntry]
    facts: dict[str, Any] = field(default_factory=dict)
    source: str = "bundled"
    schema_version: int = COMPONENT_MAP_SCHEMA_VERSION
    """The document schema this map was written against.

    Required on every document: a map published without
    it is permanently unversioned and can never be told apart from one written against an unknown revision.
    """
    deprecated_since: str | None = None
    """Set when the map is retired: loading it warns, strict mode refuses, and ``replacement`` names what to use."""
    replacement: str | None = None

    @property
    def sandwich_norms(self) -> bool:
        """Whether the block has post-sublayer norms (``ln1_post`` / ``ln2_post``), which is what decides where a
        sublayer's contribution to the residual is read."""
        return bool(self.facts.get("sandwich_norms", "blocks.{i}.ln2_post" in self.components))

    def module_for(self, component: str, layer: int | None) -> str | None:
        """The concrete module path for a block-relative or global component, or ``None`` if unmapped."""
        key = (
            f"blocks.{{i}}.{component}"
            if layer is not None and component
            else ("blocks.{i}" if layer is not None else component)
        )
        entry = self.components.get(key)
        if entry is None:
            return None
        return entry.module.replace("{i}", str(layer)) if layer is not None else entry.module

    def kind_of(self, component: str, layer: int | None) -> str | None:
        """The component's kind (block, attn, mlp, norm, linear, embed, unembed), or ``None`` if unmapped."""
        key = (
            f"blocks.{{i}}.{component}"
            if layer is not None and component
            else ("blocks.{i}" if layer is not None else component)
        )
        entry = self.components.get(key)
        return None if entry is None else entry.kind

    def block_components(self) -> list[str]:
        """Block-relative component names present in this map (``""`` for the block itself)."""
        out = []
        for key in self.components:
            if key == "blocks.{i}":
                out.append("")
            elif key.startswith("blocks.{i}."):
                out.append(key[len("blocks.{i}.") :])
        return out

    def global_components(self) -> list[str]:
        """Component names outside the block stack (embeddings, the final norm, the unembed)."""
        return [k for k in self.components if not k.startswith("blocks.")]


class ComponentMapDeprecationWarning(UserWarning):
    """A retired component map was loaded; ``replacement`` names what to use instead."""


def _validate_schema_version(doc: dict[str, Any], source: str) -> int:
    """Enforce the readable window, telling too-new from too-old from malformed.

    The three cases need three different actions from the reader, so they get three messages: too new means upgrade
    interpretune, too old means the map predates the supported floor and needs re-publishing, malformed means the
    document does not say which schema it was written against at all.
    """
    version = doc.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(
            f"{source}: component map document lacks an integer `schema_version` (current: "
            f"{COMPONENT_MAP_SCHEMA_VERSION}). Every map must say which schema it was written against; one "
            "published without it can never be told apart from one written against an unknown revision."
        )
    if version > COMPONENT_MAP_SCHEMA_VERSION:
        raise ValueError(
            f"{source}: component map schema {version} was written by a newer interpretune than this one (this "
            f"build reads {COMPONENT_MAP_SCHEMA_MIN_READABLE}-{COMPONENT_MAP_SCHEMA_VERSION}). Upgrade interpretune "
            "to read it; an older reader cannot safely guess what a newer schema means, because a newer schema may "
            "carry keys that change which rows apply."
        )
    if version < COMPONENT_MAP_SCHEMA_MIN_READABLE:
        raise ValueError(
            f"{source}: component map schema {version} is older than the minimum readable schema "
            f"({COMPONENT_MAP_SCHEMA_MIN_READABLE}). Re-publish the map against a current schema."
        )
    return version


def _validate_facts(facts: Any, source: str) -> dict[str, Any]:
    """Type-check the declared facts; ignore unknown ones (they describe, they never select rows)."""
    if facts is None:
        return {}
    if not isinstance(facts, dict):
        raise ValueError(f"{source}: `facts` must be a mapping, got {type(facts).__name__}")
    for name, expected in KNOWN_FACTS.items():
        if name in facts and (
            not isinstance(facts[name], expected) or isinstance(facts[name], bool) != (expected is bool)
        ):
            raise ValueError(
                f"{source}: fact {name!r} must be {expected.__name__}, got {type(facts[name]).__name__} "
                f"({facts[name]!r}); a mistyped fact would otherwise degrade silently to the default."
            )
    return dict(facts)


def _from_document(doc: dict[str, Any], *, source: str) -> ComponentMap:
    if not isinstance(doc, dict):
        raise ValueError(f"{source}: a component map document must be a mapping, got {type(doc).__name__}")
    version = _validate_schema_version(doc, source)
    try:
        architecture = doc["architecture"]
        rows = doc["components"]
    except KeyError as e:
        raise ValueError(f"{source}: component map is missing the {e.args[0]!r} key") from None
    if not isinstance(rows, dict):
        raise ValueError(f"{source}: `components` must be a mapping of component path -> {{module, kind}}")
    # Only `module` and `kind` are read from a row; any other row key is ignored by contract (a later schema may
    # annotate rows, and an annotation an older reader does not use costs it nothing). The same holds for unknown
    # top-level keys. A key that would change which rows apply arrives with a schema bump, refused above.
    components = {name: ComponentEntry(module=row["module"], kind=row["kind"]) for name, row in rows.items()}
    cmap = ComponentMap(
        architecture=architecture,
        components=components,
        facts=_validate_facts(doc.get("facts"), source),
        source=source,
        schema_version=version,
        deprecated_since=doc.get("deprecated_since"),
        replacement=doc.get("replacement"),
    )
    if cmap.deprecated_since is not None:
        warnings.warn(
            f"{source}: the component map for {architecture!r} is deprecated since {cmap.deprecated_since}"
            + (f"; use {cmap.replacement!r}" if cmap.replacement else "")
            + ". It still loads until the readable floor moves; strict mode refuses it.",
            ComponentMapDeprecationWarning,
            stacklevel=3,
        )
    return cmap


def load_component_map_file(path: Path) -> ComponentMap:
    """Parse one YAML document into a :class:`ComponentMap`."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return _from_document(doc, source=str(path))


_REGISTRY: dict[str, ComponentMap] = {}
_BUNDLED_LOADED = False


def _load_bundled() -> None:
    global _BUNDLED_LOADED
    if _BUNDLED_LOADED:
        return
    for path in sorted(_DATA_DIR.glob("*.yaml")):
        cmap = load_component_map_file(path)
        _REGISTRY.setdefault(cmap.architecture, cmap)
    _BUNDLED_LOADED = True


def register(cmap: ComponentMap) -> None:
    """Register (or replace) the map for an architecture at runtime."""
    _load_bundled()
    _REGISTRY[cmap.architecture] = cmap


def component_map_for(architecture: str, *, strict: bool = False) -> ComponentMap:
    """The map for an HF architecture class name, or a ``KeyError`` naming what is known.

    ``strict`` refuses a retired map (one carrying ``deprecated_since``) instead of serving it with a warning, the
    same contract the alias table's strict mode applies to deprecated spellings.
    """
    _load_bundled()
    try:
        cmap = _REGISTRY[architecture]
    except KeyError:
        raise KeyError(
            f"no component map for architecture {architecture!r}; known: {sorted(_REGISTRY)}. Register one, or add a "
            "document under interpretune/analysis/points/data/."
        ) from None
    if strict and cmap.deprecated_since is not None:
        raise ValueError(
            f"the component map for {architecture!r} is deprecated since {cmap.deprecated_since}"
            + (f"; use {cmap.replacement!r}" if cmap.replacement else "")
            + " (refused in strict mode)"
        )
    return cmap


def known_architectures() -> list[str]:
    """Every architecture with a bundled or registered map."""
    _load_bundled()
    return sorted(_REGISTRY)


#: TransformerLens bridge component classes -> component kinds. Unlisted classes (virtual q/k/v splits, rotary,
#: routing) have no module of their own and are skipped: a map row must name a real module.
_TL_KINDS: dict[str, str] = {
    "EmbeddingBridge": "embed",
    "PosEmbedBridge": "embed",
    "BlockBridge": "block",
    "NormalizationBridge": "norm",
    "AttentionBridge": "attn",
    "JointQKVAttentionBridge": "attn",
    "JointGateUpMLPBridge": "mlp",
    "MLPBridge": "mlp",
    "GatedMLPBridge": "mlp",
    "LinearBridge": "linear",
    "UnembeddingBridge": "unembed",
}


def from_transformer_lens(adapter: Any, architecture: str) -> ComponentMap:
    """Derive a map from a TransformerLens bridge ``ArchitectureAdapter``'s ``component_mapping``.

    TransformerLens maintains one adapter per architecture and each names the HF module it wraps, so this
    is an INDEPENDENT source for the same facts the bundled documents carry: a test compares the two where
    both exist, which is what keeps a bundled row from drifting the way the five hand-written tables did.
    Components without a real module (virtual attention splits) are left out; ``kind`` follows the bridge
    class. The block list becomes ``blocks.{i}`` and its children ``blocks.{i}.<name>``.
    """
    rows: dict[str, ComponentEntry] = {}

    def walk(path: str, comp: Any, module_prefix: str) -> None:
        kind = _TL_KINDS.get(type(comp).__name__)
        name = getattr(comp, "name", None)
        if kind is None or not name:
            return
        if path == "blocks":
            module = f"{name}.{{i}}"
            key = "blocks.{i}"
        else:
            module = f"{module_prefix}.{name}" if module_prefix else name
            key = path.replace("blocks.", "blocks.{i}.", 1) if path.startswith("blocks.") else path
        rows[key] = ComponentEntry(module=module, kind=kind)
        for sub_name, sub in (getattr(comp, "submodules", None) or {}).items():
            walk(f"{path}.{sub_name}", sub, module)

    for top, comp in adapter.component_mapping.items():
        walk(top, comp, "")
    facts = {"sandwich_norms": "blocks.{i}.ln2_post" in rows}
    # the derived map carries the same version as the bundled documents, so the two sources stay comparable
    return ComponentMap(
        architecture=architecture,
        components=rows,
        facts=facts,
        source="transformer_lens",
        schema_version=COMPONENT_MAP_SCHEMA_VERSION,
    )
