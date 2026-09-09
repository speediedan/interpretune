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
COMPONENT_MAP_SCHEMA_VERSION = 2

#: The oldest schema this code READS. Published maps outlive the code that wrote them, so the reader accepts a
#: window, never a single version; raising this floor retires published maps and is a documented, deliberate act.
COMPONENT_MAP_SCHEMA_MIN_READABLE = 1

#: The declared `facts` vocabulary: name -> type. A known fact of the wrong type is refused; an unknown fact is
#: ignored, because facts describe the architecture and never select rows (an older reader loses nothing it
#: would have used). New facts are added here with their type.
KNOWN_FACTS: dict[str, type] = {"sandwich_norms": bool}

#: The bundled component kinds and whether their module returns a tuple whose element 0 is the tensor (the
#: static default; the nnsight backend measures this per model because transformers 5.x changed decoder blocks).
#: Schema 2 lets a document DECLARE further kinds with their own output rule (see :class:`KindSpec`), so this
#: table is the default vocabulary rather than a closed set: a map for an architecture nobody anticipated does
#: not need a release to name its sublayer.
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
class KindSpec:
    """What the resolver needs to know about a component kind: its output rule, and whether it writes to the
    residual (a sublayer gets ``_in`` / ``_out`` / ``_contribution`` points)."""

    output: str = "tensor"
    """``"tensor"`` when the module returns the tensor, ``"tuple0"`` when it returns a tuple whose element 0 is."""
    sublayer: bool = False

    @property
    def tuple_output(self) -> bool:
        """Whether the module returns a tuple whose element 0 is the tensor."""
        return self.output == "tuple0"


#: The bundled kinds as declarations, so a document's own ``kinds:`` block merges over the same shape.
BUNDLED_KINDS: dict[str, KindSpec] = {
    name: KindSpec("tuple0" if tuple_out else "tensor", sublayer=name in ("attn", "mlp"))
    for name, tuple_out in KIND_TUPLE_OUTPUT.items()
}

#: The declared property vocabulary (schema 2 ``properties:``; schema 1 spelled it ``facts:``): name ->
#: (rule, type). A DECLARED property's source is the document; a DERIVED one's source is the model, and a
#: document value is only a cross-check that the load-time validation refuses when it disagrees. Unknown
#: names are ignored: properties describe the architecture and never select rows, so an older reader loses
#: nothing it would have used.
KNOWN_PROPERTIES: dict[str, tuple[str, type]] = {
    "sandwich_norms": ("declared", bool),
    "sublayers": ("declared", list),
    "rmsnorm_offset": ("derived", bool),
}

#: Keys that change which rows apply or how a row resolves. A reader that skipped one would resolve the wrong
#: rows silently, so they exist only from schema 2 and a schema-1 document carrying one is refused.
APPLICABILITY_KEYS = ("kinds", "stacks")
ROW_APPLICABILITY_KEYS = ("layers",)

#: The primary block stack's name: its points keep the bare ``blocks.{i}`` spelling, so every string in every
#: published map and cached artifact keeps its meaning. Additional stacks prefix the point string.
PRIMARY_STACK = "blocks"


@dataclass(frozen=True)
class LayerSet:
    """Which layers a row covers: every layer (``None`` on the row), a list, an inclusive range or a parity."""

    spec: tuple[int, ...] | str
    form: str  # "list" | "range" | "parity"

    @classmethod
    def parse(cls, raw: Any, where: str) -> "LayerSet":
        """Read a ``layers`` value (a list of ints, ``"a-b"``, ``"odd"`` or ``"even"``), refusing anything else."""
        if isinstance(raw, str):
            if raw in ("odd", "even"):
                return cls(raw, "parity")
            lo, sep, hi = raw.partition("-")
            if sep and lo.strip().isdigit() and hi.strip().isdigit():
                return cls((int(lo), int(hi)), "range")
            raise ValueError(
                f"{where}: `layers` must be a list of ints, an inclusive range 'a-b', 'odd' or 'even'; got {raw!r}"
            )
        if isinstance(raw, list) and raw and all(isinstance(i, int) and not isinstance(i, bool) for i in raw):
            return cls(tuple(sorted(set(raw))), "list")
        raise ValueError(
            f"{where}: `layers` must be a list of ints, an inclusive range 'a-b', 'odd' or 'even'; got {raw!r}"
        )

    def covers(self, layer: int) -> bool:
        """Whether the row applies at this layer."""
        if isinstance(self.spec, str):
            return (layer % 2 == 1) if self.spec == "odd" else (layer % 2 == 0)
        if self.form == "range":
            lo, hi = self.spec[0], self.spec[1]
            return lo <= layer <= hi
        return layer in self.spec

    def describe(self) -> str:
        """The predicate in words, for a refusal message."""
        if isinstance(self.spec, str):
            return f"{self.spec} layers"
        if self.form == "range":
            return f"layers {self.spec[0]}-{self.spec[1]}"
        return f"layers {list(self.spec)}"


@dataclass(frozen=True)
class ComponentEntry:
    """One row: a component path (``blocks.{i}.ln2``, ``unembed``) -> module path template + kind."""

    module: str
    kind: str
    layers: LayerSet | None = None
    """Which layers this row covers; ``None`` is every layer.

    A heterogeneous stack (attention on odd layers, a state-space mixer on even ones) needs this, and a reader that
    ignored it would resolve a plausible module path that does not exist.
    """


def _validate_kind(kind: str, kinds: dict[str, KindSpec], where: str) -> None:
    if kind not in kinds:
        raise ValueError(
            f"{where}: unknown component kind {kind!r}; expected one of {sorted(kinds)}. A kind decides the slot "
            "rule and whether the module's output is a tuple, so it is refused rather than defaulted: a guess would "
            "resolve to a plausible wrong tensor. A schema-2 document may declare a new kind under `kinds:` with "
            "its output rule."
        )


@dataclass(frozen=True)
class ComponentMap:
    """The component -> module table for one architecture, plus the structural facts the resolver needs."""

    architecture: str
    components: dict[str, ComponentEntry]
    properties: dict[str, Any] = field(default_factory=dict)
    """The typed per-architecture properties (schema 1 spelled the key ``facts``)."""
    source: str = "bundled"
    kinds: dict[str, KindSpec] = field(default_factory=lambda: dict(BUNDLED_KINDS))
    """The bundled kinds merged with the document's own declarations."""
    stacks: dict[str, str] = field(default_factory=dict)
    """Additional block stacks by name -> module template (the primary stack is implied and stays bare)."""
    schema_version: int = COMPONENT_MAP_SCHEMA_VERSION
    """The document schema this map was written against.

    Required on every document: a map published without
    it is permanently unversioned and can never be told apart from one written against an unknown revision.
    """
    deprecated_since: str | None = None
    """Set when the map is retired: loading it warns, strict mode refuses, and ``replacement`` names what to use."""
    replacement: str | None = None

    @property
    def facts(self) -> dict[str, Any]:
        """The schema-1 name for :attr:`properties`, kept for readers that compare maps."""
        return self.properties

    @property
    def sandwich_norms(self) -> bool:
        """Whether the block has post-sublayer norms (``ln1_post`` / ``ln2_post``), which is what decides where a
        sublayer's contribution to the residual is read."""
        return bool(self.properties.get("sandwich_norms", "blocks.{i}.ln2_post" in self.components))

    @property
    def sublayers(self) -> tuple[str, ...]:
        """The order in which sublayer kinds write to the residual; ``resid_mid`` presumes the first write is
        followed by a second.

        Declared, else the classic ``(attn, mlp)``.
        """
        declared = self.properties.get("sublayers")
        return tuple(declared) if declared else ("attn", "mlp")

    def tuple_output(self, kind: str) -> bool:
        """Whether a kind's module returns a tuple whose element 0 is the tensor, per this document's kinds."""
        return self.kinds[kind].tuple_output

    @staticmethod
    def _key(component: str, layer: int | None, stack: str | None) -> str:
        prefix = "" if stack in (None, PRIMARY_STACK) else f"{stack}."
        if layer is None:
            return component
        return f"{prefix}blocks.{{i}}.{component}" if component else f"{prefix}blocks.{{i}}"

    def _entry(self, component: str, layer: int | None, stack: str | None) -> ComponentEntry | None:
        entry = self.components.get(self._key(component, layer, stack))
        if entry is None or layer is None or entry.layers is None or entry.layers.covers(layer):
            return entry
        return None

    def module_for(self, component: str, layer: int | None, stack: str | None = None) -> str | None:
        """The concrete module path for a block-relative or global component, or ``None`` if unmapped (including a
        layer the row does not cover)."""
        entry = self._entry(component, layer, stack)
        if entry is None:
            return None
        return entry.module.replace("{i}", str(layer)) if layer is not None else entry.module

    def kind_of(self, component: str, layer: int | None, stack: str | None = None) -> str | None:
        """The component's kind, or ``None`` if unmapped (including a layer the row does not cover)."""
        entry = self._entry(component, layer, stack)
        return None if entry is None else entry.kind

    def unmapped_reason(self, component: str, layer: int | None, stack: str | None = None) -> str | None:
        """Why a component is unmapped when a row for it EXISTS but does not cover this layer, else ``None``."""
        entry = self.components.get(self._key(component, layer, stack))
        if entry is None or layer is None or entry.layers is None or entry.layers.covers(layer):
            return None
        return f"the {component or 'block'!r} row covers {entry.layers.describe()}, not layer {layer}"

    def block_components(self, stack: str | None = None) -> list[str]:
        """Block-relative component names present in one stack (``""`` for the block itself); the primary stack by
        default, so consumers that predate stacks see exactly what they saw."""
        head = self._key("", 0, stack)
        out = []
        for key in self.components:
            if key == head:
                out.append("")
            elif key.startswith(head + "."):
                out.append(key[len(head) + 1 :])
        return out

    def global_components(self) -> list[str]:
        """Component names outside every block stack (embeddings, the final norm, the unembed, a projector)."""
        return [k for k in self.components if "blocks.{i}" not in k]

    def stack_names(self) -> tuple[str, ...]:
        """Every stack with an indexed row: the primary one first, then the declared ones."""
        return (PRIMARY_STACK, *sorted(self.stacks))


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


def _validate_properties(raw: Any, source: str, *, key: str) -> dict[str, Any]:
    """Type-check the declared properties; ignore unknown ones (they describe, they never select rows)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: `{key}` must be a mapping, got {type(raw).__name__}")
    for name, (_rule, expected) in KNOWN_PROPERTIES.items():
        if name not in raw:
            continue
        value = raw[name]
        ok = isinstance(value, expected) and (isinstance(value, bool) == (expected is bool))
        if not ok:
            raise ValueError(
                f"{source}: property {name!r} must be {expected.__name__}, got {type(value).__name__} ({value!r}); a "
                "mistyped property would otherwise degrade silently to the default."
            )
    sublayers = raw.get("sublayers")
    if sublayers is not None and not all(isinstance(x, str) and x for x in sublayers):
        raise ValueError(f"{source}: property 'sublayers' must be a list of kind names, got {sublayers!r}")
    return dict(raw)


def _validate_kinds(raw: Any, source: str) -> dict[str, KindSpec]:
    """The document's own kind declarations merged over the bundled ones."""
    kinds = dict(BUNDLED_KINDS)
    if raw is None:
        return kinds
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: `kinds` must be a mapping of kind name -> {{output, sublayer}}")
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{source}: kind {name!r} must be a mapping with `output` (tensor|tuple0) and `sublayer`")
        output = spec.get("output", "tensor")
        if output not in ("tensor", "tuple0"):
            raise ValueError(f"{source}: kind {name!r} has output {output!r}; expected 'tensor' or 'tuple0'")
        sublayer = spec.get("sublayer", False)
        if not isinstance(sublayer, bool):
            raise ValueError(f"{source}: kind {name!r} has a non-boolean `sublayer`: {sublayer!r}")
        kinds[name] = KindSpec(output, sublayer)
    return kinds


def _validate_stacks(raw: Any, source: str) -> dict[str, str]:
    """Additional block stacks: name -> module template carrying ``{i}``."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: `stacks` must be a mapping of stack name -> {{module}}")
    stacks: dict[str, str] = {}
    for name, spec in raw.items():
        if name == PRIMARY_STACK or not isinstance(name, str) or not name.isidentifier() or not name.islower():
            raise ValueError(
                f"{source}: stack {name!r} is not a valid stack name (a lowercase identifier other than "
                f"{PRIMARY_STACK!r}, which is the implied primary stack)"
            )
        module = spec.get("module") if isinstance(spec, dict) else None
        if not isinstance(module, str) or "{i}" not in module:
            raise ValueError(f"{source}: stack {name!r} needs a `module` template containing '{{i}}'")
        stacks[name] = module
    return stacks


def _validate_rows(rows: Any, source: str, kinds: dict[str, KindSpec], stacks: dict[str, str], version: int):
    if not isinstance(rows, dict):
        raise ValueError(f"{source}: `components` must be a mapping of component path -> {{module, kind}}")
    components: dict[str, ComponentEntry] = {}
    for name, row in rows.items():
        if not isinstance(row, dict) or "module" not in row or "kind" not in row:
            raise ValueError(f"{source}: component {name!r} must be a mapping with `module` and `kind`")
        where = f"{source}: component {name!r}"
        _validate_kind(row["kind"], kinds, where)
        layers = None
        if "layers" in row:
            if version < 2:
                raise ValueError(
                    f"{where}: `layers` is a schema-2 key (a reader that skipped it would resolve the row for every "
                    "layer); declare schema_version: 2"
                )
            layers = LayerSet.parse(row["layers"], where)
        head = name.split(".blocks.{i}", 1)[0] if ".blocks.{i}" in name else None
        if head is not None and head not in stacks:
            raise ValueError(
                f"{where}: names the block stack {head!r}, which the document does not declare under `stacks:`"
            )
        # Only `module`, `kind` and `layers` are read from a row; any other row key is ignored by contract (a
        # later schema may annotate rows, and an annotation an older reader does not use costs it nothing).
        components[name] = ComponentEntry(module=row["module"], kind=row["kind"], layers=layers)
    return components


def _from_document(doc: dict[str, Any], *, source: str) -> ComponentMap:
    if not isinstance(doc, dict):
        raise ValueError(f"{source}: a component map document must be a mapping, got {type(doc).__name__}")
    version = _validate_schema_version(doc, source)
    try:
        architecture = doc["architecture"]
        rows = doc["components"]
    except KeyError as e:
        raise ValueError(f"{source}: component map is missing the {e.args[0]!r} key") from None
    if version < 2:
        for key in APPLICABILITY_KEYS:
            if key in doc:
                raise ValueError(
                    f"{source}: `{key}` is a schema-2 key (a reader that skipped it would resolve the wrong rows); "
                    "declare schema_version: 2"
                )
        if "properties" in doc and "facts" in doc:
            raise ValueError(f"{source}: a document carries `facts` (schema 1) or `properties` (schema 2), not both")
        properties = _validate_properties(doc.get("facts"), source, key="facts")
    else:
        if "facts" in doc:
            raise ValueError(
                f"{source}: `facts` was renamed to `properties` in schema 2; move the keys under `properties:`"
            )
        properties = _validate_properties(doc.get("properties"), source, key="properties")
    kinds = _validate_kinds(doc.get("kinds"), source)
    stacks = _validate_stacks(doc.get("stacks"), source)
    components = _validate_rows(rows, source, kinds, stacks, version)
    # Only the keys named above are read; an unknown top-level key is ignored by contract. A key that would change
    # which rows apply arrives with a schema bump, refused by the window check above.
    cmap = ComponentMap(
        architecture=architecture,
        components=components,
        properties=properties,
        source=source,
        schema_version=version,
        deprecated_since=doc.get("deprecated_since"),
        replacement=doc.get("replacement"),
        kinds=kinds,
        stacks=stacks,
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


#: TransformerLens bridge component classes -> component kinds, for the classes that wrap an HF module the
#: vocabulary addresses. A component without a module of its own (a virtual q/k/v split) has no ``name`` and is
#: skipped: a map row must name a real module.
_TL_KINDS: dict[str, str] = {
    "EmbeddingBridge": "embed",
    "PosEmbedBridge": "embed",
    "BlockBridge": "block",
    "NormalizationBridge": "norm",
    "RMSNormalizationBridge": "norm",
    "AttentionBridge": "attn",
    "JointQKVAttentionBridge": "attn",
    "PositionEmbeddingsAttentionBridge": "attn",
    "JointGateUpMLPBridge": "mlp",
    "MLPBridge": "mlp",
    "GatedMLPBridge": "mlp",
    "LinearBridge": "linear",
    "UnembeddingBridge": "unembed",
}

#: Bridge classes that wrap a module the vocabulary has no point for, left out of a derived map on purpose: a
#: rotary table carries no activation, and TransformerLens bridges a vision tower as one opaque component where
#: the multimodal document addresses its blocks. A class in NEITHER table is refused by name rather than
#: skipped. Skipping is how two RMSNorm-era classes were once dropped, so four bundled maps were compared with
#: TransformerLens on seven rows of fourteen and the oracle reported agreement over the whole.
_TL_UNADDRESSED: frozenset[str] = frozenset(
    {
        "RotaryEmbeddingBridge",
        "SiglipVisionEncoderBridge",
        "SiglipVisionEncoderLayerBridge",
        "VisionProjectionBridge",
        "GeneralizedComponent",
    }
)


def from_transformer_lens(adapter: Any, architecture: str) -> ComponentMap:
    """Derive a map from a TransformerLens bridge ``ArchitectureAdapter``'s ``component_mapping``.

    TransformerLens maintains one adapter per architecture and each names the HF module it wraps, so this
    is an INDEPENDENT source for the same facts the bundled documents carry: a test compares the two where
    both exist, which is what keeps a bundled row from drifting the way the five hand-written tables did.
    Components without a real module (virtual attention splits) are left out; ``kind`` follows the bridge
    class. The block list becomes ``blocks.{i}`` and its children ``blocks.{i}.<name>``. A bridge class known
    neither as a kind nor as deliberately unaddressed raises, naming both tables, because a silently skipped
    class narrows every comparison built on the result.
    """
    rows: dict[str, ComponentEntry] = {}

    def walk(path: str, comp: Any, module_prefix: str) -> None:
        cls_name = type(comp).__name__
        if cls_name in _TL_UNADDRESSED:
            return
        kind = _TL_KINDS.get(cls_name)
        if kind is None:
            raise ValueError(
                f"from_transformer_lens: {cls_name!r} at {path!r} is a bridge class this derivation does not know; "
                f"add it to _TL_KINDS with its kind, or to _TL_UNADDRESSED with a reason. Known kinds: "
                f"{sorted(_TL_KINDS)}; unaddressed: {sorted(_TL_UNADDRESSED)}"
            )
        name = getattr(comp, "name", None)
        if not name:
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
    properties = {"sandwich_norms": "blocks.{i}.ln2_post" in rows}
    # The derived map uses no schema-2 key (no declared kinds, stacks or layer predicates), so it is a schema-1
    # document, and the bundled documents it is compared against are schema 1 unless they need more.
    return ComponentMap(
        architecture=architecture,
        components=rows,
        properties=properties,
        source="transformer_lens",
        schema_version=1,
    )
