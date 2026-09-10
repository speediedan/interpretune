"""Backend capability enums, module capability aggregation, and the named-backend registry.

Part of the sanctioned :mod:`interpretune.analysis.backends` seam that op implementations (bundled,
local, or hub) may import. Ops should ask for capabilities rather than branching on backend or
adapter class names.
"""

from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from interpretune.analysis.backends.protocols import AnalysisBackend, ModelBackend


class ModelBackendCapability(Enum):
    """The gated METHOD GROUPS of a model backend: one member per ``Supports*`` protocol, no more.

    Ops and the dispatcher query ``backend.capabilities`` before calling an optional method group
    (``require_backend_capability``). A member here answers "does this backend implement that surface";
    it never answers "which configurations of the surface does it support". Those are a different
    kind of fact and carry a different shape: a typed support record on the protocol that owns the
    methods (:class:`InterventionSupport` on ``SupportsIntervention``, :class:`LatentModelSupport` on
    ``SupportsLatentModels``). Keeping the two apart is what lets a backend truthfully claim a surface
    while declaring exactly which of its modes it can honour, and lets a gate refuse the rest by name.
    """

    LATENT_MODELS = "latent_models"
    """``SupportsLatentModels``: ``fwd_w_cache_and_latent_models``, ``fwd_w_hooks_and_latent_models``,
    ``fwd_w_hooks_batched``."""

    GRADIENTS = "gradients"
    """``SupportsGradients``: forward + backward with gradient caching."""

    ACTIVATION_INTERVENTION = "activation_intervention"
    """``SupportsIntervention``: baseline-vs-intervention paired execution (``fwd_w_intervention``)."""


class PositionScope(str, Enum):
    """Which positions an intervention edits.

    A ``str`` enum so a spec built from YAML or a notebook can carry the plain string and still
    compare equal to the member.

    **Both scopes are legitimate operations, not a correct one and a broken one.** Steering the final
    token is the right shape for "change the next prediction"; steering every position is the right
    shape for "make the model read the whole prompt differently". Interpretune previously had a name
    for only the first, which is what made a backend implementing the second look like a defect
    rather than like a capability we could not express.
    """

    LAST_TOKEN = "last_token"
    ALL_POSITIONS = "all_positions"


class InterventionMode(str, Enum):
    """How an intervention combines its tensor with the activation it targets.

    A ``str`` enum for the same reason as :class:`PositionScope`. The mode is the second axis of the
    intervention contract (scope is the first): a backend can implement ``fwd_w_intervention`` and still
    be unable to express most modes. Every mode but ``add`` reads the CURRENT activation, and the
    additional thing they need is that the READ HAPPENS DURING THE FORWARD PASS: ``patch`` and ``reject``
    compute coordinates from the activation itself, so no parameter fixed when the spec was built can
    stand in for them.

    The requirement is therefore sharper than "can observe the activation": **a steering surface whose
    parameters are all static scalars satisfies that and still expresses none of these modes.** What a
    backend must support is a parameter COMPUTED AT FORWARD TIME. Stated as a capability rather than by
    naming a backend, because core must not know which adapters exist, and because a rule about what an
    implementation needs stays true when the next one appears.

    Such a backend declares ``InterventionSupport(modes={ADD})``; the declaration, not the
    implementation, is what the dispatcher consults, so the refusal is by name rather than by trial.

    A mode a backend has not declared is refused by
    :func:`~interpretune.analysis.backends.interventions.require_intervention_mode` rather than applied
    as a different mode, since every mode returns plausible logits and the substitution is undetectable
    from the result.
    """

    REPLACE = "replace"
    ADD = "add"
    PATCH = "patch"
    PROJECT = "project"
    REJECT = "reject"


@dataclass(frozen=True)
class InterventionSupport:
    """Which configurations of ``INTERVENTION`` a backend can honour.

    Declaring the capability without one of these is a contract violation, not a legacy default: the absence of a
    declaration used to mean "assume last-token", which is exactly the silent narrowing the scope field was introduced
    to remove.
    """

    position_scopes: frozenset[PositionScope]
    modes: frozenset[InterventionMode]

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_scopes", frozenset(PositionScope(s) for s in self.position_scopes))
        object.__setattr__(self, "modes", frozenset(InterventionMode(m) for m in self.modes))
        if not self.position_scopes:
            raise ValueError("InterventionSupport must declare at least one position scope")
        if not self.modes:
            raise ValueError("InterventionSupport must declare at least one intervention mode")

    @classmethod
    def every(cls) -> InterventionSupport:
        """Every scope and every mode: for a backend whose hook can compute its edit from the activation it
        sees, during the pass.

        Seeing the activation is not the criterion, which is worth stating because it reads like one: a
        steering surface can receive the activation and still express only ``add``, if every parameter it
        takes was fixed before the pass began. See :class:`InterventionMode`.
        """
        return cls(position_scopes=frozenset(PositionScope), modes=frozenset(InterventionMode))


@dataclass(frozen=True)
class LatentModelSupport:
    """How ``LATENT_MODELS`` runs on this backend.

    ``batched_hooks`` says whether ``fwd_w_hooks_batched`` fuses its hook configs into one execution
    (nnsight's multi-invoke) or loops. It lives here rather than in :class:`ModelBackendCapability` because it
    is a property of HOW a method in that group runs, not a surface of its own: every latent-models
    backend implements the method, and a sequential loop is a valid implementation.
    """

    batched_hooks: bool = False


@dataclass(frozen=True)
class CaptureSupport:
    """Which vocabulary points a backend can capture on the model it wraps, as a declaration a case can check.

    Capture is a base method every model backend has, so it is not a :class:`ModelBackendCapability` member: that enum
    answers "is the surface implemented at all", and a backend that captures 181 of 298 points implements it. What
    varies is WHICH points, so the shape is a support record beside :class:`InterventionSupport`, keyed by the
    vocabulary's layer-free base spellings (``ln2.hook_out``, ``hook_resid_pre``, ``unembed.hook_in``) so one
    declaration covers every layer. ``uncapturable`` carries the reason per base, because a point a backend cannot
    capture must be refused by name with that reason rather than returned as a cache that is silently short.

    **Valid for one model instance as it stands when asked.** The record is derived from the backend AND the model
    it wraps, and a wrapper's hooks change with what is attached to it (a HookedSAETransformer with a latent model
    attached exposes points the bare HookedTransformer does not), so a backend provides ``capture_support(model)``
    as a method rather than a property, ``get_module_capabilities`` recomputes it on every call, and a consumer that
    caches one must re-query after attaching or removing a latent model or swapping the wrapper. A cached record is
    the declaration-versus-delivery gap one level out.
    """

    capturable: frozenset[str]
    uncapturable: dict[str, str]
    n_layers: int
    architecture: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "capturable", frozenset(self.capturable))
        object.__setattr__(self, "uncapturable", dict(self.uncapturable))
        overlap = self.capturable & set(self.uncapturable)
        if overlap:
            raise ValueError(f"a point cannot be both capturable and uncapturable: {sorted(overlap)}")
        if not self.capturable:
            raise ValueError("CaptureSupport must declare at least one capturable point")
        # A declaration decides EVERY point of the architecture's inventory, or it is refused by name. Otherwise the
        # fraction it reports is self-referential (a record that omits points shrinks its own denominator, so
        # omission reads as completeness), and the understating direction is the one no capture case catches: an
        # omitted point is neither captured nor refused, only unknown.
        from interpretune.analysis.points import component_map_for
        from interpretune.analysis.points.inventory import inventory

        expected = set(inventory(component_map_for(self.architecture)))
        declared = self.capturable | set(self.uncapturable)
        undecided = sorted(expected - declared)
        if undecided:
            raise ValueError(
                f"the capture declaration for {self.architecture} decides {len(declared)} of {len(expected)} inventory "
                f"points; undecided (declare each capturable or uncapturable with a reason): {undecided}"
            )
        foreign = sorted(declared - expected)
        if foreign:
            raise ValueError(
                f"the capture declaration for {self.architecture} names points outside the architecture's inventory: "
                f"{foreign}"
            )

    @property
    def inventory_size(self) -> int:
        """How many base points the architecture's inventory has: the denominator, equal to the declaration's size
        by the construction invariant above."""
        return len(self.capturable) + len(self.uncapturable)

    def refusal(self, name: str) -> str | None:
        """Why ``name`` cannot be captured here, or ``None`` when it can.

        Parsed through the vocabulary: a layer beyond the model is refused as such, an SAE sub-hook is judged by the
        point it hangs off, a spelling outside the vocabulary is refused as unknown, and a vocabulary point the
        architecture does not define is refused as the architecture's fact rather than the backend's, since the
        declaration decides every point of the inventory and so cannot be silent about one.
        """
        from interpretune.analysis.points.vocabulary import UnknownPointError, parse

        try:
            point = parse(name)
        except UnknownPointError as exc:
            return str(exc)
        if point.layer is not None and point.layer >= self.n_layers:
            return f"{name!r} names layer {point.layer}, and this model has {self.n_layers} blocks"
        from interpretune.analysis.points.vocabulary import declaration_key

        base = declaration_key(name)
        if base in self.capturable:
            return None
        reason = self.uncapturable.get(base)
        if reason is not None:
            return f"{name!r} cannot be captured here: {reason}"
        # A declaration decides every point of the architecture's inventory, so a base that is in neither set is not
        # a gap in the declaration: the architecture defines no such point. That is the resolver's refusal, an
        # architecture-scoped fact, and the record says so rather than presenting a backend gap it does not have.
        return (
            f"{name!r} is not defined on {self.architecture}: the vocabulary resolves {base!r} to no tensor position"
            " in this architecture, so no backend could capture it; this is the resolver's refusal, not a gap in the"
            " backend's capture declaration"
        )

    def can_capture(self, name: str) -> bool:
        """Whether ``name`` is capturable here."""
        return self.refusal(name) is None

    def describe(self) -> str:
        """One line for a report: the fraction and the gaps by name."""
        gaps = ", ".join(sorted(self.uncapturable)) or "-"
        return (
            f"captures {len(self.capturable)} of {self.inventory_size} base points on {self.architecture} "
            f"({self.n_layers} blocks); cannot capture: {gaps}"
        )


class AnalysisBackendCapability(Enum):
    """Capabilities exposed by analysis adapters/backends rather than model execution backends."""

    ATTRIBUTION_GRAPH = "attribution_graph"
    """Module exposes attribution graph analysis support via an attached analysis backend."""

    FEATURE_INTERVENTION = "feature_intervention"
    """Module exposes feature intervention support via an attached analysis backend."""


@dataclass(frozen=True)
class FeatureInterventionSupport:
    """Which configurations of the feature-intervention surface an analysis backend honours.

    The settings ``resolve_feature_intervention_settings`` reads are the configuration space; the record is what
    lets a caller's settings be refused by name at the seam instead of at the first tensor op that disagrees.
    """

    value_sources: frozenset[str]
    """Accepted ``value_source`` spellings (``"constant"`` requires an explicit value)."""
    constrainable_layers: bool = True
    """Whether ``constrained_layers`` can restrict the intervention to a layer subset."""
    returns_activations: bool = False
    """Whether the intervened activations can be returned beside the logits."""

    def __post_init__(self) -> None:
        if not self.value_sources:
            raise ValueError("a feature-intervention support record must accept at least one value source")

    def refusal(self, settings: Mapping[str, Any]) -> str | None:
        """Why ``settings`` cannot be honoured here, or ``None`` when every configuration they name is declared."""
        source = settings.get("value_source")
        if source not in self.value_sources:
            return f"value_source {source!r} is not honoured here; declared: {sorted(self.value_sources)}"
        if source == "constant" and settings.get("value") is None:
            return "value_source 'constant' needs an explicit intervention value, and none was given"
        if settings.get("constrained_layers") is not None and not self.constrainable_layers:
            return (
                "constrained_layers was given, and this backend cannot restrict a feature intervention to a"
                " layer subset"
            )
        if settings.get("return_activations") and not self.returns_activations:
            return "return_activations was requested, and this backend cannot return the intervened activations"
        return None

    def describe(self) -> str:
        """One line for a card or a report."""
        return (
            f"value sources {sorted(self.value_sources)}, constrainable layers {self.constrainable_layers}, "
            f"returns activations {self.returns_activations}"
        )


@dataclass(frozen=True)
class AttributionGraphSupport:
    """What attribution-graph construction requires of the model it runs on, checked before construction.

    ``requires_own_eager_attention`` is the measured requirement: circuit-tracer resolves gemma's attention-pattern
    location through nnsight's source tracing of the function bound at the attention call site, so the modeling
    module's own ``eager_attention_forward`` must be what is bound there. A TransformerLens bridge of any llama,
    qwen or gemma model replaces that function in both gemma modeling modules for the rest of the process, with the
    config still reading ``eager``; the failure that produced is an ``AttributeError`` deep in circuit-tracer, so the
    check happens here, by provenance, where the name of the foreign function is still available.
    """

    requires_own_eager_attention: bool = True

    def refusal(self, model: Any) -> str | None:
        """Why a graph cannot be built on ``model`` here, or ``None`` when its attention is what construction
        needs."""
        if not self.requires_own_eager_attention:
            return None
        import inspect

        inner = model
        for attr in ("model", "_model", "hf_model"):
            candidate = getattr(inner, attr, None)
            if candidate is not None and hasattr(candidate, "config"):
                inner = candidate
                break
        modeling = inspect.getmodule(type(inner))
        fn = getattr(modeling, "eager_attention_forward", None) if modeling is not None else None
        if fn is None or modeling is None:
            return None  # an architecture without a module-level eager attention resolves its locations another way
        modeling_name = modeling.__name__
        impl = getattr(getattr(inner, "config", None), "_attn_implementation", None)
        if impl not in (None, "eager"):
            return f"attribution graphs need the eager attention implementation and the model is configured as {impl!r}"
        if getattr(fn, "__module__", None) != modeling_name:
            return (
                f"{modeling_name}.eager_attention_forward is {getattr(fn, '__module__', '?')}."
                f"{getattr(fn, '__qualname__', '?')}, not the modeling module's own: another library replaced it for"
                " this process, and the attention-pattern location is resolved from that function's source. Restore"
                " the original before building a graph (a TransformerLens bridge of a llama, qwen or gemma model is"
                " the known cause)"
            )
        return None

    def describe(self) -> str:
        """One line for a card or a report."""
        return (
            "requires the modeling module's own eager attention"
            if self.requires_own_eager_attention
            else "no model requirement"
        )


Capability: TypeAlias = ModelBackendCapability | AnalysisBackendCapability


@dataclass(frozen=True)
class ModuleCapabilities:
    """Execution and analysis capabilities exposed by a module, with each surface's support record.

    Each support record is present iff its surface is declared, on either level: ``intervention`` with
    ``ACTIVATION_INTERVENTION``, ``latent_models`` with ``LATENT_MODELS``, ``attribution_graph`` with
    ``ATTRIBUTION_GRAPH``, ``feature_intervention`` with ``FEATURE_INTERVENTION``. The constructor enforces that, so
    a consumer rendering this (the adapter card, ``adapter_info``, a conformance report) can rely on the record
    being there when the surface is.
    """

    model: frozenset[ModelBackendCapability]
    analysis: frozenset[AnalysisBackendCapability]
    intervention: InterventionSupport | None = None
    latent_models: LatentModelSupport | None = None
    capture: CaptureSupport | None = None
    """What the attached model backend declares it can capture on this module's model; ``None`` only when no model
    backend is attached or the backend predates the declaration (a conformance case fails the latter by name)."""
    attribution_graph: AttributionGraphSupport | None = None
    feature_intervention: FeatureInterventionSupport | None = None

    def __post_init__(self) -> None:
        for capability, record, name in (
            (ModelBackendCapability.ACTIVATION_INTERVENTION, self.intervention, "intervention"),
            (ModelBackendCapability.LATENT_MODELS, self.latent_models, "latent_models"),
            (AnalysisBackendCapability.ATTRIBUTION_GRAPH, self.attribution_graph, "attribution_graph"),
            (AnalysisBackendCapability.FEATURE_INTERVENTION, self.feature_intervention, "feature_intervention"),
        ):
            declared = capability in self.model or capability in self.analysis
            if declared and record is None:
                raise ValueError(
                    f"{capability.name} is declared but no {name} support record accompanies it; a backend "
                    "claiming the surface must say which configurations of it are supported"
                )
            if record is not None and not declared:
                raise ValueError(f"a {name} support record is present but {capability.name} is not declared")

    @property
    def all(self) -> frozenset[Capability]:
        """Model and analysis capabilities as one set, for checks that do not care which layer supplies them."""
        return frozenset({*self.model, *self.analysis})

    @property
    def values(self) -> frozenset[str]:
        """The capability names as plain strings, for logging and serialization."""
        return frozenset(cap.value for cap in self.all)

    def supports(self, capability: Capability) -> bool:
        """Whether ``capability`` is present, checked against the set its TYPE identifies.

        A model capability is looked up only among model capabilities and an analysis capability only among analysis
        ones, so the two namespaces cannot satisfy each other by coincidence.
        """
        if isinstance(capability, ModelBackendCapability):
            return capability in self.model
        return capability in self.analysis


#: Spellings that once named a capability and no longer do, each with the spelling that replaced it. Refused by
#: name rather than translated: a caller carrying the old spelling has a manifest or a config to update, and a
#: silent translation would leave it carrying a name nothing else in the vocabulary recognizes.
_RETIRED_CAPABILITY_SPELLINGS: dict[str, str] = {
    "intervention": ModelBackendCapability.ACTIVATION_INTERVENTION.value,
    "attribution": AnalysisBackendCapability.ATTRIBUTION_GRAPH.value,
}


def normalize_backend_capability(capability: Any) -> Capability:
    """Normalize capability-like values to the local execution or analysis capability enums.

    Accepts a member of either enum, a member's value, or a dotted spelling whose last segment is a member name
    (``"ModelBackendCapability.GRADIENTS"``). A retired spelling is refused by name with its replacement; an unknown one
    is refused with both vocabularies listed.
    """
    if isinstance(capability, (ModelBackendCapability, AnalysisBackendCapability)):
        return capability

    raw_value = getattr(capability, "value", capability)
    spelling = str(raw_value)
    candidate = spelling.split(".")[-1].lower() if "." in spelling else spelling
    if candidate in _RETIRED_CAPABILITY_SPELLINGS:
        raise ValueError(
            f"{spelling!r} is a retired capability spelling; the surface it named is now"
            f" {_RETIRED_CAPABILITY_SPELLINGS[candidate]!r}. Update the declaration rather than relying on a"
            " translation."
        )
    for enum_cls in (ModelBackendCapability, AnalysisBackendCapability):
        try:
            return enum_cls(candidate)
        except ValueError:
            continue
    raise ValueError(
        f"{spelling!r} is not a capability: model-level spellings are {[m.value for m in ModelBackendCapability]},"
        f" analysis-level spellings are {[m.value for m in AnalysisBackendCapability]}"
    )


def get_model_backend(module: Any) -> ModelBackend | None:
    """Return the module's model backend while avoiding mock-created private attrs."""

    module_dict = getattr(module, "__dict__", None)
    backend = module_dict.get("_model_backend") if isinstance(module_dict, dict) else None
    if backend is None and hasattr(module, "model_backend"):
        try:
            backend = module.model_backend
        except (AssertionError, AttributeError):
            backend = None
    return backend


def get_analysis_backend(module: Any) -> AnalysisBackend | None:
    """Return the module's analysis backend, or None when it has none.

    Reads ``__dict__`` directly before touching the ``analysis_backend`` property, because the property
    may assert on a module that is not fully set up -- and "not set up yet" must answer None here rather
    than raising out of a capability probe.
    """
    module_dict = getattr(module, "__dict__", None)
    backend = module_dict.get("_analysis_backend") if isinstance(module_dict, dict) else None
    if backend is None and hasattr(module, "analysis_backend"):
        try:
            backend = module.analysis_backend
        except (AssertionError, AttributeError):
            backend = None
    return backend


# Named analysis backends (hydration seam): names are the PORTABLE reference artifacts use — an
# it_artifact.json envelope can only carry a backend NAME (instances are not wire-format). Backends
# register at import; resolve_analysis_backend lazily imports interpretune.analysis.backends.<name>
# on a miss before failing, keeping resolution extensible without eager imports.
ANALYSIS_BACKEND_REGISTRY: dict[str, "AnalysisBackend"] = {}


def register_analysis_backend(name: str, backend: "AnalysisBackend") -> None:
    """Register a named analysis backend (idempotent for the same object)."""
    ANALYSIS_BACKEND_REGISTRY[name] = backend


# Bundled backend NAME -> the module whose import registers it. An explicit table rather than a path
# built from the name: the ImportError below is swallowed, so a convention-derived path that goes stale
# degrades into "no backend registered" instead of failing where the cause is visible. A table breaks
# loudly at review time when a module moves, and `test_backend_name_resolution.py` resolves every entry
# from a cold registry as a positive control -- the failure mode here is an ABSENCE, which passes
# silently when nothing checks it.
_BUNDLED_BACKEND_MODULES: dict[str, str] = {
    "circuit_tracer": "interpretune.adapters.circuit_tracer.backends",
}


def resolve_analysis_backend(name: str) -> "AnalysisBackend":
    """Resolve a backend NAME to its registered instance, lazily importing the bundled module.

    Bundled backends register themselves at import time, so a name that is not in the registry yet may simply not have
    been imported. Hub-delivered backends register from their component entrypoint and are already present by the time
    anything resolves them, so they never reach the table.
    """
    if name not in ANALYSIS_BACKEND_REGISTRY:
        module_path = _BUNDLED_BACKEND_MODULES.get(name)
        if module_path is not None:
            import importlib

            try:
                importlib.import_module(module_path)
            except ImportError:
                pass
    if name not in ANALYSIS_BACKEND_REGISTRY:
        raise KeyError(
            f"No analysis backend registered as {name!r} (known: {sorted(ANALYSIS_BACKEND_REGISTRY)}). "
            "Hydrating this artifact requires the backend's package/extra to be installed."
        )
    return ANALYSIS_BACKEND_REGISTRY[name]


def require_analysis_backend(module: Any) -> AnalysisBackend:
    """Return the module's analysis backend or raise if it is unavailable."""

    backend = get_analysis_backend(module)
    if backend is None:
        raise ValueError("Target module must expose an analysis_backend for this operation")
    return backend


def get_module_capabilities(module: Any) -> ModuleCapabilities:
    """Aggregate execution and analysis capabilities exposed by a module."""

    model_capabilities: set[ModelBackendCapability] = set()
    analysis_capabilities: set[AnalysisBackendCapability] = set()
    backend = get_model_backend(module)

    if backend is not None and hasattr(backend, "capabilities"):
        model_capabilities.update(
            capability
            for capability in (normalize_backend_capability(raw_capability) for raw_capability in backend.capabilities)
            if isinstance(capability, ModelBackendCapability)
        )

    analysis_backend = get_analysis_backend(module)
    if analysis_backend is not None and hasattr(analysis_backend, "capabilities"):
        analysis_capabilities.update(
            capability
            for capability in (
                normalize_backend_capability(raw_capability) for raw_capability in analysis_backend.capabilities
            )
            if isinstance(capability, AnalysisBackendCapability)
        )

    module_declared = getattr(module, "analysis_capabilities", None)
    if module_declared and analysis_backend is None:
        # A capability declared on the module alone has no backend to carry its support record, and a record is
        # what makes the declaration checkable; refused by name rather than aggregated as a bare set.
        raise ValueError(
            f"{type(module).__name__} declares analysis capabilities"
            f" {sorted(str(getattr(c, 'value', c)) for c in module_declared)}"
            " on the module with no analysis backend attached; attach the backend that implements them, which"
            " carries their support records"
        )

    return ModuleCapabilities(
        model=frozenset(model_capabilities),
        analysis=frozenset(analysis_capabilities),
        intervention=_support_record(
            backend, ModelBackendCapability.ACTIVATION_INTERVENTION, model_capabilities, "intervention_support"
        ),
        latent_models=_support_record(
            backend, ModelBackendCapability.LATENT_MODELS, model_capabilities, "latent_model_support"
        ),
        capture=_capture_record(backend, module),
        attribution_graph=_support_record(
            analysis_backend,
            AnalysisBackendCapability.ATTRIBUTION_GRAPH,
            analysis_capabilities,
            "attribution_graph_support",
        ),
        feature_intervention=_support_record(
            analysis_backend,
            AnalysisBackendCapability.FEATURE_INTERVENTION,
            analysis_capabilities,
            "feature_intervention_support",
        ),
    )


def _capture_record(backend: Any, module: Any) -> CaptureSupport | None:
    """The backend's capture declaration for ``module.model``, or ``None`` when the backend has none to give.

    Capturability is a property of the backend AND the model it wraps (a TransformerLens backend captures different
    points on a bridge than on a HookedTransformer), so the declaration is a method taking the model rather than a
    property. A backend without the method is not refused here, because this aggregation feeds the adapter card and
    ``adapter_info`` on a bare install; the conformance suite is where its absence fails by name.
    """
    declare = getattr(backend, "capture_support", None) if backend is not None else None
    model = getattr(module, "model", None)
    if declare is None or model is None:
        return None
    return declare(model)


def _support_record(backend: Any, capability: Capability, declared: set[Any], attr: str) -> Any:
    """The support record a backend attaches for ``capability``, or ``None`` when it does not declare it.

    Read with ``getattr`` rather than through the protocol so a backend that declares the surface and
    forgot the record fails in :class:`ModuleCapabilities`' invariant with a message naming the record,
    instead of as an ``AttributeError`` here.
    """
    if capability not in declared or backend is None:
        return None
    return getattr(backend, attr, None)
