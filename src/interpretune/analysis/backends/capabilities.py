"""Backend capability enums, module capability aggregation, and the named-backend registry.

Part of the sanctioned :mod:`interpretune.analysis.backends` seam that op implementations (bundled,
local, or hub) may import. Ops should ask for capabilities rather than branching on backend or
adapter class names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from interpretune.analysis.backends.protocols import AnalysisBackend, ModelBackend


class BackendCapability(Enum):
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

    INTERVENTION = "intervention"
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
    be unable to express most modes, because ``replace``, ``patch`` and ``project`` all need the CURRENT
    activation while an additive steering primitive never observes it. A mode a backend has not declared
    is refused by :func:`~interpretune.analysis.backends.interventions.require_intervention_mode` rather
    than applied as a different mode, since every mode returns plausible logits and the substitution is
    undetectable from the result.
    """

    REPLACE = "replace"
    ADD = "add"
    PATCH = "patch"
    PROJECT = "project"


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
        """Every scope and every mode: the declaration of a backend whose hook sees the whole activation."""
        return cls(position_scopes=frozenset(PositionScope), modes=frozenset(InterventionMode))


@dataclass(frozen=True)
class LatentModelSupport:
    """How ``LATENT_MODELS`` runs on this backend.

    ``batched_hooks`` says whether ``fwd_w_hooks_batched`` fuses its hook configs into one execution
    (nnsight's multi-invoke) or loops. It lives here rather than in :class:`BackendCapability` because it
    is a property of HOW a method in that group runs, not a surface of its own: every latent-models
    backend implements the method, and a sequential loop is a valid implementation.
    """

    batched_hooks: bool = False


class AnalysisBackendCapability(Enum):
    """Capabilities exposed by analysis adapters/backends rather than model execution backends."""

    ATTRIBUTION_GRAPH = "attribution_graph"
    """Module exposes attribution graph analysis support via an attached analysis backend."""

    FEATURE_INTERVENTION = "feature_intervention"
    """Module exposes feature intervention support via an attached analysis backend."""


Capability: TypeAlias = BackendCapability | AnalysisBackendCapability


@dataclass(frozen=True)
class ModuleCapabilities:
    """Execution and analysis capabilities exposed by a module, with each surface's support record.

    ``intervention`` is present iff ``INTERVENTION`` is declared and ``latent_models`` iff
    ``LATENT_MODELS`` is; the constructor enforces that, so a consumer rendering this (the adapter card,
    ``adapter_info``, a conformance report) can rely on the record being there when the surface is.
    """

    model: frozenset[BackendCapability]
    analysis: frozenset[AnalysisBackendCapability]
    intervention: InterventionSupport | None = None
    latent_models: LatentModelSupport | None = None

    def __post_init__(self) -> None:
        for capability, record, name in (
            (BackendCapability.INTERVENTION, self.intervention, "intervention"),
            (BackendCapability.LATENT_MODELS, self.latent_models, "latent_models"),
        ):
            declared = capability in self.model
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
        if isinstance(capability, BackendCapability):
            return capability in self.model
        return capability in self.analysis


def normalize_backend_capability(capability: Any) -> Capability:
    """Normalize capability-like values to the local execution or analysis capability enums."""

    if isinstance(capability, (BackendCapability, AnalysisBackendCapability)):
        return capability

    raw_value = getattr(capability, "value", capability)
    normalized_value = str(raw_value)
    if normalized_value == "attribution":
        normalized_value = AnalysisBackendCapability.ATTRIBUTION_GRAPH.value

    try:
        return BackendCapability(normalized_value)
    except ValueError:
        pass

    try:
        return AnalysisBackendCapability(normalized_value)
    except ValueError:
        if isinstance(raw_value, str) and "." in raw_value:
            suffix = raw_value.split(".")[-1].lower()
            if suffix == "attribution":
                suffix = AnalysisBackendCapability.ATTRIBUTION_GRAPH.value
            try:
                return BackendCapability(suffix)
            except ValueError:
                return AnalysisBackendCapability(suffix)
        raise


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

    model_capabilities: set[BackendCapability] = set()
    analysis_capabilities: set[AnalysisBackendCapability] = set()
    backend = get_model_backend(module)

    if backend is not None and hasattr(backend, "capabilities"):
        model_capabilities.update(
            capability
            for capability in (normalize_backend_capability(raw_capability) for raw_capability in backend.capabilities)
            if isinstance(capability, BackendCapability)
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

    legacy_analysis_capabilities = getattr(module, "analysis_capabilities", None)
    if legacy_analysis_capabilities:
        analysis_capabilities.update(
            capability
            for capability in (
                normalize_backend_capability(raw_capability) for raw_capability in legacy_analysis_capabilities
            )
            if isinstance(capability, AnalysisBackendCapability)
        )

    return ModuleCapabilities(
        model=frozenset(model_capabilities),
        analysis=frozenset(analysis_capabilities),
        intervention=_support_record(
            backend, BackendCapability.INTERVENTION, model_capabilities, "intervention_support"
        ),
        latent_models=_support_record(
            backend, BackendCapability.LATENT_MODELS, model_capabilities, "latent_model_support"
        ),
    )


def _support_record(backend: Any, capability: BackendCapability, declared: set[BackendCapability], attr: str) -> Any:
    """The support record a backend attaches for ``capability``, or ``None`` when it does not declare it.

    Read with ``getattr`` rather than through the protocol so a backend that declares the surface and
    forgot the record fails in :class:`ModuleCapabilities`' invariant with a message naming the record,
    instead of as an ``AttributeError`` here.
    """
    if capability not in declared or backend is None:
        return None
    return getattr(backend, attr, None)
