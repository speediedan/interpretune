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
    """Capabilities that a model backend may support.

    Ops and the dispatcher can query ``backend.capabilities`` to check support before
    calling optional methods.  Backends that do not support a capability should fall back
    to a simpler code path (e.g., looping instead of batching).
    """

    LATENT_MODELS = "latent_models"
    """Backend supports execution with latent-model handles attached (``SupportsLatentModels``:

    ``fwd_w_cache_and_latent_models``, ``fwd_w_hooks_and_latent_models``, ``fwd_w_hooks_batched``).
    """

    BATCHED_HOOKS = "batched_hooks"
    """Backend can run multiple forward passes with different hook configs in a single batched execution (e.g.,
    NNsight multi-invoke within one trace).

    An efficiency property of HOW ``fwd_w_hooks_batched`` runs --
    the method itself is part of ``LATENT_MODELS`` and a sequential loop is a valid implementation.
    """

    GRADIENTS = "gradients"
    """Backend supports forward + backward with gradient caching (``SupportsGradients``)."""

    INTERVENTION = "intervention"
    """Backend supports baseline-vs-intervention paired execution (``SupportsIntervention``:

    ``fwd_w_intervention``).
    """


class AnalysisBackendCapability(Enum):
    """Capabilities exposed by analysis adapters/backends rather than model execution backends."""

    ATTRIBUTION_GRAPH = "attribution_graph"
    """Module exposes attribution graph analysis support via an attached analysis backend."""

    FEATURE_INTERVENTION = "feature_intervention"
    """Module exposes feature intervention support via an attached analysis backend."""


Capability: TypeAlias = BackendCapability | AnalysisBackendCapability


@dataclass(frozen=True)
class ModuleCapabilities:
    """Execution and analysis capabilities exposed by a module."""

    model: frozenset[BackendCapability]
    analysis: frozenset[AnalysisBackendCapability]

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


def resolve_analysis_backend(name: str) -> "AnalysisBackend":
    """Resolve a backend NAME to its registered instance, lazily importing the in-tree module.

    In-tree backends register themselves at import time, so a name that is not in the registry yet may simply not have
    been imported. The lazy import targets ``backends.impls.<name>``, where the concrete backends live. Note this
    resolution is by module PATH built from the name, which is why moving those modules is not a pure-motion change even
    though nothing imports them by name from here: the ``ImportError`` below is swallowed, so a stale path degrades into
    "no backend registered" rather than an import failure that would point at the cause.
    """
    if name not in ANALYSIS_BACKEND_REGISTRY:
        import importlib

        try:
            importlib.import_module(f"interpretune.analysis.backends.impls.{name}")
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

    return ModuleCapabilities(model=frozenset(model_capabilities), analysis=frozenset(analysis_capabilities))
