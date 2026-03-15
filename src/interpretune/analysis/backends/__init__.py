"""Model backends for analysis operations.

Provides the ``ModelBackend`` protocol, ``BackendCapability`` enum, and backend implementations
for different model execution frameworks (TransformerLens, nnsight, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, Sequence, TypeAlias, runtime_checkable

import torch

from interpretune.protocol import NamesFilter


class BackendCapability(Enum):
    """Capabilities that a model backend may support.

    Ops and the dispatcher can query ``backend.capabilities`` to check support before
    calling optional methods.  Backends that do not support a capability should fall back
    to a simpler code path (e.g., looping instead of batching).
    """

    BATCHED_HOOKS = "batched_hooks"
    """Backend can run multiple forward passes with different hook configs in a single batched execution (e.g.,
    NNsight multi-invoke within one trace)."""

    GRADIENTS = "gradients"
    """Backend supports forward + backward with gradient caching."""


class AnalysisBackendCapability(Enum):
    """Capabilities exposed by analysis adapters/backends rather than model execution backends."""

    ATTRIBUTION_GRAPH = "attribution_graph"
    """Module exposes attribution graph analysis support via an attached analysis backend."""

    FEATURE_INTERVENTION = "feature_intervention"
    """Module exposes feature intervention support via an attached analysis backend."""

    # Future capabilities (reserved):
    # REMOTE_EXECUTION = "remote_execution"
    # SOURCE_TRACING = "source_tracing"


Capability: TypeAlias = BackendCapability | AnalysisBackendCapability


@dataclass(frozen=True)
class ModuleCapabilities:
    """Execution and analysis capabilities exposed by a module."""

    model: frozenset[BackendCapability]
    analysis: frozenset[AnalysisBackendCapability]

    @property
    def all(self) -> frozenset[Capability]:
        return frozenset({*self.model, *self.analysis})

    @property
    def values(self) -> frozenset[str]:
        return frozenset(cap.value for cap in self.all)

    def supports(self, capability: Capability) -> bool:
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


def get_analysis_backend(module: Any) -> AnalysisBackend | None:
    backend = getattr(module, "_analysis_backend", None)
    if backend is None and hasattr(module, "analysis_backend"):
        try:
            backend = module.analysis_backend
        except (AssertionError, AttributeError):
            backend = None
    return backend


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
    backend = getattr(module, "_model_backend", None)
    if backend is None and hasattr(module, "model_backend"):
        try:
            backend = module.model_backend
        except (AssertionError, AttributeError):
            backend = None

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


@runtime_checkable
class AnalysisBackend(Protocol):
    """Protocol defining analysis-adapter functionality layered above model execution backends."""

    @property
    def capabilities(self) -> frozenset[AnalysisBackendCapability]:
        """Return the set of analysis capabilities this backend supports."""
        ...

    def supports(self, capability: AnalysisBackendCapability) -> bool:
        """Check whether this backend supports a given analysis capability."""
        ...

    def get_tokenizer(self, module: Any) -> Any: ...

    def get_embedding_weight(self, module: Any) -> torch.Tensor: ...

    def token_strings_to_ids(self, tokenizer: Any, token_strings: list[str]) -> list[int]: ...

    def resolve_prompt(self, module: Any, analysis_batch: Any, batch: Any) -> str: ...

    def resolve_feature_intervention_settings(
        self,
        module: Any,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def build_feature_interventions(
        self,
        analysis_batch: Any,
        settings: dict[str, Any],
    ) -> tuple[list[tuple[int, int, int, float]], dict[str, Any]]: ...

    def feature_intervention_call_kwargs(self, settings: dict[str, Any]) -> dict[str, Any]: ...

    def decompose_graph(self, graph: Any, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def hydrate_graph_from_batch(self, analysis_batch: Any) -> Any: ...

    def build_pruned_graph(self, graph: Any, node_threshold: float, edge_threshold: float) -> Any: ...

    def select_feature_rows(self, active_features: torch.Tensor, selected_features: torch.Tensor) -> torch.Tensor: ...

    def compute_node_influence_scores(self, graph: Any) -> tuple[torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol defining the interface for model execution backends.

    Each backend wraps a specific framework's model execution API (e.g., TransformerLens hook-based execution, nnsight
    trace-based execution) behind a uniform interface used by analysis op implementations.

    .. note:: ``hook=True`` evaluation

        NNsight's ``hook=True`` parameter (on ``tracer.invoke()``) enables ``.output`` /
        ``.input`` access on auxiliary modules like SAEs.  Our current architecture calls
        ``sae.encode()`` / ``sae.decode()`` explicitly within the trace, giving direct
        proxy access to feature activations.  If SAEs were registered as model sub-modules,
        ``hook=True`` could replace explicit encode/decode calls, but the current external
        ``latent_model_handles`` design makes ``hook=True`` unnecessary.  Adding
        ``hook=True`` would require architectural changes to how SAEs are attached and is
        best evaluated in a future session.
    """

    @property
    def capabilities(self) -> frozenset[BackendCapability]:
        """Return the set of capabilities this backend supports.

        Backends must override this property to declare their capabilities. Analysis ops can check capabilities before
        calling optional methods.
        """
        ...

    def supports(self, capability: BackendCapability) -> bool:
        """Check whether this backend supports a given capability.

        Default implementation checks ``capability in self.capabilities``.
        """
        ...

    def fwd(
        self,
        model: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Run a minimal forward pass and return logits.

        Each backend handles any necessary batch-key mapping (e.g., the NNsight
        backend wraps the call in a trace context so that
        ``LanguageModel._prepare_input`` correctly routes ``input`` →
        ``input_ids`` for HuggingFace models).

        Args:
            model: The model to run.
            batch: Input batch dict.

        Returns:
            Model output logits tensor.
        """
        ...

    def fwd_w_cache_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        names_filter: NamesFilter,
    ) -> tuple[torch.Tensor, Any]:
        """Run a forward pass with activation caching and latent model hooks.

        Args:
            model: The model to run (e.g., HookedSAETransformer, SAETransformerBridge).
            batch: Input batch dict (unpacked as ``**batch`` for the model call).
            latent_model_handles: Latent model handles (e.g., SAE objects) to attach.
            names_filter: Filter specifying which hook activations to cache.

        Returns:
            Tuple of (logits, activation_cache).
        """
        ...

    def fwd_w_hooks_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        fwd_hooks: list[tuple[str, Any]],
        clear_contexts: bool = True,
    ) -> torch.Tensor:
        """Run a forward pass with custom forward hooks and latent model hooks.

        Args:
            model: The model to run.
            batch: Input batch dict (unpacked as ``**batch`` for the model call).
            latent_model_handles: Latent model handles to attach.
            fwd_hooks: List of (hook_name, hook_fn) tuples for forward hooks.
            clear_contexts: Whether to clear hook contexts after the forward pass.

        Returns:
            Model output logits.
        """
        ...

    def fwd_w_hooks_batched(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        hook_configs: Sequence[list[tuple[str, Any]]],
        clear_contexts: bool = True,
        configs_per_pass: int | None = None,
    ) -> list[torch.Tensor]:
        """Run multiple forward passes with different hook configurations, batched when possible.

        Each element of ``hook_configs`` is a ``fwd_hooks`` list (as passed to
        ``fwd_w_hooks_and_latent_models``).  Backends that support
        :attr:`BackendCapability.BATCHED_HOOKS` may batch all configs into a single
        execution context (e.g., NNsight multi-invoke within one trace) for efficiency.
        Other backends loop over configs sequentially.

        ``configs_per_pass`` limits how many configs are batched per execution context.
        When ``None`` (default), the entire ``hook_configs`` list is batched in one context.
        Setting a value (e.g., 64) chunks the work to avoid OOM with very large alive-latent
        counts.

        .. note::

            TODO: evaluate the possibility of releasing memory after each invoke within a
            trace if OOMs become a problem (would require nnsight-level support).

        Args:
            model: The model to run.
            batch: Input batch dict (unpacked as ``**batch`` for the model call).
            latent_model_handles: Latent model handles to attach.
            hook_configs: Sequence of ``fwd_hooks`` lists, one per forward pass.
            clear_contexts: Whether to clear hook contexts (for TL backend compatibility).
            configs_per_pass: Maximum number of configs to batch per execution context.
                ``None`` means unbounded (all configs in one trace).

        Returns:
            List of logits tensors, one per element in ``hook_configs``.
        """
        ...

    def fwd_w_grads_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        fwd_hooks: list[tuple[Any, Any]],
        bwd_hooks: list[tuple[Any, Any]],
        backward_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run forward + backward with latent model hooks and gradient caching.

        The backend owns the entire forward+backward execution flow.  This enables both
        eager execution (TransformerLens) and deferred/traced execution (NNsight) to
        use the same op-level code.

        The ``backward_fn`` closure is provided by the analysis op and computes a scalar
        metric from raw model logits.  The backend calls ``backward_fn(logits)`` to obtain
        the scalar, then runs ``.backward()`` on it (eager for TL, deferred via
        ``with scalar.backward():`` for NNsight).

        Forward and backward cache hooks (``fwd_hooks``, ``bwd_hooks``) are structured as
        ``[(names_filter, cache_fn), ...]`` and are invoked by the backend to populate
        ``analysis_cfg.cache_dict``.  For TL, hooks fire during execution.  For NNsight,
        the backend calls them after the trace completes with materialized tensors.

        Args:
            model: The model to run.
            batch: Input batch dict (unpacked as ``**batch`` for the model call).
            latent_model_handles: Latent model handles (e.g., SAE objects) to attach.
            fwd_hooks: Forward cache hooks ``[(names_filter, cache_fn), ...]``.
            bwd_hooks: Backward cache hooks ``[(names_filter, cache_fn), ...]``.
            backward_fn: ``raw_logits -> scalar``.  Takes the full model output logits and
                returns a scalar tensor to call ``.backward()`` on.  Must be compatible with
                both real tensors (TL) and NNsight proxy objects.

        Returns:
            Raw model output logits (always a real tensor, even for NNsight).
        """
        ...

    def wrap_activation_cache(
        self,
        cache_dict: dict[str, Any],
        model: Any,
    ) -> Any:
        """Wrap a raw activation dict into a backend-specific activation cache object.

        For TransformerLens, wraps in ``ActivationCache``. Other backends may return the dict
        as-is or wrap in their own cache type.

        Args:
            cache_dict: Raw dict mapping hook names to activation tensors.
            model: The model instance (may be needed for cache construction).

        Returns:
            A cache object suitable for indexed access by hook name.
        """
        ...


__all__ = [
    "BackendCapability",
    "AnalysisBackend",
    "AnalysisBackendCapability",
    "Capability",
    "get_module_capabilities",
    "ModelBackend",
    "ModuleCapabilities",
    "normalize_backend_capability",
]
