"""The ``ModelBackend`` and ``AnalysisBackend`` protocols.

These are the extension points a new execution framework (TransformerLens, nnsight, ...) or analysis
package (circuit-tracer, ...) implements. Op implementations consume them through the capability
helpers rather than instantiating them directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Protocol, runtime_checkable

import torch

from interpretune.analysis.backends.capabilities import AnalysisBackendCapability, BackendCapability
from interpretune.analysis.backends.interventions import InterventionDict, InterventionValue
from interpretune.protocol import NamesFilter


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

    def get_tokenizer(self, module: Any) -> Any:
        """Resolve the tokenizer for ``module``, wherever this backend keeps it."""
        ...

    def get_embedding_weight(self, module: Any) -> torch.Tensor:
        """Return the weight matrix used to map between token ids and the residual stream.

        Implementations may return the UNEMBED matrix where the backend exposes one; callers use this for
        direction/token projection and must not assume it is the input embedding specifically.
        """
        ...

    def token_strings_to_ids(self, tokenizer: Any, token_strings: list[str]) -> list[int]:
        """Map literal token strings to ids, honoring this backend's tokenization conventions.

        Concept work names tokens as they appear in a vocabulary (leading-space markers included), so
        this is deliberately not ``tokenizer.encode`` -- a string that is one vocabulary entry must map
        to exactly one id rather than being re-segmented.
        """
        ...

    def resolve_prompt(self, module: Any, analysis_batch: Any, batch: Any) -> str:
        """Recover the single prompt string an analysis is operating on.

        The prompt may have been carried on the analysis batch or only survive as token ids in the encoding, so
        implementations resolve from whichever source is populated (decoding if needed).
        """
        ...

    def build_concept_attribution_targets(
        self,
        module: Any,
        prompt: str,
        concept_direction: Any,
        concept_label: Any,
        *,
        concept_group_a_token_ids: Any = None,
        concept_group_b_token_ids: Any = None,
        concept_direction_mode: Any = None,
    ) -> list[Any] | None:
        """Build backend-native attribution targets for a concept direction, or None if unsupported.

        Returning None is a valid answer: a backend that cannot express concept-directed attribution
        targets says so here rather than raising, and the caller falls back.
        """
        ...

    def resolve_feature_intervention_settings(
        self,
        module: Any,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve the effective feature-intervention settings, applying ``overrides`` over config."""
        ...

    def build_feature_interventions(
        self,
        analysis_batch: Any,
        settings: dict[str, Any],
    ) -> tuple[list[tuple[int, int, int, float]], dict[str, Any]]:
        """Build concrete interventions from a batch plus resolved settings.

        Returns the intervention tuples the backend will apply, alongside metadata describing what was
        selected -- the second element is what makes an intervention auditable after the fact.
        """
        ...

    def feature_intervention_call_kwargs(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Translate resolved settings into this backend's forward-call keyword arguments.

        Optional settings are omitted rather than passed as None, so a backend whose signature does not accept them
        still works.
        """
        ...

    def decompose_graph(self, graph: Any, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Flatten a backend-native attribution graph into serializable, storable components.

        The inverse of :meth:`hydrate_graph_from_batch`. Tensors are detached and moved to CPU so the
        result can cross a process or land in an ``AnalysisStore`` without carrying device state.
        """
        ...

    def hydrate_graph_from_batch(self, analysis_batch: Any) -> Any:
        """Rebuild a backend-native graph from components previously stored on an analysis batch.

        The inverse of :meth:`decompose_graph`, and the reason graph-consuming ops can run against a
        replayed store rather than only against a live attribution pass.
        """
        ...

    def build_pruned_graph(self, graph: Any, node_threshold: float, edge_threshold: float) -> Any:
        """Return a copy of ``graph`` with nodes and edges below the given thresholds removed."""
        ...

    def select_feature_rows(self, active_features: torch.Tensor, selected_features: torch.Tensor) -> torch.Tensor:
        """Index the active-feature table by selected feature indices, preserving its column layout.

        Must return an empty tensor of the same column shape when nothing is selected, so callers can treat the result
        uniformly instead of special-casing the empty selection.
        """
        ...

    def compute_node_influence_scores(self, graph: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-node influence over the graph, returning the influence and logit-gradient scores."""
        ...

    def compute_signed_node_influence_scores(self, graph: Any) -> torch.Tensor:
        """Compute per-node influence that RETAINS sign, so promoting and suppressing nodes are distinguishable.

        The unsigned counterpart is :meth:`compute_node_influence_scores`; sign-aware steering needs this
        one, since magnitude alone cannot tell which direction a node pushes.
        """
        ...


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

    def fwd_w_cache(
        self,
        model: Any,
        batch: dict[str, Any],
        names_filter: NamesFilter,
    ) -> tuple[torch.Tensor, Any]:
        """Run a forward pass with activation caching but without latent model hooks.

        Args:
            model: The model to run.
            batch: Input batch dict.
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

    def fwd_w_intervention(
        self,
        model: Any,
        batch: dict[str, Any],
        interventions: InterventionDict | Mapping[str, InterventionValue],
        latent_model_handles: list[Any] | None = None,
    ) -> tuple[Any, Any]:
        """Run baseline + intervention forward passes using the given hook specs.

        Performs two forward passes:

        1. **Baseline**: captures pre-intervention logits.
        2. **Intervention**: for each key in *interventions*, matches the key (which may
           contain ``*`` wildcards) against available hook names, then applies each
           ``InterventionSpec`` at the last sequence position according to its ``mode``
           (``"replace"``, ``"add"``, or ``"project"``).

        Args:
            model: The model to run.
            batch: Input batch dict.
            interventions: Either a canonical :class:`InterventionDict` keyed by concrete hook
                names or a raw mapping from hook-name patterns to intervention payloads. Raw
                payloads may be tensors, ``InterventionSpec`` instances, mapping-style specs, or
                sequences of those values. Patterns may use ``*`` as a glob-style wildcard.
            latent_model_handles: Optional latent model handles to enable latent-hook-aware
                resolution and execution.

        Returns:
            ``(pre_intervention_logits, post_intervention_logits)`` — both real tensors.
        """
        ...
