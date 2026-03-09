"""NNsight model backend implementation.

Provides ``NNsightModelBackend`` which implements the ``ModelBackend`` protocol for NNsight-loaded
HuggingFace models.  All four protocol methods (``fwd_w_cache_and_latent_models``,
``fwd_w_hooks_and_latent_models``, ``fwd_w_hooks_batched``,
``fwd_w_grads_and_latent_models``) use NNsight's trace/proxy API for full compatibility with
both local and remote (NDIF) model execution.

Also provides ``NNsightActivationCacheAdapter`` which wraps a plain dict into a cache object
compatible with analysis op expectations (``items()``, ``keys()``, ``__getitem__``, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
from typing import Any, Callable, Literal, Sequence

import nnsight as _nnsight
import torch

from interpretune.analysis.backends import BackendCapability
from interpretune.analysis.backends.hook_mapping import HookNameResolver, ResolvedHook
from interpretune.protocol import NamesFilter

# Disable PYMOUNT C extension — this backend uses nnsight.save() exclusively.
# Must be set before the first trace context is entered.
_nnsight.CONFIG.APP.PYMOUNT = False  # type: ignore[attr-defined]  # nnsight CONFIG attrs are dynamic


# ==============================================================================
# Model introspection helpers
# ==============================================================================

# Common backbone attribute names shared by many HF causal-LM wrappers
# (GPT2LMHeadModel.transformer, LlamaForCausalLM.model, etc.).
_BACKBONE_ATTR_NAMES = ("transformer", "model")

_CONFIGS_PER_PASS_ENV = "IT_NNSIGHT_CONFIGS_PER_PASS"


def get_default_configs_per_pass() -> int:
    """Return a conservative default multi-invoke batch size for the current runtime.

    CPU-only runs are substantially more memory-sensitive for attr-ablation than GPU-backed execution, and hosted
    Windows runners have been the most fragile path in CI. Allow an explicit env override for local profiling while
    defaulting to smaller CPU batches.
    """

    env_value = os.environ.get(_CONFIGS_PER_PASS_ENV)
    if env_value is not None:
        return int(env_value)

    if not torch.cuda.is_available():
        return 2 if platform.system() == "Windows" else 4

    return 32


def _find_backbone_module(hf_model: Any) -> Any | None:
    """Return the decoder backbone module inside an HF ``*ForCausalLM`` wrapper.

    Many HF causal-LM classes (``GPT2LMHeadModel``, ``LlamaForCausalLM``, …) store
    the core decoder body under a well-known attribute (``transformer``, ``model``).
    This helper tries each known attribute name and returns the first match, or
    ``None`` if no backbone is found.
    """
    for attr in _BACKBONE_ATTR_NAMES:
        backbone = getattr(hf_model, attr, None)
        if backbone is not None:
            return backbone
    return None


# ==============================================================================
# Activation cache adapter
# ==============================================================================


class NNsightActivationCacheAdapter:
    """Wraps a plain dict as an activation cache compatible with analysis ops.

    Provides the dict-like interface (``items()``, ``keys()``, ``__getitem__``, ``__contains__``)
    that analysis operations expect from activation caches. Cache keys are TL-style hook names
    for consistency with the ``TLModelBackend``'s ``ActivationCache``.

    Args:
        cache_dict: Dict mapping TL-style hook names to activation tensors.
    """

    def __init__(self, cache_dict: dict[str, torch.Tensor]) -> None:
        self._cache = cache_dict

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._cache[key]

    def __contains__(self, key: object) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"NNsightActivationCacheAdapter(keys={list(self._cache.keys())})"

    def items(self) -> Any:
        """Return key-value pairs of (hook_name, activation_tensor)."""
        return self._cache.items()

    def keys(self) -> Any:
        """Return cache keys (TL-style hook names)."""
        return self._cache.keys()

    def values(self) -> Any:
        """Return cached activation tensors."""
        return self._cache.values()


# ==============================================================================
# Helper: minimal HookPoint stand-in for TL-style hook function calls
# ==============================================================================


@dataclass
class _DummyHookPoint:
    """Minimal stand-in for TL's ``HookPoint`` to satisfy the ``(tensor, hook)`` call signature.

    TL hook functions (caching hooks, ablation hooks) expect ``hook.name`` to identify the hook.
    This provides the same attribute without importing ``transformer_lens``.
    """

    name: str


# ==============================================================================
# NNsight model backend
# ==============================================================================


def _matches_names_filter(name: str, names_filter: NamesFilter) -> bool:
    """Check whether ``name`` matches the given ``NamesFilter``.

    Handles all ``NamesFilter`` variants: callable, list of strings, single string, or ``None``.
    """
    if names_filter is None:
        return True
    if callable(names_filter):
        return names_filter(name)
    if isinstance(names_filter, str):
        return name == names_filter
    # Sequence of strings
    return name in names_filter


# ==============================================================================
# NNsight envoy navigation helpers
# ==============================================================================


def _navigate_envoy(model: Any, path: str) -> Any:
    """Navigate the NNsight envoy tree to a sub-module by dot-separated path.

    Handles both named attributes and integer indices (for ``ModuleList``/``Sequential``):

    - ``"transformer.h.5"`` → ``model.transformer.h[5]``
    - ``"model.layers.3.mlp"`` → ``model.model.layers[3].mlp``

    Args:
        model: NNsight ``LanguageModel`` (top-level envoy).
        path: Dot-separated HF module path with layer indices.

    Returns:
        The envoy (or sub-envoy) at the given path.
    """
    current = model
    for part in path.split("."):
        try:
            idx = int(part)
            current = current[idx]
        except ValueError:
            current = getattr(current, part)
    return current


def _read_envoy_activation(envoy: Any, resolved: ResolvedHook) -> Any:
    """Read the hidden-state activation proxy from an NNsight envoy.

    For **output** hooks: accesses ``envoy.output[0]`` for modules returning tuples
    (transformer blocks, attention) or ``envoy.output`` for single-tensor modules
    (MLP, LayerNorm).

    For **input** hooks: accesses ``envoy.input`` directly.  In NNsight 0.6+,
    ``.input`` already returns the **first positional argument** (a tensor), so
    indexing with ``[0]`` would incorrectly slice into dimension 0 of that tensor.

    Args:
        envoy: NNsight envoy for the target module.
        resolved: :class:`ResolvedHook` with ``io_type`` and ``tuple_output`` flags.

    Returns:
        NNsight proxy for the activation tensor.
    """
    if resolved.io_type == "output":
        if resolved.tuple_output:
            return envoy.output[0]
        return envoy.output
    return envoy.input


def _write_envoy_activation(envoy: Any, resolved: ResolvedHook, value: Any) -> None:
    """Replace the hidden-state activation in an NNsight envoy via proxy assignment.

    Creates an NNsight intervention that substitutes the activation during trace execution.
    For input hooks, assigns to ``envoy.input`` directly (the first positional argument)
    rather than ``envoy.input[0]`` which would be an in-place modification of the tensor's
    first element along dimension 0.

    Args:
        envoy: NNsight envoy for the target module.
        resolved: :class:`ResolvedHook` with ``io_type`` and ``tuple_output`` flags.
        value: Replacement activation proxy (e.g., SAE reconstruction).
    """
    if resolved.io_type == "output":
        if resolved.tuple_output:
            envoy.output[0] = value
        else:
            envoy.output = value
    else:
        envoy.input = value


class NNsightModelBackend:
    """NNsight model execution backend.

    Implements the ``ModelBackend`` protocol for models loaded via NNsight's ``LanguageModel``.
    All three protocol methods use NNsight's trace/proxy API for full compatibility with both
    local and remote (NDIF) model execution.

    The ``HookNameResolver`` translates between TL-style hook names (used by analysis ops and
    SAELens SAE configurations) and HF module paths on the underlying model.

    Args:
        hook_resolver: A :class:`HookNameResolver` initialised for the model's architecture.
        configs_per_pass: Default maximum number of hook configurations to batch per
            execution context. ``None`` means unbounded.
    """

    def __init__(self, hook_resolver: HookNameResolver, configs_per_pass: int | None = 32) -> None:
        self._resolver = hook_resolver
        self._configs_per_pass = configs_per_pass

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> frozenset[BackendCapability]:
        """NNsight supports batched hooks (multi-invoke) and gradient caching."""
        return frozenset({BackendCapability.BATCHED_HOOKS, BackendCapability.GRADIENTS})

    def supports(self, capability: BackendCapability) -> bool:
        """Check whether this backend supports a given capability."""
        return capability in self.capabilities

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def register_model_hooks(self, model: Any) -> None:
        """Register forward pre-hooks needed for correct model execution.

        Currently a **no-op**.  Reserved for future backend-specific hooks.

        .. note:: Position IDs

            An earlier version of this method registered a hook that applied
            the legacy ``transformers`` v4 position-ID computation::

                position_ids = attention_mask.cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)

            This was removed because **both TransformerBridge and
            HookedTransformer use the default ``transformers`` v5 sequential
            position IDs** (``arange(0, seq_len)``), which ignore padding.
            Applying the cumsum fix only to the NNsight backend produced
            fundamentally different position embeddings for left-padded
            inputs, breaking Tier 1 (Bridge ↔ NNsight) parity.

            With the hook removed, all three backends (Bridge, HT, NNsight)
            use the same ``arange`` position IDs.  Attention masking
            (via ``attention_mask`` in the batch) prevents padding tokens
            from contributing to the output regardless of their position
            embeddings.

        Args:
            model: NNsight ``LanguageModel`` (unused, retained for API
                compatibility).
        """

    def _splice_sae(
        self,
        model: Any,
        sae: Any,
        *,
        hook_fn: Callable[[Any, _DummyHookPoint], Any] | None = None,
    ) -> tuple[Any, Any, ResolvedHook]:
        """Splice an SAE into the model's activation stream within an NNsight trace.

        Resolves the SAE hook point, reads the activation proxy, runs encode → (optional
        hook) → decode, and writes the reconstruction back.  This is the repeated
        pattern shared by all protocol methods.

        Must be called inside an active ``tracer.invoke()`` context.

        Args:
            model: NNsight ``LanguageModel``.
            sae: SAELens SAE handle (with ``cfg.metadata.hook_name``).
            hook_fn: Optional TL-style hook function ``(tensor, hook) -> tensor`` to apply
                to the encoded feature activations before decoding.  The hook receives a
                ``_DummyHookPoint`` with the SAE's ``hook_sae_acts_post`` sub-hook name.

        Returns:
            Tuple of ``(feature_acts_proxy, act_input_proxy, resolved_hook)``.
        """
        hook_name: str = sae.cfg.metadata.hook_name
        resolved = self._resolver.resolve_for_envoy(hook_name)
        envoy = _navigate_envoy(model, resolved.module_path)

        act_proxy = _read_envoy_activation(envoy, resolved)

        # NNsight reads post-merge activations (3D: [batch, seq, n_heads*d_head])
        # for hook_z hooks because HF attention merges heads before calling
        # c_proj.  The SAE's reshape_fn_in expects pre-merge 4D input
        # ([batch, seq, n_heads, d_head]) and would misinterpret the 3D tensor.
        # Temporarily disable reshaping since the activation is already in the
        # merged format the SAE weights operate on.
        _disable_hz = getattr(sae, "hook_z_reshaping_mode", False)
        if _disable_hz:
            sae.turn_off_forward_pass_hook_z_reshaping()

        try:
            feature_acts = sae.encode(act_proxy)

            if hook_fn is not None:
                dummy = _DummyHookPoint(name=f"{hook_name}.hook_sae_acts_post")
                feature_acts = hook_fn(feature_acts, dummy)

            sae_out = sae.decode(feature_acts)
            _write_envoy_activation(envoy, resolved, sae_out)
        finally:
            if _disable_hz:
                sae.turn_on_forward_pass_hook_z_reshaping()

        return feature_acts, act_proxy, resolved

    @staticmethod
    def _get_hf_model(model: Any) -> torch.nn.Module:
        """Extract the raw HuggingFace PyTorch model from an NNsight ``LanguageModel``.

        NNsight wraps the HF model; ``_module`` holds the underlying module.
        Falls back to returning ``model`` directly if it is already a plain ``nn.Module``.
        """
        if hasattr(model, "_model"):
            return model._model  # type: ignore[return-value]
        if hasattr(model, "_module"):
            return model._module  # type: ignore[return-value]
        return model

    @staticmethod
    def _get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
        """Navigate to a sub-module given a dot-separated path string.

        Uses ``torch.nn.Module.get_submodule`` which handles ``ModuleList`` indexing
        (e.g., ``"transformer.h.5"`` resolves to the 6th block).
        """
        return model.get_submodule(path)

    @staticmethod
    def _read_activation(data: Any, io_type: Literal["input", "output"]) -> Any:
        """Extract the hidden-state tensor (or NNsight proxy) from a module's input or output.

        For **output**: most HF transformer layers return a tuple ``(hidden_states, ...)``;
        MLP/LayerNorm layers may return a single tensor.

        For **input**: inputs arrive as an ``args`` tuple; ``args[0]`` is typically the
        hidden-state tensor.
        """
        if io_type == "output":
            if isinstance(data, tuple):
                return data[0]
            return data
        else:  # input
            if isinstance(data, tuple) and len(data) > 0:
                return data[0]
            return data

    @staticmethod
    def _replace_activation(original: Any, replacement: torch.Tensor, io_type: Literal["input", "output"]) -> Any:
        """Create a modified copy of the module output/input with the hidden-state tensor replaced."""
        if io_type == "output":
            if isinstance(original, tuple):
                return (replacement,) + original[1:]
            return replacement
        else:  # input — returns modified args tuple
            if isinstance(original, tuple):
                return (replacement,) + original[1:]
            return replacement

    # ------------------------------------------------------------------
    # Protocol: fwd (minimal forward pass)
    # ------------------------------------------------------------------

    def fwd(self, model: Any, batch: dict[str, Any]) -> torch.Tensor:
        """Minimal forward pass via NNsight trace — returns logits.

        Wraps the model call in a trace context so that
        ``LanguageModel._prepare_input`` correctly maps batch keys (e.g.,
        ``input`` → ``input_ids`` for HuggingFace models).  Calling the
        NNsight model directly (outside a trace) would bypass this mapping.
        """
        with model.trace() as tracer:
            with tracer.invoke(**batch):
                saved_logits = _nnsight.save(model.output.logits)
        return saved_logits

    # ------------------------------------------------------------------
    # Protocol: fwd_w_cache_and_latent_models
    # ------------------------------------------------------------------

    def fwd_w_cache_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        names_filter: NamesFilter,
    ) -> tuple[torch.Tensor, Any]:
        """Run a forward pass with activation caching and SAE splicing via NNsight trace.

        Uses NNsight's trace/proxy API to set up interventions declaratively.  Inside the
        trace context, SAE ``encode``/``decode`` operations execute on proxy objects, and
        envoy output assignment creates interventions that replace activations during the
        deferred forward pass.  Cached activations are extracted via ``nnsight.save()`` and
        materialized to real tensors after the trace exits.

        Supports both local and remote (NDIF) NNsight models.

        Args:
            model: NNsight ``LanguageModel``.
            batch: Input batch dict (passed to ``tracer.invoke()``).
            latent_model_handles: SAE/transcoder handles (from SAELens).
            names_filter: Filter specifying which hook activations to cache.

        Returns:
            Tuple of ``(logits, NNsightActivationCacheAdapter)``.
        """
        saved_cache: dict[str, Any] = {}
        saved_logits: Any = None

        with model.trace() as tracer:
            with tracer.invoke(**batch):
                for sae in latent_model_handles:
                    hook_name: str = sae.cfg.metadata.hook_name
                    feature_acts, act_proxy, _ = self._splice_sae(model, sae)

                    # Cache requested sub-hook activations via nnsight.save()
                    if _matches_names_filter(f"{hook_name}.hook_sae_input", names_filter):
                        saved_cache[f"{hook_name}.hook_sae_input"] = _nnsight.save(act_proxy)
                    if _matches_names_filter(f"{hook_name}.hook_sae_acts_post", names_filter):
                        saved_cache[f"{hook_name}.hook_sae_acts_post"] = _nnsight.save(feature_acts)
                # NOTE: model.output.logits is safe for single-invoke contexts.
                # For multi-invoke, use model.lm_head.output (see fwd_w_hooks_batched).
                saved_logits = _nnsight.save(model.output.logits)

        # After trace exits: nnsight.save() resolves to real tensors directly
        return saved_logits, NNsightActivationCacheAdapter(saved_cache)

    # ------------------------------------------------------------------
    # Protocol: fwd_w_hooks_and_latent_models
    # ------------------------------------------------------------------

    def fwd_w_hooks_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        fwd_hooks: list[tuple[str, Any]],
        clear_contexts: bool = True,
    ) -> torch.Tensor:
        """Run a forward pass with SAE splicing and user hooks (e.g. ablation) via NNsight trace.

        Uses NNsight's trace/proxy API.  User-provided ``fwd_hooks`` (e.g., ablation hooks)
        are applied to the encoded feature-activation proxies **before** SAE decoding, so the
        intervention is reflected in the deferred forward pass.

        Supports both local and remote (NDIF) NNsight models.

        Args:
            model: NNsight ``LanguageModel``.
            batch: Input batch dict (passed to ``tracer.invoke()``).
            latent_model_handles: SAE/transcoder handles.
            fwd_hooks: List of ``(hook_name, hook_fn)`` tuples.  ``hook_fn`` has the TL
                signature ``(tensor, hook) -> tensor``.
            clear_contexts: Unused (present for protocol compatibility with TL backend).

        Returns:
            Model output logits.
        """
        saved_logits: Any = None

        with model.trace() as tracer:
            with tracer.invoke(**batch):
                for sae in latent_model_handles:
                    hook_name: str = sae.cfg.metadata.hook_name

                    # Collect user hooks relevant to this SAE
                    relevant_user_hooks = [
                        (name, fn)
                        for name, fn in fwd_hooks
                        if name.startswith("blocks.") and _base_hook_matches(name, hook_name)
                    ]

                    # Build a composite hook_fn if any user hooks apply
                    composite_fn: Callable | None = None
                    if relevant_user_hooks:

                        def _make_composite(hooks: list[tuple[str, Any]]) -> Callable:
                            def _apply(tensor: Any, _hook: _DummyHookPoint) -> Any:
                                result = tensor
                                for uhook_name, uhook_fn in hooks:
                                    dummy = _DummyHookPoint(name=uhook_name)
                                    result = uhook_fn(result, dummy)
                                return result

                            return _apply

                        composite_fn = _make_composite(relevant_user_hooks)

                    self._splice_sae(model, sae, hook_fn=composite_fn)

                # NOTE: model.output.logits is safe for single-invoke contexts.
                # For multi-invoke, use model.lm_head.output (see fwd_w_hooks_batched).
                saved_logits = _nnsight.save(model.output.logits)

        return saved_logits

    # ------------------------------------------------------------------
    # Protocol: fwd_w_hooks_batched
    # ------------------------------------------------------------------

    def fwd_w_hooks_batched(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        hook_configs: Sequence[list[tuple[str, Any]]],
        clear_contexts: bool = True,
        configs_per_pass: int | None = None,
    ) -> list[torch.Tensor]:
        """Run multiple forward passes with different hook configs via NNsight multi-invoke.

        Batches all ``hook_configs`` into a single ``model.trace()`` context with one
        NNsight **input invoke** per config.  NNsight's ``Batcher`` scopes each invoke's
        interventions to its own batch slice (via ``narrow()`` / ``swap()``), so different
        SAE-splice + user-hook configurations in each invoke produce correctly isolated
        results — empirically verified to match separate-trace execution exactly.

        .. note::

            Logits are read via ``model.lm_head.output`` (not ``model.output.logits``)
            because NNsight only narrows outputs of envoy-wrapped ``nn.Module`` instances.
            The top-level model output (``CausalLMOutputWithCrossAttentions``) bypasses
            envoy narrowing and would return the full stacked batch.

        ``configs_per_pass`` controls how many configs are batched per trace to
        limit peak memory (total batch = ``len(chunk) × batch_size``).  When ``None``
        is passed, the backend falls back to its configured default. If both are
        ``None``, all configs are batched in one trace.

        See ``docs/nnsight_multi_invoke_analysis.md`` for detailed empirical evidence.

        Args:
            model: NNsight ``LanguageModel``.
            batch: Input batch dict (the **same** batch is repeated for each config).
            latent_model_handles: SAE/transcoder handles.
            hook_configs: Sequence of ``fwd_hooks`` lists, one per ablation variant.
            clear_contexts: Unused (present for protocol compatibility).
            configs_per_pass: Maximum number of configs to batch per execution context.
                ``None`` falls back to the backend default; if that is also ``None``,
                all configs are batched in one trace.

        Returns:
            List of logits tensors, one per element in ``hook_configs``.
        """
        if not hook_configs:
            return []

        all_results: list[torch.Tensor] = []
        chunk_size = configs_per_pass if configs_per_pass is not None else self._configs_per_pass
        chunk_size = chunk_size or len(hook_configs)

        for chunk_start in range(0, len(hook_configs), chunk_size):
            chunk = hook_configs[chunk_start : chunk_start + chunk_size]
            chunk_results: list[Any] = []

            with model.trace() as tracer:
                for fwd_hooks in chunk:
                    with tracer.invoke(**batch):
                        for sae in latent_model_handles:
                            hook_name: str = sae.cfg.metadata.hook_name

                            # Collect user hooks relevant to this SAE
                            relevant_user_hooks = [
                                (name, fn)
                                for name, fn in fwd_hooks
                                if name.startswith("blocks.") and _base_hook_matches(name, hook_name)
                            ]

                            # Build composite hook_fn
                            composite_fn: Callable | None = None
                            if relevant_user_hooks:

                                def _make_composite(hooks: list[tuple[str, Any]]) -> Callable:
                                    def _apply(tensor: Any, _hook: _DummyHookPoint) -> Any:
                                        result = tensor
                                        for uhook_name, uhook_fn in hooks:
                                            dummy = _DummyHookPoint(name=uhook_name)
                                            result = uhook_fn(result, dummy)
                                        return result

                                    return _apply

                                composite_fn = _make_composite(relevant_user_hooks)

                            self._splice_sae(model, sae, hook_fn=composite_fn)

                        # Use model.lm_head.output (narrowed per invoke) instead of
                        # model.output.logits (not narrowed, returns full stacked batch).
                        chunk_results.append(_nnsight.save(model.lm_head.output))

            all_results.extend(chunk_results)

        return all_results

    # ------------------------------------------------------------------
    # Protocol: fwd_w_grads_and_latent_models
    # ------------------------------------------------------------------

    def fwd_w_grads_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        fwd_hooks: list[tuple[Any, Any]],
        bwd_hooks: list[tuple[Any, Any]],
        backward_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run forward + backward with SAE splicing and gradient caching via NNsight trace.

        Uses NNsight's trace/proxy API so the entire forward + backward pass is expressed
        declaratively and can run on both local and remote (NDIF) models.

        Inside the trace:

        1. SAEs are spliced onto activation proxies (encode → decode → write back).
        2. ``backward_fn`` is applied to the logits proxy, producing a scalar proxy.
        3. NNsight's ``with scalar.backward():`` context captures gradients via proxy
           ``.grad`` access (backed by ``BackwardsTracer`` which patches ``torch.Tensor.grad``).
        4. Forward activations and gradients are materialized via ``nnsight.save()``.

        After the trace exits, the materialized tensors are fed through the TL-style
        ``fwd_hooks`` / ``bwd_hooks`` cache functions to populate ``analysis_cfg.cache_dict``.

        Args:
            model: NNsight ``LanguageModel``.
            batch: Input batch dict (passed to ``tracer.invoke()``).
            latent_model_handles: SAE/transcoder handles.
            fwd_hooks: Forward cache hooks ``[(names_filter, cache_fn), ...]``.
            bwd_hooks: Backward cache hooks ``[(names_filter, cache_fn), ...]``.
            backward_fn: ``logits -> scalar`` closure provided by the analysis op.  Applied
                to logits proxy inside the trace; must use only standard PyTorch ops so
                NNsight can trace through it.

        Returns:
            Raw model output logits (real tensor, after trace materialisation).
        """
        # Collectors for saved proxies, keyed by TL-style sub-hook name
        saved_fwd: dict[str, Any] = {}  # name -> SaveProxy (fwd activations)
        saved_grad: dict[str, Any] = {}  # name -> SaveProxy (gradients)

        # Guard grad state: NNsight's trace + backward execution may leave
        # torch.is_grad_enabled() as False.  The context manager restores the
        # original state on exit.
        with torch.set_grad_enabled(True), model.trace() as tracer:
            with tracer.invoke(**batch):
                # ---- SAE splicing via _splice_sae ----
                for sae in latent_model_handles:
                    hook_name: str = sae.cfg.metadata.hook_name
                    feature_acts, act_proxy, _ = self._splice_sae(model, sae)

                    # Save forward activations that match any names_filter
                    input_key = f"{hook_name}.hook_sae_input"
                    acts_key = f"{hook_name}.hook_sae_acts_post"
                    if any(_matches_names_filter(input_key, nf) for nf, _ in fwd_hooks):
                        saved_fwd[input_key] = _nnsight.save(act_proxy)
                    if any(_matches_names_filter(acts_key, nf) for nf, _ in fwd_hooks):
                        saved_fwd[acts_key] = _nnsight.save(feature_acts)

                    # Store proxy refs for gradient capture below
                    if any(_matches_names_filter(input_key, nf) for nf, _ in bwd_hooks):
                        saved_grad[f"_proxy_{input_key}"] = act_proxy
                    if any(_matches_names_filter(acts_key, nf) for nf, _ in bwd_hooks):
                        saved_grad[f"_proxy_{acts_key}"] = feature_acts

                # ---- Forward + backward ----
                logits_proxy = model.output.logits
                scalar = backward_fn(logits_proxy)

                with scalar.backward():  # type: ignore[union-attr]  # nnsight patches Tensor.backward() to return BackwardsTracer context manager
                    # Capture gradients for each proxy we stored above.
                    # IMPORTANT: NNsight's BackwardsMediator processes .grad
                    # requests sequentially. During backward(), gradient hooks
                    # fire in reverse layer order (deeper layers first). To
                    # avoid a dangling mediator error we must request .grad in
                    # the same reverse order.
                    proxy_keys = [k for k in saved_grad if k.startswith("_proxy_")]
                    for proxy_key in reversed(proxy_keys):
                        real_key = proxy_key[len("_proxy_") :]
                        saved_grad[real_key] = _nnsight.save(saved_grad.pop(proxy_key).grad)

                saved_logits = _nnsight.save(logits_proxy)

        # ---- Post-trace: populate cache_dict via TL-style hooks ----
        _apply_saved_to_cache_hooks(saved_fwd, fwd_hooks)
        _apply_saved_to_cache_hooks(saved_grad, bwd_hooks)

        return saved_logits

    # ------------------------------------------------------------------
    # Protocol: wrap_activation_cache
    # ------------------------------------------------------------------

    def wrap_activation_cache(
        self,
        cache_dict: dict[str, Any],
        model: Any,
    ) -> Any:
        """Wrap a raw activation dict in a ``NNsightActivationCacheAdapter``.

        If the input is already a ``NNsightActivationCacheAdapter``, returns it unchanged.

        Args:
            cache_dict: Raw dict mapping hook names to activation tensors.
            model: The model instance (unused for NNsight backend).

        Returns:
            ``NNsightActivationCacheAdapter`` wrapping the dict.
        """
        if isinstance(cache_dict, NNsightActivationCacheAdapter):
            return cache_dict
        return NNsightActivationCacheAdapter(cache_dict)


# ==============================================================================
# Hook matching & cache population helpers
# ==============================================================================


def _base_hook_matches(fwd_hook_name: str, sae_hook_name: str) -> bool:
    """Check if a forward hook name targets activations under a given SAE hook.

    E.g., ``"blocks.5.hook_resid_post.hook_sae_acts_post"`` matches SAE at
    ``"blocks.5.hook_resid_post"`` but not ``"blocks.5.hook_resid_pre"``.
    """
    layer, base_name, _ = HookNameResolver.parse_hook_name(fwd_hook_name)
    sae_layer, sae_base, _ = HookNameResolver.parse_hook_name(sae_hook_name)
    return layer == sae_layer and base_name == sae_base


def _apply_saved_to_cache_hooks(
    saved: dict[str, Any],
    hooks: list[tuple[Any, Callable]],
) -> None:
    """Feed materialized saved-proxy tensors through TL-style cache hooks.

    After an NNsight trace exits, ``saved`` maps TL-style sub-hook names to
    tensors (resolved by ``nnsight.save()``).  This function iterates the
    ``hooks`` list (``[(names_filter, cache_fn), ...]``) and calls matching
    ``cache_fn(tensor, dummy_hook)`` for each saved activation.

    Args:
        saved: Dict mapping sub-hook names (e.g., ``"blocks.5.hook_resid_post.hook_sae_input"``)
            to resolved tensors.
        hooks: TL-style hook list ``[(names_filter, cache_fn), ...]``.
    """
    for sub_name, tensor in saved.items():
        for names_filter, cache_fn in hooks:
            if _matches_names_filter(sub_name, names_filter):
                dummy = _DummyHookPoint(name=sub_name)
                cache_fn(tensor, dummy)


def _apply_tl_cache_hooks(
    hook_name: str,
    feature_acts: torch.Tensor,
    sae_input: torch.Tensor,
    tl_hooks: list[tuple[Any, Callable]],
    *,
    is_backward: bool = False,
) -> None:
    """Apply TL-style cache hooks for SAE sub-hook names.

    ``tl_hooks`` come from ``analysis_cfg.fwd_hooks`` / ``analysis_cfg.bwd_hooks`` and are
    structured as ``[(names_filter, cache_fn), ...]``.  ``cache_fn`` has signature
    ``(tensor, hook) -> None`` where ``hook.name`` provides the TL-style key.

    This generates the SAE sub-hook names (``hook_sae_input``, ``hook_sae_acts_post``)
    and calls matching cache functions.

    .. note::

        Retained for the ``fwd_w_hooks_and_latent_models`` user-hook matching path.
        Gradient methods now use :func:`_apply_saved_to_cache_hooks` instead.
    """
    sub_hooks = {
        f"{hook_name}.hook_sae_acts_post": feature_acts,
        f"{hook_name}.hook_sae_input": sae_input,
    }

    for names_filter, cache_fn in tl_hooks:
        for sub_name, tensor in sub_hooks.items():
            if _matches_names_filter(sub_name, names_filter):
                dummy = _DummyHookPoint(name=sub_name)
                cache_fn(tensor, dummy)


__all__ = [
    "NNsightActivationCacheAdapter",
    "NNsightModelBackend",
]
