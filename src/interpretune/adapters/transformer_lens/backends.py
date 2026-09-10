"""TransformerLens model backend implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, cast

import torch

from interpretune.analysis.backends import (
    ModelBackendCapability,
    CaptureSupport,
    InterventionSupport,
    LatentModelSupport,
    InterventionDict,
    InterventionValue,
    apply_intervention,
    build_intervention_dict,
    expand_intervention_patterns,
    get_intervention_target_shape,
)
from interpretune.analysis.backends.hook_mapping import SUBHOOK_SUFFIXES
from interpretune.protocol import NamesFilter


def _iter_hook_aliases(model: Any) -> dict[str, list[str]]:
    alias_to_canonical: dict[str, list[str]] = {}
    for attr_name in ("hook_aliases",):
        raw_aliases = getattr(model, attr_name, None)
        if not raw_aliases:
            continue
        for alias, canonical in dict(raw_aliases).items():
            values = canonical if isinstance(canonical, list) else [canonical]
            alias_to_canonical.setdefault(alias, []).extend(str(value) for value in values)

    if hasattr(model, "_collect_hook_aliases_from_registry"):
        for alias, canonical in model._collect_hook_aliases_from_registry().items():  # type: ignore[attr-defined]
            alias_to_canonical.setdefault(alias, []).append(str(canonical))
    return alias_to_canonical


def _normalize_names_filter(
    model: Any, names_filter: NamesFilter, latent_model_handles: list[Any] | None = None
) -> tuple[NamesFilter, dict[str, list[str]]]:
    """Map requested capture names onto the hooks this model exposes, through the vocabulary's spellings.

    A caller asks for a point in any accepted spelling (``blocks.5.hook_in``); a TransformerBridge exposes that name
    while a legacy HookedTransformer exposes ``blocks.5.hook_resid_pre`` for the same tensor. The same expansion the
    intervention path uses resolves it, so capture and intervention agree on what a name means. Callables and names
    the vocabulary does not know pass through untouched. Returns the filter to hand TransformerLens plus
    ``{actual hook name: requested name}`` so the cache can be re-keyed as the caller spelled it.
    """
    if callable(names_filter):
        return _wrap_callable_filter(model, names_filter, latent_model_handles)
    requested_names = [names_filter] if isinstance(names_filter, str) else list(names_filter or [])
    if not requested_names:
        return names_filter, {}
    from interpretune.analysis.backends.interventions import expand_intervention_patterns

    available = _build_available_hook_map(model, latent_model_handles=latent_model_handles)
    actual_for: dict[str, str] = {}
    for name in requested_names:
        if name in available and available[name] == name:
            continue  # already an actual hook name
        try:
            matches = expand_intervention_patterns([name], available)[name]
        except ValueError:
            # TransformerLens drops a list entry it does not know without a word (measured on both wrappers), and
            # a cache one entry short is indistinguishable from a complete one at the call site; refuse by name.
            sample = sorted(n for n in available if n.startswith(name.split(".")[0]))[:6]
            raise ValueError(
                f"{name!r} names no hook this model exposes, in any spelling the vocabulary knows; nearby names: "
                f"{sample}"
            ) from None
        if len(matches) == 1 and matches[0] != name:
            actual_for[matches[0]] = name
    if not actual_for:
        return names_filter, {}
    resolved = [actual_for.get(n, n) for n in requested_names]
    reverse = {actual: [requested] for actual, requested in actual_for.items()}
    for n in requested_names:
        actual = next((a for a, r in actual_for.items() if r == n), None)
        resolved[requested_names.index(n)] = actual or n
    return (resolved[0] if isinstance(names_filter, str) else resolved), reverse


def _wrap_callable_filter(
    model: Any, accept: Callable[[str], bool], latent_model_handles: list[Any] | None = None
) -> tuple[Callable[[str], bool], dict[str, list[str]]]:
    """A callable filter written in one spelling matches the hook this model exposes under another.

    TransformerLens applies a callable filter to the names in its own ``hook_dict``: a legacy HookedTransformer
    spells the block input ``blocks.5.hook_resid_pre`` and a bridge ``blocks.5.hook_in``, and an SAE's activations
    are ``<the SAE's hook_name>.hook_sae_acts_post`` on one and ``<canonical>.hook_sae_acts_post`` on the other. A
    filter resolved from a caller's list accepts the caller's spellings only, so on the other model it matched
    nothing: the cache came back empty and nothing raised. The wrapper accepts a hook when the filter accepts the
    hook's own name or any spelling of it (the vocabulary's, and the model's alias registry with SAE sub-hooks),
    and records ``{actual: accepted spelling}`` so the cache is re-keyed as the caller spelled it. The mapping
    fills during the forward, which is why it is returned as the same object the restore step reads.
    """
    from interpretune.analysis.points.vocabulary import UnknownPointError, spellings

    available = _build_available_hook_map(model, latent_model_handles=latent_model_handles)
    aliases_of: dict[str, list[str]] = {}
    for spelling, actual in available.items():
        if spelling != actual:
            aliases_of.setdefault(actual, []).append(spelling)
    requested: dict[str, list[str]] = {}
    decided: dict[str, bool] = {}

    def wrapped(name: str) -> bool:
        if name in decided:
            return decided[name]
        if accept(name):
            decided[name] = True
            return True
        candidates = list(aliases_of.get(name, ()))
        try:
            candidates.extend(s for s in spellings(name) if s != name)
        except UnknownPointError:
            pass
        # every accepted spelling, not the first: a caller asking for two spellings of one tensor (`ln2.hook_out`
        # and `mlp.hook_in`) gets the cache keyed under both, else the second reads as uncaptured
        accepted = [spelling for spelling in candidates if accept(spelling)]
        if accepted:
            requested[name] = accepted
            decided[name] = True
            return True
        decided[name] = False
        return False

    return wrapped, requested


def _restore_requested_names(cache: Any, requested: dict[str, list[str]]) -> Any:
    """Re-key cached activations captured under a model-specific spelling back to the name(s) the caller used."""
    if not requested:
        return cache
    store = getattr(cache, "cache_dict", None)
    target = store if isinstance(store, dict) else cache
    if not isinstance(target, dict):
        return cache
    for actual, names in requested.items():
        for name in names:
            if actual in target and name not in target:
                target[name] = target[actual]
    return cache


def _normalize_hooks(
    model: Any, hooks: Sequence[tuple[Any, Any]], latent_model_handles: list[Any] | None = None
) -> list[tuple[Any, Any]]:
    """Resolve each string hook name in ``hooks`` to the hook this model exposes, or refuse it by name.

    The same spellings the cache and intervention paths accept apply to a hook: a caller may name an SAE's
    activations as the SAE's own metadata spells them (``blocks.0.hook_resid_pre.hook_sae_acts_post``) while a
    TransformerBridge exposes ``blocks.0.hook_in.hook_sae_acts_post``. TransformerLens applies a string hook only
    when the exact name is in its ``hook_dict`` or its alias registry and otherwise SKIPS it without raising, so
    an unresolved spelling was a forward with no edit and plausible logits. Callables select hooks by predicate
    and pass through unchanged.
    """
    if not hooks:
        return list(hooks)
    available: dict[str, str] | None = None
    resolved: list[tuple[Any, Any]] = []
    for selector, fn in hooks:
        if callable(selector):
            # a predicate selector has the spelling gap a callable names_filter has; same wrapper, no re-keying
            predicate = cast(Callable[[str], bool], selector)
            resolved.append((_wrap_callable_filter(model, predicate, latent_model_handles)[0], fn))
            continue
        if available is None:
            available = _build_available_hook_map(model, latent_model_handles=latent_model_handles)
        try:
            matches = expand_intervention_patterns([selector], available)[selector]
        except ValueError:
            sample = sorted(n for n in available if n.startswith(selector.split(".")[0]))[:8]
            raise ValueError(
                f"hook {selector!r} names no hook this model exposes, in any spelling the vocabulary knows; "
                f"nearby names: {sample}"
            ) from None
        resolved.extend((actual, fn) for actual in matches)
    return resolved


def _build_available_hook_map(model: Any, latent_model_handles: list[Any] | None = None) -> dict[str, str]:
    candidate_map: dict[str, str] = {str(name): str(name) for name in model.hook_dict}
    alias_to_canonical = _iter_hook_aliases(model)

    if latent_model_handles:
        for sae in latent_model_handles:
            base_name = str(sae.cfg.metadata.hook_name)
            canonical_names = alias_to_canonical.get(base_name, [base_name])
            for canonical_name in canonical_names:
                for suffix in sorted(SUBHOOK_SUFFIXES):
                    candidate_map.setdefault(f"{canonical_name}.{suffix}", f"{canonical_name}.{suffix}")

    actual_names = list(candidate_map.values())
    for alias, canonicals in alias_to_canonical.items():
        for canonical in canonicals:
            if canonical in actual_names:
                candidate_map.setdefault(alias, canonical)
            for actual_name in actual_names:
                if actual_name.startswith(canonical + "."):
                    suffix = actual_name[len(canonical) :]
                    candidate_map.setdefault(alias + suffix, actual_name)

    return candidate_map


def _hf_architecture(model: Any) -> str:
    """The HuggingFace class name the component maps are keyed by, from either TransformerLens wrapper."""
    cfg = getattr(model, "cfg", None)
    for attr in ("architecture", "original_architecture"):
        value = getattr(cfg, attr, None)
        if isinstance(value, str) and value:
            return value
    original = getattr(model, "original_model", None)
    if original is not None:
        return type(original).__name__
    raise ValueError(
        f"cannot tell which HuggingFace architecture {type(model).__name__} wraps: its cfg carries neither "
        "'architecture' nor 'original_architecture'"
    )


class TLModelBackend:
    """TransformerLens model execution backend.

    Wraps TransformerLens model APIs (``run_with_cache_with_saes``,
    ``run_with_hooks_with_saes``, ``saes()``/``hooks()`` context managers)
    behind the ``ModelBackend`` protocol interface.

    Works identically with both ``HookedSAETransformer`` and
    ``SAETransformerBridge`` since they share the same API surface.
    """

    @property
    def capabilities(self) -> frozenset[ModelBackendCapability]:
        """TL implements every method group."""
        return frozenset(
            {
                ModelBackendCapability.GRADIENTS,
                ModelBackendCapability.LATENT_MODELS,
                ModelBackendCapability.ACTIVATION_INTERVENTION,
            }
        )

    @property
    def intervention_support(self) -> InterventionSupport:
        """Every scope and every mode: a TL hook receives the whole activation, so restricting an edit to the final
        token and editing every position are equally expressible, and every mode sees the current value."""
        return InterventionSupport.every()

    @property
    def latent_model_support(self) -> LatentModelSupport:
        """The batched-hooks path here is a sequential loop, not a fused execution."""
        return LatentModelSupport(batched_hooks=False)

    def capture_support(self, model: Any) -> CaptureSupport:
        """What this backend can capture on ``model``: every vocabulary point some hook of the wrapper spells.

        Derived from the model rather than written down, because the two TransformerLens wrappers differ: a bridge
        exposes the vocabulary's own spellings, a HookedTransformer exposes legacy names that the vocabulary maps to
        them where the tensor is the same, and deliberately does not map where it is not (its `hook_mlp_out` /
        `hook_attn_out` are the post-norm outputs on a sandwich-norm architecture, so `mlp.hook_out` / `attn.hook_out`
        stay unaliased and are uncapturable there by those spellings). The declaration reports that gap by name.
        """
        from interpretune.analysis.points import component_map_for
        from interpretune.analysis.points.inventory import inventory, spelled_at

        architecture = _hf_architecture(model)
        cmap = component_map_for(architecture)
        available = _build_available_hook_map(model)
        capturable: set[str] = set()
        uncapturable: dict[str, str] = {}
        for base in inventory(cmap):
            name = spelled_at(base, cmap)
            try:
                expand_intervention_patterns([name], available)
                capturable.add(base)
            except ValueError:
                uncapturable[base] = (
                    f"no hook of this {type(model).__name__} matches any spelling of {name!r}; the wrapper's grammar "
                    "has no tensor the vocabulary equates with it"
                )
        return CaptureSupport(
            capturable=frozenset(capturable),
            uncapturable=uncapturable,
            n_layers=int(model.cfg.n_layers),
            architecture=architecture,
        )

    def supports(self, capability: ModelBackendCapability) -> bool:
        """Check whether this backend supports a given capability."""
        return capability in self.capabilities

    def fwd(self, model: Any, batch: dict[str, Any]) -> torch.Tensor:
        """Minimal forward pass via TransformerLens — returns logits directly."""
        return model(**batch)

    def fwd_w_cache_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        names_filter: NamesFilter,
    ) -> tuple[torch.Tensor, Any]:
        """Run forward pass with activation caching and latent model hooks via TransformerLens."""
        names_filter, requested = _normalize_names_filter(model, names_filter, latent_model_handles)
        logits, cache = model.run_with_cache_with_saes(**batch, saes=latent_model_handles, names_filter=names_filter)
        return logits, _restore_requested_names(cache, requested)

    def fwd_w_cache(
        self,
        model: Any,
        batch: dict[str, Any],
        names_filter: NamesFilter,
    ) -> tuple[torch.Tensor, Any]:
        """Run forward pass with activation caching via TransformerLens."""
        names_filter, requested = _normalize_names_filter(model, names_filter)
        logits, cache = model.run_with_cache(**batch, names_filter=names_filter)
        return logits, _restore_requested_names(cache, requested)

    def fwd_w_hooks_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        fwd_hooks: list[tuple[str, Any]],
        clear_contexts: bool = True,
    ) -> torch.Tensor:
        """Run forward pass with custom hooks and latent model hooks via TransformerLens."""
        return model.run_with_hooks_with_saes(
            **batch,
            saes=latent_model_handles,
            clear_contexts=clear_contexts,
            fwd_hooks=_normalize_hooks(model, fwd_hooks, latent_model_handles),
        )

    def fwd_w_hooks_batched(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        hook_configs: Sequence[list[tuple[str, Any]]],
        clear_contexts: bool = True,
        configs_per_pass: int | None = None,
    ) -> list[torch.Tensor]:
        """Run multiple forward passes sequentially (TL does not support native batching).

        Falls back to calling ``fwd_w_hooks_and_latent_models`` once per config.
        ``configs_per_pass`` is accepted for API compatibility but has no effect.

        Args:
            model: TransformerLens model.
            batch: Input batch dict.
            latent_model_handles: SAE/transcoder handles.
            hook_configs: Sequence of ``fwd_hooks`` lists.
            clear_contexts: Passed through to each ``fwd_w_hooks_and_latent_models`` call.
            configs_per_pass: Ignored (present for protocol compatibility).

        Returns:
            List of logits tensors, one per element in ``hook_configs``.
        """
        return [
            self.fwd_w_hooks_and_latent_models(
                model=model,
                batch=batch,
                latent_model_handles=latent_model_handles,
                fwd_hooks=fwd_hooks,
                clear_contexts=clear_contexts,
            )
            for fwd_hooks in hook_configs
        ]

    def fwd_w_grads_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        fwd_hooks: list[tuple[Any, Any]],
        bwd_hooks: list[tuple[Any, Any]],
        backward_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run forward + backward with latent models and gradient hooks via TransformerLens.

        Sets up TransformerLens ``saes()`` and ``hooks()`` context managers with gradient
        tracking enabled.  Calls ``model(**batch)`` for the forward pass, applies
        ``backward_fn`` to the logits to obtain a scalar, and calls ``.backward()`` on it.

        TL's hook system populates the analysis config's ``cache_dict`` as a side effect
        during forward and backward execution.

        Args:
            model: TransformerLens model (HookedSAETransformer or SAETransformerBridge).
            batch: Input batch dict.
            latent_model_handles: SAE/transcoder handles.
            fwd_hooks: Forward cache hooks ``[(names_filter, cache_fn), ...]``.
            bwd_hooks: Backward cache hooks ``[(names_filter, cache_fn), ...]``.
            backward_fn: ``logits -> scalar`` to backpropagate.

        Returns:
            Raw model output logits.
        """
        fwd_hooks = _normalize_hooks(model, fwd_hooks, latent_model_handles)
        bwd_hooks = _normalize_hooks(model, bwd_hooks, latent_model_handles)
        with torch.set_grad_enabled(True):
            with model.saes(saes=latent_model_handles):
                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(**batch)
                    scalar = backward_fn(logits)
                    scalar.backward()
        return logits

    def wrap_activation_cache(
        self,
        cache_dict: dict[str, Any],
        model: Any,
    ) -> Any:
        """Wrap a raw activation dict in a TransformerLens ``ActivationCache``.

        If the input is already an ``ActivationCache``, returns it unchanged.
        """
        from transformer_lens import ActivationCache

        if isinstance(cache_dict, ActivationCache):
            return cache_dict
        return ActivationCache(cache_dict, model)

    def fwd_w_intervention(
        self,
        model: Any,
        batch: dict[str, Any],
        interventions: InterventionDict | Mapping[str, InterventionValue],
        latent_model_handles: list[Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply interventions at the last token position via TL forward hooks.

        Performs two forward passes:

        1. **Baseline**: ``run_with_cache`` to capture pre-intervention logits.
        2. **Intervention**: ``run_with_hooks`` with forward hooks built from *interventions*.
        """
        if isinstance(interventions, InterventionDict):
            expanded_matches = {hook_name: [hook_name] for hook_name in interventions.keys()}
            hook_names = list(interventions.keys())
            intervention_dict = interventions
        else:
            available_hook_map = _build_available_hook_map(model, latent_model_handles=latent_model_handles)
            expanded_matches = expand_intervention_patterns(list(interventions.keys()), available_hook_map)
            hook_names = [hook_name for matches in expanded_matches.values() for hook_name in matches]
            intervention_dict = None

        names_filter = hook_names[0] if len(hook_names) == 1 else hook_names

        # --- Baseline forward pass ---
        if latent_model_handles:
            pre_logits, cache = self.fwd_w_cache_and_latent_models(
                model=model,
                batch=batch,
                latent_model_handles=latent_model_handles,
                names_filter=names_filter,
            )
        else:
            pre_logits, cache = self.fwd_w_cache(model=model, batch=batch, names_filter=names_filter)

        if intervention_dict is None:
            hook_shapes = {
                hook_name: get_intervention_target_shape(torch.as_tensor(cache[hook_name])) for hook_name in hook_names
            }
            intervention_dict = build_intervention_dict(interventions, expanded_matches, hook_shapes)

        last_pos = int(pre_logits.shape[1] - 1)

        # --- Build hook list ---
        fwd_hooks: list[tuple[str, Callable]] = []
        for hook_name, spec_list in intervention_dict.items():
            for spec in spec_list:

                def _hook(value: torch.Tensor, hook: Any, _spec=spec, _last_pos=last_pos) -> torch.Tensor:
                    return apply_intervention(value, _spec, last_pos=_last_pos)

                fwd_hooks.append((hook_name, _hook))

        if latent_model_handles:
            post_logits = model.run_with_hooks_with_saes(
                **batch,
                saes=latent_model_handles,
                clear_contexts=True,
                fwd_hooks=fwd_hooks,
            )
        else:
            post_logits = model.run_with_hooks(**batch, fwd_hooks=fwd_hooks)
        return pre_logits, post_logits
