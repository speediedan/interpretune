"""TransformerLens model backend implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch

from interpretune.analysis.backends import (
    BackendCapability,
    InterventionDict,
    InterventionValue,
    apply_intervention_to_last_token,
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


class TLModelBackend:
    """TransformerLens model execution backend.

    Wraps TransformerLens model APIs (``run_with_cache_with_saes``,
    ``run_with_hooks_with_saes``, ``saes()``/``hooks()`` context managers)
    behind the ``ModelBackend`` protocol interface.

    Works identically with both ``HookedSAETransformer`` and
    ``SAETransformerBridge`` since they share the same API surface.
    """

    @property
    def capabilities(self) -> frozenset[BackendCapability]:
        """TL supports gradients but not batched hooks (uses sequential loop)."""
        return frozenset({BackendCapability.GRADIENTS})

    def supports(self, capability: BackendCapability) -> bool:
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
        return model.run_with_cache_with_saes(**batch, saes=latent_model_handles, names_filter=names_filter)

    def fwd_w_cache(
        self,
        model: Any,
        batch: dict[str, Any],
        names_filter: NamesFilter,
    ) -> tuple[torch.Tensor, Any]:
        """Run forward pass with activation caching via TransformerLens."""
        return model.run_with_cache(**batch, names_filter=names_filter)

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
            fwd_hooks=fwd_hooks,
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
                    return apply_intervention_to_last_token(value, _spec, last_pos=_last_pos)

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
