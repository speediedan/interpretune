"""TransformerLens model backend implementation."""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

import torch

from interpretune.analysis.backends import BackendCapability, InterventionSpec
from interpretune.protocol import NamesFilter


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
        interventions: dict[str, InterventionSpec | Sequence[InterventionSpec]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply interventions at the last token position via TL forward hooks.

        Performs two forward passes:

        1. **Baseline**: ``run_with_cache`` to capture pre-intervention logits.
        2. **Intervention**: ``run_with_hooks`` with forward hooks built from *interventions*.
        """
        # Collect the union of hook names for the baseline cache filter
        hook_names: set[str] = set()
        for pattern in interventions:
            if "*" in pattern:
                pat = re.compile(pattern.replace("*", ".*"))
                hook_names.update(n for n in model.hook_dict if pat.fullmatch(n))
            else:
                hook_names.add(pattern)

        names_filter = list(hook_names) if len(hook_names) > 1 else next(iter(hook_names))

        # --- Baseline forward pass ---
        pre_logits, _ = model.run_with_cache(**batch, names_filter=names_filter)
        last_pos = int(pre_logits.shape[1] - 1)

        # --- Build hook list ---
        fwd_hooks: list[tuple[str, Callable]] = []
        for pattern, specs in interventions.items():
            matched = (
                hook_names
                if "*" not in pattern
                else {n for n in hook_names if re.fullmatch(pattern.replace("*", ".*"), n)}
            )
            if "*" not in pattern:
                matched = {pattern}
            spec_list = [specs] if isinstance(specs, InterventionSpec) else list(specs)
            for hook_name in matched:
                for spec in spec_list:
                    vec = torch.as_tensor(spec.intervention_tensor, device=pre_logits.device, dtype=torch.float32)
                    sf = spec.scale_factor

                    if spec.mode == "replace":

                        def _hook(value: torch.Tensor, hook: Any, _v: torch.Tensor = vec) -> torch.Tensor:
                            value[:, last_pos, :] = _v
                            return value
                    elif spec.mode == "add":

                        def _hook(
                            value: torch.Tensor, hook: Any, _v: torch.Tensor = vec, _sf: float = sf
                        ) -> torch.Tensor:
                            value[:, last_pos, :] = value[:, last_pos, :] + _v * _sf
                            return value
                    elif spec.mode == "scale":

                        def _hook(
                            value: torch.Tensor, hook: Any, _v: torch.Tensor = vec, _sf: float = sf
                        ) -> torch.Tensor:
                            value[:, last_pos, :] = value[:, last_pos, :] * (_v * _sf)
                            return value
                    else:
                        raise ValueError(f"Unknown intervention mode: {spec.mode!r}")
                    fwd_hooks.append((hook_name, _hook))

        post_logits = model.run_with_hooks(**batch, fwd_hooks=fwd_hooks)
        return pre_logits, post_logits
