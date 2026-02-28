"""TransformerLens model backend implementation."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch

from interpretune.analysis.backends import BackendCapability
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
        max_invokes_per_trace: int | None = None,
    ) -> list[torch.Tensor]:
        """Run multiple forward passes sequentially (TL does not support native batching).

        Falls back to calling ``fwd_w_hooks_and_latent_models`` once per config.
        ``max_invokes_per_trace`` is accepted for API compatibility but has no effect.

        Args:
            model: TransformerLens model.
            batch: Input batch dict.
            latent_model_handles: SAE/transcoder handles.
            hook_configs: Sequence of ``fwd_hooks`` lists.
            clear_contexts: Passed through to each ``fwd_w_hooks_and_latent_models`` call.
            max_invokes_per_trace: Ignored (present for protocol compatibility).

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
