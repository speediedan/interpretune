"""Model backends for analysis operations.

Provides the ``ModelBackend`` protocol and backend implementations for different model execution
frameworks (TransformerLens, nnsight, etc.).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

import torch

from interpretune.protocol import NamesFilter


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol defining the interface for model execution backends.

    Each backend wraps a specific framework's model execution API (e.g., TransformerLens hook-based execution, nnsight
    trace-based execution) behind a uniform interface used by analysis op implementations.
    """

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

    def fwd_w_grads_and_latent_models(
        self,
        model: Any,
        latent_model_handles: list[Any],
        fwd_hooks: list[tuple[str, Any]],
        bwd_hooks: list[tuple[str, Any]],
    ) -> AbstractContextManager[Any]:
        """Context manager for gradient computation with latent model and gradient hooks.

        Sets up the model with gradient tracking, latent model hooks, and forward/backward
        gradient caching hooks. Yields the model ready for forward calls. The caller runs
        forward + backward inside the context.

        Args:
            model: The model to prepare.
            latent_model_handles: Latent model handles to attach.
            fwd_hooks: Forward hooks for gradient caching.
            bwd_hooks: Backward hooks for gradient caching.

        Yields:
            The model with all hooks attached, ready for forward calls.
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
    "ModelBackend",
]
