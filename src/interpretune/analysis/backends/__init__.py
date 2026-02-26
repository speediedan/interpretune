"""Model backends for analysis operations.

Provides the ``ModelBackend`` protocol and backend implementations for different model execution
frameworks (TransformerLens, nnsight, etc.).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

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
        the backend calls them after the trace completes with materialised tensors.

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
    "ModelBackend",
]
