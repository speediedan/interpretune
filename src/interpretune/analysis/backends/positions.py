"""The position-id convention every model backend shares for padded batches."""

from __future__ import annotations

import inspect
from typing import Any

import torch


def mask_derived_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids that count only the real tokens, so a left-padded prompt starts at position 0.

    A HuggingFace forward given no ``position_ids`` uses ``arange(seq_len)``, so the pad tokens of a left-padded row
    occupy the first positions and every real token is shifted. For absolute position embeddings (gpt2) that changes
    the logits. TransformerLens 4.0's bridge derives positions from the mask exactly this way (``cumsum - 1``, pad
    positions set to 1), as HookedTransformer did, so every backend applies the same convention.
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 1)


def accepts_position_ids(forward: Any) -> bool:
    """Whether a forward callable takes ``position_ids``, so passing derived ones cannot raise a TypeError."""
    try:
        parameter = inspect.signature(forward).parameters.get("position_ids")
        return parameter is not None and parameter.kind is not parameter.POSITIONAL_ONLY
    except (TypeError, ValueError):
        return False


def install_mask_derived_positions(model: torch.nn.Module) -> None:
    """Let a model derive positions from a full attention mask when the caller leaves positions unset.

    Installing on the model covers callers that forward directly, including nnsight after it merges the rows of a trace.
    A cached step's attention mask can include the prefix, so mismatched shapes are left to the model. Explicit position
    arguments also stay with the caller.
    """
    if getattr(model, "_it_mask_derived_positions", False) or not accepts_position_ids(model.forward):
        return

    positional_names = tuple(
        parameter.name
        for parameter in inspect.signature(model.forward).parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    )

    def pre_hook(_module: torch.nn.Module, args: tuple, kwargs: dict[str, Any]) -> tuple[tuple, dict[str, Any]]:
        positional = dict(zip(positional_names, args))
        tokens = kwargs.get("input_ids", positional.get("input_ids", args[0] if args else None))
        mask = kwargs.get("attention_mask", positional.get("attention_mask"))
        if (
            isinstance(mask, torch.Tensor)
            and "position_ids" not in positional
            and kwargs.get("position_ids") is None
            and isinstance(tokens, torch.Tensor)
            and mask.shape == tokens.shape
            and not bool(mask.all())
        ):
            kwargs["position_ids"] = mask_derived_position_ids(mask)
        return args, kwargs

    model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    setattr(model, "_it_mask_derived_positions", True)
