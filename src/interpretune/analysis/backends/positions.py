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
        return "position_ids" in inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
