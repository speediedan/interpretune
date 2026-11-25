"""Small measures that several example tutorials share."""

from __future__ import annotations

from pathlib import Path

import torch


def answer_margin(logits: torch.Tensor, answer_ids: tuple[int, int]) -> torch.Tensor:
    """The first answer's logit minus the second's at the last position (the answer position, with left padding),
    one value per item."""
    last = logits[:, -1, :].float()
    return last[:, answer_ids[0]] - last[:, answer_ids[1]]


def cached_revision(model_id: str) -> str:
    """The commit of the snapshot ``model_id`` loads from the local Hugging Face cache, which an artifact that
    names the model's units (a checkpoint address book, say) pins.

    Raises:
        ValueError: If the model is not in the cache (load it first).
    """
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model_id, "config.json")
    if not isinstance(path, str):
        raise ValueError(f"{model_id} is not in the local Hugging Face cache; load it before pinning its revision")
    return Path(path).parent.name
