"""Partial views over ``AnalysisStore``-shaped data, for cases that must not hold a live store.

Promoted from the core test helpers: a memory-heavy analysis run is cheaper to keep as a deep copy of the
few fields a case reads than as the store itself, and the few helpers downstream of a store only need
``by_latent_model`` to keep working on the subset.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from interpretune.analysis.core import LatentAnalysisDict


class ExtractedAnalysisStore:
    """Minimal ``AnalysisStore`` view backed by a selected subset of fields."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def __deepcopy__(self, memo: dict[int, Any]) -> ExtractedAnalysisStore:
        copied = type(self)(**deepcopy(self._fields, memo))
        memo[id(self)] = copied
        return copied

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    @property
    def field_names(self) -> tuple[str, ...]:
        """The names of the fields this view carries."""
        return tuple(self._fields)

    def by_latent_model(self, field_name: str, stack_latents: bool = True) -> LatentAnalysisDict:
        """Match ``AnalysisStore.by_latent_model`` for selected nested fields."""
        values = getattr(self, field_name)
        assert values, f"No values found for field {field_name}"
        if not isinstance(values[0], dict):
            raise TypeError(
                f"Values for field {field_name} must be dictionaries to be transformed into a LatentAnalysisDict"
            )
        result = LatentAnalysisDict()
        for latent_model in values[0].keys():
            if isinstance(values[0][latent_model], dict) and stack_latents:
                batch_tensors = []
                for batch in values:
                    latent_tensors = [tensor for tensor in batch[latent_model].values()]
                    batch_tensors.append(torch.stack(latent_tensors) if latent_tensors else None)
                result[latent_model] = batch_tensors  # type: ignore[assignment]
            else:
                result[latent_model] = [  # type: ignore[assignment]
                    None if isinstance(batch[latent_model], list) and not batch[latent_model] else batch[latent_model]
                    for batch in values
                ]
        return result


def extract_store_fields(store: Any, field_names: tuple[str, ...] | list[str]) -> ExtractedAnalysisStore:
    """Deep-copy ``field_names`` off ``store`` into a detached view."""
    return ExtractedAnalysisStore(**{name: deepcopy(getattr(store, name)) for name in field_names})
