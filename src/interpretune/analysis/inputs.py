"""Core execution-contract machinery for scoped analysis-input resolution.

This module is framework infrastructure, not op-authoring surface: it backs the ``AnalysisBatch``
scoped-lookup API (``analysis_batch.get`` / ``analysis_batch.require`` / attribute access) and the
shared execution helpers in :mod:`interpretune.analysis.execution`. Op implementations should resolve
values through the bound ``AnalysisBatch`` surface (or, transitionally, the sanctioned conveniences in
:mod:`interpretune.analysis.optools`) rather than constructing resolvers from these primitives.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

_MISSING = object()

AnalysisScope = str
DEFAULT_ANALYSIS_SCOPES: tuple[AnalysisScope, ...] = ("analysis_batch", "batch", "run", "row", "store")


def _resolve_attr_path(root: Any, *path: str) -> Any | None:
    current = root
    for attr_name in path:
        current = getattr(current, attr_name, None)
        if current is None:
            return None
    return current


def _value_for_batch(store_value: Any, batch_idx: int | None) -> Any:
    if batch_idx is None or isinstance(store_value, (str, bytes, Mapping)):
        return store_value
    if isinstance(store_value, torch.Tensor):
        return store_value if store_value.dim() == 0 else store_value[batch_idx]
    if isinstance(store_value, (list, tuple)):
        return store_value[batch_idx]
    if hasattr(store_value, "__getitem__"):
        try:
            return store_value[batch_idx]
        except Exception:
            return store_value
    return store_value


def _lookup_mapping_or_attr(container: Any, field_name: str) -> Any:
    if container is None:
        return _MISSING
    if isinstance(container, Mapping):
        try:
            value = container[field_name] if field_name in container else _MISSING
        except Exception:
            value = _MISSING
        return _MISSING if value is None else value

    value = getattr(container, field_name, _MISSING)
    if value is not _MISSING and value is not None:
        return value

    dataset = getattr(container, "dataset", None)
    column_names = getattr(dataset, "column_names", []) if dataset is not None else []
    if field_name not in column_names:
        return _MISSING

    try:
        value = container[field_name]
    except Exception:
        return _MISSING
    return _MISSING if value is None else value


def _lookup_store_row_value(store: Any, field_name: str, batch_idx: int | None) -> Any:
    if store is None or batch_idx is None:
        return _MISSING

    dataset = getattr(store, "dataset", None)
    column_names = getattr(dataset, "column_names", []) if dataset is not None else []
    if dataset is not None and field_name in column_names:
        try:
            row = dataset[batch_idx]
        except Exception:
            row = None
        if isinstance(row, Mapping):
            value = row[field_name] if field_name in row else _MISSING
            if value is not None and value is not _MISSING:
                return value

    store_value = _lookup_mapping_or_attr(store, field_name)
    if store_value is _MISSING:
        return _MISSING
    return _value_for_batch(store_value, batch_idx)


@dataclass(kw_only=True)
class AnalysisInputs:
    """Explicit scoped analysis inputs used during op execution."""

    row: Mapping[str, Any] | None = None
    batch: Mapping[str, Any] | None = None
    run: Mapping[str, Any] | None = None
    store: Any = None

    def merged(self, other: "AnalysisInputs | Mapping[str, Any] | None") -> "AnalysisInputs":
        other_inputs = coerce_analysis_inputs(other)
        if other_inputs is None:
            return AnalysisInputs(row=self.row, batch=self.batch, run=self.run, store=self.store)

        def merge_mapping(
            base: Mapping[str, Any] | None,
            override: Mapping[str, Any] | None,
        ) -> Mapping[str, Any] | None:
            if base and override:
                return {**base, **override}
            return override if override is not None else base

        return AnalysisInputs(
            row=merge_mapping(self.row, other_inputs.row),
            batch=merge_mapping(self.batch, other_inputs.batch),
            run=merge_mapping(self.run, other_inputs.run),
            store=other_inputs.store if other_inputs.store is not None else self.store,
        )

    def resolve_scope(self, scope: AnalysisScope, field_name: str, batch_idx: int | None = None) -> Any:
        if scope == "row":
            value = _lookup_mapping_or_attr(self.row, field_name)
            if value is not _MISSING:
                return value
            return _lookup_store_row_value(self.store, field_name, batch_idx)
        if scope == "batch":
            return _lookup_mapping_or_attr(self.batch, field_name)
        if scope == "run":
            return _lookup_mapping_or_attr(self.run, field_name)
        if scope == "store":
            return _lookup_mapping_or_attr(self.store, field_name)
        raise ValueError(f"Unsupported analysis input scope: {scope}")


@dataclass(kw_only=True)
class AnalysisValueResolver:
    """Resolve analysis values using explicit precedence across analysis scopes."""

    analysis_batch: Any
    analysis_inputs: AnalysisInputs
    batch_idx: int | None = None

    def resolve(
        self,
        field_name: str,
        *,
        default: Any = None,
        scopes: tuple[AnalysisScope, ...] = DEFAULT_ANALYSIS_SCOPES,
    ) -> Any:
        for scope in scopes:
            if scope == "analysis_batch":
                value = _lookup_mapping_or_attr(self.analysis_batch, field_name)
            else:
                value = self.analysis_inputs.resolve_scope(scope, field_name, batch_idx=self.batch_idx)
            if value is not _MISSING:
                return value
        return default


def coerce_analysis_inputs(value: AnalysisInputs | Mapping[str, Any] | None) -> AnalysisInputs | None:
    """Normalize user-provided analysis inputs into an AnalysisInputs object."""
    if value is None:
        return None
    if isinstance(value, AnalysisInputs):
        return value
    if isinstance(value, Mapping):
        return AnalysisInputs(run=dict(value))
    raise TypeError(f"Unsupported analysis_inputs value: {type(value).__name__}")


def get_analysis_resolver(
    analysis_batch: Any,
    module: Any,
    *,
    batch_idx: int | None = None,
    analysis_inputs: AnalysisInputs | Mapping[str, Any] | None = None,
) -> AnalysisValueResolver:
    """Build a resolver that combines config-backed and explicit analysis input scopes."""
    analysis_cfg = getattr(module, "analysis_cfg", None)
    config_inputs = AnalysisInputs(
        batch=getattr(analysis_cfg, "batch_inputs", None),
        run=getattr(analysis_cfg, "run_inputs", None),
        store=getattr(analysis_cfg, "input_store", None),
    )
    resolved_inputs = config_inputs.merged(analysis_inputs)
    return AnalysisValueResolver(analysis_batch=analysis_batch, analysis_inputs=resolved_inputs, batch_idx=batch_idx)


__all__ = [
    "AnalysisInputs",
    "AnalysisScope",
    "AnalysisValueResolver",
    "coerce_analysis_inputs",
    "DEFAULT_ANALYSIS_SCOPES",
    "get_analysis_resolver",
]
