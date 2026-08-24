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
from typing import Any, ClassVar

import torch

_MISSING = object()

AnalysisScope = str
DEFAULT_ANALYSIS_SCOPES: tuple[AnalysisScope, ...] = ("analysis_batch", "batch", "run", "row", "store")


@dataclass(frozen=True)
class OpStateSpec:
    """An op's declared cross-batch state (the ``op_state`` trait in an op's YAML).

    Declaring ``op_state`` is what authorizes an op to accumulate across batches; the input and
    output schemas remain the contract for per-batch data. ``fields`` is the namespace: writing a
    name that is not declared raises instead of silently creating it, which is what made the previous
    ``setattr(store, ...)`` contract invisible to the dispatcher and to reviewers.

    Args:
        fields: the declared state field names.
        scope: state lifetime. Only ``"run"`` is supported today.
        reset_each_epoch: clear state at each analysis-epoch boundary. Defaults to ``False``, i.e.
            accumulate across epochs.
    """

    fields: tuple[str, ...]
    scope: str = "run"
    reset_each_epoch: bool = False

    SUPPORTED_SCOPES: ClassVar[tuple[str, ...]] = ("run",)
    RECOGNIZED_KEYS: ClassVar[frozenset[str]] = frozenset({"fields", "scope", "reset_each_epoch"})

    def __post_init__(self) -> None:
        if self.scope not in self.SUPPORTED_SCOPES:
            raise ValueError(f"Unsupported op_state scope {self.scope!r} (supported: {self.SUPPORTED_SCOPES})")
        if not self.fields:
            raise ValueError("op_state must declare at least one field")
        duplicates = sorted({name for name in self.fields if list(self.fields).count(name) > 1})
        if duplicates:
            raise ValueError(f"op_state declares duplicate fields: {duplicates}")
        invalid = sorted(name for name in self.fields if not str(name).isidentifier())
        if invalid:
            raise ValueError(f"op_state field names must be valid identifiers, got: {invalid}")

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any] | "OpStateSpec" | None) -> "OpStateSpec | None":
        """Build a spec from a raw YAML mapping (``None`` when the op declares no state)."""
        if raw is None:
            return None
        if isinstance(raw, OpStateSpec):
            return raw
        if not isinstance(raw, Mapping):
            raise ValueError(f"op_state must be a mapping, got {type(raw).__name__}")
        unrecognized = sorted(set(raw) - cls.RECOGNIZED_KEYS)
        if unrecognized:
            raise ValueError(f"Unrecognized op_state keys: {unrecognized} (recognized: {sorted(cls.RECOGNIZED_KEYS)})")
        declared_fields = raw.get("fields") or ()
        if isinstance(declared_fields, (str, bytes)):
            # `fields: my_field` is a natural YAML slip for a single-field op, and tuple() would
            # silently shred it into one field per character. Declaring exists to fail at load.
            raise ValueError(
                f"op_state fields must be a sequence of names, got a bare {type(declared_fields).__name__} "
                f"({declared_fields!r}); write it as a list, e.g. fields: [{declared_fields!r}]"
            )
        return cls(
            fields=tuple(declared_fields),
            scope=str(raw.get("scope", "run")),
            reset_each_epoch=bool(raw.get("reset_each_epoch", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to the raw YAML/cache mapping form."""
        return {"fields": list(self.fields), "scope": self.scope, "reset_each_epoch": self.reset_each_epoch}


class OpStateStore:
    """Lifecycle-managed container for one op's declared cross-batch state.

    Reads and writes are restricted to the declared field set, so a typo raises where the previous
    ``getattr``/``setattr``-on-the-store contract silently returned ``None``. Ownership is the
    ``AnalysisCfg`` for the run: created on demand, cleared at run start, at epoch boundaries when
    the op asks for it, and at run end.
    """

    __slots__ = ("_spec", "_values")

    def __init__(self, spec: OpStateSpec) -> None:
        self._spec = spec
        self._values: dict[str, Any] = {}

    @property
    def spec(self) -> OpStateSpec:
        """The spec declaring which cross-batch state fields this store accepts."""
        return self._spec

    @property
    def declared(self) -> tuple[str, ...]:
        """The declared field names.

        Anything outside this tuple is rejected rather than silently stored.
        """
        return self._spec.fields

    def _validate(self, name: str) -> None:
        if name not in self._spec.fields:
            raise KeyError(f"{name!r} is not a declared op_state field (declared: {list(self._spec.fields)})")

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def __getitem__(self, name: str) -> Any:
        self._validate(name)
        return self._values[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._validate(name)
        self._values[name] = value

    def __len__(self) -> int:
        return len(self._values)

    def get(self, name: str, default: Any = None) -> Any:
        """Return a declared field's value, or ``default`` when it has not been set yet."""
        self._validate(name)
        return self._values.get(name, default)

    def set(self, name: str, value: Any) -> None:
        """Set a declared field's value."""
        self[name] = value

    def update(self, **values: Any) -> None:
        """Set several declared fields at once."""
        for name, value in values.items():
            self[name] = value

    def clear(self) -> None:
        """Drop all accumulated state, leaving the declared namespace intact."""
        self._values.clear()

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the currently-set state (for inspection and tests)."""
        return dict(self._values)

    def __getstate__(self) -> dict[str, Any]:
        """Serialize the declaration but never the accumulated values.

        The container lives on an ``AnalysisCfg``, and configs get pickled and yaml-dumped. Run-scoped
        accumulator state is meaningless outside its run and can hold sizable tensors, so it is
        dropped rather than embedded in a config dump or checkpoint.
        """
        return {"spec": self._spec}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._spec = state["spec"]
        self._values = {}

    def __repr__(self) -> str:
        return f"OpStateStore(scope={self._spec.scope!r}, set={sorted(self._values)}, declared={list(self.declared)})"


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
    # Declared cross-batch op state. Deliberately NOT an input scope: `resolve_scope` and
    # DEFAULT_ANALYSIS_SCOPES ignore it, so read precedence is unchanged and nothing resolves
    # accumulator state by accident. It is read-write working state, not a resolved input.
    op_state: "OpStateStore | None" = None

    def merged(self, other: "AnalysisInputs | Mapping[str, Any] | None") -> "AnalysisInputs":
        """Return a new inputs object with ``other`` layered over this one, per scope.

        Non-destructive: neither operand is mutated. A scope present in ``other`` replaces this one's
        entirely rather than deep-merging, so a caller can reason about which object supplied a value.
        """
        other_inputs = coerce_analysis_inputs(other)
        if other_inputs is None:
            return AnalysisInputs(
                row=self.row, batch=self.batch, run=self.run, store=self.store, op_state=self.op_state
            )

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
            op_state=other_inputs.op_state if other_inputs.op_state is not None else self.op_state,
        )

    def resolve_scope(self, scope: AnalysisScope, field_name: str, batch_idx: int | None = None) -> Any:
        """Look ``field_name`` up in exactly ONE scope, without falling back to any other.

        Single-scope by design: precedence across scopes belongs to
        :meth:`AnalysisValueResolver.resolve`, so having this method fall back too would make the
        effective precedence depend on which entry point a caller happened to use. The ``row`` scope is
        the one exception, and it is a lookup rather than a fallback -- a row value may live on the
        passed row or in the store's row at ``batch_idx``, and both are the same scope.

        Raises:
            ValueError: ``scope`` is not a recognized analysis scope.
        """
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
        """Resolve ``field_name`` across ``scopes`` in order, returning the first value found.

        This is the one place analysis-input precedence is expressed. ``op_state`` is deliberately not
        among the default scopes: it is read-write cross-batch working state, not a resolved input, and
        including it would let accumulator state satisfy an input lookup by accident.
        """
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
    "OpStateSpec",
    "OpStateStore",
]
