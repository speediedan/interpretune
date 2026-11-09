# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""The tensor-op executor registry: what applies each op, by name.

An op is a name in venngram's registry (:data:`venngram.ops.OPS`), which declares it as data (operand kinds, parameters,
targets, composition) and never executes it. This module is interpretune's half: an EXECUTOR per op name, the function
a backend's hook calls with the region of the activation a spec selects, the spec's operand and the spec itself. The
core ops are registered from the shared op math in :mod:`~interpretune.analysis.backends.interventions` when that
module is imported; a user registers another with :func:`register_executor` beside venngram's ``register_op``, and a hub
op collection's tensor op registers both when it loads (:func:`load_tensor_op`).

An executor must be shape-preserving (it returns a tensor of the region's shape), out of place (it writes neither the
region nor the operand) and differentiable in both, because a tensor op runs inside a forward pass that a training step
may differentiate through. :func:`check_tensor_op` holds an implementation to that contract on a probe before it is
registered from an op collection. The core executors are the shared op math the conformance cases test, and two of
them do not read the activation at all by design (``replace`` substitutes the operand, ``replace_with_error`` passes the
activation's value through with the gradient routed to the operand), which is why the probe is a loading gate rather
than a property every op has.

A backend whose hook computes its edit during the pass declares every op registered here
(``InterventionSupport.every()``), including an op registered after the backend was built, because nothing about a
new op's execution is the backend's: it hands the executor the region and the operand. A backend whose steering surface
takes only parameters fixed before the pass declares the ops it can express instead, and every other op is refused by
name before the backend is reached.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from interpretune.analysis.backends.interventions import InterventionSpec

#: An op's execution: ``(region, operand, spec) -> edited region``. ``region`` is what the spec's scope selected,
#: flattened so its leading axis indexes independent rows; ``operand`` is the spec's tensor on the region's device and
#: dtype; ``spec`` carries the op's scalar parameters (``scale_factor``, ``clamp_min``, ``clamp_max``, ...).
OpExecutor = Callable[[torch.Tensor, torch.Tensor, "InterventionSpec"], torch.Tensor]


@dataclass(frozen=True)
class RegisteredExecutor:
    """An executor and where it came from: ``core`` (interpretune's own op math), ``local`` (user code), or
    ``hub:<user.repo>`` with the collection, its version and the revision it was loaded from."""

    name: str
    executor: OpExecutor
    source: str = "local"
    collection: str | None = None
    version: str | None = None
    revision: str | None = None

    def describe(self) -> str:
        """One line for a report: the op and its provenance."""
        detail = [self.source]
        if self.collection:
            detail.append(f"collection {self.collection} {self.version or '(unversioned)'}")
        if self.revision:
            detail.append(f"revision {self.revision[:12]}")
        return f"{self.name} [{', '.join(detail)}]"


_EXECUTORS: dict[str, RegisteredExecutor] = {}


def _venngram_ops() -> Mapping[str, Any]:
    """Venngram's op registry, imported on first use so importing interpretune does not import venngram."""
    from venngram.ops import OPS

    return OPS


def register_executor(
    name: str,
    executor: OpExecutor,
    *,
    source: str = "local",
    collection: str | None = None,
    version: str | None = None,
    revision: str | None = None,
    replace: bool = False,
) -> None:
    """Register the executor for op ``name``.

    The op itself is declared to venngram (``venngram.ops.register_op``); an executor for a name venngram does not
    know is held here and stays unusable, since every use validates the name against venngram's registry
    (:func:`normalize_op`).

    Raises:
        ValueError: If ``name`` already has an executor and ``replace`` is not set, naming where the existing one came
            from.
    """
    existing = _EXECUTORS.get(name)
    if existing is not None and not replace:
        raise ValueError(
            f"op {name!r} already has an executor ({existing.describe()}); pass replace=True to replace it"
        )
    _EXECUTORS[name] = RegisteredExecutor(name, executor, source, collection, version, revision)


def unregister_executor(name: str) -> RegisteredExecutor:
    """Remove and return op ``name``'s executor (a test, or a collection being replaced).

    Raises:
        KeyError: If ``name`` has no executor.
    """
    return _EXECUTORS.pop(name)


def executor_record(name: str) -> RegisteredExecutor:
    """The registration of op ``name``'s executor: the function and its provenance.

    Raises:
        ValueError: If ``name`` has no executor, naming the ops that do.
    """
    record = _EXECUTORS.get(name)
    if record is None:
        raise ValueError(
            f"op {name!r} has no executor in interpretune (executors: {sorted(_EXECUTORS)}); register one with "
            "interpretune.analysis.backends.ops.register_executor, or load the hub collection that carries it"
        )
    return record


def executor_for(name: str) -> OpExecutor:
    """The executor for op ``name`` (see :func:`executor_record`)."""
    return executor_record(name).executor


def normalize_op(op: Any) -> str:
    """An op name, validated against venngram's registry, as the plain string every consumer compares.

    Accepts the name or anything carrying it as ``value``. An executor is not required here: a backend's declaration
    is what decides whether a named op runs on it, and the executor is looked up when the op is applied.

    Raises:
        ValueError: If venngram does not know the name, listing the ops it does.
    """
    name = str(getattr(op, "value", op))
    ops = _venngram_ops()
    if name not in ops:
        raise ValueError(
            f"Unknown op {name!r}; venngram's registry holds {sorted(ops)!r}. An op is registered with "
            "venngram.ops.register_op and executed through an executor registered here (a hub op collection's tensor "
            "op registers both when it loads)."
        )
    return name


def registered_ops() -> frozenset[str]:
    """Every op that is both registered with venngram and executable here: what a backend computing its edits during
    the pass declares."""
    ops = _venngram_ops()
    return frozenset(name for name in _EXECUTORS if name in ops)


# --------------------------------------------------------------------------------------------------------------------
# Tensor ops as data: an op collection's `op_spec` block
# --------------------------------------------------------------------------------------------------------------------

#: The keys an ``op_spec`` block may carry: venngram's ``OpSpec`` fields, by name.
OP_SPEC_KEYS = ("operand", "params", "requires", "targets", "value", "token_override", "composition", "at_least_one_of")
_PARAM_KEYS = ("type", "required", "range")
_PARAM_TYPES = {"float": builtins.float, "int": builtins.int, "str": builtins.str, "bool": builtins.bool}


def op_spec_from_mapping(name: str, block: Any, *, source: str = "<inline>") -> Any:
    """A ``venngram.ops.OpSpec`` built from an op definition's ``op_spec`` block, refusing a malformed one by name.

    The installed venngram reads a ``support_record`` from data but not an ``OpSpec``, so the block is read here, key
    for key against ``OpSpec``'s fields; once venngram reads one itself, this defers to it. ``params`` maps each
    parameter to ``{type: float|int|str|bool, required: bool, range: [lo, hi]}``.

    Raises:
        ValueError: If the block is not a mapping, names a key ``OpSpec`` does not have, or a parameter field or type
            this reader does not know. venngram's own ``register_op`` refuses an unknown operand kind, target or
            composition.
    """
    from venngram.ops import OpSpec, ParamSpec

    where = f"{source}: op {name!r}"
    if not isinstance(block, Mapping):
        raise ValueError(f"{where}: `op_spec` is a mapping of OpSpec fields, got {type(block).__name__}")
    if unknown := sorted(set(block) - set(OP_SPEC_KEYS)):
        raise ValueError(
            f"{where}: `op_spec` names {unknown}, which OpSpec does not have (fields: {list(OP_SPEC_KEYS)})"
        )
    fields: dict[str, Any] = {}
    for key in ("operand", "requires", "targets", "at_least_one_of"):
        if key in block:
            raw = block[key]
            if not isinstance(raw, (list, tuple)) or not all(isinstance(v, str) for v in raw):
                raise ValueError(f"{where}: `op_spec.{key}` is a list of names, got {raw!r}")
            fields[key] = tuple(raw)
    for key, kind in (("value", str), ("composition", str), ("token_override", bool)):
        if key in block:
            if not isinstance(block[key], kind):
                raise ValueError(f"{where}: `op_spec.{key}` is a {kind.__name__}, got {block[key]!r}")
            fields[key] = block[key]
    params: dict[str, Any] = {}
    for param, spec in (block.get("params") or {}).items():
        spec = spec or {}
        if not isinstance(spec, Mapping) or (unknown := sorted(set(spec) - set(_PARAM_KEYS))):
            raise ValueError(f"{where}: parameter {param!r} is a mapping of {list(_PARAM_KEYS)}, got {spec!r}")
        type_name = spec.get("type", "float")
        if type_name not in _PARAM_TYPES:
            raise ValueError(f"{where}: parameter {param!r} has type {type_name!r}; known: {sorted(_PARAM_TYPES)}")
        bounds = spec.get("range")
        if bounds is not None and (not isinstance(bounds, (list, tuple)) or len(bounds) != 2):
            raise ValueError(f"{where}: parameter {param!r} range is [lo, hi], got {bounds!r}")
        params[str(param)] = ParamSpec(
            _PARAM_TYPES[type_name], bool(spec.get("required", False)), None if bounds is None else tuple(bounds)
        )
    if params:
        fields["params"] = params
    return OpSpec(name, **fields)


#: The probe's activation width and row count: small enough to run at load, wide enough that a reduction over the
#: last axis is not the identity.
_PROBE_WIDTH, _PROBE_ROWS = 8, 3


def _probe_operand(operand_kinds: tuple[str, ...], generator: torch.Generator) -> torch.Tensor:
    """An operand of the first kind the op takes that carries a tensor: a vector, or a stack of two for a pair or a
    subspace. An op taking no operand still receives the spec's tensor, a vector of the activation's width."""
    stacked = next((k for k in operand_kinds if k in ("pair", "subspace")), None)
    shape = (2, _PROBE_WIDTH) if stacked and "vector" not in operand_kinds else (_PROBE_WIDTH,)
    return torch.randn(shape, generator=generator)


def check_tensor_op(name: str, executor: OpExecutor, *, operand_kinds: tuple[str, ...] = ("vector",)) -> None:
    """Hold an executor to the tensor-op contract on a probe, refusing it by name when it breaks one.

    The contract is what makes a tensor op safe inside a training forward pass: the result has the region's shape
    (shape-preserving), neither the region nor the operand is written (out of place, checked by value and by autograd's
    version counter, which an in-place write bumps even when it restores the values), and the result carries a gradient
    to both the region and the operand (differentiable). The probe is seeded, so a refusal reproduces.

    Raises:
        ValueError: Naming the op and the clause of the contract it broke, or the exception it raised on the probe.
    """
    from interpretune.analysis.backends.capabilities import PositionScope
    from interpretune.analysis.backends.interventions import InterventionSpec

    generator = torch.Generator().manual_seed(952)
    region_leaf = torch.randn(_PROBE_ROWS, _PROBE_WIDTH, generator=generator).requires_grad_()
    operand_leaf = _probe_operand(operand_kinds, generator).requires_grad_()
    # non-leaf views, so an in-place write is permitted (and detected) rather than refused by autograd first
    region, operand = region_leaf * 1.0, operand_leaf * 1.0
    region_before, operand_before = region.detach().clone(), operand.detach().clone()
    versions = (region._version, operand._version)
    spec = InterventionSpec(operand, op=name, position_scope=PositionScope.ALL_POSITIONS)
    where = f"tensor op {name!r}"
    try:
        result = executor(region, operand, spec)
    except Exception as exc:
        raise ValueError(f"{where} raised {type(exc).__name__} on a probe region of shape {tuple(region.shape)}: {exc}")
    if not isinstance(result, torch.Tensor):
        raise ValueError(f"{where} returned {type(result).__name__}, not a tensor")
    if tuple(result.shape) != tuple(region.shape):
        raise ValueError(
            f"{where} is not shape-preserving: a region of shape {tuple(region.shape)} came back as "
            f"{tuple(result.shape)}"
        )
    written = [
        what
        for what, tensor, before, version in (
            ("region", region, region_before, versions[0]),
            ("operand", operand, operand_before, versions[1]),
        )
        if tensor._version != version or not torch.equal(tensor.detach(), before)
    ]
    if written:
        raise ValueError(
            f"{where} is not out of place: it wrote its {' and '.join(written)}; a tensor op returns a new tensor"
        )
    if not result.requires_grad:
        raise ValueError(f"{where} is not differentiable: its result carries no gradient")
    grads = torch.autograd.grad(result.sum(), (region_leaf, operand_leaf), allow_unused=True)
    if missing := [what for what, grad in zip(("activation", "operand"), grads) if grad is None]:
        raise ValueError(f"{where} is not differentiable in the {' and the '.join(missing)}")


def load_tensor_op(
    name: str,
    block: Any,
    executor: OpExecutor,
    *,
    source: str,
    collection: str | None = None,
    version: str | None = None,
    revision: str | None = None,
) -> RegisteredExecutor:
    """Register a tensor op an op collection defines: its ``OpSpec`` with venngram and its executor here.

    The implementation is checked on a probe first (:func:`check_tensor_op`), so nothing is registered for an op that
    breaks the contract. A name another source already registered (a core op, another collection's) is refused rather
    than replaced; a re-load of the same collection (a re-pull at another revision) replaces its own registration.

    Raises:
        ValueError: For a malformed ``op_spec``, an implementation breaking the contract, or a name another source
            holds.
    """
    from venngram.ops import register_op

    where = source if revision is None else f"{source}@{revision[:12]}"
    spec = op_spec_from_mapping(name, block, source=where)
    existing = _EXECUTORS.get(name)
    if existing is not None and existing.source != source:
        raise ValueError(f"{where}: tensor op {name!r} is already registered ({existing.describe()}); rename it")
    if existing is None and name in _venngram_ops():
        raise ValueError(
            f"{where}: tensor op {name!r} is already registered with venngram by another source; rename it"
        )
    check_tensor_op(name, executor, operand_kinds=tuple(spec.operand))
    register_op(spec, replace=existing is not None)
    register_executor(
        name,
        executor,
        source=source,
        collection=collection,
        version=version,
        revision=revision,
        replace=existing is not None,
    )
    return _EXECUTORS[name]


__all__ = [
    "OP_SPEC_KEYS",
    "OpExecutor",
    "RegisteredExecutor",
    "check_tensor_op",
    "executor_for",
    "executor_record",
    "load_tensor_op",
    "normalize_op",
    "op_spec_from_mapping",
    "register_executor",
    "registered_ops",
    "unregister_executor",
]
