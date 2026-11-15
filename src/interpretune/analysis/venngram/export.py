r"""
Exporting Interventions as a venngram Book
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The inverse of :func:`~interpretune.analysis.venngram.apply.recipe_interventions`: an intervention payload written as a
venngram address book carrying it as one treatment, so a steering configuration leaves interpretune as a document that
venngram's conformance checker and other consumers read without interpretune installed.

Each hook is a set of the book (its point and layer) and each spec one op, in payload order. Consecutive ops at one
site under one token scope share a stage, so the exported treatment applies in the order the payload does:

- ``add`` / ``replace``: a scalar is a ``dose``; a vector is an operand, with ``scale_factor`` as ``add``'s ``dose``;
- ``scale``: a scalar is a ``dose``; a vector is a unit set with a ``doses`` table, the units it does not leave alone,
  since venngram's ``scale`` takes no operand;
- ``project``, ``reject``, ``patch``, ``clamp``: an operand, with ``scale_factor`` as ``reject``'s ``fraction`` and
  the clamp bounds as ``min`` and ``max``.

A vector ``add`` is an operand unless its site is one of ``unit_sites``, whose coordinates are units: there it is a
unit set with a ``doses`` table, the form a recipe's unit doses take. A mode that ignores ``scale_factor`` or
``use_intervention_tensor_as_basis`` has neither written.

**Refused by name**, since a treatment that means something else is worse than none:

- a hook naming no layer, a wildcard pattern among them, since a stage names one concrete layer;
- a position scope with no venngram spelling, and a tensor of a rank the mode's op has no operand for;
- what no op parameter carries: a ``scale_factor`` other than 1 on ``project`` or ``patch``, a ``reject`` fraction
  outside the closed unit interval, ``use_intervention_tensor_as_basis=False``, a ``clamp`` with no bound;
- an operand with no ``basis`` given, since two bases are not interchangeable and venngram assumes none;
- an operand too large to write inline with no ``sidecar_dir`` to write it to;
- anything :func:`~interpretune.analysis.venngram.apply.op_refusal` would refuse on the way back in, so every exported
  treatment applies again through :func:`~interpretune.analysis.venngram.apply.recipe_interventions`.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import torch

from interpretune.analysis.backends.capabilities import PositionScope
from interpretune.analysis.backends.interventions import (
    InterventionSpec,
    _coerce_shared_intervention_specs,
    normalize_op,
    normalize_position_scope,
)
from interpretune.analysis.venngram.apply import op_refusal, split_site
from interpretune.analysis.venngram.fetch import _require
from interpretune.utils import MisconfigurationException

#: Interpretune position scopes by their venngram token-scope names, the inverse of the apply path's.
_TOKENS = {PositionScope.ALL_POSITIONS: "all", PositionScope.LAST_TOKEN: "last"}

#: The operand ranks each operand-bearing op has a kind for: a vector, or a stack of them on a leading axis. An op
#: outside this table (``replace_with_error``, an op a collection registers) has no writer here and is refused by name.
_OPERAND_RANKS = {
    "add": (1,),
    "replace": (1,),
    "project": (1,),
    "reject": (1, 2),
    "patch": (2,),
    "clamp": (1, 2),
}

_NOT_SET_NAME = re.compile(r"[^A-Za-z0-9_-]")


class _BookWriter:
    """Accumulates a book's sets and a treatment's operands and stages, one spec at a time."""

    def __init__(self, basis: str | None, unit_sites: frozenset[str]) -> None:
        self.basis = basis
        self.unit_sites = unit_sites
        self.sets: dict[str, dict[str, Any]] = {}
        self.operands: dict[str, torch.Tensor] = {}
        self.stages: list[dict[str, Any]] = []

    def _operand(self, tensor: torch.Tensor, op: str, where: str) -> str:
        if tensor.ndim not in _OPERAND_RANKS[op]:
            raise MisconfigurationException(
                f"{where}: venngram's `{op}` takes an operand of rank {_OPERAND_RANKS[op]}, not a tensor of"
                f" shape {tuple(tensor.shape)}"
            )
        if op == "patch" and tensor.shape[0] != 2:
            raise MisconfigurationException(f"{where}: `patch` needs a pair, not a shape {tuple(tensor.shape)} stack")
        if self.basis is None:
            raise MisconfigurationException(
                f"{where}: an operand must declare the basis it is written in, and none is assumed; pass `basis`"
            )
        name = f"operand_{len(self.operands)}"
        self.operands[name] = tensor
        return name

    def _unit_op(self, op: str, site: str, layer: int, multiplier: torch.Tensor, fill: float) -> dict[str, Any]:
        """A vector as a unit set and its doses: the coordinates that differ from what the op leaves alone."""
        units = [i for i, v in enumerate(multiplier.tolist()) if v != fill]
        set_name = _NOT_SET_NAME.sub("_", f"blocks_{layer}_{site}_units_{len(self.sets)}")
        self.sets[set_name] = {"point": site, "layers": [layer], "units": {layer: units}}
        doses = {layer: {u: multiplier[u].item() for u in units}}
        return {"op": op, "target": {"set": set_name}, "value": {"doses": doses}}

    def _broadcast_op(self, op: str, tensor: torch.Tensor, scale: float) -> dict[str, Any]:
        """A tensor every element of which is the one value: venngram's ``dose``."""
        return {"op": op, "value": {"dose": tensor.flatten()[0].item() * scale}}

    def _scale_or_add(self, spec: InterventionSpec, tensor: torch.Tensor, site: str, layer: int, where: str):
        """A scale or add op; returns it and the stage site it needs (a unit set's stage names no layer)."""
        name = normalize_op(spec.op)
        layered = f"blocks.{layer}.{site}"
        per_unit = name == "scale" or site in self.unit_sites
        if tensor.ndim > 1:
            raise MisconfigurationException(
                f"{where}: venngram's `{name}` takes a dose or a vector, not a tensor of shape {tuple(tensor.shape)}"
            )
        constant = tensor.ndim == 0 or (per_unit and bool((tensor == tensor.flatten()[0]).all()))
        if constant:
            return self._broadcast_op(name, tensor, spec.scale_factor), layered
        if per_unit:
            fill = 1.0 if name == "scale" else 0.0
            return self._unit_op(name, site, layer, tensor * spec.scale_factor, fill), site
        op: dict[str, Any] = {"op": name, "operand": self._operand(tensor, name, where)}
        if spec.scale_factor != 1.0:
            op["value"] = {"dose": spec.scale_factor}
        return op, layered

    def _replace_op(self, tensor: torch.Tensor, where: str) -> dict[str, Any]:
        if tensor.ndim == 0:
            return self._broadcast_op("replace", tensor, 1.0)  # `replace` ignores `scale_factor`
        return {"op": "replace", "operand": self._operand(tensor, "replace", where)}

    def _span_op(self, spec: InterventionSpec, tensor: torch.Tensor, where: str) -> dict[str, Any]:
        """A ``project``, ``reject``, ``patch`` or ``clamp`` op, all of which act through an operand."""
        name = normalize_op(spec.op)
        if name in ("project", "patch") and spec.scale_factor != 1.0:
            raise MisconfigurationException(f"{where}: venngram's `{name}` carries no `scale_factor`")
        if name == "project" and not spec.use_intervention_tensor_as_basis:
            raise MisconfigurationException(
                f"{where}: venngram's `project` projects onto its operand; projecting the operand onto the activation"
                " (`use_intervention_tensor_as_basis=False`) has no spelling"
            )
        params: dict[str, Any] = {}
        if name == "reject":
            if spec.scale_factor < 0.0:  # above 1 over-rejects, which venngram's `fraction` admits
                raise MisconfigurationException(
                    f"{where}: venngram's `reject` takes a non-negative `fraction`, not {spec.scale_factor}"
                )
            if spec.scale_factor != 1.0:
                params["fraction"] = spec.scale_factor
        if name == "clamp":
            params = {k: v for k, v in (("min", spec.clamp_min), ("max", spec.clamp_max)) if v is not None}
            if not params:
                raise MisconfigurationException(f"{where}: a `clamp` with neither bound is the identity")
        op: dict[str, Any] = {"op": name, "operand": self._operand(tensor, name, where)}
        if params:
            op["params"] = params
        return op

    def add_spec(self, layer: int, site: str, spec: InterventionSpec, where: str) -> None:
        """Write ``spec`` as one op, in the stage before it when it shares that stage's site and token scope."""
        name = normalize_op(spec.op)
        scope = normalize_position_scope(spec.position_scope)
        if scope not in _TOKENS:
            raise MisconfigurationException(f"{where}: position scope `{scope.value}` has no venngram token scope")
        tensor = torch.as_tensor(spec.intervention_tensor).detach().cpu().float()
        if tensor.numel() == 0:
            raise MisconfigurationException(f"{where}: an empty tensor has nothing to write")
        stage_site = f"blocks.{layer}.{site}"
        if name in ("add", "scale"):
            op, stage_site = self._scale_or_add(spec, tensor, site, layer, where)
        elif name == "replace":
            op = self._replace_op(tensor, where)
        elif name in _OPERAND_RANKS:
            op = self._span_op(spec, tensor, where)
        else:
            raise MisconfigurationException(
                f"{where}: op `{name}` has no writer here (written: {sorted({'scale', *_OPERAND_RANKS})})"
            )
        tokens = _TOKENS[scope]
        if self.stages and (self.stages[-1]["site"], self.stages[-1]["tokens"]) == (stage_site, tokens):
            self.stages[-1]["ops"].append(op)
        else:
            self.stages.append({"site": stage_site, "tokens": tokens, "ops": [op]})

    def operand_documents(self, name: str, sidecar_dir: str | Path | None) -> dict[str, dict[str, Any]]:
        """Each operand's document: inline up to venngram's limit, in a sidecar beside the book above it."""
        operands = _require("venngram.operands")
        docs: dict[str, dict[str, Any]] = {}
        sidecar: dict[str, tuple[list[float], list[int]]] = {}
        for key, tensor in self.operands.items():
            values, shape = tensor.flatten().tolist(), list(tensor.shape)
            if len(values) <= operands.INLINE_LIMIT:
                docs[key] = {"values": values, "shape": shape, "basis": self.basis}
            elif sidecar_dir is None:
                raise MisconfigurationException(
                    f"`{key}` has {len(values)} values, above venngram's inline limit of {operands.INLINE_LIMIT};"
                    " pass `sidecar_dir` to write it to a sidecar"
                )
            else:
                sidecar[key] = (values, shape)
        if sidecar:
            filename = f"{name}.operands.safetensors"
            sha256 = operands.write_operands(Path(str(sidecar_dir)) / filename, sidecar)
            for key, (_, shape) in sidecar.items():
                docs[key] = {"file": filename, "key": key, "sha256": sha256, "shape": shape, "basis": self.basis}
        return {key: docs[key] for key in self.operands}


def export_book(
    interventions: Mapping[str, Any],
    *,
    name: str,
    model_id: str,
    revision: str,
    basis: str | None = None,
    unit_sites: Collection[str] = (),
    sidecar_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write an intervention payload as a venngram address book carrying it as the treatment ``name``.

    Args:
        interventions: Concrete hook names (``blocks.<layer>.<site>``) mapped to what a backend's payload accepts at a
            hook: a spec, a tensor, a mapping, or a sequence of them applied in order.
        name: The book's name and its treatment's.
        model_id: The model the treatment was written for.
        revision: The model's pinned revision: a unit means something in one revision only.
        basis: The basis every operand is written in. Required when the payload writes an operand.
        unit_sites: The layer-free sites whose coordinates are units, where a vector ``add`` is a unit set with a dose
            table rather than an operand.
        sidecar_dir: Where to write the sidecar of any operand above venngram's inline limit. The book must then be
            read from this directory.

    Returns:
        The book's document (``yaml.safe_dump`` writes it), parsed by venngram before it is returned.

    Raises:
        MisconfigurationException: For anything the module docstring lists as refused, naming the hook and spec.
    """
    book_mod = _require("venngram.book")
    if not interventions:
        raise MisconfigurationException("There are no interventions to export.")
    writer = _BookWriter(basis, frozenset(unit_sites))
    for hook, raw in interventions.items():
        layer, site = split_site(hook)
        if layer is None:
            raise MisconfigurationException(
                f"`{hook}`: a stage names one concrete layer (`blocks.<layer>.<site>`), and this hook names none;"
                " expand a wildcard pattern before exporting"
            )
        writer.sets[_NOT_SET_NAME.sub("_", hook)] = {"point": site, "layers": [layer]}
        for i, spec in enumerate(_coerce_shared_intervention_specs(raw)):
            writer.add_spec(layer, site, spec, f"`{hook}` spec {i}")
    body: dict[str, Any] = {"model": {"id": model_id, "revision": revision}, "stages": writer.stages}
    if writer.operands:
        body["operands"] = writer.operand_documents(name, sidecar_dir)
    document = {
        "schema_version": book_mod.SCHEMA_VERSION,
        "name": name,
        "portability": "checkpoint",
        "applies_to": {"models": [{"id": model_id, "revision": revision}]},
        "sets": writer.sets,
        "treatments": {name: body},
    }
    book = book_mod.AddressBook.from_mapping(document, source=f"exported book `{name}`", base_dir=sidecar_dir)
    recipe = book.treatments[name]
    for i, stage in enumerate(recipe.stages):
        for j, op in enumerate(stage.ops):
            if (refusal := op_refusal(recipe, stage, op)) is not None:
                raise MisconfigurationException(f"stage {i} ({stage.site}) op {j} (`{op.op}`): {refusal}")
    return document
