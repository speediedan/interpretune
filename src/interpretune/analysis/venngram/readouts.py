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
r"""Executing a readout a venngram treatment declares.

A treatment whose stages act on gradients (``of: grad``) says which scalar they are gradients of with a ``readout``:
a weighted sum of terms, each a registered reduction of the activations a book set addresses, at a token scope. A
stage may state its own, replacing the treatment's. venngram declares and validates it; :func:`recipe_readout` turns
it into the :class:`~interpretune.analysis.backends.readouts.Readout` ``model_bwd_intervention`` (and a model backend's
``bwd_w_intervention``) differentiates, beside the payload ``recipe_interventions(..., flow="grad")`` translates.

**What a term reads.** Its set addresses sites and, optionally, units there:

- a block site (``point:hook_in & layers:5``) is read from the forward pass the backward runs from, as a callable
  readout's sites are; a set selecting several sites concatenates their addressed units in layer order;
- ``point:unembed.hook_out`` is the logits, whose units are the vocabulary, so ``unembed.hook_out & units:[1000]`` is
  one token's logit. Any other point outside the block stack is refused.

Units are resolved against each activation's width when the readout is evaluated, so slices and negative units count
from the real width. A term without ``units`` reads every unit of its sites.

**Token scopes.** ``last`` (the final column), ``all``, and explicit ``positions`` (distinct, non-negative columns of
the padded batch tensor, as for interventions). ``prompt``, ``generated`` and ``slice`` have no interpretune spelling
and are refused.

**Reductions.** :data:`READOUT_EXECUTORS` computes each declared reduction per row: ``sum`` over the addressed units and
tokens, ``project`` (each token's values dotted with the operand, a vector over the addressed units in their order,
summed over the tokens) and ``norm`` (each token's Euclidean norm, summed over the tokens). The readout is the weighted
sum of its terms, summed over the batch's rows: venngram's readout is a scalar per sequence, and a sum keeps each
row's gradient that of its own readout, as the default last-token readout does. :func:`evaluate_readout_case` runs an
executor on venngram's conformance vectors (``venngram.readouts.check_readouts``), and
:func:`register_readout_executor` adds one for a reduction registered in venngram, refusing it unless it passes them.

**One scalar per backward pass.** Every stage acting on gradients must take them of the same readout. Stages declaring
different ones are refused rather than applied under one of them, since a single backward pass differentiates a single
scalar. A treatment written before readouts existed (recipe schema 3 or earlier) may declare none; the consumer's
default then applies, and :func:`recipe_readout` returns ``None``.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from interpretune.analysis.backends.readouts import Readout
from interpretune.analysis.points import UnknownPointError, parse
from interpretune.analysis.venngram.sites import SITE_FIELDS, _SiteEvaluator, _UnitEvaluator, _venngram
from interpretune.utils import MisconfigurationException

#: A reduction's executor: ``(values, operand, params) -> per-row value``, with ``values`` shaped
#: ``(rows, tokens, units)`` and ``operand`` a ``(units,)`` vector for a reduction that takes one, else ``None``.
ReadoutExecutor = Callable[[torch.Tensor, "torch.Tensor | None", Mapping[str, Any]], torch.Tensor]

#: The point whose activation is the logits the backward pass starts from.
LOGITS_POINT = "unembed.hook_out"

#: The token scopes a readout term may use here.
_TOKEN_KINDS = frozenset({"all", "last", "positions"})


def _sum(values: torch.Tensor, operand: torch.Tensor | None, params: Mapping[str, Any]) -> torch.Tensor:
    return values.sum(dim=(-2, -1))


def _project(values: torch.Tensor, operand: torch.Tensor | None, params: Mapping[str, Any]) -> torch.Tensor:
    assert operand is not None  # venngram refuses a `project` term without an operand
    return (values @ operand.to(values)).sum(dim=-1)


def _norm(values: torch.Tensor, operand: torch.Tensor | None, params: Mapping[str, Any]) -> torch.Tensor:
    return torch.linalg.vector_norm(values, dim=-1).sum(dim=-1)


#: The reductions interpretune computes, by venngram name.
READOUT_EXECUTORS: dict[str, ReadoutExecutor] = {"sum": _sum, "project": _project, "norm": _norm}


def executed_readouts() -> frozenset[str]:
    """The reductions a readout may use here: venngram's core ones and any registered with an executor."""
    return frozenset(READOUT_EXECUTORS)


def evaluate_readout_case(case: Any) -> float:
    """One venngram readout conformance vector (``venngram.readouts.ReadoutCase``) evaluated with
    :data:`READOUT_EXECUTORS`: the adapter ``venngram.readouts.check_readouts`` takes.

    Raises:
        MisconfigurationException: If a term names a reduction with no executor.
    """
    total = 0.0
    for term in case.terms:
        if term.reduce not in READOUT_EXECUTORS:
            raise MisconfigurationException(
                f"readout reduction `{term.reduce}` has no executor here (executed: {sorted(READOUT_EXECUTORS)})"
            )
        values = torch.tensor(term.values, dtype=torch.float64).unsqueeze(0)
        operand = None if term.operand is None else torch.tensor(term.operand, dtype=torch.float64)
        total += term.weight * float(READOUT_EXECUTORS[term.reduce](values, operand, dict(term.params))[0])
    return total


def register_readout_executor(name: str, executor: ReadoutExecutor, *, replace: bool = False) -> None:
    """Execute the venngram reduction ``name`` with ``executor``, once it computes venngram's conformance vectors.

    Raises:
        MisconfigurationException: If venngram has not registered ``name``, ``name`` already has an executor and
            ``replace`` is not set, or the executor gets any conformance vector wrong (each named).
    """
    vg_readouts = _venngram("readouts")
    if name not in vg_readouts.READOUTS:
        raise MisconfigurationException(
            f"readout reduction `{name}` is not registered in venngram (registered: {sorted(vg_readouts.READOUTS)}); "
            "register it there first, with what it computes and its conformance vectors"
        )
    if name in READOUT_EXECUTORS and not replace:
        raise MisconfigurationException(f"readout reduction `{name}` already has an executor; pass replace=True")
    previous = READOUT_EXECUTORS.get(name)
    READOUT_EXECUTORS[name] = executor
    failures = vg_readouts.check_readouts(evaluate_readout_case)
    if failures:
        if previous is None:
            del READOUT_EXECUTORS[name]
        else:
            READOUT_EXECUTORS[name] = previous
        raise MisconfigurationException(
            f"the executor for readout reduction `{name}` fails venngram's conformance vectors: {failures}"
        )


@dataclass(frozen=True)
class _Term:
    """One declared term, with its sites resolved: ``(key, layer, point)``, the key ``None`` for the logits."""

    reduce: str
    weight: float
    operand: torch.Tensor | None
    params: Mapping[str, Any]
    tokens: Any
    expr: Any
    sites: tuple[tuple[str | None, int, str], ...]
    where: str


class _ReadSiteEvaluator(_SiteEvaluator):
    """Site algebra for a readout term, whose set may also name ``units``: units narrow what is read at a site, not
    which sites are read, so they are resolved against each activation's width when the readout is evaluated."""

    def _term(self, term: Any, universe: frozenset) -> frozenset:
        if term.units is None:
            return super()._term(term, universe)
        if unknown := set(term.set_fields()) - SITE_FIELDS - {"units"}:
            raise self._refuse(f"{sorted(unknown)} have no meaning in a readout term here")
        layers = None if term.layers is None else set(self._layers(term.layers))
        return frozenset(
            (layer, point)
            for layer, point in universe
            if (term.point is None or point == term.point) and (layers is None or layer in layers)
        )


def _is_logits(point: str) -> bool:
    try:
        return parse(point).base == LOGITS_POINT
    except UnknownPointError:
        return False


def _term_sites(
    expr: Any, sets: Mapping[str, Any], n_layers: int, where: str
) -> tuple[tuple[str | None, int, str], ...]:
    address = _venngram("address")
    points = _SiteEvaluator(address, sets, n_layers, where=where).points(expr, ())
    if not points:
        raise MisconfigurationException(f"{where}: its set names no `point`, so the readout reads nothing")
    logits = {point for point in points if _is_logits(point)}
    if logits and logits != points:
        raise MisconfigurationException(
            f"{where}: its set mixes the logits (`{LOGITS_POINT}`) with block sites {sorted(points - logits)}; a term "
            "reads one tensor, so write one term for each"
        )
    if logits:
        return ((None, 0, logits.pop()),)
    evaluator = _ReadSiteEvaluator(address, sets, n_layers, where=where)
    if outside := sorted(point for point in points if evaluator.is_global(point)):
        raise MisconfigurationException(
            f"{where}: {outside} are global points, and a readout reads the logits (`{LOGITS_POINT}`) or points inside "
            "the block stack"
        )
    universe = frozenset((layer, point) for point in points for layer in range(n_layers))
    selected = evaluator.evaluate(expr, universe, ())
    if not selected:
        raise MisconfigurationException(f"{where}: its set selects no site of a {n_layers}-layer stack")
    ordered = sorted(selected, key=lambda pair: (pair[0], sorted(points).index(pair[1])))
    return tuple((f"blocks.{layer}.{point}", layer, point) for layer, point in ordered)


def _operand(recipe: Any, name: str | None, bases: frozenset[str] | None, where: str) -> torch.Tensor | None:
    if name is None:
        return None
    operand = recipe.operands[name]
    if bases is not None and operand.basis not in bases:
        raise MisconfigurationException(
            f"{where}: operand `{name}` is written in basis `{operand.basis}`, which is not one of {sorted(bases)}"
        )
    values, shape = operand.load(recipe.base_dir or ".")
    tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
    if tensor.ndim != 1:
        raise MisconfigurationException(f"{where}: operand `{name}` must be a vector, got shape {tuple(shape)}")
    return tensor


def _check_tokens(tokens: Any, where: str) -> None:
    if tokens.kind not in _TOKEN_KINDS:
        raise MisconfigurationException(
            f"{where}: token scope `{tokens.kind}` has no interpretune spelling for a readout (supported: "
            f"{sorted(_TOKEN_KINDS)})"
        )
    if tokens.kind == "positions":
        positions = list(tokens.positions or ())
        if not positions or any(p < 0 for p in positions) or len(set(positions)) != len(positions):
            raise MisconfigurationException(
                f"{where}: explicit positions must be distinct, non-negative columns of the padded batch, got "
                f"{positions}"
            )


def _at_tokens(values: torch.Tensor, tokens: Any, where: str) -> torch.Tensor:
    if tokens.kind == "last":
        return values[:, -1:, :]
    if tokens.kind == "all":
        return values
    positions = list(tokens.positions)
    if max(positions) >= values.shape[1]:
        raise MisconfigurationException(
            f"{where}: position {max(positions)} is outside a sequence of {values.shape[1]} columns"
        )
    return values[:, positions, :]


def _term_readout(recipe: Any, term: Any, sets: Mapping[str, Any], n_layers: int, bases: Any, where: str) -> _Term:
    if term.reduce not in READOUT_EXECUTORS:
        raise MisconfigurationException(
            f"{where}: reduction `{term.reduce}` has no executor here (executed: {sorted(READOUT_EXECUTORS)}); "
            "register one with `register_readout_executor`"
        )
    _check_tokens(term.tokens, where)
    expr = _venngram("address").parse(f"@{term.set}")
    return _Term(
        reduce=term.reduce,
        weight=float(term.weight),
        operand=_operand(recipe, term.operand, bases, where),
        params=dict(term.params),
        tokens=term.tokens,
        expr=expr,
        sites=_term_sites(expr, sets, n_layers, where),
        where=where,
    )


def _readout_fn(terms: tuple[_Term, ...], sets: Mapping[str, Any], n_layers: int) -> Callable[..., Any]:
    address = _venngram("address")
    unit_cache: dict[tuple[int, str | None, int], list[int]] = {}

    def units(t: int, term: _Term, key: str | None, layer: int, point: str, width: int) -> list[int]:
        if (t, key, width) not in unit_cache:
            evaluator = _UnitEvaluator(address, sets, n_layers, where=term.where)
            evaluator.layer, evaluator.point, evaluator.width = layer, point, width
            unit_cache[(t, key, width)] = sorted(evaluator.evaluate(term.expr, frozenset(range(width)), ()))
        return unit_cache[(t, key, width)]

    def fn(logits: Any, activations: Mapping[str, Any]) -> Any:
        total = None
        for t, term in enumerate(terms):
            parts = []
            for key, layer, point in term.sites:
                value = logits if key is None else activations[key]
                if value.ndim != 3:
                    raise MisconfigurationException(
                        f"{term.where}: the activation at `{key or LOGITS_POINT}` has shape {tuple(value.shape)}; a "
                        "readout term reads a (batch, sequence, units) tensor"
                    )
                selected = units(t, term, key, layer, point, value.shape[-1])
                if selected:
                    parts.append(value[..., selected])
            if not parts:
                raise MisconfigurationException(f"{term.where}: its set selects no unit at any of its sites")
            values = _at_tokens(torch.cat(parts, dim=-1), term.tokens, term.where)
            if term.operand is not None and term.operand.shape[0] != values.shape[-1]:
                raise MisconfigurationException(
                    f"{term.where}: operand of {term.operand.shape[0]} values for {values.shape[-1]} addressed units"
                )
            value = term.weight * READOUT_EXECUTORS[term.reduce](values, term.operand, term.params).sum()
            total = value if total is None else total + value
        return total

    return fn


def _gradient_stages(recipe: Any) -> list[int]:
    return [
        i
        for i, stage in enumerate(recipe.stages)
        if any(op.target is not None and op.target_of(recipe.book) == "grad" for op in stage.ops)
    ]


def recipe_readout(recipe: Any, *, n_layers: int, bases: frozenset[str] | None = None) -> Readout | None:
    """The :class:`~interpretune.analysis.backends.readouts.Readout` a recipe's gradient stages take their
    gradients of.

    Pass it as ``model_bwd_intervention``'s ``readout`` (or a backend's ``bwd_w_intervention``) beside the payload
    ``recipe_interventions(recipe, ..., flow="grad")`` translates.

    Args:
        recipe: A ``venngram.recipe.Recipe`` with at least one stage acting on ``of: grad`` targets.
        n_layers: The model's block count, which layer slices and negative layers count from.
        bases: The bases a ``project`` term's operand may be written in, or ``None`` for any. The operand is applied
            to the addressed units in their order, so it must be written in those coordinates.

    Returns:
        The readout, or ``None`` when no gradient stage declares one (a recipe written before readouts existed), so
        the consumer's default readout applies.

    Raises:
        MisconfigurationException: If no stage acts on a gradient, the gradient stages declare different readouts, or a
            term names a reduction with no executor, an unsupported token scope, a set mixing the logits with block
            sites, a point outside the block stack other than the logits, or an operand outside ``bases``.
    """
    stages = _gradient_stages(recipe)
    if not stages:
        raise MisconfigurationException(
            f"recipe {recipe.name!r} has no stage acting on a gradient (`of: grad`), so no gradient is taken of a "
            "readout"
        )
    declared = {i: recipe.stages[i].readout or recipe.readout for i in stages}
    distinct = []
    for readout in declared.values():
        if readout not in distinct:
            distinct.append(readout)
    if len(distinct) > 1:
        raise MisconfigurationException(
            f"recipe {recipe.name!r}: its gradient stages take their gradients of different readouts (stages "
            f"{sorted(declared)}), and one backward pass differentiates one scalar; translate and apply the stages "
            "sharing a readout separately"
        )
    (readout,) = distinct
    if readout is None:
        return None
    sets = recipe.book.sets()
    origin = "readout" if all(recipe.stages[i].readout is None for i in stages) else f"stage {stages[0]} readout"
    terms = tuple(
        _term_readout(recipe, term, sets, n_layers, bases, f"recipe {recipe.name!r} {origin} term {k} (`{term.set}`)")
        for k, term in enumerate(readout.terms)
    )
    sites = tuple(dict.fromkeys(key for term in terms for key, _, _ in term.sites if key is not None))
    return Readout(_readout_fn(terms, sets, n_layers), sites=sites)


__all__ = [
    "LOGITS_POINT",
    "READOUT_EXECUTORS",
    "ReadoutExecutor",
    "evaluate_readout_case",
    "executed_readouts",
    "recipe_readout",
    "register_readout_executor",
]
