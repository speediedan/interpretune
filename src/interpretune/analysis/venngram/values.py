r"""
A Recipe Op's Members and Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

What a unit or feature target names, and the value each of its members takes. Venngram declares a value's source and
its transforms and leaves evaluating them to the consumer, so this is where Interpretune evaluates them. An
evidence-sourced value reads a column the caller supplies: ``evidence`` maps an evidence name to a table, a table maps
a column name to its values, and a column is keyed by ``(layer, unit)``. Interpretune does not compute evidence, and a
member the column has no row for is refused rather than dropped, since a treatment missing a member is a different
treatment.

The transforms run in the order the recipe writes them, each over the values the last one produced:

- ``scale``: $v \cdot f$ for the ``factor`` $f$;
- ``abs``: $\lvert v \rvert$;
- ``normalize``: by ``max_abs``, $v / \max_m \lvert v_m \rvert$ over the op's members (zero throughout when every value
  is zero); by any other name, $v / t$ for that statistic $t$;
- ``sign_of``: $\mathrm{sign}(c) \cdot \lvert v \rvert$ for the same member's value $c$ in the named ``column`` of the
  same table;
- ``sign_aware_scale``: $\mathrm{sign}(s) \cdot \lvert v \rvert \cdot f$ for the ``factor`` $f$ and the score $s$, the
  value the op read from its ``from`` column. This is circuit-tracer's meaning of the name
  (``CircuitTracerAnalysisBackend._signed_feature_intervention_value``);
- ``statistic``: $v \cdot t$ for the named statistic $t$, the inverse of ``normalize`` by that name. A dose written in
  units of a statistic (neuronary's, in units of a layer's median residual norm, which ``venngram.recipe.migrate``
  translates into this transform) becomes an activation-space dose this way, as venngram#58's draft defines it.

The sign of zero is zero, as circuit-tracer takes it. A statistic is computed from the model, so the caller supplies it
(``statistics`` maps a name to a value, or to a value per layer), and one it does not supply is refused by name.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from interpretune.utils import MisconfigurationException

#: The transforms :func:`member_values` evaluates without a statistic from the caller.
EVALUATED_TRANSFORMS = frozenset({"scale", "abs", "normalize", "sign_of", "sign_aware_scale"})

#: A ``features`` value naming the features themselves, rather than a latent model whose ``units`` name them.
_FEATURE_IDS = re.compile(r"^\d+(,\s*\d+)*$")

Member = tuple[int, int]


def target_kind(recipe: Any, op: Any) -> str:
    """``activation``, ``units`` or ``features``, the last for any set naming ``features``.

    Venngram's ``Op.target_kind`` reads a set with both ``features`` and ``units`` as units. Here it is a feature
    target whose ``units`` list the features, so its members are never scattered into the site's own activation.
    """
    if op.target is None:
        return "activation"
    term = recipe.book.sets()[op.target]
    return "features" if term.features is not None else op.target_kind(recipe.book)


def _explicit(indices: Any, name: str, what: str) -> tuple[int, ...]:
    if indices.indices is None:
        raise MisconfigurationException(f"set `{name}` names a slice of {what}; a recipe needs explicit {what}")
    return tuple(indices.indices)


def _explicit_layers(term: Any, name: str) -> tuple[int, ...]:
    if term.layers is None or term.layers.indices is None:
        raise MisconfigurationException(f"set `{name}` does not name its layers explicitly")
    return tuple(term.layers.indices)


def target_members(recipe: Any, op: Any) -> list[Member]:
    """The ``(layer, unit)`` pairs a unit or feature target names.

    A feature target names its features in ``units``, or directly in ``features`` (``features: "3,7"``), as venngram's
    recipes do. A ``features`` value that is a name and comes with no ``units`` names a whole latent space.

    Raises:
        MisconfigurationException: If the set does not name explicit layers and members.
    """
    name = op.target
    term = recipe.book.sets()[name]
    units = term.units
    if units is None:
        if not _FEATURE_IDS.match(term.features):
            raise MisconfigurationException(
                f"set `{name}` names the latent space `{term.features}` but no features: list them in `units`"
            )
        features = tuple(int(f) for f in term.features.split(","))
        return [(layer, f) for layer in _explicit_layers(term, name) for f in features]
    if units.per_layer is not None:
        return [(layer, u) for layer, unit_set in units.per_layer for u in _explicit(unit_set, name, "units")]
    flat = _explicit(units.flat, name, "units")
    return [(layer, u) for layer in _explicit_layers(term, name) for u in flat]


def member_refusal(recipe: Any, op: Any) -> str | None:
    """Why a unit or feature target's members cannot be listed, or ``None`` when they can."""
    try:
        target_members(recipe, op)
    except MisconfigurationException as e:
        return str(e)
    return None


def _sign(value: float) -> float:
    return float((value > 0) - (value < 0))


@dataclass(frozen=True)
class _Evaluation:
    """What a transform reads beyond the values it is given."""

    table: Mapping[str, Mapping[Member, float]]
    scores: Mapping[Member, float]
    statistics: Mapping[str, float | Mapping[int, float]]
    where: str

    def column(self, name: str, members: Any) -> dict[Member, float]:
        """The named column of the op's evidence table, for ``members``."""
        if name not in self.table:
            raise MisconfigurationException(
                f"{self.where}: the evidence table has no column `{name}` (columns: {sorted(self.table)})"
            )
        column = self.table[name]
        if missing := sorted(set(members) - set(column)):
            raise MisconfigurationException(
                f"{self.where}: column `{name}` has no value for (layer, unit) {missing}, which the op's target names"
            )
        return {m: float(column[m]) for m in members}

    def statistic(self, name: str, layer: int) -> float:
        """The named statistic at ``layer``."""
        if name not in self.statistics:
            raise MisconfigurationException(
                f"{self.where}: no statistic `{name}` was supplied (supplied: {sorted(self.statistics)})"
            )
        statistic = self.statistics[name]
        if isinstance(statistic, Mapping) and layer not in statistic:
            raise MisconfigurationException(f"{self.where}: statistic `{name}` has no value at layer {layer}")
        return float(statistic[layer] if isinstance(statistic, Mapping) else statistic)

    def divide(self, values: Mapping[Member, float], name: str) -> dict[Member, float]:
        """``values``, each over the named statistic at its layer."""
        out = {}
        for (layer, unit), value in values.items():
            if (divisor := self.statistic(name, layer)) == 0.0:
                raise MisconfigurationException(f"{self.where}: statistic `{name}` is zero at layer {layer}")
            out[(layer, unit)] = value / divisor
        return out

    def multiply(self, values: Mapping[Member, float], name: str) -> dict[Member, float]:
        """``values``, each times the named statistic at its layer."""
        return {(layer, unit): value * self.statistic(name, layer) for (layer, unit), value in values.items()}


def _normalize(values: Mapping[Member, float], params: Mapping[str, Any], ctx: _Evaluation) -> dict[Member, float]:
    if params["by"] != "max_abs":
        return ctx.divide(values, params["by"])
    peak = max((abs(v) for v in values.values()), default=0.0)
    return {m: 0.0 if peak == 0.0 else v / peak for m, v in values.items()}


def _sign_of(values: Mapping[Member, float], params: Mapping[str, Any], ctx: _Evaluation) -> dict[Member, float]:
    signs = ctx.column(params["column"], values)
    return {m: _sign(signs[m]) * abs(v) for m, v in values.items()}


_TRANSFORMS: dict[str, Callable[[Mapping[Member, float], Mapping[str, Any], _Evaluation], dict[Member, float]]] = {
    "scale": lambda values, params, _: {m: v * params["factor"] for m, v in values.items()},
    "abs": lambda values, _, __: {m: abs(v) for m, v in values.items()},
    "normalize": _normalize,
    "sign_of": _sign_of,
    "sign_aware_scale": lambda values, params, ctx: {
        m: _sign(ctx.scores[m]) * abs(v) * params["factor"] for m, v in values.items()
    },
    "statistic": lambda values, params, ctx: ctx.multiply(values, params["name"]),
}


def _from_evidence(
    value: Any,
    members: list[Member],
    evidence: Mapping[str, Mapping[str, Mapping[Member, float]]],
    statistics: Mapping[str, float | Mapping[int, float]],
    where: str,
) -> dict[Member, float]:
    if value.evidence not in evidence:
        raise MisconfigurationException(
            f"{where}: the value reads evidence `{value.evidence}`, which the caller did not supply"
            f" (supplied: {sorted(evidence)})"
        )
    ctx = _Evaluation(evidence[value.evidence], {}, statistics, where)
    values = ctx.column(value.column, members)
    ctx = _Evaluation(ctx.table, values, statistics, where)
    for transform in value.transform:
        values = _TRANSFORMS[transform.name](values, transform.params, ctx)
    return values


def member_values(
    recipe: Any,
    op: Any,
    *,
    evidence: Mapping[str, Mapping[str, Mapping[Member, float]]] | None = None,
    statistics: Mapping[str, float | Mapping[int, float]] | None = None,
    where: str = "<recipe>",
) -> dict[Member, float]:
    """``(layer, unit) -> value`` for a unit or feature target (a ``doses`` table may leave members undosed).

    Raises:
        MisconfigurationException: If the target's members cannot be listed, or an evidence-sourced value reads
            evidence, a column, a member or a statistic the caller did not supply.
    """
    value = op.value
    members = target_members(recipe, op)
    if value.source == "dose":
        return dict.fromkeys(members, float(value.dose))
    if value.source == "doses":
        doses = value.doses or {}
        return {(ly, u): doses[ly][u] for ly, u in members if u in doses.get(ly, {})}
    return _from_evidence(value, members, evidence or {}, statistics or {}, where)
