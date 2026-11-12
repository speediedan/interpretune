r"""
Applying a Recipe Through Native Interventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A supported recipe is translated into interpretune's own intervention payload, the raw mapping every backend
canonicalizes with :func:`~interpretune.analysis.backends.interventions.build_intervention_dict` once concrete
hook shapes are known. Each op becomes one
:class:`~interpretune.analysis.backends.interventions.InterventionSpec` per layer it acts at, keyed by the hook
pattern ``blocks.<layer>.<site>`` and appended in recipe order, so the backend's left-to-right fold applies the ops
as venngram orders them. The stage's token scope is the spec's ``position_scope`` (``all`` -> ``all_positions``,
``last`` -> ``last_token``); an op may not override its stage's scope (venngram refuses that at parse).

**Unit-targeted ops** take their layers from the book set they target, so their stage names a layer-free site:

- ``add``: the op's values per unit scattered into a dense vector at the site's width, applied as ``add``;
- ``scale``: the same table as a multiplier, ones where no unit is named, applied as ``scale``.

The values are a ``dose``, a ``doses`` table, or a column of evidence the caller supplies through a transform pipeline
(:mod:`~interpretune.analysis.venngram.values`).

**Feature-targeted ops** take the same path one level down, into a latent model's feature activations: ``add`` and
``scale`` scatter at the latent sub-hook, ``blocks.<layer>.<site>.hook_sae_acts_post``, which the backend resolves to
the attached latent model. The stage names the hooked site (``hook_resid_post``), and the width supplied is the
latent's, keyed ``<site>.hook_sae_acts_post`` because the site's own width differs. A ``replace`` on features has no
native mode here, since a spec replaces the whole tensor; circuit-tracer's feature tuples express it.

**Whole-activation ops** (no ``target``) name no layer of their own, so their stage names a layered site,
``blocks.<layer>.<site>``. A layer-free one is refused by name rather than applied at a guessed layer or at every
layer. The operand (a vector, a pair, or a subspace stacked on its leading axis) is read from the recipe inline or
from its sidecar, relative to the recipe's own directory:

- ``add`` / ``replace``: the operand vector, times the op's ``dose`` when it has one (``add`` through
  ``scale_factor``; ``replace``, which ignores ``scale_factor``, by scaling the tensor). Without an operand, the
  ``dose`` itself, broadcast over the activation;
- ``scale``: the ``dose``, as a scalar multiplier;
- ``project``: onto a vector operand, keeping all of it (``fraction`` 1). A subspace operand, or a fraction below
  1 (which keeps the span and only part of its complement), has no native mode yet and is refused by name;
- ``reject``: the operand's span, removing ``fraction`` of it (``scale_factor`` is the native removal fraction);
- ``patch``: the operand pair, exchanging its two coordinates;
- ``clamp``: the operand's pseudoinverse coordinates bounded into ``[min, max]``.

The caller supplies each unit-targeted site's width (from the model's own configuration: the MLP hidden width for
``mlp.out.hook_in`` / ``hook_mlp_hidden_in``, the residual width for ``mlp.hook_out``), because the width is a
fact of the checkpoint the recipe pins, not of the recipe, and guessing it from the largest unit named would
silently truncate a dose table that names a unit past the guess. A supplied width also checks an operand's.

:func:`recipe_feature_interventions` is the analysis-backend path: a feature ``replace``, ``add`` or ``scale`` at
explicit positions becomes circuit-tracer's ``(layer, position, feature_idx, value)`` tuples. A tuple sets a value, so
``add`` and ``scale`` are computed from the feature's clean activation, which the backend captures first
(``capture_feature_activations``, declared by ``captures_clean_activations``). The values are computed here and passed
as constants, so a backend must honour the ``constant`` value source.

Anything the declaration does not cover is refused by
:func:`~interpretune.analysis.venngram.support.require_supported`, which lists every refusal this module would
raise (:func:`op_refusal`) at once; this module refuses again what it cannot translate, by name, so a caller that
skipped the check still cannot get a silently narrowed treatment.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import torch

from interpretune.analysis.backends.capabilities import FeatureInterventionSupport, InterventionMode, PositionScope
from interpretune.analysis.backends.interventions import InterventionSpec
from interpretune.analysis.venngram.values import Statistic, member_refusal, member_values, target_kind
from interpretune.utils import MisconfigurationException

#: Recipe token scopes by their interpretune position-scope names.
_POSITION_SCOPES = {"all": PositionScope.ALL_POSITIONS, "last": PositionScope.LAST_TOKEN}

#: The unit-targeted ops, with the native mode each applies as and the value units no dose names keep.
_UNIT_OPS = {"add": (InterventionMode.ADD, 0.0), "scale": (InterventionMode.SCALE, 1.0)}

#: The whole-activation ops and the native mode each applies as.
_ACTIVATION_OPS = {
    "add": InterventionMode.ADD,
    "replace": InterventionMode.REPLACE,
    "scale": InterventionMode.SCALE,
    "project": InterventionMode.PROJECT,
    "reject": InterventionMode.REJECT,
    "patch": InterventionMode.PATCH,
    "clamp": InterventionMode.CLAMP,
}

#: Every recipe op this module translates (for at least one target kind).
APPLIED_OPS = frozenset(_UNIT_OPS) | frozenset(_ACTIVATION_OPS)

#: The feature ops a feature tuple's value is computed for: ``replace`` sets the op's value; ``add`` and ``scale`` set a
#: value relative to the feature's clean activation $a$ ($a + d$ and $a \cdot d$ for the op's value $d$).
_RELATIVE_FEATURE_OPS = {"add": lambda a, d: a + d, "scale": lambda a, d: a * d}
FEATURE_TUPLE_OPS = frozenset({"replace"}) | frozenset(_RELATIVE_FEATURE_OPS)

_LAYERED_SITE = re.compile(r"^blocks\.(\d+)\.(.+)$")

#: The sub-hook a latent model's feature activations are written at.
_LATENT_SUBHOOK = "hook_sae_acts_post"


def split_site(site: str) -> tuple[int | None, str]:
    """A stage site's layer (``None`` when it names none) and its layer-free spelling."""
    m = _LAYERED_SITE.match(site)
    return (int(m.group(1)), m.group(2)) if m else (None, site)


def _value_refusal(op: Any) -> str | None:
    """Why a whole-activation op's value cannot be applied: its transforms have no members to evaluate over."""
    value = op.value
    if value is not None and value.transform:
        return "a whole-activation op's dose takes no transform; write the dose as its final value"
    return None


def _member_refusal(recipe: Any, stage: Any, op: Any, kind: str) -> str | None:
    """Why a unit or feature target cannot take the unit path, or ``None`` when it can."""
    layer, site = split_site(stage.site)
    if op.op not in _UNIT_OPS:
        if kind == "features" and op.op == "replace":
            return (
                "a feature-targeted `replace` has no native mode at a sub-hook: a spec replaces the whole tensor"
                " (`recipe_feature_interventions` gives circuit-tracer's tuples)"
            )
        return f"a {kind[:-1]}-targeted `{op.op}` has no native mode (supported: {sorted(_UNIT_OPS)})"
    if op.operand is not None:
        return f"a {kind[:-1]}-targeted op takes a dose, not an operand"
    if layer is not None:
        return f"a {kind[:-1]} set names its own layers, so its stage names the layer-free site, not `{stage.site}`"
    if kind == "features" and site.endswith(_LATENT_SUBHOOK):
        return f"a feature target's stage names the hooked site; `{_LATENT_SUBHOOK}` is implied, not `{site}`"
    return member_refusal(recipe, op)


def op_refusal(recipe: Any, stage: Any, op: Any) -> str | None:
    """Why ``op`` (at ``stage`` of ``recipe``) cannot be translated, or ``None`` when it can."""
    layer, _ = split_site(stage.site)
    kind = target_kind(recipe, op)
    if kind in ("units", "features"):
        return _member_refusal(recipe, stage, op, kind)
    value = op.value
    if (refusal := _value_refusal(op)) is not None:
        return refusal
    if op.op not in _ACTIVATION_OPS:
        return f"the op has no native mode (supported: {sorted(_ACTIVATION_OPS)})"
    if layer is None:
        return f"a whole-activation op names no layer: write its stage's site as `blocks.<layer>.{stage.site}`"
    if value is not None and value.source != "dose":
        return "a whole-activation op takes a single `dose`, not a per-unit table"
    if op.op == "project":
        if recipe.operands[op.operand].kind != "vector":
            return "projecting onto a subspace has no native mode yet (a vector operand projects)"
        if op.params.get("fraction", 1.0) != 1.0:
            return "a `project` fraction below 1 has no native mode yet (fraction 1 projects)"
    return None


def feature_op_refusal(recipe: Any, stage: Any, op: Any) -> str | None:
    """Why ``op`` cannot become circuit-tracer feature tuples, or ``None`` when it can.

    Whether the backend can capture the clean activations a feature ``add`` or ``scale`` is computed from is the
    backend's declaration, not the recipe's, so :func:`recipe_feature_interventions` and the derived support record
    decide it.
    """
    if target_kind(recipe, op) != "features":
        return "circuit-tracer's feature tuples act on a feature target only"
    if op.op not in FEATURE_TUPLE_OPS:
        return f"circuit-tracer's feature tuples express {sorted(FEATURE_TUPLE_OPS)}, not `{op.op}`"
    if (op.tokens or stage.tokens).kind != "positions":
        return "circuit-tracer's feature tuples name their positions: write the stage's tokens as `{positions: [...]}`"
    if op.value is None:
        return "a feature-targeted op takes a value"
    return member_refusal(recipe, op)


def _scatter(doses: Mapping[int, float], width: int, fill: float, where: str, layer: int) -> torch.Tensor:
    tensor = torch.full((width,), fill)
    for unit, dose in doses.items():
        if not 0 <= unit < width:
            raise MisconfigurationException(f"{where}: unit {unit} at layer {layer} is outside the site width {width}")
        tensor[unit] = dose
    return tensor


def _operand(recipe: Any, name: str, width: int | None, where: str) -> torch.Tensor:
    values, shape = recipe.operands[name].load(recipe.base_dir or ".")
    tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
    if width is not None and tensor.shape[-1] != width:
        raise MisconfigurationException(
            f"{where}: operand `{name}` has width {tensor.shape[-1]} but the site width is {width}"
        )
    return tensor


def _member_specs(
    op: Any, values: Mapping[tuple[int, int], float], width: int, scope: PositionScope, where: str
) -> dict[int, InterventionSpec]:
    mode, fill = _UNIT_OPS[op.op]
    by_layer: dict[int, dict[int, float]] = {}
    for (layer, unit), value in values.items():
        by_layer.setdefault(layer, {})[unit] = value
    return {
        layer: InterventionSpec(
            intervention_tensor=_scatter(by_layer[layer], width, fill, where, layer), mode=mode, position_scope=scope
        )
        for layer in sorted(by_layer)
    }


def _activation_spec(recipe: Any, op: Any, width: int | None, scope: PositionScope, where: str) -> InterventionSpec:
    mode = _ACTIVATION_OPS[op.op]
    dose = op.value.dose if op.value is not None else 1.0
    tensor = torch.tensor(dose) if op.operand is None else _operand(recipe, op.operand, width, where)
    fields: dict[str, Any] = {}
    if op.op == "add" and op.operand is not None:
        fields["scale_factor"] = dose
    elif op.op == "replace" and op.operand is not None:
        tensor = tensor * dose
    elif op.op == "reject":
        fields["scale_factor"] = op.params.get("fraction", 1.0)
    elif op.op == "clamp":
        fields.update(clamp_min=op.params.get("min"), clamp_max=op.params.get("max"))
    return InterventionSpec(intervention_tensor=tensor, mode=mode, position_scope=scope, **fields)


def _require_width(site_widths: Mapping[str, int], site: str, i: int, stage: Any) -> int:
    if site not in site_widths:
        raise MisconfigurationException(
            f"stage {i} ({stage.site}): no width was supplied for this site, `{site}` (supplied: {sorted(site_widths)})"
        )
    return site_widths[site]


def recipe_interventions(
    recipe: Any,
    *,
    site_widths: Mapping[str, int],
    evidence: Mapping[str, Mapping[str, Mapping[tuple[int, int], float]]] | None = None,
    statistics: Mapping[str, Statistic] | None = None,
) -> dict[str, tuple[InterventionSpec, ...]]:
    """Translate a supported ``venngram.recipe.Recipe`` into a backend intervention payload.

    Args:
        recipe: The recipe to translate.
        site_widths: The width of every unit-targeted site the recipe names, keyed by the recipe's own layer-free
            site spelling, and of every feature-targeted one, keyed ``<site>.hook_sae_acts_post``. A width given for a
            whole-activation op's site also checks its operand's width.
        evidence: The tables evidence-sourced values read (:mod:`~interpretune.analysis.venngram.values`).
        statistics: The statistics their transforms name.

    Returns:
        A mapping of hook patterns (``blocks.<layer>.<site>``) to the ordered specs to apply at each, in the raw
        form :func:`~interpretune.analysis.backends.interventions.build_intervention_dict` canonicalizes.

    Raises:
        MisconfigurationException: For anything this module cannot translate (see :func:`op_refusal`), an
            unmapped token scope, a unit- or feature-targeted site whose width was not supplied, a unit outside its
            site's width, an operand whose width differs from a supplied site width, or a value the supplied evidence
            cannot give, naming the stage and op. Run
            :func:`~interpretune.analysis.venngram.support.require_supported` first for the full list at once.
    """
    payload: dict[str, list[InterventionSpec]] = {}
    for i, stage in enumerate(recipe.stages):
        if stage.tokens.kind not in _POSITION_SCOPES:
            raise MisconfigurationException(
                f"stage {i} ({stage.site}): token scope `{stage.tokens.kind}` has no interpretune position scope"
                f" (supported: {sorted(_POSITION_SCOPES)})"
            )
        scope = _POSITION_SCOPES[stage.tokens.kind]
        layer, site = split_site(stage.site)
        for j, op in enumerate(stage.ops):
            where = f"stage {i} ({stage.site}) op {j} (`{op.op}`)"
            if (refusal := op_refusal(recipe, stage, op)) is not None:
                raise MisconfigurationException(f"{where}: {refusal}")
            kind = target_kind(recipe, op)
            if kind == "activation":
                assert layer is not None  # op_refusal refuses a layer-free whole-activation op
                specs = {layer: _activation_spec(recipe, op, site_widths.get(site), scope, where)}
                hooked = site
            else:
                hooked = f"{site}.{_LATENT_SUBHOOK}" if kind == "features" else site
                width = _require_width(site_widths, hooked, i, stage)
                values = member_values(recipe, op, evidence=evidence, statistics=statistics, where=where)
                specs = _member_specs(op, values, width, scope, where)
            for spec_layer, spec in specs.items():
                payload.setdefault(f"blocks.{spec_layer}.{hooked}", []).append(spec)
    return {pattern: tuple(specs) for pattern, specs in payload.items()}


def _clean_activation(clean_activations: Any, layer: int, position: int, feature: int, where: str) -> float:
    shape = tuple(clean_activations.shape)
    if len(shape) != 3 or not (0 <= layer < shape[0] and 0 <= position < shape[1] and 0 <= feature < shape[2]):
        raise MisconfigurationException(
            f"{where}: the clean capture, of shape {shape} ([layer, position, feature]), has no activation at"
            f" layer {layer}, position {position}, feature {feature}"
        )
    return float(clean_activations[layer, position, feature])


def _relative_op_refusal(op: Any, support: FeatureInterventionSupport, clean_activations: Any | None) -> str | None:
    """Why a feature op computed from a clean activation cannot be computed here, or ``None``."""
    if op.op not in _RELATIVE_FEATURE_OPS:
        return None
    if not support.captures_clean_activations:
        return (
            "a feature `add` or `scale` is computed from the feature's clean activation, and this backend declares"
            " no clean capture (`captures_clean_activations`)"
        )
    if clean_activations is None:
        return (
            "a feature `add` or `scale` is computed from the feature's clean activation: pass `clean_activations`,"
            " from the backend's `capture_feature_activations` over the same prompt"
        )
    return None


def recipe_feature_interventions(
    recipe: Any,
    *,
    support: FeatureInterventionSupport,
    clean_activations: Any | None = None,
    evidence: Mapping[str, Mapping[str, Mapping[tuple[int, int], float]]] | None = None,
    statistics: Mapping[str, Statistic] | None = None,
) -> list[tuple[int, int, int, float]]:
    """Translate a recipe's feature ``replace``, ``add`` and ``scale`` ops into circuit-tracer's feature
    intervention tuples.

    A tuple sets a feature's value. ``replace`` sets the op's value $d$; ``add`` and ``scale`` set $a + d$ and
    $a \\cdot d$, where $a$ is the feature's clean activation at that layer and position, read from
    ``clean_activations``. The clean activation is the one the unintervened prompt produces, so where a recipe also
    intervenes at an earlier layer, an ``add`` is relative to the clean value rather than to the value the earlier
    intervention propagates to (the model-backend surface's ``add`` at the latent sub-hook is relative to the latter).

    Args:
        recipe: The recipe to translate; every op must be one :func:`feature_op_refusal` admits.
        support: The analysis backend's declaration. The tuples carry computed values, so it must honour ``constant``,
            and an ``add`` or ``scale`` needs it to declare ``captures_clean_activations``.
        clean_activations: The clean feature activations, indexed ``[layer, position, feature]``: the backend's
            ``capture_feature_activations`` over the prompt the tuples will be applied to, with the same
            ``apply_activation_function`` as the intervention call. Needed only for ``add`` and ``scale``.
        evidence: The tables evidence-sourced values read (:mod:`~interpretune.analysis.venngram.values`).
        statistics: The statistics their transforms name.

    Returns:
        ``(layer, position, feature_idx, value)`` tuples, in recipe order, for
        ``replacement_model.feature_intervention``.

    Raises:
        MisconfigurationException: For any op :func:`feature_op_refusal` refuses, a backend not honouring ``constant``,
            an ``add`` or ``scale`` without a clean capture (undeclared, or not passed), a member or position the
            capture has no activation for, or a value the supplied evidence cannot give, naming the stage and op.
    """
    if (why := support.refusal({"value_source": "constant", "value": 0.0})) is not None:
        raise MisconfigurationException(f"this backend cannot take a recipe's computed values: {why}")
    interventions: list[tuple[int, int, int, float]] = []
    for i, stage in enumerate(recipe.stages):
        for j, op in enumerate(stage.ops):
            where = f"stage {i} ({stage.site}) op {j} (`{op.op}`)"
            refusal = feature_op_refusal(recipe, stage, op) or _relative_op_refusal(op, support, clean_activations)
            if refusal is not None:
                raise MisconfigurationException(f"{where}: {refusal}")
            values = member_values(recipe, op, evidence=evidence, statistics=statistics, where=where)
            positions = (op.tokens or stage.tokens).positions
            relative = _RELATIVE_FEATURE_OPS.get(op.op)
            interventions += [
                (
                    layer,
                    position,
                    feature,
                    value
                    if relative is None
                    else relative(_clean_activation(clean_activations, layer, position, feature, where), value),
                )
                for (layer, feature), value in values.items()
                for position in positions
            ]
    return interventions
