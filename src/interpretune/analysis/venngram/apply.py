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

- ``add``: the op's per-unit dose table (venngram's own ``Op.unit_doses``) scattered into a dense vector at the
  site's width, applied as ``add``;
- ``scale``: the same table as a multiplier, ones where no unit is named, applied as ``scale``.

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

from interpretune.analysis.backends.capabilities import InterventionMode, PositionScope
from interpretune.analysis.backends.interventions import InterventionSpec
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

_LAYERED_SITE = re.compile(r"^blocks\.(\d+)\.(.+)$")


def split_site(site: str) -> tuple[int | None, str]:
    """A stage site's layer (``None`` when it names none) and its layer-free spelling."""
    m = _LAYERED_SITE.match(site)
    return (int(m.group(1)), m.group(2)) if m else (None, site)


def op_refusal(recipe: Any, stage: Any, op: Any) -> str | None:
    """Why ``op`` (at ``stage`` of ``recipe``) cannot be translated, or ``None`` when it can."""
    layer, _ = split_site(stage.site)
    kind = op.target_kind(recipe.book)
    value = op.value
    if value is not None and (value.source not in ("dose", "doses") or value.transform):
        return "the value is not a plain dose or doses (evidence-sourced values are not applied yet)"
    if kind == "features":
        return "feature targets are not applied yet"
    if kind == "units":
        if op.op not in _UNIT_OPS:
            return f"a unit-targeted `{op.op}` has no native mode (unit-targeted: {sorted(_UNIT_OPS)})"
        if op.operand is not None:
            return "a unit-targeted op takes a dose, not an operand"
        if layer is not None:
            return f"a unit set names its own layers, so its stage names the layer-free site, not `{stage.site}`"
        return None
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


def _unit_specs(recipe: Any, op: Any, width: int, scope: PositionScope, where: str) -> dict[int, InterventionSpec]:
    mode, fill = _UNIT_OPS[op.op]
    by_layer: dict[int, dict[int, float]] = {}
    for (layer, unit), dose in op.unit_doses(recipe.book).items():
        by_layer.setdefault(layer, {})[unit] = dose
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


def recipe_interventions(recipe: Any, *, site_widths: Mapping[str, int]) -> dict[str, tuple[InterventionSpec, ...]]:
    """Translate a supported ``venngram.recipe.Recipe`` into a backend intervention payload.

    Args:
        recipe: The recipe to translate.
        site_widths: The width of every unit-targeted site the recipe names, keyed by the recipe's own layer-free
            site spelling. A width given for a whole-activation op's site also checks its operand's width.

    Returns:
        A mapping of hook patterns (``blocks.<layer>.<site>``) to the ordered specs to apply at each, in the raw
        form :func:`~interpretune.analysis.backends.interventions.build_intervention_dict` canonicalizes.

    Raises:
        MisconfigurationException: For anything this module cannot translate (see :func:`op_refusal`), an
            unmapped token scope, a unit-targeted site whose width was not supplied, a unit outside its site's
            width, or an operand whose width differs from a supplied site width, naming the stage and op. Run
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
            if op.target_kind(recipe.book) == "units":
                if site not in site_widths:
                    raise MisconfigurationException(
                        f"stage {i} ({stage.site}): no width was supplied for this site"
                        f" (supplied: {sorted(site_widths)})"
                    )
                specs = _unit_specs(recipe, op, site_widths[site], scope, where)
            else:
                assert layer is not None  # op_refusal refuses a layer-free whole-activation op
                specs = {layer: _activation_spec(recipe, op, site_widths.get(site), scope, where)}
            for spec_layer, spec in specs.items():
                payload.setdefault(f"blocks.{spec_layer}.{site}", []).append(spec)
    return {pattern: tuple(specs) for pattern, specs in payload.items()}
