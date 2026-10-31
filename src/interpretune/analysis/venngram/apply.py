r"""
Applying a Recipe Through Native Interventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A supported recipe is translated into interpretune's own intervention payload, the raw mapping every backend
canonicalizes with :func:`~interpretune.analysis.backends.interventions.build_intervention_dict` once concrete
hook shapes are known. The translation is deliberately the plainest one the declared support
(:mod:`~interpretune.analysis.venngram.support`) covers:

- each stage's unit-targeted ``add`` becomes, per layer its book set names, one
  :class:`~interpretune.analysis.backends.interventions.InterventionSpec` keyed by the hook pattern
  ``blocks.<layer>.<site>``, whose intervention tensor is the op's per-unit dose table (from venngram's own
  ``Op.unit_doses``) scattered into a dense vector at the site's width. A unit outside that width is refused by
  name here rather than dropped or wrapped;
- the stage's token scope is the spec's ``position_scope`` (``all`` -> ``all_positions``, ``last`` ->
  ``last_token``); an op may not override its stage's scope (venngram refuses that at parse).

The caller supplies each site's width (from the model's own configuration: the MLP hidden width for
``mlp.out.hook_in`` / ``hook_mlp_hidden_in``, the residual width for ``mlp.hook_out``), because the width is a
fact of the checkpoint the recipe pins, not of the recipe, and guessing it from the largest unit named would
silently truncate a dose table that names a unit past the guess.

Anything the declaration does not cover was already refused by
:func:`~interpretune.analysis.venngram.support.require_supported`; this module refuses again what it cannot
translate, by name, so a caller that skipped the check still cannot get a silently narrowed treatment.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from interpretune.analysis.backends.capabilities import InterventionMode, PositionScope
from interpretune.analysis.backends.interventions import InterventionSpec
from interpretune.utils import MisconfigurationException

#: Recipe token scopes by their interpretune position-scope names.
_POSITION_SCOPES = {"all": PositionScope.ALL_POSITIONS, "last": PositionScope.LAST_TOKEN}


def _scatter(doses: Mapping[int, float], width: int, where: str, layer: int) -> torch.Tensor:
    tensor = torch.zeros(width)
    for unit, dose in doses.items():
        if not 0 <= unit < width:
            raise MisconfigurationException(f"{where}: unit {unit} at layer {layer} is outside the site width {width}")
        tensor[unit] = dose
    return tensor


def recipe_interventions(recipe: Any, *, site_widths: Mapping[str, int]) -> dict[str, tuple[InterventionSpec, ...]]:
    """Translate a supported ``venngram.recipe.Recipe`` into a backend intervention payload.

    Args:
        recipe: The recipe to translate.
        site_widths: The width of every site the recipe names, keyed by the recipe's own site spelling.

    Returns:
        A mapping of hook patterns (``blocks.<layer>.<site>``) to the ordered specs to apply at each, in the raw
        form :func:`~interpretune.analysis.backends.interventions.build_intervention_dict` canonicalizes.

    Raises:
        MisconfigurationException: For anything outside the declared support (a non-``add`` op, an operand, a
            non-unit target, an evidence or transformed value, an unmapped token scope), a site whose width was
            not supplied, or a unit outside its site's width, naming the stage and op. Run
            :func:`~interpretune.analysis.venngram.support.require_supported` first for the full list at once.
    """
    payload: dict[str, list[InterventionSpec]] = {}
    for i, stage in enumerate(recipe.stages):
        if stage.site not in site_widths:
            raise MisconfigurationException(
                f"stage {i} ({stage.site}): no width was supplied for this site (supplied: {sorted(site_widths)})"
            )
        if stage.tokens.kind not in _POSITION_SCOPES:
            raise MisconfigurationException(
                f"stage {i} ({stage.site}): token scope `{stage.tokens.kind}` has no interpretune position scope"
                f" (supported: {sorted(_POSITION_SCOPES)})"
            )
        scope = _POSITION_SCOPES[stage.tokens.kind]
        for j, op in enumerate(stage.ops):
            where = f"stage {i} ({stage.site}) op {j} (`{op.op}`)"
            if op.op != "add" or op.operand is not None:
                raise MisconfigurationException(f"{where}: only unit-targeted `add` doses are applied today")
            if op.target_kind(recipe.book) != "units":
                raise MisconfigurationException(f"{where}: the target is not a unit set")
            if op.value is None or op.value.source not in ("dose", "doses") or op.value.transform:
                raise MisconfigurationException(f"{where}: the value is not a plain dose or doses")
            by_layer: dict[int, dict[int, float]] = {}
            for (layer, unit), dose in op.unit_doses(recipe.book).items():
                by_layer.setdefault(layer, {})[unit] = dose
            for layer in sorted(by_layer):
                payload.setdefault(f"blocks.{layer}.{stage.site}", []).append(
                    InterventionSpec(
                        intervention_tensor=_scatter(by_layer[layer], site_widths[stage.site], where, layer),
                        mode=InterventionMode.ADD,
                        position_scope=scope,
                    )
                )
    return {pattern: tuple(specs) for pattern, specs in payload.items()}
