r"""
The venngram Support Declaration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A venngram consumer declares what it honours as a ``venngram.ops.Support`` record and checks a recipe against it
with ``venngram.ops.check_support`` before running anything, so a recipe asking for more is refused by name ahead
of time rather than narrowed silently at run time. Interpretune's declaration is DERIVED from the records a
backend already publishes, never restated by hand:

- **ops** from :class:`~interpretune.analysis.backends.capabilities.InterventionSupport`'s ops, intersected
  with the ops this integration's apply path implements (:data:`~interpretune.analysis.venngram.apply.APPLIED_OPS`:
  every core venngram op). The record's op names are venngram's op names.
- **token scopes** from the same record's position scopes: ``last_token`` is venngram's ``last``,
  ``all_positions`` its ``all`` and ``positions`` its ``positions``. The remaining scopes (``prompt``,
  ``generated``, ``slice``) have no interpretune spelling yet and are not declared.
- **sites** from :class:`~interpretune.analysis.backends.capabilities.CaptureSupport`'s capturable points, which
  are layer-free base spellings like a recipe's sites, each also declared in its layered form
  (``blocks.*.<point>``), which a whole-activation op's stage names. A site the backend cannot capture on this
  model is a site a treatment cannot be verified at, so it is not declared reachable. Recipes name
  ``mlp.out.hook_in`` for the MLP hidden activation; the vocabulary's semantic spelling for that tensor is
  ``hook_mlp_hidden_in`` (the down projection's input), and both spellings parse to the one point.
- **targets and values** from the record's targets and value sources, through the transforms the apply path
  evaluates (:mod:`~interpretune.analysis.venngram.values`). A ``features`` target is declared only when the caller
  says a latent model is attached (``latent_models``), since that is a fact of the model at hand rather than of the
  backend. ``statistic`` is declared when the caller names statistics it will supply.
- **bases**: the bases an operand may be written in, from the record when it narrows them. An operand is applied as
  a tensor in the site's own activation coordinates, so a record naming no bases is declared with the bases known to
  be written there: by default interpretune's two J-lens bases (``jlens_folded``, ``jlens_unfolded``), both
  residual-stream directions. A caller whose operands come from elsewhere names their bases explicitly.

venngram's ``Support`` has no ``of`` field in the installed release, so the record's flows are not carried into it;
a recipe acting on a gradient is refused by the apply path until it does.

:func:`venngram_feature_support` is the analysis-backend path's declaration, derived from the provider of the
``features`` target that applies feature tuples: a feature ``replace`` at explicit positions, which circuit-tracer's
feature tuples express, and a feature ``add`` and ``scale`` where the target's detail declares a clean capture to
compute them from.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from interpretune.analysis.backends.capabilities import InterventionSupport, PositionScope
from interpretune.analysis.venngram.apply import APPLIED_OPS, FEATURE_TUPLE_OPS, feature_op_refusal, op_refusal
from interpretune.analysis.venngram.values import EVALUATED_TRANSFORMS
from interpretune.utils import MisconfigurationException

if TYPE_CHECKING:
    from interpretune.analysis.backends.capabilities import CaptureSupport

#: Interpretune position scopes by their venngram token-scope names.
_TOKEN_SCOPES = {
    PositionScope.LAST_TOKEN: "last",
    PositionScope.ALL_POSITIONS: "all",
    PositionScope.POSITIONS: "positions",
}

#: The operand bases declared by default: the directions interpretune itself writes in a site's activation space
#: (``optools.JLENS_BASIS_NAMES``, spelled out so this module imports no analysis machinery; a test keeps them equal).
DEFAULT_BASES = frozenset({"jlens_folded", "jlens_unfolded"})


def _require_venngram():
    try:
        return importlib.import_module("venngram.ops")
    except ImportError as e:
        raise MisconfigurationException(
            "The venngram integration needs venngram, an interpretune dependency: reinstall interpretune."
        ) from e


def _sites(points: Any) -> frozenset[str]:
    return frozenset(points) | {f"blocks.*.{p}" for p in points}


def venngram_support(
    intervention: InterventionSupport,
    capture: CaptureSupport | None = None,
    *,
    ops: frozenset[str] = APPLIED_OPS,
    bases: frozenset[str] | None = DEFAULT_BASES,
    latent_models: bool = False,
    statistics: frozenset[str] = frozenset(),
    weights: bool = False,
    name: str = "interpretune",
) -> Any:
    """Derive a ``venngram.ops.Support`` from a backend's own support records (see the module docstring).

    Args:
        intervention: The provider's intervention record (``ModuleCapabilities.intervention_for(target)``).
        capture: The backend's capture declaration for the model at hand; its capturable points bound the
            reachable sites. Without one, no site bound is declared (any site passes this check, and site
            resolution is left to the apply path).
        ops: The recipe ops to declare, intersected with the record's ops. Narrowing this declares less than the apply
            path implements; it must only name ops the apply path implements.
        bases: The operand bases to honour when the record names none (see the module docstring), or ``None`` for
            any.
        latent_models: Whether a latent model is attached, so feature targets have a sub-hook to act at.
        statistics: The statistics the caller will supply to ``recipe_interventions``; any declares ``statistic``.
        weights: Whether the caller materializes a recipe's ``weights`` section
            (:func:`~interpretune.tuning.weights.apply_recipe_weights`), so the parameter targets (``param``,
            ``param_units``) are declared. Their ops are the declared ops interpretune materializes on a tensor
            (:data:`~interpretune.tuning.weights.WEIGHT_OPS`).
        name: The consumer name the record carries.

    Returns:
        A ``venngram.ops.Support``. Typed ``Any`` so importing this module needs no venngram install.

    Raises:
        MisconfigurationException: If venngram is not installed.
    """
    vg_ops = _require_venngram()
    declared = intervention.declared_ops
    return vg_ops.Support(
        ops=frozenset(ops & declared & set(vg_ops.OPS)),
        targets=frozenset(t for t in intervention.targets if t != "features" or latent_models)
        | (frozenset(vg_ops.PARAM_TARGETS) if weights else frozenset()),
        tokens=frozenset(_TOKEN_SCOPES[s] for s in intervention.position_scopes),
        values=frozenset(intervention.values),
        transforms=EVALUATED_TRANSFORMS | ({"statistic"} if statistics else frozenset()),
        sites=None if capture is None else _sites(capture.capturable),
        bases=intervention.bases if intervention.bases is not None else bases,
        name=name,
    )


def venngram_feature_support(
    intervention: InterventionSupport,
    *,
    statistics: frozenset[str] = frozenset(),
    name: str = "interpretune",
) -> Any:
    """Derive a ``venngram.ops.Support`` for the analysis-backend path from the intervention record of the
    component applying feature tuples.

    Args:
        intervention: That provider's record (``ModuleCapabilities.intervention_for("features",
            provider="analysis_backend")``); its ``features`` target carries the detail.
            ``recipe_feature_interventions`` passes computed values, so a detail not honouring ``constant`` declares no
            value source at all, and one declaring no clean capture declares ``replace`` alone.
        statistics: The statistics the caller will supply; any declares ``statistic``.
        name: The consumer name the record carries.

    Returns:
        A ``venngram.ops.Support``. Typed ``Any`` so importing this module needs no venngram install.

    Raises:
        MisconfigurationException: If venngram is not installed.
    """
    vg_ops = _require_venngram()
    detail = intervention.target_detail("features")
    constant = detail is not None and "constant" in detail.value_sources
    tuple_ops = frozenset(intervention.declared_ops & {"replace"})
    captures = bool(tuple_ops) and detail is not None and detail.captures_clean_activations
    relative = FEATURE_TUPLE_OPS - {"replace"} if captures else frozenset()
    return vg_ops.Support(
        ops=tuple_ops | relative,
        targets=frozenset({"features"}),
        tokens=frozenset(_TOKEN_SCOPES[s] for s in intervention.position_scopes),
        values=frozenset(intervention.values) if constant else frozenset(),
        transforms=EVALUATED_TRANSFORMS | ({"statistic"} if statistics else frozenset()),
        sites=None,
        bases=None,
        name=name,
    )


def support_mapping(support: Any) -> dict[str, Any]:
    """A derived ``venngram.ops.Support`` as its publishable ``support_record`` document.

    The document is what a recipe is checked against without this consumer installed (venngram's conformance command
    takes it directly). It is venngram's own schema 2 form, with the one contract interpretune offers as its
    ``steering`` section: the same fields as the in-process record, sorted, with an unbounded site set omitted rather
    than spelled as an empty list. A section it omits (``tuning`` is finetuning-scheduler's to declare) is a contract
    interpretune does not offer. A callable site predicate cannot be published and is refused by name rather than
    dropped.
    """
    if callable(support.sites):
        raise MisconfigurationException(
            "A support record with a callable site predicate cannot be published as a document; derive it from a "
            "CaptureSupport instead."
        )
    _require_venngram()
    return importlib.import_module("venngram.support").as_record(support).to_mapping()


def check_recipe(recipe: Any, support: Any, *, analysis_backend: bool = False) -> list[str]:
    """Every way ``recipe`` (a ``venngram.recipe.Recipe``) asks for something interpretune does not honour.

    Venngram's own ``check_support`` covers the declared record (ops, targets, token scopes, value sources,
    transforms, bases, reachable sites). This adds the refusals the record cannot express, the ones the apply path
    would raise (:func:`~interpretune.analysis.venngram.apply.op_refusal`): an op on a target kind it has no native
    mode for, a whole-activation op at a layer-free site, a unit set at a layered one, or an operand shape or
    parameter no native translation expresses yet. With ``analysis_backend``, the refusals are those of
    :func:`~interpretune.analysis.venngram.apply.recipe_feature_interventions` instead.

    Evidence is supplied when the recipe is applied, so whether it has what a value reads is decided then.
    """
    vg_ops = _require_venngram()
    refuse = feature_op_refusal if analysis_backend else op_refusal
    refusals = list(vg_ops.check_support(recipe, support))
    for i, stage in enumerate(recipe.stages):
        for j, op in enumerate(stage.ops):
            if (refusal := refuse(recipe, stage, op)) is not None:
                refusals.append(f"stage {i} ({stage.site}) op {j} (`{op.op}`): {refusal}")
    return refusals


def require_supported(recipe: Any, support: Any, *, analysis_backend: bool = False) -> None:
    """Refuse a recipe this consumer cannot honour in full, by name, before anything runs.

    Raises:
        MisconfigurationException: Listing every refusal :func:`check_recipe` finds.
    """
    if refusals := check_recipe(recipe, support, analysis_backend=analysis_backend):
        raise MisconfigurationException(
            "This recipe asks for more than interpretune honours:\n" + "\n".join(f"- {r}" for r in refusals)
        )
