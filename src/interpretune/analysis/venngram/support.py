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
- **of**: what targets may address, ``value`` or ``grad``, from the record. Both model backends declare ``grad`` beside
  ``value``, since each applies its ops to gradients through ``bwd_w_intervention``; a provider declaring ``value``
  alone has a recipe acting on a gradient refused here. Declaring both flows does not let one translation apply both:
  :func:`check_recipe` checks the ops for the one flow its caller applies, as the apply path does
  (:func:`~interpretune.analysis.venngram.apply.op_refusal`).
- **readouts**: the reductions a gradient stage's readout may use, those
  :mod:`~interpretune.analysis.venngram.readouts` executes, declared where the record declares ``grad``, since only a
  consumer acting on gradients takes them of a readout.

:func:`venngram_feature_support` is the analysis-backend path's declaration, derived from the provider of the
``features`` target that applies feature tuples: a feature ``replace`` at explicit positions, which circuit-tracer's
feature tuples express, and a feature ``add`` and ``scale`` where the target's detail declares a clean capture to
compute them from. :func:`venngram_feature_subsets` derives, from the same record, the graph-subset treatments that
path applies (venngram's ``circuits.subsets`` section): ``replace``, and ``add`` with a clean capture, on transcoder
and CLT nodes with scalar values, by any anchor, through the transforms that act on each value alone, and no
restriction. :func:`venngram_subsets` is the model-backend surface's graph-subset declaration, derived from
:func:`venngram_support`'s record: ``replace`` and ``add`` at an attached latent model's sub-hook, and a restriction
(the complement in a finite feature set), on SAE, transcoder and block-sparse nodes with scalar or vector values, where
feature targets and explicit positions are declared.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from interpretune.analysis.backends.capabilities import InterventionSupport, PositionScope
from interpretune.analysis.venngram.apply import (
    APPLIED_OPS,
    FEATURE_TUPLE_OPS,
    GRAPH_LATENT_FEATURIZERS,
    GRAPH_TUPLE_FEATURIZERS,
    GRAPH_TUPLE_OPS,
    GRAPH_TUPLE_TRANSFORMS,
    feature_op_refusal,
    graph_layout_refusal,
    op_refusal,
)
from interpretune.analysis.venngram.readouts import executed_readouts
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
        of=frozenset(intervention.of),
        readouts=executed_readouts() if "grad" in intervention.of else frozenset(),
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


def venngram_feature_subsets(intervention: InterventionSupport) -> Any:
    """Derive a ``venngram.support.SubsetSupport`` for the analysis-backend path: the treatments over graph subsets
    that :func:`~interpretune.analysis.venngram.apply.recipe_feature_interventions` applies.

    Its ops are those of :func:`venngram_feature_support` a graph target admits, so a backend that does not honour
    ``constant`` applies none, and ``add`` needs the clean capture. Every anchor kind is resolved, by venngram's
    ``GraphSubset.plan``.

    Args:
        intervention: The provider's record, as :func:`venngram_feature_support` takes it.

    Returns:
        A ``venngram.support.SubsetSupport``. Typed ``Any`` so importing this module needs no venngram install.

    Raises:
        MisconfigurationException: If venngram is not installed.
    """
    return _feature_subsets(venngram_feature_support(intervention))


def _feature_subsets(features: Any) -> Any:
    """The graph-subset declaration of the analysis-backend path whose feature declaration is ``features``."""
    graphs = importlib.import_module("venngram.graphs")
    return importlib.import_module("venngram.support").SubsetSupport(
        ops=frozenset(features.ops & GRAPH_TUPLE_OPS) if features.values else frozenset(),
        anchors=frozenset(graphs.ANCHORS),
        featurizers=GRAPH_TUPLE_FEATURIZERS,
        values=frozenset({"scalar"}),
        transforms=GRAPH_TUPLE_TRANSFORMS,
        complement=False,
    )


def venngram_subsets(intervention: InterventionSupport, *, latent_models: bool = False) -> Any:
    """Derive a ``venngram.support.SubsetSupport`` for the model-backend surface: the treatments over graph subsets
    that :func:`~interpretune.analysis.venngram.apply.recipe_interventions` writes at an attached latent model's
    sub-hook.

    A graph node is a feature at one position, so the surface needs the ``features`` target (a latent model attached,
    ``latent_models``) and the ``positions`` token scope; without either it applies no graph op.

    Args:
        intervention: The provider's intervention record, as :func:`venngram_support` takes it.
        latent_models: Whether a latent model is attached, as :func:`venngram_support` takes it.

    Returns:
        A ``venngram.support.SubsetSupport``. Typed ``Any`` so importing this module needs no venngram install.

    Raises:
        MisconfigurationException: If venngram is not installed.
    """
    return _latent_subsets(venngram_support(intervention, latent_models=latent_models))


def _latent_subsets(support: Any) -> Any:
    """The graph-subset declaration of the model-backend surface whose declaration is ``support``."""
    graphs = importlib.import_module("venngram.graphs")
    places = "features" in support.targets and "positions" in support.tokens
    return importlib.import_module("venngram.support").SubsetSupport(
        ops=frozenset(support.ops & GRAPH_TUPLE_OPS) if places else frozenset(),
        anchors=frozenset(graphs.ANCHORS),
        featurizers=GRAPH_LATENT_FEATURIZERS,
        values=frozenset({"scalar", "vector"}),
        transforms=GRAPH_TUPLE_TRANSFORMS,
        complement=bool(places and "replace" in support.ops),
        sites=None if support.sites is None or callable(support.sites) else frozenset(support.sites),
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


def check_recipe(recipe: Any, support: Any, *, analysis_backend: bool = False, flow: str = "value") -> list[str]:
    """Every way ``recipe`` (a ``venngram.treatment.Treatment``) asks for something interpretune does not honour.

    Venngram's own ``check_support`` covers the declared record (ops, targets, token scopes, value sources,
    transforms, bases, reachable sites). This adds the refusals the record cannot express, the ones the apply path
    would raise (:func:`~interpretune.analysis.venngram.apply.op_refusal`): an op on a target kind it has no native
    mode for, a whole-activation op at a layer-free site, a unit set at a layered one, or an operand shape or
    parameter no native translation expresses yet. With ``analysis_backend``, the refusals are those of
    :func:`~interpretune.analysis.venngram.apply.recipe_feature_interventions` instead. ``flow`` is the flow the
    caller will apply (``value`` through ``fwd_w_intervention``, ``grad`` through ``bwd_w_intervention``), and an op on
    the other flow is a refusal. An op on a graph subset is checked against the derived graph-subset declaration
    (:func:`venngram_feature_subsets` on the analysis backend, :func:`venngram_subsets` on the model backend), which
    venngram's ``check_support`` leaves to it; on the model backend, also against the layout one latent model's columns
    admit (:func:`~interpretune.analysis.venngram.apply.graph_layout_refusal`).

    A unit table the caller supplies is passed when the recipe is applied, so whether it has what a value reads is
    decided then (venngram checks the book's own tables when it reads the book).
    """
    vg_ops = _require_venngram()
    refusals = list(vg_ops.check_support(recipe, support))
    refusals += (_feature_subsets(support) if analysis_backend else _latent_subsets(support)).check_treatment(recipe)
    for i, stage in enumerate(recipe.stages):
        for j, op in enumerate(stage.ops):
            if getattr(op, "graph", None) is not None:
                # the declaration above checks what it can express; the model backend adds its columns' layout
                refusal = None if analysis_backend else graph_layout_refusal(recipe, stage, op, flow)
            elif analysis_backend:
                refusal = feature_op_refusal(recipe, stage, op)
            else:
                refusal = op_refusal(recipe, stage, op, flow)
            if refusal is not None:
                refusals.append(f"stage {i} ({stage.site}) op {j} (`{op.op}`): {refusal}")
    return refusals


def require_supported(recipe: Any, support: Any, *, analysis_backend: bool = False, flow: str = "value") -> None:
    """Refuse a recipe this consumer cannot honour in full, by name, before anything runs.

    Raises:
        MisconfigurationException: Listing every refusal :func:`check_recipe` finds.
    """
    if refusals := check_recipe(recipe, support, analysis_backend=analysis_backend, flow=flow):
        raise MisconfigurationException(
            "This recipe asks for more than interpretune honours:\n" + "\n".join(f"- {r}" for r in refusals)
        )
