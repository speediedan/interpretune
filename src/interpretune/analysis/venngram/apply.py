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
- ``replace``: the values at the named units only, preserving the unselected coordinates.

The values are a ``dose``, a ``doses`` table, or a column of evidence the caller supplies through a transform pipeline
(:mod:`~interpretune.analysis.venngram.values`).

**Feature-targeted ops** take the same path one level down, into a latent model's feature activations: ``add`` and
``scale`` scatter at the latent sub-hook, ``blocks.<layer>.<site>.hook_sae_acts_post``, which the backend resolves to
the attached latent model. The stage names the hooked site (``hook_resid_post``), and the width supplied is the
latent's, keyed ``<site>.hook_sae_acts_post`` because the site's own width differs. A ``replace`` on features has no
native mode here, since a spec replaces the whole tensor; circuit-tracer's feature tuples express it.

**Whole-activation ops** (no ``target``) name no layer of their own, so their stage names a layered site,
``blocks.<layer>.<site>``. A layer-free one is refused rather than applied at a guessed layer or at every
layer. The operand (a vector, a pair, or a subspace stacked on its leading axis) is read from the recipe inline or
from its sidecar, relative to the recipe's own directory:

- ``add`` / ``replace``: the operand vector, times the op's ``dose`` when it has one (``add`` through
  ``scale_factor``; ``replace``, which ignores ``scale_factor``, by scaling the tensor). Without an operand, the
  ``dose`` itself, broadcast over the activation;
- ``scale``: the ``dose``, as a scalar multiplier;
- ``project``: onto a vector operand, keeping all of it (``fraction`` 1). A subspace operand, or a fraction below
  1 (which keeps the span and only part of its complement), has no native translation yet and is refused;
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

**Graph-targeted ops** (``target: {graph: <name>}``, :mod:`venngram.graphs`) apply a graph subset distilled from one
prompt in another, on the same analysis-backend path. Each feature node at the stage's site is placed by its anchor in
the destination prompt (``token_ids``), and becomes a tuple at that layer, position and feature: ``replace`` sets the
node's value (its column, or its delta from a baseline column), and ``add`` sets the feature's clean activation plus
it. Only transcoder and CLT nodes (one feature index, a scalar value) are placed. A restriction (``outside``), which
ablates every member of a universe the graph does not place, is refused here: as tuples it would enumerate each one,
at every position. The model-backend surface does not place graph nodes yet, so :func:`op_refusal` refuses them.
:func:`graph_subset_overlap` places a subset the same way in the prompt of another attribution graph and reports which
of its nodes that graph selects and what share of the graph's influence they carry, a measure of how much of the
reasoning the subset describes transferred.

**Gradient-targeted ops** are unit- or feature-targeted ops whose target set addresses ``of: grad``: the gradient with
respect to those units of the activation at the site. ``recipe_interventions(..., flow="grad")`` translates them into
the same specs, for the backward surface (``bwd_w_intervention``, the ``model_bwd_intervention`` op), and refuses the
recipe's value ops there, as the default ``flow="value"`` refuses its gradient ops: one translation, one flow.

Anything the declaration does not cover is refused by
:func:`~interpretune.analysis.venngram.support.require_supported`, which lists every refusal this module would
raise (:func:`op_refusal`) at once; this module refuses again what it cannot translate, by name, so a caller that
skipped the check still cannot get a silently narrowed treatment.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from interpretune.analysis.backends.capabilities import FLOWS, FeatureTargetSupport, PositionScope
from interpretune.analysis.backends.interventions import InterventionSpec
from interpretune.analysis.venngram.values import Statistic, member_refusal, member_values, target_kind
from interpretune.utils import MisconfigurationException

#: Recipe token scopes by their interpretune position-scope names.
_POSITION_SCOPES = {"all": PositionScope.ALL_POSITIONS, "last": PositionScope.LAST_TOKEN}

#: The dense unit ops' identity values; replacement instead preserves its complement through a selected spec.
_DENSE_UNIT_OPS = {"add": 0.0, "scale": 1.0}
_UNIT_OPS = frozenset(_DENSE_UNIT_OPS) | {"replace"}

#: The whole-activation ops this path translates; each applies as the interpretune op of the same name.
_ACTIVATION_OPS = frozenset({"add", "replace", "scale", "project", "reject", "patch", "clamp"})

#: Every recipe op this module translates (for at least one target kind).
APPLIED_OPS = frozenset(_UNIT_OPS) | _ACTIVATION_OPS

#: The feature ops a feature tuple's value is computed for: ``replace`` sets the op's value; ``add`` and ``scale`` set a
#: value relative to the feature's clean activation $a$ ($a + d$ and $a \cdot d$ for the op's value $d$).
_RELATIVE_FEATURE_OPS = {"add": lambda a, d: a + d, "scale": lambda a, d: a * d}
FEATURE_TUPLE_OPS = frozenset({"replace"}) | frozenset(_RELATIVE_FEATURE_OPS)

#: The graph-target ops the feature tuples apply: ``replace`` sets a node's value; ``add`` sets the feature's clean
#: activation plus it.
GRAPH_TUPLE_OPS = frozenset({"replace", "add"})

#: The featurizer kinds whose graph nodes are circuit-tracer features: one feature index and a scalar value.
GRAPH_TUPLE_FEATURIZERS = frozenset({"transcoder", "clt"})

#: The transforms applied to a graph value, each acting on one value alone.
_GRAPH_TRANSFORMS = {"scale": lambda v, params: v * params["factor"], "abs": lambda v, _: abs(v)}
GRAPH_TUPLE_TRANSFORMS = frozenset(_GRAPH_TRANSFORMS)

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
    if kind == "features" and op.op == "replace":
        return (
            "a feature-targeted `replace` has no native translation at a sub-hook: a spec replaces the whole tensor"
            " (`recipe_feature_interventions` gives circuit-tracer's tuples)"
        )
    if op.op not in _UNIT_OPS:
        return f"a {kind[:-1]}-targeted `{op.op}` has no native translation (supported: {sorted(_UNIT_OPS)})"
    if op.operand is not None:
        return f"a {kind[:-1]}-targeted op takes a dose, not an operand"
    if layer is not None:
        return f"a {kind[:-1]} set names its own layers, so its stage names the layer-free site, not `{stage.site}`"
    if kind == "features" and site.endswith(_LATENT_SUBHOOK):
        return f"a feature target's stage names the hooked site; `{_LATENT_SUBHOOK}` is implied, not `{site}`"
    return member_refusal(recipe, op)


def _flow_refusal(recipe: Any, op: Any, flow: str) -> str | None:
    """Why an op acting on one flow cannot be translated for the surface applying the other: the forward surface writes
    activation values and the backward surface writes gradients, so translating it would apply the op to a different
    tensor. A target set addressing ``of: grad`` names the gradient through those units; an op with no target acts on
    the value.
    """
    acts_on = "value" if op.target is None else op.target_of(recipe.book)
    if acts_on == flow:
        return None
    if acts_on == "grad":
        return (
            "its target addresses the gradient (`of: grad`), and this translation is for activation values: translate "
            "it with `flow='grad'` and apply it through `bwd_w_intervention`, rather than to the value"
        )
    return (
        "it acts on the activation value, and this translation is for gradients: a recipe's ops are applied one flow "
        "at a time, so translate it with `flow='value'` and apply it through `fwd_w_intervention`"
    )


def op_refusal(recipe: Any, stage: Any, op: Any, flow: str = "value") -> str | None:
    """Why ``op`` (at ``stage`` of ``recipe``) cannot be translated for the ``flow`` surface (``value`` for
    ``fwd_w_intervention``, ``grad`` for ``bwd_w_intervention``), or ``None`` when it can."""
    if (refusal := _flow_refusal(recipe, op, flow)) is not None:
        return refusal
    if getattr(op, "graph", None) is not None:
        return (
            "a graph target is placed on the analysis-backend path (`recipe_feature_interventions`); this surface does"
            " not place graph nodes yet"
        )
    layer, _ = split_site(stage.site)
    kind = target_kind(recipe, op)
    if kind in ("units", "features"):
        return _member_refusal(recipe, stage, op, kind)
    value = op.value
    if (refusal := _value_refusal(op)) is not None:
        return refusal
    if op.op not in _ACTIVATION_OPS:
        return f"the op has no native translation (supported: {sorted(_ACTIVATION_OPS)})"
    if layer is None:
        return f"a whole-activation op names no layer: write its stage's site as `blocks.<layer>.{stage.site}`"
    if value is not None and value.source != "dose":
        return "a whole-activation op takes a single `dose`, not a per-unit table"
    if op.op == "project":
        if recipe.operands[op.operand].kind != "vector":
            return "projecting onto a subspace has no native translation yet (a vector operand projects)"
        if op.params.get("fraction", 1.0) != 1.0:
            return "a `project` fraction below 1 has no native translation yet (fraction 1 projects)"
    return None


def _graph_op_refusal(recipe: Any, stage: Any, op: Any) -> str | None:
    """Why a graph-targeted op cannot become circuit-tracer feature tuples, or ``None`` when it can."""
    if op.outside is not None:
        return (
            "a restriction (`outside`) ablates every member of its universe the graph does not place, at every"
            " position; as feature tuples it would enumerate each one, so it is not applied here"
        )
    if op.op not in GRAPH_TUPLE_OPS:
        return f"circuit-tracer's feature tuples apply {sorted(GRAPH_TUPLE_OPS)} to a graph target, not `{op.op}`"
    graph = recipe.book.graphs[op.graph]
    spaces = {n.space: graph.spaces[str(n.space)] for n in graph.nodes if n.kind == "feature" and n.site == stage.site}
    for name in sorted(spaces):
        space = spaces[name]
        if space.kind not in GRAPH_TUPLE_FEATURIZERS:
            return (
                f"space `{name}` is a `{space.kind}` featurizer, which graph targets do not yet support:"
                f" circuit-tracer's feature tuples place {sorted(GRAPH_TUPLE_FEATURIZERS)} features"
            )
        if space.value_shape or len(space.index) != 1:
            return f"space `{name}`'s nodes are not one feature with a scalar value, as a feature tuple sets"
    if unapplied := sorted({t.name for t in op.value.transform} - GRAPH_TUPLE_TRANSFORMS):
        return f"{unapplied} are not applied to a graph value (applied: {sorted(GRAPH_TUPLE_TRANSFORMS)})"
    return None


def _graph_targets(
    recipe: Any, stage: Any, op: Any, token_ids: Sequence[int] | None, tokenizer: Mapping[str, str] | None, where: str
) -> list[tuple[int, int, int, float]]:
    """``(layer, position, feature, value)`` for each of a graph's feature nodes at the stage's site, placed by its
    anchor in the destination prompt, before any clean activation is added."""
    if token_ids is None:
        raise MisconfigurationException(
            f"{where}: a graph target places its nodes by their anchors in the prompt: pass `token_ids`, the prompt's"
            " unpadded input ids"
        )
    from venngram.errors import AddressError

    value = op.value
    from_graph = value.source == "graph"
    graph = recipe.book.graphs[op.graph]
    try:
        plan = graph.plan(
            list(token_ids),
            column=value.column if from_graph else graph.columns[0],
            baseline=value.baseline if from_graph else None,
            sites=[stage.site],
            tokenizer=tokenizer,
        )
    except AddressError as err:
        raise MisconfigurationException(f"{where}: {err}") from err
    targets = []
    for placement in plan:
        v = float(placement.value[0]) if from_graph else float(value.dose)
        for transform in value.transform:
            v = _GRAPH_TRANSFORMS[transform.name](v, transform.params)
        targets.append((placement.layer, placement.position, placement.index[0], v))
    return targets


@dataclass(frozen=True)
class GraphSubsetOverlap:
    """How much of a graph subset an attribution graph of another prompt contains
    (:func:`graph_subset_overlap`)."""

    placed: tuple[tuple[str, int, int, int], ...]
    """``(node, layer, position, feature)`` for each placement of the subset's feature nodes in the graph's
    prompt."""
    matched: Mapping[tuple[str, int, int, int], float]
    """Each placement that is a selected feature of the graph, with its activation there."""
    node_overlap: float
    """The fraction of placements the graph selects."""
    recovered_influence: float | None
    """The matched features' share of the graph's total feature influence, when the influence is given."""


def graph_subset_overlap(
    subset: Any,
    graph: Any,
    *,
    token_ids: Sequence[int] | None = None,
    tokenizer: Mapping[str, str] | None = None,
    influence: Any | None = None,
    sites: Sequence[str] | None = None,
) -> GraphSubsetOverlap:
    """Compare a graph subset with an attribution graph of another prompt: which of its feature nodes, placed by their
    anchors in that prompt, the graph selects, and how much of the graph's influence they carry.

    After a subset is applied in a new prompt, attributing that prompt and comparing its graph with the subset measures
    how much of the reasoning the subset describes the new prompt's graph contains. A placement matches when the graph
    selects a feature at its layer, position and feature index.

    Args:
        subset: A venngram ``GraphSubset`` (:func:`~interpretune.analysis.venngram.graphs.export_graph_subset`
            returns one), or a ``graph_subset`` document.
        graph: The circuit-tracer ``Graph`` of the destination prompt.
        token_ids: The destination prompt's unpadded input ids the anchors resolve in; the graph's ``input_tokens``
            when not given.
        tokenizer: ``{id, revision}`` of the destination tokenizer, which a ``token_match`` anchor needs to be the one
            the subset's source pins.
        influence: One influence score per selected feature of ``graph``, in its order (the analysis backend's
            ``compute_node_influence_scores``), for ``recovered_influence``.
        sites: Only the subset's nodes at these sites (all of them when ``None``).

    Returns:
        A :class:`GraphSubsetOverlap`.

    Raises:
        MisconfigurationException: If the subset is malformed or a node does not resolve in the prompt (the whole
            comparison is refused, as a graph target's placement is), a node lies in a space whose units are not one
            feature with a scalar value, or ``influence`` is not one score per selected feature.
    """
    from venngram.errors import AddressError

    graphs_mod = importlib.import_module("venngram.graphs")
    try:
        if isinstance(subset, Mapping):
            subset = graphs_mod.GraphSubset.from_mapping(subset, source="the compared graph subset")
        unsupported = sorted(
            name for name, space in subset.spaces.items() if len(space.index) != 1 or space.value_shape
        )
        if unsupported:
            raise MisconfigurationException(
                f"spaces {unsupported} address a unit by more than one index or carry a vector value, which a"
                " circuit-tracer graph's features do not"
            )
        prompt = graph.input_tokens.detach().cpu().flatten().tolist() if token_ids is None else token_ids
        plan = subset.plan([int(t) for t in prompt], sites=sites, tokenizer=tokenizer)
    except AddressError as err:
        raise MisconfigurationException(str(err)) from err
    selected = graph.selected_features.detach().cpu()
    rows = {
        tuple(coordinate): i
        for i, coordinate in enumerate(graph.active_features.detach().cpu().index_select(0, selected).tolist())
    }
    activations = graph.activation_values.detach().cpu().float().tolist()
    placed = tuple((p.node, p.layer, p.position, p.index[0]) for p in plan)
    hits = {key: rows[key[1:]] for key in placed if key[1:] in rows}
    recovered = None
    if influence is not None:
        scores = torch.as_tensor(influence).detach().cpu().float().flatten()
        if scores.numel() != len(rows):
            raise MisconfigurationException(
                f"`influence` has {scores.numel()} scores; it needs one per selected feature of the graph ({len(rows)})"
            )
        total = float(scores.sum())
        recovered = float(scores[sorted(set(hits.values()))].sum()) / total if total else 0.0
    return GraphSubsetOverlap(
        placed=placed,
        matched={key: activations[row] for key, row in hits.items()},
        node_overlap=len(hits) / len(placed),
        recovered_influence=recovered,
    )


def feature_op_refusal(recipe: Any, stage: Any, op: Any) -> str | None:
    """Why ``op`` cannot become circuit-tracer feature tuples, or ``None`` when it can.

    Whether the backend can capture the clean activations a feature ``add`` or ``scale`` is computed from is the
    backend's declaration, not the recipe's, so :func:`recipe_feature_interventions` and the derived support record
    decide it.
    """
    if getattr(op, "graph", None) is not None:
        return _graph_op_refusal(recipe, stage, op)
    if target_kind(recipe, op) != "features":
        return "circuit-tracer's feature tuples act on a feature target only"
    if (refusal := _flow_refusal(recipe, op, "value")) is not None:
        return refusal
    if op.op not in FEATURE_TUPLE_OPS:
        return f"circuit-tracer's feature tuples express {sorted(FEATURE_TUPLE_OPS)}, not `{op.op}`"
    if (op.tokens or stage.tokens).kind != "positions":
        return "circuit-tracer's feature tuples name their positions: write the stage's tokens as `{positions: [...]}`"
    if op.value is None:
        return "a feature-targeted op takes a value"
    return member_refusal(recipe, op)


def _unit_bounds(doses: Mapping[int, float], width: int, where: str, layer: int) -> None:
    for unit in doses:
        if not 0 <= unit < width:
            raise MisconfigurationException(f"{where}: unit {unit} at layer {layer} is outside the site width {width}")


def _scatter(doses: Mapping[int, float], width: int, fill: float, where: str, layer: int) -> torch.Tensor:
    _unit_bounds(doses, width, where, layer)
    tensor = torch.full((width,), fill)
    for unit, dose in doses.items():
        tensor[unit] = dose
    return tensor


def _unit_clamp_args(value: Any, lower: float | None, upper: float | None) -> None:
    if lower is None and upper is None:
        raise MisconfigurationException("a unit clamp needs clamp_min or clamp_max")
    if value is not None:
        raise MisconfigurationException("a unit clamp takes coordinate bounds, not a value")
    if lower is not None and upper is not None and lower > upper:
        raise MisconfigurationException(f"unit clamp bounds {lower} and {upper} are reversed")


def _unit_mode_args(op: str, value: Any, lower: float | None, upper: float | None) -> None:
    if op == "clamp":
        _unit_clamp_args(value, lower, upper)
        return
    if op not in {"add", "scale", "replace"}:
        raise MisconfigurationException(f"unit interventions apply add, scale, replace or clamp, not {op!r}")
    if lower is not None or upper is not None:
        raise MisconfigurationException(f"unit op {op!r} takes a value, not clamp bounds")
    if value is None:
        raise MisconfigurationException(f"unit op {op!r} needs an explicit value")


def _unit_value(value: Any, layer: int, count: int, hook: str) -> torch.Tensor:
    if isinstance(value, Mapping):
        if layer not in value:
            raise MisconfigurationException(f"no unit intervention value was supplied for layer {layer}")
        value = value[layer]
    tensor = torch.as_tensor(value)
    if tensor.shape not in (torch.Size(), torch.Size([count])):
        raise MisconfigurationException(
            f"{hook}: value shape {tuple(tensor.shape)} needs a scalar or {count} unit values"
        )
    return tensor


def unit_interventions(
    address: Any,
    *,
    n_layers: int,
    site_widths: Mapping[str, int],
    op: str = "add",
    value: Any = None,
    position_scope: PositionScope | str = PositionScope.LAST_TOKEN,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    sets: Mapping[str, Any] | None = None,
    flow: str = "value",
) -> dict[str, InterventionSpec]:
    """Build a direct backend payload editing only the units a venngram address selects.

    ``add``, ``scale`` and ``replace`` take a scalar or a vector in ascending selected-unit order, or a mapping
    from layer index to either. ``clamp`` bounds those coordinates in the model's unit basis; it takes bounds and
    no dose. This direct payload does not change venngram's recipe registry, whose clamp target is a whole
    activation with a basis operand. A gradient address requires ``flow="grad"`` and the backward surface.

    Args:
        address: A unit-site address or book reference, including per-layer unit sets.
        n_layers: The block count.
        site_widths: Model last-axis widths by layer-free point spelling.
        op: One of ``add``, ``scale``, ``replace`` and ``clamp``.
        value: The dose or replacement values (unused by ``clamp``).
        position_scope: The token scope, independent of which units the address selects.
        clamp_min: Optional lower coordinate bound.
        clamp_max: Optional upper coordinate bound.
        sets: Address-book sets for references.
        flow: The surface the caller will apply: ``value`` or ``grad``.

    Raises:
        MisconfigurationException: The op, address, layer dose, value shape or clamp bounds are unsupported.
    """
    from interpretune.analysis.venngram.sites import resolve_unit_sites

    _unit_mode_args(op, value, clamp_min, clamp_max)
    selection = resolve_unit_sites(address, n_layers=n_layers, site_widths=site_widths, sets=sets)
    if flow not in FLOWS or selection.flow != flow:
        raise MisconfigurationException(f"unit address acts on {selection.flow!r}, but the requested flow is {flow!r}")
    payload = {}
    for hook, indices in selection.units.items():
        layer, _ = split_site(hook)
        assert layer is not None
        tensor = torch.eye(len(indices)) if op == "clamp" else _unit_value(value, layer, len(indices), hook)
        payload[hook] = InterventionSpec(
            tensor, op=op, unit_indices=indices, position_scope=position_scope, clamp_min=clamp_min, clamp_max=clamp_max
        )
    return payload


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
    by_layer: dict[int, dict[int, float]] = {}
    for (layer, unit), value in values.items():
        by_layer.setdefault(layer, {})[unit] = value
    if op.op == "replace":
        specs = {}
        for layer, doses in sorted(by_layer.items()):
            _unit_bounds(doses, width, where, layer)
            units = tuple(sorted(doses))
            specs[layer] = InterventionSpec(
                intervention_tensor=torch.tensor([doses[unit] for unit in units]),
                op=op.op,
                position_scope=scope,
                unit_indices=units,
            )
        return specs
    fill = _DENSE_UNIT_OPS[op.op]
    return {
        layer: InterventionSpec(
            intervention_tensor=_scatter(by_layer[layer], width, fill, where, layer), op=op.op, position_scope=scope
        )
        for layer in sorted(by_layer)
    }


def _activation_spec(recipe: Any, op: Any, width: int | None, scope: PositionScope, where: str) -> InterventionSpec:
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
    return InterventionSpec(intervention_tensor=tensor, op=op.op, position_scope=scope, **fields)


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
    weights_applied: bool = False,
    flow: str = "value",
) -> dict[str, tuple[InterventionSpec, ...]]:
    """Translate a supported ``venngram.recipe.Recipe`` into a backend intervention payload.

    With ``flow="grad"``, the payload is for ``bwd_w_intervention``: each unit- or feature-targeted op whose target set
    addresses ``of: grad`` acts on the gradient at its site, through the same specs a value op becomes. A recipe's ops
    are translated for one flow at a time, and an op on the other flow is refused (see :func:`op_refusal`).

    Args:
        recipe: The recipe to translate.
        site_widths: The width of every unit-targeted site the recipe names, keyed by the recipe's own layer-free
            site spelling, and of every feature-targeted one, keyed ``<site>.hook_sae_acts_post``. A width given for a
            whole-activation op's site also checks its operand's width.
        evidence: The tables evidence-sourced values read (:mod:`~interpretune.analysis.venngram.values`).
        statistics: The statistics their transforms name.
        weights_applied: Whether the recipe's ``weights`` section (ops on parameters, which a recipe applies before any
            stage) has been applied (:func:`~interpretune.tuning.weights.apply_recipe_weights`). A recipe with weights
            is refused until it has, since its stages alone are a different treatment.
        flow: ``value`` (the default) for ``fwd_w_intervention``, or ``grad`` for ``bwd_w_intervention``.

    Returns:
        A mapping of hook patterns (``blocks.<layer>.<site>``) to the ordered specs to apply at each, in the raw
        form :func:`~interpretune.analysis.backends.interventions.build_intervention_dict` canonicalizes.

    Raises:
        MisconfigurationException: For a recipe whose weights were not applied, anything this module cannot translate
            (see :func:`op_refusal`), an unmapped token scope, a unit- or feature-targeted site whose width was not
            supplied, a unit outside its site's width, an operand whose width differs from a supplied site width, or a
            value the supplied evidence cannot give, naming the stage and op. Run
            :func:`~interpretune.analysis.venngram.support.require_supported` first for the full list at once.
    """
    if flow not in FLOWS:
        raise MisconfigurationException(f"flow {flow!r} is not a flow; a recipe is translated for one of {list(FLOWS)}")
    if getattr(recipe, "weights", ()) and not weights_applied:
        raise MisconfigurationException(
            f"recipe {recipe.name!r} has a `weights` section ({len(recipe.weights)} ops on parameters), applied before "
            "its stages: apply it first (`interpretune.tuning.weights.apply_recipe_weights`), then pass "
            "`weights_applied=True`"
        )
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
            if (refusal := op_refusal(recipe, stage, op, flow)) is not None:
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


def _relative_op_refusal(op: Any, support: FeatureTargetSupport, clean_activations: Any | None) -> str | None:
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
    support: FeatureTargetSupport,
    clean_activations: Any | None = None,
    evidence: Mapping[str, Mapping[str, Mapping[tuple[int, int], float]]] | None = None,
    statistics: Mapping[str, Statistic] | None = None,
    token_ids: Sequence[int] | None = None,
    tokenizer: Mapping[str, str] | None = None,
) -> list[tuple[int, int, int, float]]:
    """Translate a recipe's feature ``replace``, ``add`` and ``scale`` ops, and its graph-targeted ``replace`` and
    ``add`` ops, into circuit-tracer's feature intervention tuples.

    A tuple sets a feature's value. ``replace`` sets the op's value $d$; ``add`` and ``scale`` set $a + d$ and
    $a \\cdot d$, where $a$ is the feature's clean activation at that layer and position, read from
    ``clean_activations``. The clean activation is the one the unintervened prompt produces, so where a recipe also
    intervenes at an earlier layer, an ``add`` is relative to the clean value rather than to the value the earlier
    intervention propagates to (the model-backend surface's ``add`` at the latent sub-hook is relative to the latter).

    A graph-targeted op places each of the graph's feature nodes at the stage's site by its anchor in the prompt
    (venngram's ``GraphSubset.plan``, which refuses the whole plan if any node does not resolve, or two land on one
    coordinate), and its value $d$ is the node's column, less its baseline column when the op names one.

    Args:
        recipe: The recipe to translate; every op must be one :func:`feature_op_refusal` admits.
        support: The ``features`` target's detail in the analysis backend's intervention record. The tuples carry
            computed values, so it must honour ``constant``, and an ``add`` or ``scale`` needs it to declare
            ``captures_clean_activations``.
        clean_activations: The clean feature activations, indexed ``[layer, position, feature]``: the backend's
            ``capture_feature_activations`` over the prompt the tuples will be applied to, with the same
            ``apply_activation_function`` as the intervention call. Needed only for ``add`` and ``scale``.
        evidence: The tables evidence-sourced values read (:mod:`~interpretune.analysis.venngram.values`).
        statistics: The statistics their transforms name.
        token_ids: The prompt's unpadded input token ids, which a graph target's anchors resolve in. Needed only for
            graph targets.
        tokenizer: ``{id, revision}`` of the prompt's tokenizer, which a ``token_match`` anchor needs to be the one the
            graph's source pins.

    Returns:
        ``(layer, position, feature_idx, value)`` tuples, in recipe order, for
        ``replacement_model.feature_intervention``.

    Raises:
        MisconfigurationException: For any op :func:`feature_op_refusal` refuses, a backend not honouring ``constant``,
            an ``add`` or ``scale`` without a clean capture (undeclared, or not passed), a member or position the
            capture has no activation for, a value the supplied evidence cannot give, or a graph target without
            ``token_ids`` or with a node its anchor does not place, naming the stage and op.
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
            if getattr(op, "graph", None) is not None:
                targets = _graph_targets(recipe, stage, op, token_ids, tokenizer, where)
            else:
                values = member_values(recipe, op, evidence=evidence, statistics=statistics, where=where)
                positions = (op.tokens or stage.tokens).positions
                targets = [
                    (layer, position, feature, value)
                    for (layer, feature), value in values.items()
                    for position in positions
                ]
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
                for layer, position, feature, value in targets
            ]
    return interventions
