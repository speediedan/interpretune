"""Weight-space interventions (#961): venngram ops on parameters, materialized reversibly, checked against
venngram's parameter op vectors and against their activation counterparts."""

import pytest
import torch
from torch import nn

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from venngram.maps.schema import from_document  # noqa: E402
from venngram.maps.view import MappedView  # noqa: E402
from venngram.ops import check_ops  # noqa: E402
from venngram.patches import tensor_sha256, write_weight_patch  # noqa: E402

from interpretune.analysis.backends.capabilities import PositionScope  # noqa: E402
from interpretune.analysis.backends.interventions import InterventionSpec, apply_op  # noqa: E402
from interpretune.analysis.venngram import load_recipe, recipe_interventions  # noqa: E402
from interpretune.tuning.weights import (  # noqa: E402
    _Delta,
    _Span,
    _Units,
    _Vector,
    apply_recipe_weights,
    apply_weight_patch,
    bias_patch,
    check_weight_equivalence,
    resolve_weight_targets,
    weight_edit_roles,
)
from interpretune.utils import MisconfigurationException  # noqa: E402

D, H, LAYERS, VOCAB = 6, 10, 2, 11


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up, self.down = nn.Linear(D, H), nn.Linear(H, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.tanh(self.up(x)))


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _Mlp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList(_Layer() for _ in range(LAYERS))
        self.head = nn.Linear(D, VOCAB, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# venngram's schema-5 golden map shape: each linear row records its measured layout (nn.Linear stores [out, in])
MAP = {
    "schema_version": 5,
    "architecture": "TinyDecoder",
    "components": {
        "embed": {"module": "embed", "kind": "embed", "layout": "in_out"},
        "blocks.{i}": {"module": "layers.{i}", "kind": "block"},
        "blocks.{i}.mlp": {"module": "layers.{i}.mlp", "kind": "mlp"},
        "blocks.{i}.mlp.in": {"module": "layers.{i}.mlp.up", "kind": "linear", "layout": "out_in"},
        "blocks.{i}.mlp.out": {"module": "layers.{i}.mlp.down", "kind": "linear", "layout": "out_in"},
    },
}
UNMAPPED_LAYOUT = {
    **MAP,
    "schema_version": 4,
    "components": {k: {f: v for f, v in row.items() if f != "layout"} for k, row in MAP["components"].items()},
}


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyDecoder()


def _view(model, doc=MAP):
    return MappedView(model, from_document(doc))


def _recipe(weights, operands=None, sets=None):
    sets = sets or {
        "down": {"component": "mlp.out", "layers": [1], "leaf": "weight"},
        "down_bias": {"component": "mlp.out", "layers": [1], "leaf": "bias"},
        "neurons": {"component": "mlp.out", "layers": [1], "leaf": "weight", "units": [2, 7]},
    }
    doc = {
        "schema_version": 3,
        "kind": "recipe",
        "name": "edit",
        "model": {"id": "tiny", "revision": "r1"},
        "book": {
            "schema_version": 2,
            "name": "tiny",
            "portability": "checkpoint",
            "applies_to": {"models": [{"id": "tiny", "revision": "r1"}]},
            "sets": sets,
        },
        "weights": weights,
    }
    if operands:
        doc["operands"] = operands
    return load_recipe(doc).recipe


IDS = torch.tensor([[1, 4, 2, 9]])


# venngram's parameter op vectors, each through the edit interpretune materializes, and every stated activation
# counterpart through interpretune's own activation op math
def _evaluate(case):
    i = case.inputs
    w = torch.tensor(i["w"]).reshape([int(d) for d in i["shape"]])
    if case.op == "scale":
        dim, units = int(i["dim"][0]), [int(u) for u in i["units"]]
        edited = _Units("v", "instrument", "scale", dim, units, [i["dose"][0]] * len(units))(w)
        out = {"w": edited.flatten().tolist()}
        if "x" in i:
            x = torch.tensor(i["x"])
            multiplier = torch.ones_like(x).index_fill(0, torch.tensor(units), i["dose"][0])
            spec = InterventionSpec(multiplier, op="scale", position_scope=PositionScope.ALL_POSITIONS)
            scaled = apply_op(x.reshape(1, 1, -1), spec).reshape(-1)
            out.update(y_weight=(edited @ x).tolist(), y_activation=(w @ scaled).tolist())
        return out
    if case.op == "add" and "b" in i:
        b, x, j, d = torch.tensor(i["b"]), torch.tensor(i["x"]), int(i["units"][0]), i["dose"][0]
        patched = _Vector("v", "instrument", "add", bias_patch(w, {j: d}, dim=1))(b)
        dose = torch.zeros_like(x).index_fill(0, torch.tensor([j]), d)
        shifted = apply_op(
            x.reshape(1, 1, -1), InterventionSpec(dose, op="add", position_scope=PositionScope.ALL_POSITIONS)
        )
        return {
            "b": patched.tolist(),
            "y_weight": (w @ x + patched).tolist(),
            "y_activation": (w @ shifted.reshape(-1) + b).tolist(),
        }
    if case.op == "add":
        u, v = torch.tensor(i["u"]).reshape(-1, 1), torch.tensor(i["v"]).reshape(-1, 1)
        return {"w": _Delta("v", "instrument", "add", "low_rank", {"u": u, "v": v})(w).flatten().tolist()}
    if case.op == "reject":
        edited = _Span("v", "instrument", "reject", torch.tensor(i["u"]), i["fraction"][0], dim=0)(w)
        return {"w": edited.flatten().tolist()}
    sparse = {"indices": torch.tensor([int(k) for k in i["indices"]]), "values": torch.tensor(i["values"])}
    return {"w": _Delta("v", "instrument", "replace", "sparse", sparse)(w).flatten().tolist()}


@pytest.mark.parametrize("target", ["param", "param_units"])
def test_the_edits_meet_venngrams_parameter_op_vectors_and_their_activation_counterparts(target):
    assert check_ops(_evaluate, target=target, rel_tol=1e-6, abs_tol=1e-5) == []


def test_an_address_resolves_through_the_view_to_the_tensor_the_model_holds(model):
    view = _view(model)
    assert resolve_weight_targets(model, "component:mlp.out & layers:1 & leaf:weight", view=view) == [
        "layers.1.mlp.down.weight"
    ]
    assert resolve_weight_targets(model, "layers.0.mlp.up.bias") == ["layers.0.mlp.up.bias"]
    with pytest.raises(ValueError, match="does not resolve"):
        resolve_weight_targets(model, "component:mlp.out & layers:7 & leaf:weight", view=view)


def _patch(tmp_path, model, *, base=None, form="low_rank"):
    w = model.layers[1].mlp.down.weight
    rows, cols = w.shape
    tensors = (
        {"u": [float(r + 1) for r in range(rows)], "v": [0.0] * (cols - 1) + [0.5]}
        if form == "low_rank"
        else {"indices": [0, 7], "values": [3.0, -2.0]}
    )
    return write_weight_patch(
        tmp_path / "edit.yaml",
        name="neuron write",
        target="component:mlp.out & layers:1 & leaf:weight",
        shape=list(w.shape),
        form=form,
        tensors=tensors,
        base_sha256=base or tensor_sha256(w),
    )


def test_a_weight_patch_applies_reversibly_after_its_base_is_checked(tmp_path, model):
    base = model.layers[1].mlp.down.weight.detach().clone()
    patch = _patch(tmp_path, model)
    handle = apply_weight_patch(model, patch, view=_view(model))
    down = model.layers[1].mlp.down
    delta = torch.arange(1.0, D + 1).reshape(-1, 1) @ torch.tensor([[0.0] * (H - 1) + [0.5]])
    torch.testing.assert_close(down.weight, base + delta)
    assert handle.target == "layers.1.mlp.down.weight" and handle.edit.edit_name == "neuron_write"
    # the operand is an instrument (loaded from an artifact): never trained
    assert weight_edit_roles(model) == {
        "layers.1.mlp.down.parametrizations.weight.0.u": "instrument",
        "layers.1.mlp.down.parametrizations.weight.0.v": "instrument",
    }
    assert not any(p.requires_grad for p in handle.edit.parameters())
    handle.remove()
    assert not hasattr(down, "parametrizations") and torch.equal(down.weight, base)
    assert weight_edit_roles(model) == {}
    with pytest.raises(ValueError, match="no longer applied"):
        handle.remove()


def test_a_patch_made_for_another_base_is_refused(tmp_path, model):
    patch = _patch(tmp_path, model, base="0" * 64)
    with pytest.raises(ValueError, match="made for another base"):
        apply_weight_patch(model, patch, view=_view(model))
    assert not hasattr(model.layers[1].mlp.down, "parametrizations")


def test_a_trainable_patch_is_an_adapter_and_a_permanent_one_is_written_into_the_tensor(tmp_path, model):
    patch = _patch(tmp_path, model, form="sparse")
    handle = apply_weight_patch(model, patch, name="steer", role="adapter", view=_view(model))
    assert set(weight_edit_roles(model).values()) == {"adapter:steer"}
    assert handle.edit.values.requires_grad
    # a view built after the edit schedules its operand by role and still places the base tensor
    view = MappedView(model, from_document(MAP), roles=weight_edit_roles(model))
    assert view.roles()["layers.1.mlp.down.parametrizations.weight.0.values"] == "adapter:steer"
    assert view.structure().locate("layers.1.mlp.down.parametrizations.weight.original").component == "mlp.out"
    model(IDS).sum().backward()
    assert (
        handle.edit.values.grad is not None
        and model.layers[1].mlp.down.parametrizations.weight.original.grad is not None
    )
    handle.remove()
    with pytest.raises(ValueError, match="is reversible"):
        apply_weight_patch(model, patch, role="adapter", reversible=False, view=_view(model))
    base = model.layers[1].mlp.down.weight.detach().clone()
    kept = apply_weight_patch(model, patch, reversible=False, view=_view(model))
    assert not hasattr(model.layers[1].mlp.down, "parametrizations")
    assert model.layers[1].mlp.down.weight.flatten()[7] == base.flatten()[7] - 2.0
    with pytest.raises(ValueError, match="cannot be taken back"):
        kept.remove()


def test_a_recipes_weights_scale_units_along_the_unit_axis_and_reject_on_the_output_side(model):
    down = model.layers[1].mlp.down
    base = down.weight.detach().clone()
    u = torch.tensor([1.0, 0, 0, 0, 0, 0])
    recipe = _recipe(
        [
            {"op": "scale", "target": {"set": "neurons"}, "value": {"dose": 0.0}},
            {"op": "reject", "target": {"set": "down"}, "operand": "concept", "params": {"fraction": 1.0}},
        ],
        operands={"concept": {"values": u.tolist(), "shape": [D], "basis": "model"}},
    )
    handles = apply_recipe_weights(model, recipe, view=_view(model))
    assert [h.target for h in handles] == ["layers.1.mlp.down.weight"] * 2
    want = base.clone()
    want[:, [2, 7]] = 0.0  # mlp.out's units are its input side: the columns of an [out, in] weight
    want[0] = 0.0  # rejecting e_0 from the output removes row 0
    torch.testing.assert_close(down.weight, want)
    for handle in reversed(handles):
        handle.remove()
    assert torch.equal(down.weight, base)


def test_units_of_a_tensor_need_a_recorded_layout_and_a_failed_recipe_is_taken_back(model):
    recipe = _recipe(
        [
            {"op": "scale", "target": {"set": "down"}, "value": {"dose": 2.0}},
            {"op": "scale", "target": {"set": "neurons"}, "value": {"dose": 0.0}},
        ]
    )
    base = model.layers[1].mlp.down.weight.detach().clone()
    with pytest.raises(ValueError, match="records no `layout`"):
        apply_recipe_weights(model, recipe, view=_view(model, UNMAPPED_LAYOUT))
    assert not hasattr(model.layers[1].mlp.down, "parametrizations")
    assert torch.equal(model.layers[1].mlp.down.weight, base)


def _scale_inputs(module, units, factor):
    def pre(_, args):
        (x,) = args
        return (x * torch.ones(x.shape[-1]).index_fill(0, torch.tensor(units), factor),)

    return module.register_forward_pre_hook(pre)


def _add_to_inputs(module, units, dose):
    def pre(_, args):
        (x,) = args
        return (x + torch.zeros(x.shape[-1]).index_fill(0, torch.tensor(units), dose),)

    return module.register_forward_pre_hook(pre)


def test_a_unit_scale_and_a_bias_patch_equal_their_activation_ops_on_the_model(model):
    view, down = _view(model), model.layers[1].mlp.down

    def forward():
        return model(IDS)

    scaled = check_weight_equivalence(
        forward,
        lambda: apply_recipe_weights(
            model, _recipe([{"op": "scale", "target": {"set": "neurons"}, "value": {"dose": 3.0}}]), view=view
        ),
        lambda: _scale_inputs(down, [2, 7], 3.0),
    )
    assert scaled.equivalent, scaled
    dose = {4: 1.5}
    patched = check_weight_equivalence(
        forward,
        lambda: apply_recipe_weights(
            model,
            _recipe(
                [{"op": "add", "target": {"set": "down_bias"}, "operand": "patch"}],
                operands={
                    "patch": {"values": bias_patch(down.weight.detach(), dose).tolist(), "shape": [D], "basis": "model"}
                },
            ),
            view=view,
        ),
        lambda: _add_to_inputs(down, list(dose), 1.5),
    )
    assert patched.equivalent, patched
    # a change with no counterpart is told apart
    unequal = check_weight_equivalence(
        forward,
        lambda: apply_recipe_weights(
            model, _recipe([{"op": "scale", "target": {"set": "neurons"}, "value": {"dose": 3.0}}]), view=view
        ),
        lambda: _scale_inputs(down, [2], 3.0),
    )
    assert not unequal.equivalent and not hasattr(down, "parametrizations")


def test_a_recipe_with_weights_is_refused_by_the_activation_path_until_they_are_applied(model):
    recipe = _recipe([{"op": "scale", "target": {"set": "neurons"}, "value": {"dose": 0.0}}])
    with pytest.raises(MisconfigurationException, match="has a `weights` section"):
        recipe_interventions(recipe, site_widths={})
    assert recipe_interventions(recipe, site_widths={}, weights_applied=True) == {}


def test_parameter_targets_are_declared_only_where_weights_are_materialized():
    from venngram.ops import check_support

    from interpretune.analysis.backends.capabilities import InterventionSupport
    from interpretune.analysis.venngram import venngram_support

    record = InterventionSupport(position_scopes={PositionScope.ALL_POSITIONS}, ops={"add", "scale", "reject"})
    recipe = _recipe([{"op": "scale", "target": {"set": "neurons"}, "value": {"dose": 0.0}}])
    assert any("param_units" in r for r in check_support(recipe, venngram_support(record)))
    supported = venngram_support(record, weights=True)
    assert {"param", "param_units"} <= supported.targets and check_support(recipe, supported) == []


def _stage_recipe(ops, tokens="all", site="mlp.out.hook_in"):
    return load_recipe(
        {
            "schema_version": 2,
            "kind": "recipe",
            "name": "neurons",
            "model": {"id": "tiny", "revision": "r1"},
            "book": {
                "schema_version": 1,
                "name": "tiny",
                "portability": "checkpoint",
                "applies_to": {"models": [{"id": "tiny", "revision": "r1"}]},
                "sets": {"units": {"point": "mlp.out.hook_in", "layers": [0, 1], "units": {0: [3], 1: [2, 7]}}},
            },
            "stages": [{"site": site, "tokens": tokens, "ops": ops}],
        }
    ).recipe


def _hooked(model, recipe):
    """The recipe through interpretune's activation path: its payload, applied at each layer's ``down`` input."""
    from interpretune.analysis.backends.interventions import apply_intervention

    payload = recipe_interventions(recipe, site_widths={"mlp.out.hook_in": H})
    hooks = []
    for pattern, specs in payload.items():
        layer = int(pattern.split(".")[1])

        def pre(_, args, specs=specs):
            (x,) = args
            for spec in specs:
                x = apply_intervention(x.clone(), spec, last_pos=x.shape[1] - 1)
            return (x,)

        hooks.append(model.layers[layer].mlp.down.register_forward_pre_hook(pre))

    class _Hooks:
        def remove(self):
            for h in hooks:
                h.remove()

    return _Hooks()


SCALE = {"op": "scale", "target": {"set": "units"}, "value": {"dose": 0.0}}
SHIFT = {"op": "add", "target": {"set": "units"}, "value": {"doses": {0: {3: 0.75}, 1: {2: -1.0, 7: 2.0}}}}
TRIPLE = {"op": "scale", "target": {"set": "units"}, "value": {"dose": 3.0}}


# an ablation then a dose (the dose survives), and a dose then a scale (the later scale scales the dose too)
@pytest.mark.parametrize("ops", [[SCALE, SHIFT], [SHIFT, TRIPLE]], ids=["ablate-then-dose", "dose-then-scale"])
def test_a_per_neuron_recipe_at_scope_all_has_a_weight_form_equal_to_its_hooked_form(model, ops):
    from interpretune.tuning.weights import weight_form

    view = _view(model)
    recipe = _stage_recipe(ops)
    handles = weight_form(model, recipe, view=view)
    assert [h.target for h in handles] == [
        "layers.0.mlp.down.weight",
        "layers.0.mlp.down.bias",
        "layers.1.mlp.down.weight",
        "layers.1.mlp.down.bias",
    ]
    for handle in reversed(handles):
        handle.remove()
    result = check_weight_equivalence(
        lambda: model(IDS), lambda: weight_form(model, recipe, view=view), lambda: _hooked(model, recipe)
    )
    assert result.equivalent, result
    for down in (layer.mlp.down for layer in model.layers):
        assert not hasattr(down, "parametrizations")


@pytest.mark.parametrize(
    "tokens, site, match",
    [
        ("last", "mlp.out.hook_in", "tokens `last`"),
        ("all", "resid_post", "only unit ops at `mlp.out.hook_in`"),
    ],
)
def test_a_token_scoped_or_other_site_recipe_has_no_weight_form(model, tokens, site, match):
    from interpretune.tuning.weights import weight_form

    ops = [{"op": "scale", "target": {"set": "units"}, "value": {"dose": 0.0}}]
    with pytest.raises(ValueError, match=match):
        weight_form(model, _stage_recipe(ops, tokens=tokens, site=site), view=_view(model))
