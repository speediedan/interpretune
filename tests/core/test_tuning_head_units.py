"""Per-head attention units for unit adapters (#868): heads at an attention site project onto the output
projection's blocks and their key/value group's value blocks, and arbitrary coordinates inside an attention
projection stay expressible beside the per-head form."""

import pytest
import torch
from torch import nn

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from venngram.maps.schema import from_document  # noqa: E402
from venngram.maps.verification import check_unit_blocks, with_layouts  # noqa: E402
from venngram.maps.view import MappedView  # noqa: E402
from venngram.resolve import gradient_selections  # noqa: E402

from interpretune.tuning.weights import _space, attach_unit_adapter, weight_edit_roles  # noqa: E402

D_HEAD, N_HEADS, N_KV, LAYERS = 3, 4, 2, 2
D = D_HEAD * N_HEADS
GROUP = N_HEADS // N_KV


class _Attention(nn.Module):
    """Grouped-query attention: 4 query heads of width 3 over 2 key/value heads, so each value head serves 2."""

    head_dim, num_heads, num_key_value_heads = D_HEAD, N_HEADS, N_KV

    def __init__(self) -> None:
        super().__init__()
        self.q, self.o = nn.Linear(D, D), nn.Linear(D, D)
        self.k, self.v = nn.Linear(D, D_HEAD * N_KV), nn.Linear(D, D_HEAD * N_KV)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, p, _ = x.shape
        q = self.q(x).view(b, p, N_HEADS, D_HEAD).transpose(1, 2)
        k, v = (f(x).view(b, p, N_KV, D_HEAD).transpose(1, 2).repeat_interleave(GROUP, 1) for f in (self.k, self.v))
        z = torch.softmax(q @ k.transpose(-1, -2) / D_HEAD**0.5, -1) @ v
        return self.o(z.transpose(1, 2).reshape(b, p, D))


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(LAYERS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


ROWS = {
    "blocks.{i}": {"module": "layers.{i}", "kind": "block"},
    "blocks.{i}.attn": {"module": "layers.{i}.attn", "kind": "attn"},
    **{f"blocks.{{i}}.attn.{c}": {"module": f"layers.{{i}}.attn.{c}", "kind": "linear"} for c in "qkvo"},
}
DRAFT = {"schema_version": 3, "architecture": "TinyAttention", "components": ROWS}
W_O, W_V, V_BIAS = "layers.1.attn.o.weight", "layers.1.attn.v.weight", "layers.1.attn.v.bias"
SITE = "point:attn.hook_z & layers:1"


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyAttention()


@pytest.fixture
def measured(model):
    """The map as venngram's builder records it: layouts and head block sizes measured on the model."""
    return with_layouts(DRAFT, model)


def _view(model, doc):
    return MappedView(model, from_document(doc))


def _without(doc, component, key):
    rows = {
        k: ({f: v for f, v in r.items() if f != key} if k.endswith(component) else r)
        for k, r in doc["components"].items()
    }
    return {**doc, "components": rows}


def _moved(model, handles):
    """Each tensor's changed coordinates along dim 0 (rows) or 1 (columns), after setting every delta to one."""
    before = {h.target: getattr(h.owner, h.leaf).detach().clone() for h in handles}
    with torch.no_grad():
        for h in handles:
            for p in h.edit.parameters():
                p.fill_(1.0)
    out = {}
    for h in handles:
        diff = (getattr(h.owner, h.leaf) - before[h.target]).ne(0)
        moved = diff.any(0) if h.target == W_O else diff if diff.ndim == 1 else diff.any(1)
        out[h.target] = moved.nonzero().flatten()
    return {t: idx.tolist() for t, idx in out.items()}


def test_the_measured_map_records_each_projections_head_block_and_passes_verification(model, measured):
    rows = measured["components"]
    assert measured["schema_version"] == 6
    assert {c: rows[f"blocks.{{i}}.attn.{c}"]["unit_block"] for c in "qkvo"} == dict.fromkeys("qkvo", D_HEAD)
    assert check_unit_blocks(from_document(measured), model).status == "passed"


def test_whole_key_value_groups_tune_their_output_and_value_blocks(model, measured):
    x = torch.randn(1, 5, D)
    reference = model(x)
    handles = attach_unit_adapter(model, SITE, [2, 3], name="heads", view=_view(model, measured))
    assert torch.equal(model(x), reference)  # the identity at attach
    # heads 2 and 3 are columns 6 to 11 of W_O, and they read value head 1: rows (and bias entries) 3 to 5 of W_V
    assert _moved(model, handles) == {W_O: list(range(6, 12)), W_V: [3, 4, 5], V_BIAS: [3, 4, 5]}
    assert set(weight_edit_roles(model).values()) == {"adapter:heads"}


def test_a_partial_key_value_group_is_refused_on_the_value_side_and_tunes_w_o_alone_on_request(model, measured):
    view = _view(model, measured)
    with pytest.raises(ValueError, match=r"value head 1 .* serves query heads \[2, 3\].*lacks \[3\].*value_side=False"):
        attach_unit_adapter(model, SITE, [2], name="heads", view=view)
    assert not weight_edit_roles(model)  # nothing attached
    handles = attach_unit_adapter(model, SITE, {"heads": [2]}, name="heads", view=view, value_side=False)
    assert _moved(model, handles) == {W_O: [6, 7, 8]}


def test_heads_need_the_maps_recorded_block_size(model, measured):
    with pytest.raises(ValueError, match="no recorded head block size.*unit_block.*schema 6"):
        attach_unit_adapter(
            model, SITE, [2, 3], name="heads", view=_view(model, _without(measured, "attn.o", "unit_block"))
        )
    no_value_block = _view(model, _without(measured, "attn.v", "unit_block"))
    with pytest.raises(ValueError, match="value side needs .* recorded head block size.*value_side=False"):
        attach_unit_adapter(model, SITE, [2, 3], name="heads", view=no_value_block)
    assert len(attach_unit_adapter(model, SITE, [2, 3], name="heads", view=no_value_block, value_side=False)) == 1


def test_an_arbitrary_coordinate_slice_of_a_projection_tunes_beside_the_per_head_form(model, measured):
    """Units outside the head convenience stay expressible: a slice crossing heads 1 and 2 of W_O, addressed by its
    tensor, tunes exactly those columns, and the same tensor's head 1 by `{heads: [1]}` tunes its block."""
    view = _view(model, measured)
    o = "component:attn.o & layers:1 & leaf:weight"
    (crossing,) = attach_unit_adapter(model, o, [4, 5, 6, 7], name="slice", view=view)
    assert _moved(model, [crossing]) == {W_O: [4, 5, 6, 7]}
    crossing.remove()
    (head,) = attach_unit_adapter(model, o, {"heads": [1]}, name="head", view=view)
    assert _moved(model, [head]) == {W_O: [3, 4, 5]}
    head.remove()
    # both are what venngram's own resolver gives the same units as an architecture-independent address, along W_O's
    # unit axis (its columns)
    space, shapes = _space(model, view), {n: tuple(p.shape) for n, p in model.named_parameters()}
    for units, expected in (("[4, 5, 6, 7] & axis:unit", (4, 5, 6, 7)), ("{heads: [1]}", (3, 4, 5))):
        (selection,) = gradient_selections(f"{o} & units:{units} & of:grad", space, shapes=shapes).values()
        assert (selection.axis, selection.indices) == (1, expected)


def test_heads_are_refused_where_the_units_are_neurons_and_value_side_off_an_attention_site(model, measured):
    view = _view(model, measured)
    with pytest.raises(ValueError, match="value_side.*attention site"):
        attach_unit_adapter(
            model, "component:attn.o & layers:1 & leaf:weight", [2], name="a", view=view, value_side=False
        )
    with pytest.raises(ValueError, match="outside the target's 4 heads"):
        attach_unit_adapter(model, SITE, [4], name="a", view=view, value_side=False)
    with pytest.raises(ValueError, match="selects no units"):
        attach_unit_adapter(model, SITE, {"heads": []}, name="a", view=view)


def test_the_transformer_lens_views_structure_reports_the_wrapped_models_head_blocks(model, measured):
    """A bridge's tensors are located through the wrapped model's map, so their head block sizes come from it
    too."""
    from interpretune.adapters.transformer_lens.tunable_view import _WrappedModelStructure

    inner = _view(model, measured).structure()
    structure = _WrappedModelStructure(inner, {"bridge.o.weight": W_O})
    assert structure.unit_block("bridge.o.weight") == D_HEAD
    assert structure.unit_block("bridge.unknown.weight") is None
    # the same tensor as a view the bridge makes has neither a unit axis nor a head block size of its own
    as_view = _WrappedModelStructure(inner, {"bridge.o.weight": W_O}, views={"bridge.o.weight": W_O})
    assert as_view.unit_block("bridge.o.weight") is None
