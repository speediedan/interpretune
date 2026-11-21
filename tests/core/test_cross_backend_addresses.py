"""One address selects the same tensors of GPT-2 on its core/HF, nnsight and TransformerLens compositions
(#875)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytest.importorskip("venngram.tuning", reason="venngram, which holds the tunable view protocol, is not installed")

from interpretune.testing.conformance.tuning import (  # noqa: E402
    assert_same_selection,
    selection_by_wrapped_name,
    wrapped_model_names,
)
from interpretune.tuning.latents import (  # noqa: E402
    LatentSite,
    attach_latent_model,
    detach_latent_model,
    latent_model_placements,
    latent_model_role_addresses,
)
from interpretune.tuning.views import VIEW_ATTR, default_view  # noqa: E402

# structure (a layer, a range from the top, a parity, a fraction of the depth) by component, on GPT-2's twelve blocks
ADDRESSES = (
    "layers:9 & component:attn.*",
    "component:mlp.* & layers:-2:",
    "layers:0 & component:attn.o",
    "layers:odd & component:mlp.out",
    "layers:0.75d: & component:attn.*",
)

# a latent model at block 3's residual output and a transcoder at block 1's MLP, by site address on every backend
SPLICES = {
    "sae": LatentSite("point:hook_resid_post & layers:3"),
    "tc": LatentSite("point:mlp.hook_in & layers:1", write="point:mlp.hook_out & layers:1"),
}
LATENT_ADDRESSES = ("role:latent & layers:3:", "role:latent & component:mlp", "role:latent:tc")


class Dictionary(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.enc, self.dec = nn.Linear(width, 2 * width), nn.Linear(2 * width, width)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)


class Holder(nn.Module):
    """A core/HF composition's shape: the model under ``model``, and modules spliced in beside it."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


@pytest.fixture(scope="module")
def bridge():
    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers("gpt2", device="cpu")


@pytest.fixture(scope="module")
def hf_holder(bridge):
    # the bridge rewires its wrapped model in place, so the core/HF composition gets a fresh model of the same config
    wrapped = bridge.original_model
    with torch.device("meta"):
        return Holder(type(wrapped)(wrapped.config))


@pytest.fixture
def nnsight_module(request):
    return request.getfixturevalue("get_it_session__ns_gpt2__setup").it_session.module


def _views(hf_holder, nnsight_module, bridge, spliced: bool = False) -> dict:
    """Each backend's view of GPT-2, built as its composition builds it, with the spliced modules placed."""
    from interpretune.adapters.transformer_lens.tunable_view import TransformerBridgeView

    tl_holder = Holder(bridge)
    holders = (hf_holder, tl_holder, nnsight_module)
    if spliced:
        width = bridge.cfg.d_model
        for holder in holders:
            for name, site in SPLICES.items():
                attach_latent_model(holder, name, Dictionary(width), site)

    def args(holder) -> dict:
        return {"roles": latent_model_role_addresses(holder), "attached": latent_model_placements(holder)}

    return {
        "core/HF": default_view(hf_holder, **args(hf_holder)),
        "nnsight": nnsight_module.build_tunable_view(),
        "TransformerLens": TransformerBridgeView(tl_holder, implicit_norm_thaw=False, **args(tl_holder)),
    }


def _detach(*holders) -> None:
    for holder in holders:
        for name in SPLICES:
            if name in getattr(holder, "latent_models", {}):
                detach_latent_model(holder, name)


class TestCrossBackendAddresses:
    def test_the_wrapped_models_names_agree(self, hf_holder, nnsight_module, bridge):
        """The comparison's premise: every backend names GPT-2's tensors as GPT-2 does."""
        views = _views(hf_holder, nnsight_module, bridge)
        named = {b: set(wrapped_model_names(v).values()) for b, v in views.items()}
        assert "transformer.h.0.attn.c_attn.weight" in named["core/HF"]
        assert named["core/HF"] <= named["nnsight"] and named["core/HF"] <= named["TransformerLens"]

    @pytest.mark.parametrize("address", ADDRESSES)
    def test_a_structural_address_selects_the_same_tensors(self, hf_holder, nnsight_module, bridge, address):
        assert_same_selection(_views(hf_holder, nnsight_module, bridge), [address])

    def test_spliced_latent_models_are_selected_alike(self, hf_holder, nnsight_module, bridge):
        """Latent models spliced by site address take the same layer, component and role on every backend."""
        views = _views(hf_holder, nnsight_module, bridge, spliced=True)
        try:
            assert_same_selection(views, LATENT_ADDRESSES)
            assert selection_by_wrapped_name(views["nnsight"], "role:latent & layers:3:") == {
                f"latent_models.sae.{n}" for n in ("enc.weight", "enc.bias", "dec.weight", "dec.bias")
            }
        finally:
            _detach(*(view.root for view in views.values()))

    def test_a_selection_that_differs_or_is_empty_is_named(self, bridge):
        width, holders = bridge.cfg.d_model, []
        for layer in (4, 5):  # the same latent model, spliced one block apart
            with torch.device("meta"):
                holder = Holder(type(bridge.original_model)(bridge.original_model.config))
            attach_latent_model(holder, "sae", Dictionary(width), LatentSite(f"point:hook_resid_post & layers:{layer}"))
            holders.append(holder)
        views = {
            f"at {k}": default_view(h, roles=latent_model_role_addresses(h), attached=latent_model_placements(h))
            for k, h in zip((4, 5), holders)
        }
        try:
            with pytest.raises(AssertionError, match="does not resolve on at 5: .*matches no tensor"):
                assert_same_selection(views, ["role:latent & layers:4"])
            with pytest.raises(AssertionError, match="different tensors across backends"):
                assert_same_selection(views, ["layers:4"])  # block 4's tensors, and the latent model only where spliced
        finally:
            _detach(*holders)


def test_the_nnsight_view_is_the_attached_one(nnsight_module):
    """The nnsight view compared above is the one its session attaches."""
    assert type(nnsight_module.build_tunable_view()) is type(getattr(nnsight_module, VIEW_ATTR))
