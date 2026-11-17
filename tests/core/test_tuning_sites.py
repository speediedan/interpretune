"""Site addresses for the tuning seam (#864): an intervention or latent model names its site once, as a venngram
site address, and each composition resolves it to its own module."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from interpretune.tuning.interventions import OpIntervention, attach_intervention, intervention_sites
from interpretune.tuning.latents import LatentSite, attach_latent_model, latent_model_sites
from interpretune.tuning.sites import is_site_address, resolve_site

D = 8
IDS = torch.tensor([[1, 2, 3, 4, 5]])


def _gpt2() -> nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(n_layer=2, n_embd=D, n_head=2, vocab_size=16, n_positions=8, architectures=["GPT2LMHeadModel"])
    return GPT2LMHeadModel(config).eval()


class Core(nn.Module):
    """A core composition's layout: the HF model under ``model``."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _gpt2()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model(ids).logits


class NNsightLayout(nn.Module):
    """An nnsight composition's layout: the HF module registered as ``hf_model``."""

    def __init__(self) -> None:
        super().__init__()
        self.hf_model = _gpt2()


class TinySAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.enc, self.dec = nn.Linear(D, 2 * D), nn.Linear(2 * D, D)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.enc(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)


def _steered(root: Core, site: str, value: float = 0.5) -> torch.Tensor:
    steer = attach_intervention(root, "steer", OpIntervention("add", D), site)
    with torch.no_grad():
        steer.operand.value.fill_(value)
        return root(IDS)


class TestResolution:
    def test_module_paths_are_not_addresses(self):
        assert not is_site_address("model.transformer.h.0")
        assert is_site_address("point:hook_resid_post & layers:0")

    @pytest.mark.parametrize(
        ("address", "expected"),
        [
            ("point:hook_resid_post & layers:1", ("transformer.h.1", "output")),
            ("point:hook_resid_pre & layers:1", ("transformer.h.1", "input")),
        ],
    )
    def test_an_address_resolves_under_the_hf_model_path(self, address, expected):
        path, io = expected
        assert (resolve_site(Core(), address).path, resolve_site(Core(), address).io) == (f"model.{path}", io)
        assert resolve_site(NNsightLayout(), address).path == f"hf_model.{path}"
        assert resolve_site(Core(), address).hook == f"blocks.1.{address.split()[0].removeprefix('point:')}"

    @pytest.mark.parametrize(
        ("address", "match"),
        [
            ("point:hook_resid_post & layers:0:2", "selects 2 sites"),
            ("point:hook_resid_post & layers:0 & of:grad", "names a gradient"),
            ("point:hook_no_such_point & layers:0", "Site address 'point:hook_no_such_point"),
            ("layers:0", "Site address 'layers:0'"),
        ],
    )
    def test_an_address_that_is_not_one_value_site_is_refused(self, address, match):
        with pytest.raises(ValueError, match=match):
            resolve_site(Core(), address)

    def test_a_root_without_a_model_is_refused(self):
        root = nn.Module()
        root.add_module("proj", nn.Linear(D, D))
        with pytest.raises(ValueError, match="needs the model's layer count"):
            resolve_site(root, "point:hook_resid_post & layers:0")


class TestInterventions:
    def test_an_address_steers_exactly_as_its_module_path_does(self):
        base = Core()(IDS)
        by_address, by_path = Core(), Core()
        steered = _steered(by_address, "point:hook_resid_post & layers:0")
        assert torch.equal(steered, _steered(by_path, "model.transformer.h.0"))
        assert not torch.equal(steered, base)
        assert intervention_sites(by_address) == {"steer": "point:hook_resid_post & layers:0"}

    def test_an_input_site_is_rewritten_before_its_module_runs(self):
        """GPT-2's residual input to block 1 is block 0's output, so steering either is one edit."""
        pre, post = Core(), Core()
        assert torch.equal(
            _steered(pre, "point:hook_resid_pre & layers:1"), _steered(post, "point:hook_resid_post & layers:0")
        )

    def test_a_refused_address_attaches_nothing(self):
        root = Core()
        with pytest.raises(ValueError, match="selects 2 sites"):
            attach_intervention(root, "steer", OpIntervention("add", D), "point:hook_resid_post & layers:0:2")
        assert intervention_sites(root) == {}


class TestLatentModels:
    def test_a_latent_model_spliced_by_address_matches_its_module_path(self):
        out, roots = [], []
        for site in (LatentSite("point:hook_resid_post & layers:0"), LatentSite("model.transformer.h.0")):
            roots.append(root := Core())
            attach_latent_model(root, "sae", TinySAE(), site)
            with torch.no_grad():
                out.append(root(IDS))
        assert torch.equal(*out)
        assert latent_model_sites(roots[0]) == {"sae": LatentSite("point:hook_resid_post & layers:0")}

    def test_an_input_site_reads_and_writes_the_input(self):
        """At block 1's residual input the SAE reconstructs block 0's output, as it does spliced at that output."""
        out = []
        for address in ("point:hook_resid_pre & layers:1", "point:hook_resid_post & layers:0"):
            root = Core()
            attach_latent_model(root, "sae", TinySAE(), LatentSite(address, op="replace"))
            with torch.no_grad():
                out.append(root(IDS))
        assert torch.allclose(*out)

    def test_a_transcoder_placed_by_address_matches_its_module_paths(self):
        """``mlp.hook_in`` is the MLP's argument (the block norm's output) and ``mlp.hook_out`` the MLP's output,
        so the addresses place a transcoder exactly where the MLP's input and output paths do."""
        sites = (
            LatentSite("point:mlp.hook_in & layers:1", write="point:mlp.hook_out & layers:1", op="replace"),
            LatentSite("model.transformer.h.1.mlp", read_from="input", op="replace"),
        )
        out = []
        for site in sites:
            root = Core()
            attach_latent_model(root, "tc", TinySAE(), site)
            with torch.no_grad():
                out.append(root(IDS))
        assert torch.equal(*out)

    def test_read_from_input_against_an_output_address_is_refused(self):
        with pytest.raises(ValueError, match="an address decides the side"):
            attach_latent_model(
                Core(), "sae", TinySAE(), LatentSite("point:hook_resid_post & layers:0", read_from="input")
            )
