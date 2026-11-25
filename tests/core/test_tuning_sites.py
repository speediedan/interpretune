"""Site addresses for the tuning seam (#864): an intervention or latent model names its site once, as a venngram
site address, and each composition resolves it to its own module."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from interpretune.tuning.interventions import (
    OpIntervention,
    attach_intervention,
    detach_intervention,
    intervention_placements,
    intervention_roles,
    intervention_sites,
)
from interpretune.tuning.latents import (
    LatentSite,
    attach_latent_model,
    detach_latent_model,
    latent_model_placements,
    latent_model_roles,
    latent_model_sites,
)
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


def _view(root: nn.Module):
    """``root``'s default view with the roles and placements of what is spliced into it, as ``BaseITModule``'s."""
    from interpretune.tuning.views import default_view

    roles = {**latent_model_roles(root), **intervention_roles(root)}
    return default_view(root, roles=roles, attached={**latent_model_placements(root), **intervention_placements(root)})


def _selected(view, address: str) -> set[str]:
    """The parameters ``address`` selects on ``view``, as finetuning-scheduler resolves a schedule's addresses."""
    from venngram.errors import VenngramError
    from venngram.resolve import ParamSpace, select

    names = [n for n, _ in view.root.named_parameters()]
    try:
        return set(select(address, ParamSpace(names, roles=dict(view.roles()), structure=view.structure())))
    except VenngramError:  # selects nothing
        return set()


class TestPlacement:
    """A module spliced in beside the model locates at its site (#875), so an address selects it by depth."""

    @pytest.mark.parametrize("layout, mount", [(Core, "model"), (NNsightLayout, "hf_model")])
    def test_a_latent_model_takes_the_layer_and_component_of_its_site(self, layout, mount):
        root = layout()
        attach_latent_model(root, "sae0", TinySAE(), LatentSite("point:hook_resid_post & layers:0"))
        attach_latent_model(root, "sae1", TinySAE(), LatentSite("point:hook_resid_post & layers:1"))
        assert latent_model_placements(root) == {
            "latent_models.sae0": f"{mount}.transformer.h.0",
            "latent_models.sae1": f"{mount}.transformer.h.1",
        }
        view = _view(root)
        structure = view.structure()
        assert [structure.locate(f"latent_models.sae{i}.enc.weight").layer for i in (0, 1)] == [0, 1]
        sae1 = {f"latent_models.sae1.{n}" for n in ("enc.weight", "enc.bias", "dec.weight", "dec.bias")}
        assert _selected(view, "role:latent & layers:1:") == sae1

    def test_a_transcoder_is_placed_at_the_module_it_rewrites(self):
        root = Core()
        site = LatentSite("point:mlp.hook_in & layers:1", write="point:mlp.hook_out & layers:1")
        attach_latent_model(root, "tc", TinySAE(), site)
        assert latent_model_placements(root) == {"latent_models.tc": "model.transformer.h.1.mlp"}
        location = _view(root).structure().locate("latent_models.tc.dec.weight")
        assert (location.layer, location.component) == (1, "mlp")

    def test_an_intervention_takes_the_layer_of_its_site(self):
        root = Core()
        attach_intervention(root, "steer", OpIntervention("add", D), "point:hook_resid_post & layers:1")
        assert intervention_placements(root) == {"interventions.steer": "model.transformer.h.1"}
        view = _view(root)
        steer = {f"interventions.steer.{n}" for n, _ in root.interventions["steer"].named_parameters()}
        assert _selected(view, "role:intervention & layers:1") == steer
        assert not _selected(view, "role:intervention & layers:0")

    def test_detaching_removes_the_placement(self):
        root = Core()
        attach_latent_model(root, "sae", TinySAE(), LatentSite("point:hook_resid_post & layers:0"))
        attach_intervention(root, "steer", OpIntervention("add", D), "point:hook_resid_post & layers:0")
        detach_latent_model(root, "sae")
        detach_intervention(root, "steer")
        assert latent_model_placements(root) == {} and intervention_placements(root) == {}

    def test_a_site_the_map_does_not_place_is_left_unplaced_with_a_warning(self):
        root = Core()
        root.side = nn.Linear(D, D)  # beside the mapped model: no map row places it
        attach_latent_model(root, "sae", TinySAE(), LatentSite("point:hook_resid_post & layers:0"))
        attach_latent_model(root, "aside", TinySAE(), LatentSite("side"))
        with pytest.warns(UserWarning, match="latent_models.aside could not be placed at their sites"):
            view = _view(root)
        structure = view.structure()
        assert structure.locate("latent_models.sae.enc.weight").layer == 0  # the placeable one keeps its place
        assert structure.locate("latent_models.aside.enc.weight").layer is None
        assert "latent_models.aside.enc.weight" in _selected(view, "role:latent:aside")


BOOK = {
    "schema_version": 1,
    "name": "steering",
    "sets": {"resid1": "point:hook_resid_post & layers:1", "mlp1_in": "point:mlp.hook_in & layers:1"},
}


class TestBookSites:
    """A site address may reference an address book's named set (#876), so a shared book names a splice site once
    for every backend."""

    def test_an_intervention_sited_by_a_book_set_matches_the_address(self):
        from interpretune.tuning.sites import book_sets

        sets = book_sets(BOOK)
        assert is_site_address("@resid1")
        assert resolve_site(Core(), "@resid1", sets) == resolve_site(Core(), "point:hook_resid_post & layers:1")
        assert resolve_site(NNsightLayout(), "@resid1", sets).path == "hf_model.transformer.h.1"
        root = Core()
        attach_intervention(root, "steer", OpIntervention("add", D), "@resid1", sets=sets)
        assert intervention_placements(root) == {"interventions.steer": "model.transformer.h.1"}

    def test_a_latent_model_sited_by_book_sets(self):
        from interpretune.tuning.sites import book_sets

        root = Core()
        site = LatentSite("@mlp1_in", write="point:mlp.hook_out & layers:1")
        attach_latent_model(root, "tc", TinySAE(), site, sets=book_sets(BOOK))
        assert latent_model_placements(root) == {"latent_models.tc": "model.transformer.h.1.mlp"}

    def test_a_configured_intervention_reads_its_book_from_a_path(self, tmp_path):
        import yaml

        from interpretune.tuning.interventions import attach_configured_interventions

        path = tmp_path / "steering.yaml"
        path.write_text(yaml.safe_dump(BOOK), encoding="utf-8")
        root = Core()
        attach_configured_interventions(root, [{"name": "steer", "site": "@resid1", "book": str(path), "shape": D}])
        assert intervention_sites(root) == {"steer": "@resid1"}
        assert intervention_placements(root) == {"interventions.steer": "model.transformer.h.1"}

    def test_a_set_reference_without_its_book_is_refused(self):
        with pytest.raises(ValueError, match="@resid1"):
            attach_intervention(Core(), "steer", OpIntervention("add", D), "@resid1")
