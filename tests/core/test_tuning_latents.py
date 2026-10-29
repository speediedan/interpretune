"""Latent models spliced into a module's forward pass and scheduled by role (#812)."""

from __future__ import annotations

from functools import partial

import pytest
import torch
from torch import nn

from interpretune.tuning.latents import (
    LATENT_MODELS_ATTR,
    LatentModel,
    LatentModelsMixin,
    LatentSite,
    attach_latent_model,
    detach_latent_model,
    latent_model_roles,
    latent_model_sites,
)

D, K = 8, 16


class TinySAE(nn.Module):
    """Encodes and decodes one site's activations."""

    def __init__(self, d: int = D, k: int = K) -> None:
        super().__init__()
        self.enc, self.dec = nn.Linear(d, k), nn.Linear(k, d)

    def encode(self, x):
        return torch.relu(self.enc(x))

    def decode(self, z):
        return self.dec(z)


class TinyTranscoder(TinySAE):
    """Reads an MLP's input and writes its output (here the same width)."""


class TupleBlock(nn.Module):
    """Returns ``(hidden, extra)``, as HF decoder layers do."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(D, D)

    def forward(self, x):
        return self.proj(x), x.sum()


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

    def forward(self, x):
        return x + self.mlp(x)


class Root(nn.Module):
    """An interpretune-module stand-in: the model under ``model`` and a steering vector beside it."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = nn.ModuleDict({"blocks": nn.ModuleList([Block(), Block()]), "tuple_block": TupleBlock()})
        self.head = nn.Linear(D, 2)

    def forward(self, x):
        for block in self.model["blocks"]:
            x = block(x)
        x, _ = self.model["tuple_block"](x)
        return self.head(x)


def _x():
    return torch.randn(4, D, generator=torch.Generator().manual_seed(1))


class TestSplicing:
    def test_replace_runs_the_model_on_the_reconstruction(self):
        root, x = Root(), _x()
        sae = attach_latent_model(root, "sae", TinySAE(), LatentSite("model.blocks.0", mode="replace"))
        block_out = {}
        root.model["blocks"][1].register_forward_pre_hook(lambda m, a: block_out.setdefault("x", a[0]))
        root(x)
        clean = Root().model["blocks"][0](x)  # the same seed: the unspliced block's output
        assert torch.allclose(block_out["x"], sae.decode(sae.encode(clean)))

    def test_replace_with_error_keeps_the_forward_value_and_trains_the_latent_model(self):
        root, x = Root(), _x()
        expected = root(x)
        sae = attach_latent_model(root, "sae", TinySAE(), LatentSite("model.blocks.0"))
        out = root(x)
        assert torch.allclose(out, expected)
        out.sum().backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in sae.parameters())

    def test_observe_records_latents_and_changes_nothing(self):
        root, x = Root(), _x()
        expected = root(x)
        sae = attach_latent_model(root, "sae", TinySAE(), LatentSite("model.blocks.1", mode="observe"))
        assert torch.allclose(root(x), expected)
        assert sae.last_latents.shape == (4, K) and not sae.last_latents.requires_grad

    def test_a_transcoder_reads_an_mlp_input_and_writes_its_output(self):
        root, x = Root(), _x()
        site = LatentSite("model.blocks.0.mlp", read_from="input", mode="replace")
        tc = attach_latent_model(root, "mlp0", TinyTranscoder(), site)
        mlp_out = {}
        root.model["blocks"][0].mlp.register_forward_hook(lambda m, a, o: mlp_out.setdefault("o", o))
        root(x)
        assert torch.allclose(mlp_out["o"], tc.decode(tc.encode(x)))

    def test_a_tuple_output_has_its_first_element_rewritten(self):
        root, x = Root(), _x()
        attach_latent_model(root, "tb", TinySAE(), LatentSite("model.tuple_block", mode="replace"))
        hidden, extra = root.model["tuple_block"](x)
        assert torch.allclose(extra, x.sum())
        assert not torch.allclose(hidden, root.model["tuple_block"].proj(x))

    def test_detaching_restores_the_forward_pass(self):
        root, x = Root(), _x()
        expected = root(x)
        attach_latent_model(root, "sae", TinySAE(), LatentSite("model.blocks.0", mode="replace"))
        assert not torch.allclose(root(x), expected)
        detach_latent_model(root, "sae")
        assert torch.allclose(root(x), expected)
        assert "sae" not in getattr(root, LATENT_MODELS_ATTR) and latent_model_sites(root) == {}

    def test_the_latent_model_is_part_of_the_root(self):
        root = Root()
        attach_latent_model(root, "sae", TinySAE(), LatentSite("model.blocks.0"))
        assert {"latent_models.sae.enc.weight", "latent_models.sae.dec.bias"} <= set(root.state_dict())
        assert latent_model_sites(root) == {"sae": LatentSite("model.blocks.0")}


class TestRefusals:
    @pytest.mark.parametrize("name", ["", "a.b", "a b", "latent:x"])
    def test_names_must_be_role_and_module_safe(self, name):
        with pytest.raises(ValueError, match="must be letters"):
            attach_latent_model(Root(), name, TinySAE(), LatentSite("model.blocks.0"))

    def test_a_name_is_attached_once(self):
        root = Root()
        attach_latent_model(root, "sae", TinySAE(), LatentSite("model.blocks.0"))
        with pytest.raises(ValueError, match="already attached"):
            attach_latent_model(root, "sae", TinySAE(), LatentSite("model.blocks.1"))

    def test_a_module_without_encode_and_decode_is_refused(self):
        assert isinstance(TinySAE(), LatentModel) and not isinstance(nn.Linear(D, D), LatentModel)
        with pytest.raises(ValueError, match="encode"):
            attach_latent_model(Root(), "lin", nn.Linear(D, D), LatentSite("model.blocks.0"))

    def test_a_missing_site_is_refused_before_anything_is_registered(self):
        root = Root()
        with pytest.raises(AttributeError):
            attach_latent_model(root, "sae", TinySAE(), LatentSite("model.blocks.7"))
        assert "sae" not in getattr(root, LATENT_MODELS_ATTR)

    def test_detaching_an_unknown_name_raises(self):
        with pytest.raises(KeyError):
            detach_latent_model(Root(), "sae")


class SteeredRoot(Root):
    """A cooperative ``tunable_roles()`` below the mixin, as ``BaseITModule``'s is."""

    def __init__(self) -> None:
        super().__init__()
        self.steer = nn.Parameter(torch.zeros(D))

    def tunable_roles(self) -> dict[str, str]:
        return {"steer": "intervention:steer"}


class LatentRoot(LatentModelsMixin, SteeredRoot):
    pass


class TestRoles:
    def test_tunable_and_frozen_latent_models_take_their_roles(self):
        root = Root()
        attach_latent_model(root, "sae", TinySAE(), LatentSite("model.blocks.0"))
        attach_latent_model(root, "held", TinySAE(), LatentSite("model.blocks.1"), frozen=True)
        roles = latent_model_roles(root)
        assert {n for n, r in roles.items() if r == "latent:sae"} == {
            f"latent_models.sae.{n}" for n in ("enc.weight", "enc.bias", "dec.weight", "dec.bias")
        }
        assert {r for n, r in roles.items() if n.startswith("latent_models.held.")} == {"instrument"}

    def test_no_latent_models_no_roles(self):
        assert latent_model_roles(Root()) == {}

    def test_the_mixin_adds_its_roles_to_the_cooperative_ones(self):
        root = LatentRoot()
        root.attach_latent_model("sae", TinySAE(), LatentSite("model.blocks.0"))
        roles = root.tunable_roles()
        assert roles["steer"] == "intervention:steer" and roles["latent_models.sae.enc.weight"] == "latent:sae"
        root.detach_latent_model("sae")
        assert root.tunable_roles() == {"steer": "intervention:steer"}


class TestScheduling:
    """Finetuning-scheduler compiles and runs schedules over latent models by role."""

    @pytest.fixture(autouse=True)
    def _fts(self):
        pytest.importorskip("finetuning_scheduler.tunable_view", reason="finetuning-scheduler predates the view")

    @staticmethod
    def _root(frozen: bool = False):
        root = LatentRoot()
        root.attach_latent_model("sae", TinySAE(), LatentSite("model.blocks.0"), frozen=frozen)
        from finetuning_scheduler import CanonicalView

        return root, CanonicalView(root, roles=root.tunable_roles())

    @staticmethod
    def _train(fts, root, epochs):
        x, y = _x(), torch.tensor([0, 1, 0, 1])
        seen = []
        for _ in fts.epochs(epochs):
            loss = nn.functional.cross_entropy(root(x + root.steer), y)
            loss.backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
            seen.append((fts.curr_depth, sorted(n for n, p in root.named_parameters() if p.requires_grad)))
            fts.epoch_end({})
        return seen

    def test_a_schedule_thaws_the_latent_model_by_role_first(self):
        from finetuning_scheduler import TuningScheduler

        root, view = self._root()
        schedule = {
            0: {"params": ["role:latent:sae"], "max_transition_epoch": 1},
            1: {"params": ["role:intervention:steer"]},
        }
        fts = TuningScheduler(root, schedule, optimizer=partial(torch.optim.SGD, lr=0.1), view=view)
        fts.setup()
        before = {n: p.detach().clone() for n, p in root.named_parameters()}
        seen = self._train(fts, root, 2)
        latent = sorted(n for n in before if n.startswith("latent_models.sae."))
        assert seen == [(0, latent), (1, sorted([*latent, "steer"]))]
        changed = {n for n, p in root.named_parameters() if not torch.equal(before[n], p)}
        assert set(latent) <= changed and not any(n.startswith("model.") or n.startswith("head.") for n in changed)

    def test_a_frozen_latent_model_trains_once_promoted(self):
        from finetuning_scheduler import TuningScheduler

        root, view = self._root(frozen=True)
        schedule = {
            0: {"params": ["role:intervention:steer"], "max_transition_epoch": 1},
            1: {"params": ["role:latent:sae"], "promote": {"role:instrument": "latent:sae"}},
        }
        fts = TuningScheduler(root, schedule, optimizer=partial(torch.optim.SGD, lr=0.1), view=view)
        fts.setup()
        seen = self._train(fts, root, 2)
        latent = sorted(n for n, _ in root.named_parameters() if n.startswith("latent_models.sae."))
        assert seen == [(0, ["steer"]), (1, sorted([*latent, "steer"]))]
        assert fts.roles["latent_models.sae.enc.weight"] == "latent:sae"


class TestOnAnInterpretuneModule:
    """The seam on a real composition: a core module's model, a latent model beside it, its identity view."""

    def test_a_core_module_compiles_and_runs_a_latent_model_by_role(self, request):
        pytest.importorskip("finetuning_scheduler.tunable_view", reason="finetuning-scheduler predates the view")
        from copy import deepcopy

        from finetuning_scheduler.core.compile import compile_schedule

        from interpretune.session import ITSession
        from interpretune.tuning.views import identity_view

        module = ITSession(deepcopy(request.getfixturevalue("get_it_session_cfg__core_cust"))).module
        width = module.model.tok_embeddings.embedding_dim
        attach_latent_model(module, "sae", TinySAE(width, 2 * width), LatentSite("model.layers.0"))
        view = identity_view(module, roles={**module.tunable_roles(), **latent_model_roles(module)})
        compiled = compile_schedule({0: {"params": ["role:latent:sae"]}, 1: {"params": ["model.layers.0.*"]}}, view)
        assert sorted(compiled.physical_for(0)) == sorted(
            f"latent_models.sae.{n}" for n in ("enc.weight", "enc.bias", "dec.weight", "dec.bias")
        )
        ids = torch.randint(0, module.model.tok_embeddings.num_embeddings, (2, 5))
        module.model(ids).float().sum().backward()  # replace_with_error: the spliced forward trains the latent model
        assert all(p.grad is not None for p in module.latent_models["sae"].parameters())
