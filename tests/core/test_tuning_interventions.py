"""Learned interventions spliced into a module's forward pass and scheduled by role (#812)."""

from __future__ import annotations

from functools import partial

import pytest
import torch
from torch import nn

from interpretune.tuning.interventions import (
    INTERVENTIONS_ATTR,
    InterventionsMixin,
    SteeringVector,
    attach_intervention,
    detach_intervention,
    intervention_roles,
    intervention_sites,
)
from interpretune.tuning.latents import LatentModelsMixin, LatentSite

D = 8


class TupleBlock(nn.Module):
    """Returns ``(hidden, extra)``, as HF decoder layers do."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(D, D)

    def forward(self, x):
        return self.proj(x), x.sum()


class Root(nn.Module):
    """An interpretune-module stand-in: the model under ``model``, a head, and a cooperative ``tunable_roles()``."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = nn.ModuleDict(
            {"blocks": nn.ModuleList([nn.Linear(D, D), nn.Linear(D, D)]), "tuple_block": TupleBlock()}
        )
        self.head = nn.Linear(D, 2)

    def forward(self, x):
        for block in self.model["blocks"]:
            x = torch.relu(block(x))
        x, _ = self.model["tuple_block"](x)
        return self.head(x)

    def tunable_roles(self) -> dict[str, str]:
        return {}


class TinySAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc, self.dec = nn.Linear(D, 2 * D), nn.Linear(2 * D, D)

    def encode(self, x):
        return torch.relu(self.enc(x))

    def decode(self, z):
        return self.dec(z)


def _x():
    return torch.randn(4, D, generator=torch.Generator().manual_seed(1))


class TestSplicing:
    def test_a_steering_vector_changes_nothing_until_it_trains(self):
        root, x = Root(), _x()
        before = root(x)
        attach_intervention(root, "steer", SteeringVector(D), "model.blocks.0")
        assert torch.equal(root(x), before)
        root(x).sum().backward()
        assert root.interventions["steer"].vector.grad is not None

    def test_the_site_output_becomes_the_intervention_of_it(self):
        root, x = Root(), _x()
        steer = attach_intervention(root, "steer", SteeringVector(D, scale=2.0), "model.blocks.1")
        with torch.no_grad():
            steer.vector.fill_(0.5)
        seen = {}
        root.model["tuple_block"].register_forward_pre_hook(lambda m, args: seen.update(x=args[0]))
        root(x)
        b0, b1 = root.model["blocks"]
        h = nn.functional.linear(torch.relu(b0(x)), b1.weight, b1.bias)  # unhooked, so unsteered
        assert torch.allclose(seen["x"], torch.relu(h + 1.0))

    def test_a_tuple_output_has_its_first_element_rewritten(self):
        root = Root()
        steer = attach_intervention(root, "steer", SteeringVector(D), "model.tuple_block")
        with torch.no_grad():
            steer.vector.fill_(1.0)
        out = root.model["tuple_block"](_x())
        assert isinstance(out, tuple) and torch.allclose(out[0], root.model["tuple_block"].proj(_x()) + 1.0)

    def test_detaching_restores_the_forward_pass(self):
        root, x = Root(), _x()
        before = root(x)
        steer = attach_intervention(root, "steer", SteeringVector(D), "model.blocks.0")
        with torch.no_grad():
            steer.vector.fill_(1.0)
        assert not torch.equal(root(x), before)
        assert detach_intervention(root, "steer") is steer
        assert torch.equal(root(x), before) and "steer" not in root.interventions
        assert intervention_sites(root) == {}

    def test_the_intervention_is_part_of_the_root(self):
        root = Root()
        attach_intervention(root, "steer", SteeringVector(D), "model.blocks.0")
        assert f"{INTERVENTIONS_ATTR}.steer.vector" in root.state_dict()
        assert intervention_sites(root) == {"steer": "model.blocks.0"}

    def test_the_vector_follows_the_activations_dtype(self):
        steer = SteeringVector(D)
        assert steer(torch.zeros(2, D, dtype=torch.bfloat16)).dtype == torch.bfloat16


class TestRefusals:
    @pytest.mark.parametrize("name", ["a.b", "a:b", "", "a b"])
    def test_names_must_be_role_and_module_safe(self, name):
        with pytest.raises(ValueError, match="letters, digits"):
            attach_intervention(Root(), name, SteeringVector(D), "model.blocks.0")

    def test_a_kind_is_required(self):
        with pytest.raises(ValueError, match="kind None"):
            attach_intervention(Root(), "edit", nn.Linear(D, D), "model.blocks.0")

    def test_a_name_is_attached_once(self):
        root = Root()
        attach_intervention(root, "steer", SteeringVector(D), "model.blocks.0")
        with pytest.raises(ValueError, match="already attached"):
            attach_intervention(root, "steer", SteeringVector(D), "model.blocks.1")

    def test_a_missing_site_is_refused_before_anything_is_registered(self):
        root = Root()
        with pytest.raises(AttributeError):
            attach_intervention(root, "steer", SteeringVector(D), "model.blocks.7")
        assert len(root.interventions) == 0 and intervention_roles(root) == {}

    def test_detaching_an_unknown_name_raises(self):
        with pytest.raises(KeyError):
            detach_intervention(Root(), "steer")


class SteeredRoot(InterventionsMixin, LatentModelsMixin, Root):
    pass


class TestRoles:
    def test_tunable_and_frozen_interventions_take_their_roles(self):
        root = Root()
        attach_intervention(root, "steer", SteeringVector(D), "model.blocks.0")
        attach_intervention(root, "edit", nn.Linear(D, D), "model.blocks.1", kind="edit")
        attach_intervention(root, "held", SteeringVector(D), "model.tuple_block", frozen=True)
        assert intervention_roles(root) == {
            "interventions.steer.vector": "intervention:steer",
            "interventions.edit.weight": "intervention:edit",
            "interventions.edit.bias": "intervention:edit",
            "interventions.held.vector": "instrument",
        }

    def test_no_interventions_no_roles(self):
        assert intervention_roles(Root()) == {}

    def test_the_mixin_composes_with_the_latent_models_mixin(self):
        root = SteeredRoot()
        root.attach_intervention("steer", SteeringVector(D), "model.blocks.1")
        root.attach_latent_model("sae", TinySAE(), LatentSite("model.blocks.0"))
        roles = root.tunable_roles()
        assert roles["interventions.steer.vector"] == "intervention:steer"
        assert roles["latent_models.sae.enc.weight"] == "latent:sae"
        root.detach_intervention("steer")
        assert "interventions.steer.vector" not in root.tunable_roles()


class TestScheduling:
    """Finetuning-scheduler compiles and runs schedules over interventions by role."""

    @pytest.fixture(autouse=True)
    def _fts(self):
        pytest.importorskip("finetuning_scheduler.tunable_view", reason="finetuning-scheduler predates the view")

    @staticmethod
    def _train(fts, root, epochs):
        x, y = _x(), torch.tensor([0, 1, 0, 1])
        seen = []
        for _ in fts.epochs(epochs):
            nn.functional.cross_entropy(root(x), y).backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
            seen.append((fts.curr_depth, sorted(n for n, p in root.named_parameters() if p.requires_grad)))
            fts.epoch_end({})
        return seen

    def test_a_schedule_steers_first_then_thaws_blocks_by_name(self):
        from finetuning_scheduler import CanonicalView, TuningScheduler

        root = SteeredRoot()
        root.attach_intervention("steer", SteeringVector(D), "model.blocks.1")
        schedule = {
            0: {"params": ["role:intervention:steer"], "max_transition_epoch": 1},
            1: {"params": [r"model\.tuple_block\..*"]},
        }
        view = CanonicalView(root, roles=root.tunable_roles())
        fts = TuningScheduler(root, schedule, optimizer=partial(torch.optim.SGD, lr=0.1), view=view)
        fts.setup()
        before = {n: p.detach().clone() for n, p in root.named_parameters()}
        seen = self._train(fts, root, 2)
        block = ["model.tuple_block.proj.bias", "model.tuple_block.proj.weight"]
        assert seen == [(0, ["interventions.steer.vector"]), (1, sorted(["interventions.steer.vector", *block]))]
        changed = {n for n, p in root.named_parameters() if not torch.equal(before[n], p)}
        assert changed == {"interventions.steer.vector", *block}

    def test_a_frozen_intervention_steers_and_never_trains(self):
        from finetuning_scheduler import CanonicalView, TuningScheduler

        root = SteeredRoot()
        steer = root.attach_intervention("steer", SteeringVector(D), "model.blocks.1", frozen=True)
        with torch.no_grad():
            steer.vector.fill_(0.25)  # a precomputed direction, applied from the start
        schedule = {
            0: {"params": [r"model\.tuple_block\..*"], "max_transition_epoch": 1},
            1: {"params": [r"model\.blocks\..*"]},
        }
        view = CanonicalView(root, roles=root.tunable_roles())
        fts = TuningScheduler(root, schedule, optimizer=partial(torch.optim.SGD, lr=0.1), view=view)
        fts.setup()
        seen = self._train(fts, root, 2)
        assert all("interventions.steer.vector" not in trainable for _, trainable in seen)
        assert fts.roles["interventions.steer.vector"] == "instrument"
        assert torch.equal(steer.vector, torch.full((D,), 0.25))


class TestOnAnInterpretuneModule:
    """The seam on a real composition: a core module's model, a steering vector beside it, its identity view."""

    def test_a_core_module_compiles_and_trains_a_steering_vector_by_role(self, request):
        pytest.importorskip("finetuning_scheduler.tunable_view", reason="finetuning-scheduler predates the view")
        from copy import deepcopy

        from finetuning_scheduler.core.compile import compile_schedule

        from interpretune.session import ITSession
        from interpretune.tuning.views import identity_view

        module = ITSession(deepcopy(request.getfixturevalue("get_it_session_cfg__core_cust"))).module
        width = module.model.tok_embeddings.embedding_dim
        attach_intervention(module, "steer", SteeringVector(width), "model.layers.0")
        view = identity_view(module, roles={**module.tunable_roles(), **intervention_roles(module)})
        compiled = compile_schedule(
            {0: {"params": ["role:intervention:steer"]}, 1: {"params": ["model.layers.0.*"]}}, view
        )
        assert compiled.physical_for(0) == ("interventions.steer.vector",)
        ids = torch.randint(0, module.model.tok_embeddings.num_embeddings, (2, 5))
        module.model(ids).float().sum().backward()
        assert module.interventions["steer"].vector.grad is not None


def _site(module, cfg_fixture: str) -> str:
    """A first-block residual site, in the composition's own module names."""
    if cfg_fixture.startswith("get_it_session_cfg__core"):
        return "model.layers.0"
    if cfg_fixture.startswith("get_it_session_cfg__ns"):
        return "hf_model.transformer.h.0"
    # a TransformerLens bridge (sae_lens's bridge composition too): the first block's output hook point
    return next(
        f"model.{n}" for n, _ in module.model.named_modules() if n.startswith("blocks.0.") and n.endswith("hook_out")
    )


@pytest.mark.parametrize(
    "cfg_fixture",
    [
        pytest.param("get_it_session_cfg__core_cust", id="core"),
        pytest.param("get_it_session_cfg__tl_cust", id="tl"),
        pytest.param("get_it_session_cfg__ns_gpt2", id="nnsight"),
        pytest.param("get_it_session_cfg__sl_br_gpt2", id="sae_lens_bridge"),
    ],
)
def test_a_configured_steering_vector_is_attached_and_scheduled_by_role(request, cfg_fixture):
    """``tuning_interventions`` attaches a steering vector on every composition, declared in its own view."""
    pytest.importorskip("finetuning_scheduler.tunable_view", reason="finetuning-scheduler predates the view")
    from copy import deepcopy

    from finetuning_scheduler.core.compile import compile_schedule
    from finetuning_scheduler.tunable_view import as_view

    from interpretune.session import ITSession
    from interpretune.tuning.interventions import InterventionCfg
    from interpretune.tuning.views import VIEW_ATTR

    probe = ITSession(deepcopy(request.getfixturevalue(cfg_fixture))).module
    site = _site(probe, cfg_fixture)
    cfg = deepcopy(request.getfixturevalue(cfg_fixture))
    width = probe.model.tok_embeddings.embedding_dim if cfg_fixture.endswith("core_cust") else None
    cfg.module_cfg.tuning_interventions = [InterventionCfg(name="steer", site=site, width=width)]
    module = ITSession(cfg).module
    assert intervention_sites(module) == {"steer": site}
    view = getattr(module, VIEW_ATTR)
    roles = as_view(view).roles()
    assert {n for n, r in roles.items() if r == "intervention:steer"} == {"interventions.steer.vector"}
    compiled = compile_schedule({0: {"params": ["role:intervention:steer"]}}, view)
    assert compiled.physical_for(0) == ("interventions.steer.vector",)


def test_a_configured_kind_must_be_one_a_configuration_builds():
    from interpretune.tuning.interventions import InterventionCfg, attach_configured_interventions

    with pytest.raises(ValueError, match="builds only 'steer'"):
        attach_configured_interventions(Root(), [InterventionCfg(name="edit", site="model.blocks.0", kind="edit")])
    with pytest.raises(ValueError, match="needs a `width`"):
        attach_configured_interventions(Root(), [InterventionCfg(name="steer", site="model.blocks.0")])
    (steer,) = attach_configured_interventions(
        root := Root(), [InterventionCfg(name="s", site="model.blocks.0", width=D)]
    )
    assert root.interventions["s"] is steer and isinstance(steer, SteeringVector)
