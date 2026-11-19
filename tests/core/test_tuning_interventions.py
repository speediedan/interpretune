"""Trainable interventions: an op with a trainable operand, spliced in and scheduled by role (#812, #950)."""

from __future__ import annotations

from functools import partial

import pytest
import torch
from torch import nn

from interpretune.tuning.interventions import (
    INTERVENTIONS_ATTR,
    OpIntervention,
    attach_intervention,
    detach_intervention,
    intervention_role_addresses,
    intervention_roles,
    intervention_sites,
)
from interpretune.tuning.latents import LatentSite, latent_model_role_addresses, latent_model_roles
from interpretune.tuning.views import expand_declared_roles, merge_roles

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
        attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.0")
        assert torch.equal(root(x), before)
        root(x).sum().backward()
        assert root.interventions["steer"].operand.value.grad is not None

    def test_the_site_output_becomes_the_intervention_of_it(self):
        root, x = Root(), _x()
        steer = attach_intervention(root, "steer", OpIntervention("add", D, scale_factor=2.0), "model.blocks.1")
        with torch.no_grad():
            steer.operand.value.fill_(0.5)
        seen = {}
        root.model["tuple_block"].register_forward_pre_hook(lambda m, args: seen.update(x=args[0]))
        root(x)
        b0, b1 = root.model["blocks"]
        h = nn.functional.linear(torch.relu(b0(x)), b1.weight, b1.bias)  # unhooked, so unsteered
        assert torch.allclose(seen["x"], torch.relu(h + 1.0))

    def test_a_tuple_output_has_its_first_element_rewritten(self):
        root = Root()
        steer = attach_intervention(root, "steer", OpIntervention("add", D), "model.tuple_block")
        with torch.no_grad():
            steer.operand.value.fill_(1.0)
        out = root.model["tuple_block"](_x())
        assert isinstance(out, tuple) and torch.allclose(out[0], root.model["tuple_block"].proj(_x()) + 1.0)

    def test_detaching_restores_the_forward_pass(self):
        root, x = Root(), _x()
        before = root(x)
        steer = attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.0")
        with torch.no_grad():
            steer.operand.value.fill_(1.0)
        assert not torch.equal(root(x), before)
        assert detach_intervention(root, "steer") is steer
        assert torch.equal(root(x), before) and "steer" not in root.interventions
        assert intervention_sites(root) == {}

    def test_the_intervention_is_part_of_the_root(self):
        root = Root()
        attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.0")
        assert f"{INTERVENTIONS_ATTR}.steer.operand.value" in root.state_dict()
        assert intervention_sites(root) == {"steer": "model.blocks.0"}

    def test_the_intervention_moves_to_the_sites_device_and_keeps_its_dtype(self):
        root = Root().to(device="meta", dtype=torch.float16)
        steer = attach_intervention(root, "s", OpIntervention("add", D), "model.blocks.0")
        assert steer.operand.value.device.type == "meta" and steer.operand.value.dtype == torch.float32
        root.model["hook"] = nn.Identity()  # a site holding no tensors (a hook point) takes the root's device
        hooked = attach_intervention(root, "h", OpIntervention("add", D), "model.hook")
        assert hooked.operand.value.device.type == "meta"

    def test_the_vector_follows_the_activations_dtype(self):
        steer = OpIntervention("add", D)
        assert steer(torch.zeros(2, D, dtype=torch.bfloat16)).dtype == torch.bfloat16


class TestRefusals:
    @pytest.mark.parametrize("name", ["a.b", "a:b", "", "a b"])
    def test_names_must_be_role_and_module_safe(self, name):
        with pytest.raises(ValueError, match="letters, digits"):
            attach_intervention(Root(), name, OpIntervention("add", D), "model.blocks.0")

    def test_a_circuit_tracer_composition_takes_one_only_where_its_forward_runs(self):
        class BaseCircuitTracerModule(Root):  # matched by name: circuit-tracer is optional
            pass

        with pytest.raises(ValueError, match="no torch module runs its replacement model's forward"):
            attach_intervention(BaseCircuitTracerModule(), "steer", OpIntervention("add", D), "model.blocks.0")
        root = BaseCircuitTracerModule()
        object.__setattr__(root, "_replacement_model", root.model["blocks"])  # the model circuit-tracer runs
        with pytest.raises(ValueError, match="only at modules of the model circuit-tracer runs"):
            attach_intervention(root, "steer", OpIntervention("add", D), "model.tuple_block")
        assert "steer" not in getattr(root, INTERVENTIONS_ATTR)
        assert (
            attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.1")
            is root.interventions["steer"]
        )

    def test_a_configured_vector_cannot_be_frozen(self):
        from interpretune.tuning.interventions import as_intervention_cfg

        # it would start at zero and never train: hold a trained vector fixed with `refreeze` instead
        with pytest.raises(TypeError, match="frozen"):
            as_intervention_cfg({"name": "steer", "site": "model.blocks.0", "frozen": True})

    def test_a_name_is_attached_once(self):
        root = Root()
        attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.0")
        with pytest.raises(ValueError, match="already attached"):
            attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.1")

    def test_a_missing_site_is_refused_before_anything_is_registered(self):
        root = Root()
        with pytest.raises(AttributeError):
            attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.7")
        assert len(root.interventions) == 0 and intervention_roles(root) == {}

    def test_detaching_an_unknown_name_raises(self):
        with pytest.raises(KeyError):
            detach_intervention(Root(), "steer")


class SteeredRoot(Root):
    """Declares the attached latent models' and interventions' roles as ``BaseITModule.tunable_roles`` does."""

    def tunable_roles(self) -> dict[str, str]:
        return {**latent_model_roles(self), **intervention_roles(self)}


class TestRoles:
    def test_tunable_and_frozen_interventions_take_their_roles(self):
        root = Root()
        attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.0")
        attach_intervention(root, "edit", nn.Linear(D, D), "model.blocks.1")
        attach_intervention(root, "held", OpIntervention("add", D), "model.tuple_block", frozen="artifact")
        attach_intervention(root, "mean", OpIntervention("add", D), "model.blocks.0", frozen="fitted")
        assert intervention_roles(root) == {
            "interventions.steer.operand.value": "intervention:steer",
            "interventions.edit.weight": "intervention:edit",
            "interventions.edit.bias": "intervention:edit",
            "interventions.held.operand.value": "artifact",
            "interventions.mean.operand.value": "fitted",
        }

    @pytest.mark.parametrize("frozen", [True, "instrument", "latent"])
    def test_frozen_must_say_whether_the_value_is_saved(self, frozen):
        """``True`` is refused: a value computed in the run (``fitted``) is checkpointed, a loaded one
        (``artifact``) is not, and guessing loses one or the other on a resumed run."""
        root = Root()
        with pytest.raises(ValueError, match="'fitted'.*'artifact'"):
            attach_intervention(root, "held", OpIntervention("add", D), "model.blocks.0", frozen=frozen)
        assert intervention_roles(root) == {}

    def test_a_fitted_intervention_is_checkpointed_and_an_artifact_is_not(self):
        """A value computed in the run survives a resume through the checkpoint; a loaded one is reloaded
        instead."""
        from venngram.tuning import CanonicalView

        root = SteeredRoot()
        attach_intervention(root, "mean", OpIntervention("add", D), "model.blocks.0", frozen="fitted")
        attach_intervention(root, "held", OpIntervention("add", D), "model.blocks.1", frozen="artifact")
        saved = CanonicalView(root, roles=root.tunable_roles()).checkpoint_state_dict()
        assert "interventions.mean.operand.value" in saved
        assert "interventions.held.operand.value" not in saved

    def test_no_interventions_no_roles(self):
        assert intervention_roles(Root()) == {}

    def test_latent_model_and_intervention_roles_are_declared_together(self):
        from interpretune.tuning.latents import attach_latent_model

        root = SteeredRoot()
        attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.1")
        attach_latent_model(root, "sae", TinySAE(), LatentSite("model.blocks.0"))
        roles = root.tunable_roles()
        assert roles["interventions.steer.operand.value"] == "intervention:steer"
        assert roles["latent_models.sae.enc.weight"] == "latent:sae"
        detach_intervention(root, "steer")
        assert "interventions.steer.operand.value" not in root.tunable_roles()


class TestRoleAddresses:
    """``BaseITModule.tunable_roles()`` declares these roles by address, and each address selects exactly the
    tensors the name map named (#875)."""

    @staticmethod
    def _spliced():
        from interpretune.tuning.latents import attach_latent_model

        root = Root()
        attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.0")
        attach_intervention(root, "edit", nn.Linear(D, D), "model.blocks.1")
        attach_intervention(root, "held", OpIntervention("add", D), "model.tuple_block", frozen="artifact")
        attach_intervention(root, "mean", OpIntervention("add", D), "model.blocks.0", frozen="fitted")
        attach_latent_model(root, "sae", TinySAE(), LatentSite("model.blocks.0"))
        attach_latent_model(root, "loaded", TinySAE(), LatentSite("model.blocks.1"), frozen=True)
        return root

    def test_each_role_is_declared_by_address_and_expands_to_the_names(self):
        root = self._spliced()
        declared = merge_roles(latent_model_role_addresses(root), intervention_role_addresses(root))
        assert declared == {
            "latent:sae": "module:latent_models.sae",
            "artifact": "module:latent_models.loaded | module:interventions.held",  # one role, both frozen modules
            "intervention:steer": "module:interventions.steer",
            "intervention:edit": "module:interventions.edit",
            "fitted": "module:interventions.mean",
        }
        assert expand_declared_roles(root, declared) == {**latent_model_roles(root), **intervention_roles(root)}

    def test_a_view_given_the_addresses_has_the_roles_of_the_names(self):
        from venngram.tuning import CanonicalView
        from venngram.tuning.conformance import role_address_case

        root = self._spliced()
        declared = merge_roles(latent_model_role_addresses(root), intervention_role_addresses(root))
        role_address_case(lambda roles: CanonicalView(root, roles=roles), declared).run()

    def test_nothing_attached_declares_nothing(self):
        assert latent_model_role_addresses(Root()) == {} and intervention_role_addresses(Root()) == {}

    def test_merging_joins_one_roles_addresses_and_a_later_name_wins(self):
        assert merge_roles(
            {"artifact": "module:a", "head.weight": "fitted"},
            {"artifact": "module:b", "head.weight": "base", "latent:sae": "module:c"},
            {"latent:sae": "module:c"},  # the same address again adds nothing
        ) == {"artifact": "module:a | module:b", "head.weight": "base", "latent:sae": "module:c"}


class TestScheduling:
    """Finetuning-scheduler compiles and runs schedules over interventions by role."""

    @pytest.fixture(autouse=True)
    def _fts(self):
        pytest.importorskip("finetuning_scheduler.core.views", reason="finetuning-scheduler predates tunable views")

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
        attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.1")
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
        assert seen == [
            (0, ["interventions.steer.operand.value"]),
            (1, sorted(["interventions.steer.operand.value", *block])),
        ]
        changed = {n for n, p in root.named_parameters() if not torch.equal(before[n], p)}
        assert changed == {"interventions.steer.operand.value", *block}

    def test_a_stage_refreezes_a_trained_vector_and_trains_the_block(self):
        """Phases as stages: steer first, then hold the trained vector fixed (finetuning-scheduler's ``refreeze``)
        while the block trains from where the steering stage ended (``restore: none``)."""
        from finetuning_scheduler import TuningScheduler

        from interpretune.tuning.views import identity_view

        root = SteeredRoot()
        steer = attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.1")
        schedule = {
            0: {"params": ["role:intervention:steer"], "max_transition_epoch": 1},
            1: {"params": [r"model\.tuple_block\..*"], "refreeze": ["role:intervention"], "restore": "none"},
        }
        view = identity_view(root, roles=root.tunable_roles())
        fts = TuningScheduler(root, schedule, optimizer=partial(torch.optim.SGD, lr=0.1), view=view)
        fts.setup()
        seen, trained = [], None
        x, y = _x(), torch.tensor([0, 1, 0, 1])
        for _ in fts.epochs(2):
            if fts.curr_depth == 1 and trained is None:
                trained = steer.operand.value.detach().clone()
            nn.functional.cross_entropy(root(x), y).backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
            seen.append((fts.curr_depth, sorted(n for n, p in root.named_parameters() if p.requires_grad)))
            fts.epoch_end({})
        block = ["model.tuple_block.proj.bias", "model.tuple_block.proj.weight"]
        assert seen == [(0, ["interventions.steer.operand.value"]), (1, block)]
        assert trained is not None and trained.abs().sum() > 0  # the steering stage moved it
        assert torch.equal(steer.operand.value, trained)  # and the block stage held it fixed

    @pytest.mark.parametrize("frozen", ["fitted", "artifact"])
    def test_a_frozen_intervention_steers_and_never_trains(self, frozen):
        from finetuning_scheduler import CanonicalView, TuningScheduler

        root = SteeredRoot()
        steer = attach_intervention(root, "steer", OpIntervention("add", D), "model.blocks.1", frozen=frozen)
        with torch.no_grad():
            steer.operand.value.fill_(0.25)  # a precomputed direction, applied from the start
        schedule = {
            0: {"params": [r"model\.tuple_block\..*"], "max_transition_epoch": 1},
            1: {"params": [r"model\.blocks\..*"]},
        }
        view = CanonicalView(root, roles=root.tunable_roles())
        fts = TuningScheduler(root, schedule, optimizer=partial(torch.optim.SGD, lr=0.1), view=view)
        fts.setup()
        seen = self._train(fts, root, 2)
        assert all("interventions.steer.operand.value" not in trainable for _, trainable in seen)
        assert fts.roles["interventions.steer.operand.value"] == frozen
        assert torch.equal(steer.operand.value, torch.full((D,), 0.25))


class TestOnAnInterpretuneModule:
    """The seam on a real composition: a core module's model, a steering vector beside it, its identity view."""

    def test_a_core_module_compiles_and_trains_a_steering_vector_by_role(self, request):
        pytest.importorskip("finetuning_scheduler.core.views", reason="finetuning-scheduler predates tunable views")
        from copy import deepcopy

        from finetuning_scheduler.core.compile import compile_schedule

        from interpretune.session import ITSession
        from interpretune.tuning.views import identity_view

        module = ITSession(deepcopy(request.getfixturevalue("get_it_session_cfg__core_cust"))).module
        d_model = module.model.tok_embeddings.embedding_dim
        attach_intervention(module, "steer", OpIntervention("add", d_model), "model.layers.0")
        view = identity_view(module, roles={**module.tunable_roles(), **intervention_roles(module)})
        compiled = compile_schedule(
            {0: {"params": ["role:intervention:steer"]}, 1: {"params": ["model.layers.0.*"]}}, view
        )
        assert compiled.physical_for(0) == ("interventions.steer.operand.value",)
        ids = torch.randint(0, module.model.tok_embeddings.num_embeddings, (2, 5))
        module.model(ids).float().sum().backward()
        assert module.interventions["steer"].operand.value.grad is not None


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
@pytest.mark.parametrize("as_mapping", [False, True], ids=["cfg", "mapping"])
def test_a_configured_steering_vector_is_attached_and_scheduled_by_role(request, cfg_fixture, as_mapping):
    """``tuning_interventions`` attaches a steering vector on every composition, declared in its own view, whether
    an entry is an ``InterventionCfg`` or the mapping a YAML configuration gives."""
    pytest.importorskip("finetuning_scheduler.core.views", reason="finetuning-scheduler predates tunable views")
    from copy import deepcopy

    from finetuning_scheduler.core.compile import compile_schedule
    from finetuning_scheduler.core.views import as_view

    from interpretune.session import ITSession
    from interpretune.tuning.interventions import InterventionCfg
    from interpretune.tuning.views import VIEW_ATTR

    probe = ITSession(deepcopy(request.getfixturevalue(cfg_fixture))).module
    site = _site(probe, cfg_fixture)
    cfg = deepcopy(request.getfixturevalue(cfg_fixture))
    d_model = probe.model.tok_embeddings.embedding_dim if cfg_fixture.endswith("core_cust") else None
    entry = {"name": "steer", "site": site, "shape": d_model}
    cfg.module_cfg.tuning_interventions = [entry if as_mapping else InterventionCfg(**entry)]
    module = ITSession(cfg).module
    assert intervention_sites(module) == {"steer": site}
    view = getattr(module, VIEW_ATTR)
    roles = as_view(view).roles()
    assert {n for n, r in roles.items() if r == "intervention:steer"} == {"interventions.steer.operand.value"}
    compiled = compile_schedule({0: {"params": ["role:intervention:steer"]}}, view)
    assert compiled.physical_for(0) == ("interventions.steer.operand.value",)


@pytest.mark.parametrize(
    "cfg_fixture",
    [
        pytest.param("get_it_session_cfg__tl_cust", id="tl"),
        pytest.param("get_it_session_cfg__ns_gpt2", id="nnsight"),
        pytest.param("get_it_session_cfg__sl_br_gpt2", id="sae_lens_bridge"),
    ],
)
def test_a_configured_site_address_resolves_on_every_composition(request, cfg_fixture):
    """One site address in ``tuning_interventions`` attaches at the first block's residual output on each
    composition: the module ``_site`` names in that composition's own spelling (#864)."""
    from copy import deepcopy

    from interpretune.session import ITSession
    from interpretune.tuning.sites import resolve_site

    address = "point:hook_resid_post & layers:0"
    probe = ITSession(deepcopy(request.getfixturevalue(cfg_fixture))).module
    resolved = resolve_site(probe, address)
    assert resolved.io == "output"
    if cfg_fixture.startswith("get_it_session_cfg__ns"):
        assert resolved.path == _site(probe, cfg_fixture)
    else:  # a bridge names the block's residual output `blocks.0.hook_out`
        assert resolved.path == "model.blocks.0.hook_out"
    cfg = deepcopy(request.getfixturevalue(cfg_fixture))
    cfg.module_cfg.tuning_interventions = [{"name": "steer", "site": address}]
    assert intervention_sites(ITSession(cfg).module) == {"steer": address}


def test_a_site_address_is_refused_on_a_model_without_a_component_map(request):
    """The custom core model has no HF config or TransformerLens hooks, so an address cannot be placed."""
    from copy import deepcopy

    from interpretune.session import ITSession

    module = ITSession(deepcopy(request.getfixturevalue("get_it_session_cfg__core_cust"))).module
    with pytest.raises(ValueError, match="layer count|neither a TransformerLens model nor an HF model"):
        attach_intervention(module, "steer", OpIntervention("add", D), "point:hook_resid_post & layers:0")


def test_a_configured_op_must_be_one_a_trainable_intervention_applies():
    from interpretune.tuning.interventions import InterventionCfg, OpIntervention, attach_configured_interventions

    with pytest.raises(ValueError, match="'reject' is a registered op that this seam does not yet apply"):
        attach_configured_interventions(
            Root(), [InterventionCfg(name="edit", site="model.blocks.0", op="reject", shape=D)]
        )
    with pytest.raises(ValueError, match="'steer' is not a registered op"):
        attach_configured_interventions(
            Root(), [InterventionCfg(name="edit", site="model.blocks.0", op="steer", shape=D)]
        )
    with pytest.raises(ValueError, match="needs a `shape`"):
        attach_configured_interventions(Root(), [InterventionCfg(name="steer", site="model.blocks.0")])
    with pytest.raises(TypeError, match="InterventionCfg or a mapping"):
        attach_configured_interventions(Root(), ["model.blocks.0"])
    (steer, gain) = attach_configured_interventions(
        root := Root(),
        [
            InterventionCfg(name="s", site="model.blocks.0", shape=D),
            InterventionCfg(name="g", site="model.blocks.1", op="scale", shape=D),
        ],
    )
    assert root.interventions["s"] is steer and isinstance(steer, OpIntervention) and steer.op == "add"
    assert gain.op == "scale"


class TestOps:
    """A trainable intervention is an op applied with a trainable operand, through the shared op math."""

    def test_each_op_starts_at_its_identity_and_trains(self):
        from interpretune.tuning.interventions import OpIntervention

        x = _x()
        for op in ("add", "scale"):
            intervention = OpIntervention(op, D, scale_factor=2.0)
            assert torch.equal(intervention(x), x)  # attaching it changes nothing until it trains
            intervention(x).sum().backward()
            assert intervention.operand.value.grad is not None

    def test_a_tuple_shape_gives_a_per_head_operand(self):
        from interpretune.tuning.interventions import InterventionCfg, OpIntervention

        x = torch.randn(2, 3, 2, D // 2, generator=torch.Generator().manual_seed(3))  # (batch, pos, n_heads, d_head)
        gain = OpIntervention("scale", (2, D // 2))
        assert gain.operand.value.shape == (2, D // 2) and torch.equal(gain(x), x)
        gain(x).sum().backward()
        assert gain.operand.value.grad is not None
        assert InterventionCfg(name="g", site="model.blocks.0", shape=[2, D // 2]).shape == (2, D // 2)  # a YAML list

    def test_the_ops_match_the_analysis_op_math(self):
        from interpretune.analysis.backends.interventions import InterventionSpec, apply_intervention
        from interpretune.tuning.interventions import OpIntervention

        x = torch.randn(2, 3, D, generator=torch.Generator().manual_seed(2))
        for op in ("add", "scale"):
            intervention = OpIntervention(op, D, scale_factor=0.5)
            with torch.no_grad():
                intervention.operand.value.uniform_(0.5, 1.5)
            spec = InterventionSpec(
                intervention.operand.value.detach(), op=op, scale_factor=0.5, position_scope="all_positions"
            )
            assert torch.allclose(intervention(x), apply_intervention(x.clone(), spec, last_pos=-1))

    def test_an_operand_module_supplies_the_ops_tensor(self):
        from interpretune.tuning.interventions import LearnedOperand, OpIntervention

        heads = OpIntervention("scale", LearnedOperand((2, D // 2), init=1.0))
        x = torch.randn(4, 3, 2, D // 2)
        with torch.no_grad():
            heads.operand.value[0].zero_()  # a learned per-head gain: the first head ablated
        assert torch.equal(heads(x)[..., 0, :], torch.zeros(4, 3, D // 2)) and torch.equal(
            heads(x)[..., 1, :], x[..., 1, :]
        )

    def test_apply_op_is_out_of_place_and_every_position_only(self):
        from interpretune.analysis.backends.interventions import InterventionSpec, apply_op

        x = _x()
        before = x.clone()
        apply_op(x, InterventionSpec(torch.ones(D), op="add", position_scope="all_positions"))
        assert torch.equal(x, before)
        with pytest.raises(ValueError, match="edits every position"):
            apply_op(x, InterventionSpec(torch.ones(D), op="add"))
