"""Each composition's finetuning-scheduler ``TunableView`` (#758) and its support record and attachment (#759)."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

pytest.importorskip("finetuning_scheduler.tunable_view", reason="finetuning-scheduler predates the tunable view")

from finetuning_scheduler.core.compile import compile_schedule  # noqa: E402
from finetuning_scheduler.core.exceptions import FTSConfigurationError  # noqa: E402
from finetuning_scheduler.tunable_view import as_view  # noqa: E402
from finetuning_scheduler.testing import pytest_params, TunableViewConformance  # noqa: E402
from finetuning_scheduler.tunable_view import TunableViewError, VIEW_ATTR as FTS_VIEW_ATTR  # noqa: E402

from interpretune.adapters.circuit_tracer.adapter import BaseCircuitTracerModule  # noqa: E402
from interpretune.session import ITSession  # noqa: E402
from interpretune.tuning import views as tuning_views  # noqa: E402
from tests.core import test_adapters_sae_lens as sl_tests  # noqa: E402
from interpretune.tuning.views import (  # noqa: E402
    VIEW_ATTR,
    RefusingView,
    attach_tunable_view,
    identity_view,
    peft_adapter_roles,
)


# an SAE at a hook the bridge splices (the fixture's default is skipped by `add_sae`, so nothing would register)
_SPLICEABLE_SAES = [sl_tests.TestClassSAELens.test_sl_cust_config]


def _fresh_module(request, cfg_fixture: str, **overrides):
    cfg = deepcopy(request.getfixturevalue(cfg_fixture))
    for k, v in overrides.items():
        setattr(cfg.module_cfg, k, v)
    return ITSession(cfg).module


def _view_factory(request, cfg_fixture: str, **overrides):
    # fixtures resolve once, here, not inside the factory each case calls
    cfg = deepcopy(request.getfixturevalue(cfg_fixture))
    for k, v in overrides.items():
        setattr(cfg.module_cfg, k, v)
    return lambda: getattr(ITSession(deepcopy(cfg)).module, VIEW_ATTR)


class TestAttachment:
    def test_core_composition_attaches_the_identity_view(self, request):
        module = request.getfixturevalue("get_it_session__core_cust__initonly").it_session.module
        view = getattr(module, VIEW_ATTR)
        support = as_view(view).support()
        assert view.root is module
        assert (support.trainable, support.naming_scheme) == (True, "physical")

    def test_tl_composition_attaches_the_bridge_view_without_building_it(self, request):
        from interpretune.adapters.transformer_lens.tunable_view import TransformerBridgeView

        module = _fresh_module(request, "get_it_session_cfg__tl_cust")
        view = getattr(module, VIEW_ATTR)
        assert isinstance(view, TransformerBridgeView) and view.root is module and view.bridge is module.model
        assert not view._built  # a session that never tunes never traces the bridge
        assert as_view(view).support().naming_scheme == "tl"

    def test_spliced_saes_take_the_latent_role(self, request):
        module = _fresh_module(request, "get_it_session_cfg__sl_cust", add_saes_on_init=True, sae_cfgs=_SPLICEABLE_SAES)
        roles = as_view(getattr(module, VIEW_ATTR)).roles()
        sae_names = {f"attached_saes.{n}" for n, _ in module.attached_saes.named_parameters()}
        assert sae_names and {n for n, r in roles.items() if r == "latent:sae"} == sae_names
        assert "latent" in as_view(getattr(module, VIEW_ATTR)).support().declared_roles
        # a schedule can select them by role
        compiled = compile_schedule({0: {"params": ["role:latent:sae"]}}, getattr(module, VIEW_ATTR))
        assert set(compiled.physical_for(0)) >= sae_names

    def test_unspliced_saes_have_no_role(self, request):
        module = _fresh_module(request, "get_it_session_cfg__sl_cust", add_saes_on_init=False)
        assert not as_view(getattr(module, VIEW_ATTR)).roles()

    def test_local_nnsight_composition_registers_its_hf_module(self, request):
        """The HF module is registered as `hf_model`, so the identity view tunes it (#774)."""
        module = request.getfixturevalue("get_it_session__ns_gpt2__setup").it_session.module
        assert module._modules["hf_model"] is module.model._model
        view = getattr(module, VIEW_ATTR)
        assert as_view(view).support().trainable
        names = [n for n, _ in module.named_parameters()]
        assert names and all(n.startswith("hf_model.") for n in names)
        # the view and `module.model.parameters()` reach the same tensors: the default optimizer lists each once
        opt_params = list(module.optimizer_parameters())
        assert len(opt_params) == len({id(p) for p in opt_params})
        assert {id(p) for p in opt_params} == {id(p) for p in module.model._model.parameters()}

    def test_remote_or_undispatched_nnsight_is_refused_with_its_reason(self):
        from types import SimpleNamespace

        from interpretune.adapters.nnsight.adapter import BaseNNsightModule

        remote = SimpleNamespace(it_cfg=SimpleNamespace(nnsight_cfg=SimpleNamespace(remote=True)))
        local = SimpleNamespace(it_cfg=SimpleNamespace(nnsight_cfg=SimpleNamespace(remote=False)))
        assert "#775" in BaseNNsightModule._hf_model_unregistered_reason(remote)  # type: ignore[arg-type]
        assert "no local HF module" in BaseNNsightModule._hf_model_unregistered_reason(local)  # type: ignore[arg-type]

    def test_an_undispatched_nnsight_composition_is_refused(self, request):
        """A real `dispatch=False` composition: weights stay on the meta device, so nothing is registered."""
        cfg = deepcopy(request.getfixturevalue("get_it_session_cfg__ns_gpt2"))
        cfg.module_cfg.nnsight_cfg.dispatch = False
        module = ITSession(cfg).module
        assert "hf_model" not in module._modules
        support = as_view(getattr(module, VIEW_ATTR)).support()
        assert not support.trainable and "not dispatched" in support.reason

    def test_nnsight_registration_survives_a_template_hook_override(self, request):
        """`post_auto_model_init` is a template hook (no `super()` required), so registration does not live
        there."""
        from tests.modules import TestITModule

        class NoSuperHook(TestITModule):
            def post_auto_model_init(self) -> None:
                pass

        cfg = deepcopy(request.getfixturevalue("get_it_session_cfg__ns_gpt2"))
        cfg.module_cls = NoSuperHook
        module = ITSession(cfg).module
        assert module._modules["hf_model"] is module.model._model
        assert as_view(getattr(module, VIEW_ATTR)).support().trainable

    def test_a_composition_without_nnsight_cfg_registers_nothing(self):
        """circuit-tracer over nnsight carries the nnsight module without an `nnsight_cfg` (its config is its own), and
        the registration hook runs after its init too: it must neither raise nor register its model."""
        from types import SimpleNamespace

        from interpretune.adapters.nnsight.adapter import BaseNNsightModule

        hf = nn.Linear(2, 2)
        module = SimpleNamespace(
            it_cfg=SimpleNamespace(), model=SimpleNamespace(_model=hf, dispatched=True), _modules={}
        )
        BaseNNsightModule._register_hf_model(module)  # type: ignore[arg-type]
        assert "hf_model" not in module._modules
        assert "not built through nnsight" in BaseNNsightModule._hf_model_unregistered_reason(module)  # type: ignore[arg-type]

    def test_circuit_tracer_is_refused_with_its_reason(self):
        view = BaseCircuitTracerModule.build_tunable_view(nn.Linear(2, 2))  # type: ignore[arg-type]
        support = as_view(view).support()
        assert not support.trainable and "frozen at construction" in support.reason

    def test_every_circuit_tracer_composition_takes_the_refusal(self):
        """The refusal reaches a composed circuit-tracer module only by MRO order (ahead of the TransformerLens and
        nnsight modules, whose views are trainable), so assert it on every registered composition."""
        from interpretune.adapter_registry import ADAPTER_REGISTRY
        from interpretune.protocol import Adapter

        keys = [k for k in ADAPTER_REGISTRY.keys() if k[0] == "module" and Adapter.circuit_tracer in k]
        assert keys
        for key in keys:
            composed = type("Composed", tuple(ADAPTER_REGISTRY.get(key)), {})
            assert composed.build_tunable_view is BaseCircuitTracerModule.build_tunable_view, key


class TestAttachRules:
    def test_undeclared_composition_is_refused_not_assumed_trainable(self):
        class Undeclared(nn.Linear):
            def build_tunable_view(self):
                return None

        module = Undeclared(2, 2)
        view = attach_tunable_view(module)
        assert isinstance(view, RefusingView) and getattr(module, VIEW_ATTR) is view
        with pytest.raises(FTSConfigurationError, match="declares no tuning support"):
            compile_schedule({0: {"params": ["weight"]}}, view)

    def test_the_view_is_attached_where_finetuning_scheduler_looks(self):
        assert VIEW_ATTR == FTS_VIEW_ATTR

    def test_an_attached_view_is_never_replaced(self):
        module = nn.Linear(2, 2)
        sentinel = identity_view(module)
        object.__setattr__(module, VIEW_ATTR, sentinel)
        assert attach_tunable_view(module) is None and getattr(module, VIEW_ATTR) is sentinel

    def test_nothing_is_attached_without_view_support(self, monkeypatch):
        monkeypatch.setattr(tuning_views, "fts_views_available", lambda: False)
        module = nn.Linear(2, 2)
        assert attach_tunable_view(module) is None and getattr(module, VIEW_ATTR, None) is None


_CANONICAL = {"class_path": "finetuning_scheduler.tunable_view.CanonicalView"}
_BRIDGE = "interpretune.adapters.transformer_lens.tunable_view.TransformerBridgeView"


class TestConfiguredView:
    """``it_cfg.tunable_view`` chooses the view in configuration, ahead of ``build_tunable_view()`` (#841)."""

    def test_a_tl_composition_can_schedule_its_own_parameter_names(self, request):
        from finetuning_scheduler.tunable_view import CanonicalView

        module = _fresh_module(request, "get_it_session_cfg__tl_cust", tunable_view=_CANONICAL)
        view = getattr(module, VIEW_ATTR)
        assert isinstance(view, CanonicalView) and view.root is module
        assert as_view(view).support().naming_scheme == "physical"
        name = next(n for n, _ in module.named_parameters() if n.endswith("ln1._original_component.weight"))
        assert compile_schedule({0: {"params": [name]}}, view).phases[0].physical == (name,)

    def test_the_bridge_view_can_leave_norms_to_the_schedule(self, request):
        no_thaw = {"class_path": _BRIDGE, "init_args": {"implicit_norm_thaw": False}}
        module = _fresh_module(request, "get_it_session_cfg__tl_cust", tunable_view=no_thaw)
        view = getattr(module, VIEW_ATTR)
        assert view.implicit_norm_thaw is False and view.root is module
        default = getattr(_fresh_module(request, "get_it_session_cfg__tl_cust"), VIEW_ATTR)
        physical = view.to_physical(["blocks.0.attn.W_Q"])["blocks.0.attn.W_Q"]
        norms = set(default.coupled(physical)) - set(view.coupled(physical))
        assert norms and all(".ln" in n for n in norms)  # the default thaws the block's norms, this view does not

    def test_the_compositions_roles_reach_a_configured_view(self, request):
        overrides = {"add_saes_on_init": True, "sae_cfgs": _SPLICEABLE_SAES, "tunable_view": _CANONICAL}
        module = _fresh_module(request, "get_it_session_cfg__sl_cust", **overrides)
        roles = as_view(getattr(module, VIEW_ATTR)).roles()
        assert roles and roles == {**peft_adapter_roles(module), **module.tunable_roles()}
        assert any(r.startswith("latent") for r in roles.values())

    def test_init_args_roles_are_used_as_given(self, request):
        cfg = {**_CANONICAL, "init_args": {"roles": {}}}
        module = _fresh_module(
            request, "get_it_session_cfg__sl_cust", add_saes_on_init=True, sae_cfgs=_SPLICEABLE_SAES, tunable_view=cfg
        )
        assert as_view(getattr(module, VIEW_ATTR)).roles() == {}

    @pytest.mark.parametrize(
        "bad, match",
        [
            pytest.param("finetuning_scheduler.tunable_view.CanonicalView", "mapping with a `class_path`", id="str"),
            pytest.param({"init_args": {}}, "mapping with a `class_path`", id="no_class_path"),
            pytest.param({**_CANONICAL, "args": [1]}, "not \\['args'\\]", id="extra_key"),
            pytest.param({**_CANONICAL, "init_args": [1]}, "must be a mapping", id="init_args"),
            pytest.param({**_CANONICAL, "init_args": {"root": None}}, "cannot set `root`", id="root"),
        ],
    )
    def test_a_malformed_configuration_is_refused(self, request, bad, match):
        from interpretune.utils import MisconfigurationException

        it_cfg = deepcopy(request.getfixturevalue("get_it_session_cfg__core_cust").module_cfg)
        it_cfg.tunable_view = bad
        with pytest.raises(MisconfigurationException, match=match):
            it_cfg._check_tunable_view()


class TestDefaultOptimizer:
    """The default optimizer covers what a trainable view selects, so modules beside the model train (#761)."""

    @staticmethod
    def _ids(params):
        return {id(p) for p in params}

    def test_modules_registered_beside_the_model_are_optimized(self, request):
        module = _fresh_module(request, "get_it_session_cfg__core_cust")
        module.intervention = nn.Linear(2, 2)
        covered = self._ids(module.optimizer_parameters())
        assert self._ids(module.model.parameters()) <= covered
        assert self._ids(module.intervention.parameters()) <= covered
        if module.it_cfg.optimizer_init:
            [optimizer], _ = module.configure_optimizers()
            assert self._ids(module.intervention.parameters()) <= self._ids(
                p for g in optimizer.param_groups for p in g["params"]
            )
        # control: without the view, only the model's own parameters are covered
        object.__setattr__(module, VIEW_ATTR, None)
        assert self._ids(module.optimizer_parameters()) == self._ids(module.model.parameters())

    def test_fitted_and_instrument_tensors_are_not_optimized(self, request):
        module = _fresh_module(request, "get_it_session_cfg__core_cust")
        module.fitted = nn.Linear(2, 2)
        module.instrument = nn.Linear(2, 2)
        module.intervention = nn.Linear(2, 2)
        roles = {f"fitted.{n}": "fitted" for n, _ in module.fitted.named_parameters()}
        roles |= {f"instrument.{n}": "instrument" for n, _ in module.instrument.named_parameters()}
        object.__setattr__(module, VIEW_ATTR, identity_view(module, roles=roles))
        covered = self._ids(module.optimizer_parameters())
        assert self._ids(module.intervention.parameters()) <= covered
        assert not covered & self._ids([*module.fitted.parameters(), *module.instrument.parameters()])

    def test_an_attached_non_view_breaks_the_protocol(self, request):
        module = _fresh_module(request, "get_it_session_cfg__core_cust")
        object.__setattr__(module, VIEW_ATTR, object())
        with pytest.raises(TunableViewError):
            list(module.optimizer_parameters())

    def test_a_refused_composition_keeps_the_models_parameters(self, request):
        module = _fresh_module(request, "get_it_session_cfg__core_cust")
        module.intervention = nn.Linear(2, 2)
        object.__setattr__(module, VIEW_ATTR, RefusingView(module, "refused for this test"))
        assert self._ids(module.optimizer_parameters()) == self._ids(module.model.parameters())


class TestBridgeView:
    """The TransformerLens view's storage refusal, live tensors and complete default schedule."""

    def test_packed_storage_is_refused_and_does_not_break_the_trace(self, request):
        from interpretune.adapters.transformer_lens.tunable_view import TransformerBridgeView

        module = _fresh_module(request, "get_it_session_cfg__tl_cust")
        holder = next(m for m in module.model.modules() if isinstance(m, nn.Linear))
        holder.register_parameter("packed", nn.Parameter(torch.zeros(4, dtype=torch.int8), requires_grad=False))
        view = TransformerBridgeView(module)
        support = as_view(view).support()
        assert not support.trainable and "packed" in support.reason
        assert view.untraced is not None  # tracing skips the integer tensor rather than raising

    def test_a_parameter_no_tl_name_reaches_is_scheduled_by_its_own_name(self, request):
        """A parameter no TransformerLens name reaches (here one registered on a block, as an unnamed norm or a
        fused projection is on a real architecture) resolves to itself; a reached parameter is scheduled by its
        TransformerLens name (#841)."""
        from finetuning_scheduler.tunable_view import Unresolvable

        module = _fresh_module(request, "get_it_session_cfg__tl_cust")
        module.model.blocks[0].register_parameter("extra", nn.Parameter(torch.zeros(2)))
        view = getattr(module, VIEW_ATTR)
        (unreached,) = [n for n, _ in module.named_parameters() if n.endswith(".extra")]
        assert dict(view.to_physical([unreached])) == {unreached: (unreached,)}
        assert view.to_logical([unreached]) == [unreached]  # and it round-trips
        compiled = compile_schedule({0: {"params": [unreached]}}, view)
        assert compiled.phases[0].physical == (unreached,)
        reached = view.to_physical(["blocks.0.attn.W_Q"])["blocks.0.attn.W_Q"][0]
        with pytest.raises(Unresolvable, match="TransformerLens name"):
            view.to_physical([reached])

    def test_logical_tensors_are_live(self, request):
        module = _fresh_module(request, "get_it_session_cfg__tl_cust")
        view = getattr(module, VIEW_ATTR)
        name, before = next(n_t for n_t in view.logical_named_parameters() if n_t[0].endswith("W_Q"))
        before = before.clone()
        (physical,) = view.to_physical([name])[name][:1]
        with torch.no_grad():
            module.get_parameter(physical).add_(1.0)
        after = dict(view.logical_named_parameters())[name]
        assert not torch.equal(before, after)

    def test_an_unordered_traced_name_gets_a_final_group(self, request):
        module = _fresh_module(request, "get_it_session_cfg__tl_cust")
        view = getattr(module, VIEW_ATTR)
        view._build()
        some_physical = next(iter(view._table.values()))
        view._table["extra.W_new"] = some_physical
        groups = view.depth_groups()
        assert groups[-1] == ["extra.W_new"]
        assert {n for g in groups for n in g} == set(view._table)


def test_peft_adapter_tensors_take_the_adapter_role():
    peft = pytest.importorskip("peft")

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x):
            return self.proj(x)

    class Holder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

    holder = Holder(peft.get_peft_model(Tiny(), peft.LoraConfig(r=2, target_modules=["proj"])))
    roles = peft_adapter_roles(holder)
    lora = {n for n, _ in holder.named_parameters() if ".lora_" in n}
    assert lora and set(roles) == lora and set(roles.values()) == {"adapter:default"}
    # the base weight PEFT renames stays `base`
    assert not any(n.endswith("base_layer.weight") for n in roles)
    view = identity_view(holder)
    assert as_view(view).support().declared_roles == frozenset({"base", "adapter"})


@pytest.mark.parametrize(
    "cfg_fixture, overrides",
    [
        pytest.param("get_it_session_cfg__core_cust", {}, id="core"),
        pytest.param("get_it_session_cfg__tl_cust", {}, id="tl"),
        pytest.param(
            "get_it_session_cfg__sl_cust", {"add_saes_on_init": True, "sae_cfgs": _SPLICEABLE_SAES}, id="sl_spliced"
        ),
    ],
)
def test_view_conformance(request, cfg_fixture, overrides):
    """Finetuning-scheduler's conformance cases, each on a fresh session, for every trainable composition here."""
    make_view = _view_factory(request, cfg_fixture, **overrides)
    report = TunableViewConformance(make_view).run()
    assert report.ok, report.failures
    assert "checkpoint_round_trip" in report.ran


def test_refusal_conformance():
    """The refusal case for a composition declaring `trainable=False`."""
    reason = "this composition cannot be tuned (a test reason)"
    (case,) = pytest_params(lambda: RefusingView(nn.Linear(2, 2), reason))
    case()


def test_tunable_support_requires_fts_views_only_when_built():
    """Importing the tuning package imports neither finetuning-scheduler nor any backend."""
    import subprocess
    import sys

    code = (
        "import sys, interpretune.tuning;"
        "print(any(m.startswith('finetuning_scheduler') for m in sys.modules), 'transformer_lens' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]
    assert torch  # torch is imported by interpretune anyway
