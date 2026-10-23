"""Each composition's finetuning-scheduler ``TunableView`` (#758) and its support record and attachment (#759)."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

pytest.importorskip("finetuning_scheduler.core.views", reason="finetuning-scheduler predates TunableView support")

from finetuning_scheduler.core.compile import compile_schedule  # noqa: E402
from finetuning_scheduler.core.exceptions import FTSConfigurationError  # noqa: E402
from finetuning_scheduler.core.views import as_view  # noqa: E402
from finetuning_scheduler.testing import pytest_params, TunableViewConformance  # noqa: E402

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

    def test_nnsight_composition_is_refused_with_its_reason(self, request):
        module = request.getfixturevalue("get_it_session__ns_gpt2__setup").it_session.module
        view = getattr(module, VIEW_ATTR)
        with pytest.raises(FTSConfigurationError, match="nnsight Envoy"):
            compile_schedule({0: {"params": ["anything"]}}, view)

    def test_circuit_tracer_is_refused_with_its_reason(self):
        view = BaseCircuitTracerModule.build_tunable_view(nn.Linear(2, 2))  # type: ignore[arg-type]
        support = as_view(view).support()
        assert not support.trainable and "frozen at construction" in support.reason


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

    def test_an_attached_view_is_never_replaced(self):
        module = nn.Linear(2, 2)
        sentinel = identity_view(module)
        object.__setattr__(module, VIEW_ATTR, sentinel)
        assert attach_tunable_view(module) is None and getattr(module, VIEW_ATTR) is sentinel

    def test_nothing_is_attached_without_view_support(self, monkeypatch):
        monkeypatch.setattr(tuning_views, "fts_views_available", lambda: False)
        module = nn.Linear(2, 2)
        assert attach_tunable_view(module) is None and getattr(module, VIEW_ATTR, None) is None


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

    def test_a_refused_composition_keeps_the_models_parameters(self, request):
        module = _fresh_module(request, "get_it_session_cfg__core_cust")
        module.intervention = nn.Linear(2, 2)
        object.__setattr__(module, VIEW_ATTR, RefusingView(module, "refused for this test"))
        assert self._ids(module.optimizer_parameters()) == self._ids(module.model.parameters())


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


def test_refusal_conformance(request):
    """The refusal case for a composition declaring `trainable=False`."""
    module = request.getfixturevalue("get_it_session__ns_gpt2__setup").it_session.module
    reason = as_view(getattr(module, VIEW_ATTR)).support().reason
    (case,) = pytest_params(lambda: RefusingView(module, reason))
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
