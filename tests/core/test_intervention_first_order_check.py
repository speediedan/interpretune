"""The first-order intervention check: predict the metric change from the edit.

Runs downstream of `model_fwd_intervention` in a composite: the measured change comes from its
recorded pre/post logits, while the prediction is built here from the metric gradient at the clean
site activation (captured through the backend gradient seam) and the applied edit recomputed with
the SHARED `apply_intervention` math the backends execute -- never a re-derivation.

The fake backend below is minimal but genuine: it executes real forwards, real hook-sited edits,
and real autograd. The gradient half is backend-shaped (hook lists in, cache dict out) so the op
under test cannot tell it from a real one; what it does not cover -- TL-bridge and NNsight tracing
behavior -- is the flagged backend fork, not a gap in these cases.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.backends import (
    InterventionMode,
    InterventionSupport,
    ModelBackendCapability,
    PositionScope,
)
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.bundled.concept import concept_ops

D, VOCAB, SEQ = 8, 16, 5
SITE = "proj"


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, D)
        self.proj = torch.nn.Linear(D, D, bias=False)
        self.act = torch.nn.GELU()
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.act(self.proj(self.embed(input_ids))))


class _GradBackend:
    """Declares the intervention + gradients surfaces and executes both for real."""

    capabilities = frozenset({ModelBackendCapability.ACTIVATION_INTERVENTION, ModelBackendCapability.GRADIENTS})
    intervention_support = InterventionSupport(
        position_scopes=frozenset(PositionScope),
        modes=frozenset(InterventionMode),
    )

    def _site_module(self, model):
        assert SITE in dict(model.named_modules()), "hook site missing from tiny model"
        return dict(model.named_modules())[SITE]

    def fwd_w_intervention(self, model, batch, interventions, latent_model_handles=None):
        from interpretune.analysis.backends import (
            apply_intervention,
            build_intervention_dict,
            get_intervention_target_shape,
        )

        with torch.no_grad():
            site = self._site_module(model)
            shapes = {}

            def shape_hook(_m, _i, out):
                shapes[SITE] = tuple(out.shape)

            handle = site.register_forward_hook(shape_hook)
            try:
                clean = model(**batch)
            finally:
                handle.remove()
            hook_shapes = {SITE: get_intervention_target_shape(torch.empty(shapes[SITE]))}
            canonical = build_intervention_dict(interventions, {SITE: [SITE]}, hook_shapes)
            _hook, specs = list(canonical.items())[0]
            assert len(specs) == 1
            spec = specs[0]

            def edit_hook(_m, _i, out):
                return apply_intervention(out.clone(), spec, last_pos=out.shape[1] - 1)

            handle = site.register_forward_hook(edit_hook)
            try:
                intervened = model(**batch)
            finally:
                handle.remove()
        return clean, intervened

    def fwd_w_grads_and_latent_models(self, model, batch, latent_model_handles, fwd_hooks, bwd_hooks, backward_fn):
        cache = {}

        class _Hook:
            def __init__(self, name):
                self.name = name

        handles = []
        site = self._site_module(model)
        for _pattern, cache_fn in fwd_hooks:

            def _fwd(_m, _i, out, _fn=cache_fn):
                _fn(out, _Hook(SITE))

            handles.append(site.register_forward_hook(_fwd))
        for _pattern, cache_fn in bwd_hooks:

            def _bwd(_m, _gi, grad_out, _fn=cache_fn):
                _fn(grad_out[0], _Hook(SITE))

            handles.append(site.register_full_backward_hook(_bwd))
        with torch.set_grad_enabled(True):
            logits = model(**batch)
            backward_fn(logits).backward()
        for handle in handles:
            handle.remove()
        self.last_cache = cache
        return logits


class _FakeCfg:
    auto_prune_batch_encoding = False
    names_filter = None
    cache_dict = None
    fwd_hooks = []
    bwd_hooks = []

    def __init__(self):
        self.cache_dict = {}

    def add_default_cache_hooks(self, include_backward=True):
        from interpretune.analysis.core import _make_simple_cache_hook

        self.fwd_hooks = [(self.names_filter, _make_simple_cache_hook(self.cache_dict))]
        self.bwd_hooks = (
            [(self.names_filter, _make_simple_cache_hook(self.cache_dict, is_backward=True))]
            if include_backward
            else []
        )


def _module():
    module = type("Module", (), {})()
    module.model = _Tiny().eval()
    module._model_backend = _GradBackend()
    module.analysis_cfg = _FakeCfg()
    module.sae_handles = []
    return module


def _ids():
    return torch.randint(0, VOCAB, (1, SEQ))


def _run_check(module, batch_fields, raw_batch, basis="embed"):
    batch = AnalysisBatch(**batch_fields, concept_basis=basis)
    out = concept_ops.model_fwd_intervention_impl(module, batch, raw_batch, 0)
    return concept_ops.intervention_first_order_check_impl(module, out, raw_batch, 0)


def _add_fields(scale, direction):
    return {
        "intervention_tensor": direction,
        "intervention_mode": "add",
        "intervention_scale_factor": scale,
        "intervention_hook_pattern": SITE,
        "intervention_position_scope": "last_token",
        "logit_target_ids": [3, 7],
    }


class TestAgreementAndBreakdown:
    def test_add_small_alpha_agrees_tightly(self):
        """A small edit stays in the linear regime: predicted ≈ measured."""
        torch.manual_seed(1)
        module, raw = _module(), {"input_ids": _ids()}
        direction = torch.randn(D)
        out = _run_check(module, _add_fields(0.05, direction), raw)
        assert out.concept_basis == "embed"
        assert abs(float(out.fo_residual)) < 1e-3, out.fo_residual
        assert abs(float(out.fo_predicted_delta) - float(out.fo_measured_delta)) < 1e-3

    def test_add_large_alpha_breaks_down(self):
        """Positive control: a large edit leaves the linear regime, so the residual must be large.

        Without it, an instrument that always reported agreement would pass.
        """
        torch.manual_seed(1)
        module, raw = _module(), {"input_ids": _ids()}
        direction = torch.randn(D)
        out = _run_check(module, _add_fields(5.0, direction), raw)
        assert abs(float(out.fo_residual)) > 1e-2, out.fo_residual


class TestAllModesReport:
    @pytest.mark.parametrize("mode", ["reject", "patch", "clamp"])
    def test_modes_report_finite_prediction_and_basis(self, mode):
        """Clamp travels by explicit mapping (its band has no shorthand fields); the rest shorthand."""
        torch.manual_seed(2)
        module, raw = _module(), {"input_ids": _ids()}
        base = {
            "intervention_hook_pattern": SITE,
            "intervention_position_scope": "last_token",
            "logit_target_ids": [3, 7],
            "concept_basis": "jlens_norm_aware",
        }
        if mode == "clamp":
            fields = {
                **base,
                "interventions": {
                    SITE: {
                        "intervention_tensor": torch.randn(D),
                        "mode": "clamp",
                        "clamp_min": -1.0,
                        "clamp_max": 1.0,
                    }
                },
            }
        else:
            fields = {
                **base,
                "intervention_tensor": torch.randn(2, D) if mode == "patch" else torch.randn(D),
                "intervention_mode": mode,
                "intervention_scale_factor": 1.0,
            }
        batch = AnalysisBatch(**fields)
        out = concept_ops.model_fwd_intervention_impl(module, batch, raw, 0)
        checked = concept_ops.intervention_first_order_check_impl(module, out, raw, 0)
        assert checked.concept_basis == "jlens_norm_aware"
        for field in ("fo_predicted_delta", "fo_measured_delta", "fo_residual"):
            assert torch.isfinite(getattr(checked, field)), (mode, field)


class TestRefusals:
    def test_absent_basis_is_refused(self):
        with pytest.raises(ValueError, match="concept_basis"):
            concept_ops.intervention_first_order_check_impl(_module(), AnalysisBatch(), {"input_ids": _ids()}, 0)

    def test_missing_pre_post_is_refused_naming_the_composition(self):
        with pytest.raises(ValueError, match="downstream of `model_fwd_intervention`"):
            concept_ops.intervention_first_order_check_impl(
                _module(),
                AnalysisBatch(
                    concept_basis="embed",
                    intervention_tensor=torch.randn(D),
                    intervention_mode="add",
                    intervention_hook_pattern=SITE,
                ),
                {"input_ids": _ids()},
                0,
            )

    def test_multi_hook_spec_is_refused(self):
        torch.manual_seed(3)
        module, raw = _module(), {"input_ids": _ids()}
        batch = AnalysisBatch(
            **_add_fields(0.05, torch.randn(D)),
            concept_basis="embed",
            pre_intervention_logits=torch.randn(1, SEQ, VOCAB),
            post_intervention_logits=torch.randn(1, SEQ, VOCAB),
            interventions={"a": {"mode": "add"}, "b": {"mode": "add"}},
        )
        with pytest.raises(ValueError, match="one site"):
            concept_ops.intervention_first_order_check_impl(module, batch, raw, 0)
