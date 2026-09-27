"""Holding residual-stream norm denominators at their clean values during a feature intervention, and the far-
upstream attribution share recorded for targets read at a block's output."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from interpretune.adapters.circuit_tracer.backends import CircuitTracerAnalysisBackend
from interpretune.adapters.circuit_tracer.norm_freeze import frozen_norm_denominators, residual_norm_modules
from tests.core.circuit_tracer_toy import TOY_PROMPT, only_token_zero_special, tiny_gemma2_replacement_model


@pytest.fixture(scope="module")
def toy():
    model = tiny_gemma2_replacement_model(n_layers=3)
    with only_token_zero_special(model):
        yield model


def _norm_io(norm: nn.Module):
    """Record a norm module's input and output on each call."""
    seen: dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        seen["x"], seen["y"] = inputs[0].detach().float(), output.detach().float()

    return seen, norm.register_forward_hook(hook)


def _inverse_rms(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + module.eps)


def _layer0_edit(toy) -> list[tuple[int, int, int, float]]:
    """Scale up the first active layer-0 feature fivefold: a write that every later norm sees."""
    _, acts = toy.get_activations(TOY_PROMPT, apply_activation_function=False)
    pos, feat = (int(i) for i in torch.nonzero(acts[0] > 0)[0])
    return [(0, pos, feat, float(acts[0, pos, feat]) * 5.0)]


def test_every_residual_norm_is_found_and_attention_norms_are_not(toy):
    norms = residual_norm_modules(toy)
    # four per Gemma 2 block (input, post-attention, pre- and post-feedforward) plus the final norm
    assert len(norms) == 4 * 3 + 1
    assert all(kind == "rms" for _, kind in norms)


def test_the_freeze_is_the_identity_on_a_clean_pass(toy):
    """With nothing intervened on, freezing the denominators to their clean values must change nothing."""
    clean_logits, _ = toy.get_activations(TOY_PROMPT)
    with frozen_norm_denominators(toy, TOY_PROMPT):
        frozen_logits, _ = toy.get_activations(TOY_PROMPT)
    torch.testing.assert_close(frozen_logits, clean_logits, atol=1e-5, rtol=1e-5)


def test_an_intervention_moves_a_downstream_denominator(toy):
    """Positive control for the test below: the edit does move a downstream norm's denominator, so holding it fixed
    is a measurable act rather than a no-op on this model."""
    norm = residual_norm_modules(toy)[4 + 2][0]  # block 1's pre-feedforward norm, downstream of the edit
    intervention = _layer0_edit(toy)
    seen, handle = _norm_io(norm)
    try:
        toy.get_activations(TOY_PROMPT)
        clean_inv = _inverse_rms(norm, seen["x"])
        toy.feature_intervention(TOY_PROMPT, intervention, apply_activation_function=False)
    finally:
        handle.remove()
    assert not torch.allclose(_inverse_rms(norm, seen["x"]), clean_inv, rtol=1e-3), "the edit never moved the norm"


def test_the_frozen_output_is_what_the_next_module_receives(toy):
    """The freeze replaces the norm's output: the module after it must see the clean-scale value."""
    norm = residual_norm_modules(toy)[4 + 2][0]
    block = toy.pre_logit_location.layers[1]
    mlp = getattr(block, "_module", block).mlp
    intervention = _layer0_edit(toy)
    seen_norm, h1 = _norm_io(norm)
    seen_mlp: dict[str, torch.Tensor] = {}
    h2 = mlp.register_forward_pre_hook(lambda m, inputs: seen_mlp.__setitem__("x", inputs[0].detach().float()))
    try:
        toy.get_activations(TOY_PROMPT)
        clean_inv = _inverse_rms(norm, seen_norm["x"])
        with frozen_norm_denominators(toy, TOY_PROMPT):
            toy.feature_intervention(TOY_PROMPT, intervention, apply_activation_function=False)
    finally:
        h1.remove()
        h2.remove()
    expected = seen_norm["x"] * clean_inv * (1.0 + norm.weight.float())
    unfrozen = seen_norm["x"] * _inverse_rms(norm, seen_norm["x"]) * (1.0 + norm.weight.float())
    assert not torch.allclose(expected, unfrozen, rtol=1e-3)  # the two readings differ, so the check can fail
    torch.testing.assert_close(seen_mlp["x"], expected, atol=1e-5, rtol=1e-5)


def test_a_norm_class_it_cannot_rescale_is_refused_by_name():
    class OddNorm(nn.Module):
        def forward(self, x):
            return x

    block = nn.Module()
    block.odd_norm = OddNorm()
    model = types.SimpleNamespace(pre_logit_location=types.SimpleNamespace(layers=nn.ModuleList([block]), norm=None))
    with pytest.raises(ValueError, match="OddNorm is a normalization module"):
        residual_norm_modules(model)


def test_a_model_without_residual_norms_is_refused():
    model = types.SimpleNamespace(pre_logit_location=types.SimpleNamespace(layers=nn.ModuleList([nn.Linear(2, 2)])))
    with pytest.raises(ValueError, match="found no RMSNorm or LayerNorm"):
        residual_norm_modules(model)


def test_the_context_is_null_unless_freeze_norms_is_set(toy):
    backend = CircuitTracerAnalysisBackend()
    module = types.SimpleNamespace(replacement_model=toy)
    with backend.feature_intervention_context(module, TOY_PROMPT, {"freeze_norms": False}) as held:
        assert held is None
    with backend.feature_intervention_context(module, TOY_PROMPT, {"freeze_norms": True}) as held:
        assert len(held) == 4 * 3 + 1


def test_the_far_upstream_share_is_computed_from_the_target_row(toy):
    """Checked against an independent computation from the graph, for a read at block 2 (features at layers 0 and 1
    count as far upstream) beside an unsited target that must not be reported."""
    from circuit_tracer import attribute
    from circuit_tracer.attribution.targets import CustomTarget

    torch.manual_seed(0)
    v = torch.randn(toy.cfg.d_model)
    targets = [CustomTarget("read@2", 1.0, v / v.norm(), layer=2), CustomTarget("final", 1.0, v / v.norm())]
    graph = attribute(TOY_PROMPT, toy, attribution_targets=targets)
    (entry,) = CircuitTracerAnalysisBackend().layer_local_target_provenance(graph, targets)
    assert entry["target"] == "read@2" and entry["layer"] == 2 and entry["position"] is None

    row = graph.adjacency_matrix[-2].abs()
    layers = graph.active_features[graph.selected_features][:, 0]
    n_feat = len(layers)
    n_src = n_feat + toy.cfg.n_layers * graph.n_pos + graph.n_pos
    expected = float(row[:n_feat][layers <= 1].sum() / row[:n_src].sum())
    assert 0.0 < expected < 1.0  # the check can fail in both directions
    assert entry["far_upstream_feature_share"] == pytest.approx(expected)


def test_no_share_is_reported_without_a_sited_target(toy):
    backend = CircuitTracerAnalysisBackend()
    assert backend.layer_local_target_provenance(object(), None) == []
    assert backend.layer_local_target_provenance(object(), torch.tensor([1, 2])) == []
