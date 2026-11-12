"""A recipe's feature ``add`` and ``scale`` on the analysis-backend surface, computed from a clean capture, through
a real circuit-tracer replacement model (#965)."""

from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("venngram", reason="the venngram extra is not installed")
pytest.importorskip("circuit_tracer", reason="the circuit_tracer extra is not installed")

from interpretune.adapters.circuit_tracer.backends import CircuitTracerAnalysisBackend  # noqa: E402
from interpretune.analysis.backends.interventions import apply_intervention  # noqa: E402
from interpretune.analysis.venngram import (  # noqa: E402
    load_recipe,
    recipe_feature_interventions,
    recipe_interventions,
    require_supported,
    venngram_feature_support,
)
from tests.core.circuit_tracer_toy import TOY_PROMPT, only_token_zero_special, tiny_gemma2_replacement_model  # noqa: E402

LAYER = 1


@pytest.fixture(scope="module")
def toy():
    model = tiny_gemma2_replacement_model(n_layers=3)
    with only_token_zero_special(model):
        yield model


def _recipe(op: str, dose: float, feature: int, tokens):
    return load_recipe(
        {
            "schema_version": 2,
            "kind": "recipe",
            "name": "relative-feature-treatment",
            "model": {"id": "m", "revision": "r1"},
            "book": {
                "schema_version": 1,
                "name": "features",
                "portability": "checkpoint",
                "applies_to": {"models": [{"id": "m", "revision": "r1"}]},
                "sets": {"top": {"features": str(feature), "layers": [LAYER]}},
            },
            "stages": [
                {
                    "site": "hook_resid_post",
                    "tokens": tokens,
                    "ops": [{"op": op, "target": {"set": "top"}, "value": {"dose": dose}}],
                }
            ],
        }
    ).recipe


def _active_feature(clean: torch.Tensor) -> tuple[int, int]:
    """A (position, feature) active at ``LAYER``, so a scale moves it and the check is not trivially zero."""
    position, feature = (int(i) for i in torch.nonzero(clean[LAYER, 1:] > 0)[0])
    return position + 1, feature


def _intervened_logits(toy, op: str, dose: float) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    backend = CircuitTracerAnalysisBackend()
    clean = backend.capture_feature_activations(types.SimpleNamespace(replacement_model=toy), TOY_PROMPT)
    position, feature = _active_feature(clean)
    recipe = _recipe(op, dose, feature, {"positions": [position]})
    support = backend.feature_intervention_support
    require_supported(recipe, venngram_feature_support(support), analysis_backend=True)
    (intervention,) = recipe_feature_interventions(recipe, support=support, clean_activations=clean)
    clean_logits, _ = toy.get_activations(TOY_PROMPT)
    logits, _ = toy.feature_intervention(TOY_PROMPT, [intervention])
    return clean_logits, logits, (clean, position, feature, intervention)


@pytest.mark.parametrize("op, identity", [("add", 0.0), ("scale", 1.0)])
def test_the_identity_dose_leaves_the_model_unchanged(toy, op, identity):
    """The tuple sets a value, so an `add` of 0 or a `scale` of 1 is the identity only if the capture reads the
    very activation the intervention writes (same positions, same tokenization, same side of the activation
    function)."""
    clean_logits, logits, (clean, position, feature, intervention) = _intervened_logits(toy, op, identity)
    assert intervention == (LAYER, position, feature, pytest.approx(float(clean[LAYER, position, feature])))
    torch.testing.assert_close(logits, clean_logits, atol=1e-5, rtol=1e-5)


def test_a_misaligned_capture_breaks_the_identity(toy):
    """Negative control for the identity test: a capture one position off gives an `add` of 0 another token's value,
    and the logits move, so the identity above is evidence of alignment rather than of an intervention that does
    nothing."""
    backend = CircuitTracerAnalysisBackend()
    clean = backend.capture_feature_activations(types.SimpleNamespace(replacement_model=toy), TOY_PROMPT)
    position, feature = _active_feature(clean)
    shifted = torch.roll(clean, shifts=1, dims=1)
    assert float(shifted[LAYER, position, feature]) != pytest.approx(float(clean[LAYER, position, feature]))
    (intervention,) = recipe_feature_interventions(
        _recipe("add", 0.0, feature, {"positions": [position]}),
        support=backend.feature_intervention_support,
        clean_activations=shifted,
    )
    clean_logits, _ = toy.get_activations(TOY_PROMPT)
    logits, _ = toy.feature_intervention(TOY_PROMPT, [intervention])
    assert not torch.allclose(logits, clean_logits, atol=1e-4)


@pytest.mark.parametrize("op, dose", [("add", 0.75), ("scale", 3.0)])
def test_a_non_identity_dose_moves_the_model(toy, op, dose):
    """Positive control for the identity test: the same path with a real dose changes the logits."""
    clean_logits, logits, (clean, position, feature, intervention) = _intervened_logits(toy, op, dose)
    a = float(clean[LAYER, position, feature])
    assert intervention[3] == pytest.approx(a + dose if op == "add" else a * dose)
    assert not torch.allclose(logits, clean_logits, atol=1e-4), "the intervention never moved the logits"


@pytest.mark.parametrize("op, dose", [("add", 0.75), ("scale", 3.0)])
def test_both_surfaces_give_one_value_for_the_same_op(toy, op, dose):
    """The model-backend surface applies the op at the latent sub-hook to the activation it meets; on a clean pass
    that is the clean activation, so its result at the named position is the analysis surface's tuple value."""
    backend = CircuitTracerAnalysisBackend()
    clean = backend.capture_feature_activations(types.SimpleNamespace(replacement_model=toy), TOY_PROMPT)
    position, feature = _active_feature(clean)
    width = clean.shape[-1]
    ((_, _, _, value),) = recipe_feature_interventions(
        _recipe(op, dose, feature, {"positions": [position]}),
        support=backend.feature_intervention_support,
        clean_activations=clean,
    )
    (spec,) = recipe_interventions(
        _recipe(op, dose, feature, "all"), site_widths={"hook_resid_post.hook_sae_acts_post": width}
    )[f"blocks.{LAYER}.hook_resid_post.hook_sae_acts_post"]
    applied = apply_intervention(clean[LAYER].unsqueeze(0), spec, last_pos=clean.shape[1] - 1)
    assert float(applied[0, position, feature]) == pytest.approx(value)


def test_a_backend_without_a_replacement_model_cannot_capture():
    with pytest.raises(ValueError, match="requires module.replacement_model"):
        CircuitTracerAnalysisBackend().capture_feature_activations(types.SimpleNamespace(), TOY_PROMPT)
