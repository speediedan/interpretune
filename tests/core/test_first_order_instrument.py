"""Tests for the first-order intervention-validation instrument (#539).

The instrument predicts a metric change as grad_h(m)^T @ delta_h and reports it against the measured change, per basis.
Two properties make the suite trustworthy rather than merely green: agreement is asserted TIGHT at small alpha (the
linear regime), and a positive control at large alpha asserts the instrument demonstrably breaks down outside it.
Without the second, an instrument that always reported agreement would pass.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.backends.interventions import InterventionSpec
from interpretune.analysis.first_order import FirstOrderReport, first_order_check

N_LAYERS = 3
D_MODEL = 32
HOOK_LAYER = 1
PROMPT_IDS = [3, 17, 9, 25, 11]
TOKEN_A = 5
TOKEN_B = 7


@pytest.fixture(scope="module")
def tiny_gpt2_dir_539(tmp_path_factory):
    """A seeded random tiny GPT-2 saved to disk (mirrors the validation-test fixture)."""
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    torch.manual_seed(7)
    config = GPT2Config(n_layer=N_LAYERS, n_head=4, n_embd=D_MODEL, vocab_size=64, n_positions=32)
    model = GPT2LMHeadModel(config).eval()
    path = tmp_path_factory.mktemp("tiny_gpt2_539")
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def tiny_gpt2_539(tiny_gpt2_dir_539):
    from transformers import GPT2LMHeadModel

    return GPT2LMHeadModel.from_pretrained(tiny_gpt2_dir_539).eval()


def _gap_metric(logits: torch.Tensor) -> torch.Tensor:
    return logits[0, -1, TOKEN_A] - logits[0, -1, TOKEN_B]


def _add_spec(alpha: float) -> InterventionSpec:
    direction = torch.randn(D_MODEL)
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
    return InterventionSpec(
        intervention_tensor=(direction * alpha).float(),
        mode="add",
        scale_factor=1.0,
        position_scope="last_token",
    )


class TestFirstOrderAgreement:
    def test_small_alpha_prediction_matches_measured(self, tiny_gpt2_539) -> None:
        report = first_order_check(
            tiny_gpt2_539,
            tiny_gpt2_539.transformer.h[HOOK_LAYER],
            torch.tensor([PROMPT_IDS]),
            _add_spec(1e-3),
            _gap_metric,
            basis="embed",
            last_pos=len(PROMPT_IDS) - 1,
        )
        assert isinstance(report, FirstOrderReport)
        assert report.basis == "embed"
        assert report.mode == "add"
        assert abs(report.predicted) > 1e-9  # non-vacuous: the metric actually moves
        assert abs(report.residual) <= 0.1 * abs(report.predicted)

    def test_large_alpha_breaks_agreement_positive_control(self, tiny_gpt2_539) -> None:
        report = first_order_check(
            tiny_gpt2_539,
            tiny_gpt2_539.transformer.h[HOOK_LAYER],
            torch.tensor([PROMPT_IDS]),
            _add_spec(5.0),
            _gap_metric,
            basis="embed",
            last_pos=len(PROMPT_IDS) - 1,
        )
        assert abs(report.predicted) > 1e-9
        assert abs(report.residual) > abs(report.measured)

    def test_identity_clamp_reports_zero_change(self, tiny_gpt2_539) -> None:
        n = D_MODEL
        eye = torch.eye(n)
        spec = InterventionSpec(
            intervention_tensor=eye,
            mode="clamp",
            scale_factor=1.0,
            position_scope="last_token",
            clamp_min=-1e9,
            clamp_max=1e9,
        )
        report = first_order_check(
            tiny_gpt2_539,
            tiny_gpt2_539.transformer.h[HOOK_LAYER],
            torch.tensor([PROMPT_IDS]),
            spec,
            _gap_metric,
            basis="jlens_paper",
            last_pos=len(PROMPT_IDS) - 1,
        )
        assert report.basis == "jlens_paper"
        assert report.predicted == pytest.approx(0.0, abs=1e-9)
        assert report.measured == pytest.approx(0.0, abs=1e-6)
