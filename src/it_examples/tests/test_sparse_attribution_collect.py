"""Checkpoint-report source conversion and measured basis identity through the real public head op."""

import json

import pytest
import torch
from transformers import BatchEncoding

from interpretune.analysis.ops.base import AnalysisBatch
from tests.core.circuit_tracer_toy import TOY_PROMPT, only_token_zero_special, tiny_gemma2_replacement_model
from tests.core.test_circuit_tracer_attention_attribution import FROZEN, _ct_module


@pytest.fixture(scope="module")
def toy():
    model = tiny_gemma2_replacement_model(n_layers=3)
    with only_token_zero_special(model):
        yield model


class TestMeasuredBasisIdentity:
    @pytest.fixture
    def basis_toy(self):
        # Structural mutations below must not leave the shared model's Envoy children pointing at replaced modules.
        return tiny_gemma2_replacement_model(n_layers=2)

    @pytest.mark.parametrize("field", ["decoder", "threshold", "bandwidth", "zero-positions", "hook"])
    def test_actual_basis_changes_cannot_keep_the_same_identity(self, basis_toy, monkeypatch, field):
        from it_examples.utils.sparse_attribution_collect import transcoder_basis_id

        before = transcoder_basis_id(basis_toy)
        basis = basis_toy.transcoders._module
        tc = basis[0]
        if field in ("decoder", "threshold"):
            tensor = tc.W_dec if field == "decoder" else tc.activation_function.threshold
            saved = tensor.detach().clone()
            try:
                with torch.no_grad():
                    tensor.add_(0.25)
                assert transcoder_basis_id(basis_toy) != before
            finally:
                with torch.no_grad():
                    tensor.copy_(saved)
        else:
            obj, attribute, value = {
                "bandwidth": (tc.activation_function, "bandwidth", 0.25),
                "zero-positions": (basis_toy, "zero_positions", slice(0, 2)),
                "hook": (basis, "feature_input_hook", "another-hook"),
            }[field]
            monkeypatch.setattr(obj, attribute, value)
            assert transcoder_basis_id(basis_toy) != before

    def test_an_unrecorded_activation_kind_is_refused(self, basis_toy, monkeypatch):
        from it_examples.utils.sparse_attribution_collect import transcoder_basis_id

        monkeypatch.setattr(basis_toy.transcoders._module[0], "activation_function", torch.nn.ReLU())
        with pytest.raises(ValueError, match="supported JumpReLU"):
            transcoder_basis_id(basis_toy)


class TestExampleHeadReport:
    def test_a_logit_gap_report_keeps_the_decision_separate_from_source_edges(self, toy):
        import interpretune as it
        import interpretune.analysis
        from circuit_tracer.attribution.targets import CustomTarget

        from it_examples.utils.sparse_attribution_collect import head_loading_row
        from it_examples.utils.sparse_attribution_report import checkpoint_attribution_report

        positive, negative = 3, 8
        with torch.no_grad():
            logits = toy._module(input_ids=TOY_PROMPT[None]).logits[0, -1]
            decision = float(logits[positive] - logits[negative])
        unembed = toy._module.get_output_embeddings().weight.detach()
        target = CustomTarget("logit_gap", 1.0, unembed[positive] - unembed[negative])
        raw = it.head_loadings(
            _ct_module(toy), AnalysisBatch(), BatchEncoding(), 0, prompt=TOY_PROMPT, head_target=target
        )
        row = head_loading_row(
            raw, row_id="validation-1", checkpoint_sha256="a" * 64, basis_id="toy-transcoders",
            decision=decision, answer_position=len(TOY_PROMPT) - 1,
        )  # fmt: skip
        summary = checkpoint_attribution_report(
            [row], checkpoint_sha256="a" * 64, basis_id="toy-transcoders", model_id="tiny-gemma2",
            model_revision="random-test-model", target_token_ids=(positive, negative),
            measurement_dtype="float32",
        )  # fmt: skip
        assert row.decision == decision
        assert row.bias is None
        assert summary["attribution_regime"] == FROZEN
        assert summary["source_shares"]["bias"] is None
        assert summary["source_shares"]["unassigned"] == pytest.approx(1 - row.edge_total / decision, abs=1e-5)
        assert row.edge_total != pytest.approx(decision, abs=1e-2)
        assert torch.tensor(summary["head_shares"]) == pytest.approx(raw.head_loading_head_edges / decision, abs=1e-5)
        assert row.feature_nodes == [tuple(node) for node in raw.head_loading_feature_nodes.tolist()]

    @pytest.mark.parametrize("regime", [{}, {"freeze_attention": True, "freeze_norms": "false"}])
    def test_a_row_with_incomplete_or_ambiguous_regime_is_refused(self, regime):
        from it_examples.utils.sparse_attribution_collect import head_loading_row

        with pytest.raises(ValueError, match="regime must"):
            head_loading_row(
                {"head_loading_regime_json": json.dumps(regime)},
                row_id="validation-1", checkpoint_sha256="a" * 64, basis_id="toy-transcoders",
                decision=1.0, answer_position=1,
            )  # fmt: skip
