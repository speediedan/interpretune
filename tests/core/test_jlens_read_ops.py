"""The J-lens read family: readout, concept probe, sparse inventory.

Synthetic lenses on CPU. Each case is chosen so the right answer is known independently of the
implementation: the identity lens must reduce to a logit lens, a concept direction must be perfectly
aligned with itself, and a vector built from known atoms must be recovered as those atoms.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.bundled.jlens import jlens_ops
from interpretune.analysis.optools import JLensArtifact
from tests.runif import RunIf

VOCAB, D, LAYERS = 24, 8, (0, 4, 8)
KEY = "blocks.4.hook_in"


class _Tok:
    def decode(self, ids):
        return f"<{ids[0]}>"


class _Norm(torch.nn.Module):  # class name drives kind detection
    def __init__(self, scale):
        super().__init__()
        self.weight = torch.nn.Parameter(scale)


class _RMSNorm(_Norm): ...


class _LayerNorm(_Norm): ...


def _module(norm_cls=_RMSNorm, scale=None, model_type="llama"):
    torch.manual_seed(0)
    inner = type("Inner", (), {})()
    inner.norm = norm_cls(torch.ones(D) if scale is None else scale)
    model = type("Model", (), {})()
    model.config = type("Cfg", (), {"model_type": model_type})()
    model.lm_head = type("Head", (), {})()
    model.lm_head.weight = torch.randn(VOCAB, D)
    model.model = inner
    model.tokenizer = _Tok()
    module = type("Module", (), {})()
    module.model = model
    return module


@pytest.fixture
def synthetic_lens(monkeypatch):
    """A lens whose J is the identity, so the readout has a known closed form."""

    def _resolve(module, **kwargs):
        return JLensArtifact(
            j_by_layer={ell: torch.eye(D) for ell in LAYERS},
            source_layers=list(LAYERS),
            d_model=D,
            repo_id="synthetic",
            path="synthetic/jlens/c/synthetic_jacobian_lens.pt",
            hf_model_name="org/synthetic",
            provenance={"results": {"prompts_fitted": 277}},
        )

    monkeypatch.setattr(jlens_ops, "resolve_jlens", _resolve)


def _batch(activations):
    return AnalysisBatch(cache={KEY: activations}, jlens_layer=4, jlens_cache_key=KEY)


class TestJLensRead:
    def test_an_identity_lens_reduces_to_the_logit_lens(self, synthetic_lens):
        """J = I is the degenerate case, and the readout must land exactly on the logit lens there."""
        module = _module()
        acts = torch.randn(2, 5, D)
        out = jlens_ops.jlens_read_impl(module, _batch(acts), None, 0, jlens_top_k=VOCAB)

        expected = acts[:, -1, :] @ module.model.lm_head.weight.T
        # top-k over the full vocabulary is the sorted logits, so compare against those directly
        torch.testing.assert_close(
            out["jlens_top_token_scores"][:, 0, :],
            expected.sort(dim=-1, descending=True).values,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(out["jlens_top_token_ids"][:, 0, :], expected.sort(dim=-1, descending=True).indices)

    def test_top_token_matches_the_largest_logit(self, synthetic_lens):
        module = _module()
        acts = torch.randn(3, 4, D)
        out = jlens_ops.jlens_read_impl(module, _batch(acts), None, 0, jlens_top_k=1)
        expected = (acts[:, -1, :] @ module.model.lm_head.weight.T).argmax(dim=-1)
        torch.testing.assert_close(out["jlens_top_token_ids"][:, 0, 0], expected)

    def test_rms_scale_changes_magnitudes_and_not_within_position_ranking(self, synthetic_lens):
        """The flag exists because these two answers differ; asserting both is what pins it."""
        module = _module()
        acts = torch.randn(2, 3, D)
        without = jlens_ops.jlens_read_impl(module, _batch(acts), None, 0, jlens_top_k=VOCAB)
        with_scale = jlens_ops.jlens_read_impl(
            module, _batch(acts), None, 0, jlens_top_k=VOCAB, jlens_include_rms_scale=True
        )
        torch.testing.assert_close(without["jlens_top_token_ids"], with_scale["jlens_top_token_ids"])
        assert not torch.allclose(without["jlens_top_token_scores"], with_scale["jlens_top_token_scores"]), (
            "if magnitudes were unchanged the flag would be inert and cross-position comparison unaffected"
        )

    def test_a_layer_the_lens_was_not_fit_at_is_refused_not_interpolated(self, synthetic_lens):
        with pytest.raises(ValueError, match="fit at layers"):
            jlens_ops.jlens_read_impl(_module(), _batch(torch.randn(1, 2, D)), None, 0, jlens_layer=5)

    def test_positions_select_what_is_read(self, synthetic_lens):
        module = _module()
        acts = torch.randn(1, 6, D)
        out = jlens_ops.jlens_read_impl(module, _batch(acts), None, 0, jlens_positions=[0, 2], jlens_top_k=1)
        assert out["jlens_top_token_ids"].shape == (1, 2, 1)
        expected = (acts[0, [0, 2], :] @ module.model.lm_head.weight.T).argmax(dim=-1)
        torch.testing.assert_close(out["jlens_top_token_ids"][0, :, 0], expected)

    def test_an_out_of_range_position_is_refused(self, synthetic_lens):
        with pytest.raises(ValueError, match="out of range"):
            jlens_ops.jlens_read_impl(_module(), _batch(torch.randn(1, 3, D)), None, 0, jlens_positions=[9])

    def test_a_width_mismatch_is_an_error_rather_than_a_broadcast(self, synthetic_lens):
        with pytest.raises(ValueError, match="does not match lens d_model"):
            jlens_ops.jlens_read_impl(_module(), _batch(torch.randn(1, 2, D + 3)), None, 0)

    def test_fit_provenance_travels_with_the_result(self, synthetic_lens):
        out = jlens_ops.jlens_read_impl(_module(), _batch(torch.randn(1, 2, D)), None, 0)
        assert out["jlens_provenance"]["results"]["prompts_fitted"] == 277


class TestConceptProbe:
    def test_a_concept_direction_is_perfectly_aligned_with_itself(self, synthetic_lens):
        """With J = I the direction for token c is the folded unembed row, so cosine must be exactly 1."""
        module = _module()
        row = module.model.lm_head.weight[7]
        acts = row.reshape(1, 1, D)
        out = jlens_ops.jlens_concept_probe_impl(
            module, _batch(acts), None, 0, jlens_concept_token_ids=[7], jlens_positions=[0]
        )
        torch.testing.assert_close(out["jlens_concept_cosine"][0, 0, 0], torch.tensor(1.0), atol=1e-5, rtol=0)

    def test_cosine_is_reported_per_concept(self, synthetic_lens):
        out = jlens_ops.jlens_concept_probe_impl(
            _module(), _batch(torch.randn(2, 3, D)), None, 0, jlens_concept_token_ids=[1, 5, 9]
        )
        assert out["jlens_concept_cosine"].shape == (2, 1, 3)
        assert out["jlens_concept_cosine"].abs().max() <= 1.0 + 1e-5

    def test_layernorm_centering_reaches_the_probe(self, synthetic_lens):
        """The probe must inherit the seam's per-kind rule rather than folding the scale itself."""
        scale = torch.linspace(0.4, 2.2, D)
        ln_out = jlens_ops.jlens_concept_probe_impl(
            _module(_LayerNorm, scale), _batch(torch.randn(1, 1, D)), None, 0, jlens_concept_token_ids=[3]
        )
        rms_out = jlens_ops.jlens_concept_probe_impl(
            _module(_RMSNorm, scale), _batch(torch.randn(1, 1, D)), None, 0, jlens_concept_token_ids=[3]
        )
        assert not torch.allclose(ln_out["jlens_concept_cosine"], rms_out["jlens_concept_cosine"]), (
            "identical results would mean the centering never reached the direction"
        )

    def test_missing_concept_tokens_is_an_error(self, synthetic_lens):
        with pytest.raises(ValueError, match="requires jlens_concept_token_ids"):
            jlens_ops.jlens_concept_probe_impl(_module(), _batch(torch.randn(1, 1, D)), None, 0)


class TestSparseInventory:
    def test_a_vector_built_from_known_atoms_recovers_them(self, synthetic_lens):
        """The only case where the right answer is known independently of the solver."""
        module = _module()
        rows = module.model.lm_head.weight  # J = I, uniform scale, so atoms are the unembed rows
        target = 3.0 * rows[2] + 1.5 * rows[11]
        out = jlens_ops.jlens_sparse_inventory_impl(
            module, _batch(target.reshape(1, 1, D)), None, 0, jlens_inventory_k=4, jlens_positions=[0]
        )
        chosen = out["jlens_inventory_token_ids"][0]
        coefficients = dict(zip(chosen, out["jlens_inventory_coefficients"][0]))
        assert {2, 11} <= set(chosen)
        # Tight on purpose. The solver is exact when every coefficient is nonnegative, which it is
        # here, so a loose bound would pass on an under-converged solver and hide the regression.
        assert out["jlens_inventory_residual_ratio"][0] < 1e-5
        assert coefficients[2] == pytest.approx(3.0, abs=1e-4)
        assert coefficients[11] == pytest.approx(1.5, abs=1e-4)

    def test_the_residual_ratio_is_reported_and_is_not_trivially_zero(self, synthetic_lens):
        """A decomposition quoted without its residual cannot distinguish a good fit from a poor one."""
        out = jlens_ops.jlens_sparse_inventory_impl(
            _module(), _batch(torch.randn(1, 1, D)), None, 0, jlens_inventory_k=1, jlens_positions=[0]
        )
        assert 0.0 < float(out["jlens_inventory_residual_ratio"][0]) <= 1.0

    def test_more_atoms_never_fit_worse(self, synthetic_lens):
        module = _module()
        acts = torch.randn(1, 1, D)
        ratios = [
            float(
                jlens_ops.jlens_sparse_inventory_impl(
                    module, _batch(acts), None, 0, jlens_inventory_k=k, jlens_positions=[0]
                )["jlens_inventory_residual_ratio"][0]
            )
            for k in (1, 3, 6)
        ]
        assert ratios[1] <= ratios[0] + 1e-4 and ratios[2] <= ratios[1] + 1e-4, ratios

    def test_coefficients_are_nonnegative(self, synthetic_lens):
        out = jlens_ops.jlens_sparse_inventory_impl(
            _module(), _batch(torch.randn(1, 1, D)), None, 0, jlens_inventory_k=5, jlens_positions=[0]
        )
        assert all(c >= 0.0 for c in out["jlens_inventory_coefficients"][0])


class TestCrossBackendReadoutAgreement:
    """The readout must not depend on which backend resolved the unembed and the final norm.

    This exercises the seam's TransformerLens row against a real TL model rather than a stub, which is
    the gap the folding investigation left open: the row was asserted by construction and never
    measured. Standalone-marked at the METHOD level, since class-level marks are invisible to the
    collection filter and a real model load is too heavy for the default CPU lane.
    """

    @RunIf(standalone=True)
    def test_tl_and_hf_resolved_readouts_agree_on_gpt2(self):
        from transformer_lens import HookedTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from interpretune.analysis.optools import resolve_unembed_and_norm_scale

        torch.manual_seed(3)
        # NO-processing load, and it is load-bearing here for the same reason it is in the patch
        # tests: the default from_pretrained folds LayerNorm and centers weights, which changes both
        # the unembed and the residual basis. Comparing a processed TL model against HF would measure
        # that transformation rather than the seam, and would fail while nothing was wrong.
        tl_model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")
        hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
        hf_model.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        tl_module = type("M", (), {})()
        tl_module.model = tl_model
        tl_model.tokenizer = tl_model.tokenizer or AutoTokenizer.from_pretrained("gpt2")
        hf_module = type("M", (), {})()
        hf_module.model = hf_model

        d_model = resolve_unembed_and_norm_scale(hf_module).w_u.shape[1]
        lens = {6: torch.randn(d_model, d_model) * (1.0 / d_model**0.5)}
        activations = torch.randn(1, 3, d_model)

        def _read(module):
            artifact = JLensArtifact(
                j_by_layer=lens,
                source_layers=[6],
                d_model=d_model,
                repo_id="synthetic",
                path="synthetic",
                hf_model_name="openai-community/gpt2",
                provenance={},
            )
            batch = AnalysisBatch(cache={"blocks.6.hook_in": activations})
            original = jlens_ops.resolve_jlens
            jlens_ops.resolve_jlens = lambda m, **k: artifact
            try:
                return jlens_ops.jlens_read_impl(
                    module, batch, None, 0, jlens_layer=6, jlens_cache_key="blocks.6.hook_in", jlens_top_k=20
                )
            finally:
                jlens_ops.resolve_jlens = original

        tl_out, hf_out = _read(tl_module), _read(hf_module)
        torch.testing.assert_close(tl_out["jlens_top_token_ids"], hf_out["jlens_top_token_ids"])
        torch.testing.assert_close(
            tl_out["jlens_top_token_scores"], hf_out["jlens_top_token_scores"], rtol=1e-4, atol=1e-4
        )


class TestRealLensSmoke:
    """One end-to-end pass against a published lens, so resolution and the readout are exercised together."""

    @RunIf(min_cuda_gpus=1)
    def test_a_published_gpt2_lens_resolves_and_reads(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from interpretune.analysis.optools import resolve_jlens

        model = AutoModelForCausalLM.from_pretrained("gpt2").cuda().eval()
        model.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        module = type("M", (), {})()
        module.model = model

        artifact = resolve_jlens(module)
        # The published gpt2 lens lives under `gpt2-small/` with a stem of `gpt2_...`, so resolving it
        # at all is the discovery path working against the real repository rather than a recording.
        assert artifact.hf_model_name == "openai-community/gpt2"
        assert artifact.path.startswith("gpt2-small/")
        assert artifact.d_model == model.config.n_embd
        assert artifact.provenance.get("results", {}).get("prompts_fitted", 0) > 0

        layer = jlens_ops.jlens_layer_for_percentile(artifact, 0.85)
        activations = torch.randn(1, 4, artifact.d_model)
        out = jlens_ops.jlens_read_impl(
            module,
            AnalysisBatch(cache={f"blocks.{layer}.hook_in": activations}),
            None,
            0,
            jlens_layer=layer,
            jlens_top_k=5,
        )
        assert out["jlens_top_token_ids"].shape == (1, 1, 5)
        assert len(out["jlens_top_token_strings"][0][0]) == 5
