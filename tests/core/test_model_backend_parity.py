"""Model Backend Parity Tests: TransformerBridge ↔ NNsight.

Validates that analysis operations produce equivalent results regardless of
whether the TransformerBridge or NNsight model backend is used for SAE analysis.

**Parity validation (TransformerBridge ↔ NNsight on GPT-2):**

    Both backends run the HF model's native forward pass; differences arise
    only from hook interception mechanism (PyTorch hooks vs thread interleaving).
    TransformerBridge serves as the source of truth (canonical HF behavior).
    Tolerance: rtol=1e-4, atol=1e-4 (gradients: rtol=5e-3, atol=5e-3).

**Note on HookedTransformer (HT) parity:**
    Prior Tier 2 tests compared Bridge ↔ HookedTransformer with bounded
    divergence tolerances.  These were removed because HT and Bridge use
    fundamentally different forward computation paths — HT reimplements
    attention with multiplicative boolean masking and zeroed pad-position
    embeddings, while Bridge wraps HF's native additive float masking with
    real W_pos values.  On 178-token padded GPT-2 RTE inputs, answer_logits
    diverge by ~85 units and alive-latent Jaccard overlap drops to 0%.
    HT functional correctness is validated separately by existing
    HookedTransformer-specific tests in the test suite; see
    docs/ht_bridge_parity_behavior.md for the full analysis.

Fixture mapping (NNsight → Bridge):
    sl_ns_gpt2_logit_diffs_base          → sl_br_gpt2_logit_diffs_base
    sl_ns_gpt2_logit_diffs_sae           → sl_br_gpt2_logit_diffs_sae
    sl_ns_gpt2_logit_diffs_attr_grad     → sl_br_gpt2_logit_diffs_attr_grad
    sl_ns_gpt2_logit_diffs_attr_ablation → sl_br_gpt2_logit_diffs_attr_ablation
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import torch
from torch.testing import assert_close

import interpretune as it
from interpretune.analysis.core import (
    AnalysisStore,
    base_vs_sae_logit_diffs,
    compute_correct,
)
from interpretune.analysis.ops.base import AnalysisBatch
from tests.analysis_resource_utils import (
    AnalysisFixtureSpec,
    AnalysisExtractionMixin,
)

# ---------------------------------------------------------------------------
# Fixture key constants
# ---------------------------------------------------------------------------
_NS_BASE = "get_analysis_session__sl_ns_gpt2_logit_diffs_base__initonly_runanalysis"
_NS_SAE = "get_analysis_session__sl_ns_gpt2_logit_diffs_sae__initonly_runanalysis"
_NS_GRAD = "get_analysis_session__sl_ns_gpt2_logit_diffs_attr_grad__initonly_runanalysis"
_NS_ABLATION = "get_analysis_session__sl_ns_gpt2_logit_diffs_attr_ablation__initonly_runanalysis"

_BR_BASE = "get_analysis_session__sl_br_gpt2_logit_diffs_base__initonly_runanalysis"
_BR_SAE = "get_analysis_session__sl_br_gpt2_logit_diffs_sae__initonly_runanalysis"
_BR_GRAD = "get_analysis_session__sl_br_gpt2_logit_diffs_attr_grad__initonly_runanalysis"
_BR_ABLATION = "get_analysis_session__sl_br_gpt2_logit_diffs_attr_ablation__initonly_runanalysis"


def _compare_tensor_lists(
    ref_list: list[torch.Tensor],
    cmp_list: list[torch.Tensor],
    *,
    label: str,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    """Compare two lists of tensors element-wise with assert_close."""
    assert len(ref_list) == len(cmp_list), f"{label}: batch count mismatch ({len(ref_list)} vs {len(cmp_list)})"
    for i, (t_ref, t_cmp) in enumerate(zip(ref_list, cmp_list)):
        assert_close(t_ref, t_cmp, rtol=rtol, atol=atol, msg=f"{label} batch {i}")


def _compare_dict_tensor_lists(
    ref_list: list[dict[str, torch.Tensor]],
    cmp_list: list[dict[str, torch.Tensor]],
    *,
    label: str,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    """Compare two lists of dicts mapping hook names → Tensors."""
    assert len(ref_list) == len(cmp_list), f"{label}: batch count mismatch ({len(ref_list)} vs {len(cmp_list)})"
    for i, (d_ref, d_cmp) in enumerate(zip(ref_list, cmp_list)):
        assert set(d_ref.keys()) == set(d_cmp.keys()), (
            f"{label} batch {i}: hook key mismatch ({d_ref.keys()} vs {d_cmp.keys()})"
        )
        for hook_name in d_ref:
            assert_close(
                d_ref[hook_name],
                d_cmp[hook_name],
                rtol=rtol,
                atol=atol,
                msg=f"{label} batch {i} hook {hook_name}",
            )


class TestLogitDiffsBaseBackendParity(AnalysisExtractionMixin):
    """logit_diffs_base: TransformerBridge ↔ NNsight forward pass (no SAE).

    Simplest parity check — both backends run the HF model's native forward
    pass and compute logit diffs.  Bridge (source of truth) and NNsight
    should produce very close results.
    """

    _analysis_fixture_specs: ClassVar[dict[str, AnalysisFixtureSpec]] = {
        "br": AnalysisFixtureSpec(fixture_key=_BR_BASE),
        "ns": AnalysisFixtureSpec(fixture_key=_NS_BASE),
    }

    def test_logit_diffs_match(self, request):
        """Core logit_diffs values should match across backends."""
        extracted = cast(dict[str, AnalysisStore], self.extract_values(request))
        br_store = extracted["br"]
        ns_store = extracted["ns"]
        _compare_tensor_lists(br_store.logit_diffs, ns_store.logit_diffs, label="logit_diffs")

    def test_answer_logits_match(self, request):
        """Answer logits should match across backends."""
        extracted = cast(dict[str, AnalysisStore], self.extract_values(request))
        br_store = extracted["br"]
        ns_store = extracted["ns"]
        _compare_tensor_lists(br_store.answer_logits, ns_store.answer_logits, label="answer_logits")

    def test_predictions_match(self, request):
        """Model predictions should be identical across backends."""
        extracted = cast(dict[str, AnalysisStore], self.extract_values(request))
        br_store = extracted["br"]
        ns_store = extracted["ns"]
        _compare_tensor_lists(br_store.preds, ns_store.preds, label="preds", rtol=0, atol=0)

    def test_labels_match(self, request):
        """Ground truth labels should be identical (dataset-level sanity check)."""
        extracted = cast(dict[str, AnalysisStore], self.extract_values(request))
        br_store = extracted["br"]
        ns_store = extracted["ns"]
        _compare_tensor_lists(br_store.orig_labels, ns_store.orig_labels, label="orig_labels", rtol=0, atol=0)
        _compare_tensor_lists(br_store.label_ids, ns_store.label_ids, label="label_ids", rtol=0, atol=0)


class TestLogitDiffsSAEBackendParity(AnalysisExtractionMixin):
    """logit_diffs_sae: TransformerBridge ↔ NNsight SAE-spliced forward (``model_fwd_w_cache_latent_models``).

    Compares SAE-spliced forward pass results including cached activations,
    alive latents, and logit diffs.  Bridge uses TL hook-based SAE splicing;
    NNsight uses thread-interleaved SAE splicing.
    """

    _analysis_fixture_specs: ClassVar[dict[str, AnalysisFixtureSpec]] = {
        "br_sae": AnalysisFixtureSpec(fixture_key=_BR_SAE),
        "ns_sae": AnalysisFixtureSpec(
            fixture_key=_NS_SAE,
            include_result=True,
            extra_extractors={"tokenizer": lambda fixture: fixture.it_session.datamodule.tokenizer},
        ),
        "ns_base": AnalysisFixtureSpec(fixture_key=_NS_BASE),
    }

    def test_logit_diffs_match(self, request):
        """Logit diffs with SAE should match across backends."""
        extracted = cast(dict[str, Any], self.extract_values(request))
        br_store = extracted["br_sae"]
        ns_store = extracted["ns_sae"].result
        _compare_tensor_lists(br_store.logit_diffs, ns_store.logit_diffs, label="logit_diffs")

    def test_answer_logits_match(self, request):
        """Answer logits with SAE should match across backends."""
        extracted = cast(dict[str, Any], self.extract_values(request))
        br_store = extracted["br_sae"]
        ns_store = extracted["ns_sae"].result
        _compare_tensor_lists(br_store.answer_logits, ns_store.answer_logits, label="answer_logits")

    def test_predictions_match(self, request):
        """Predictions with SAE should match across backends."""
        extracted = cast(dict[str, Any], self.extract_values(request))
        br_store = extracted["br_sae"]
        ns_store = extracted["ns_sae"].result
        _compare_tensor_lists(br_store.preds, ns_store.preds, label="preds", rtol=0, atol=0)

    def test_alive_latents_match(self, request):
        """Alive latent indices should be identical across backends."""
        extracted = cast(dict[str, Any], self.extract_values(request))
        br_store = extracted["br_sae"]
        ns_store = extracted["ns_sae"].result
        br_alive = br_store.alive_latents
        ns_alive = ns_store.alive_latents
        assert len(br_alive) == len(ns_alive), (
            f"alive_latents batch count mismatch ({len(br_alive)} vs {len(ns_alive)})"
        )
        for i, (d_br, d_ns) in enumerate(zip(br_alive, ns_alive)):
            assert set(d_br.keys()) == set(d_ns.keys()), f"alive_latents batch {i}: hook key mismatch"
            for hook_name in d_br:
                assert sorted(d_br[hook_name]) == sorted(d_ns[hook_name]), (
                    f"alive_latents batch {i} hook {hook_name}: index mismatch"
                )

    def test_prompts_and_tokens_saved(self, request):
        """Verify prompts/tokens are saved for SAE op (save_prompts=True, save_tokens=True)."""
        extracted = cast(dict[str, Any], self.extract_values(request))
        br_store = extracted["br_sae"]
        ns_store = extracted["ns_sae"].result
        # Both stores should have prompts and tokens
        assert br_store.prompts is not None and ns_store.prompts is not None
        assert br_store.tokens is not None and ns_store.tokens is not None
        # Prompt text content should be identical
        assert br_store.prompts == ns_store.prompts

    def test_base_vs_sae_comparison_works(self, monkeypatch, request):
        """``base_vs_sae_logit_diffs`` should work with NNsight backend results."""
        extracted = cast(dict[str, Any], self.extract_values(request))
        ns_base = extracted["ns_base"]
        ns_sae = extracted["ns_sae"].result
        monkeypatch.setattr("tabulate.tabulate", lambda *a, **k: "table")
        base_vs_sae_logit_diffs(
            sae=ns_sae,
            base_ref=ns_base,
            top_k=3,
            tokenizer=extracted["ns_sae"].tokenizer,
        )


class TestLogitDiffsAttrGradBackendParity(AnalysisExtractionMixin):
    """logit_diffs_attr_grad: TransformerBridge ↔ NNsight gradient attribution (``model_gradient``).

    Highest-risk parity comparison — gradient paths differ between Bridge
    (PyTorch hooks) and NNsight (thread-interleaved tracing).  Tolerances
    are relaxed to accommodate numerical differences in autograd graphs.
    """

    # Relaxed tolerances for gradient comparisons — different autograd graphs
    # between Bridge hooks and NNsight tracing can yield small numerical differences.
    _GRAD_RTOL: float = 5e-3
    _GRAD_ATOL: float = 5e-3
    _analysis_fixture_specs: ClassVar[dict[str, AnalysisFixtureSpec]] = {
        "br": AnalysisFixtureSpec(fixture_key=_BR_GRAD),
        "ns": AnalysisFixtureSpec(fixture_key=_NS_GRAD),
    }

    def test_logit_diffs_match(self, request):
        """Logit diffs from the gradient forward pass should match."""
        extracted = cast(dict[str, AnalysisStore], self.extract_values(request))
        br_store = extracted["br"]
        ns_store = extracted["ns"]
        _compare_tensor_lists(
            br_store.logit_diffs,
            ns_store.logit_diffs,
            label="logit_diffs",
            rtol=self._GRAD_RTOL,
            atol=self._GRAD_ATOL,
        )

    def test_attribution_values_match(self, request):
        """Gradient-based attribution values should match within tolerance."""
        extracted = cast(dict[str, AnalysisStore], self.extract_values(request))
        br_store = extracted["br"]
        ns_store = extracted["ns"]
        _compare_dict_tensor_lists(
            br_store.attribution_values,
            ns_store.attribution_values,
            label="attribution_values",
            rtol=self._GRAD_RTOL,
            atol=self._GRAD_ATOL,
        )

    def test_correct_activations_match(self, request):
        """Correct activations (subset where logit_diff > 0) should match."""
        extracted = cast(dict[str, AnalysisStore], self.extract_values(request))
        br_store = extracted["br"]
        ns_store = extracted["ns"]
        _compare_dict_tensor_lists(
            br_store.correct_activations,
            ns_store.correct_activations,
            label="correct_activations",
            rtol=self._GRAD_RTOL,
            atol=self._GRAD_ATOL,
        )

    def test_alive_latents_match(self, request):
        """Alive latents should be identical (same SAE, same data)."""
        extracted = cast(dict[str, AnalysisStore], self.extract_values(request))
        br_store = extracted["br"]
        ns_store = extracted["ns"]
        br_alive = br_store.alive_latents
        ns_alive = ns_store.alive_latents
        assert len(br_alive) == len(ns_alive)
        for i, (d_br, d_ns) in enumerate(zip(br_alive, ns_alive)):
            assert set(d_br.keys()) == set(d_ns.keys())
            for hook_name in d_br:
                assert sorted(d_br[hook_name]) == sorted(d_ns[hook_name])


class TestLogitDiffsAttrAblationBackendParity(AnalysisExtractionMixin):
    """logit_diffs_attr_ablation: TransformerBridge ↔ NNsight ablation (``model_ablation``).

    Compares per-latent ablation results — model runs once per alive latent
    with that latent zeroed out.  Bridge and NNsight should agree closely.

    Fixture scope is selected dynamically: low-RAM runners use function-scoped
    analysis fixtures for prompt teardown, while higher-RAM runners keep
    class-scoped reuse.
    """

    _analysis_fixture_specs: ClassVar[dict[str, AnalysisFixtureSpec]] = {
        "br": AnalysisFixtureSpec(
            fixture_key=_BR_ABLATION,
            field_names=("logit_diffs", "attribution_values", "alive_latents", "orig_labels", "preds"),
        ),
        "ns": AnalysisFixtureSpec(
            fixture_key=_NS_ABLATION,
            field_names=("logit_diffs", "attribution_values", "alive_latents", "orig_labels", "preds"),
        ),
    }

    def test_logit_diffs_match(self, request):
        """Ablation logit diffs should match across backends."""
        br_store = self.extract_field_store(request, "br", "logit_diffs")
        ns_store = self.extract_field_store(request, "ns", "logit_diffs")
        _compare_tensor_lists(br_store.logit_diffs, ns_store.logit_diffs, label="logit_diffs")

    def test_attribution_values_match(self, request):
        """Per-latent ablation attribution values should match."""
        br_store = self.extract_field_store(request, "br", "attribution_values")
        ns_store = self.extract_field_store(request, "ns", "attribution_values")
        _compare_dict_tensor_lists(
            br_store.attribution_values,
            ns_store.attribution_values,
            label="attribution_values",
        )

    def test_alive_latents_match(self, request):
        """Alive latents should be identical."""
        br_store = self.extract_field_store(request, "br", "alive_latents")
        ns_store = self.extract_field_store(request, "ns", "alive_latents")
        br_alive = br_store.alive_latents
        ns_alive = ns_store.alive_latents
        assert len(br_alive) == len(ns_alive)
        for i, (d_br, d_ns) in enumerate(zip(br_alive, ns_alive)):
            assert set(d_br.keys()) == set(d_ns.keys())
            for hook_name in d_br:
                assert sorted(d_br[hook_name]) == sorted(d_ns[hook_name])

    def test_compute_correct_parity(self, request):
        """compute_correct should return equivalent summaries for both backends."""
        br_summ = compute_correct(
            cast(Any, self.extract_field_store(request, "br", "orig_labels", "preds")),
            op="logit_diffs_attr_ablation",
        )
        ns_summ = compute_correct(
            cast(Any, self.extract_field_store(request, "ns", "orig_labels", "preds")),
            op="logit_diffs_attr_ablation",
        )
        assert br_summ.total_correct == ns_summ.total_correct
        assert br_summ.percentage_correct == pytest.approx(ns_summ.percentage_correct, abs=1e-4)


class TestBackendParityEdgeCases(AnalysisExtractionMixin):
    """Cross-cutting edge-case and structural checks (TransformerBridge ↔ NNsight)."""

    _analysis_fixture_specs: ClassVar[dict[str, AnalysisFixtureSpec]] = {
        "base_br": AnalysisFixtureSpec(
            fixture_key=_BR_BASE,
            field_names=("answer_logits", "loss"),
            include_dataset_metadata=True,
        ),
        "base_ns": AnalysisFixtureSpec(
            fixture_key=_NS_BASE,
            field_names=("answer_logits", "loss"),
            include_dataset_metadata=True,
        ),
        "sae_br": AnalysisFixtureSpec(
            fixture_key=_BR_SAE,
            field_names=("answer_logits",),
            include_dataset_metadata=True,
        ),
        "sae_ns": AnalysisFixtureSpec(
            fixture_key=_NS_SAE,
            field_names=("answer_logits",),
            include_dataset_metadata=True,
        ),
        "grad_br": AnalysisFixtureSpec(
            fixture_key=_BR_GRAD,
            field_names=("answer_logits",),
            include_dataset_metadata=True,
        ),
        "grad_ns": AnalysisFixtureSpec(
            fixture_key=_NS_GRAD,
            field_names=("answer_logits",),
            include_dataset_metadata=True,
        ),
        "ablation_br": AnalysisFixtureSpec(
            fixture_key=_BR_ABLATION,
            field_names=("answer_logits",),
            include_dataset_metadata=True,
        ),
        "ablation_ns": AnalysisFixtureSpec(
            fixture_key=_NS_ABLATION,
            field_names=("answer_logits",),
            include_dataset_metadata=True,
        ),
    }

    @pytest.mark.parametrize(
        ("br_key", "ns_key", "op_name"),
        [
            pytest.param("base_br", "base_ns", "logit_diffs_base", id="base"),
            pytest.param("sae_br", "sae_ns", "logit_diffs_sae", id="sae"),
        ],
    )
    def test_result_column_names_match(self, request, br_key, ns_key, op_name):
        """AnalysisStore dataset column names should be identical across backends."""
        br_meta = self.extract_dataset_metadata(request, br_key)
        ns_meta = self.extract_dataset_metadata(request, ns_key)
        assert set(cast(list[str], br_meta["column_names"])) == set(cast(list[str], ns_meta["column_names"])), (
            f"Column name mismatch for {op_name}"
        )

    @pytest.mark.parametrize(
        ("br_key", "ns_key", "op_name"),
        [
            pytest.param("base_br", "base_ns", "logit_diffs_base", id="base"),
            pytest.param("sae_br", "sae_ns", "logit_diffs_sae", id="sae"),
        ],
    )
    def test_result_row_counts_match(self, request, br_key, ns_key, op_name):
        """AnalysisStore datasets should have the same number of rows."""
        br_meta = self.extract_dataset_metadata(request, br_key)
        ns_meta = self.extract_dataset_metadata(request, ns_key)
        assert cast(int, br_meta["num_rows"]) == cast(int, ns_meta["num_rows"]), f"Row count mismatch for {op_name}"

    @pytest.mark.parametrize(
        ("br_key", "ns_key"),
        [
            pytest.param("base_br", "base_ns", id="base"),
            pytest.param("sae_br", "sae_ns", id="sae"),
            pytest.param("grad_br", "grad_ns", id="attr_grad"),
            pytest.param("ablation_br", "ablation_ns", id="attr_ablation"),
        ],
    )
    def test_answer_logits_dtype_match(self, request, br_key, ns_key):
        """Output tensor dtypes should be consistent across backends."""
        br_store = self.extract_field_store(request, br_key, "answer_logits")
        ns_store = self.extract_field_store(request, ns_key, "answer_logits")
        # Check first batch answer_logits dtype
        br_logits = br_store.answer_logits
        ns_logits = ns_store.answer_logits
        if br_logits and ns_logits:
            br_sample = br_logits[0] if isinstance(br_logits[0], torch.Tensor) else next(iter(br_logits[0].values()))
            ns_sample = ns_logits[0] if isinstance(ns_logits[0], torch.Tensor) else next(iter(ns_logits[0].values()))
            # Both should be float32 on CPU
            if isinstance(br_sample, torch.Tensor) and isinstance(ns_sample, torch.Tensor):
                assert br_sample.dtype == ns_sample.dtype, "dtype mismatch for answer_logits"

    @pytest.mark.parametrize(
        ("br_key", "ns_key"),
        [
            pytest.param("base_br", "base_ns", id="base"),
            pytest.param("sae_br", "sae_ns", id="sae"),
        ],
    )
    def test_answer_logits_shape_match(self, request, br_key, ns_key):
        """Output tensor shapes should match across backends."""
        br_store = self.extract_field_store(request, br_key, "answer_logits")
        ns_store = self.extract_field_store(request, ns_key, "answer_logits")
        br_logits = br_store.answer_logits
        ns_logits = ns_store.answer_logits
        assert len(br_logits) == len(ns_logits), "batch count mismatch"
        for i, (t_br, t_ns) in enumerate(zip(br_logits, ns_logits)):
            assert t_br.shape == t_ns.shape, f"shape mismatch in batch {i}: {t_br.shape} vs {t_ns.shape}"

    def test_loss_values_close(self, request):
        """Loss values from logit_diffs_base should be close across backends."""
        br_store = self.extract_field_store(request, "base_br", "loss")
        ns_store = self.extract_field_store(request, "base_ns", "loss")
        _compare_tensor_lists(br_store.loss, ns_store.loss, label="loss")


# ---------------------------------------------------------------------------
# Direction intervention parity
# ---------------------------------------------------------------------------


def _extract_module_and_batch(fixture):
    """Pull the module and first eval batch from an analysis session fixture."""
    module = fixture.it_session.module
    datamodule = fixture.it_session.datamodule
    dl = datamodule.test_dataloader()
    batch = next(iter(dl))
    return module, batch


class TestDirectionInterventionBackendParity:
    """direct_concept_direction_intervention: TransformerBridge ↔ NNsight.

    Both backends add a scaled direction vector at the ``unembed.hook_in``
    site and return pre/post logits.  Pre-intervention logits should already
    match (same model, same data) and the intervention delta should be very
    close since the hook application is numerically equivalent.
    """

    @pytest.fixture(scope="class")
    def _br_module_and_batch(self, request):
        fixture = request.getfixturevalue(_BR_BASE)
        return _extract_module_and_batch(fixture)

    @pytest.fixture(scope="class")
    def _ns_module_and_batch(self, request):
        fixture = request.getfixturevalue(_NS_BASE)
        return _extract_module_and_batch(fixture)

    @pytest.fixture(scope="class")
    def _concept_direction(self, _br_module_and_batch):
        """Build a simple concept direction from token embedding difference."""
        module, _ = _br_module_and_batch
        # GPT-2 token IDs for " Dallas" and " Austin" (hardcoded to avoid tokenizer dependency)
        tok_a = 8533  # " Dallas"
        tok_b = 9533  # " Austin"
        embed = module.model.embed.W_E if hasattr(module.model, "embed") else module.model.transformer.wte.weight
        if hasattr(embed, "detach"):
            embed = embed.detach()
        else:
            embed = torch.as_tensor(embed)
        diff = embed[tok_a].float() - embed[tok_b].float()
        return diff / torch.linalg.vector_norm(diff)

    @staticmethod
    def _run_direction_intervention(module, batch, direction, scale_factor=1.0):
        """Call the direction intervention op and return the result AnalysisBatch."""
        # Prune batch to forward-compatible keys (mirrors analysis pipeline auto_prune_batch_encoding)
        if hasattr(module, "auto_prune_batch") and hasattr(batch, "data"):
            batch = module.auto_prune_batch(batch, "forward")
        analysis_batch = AnalysisBatch(
            concept_direction=direction,
            direction_scale_factor=scale_factor,
        )
        return it.direct_concept_direction_intervention(module, analysis_batch, batch=batch, batch_idx=0)

    def test_pre_intervention_logits_match(self, _br_module_and_batch, _ns_module_and_batch, _concept_direction):
        """Pre-intervention last-token logits should match across backends."""
        br_module, br_batch = _br_module_and_batch
        ns_module, ns_batch = _ns_module_and_batch

        br_result = self._run_direction_intervention(br_module, br_batch, _concept_direction)
        ns_result = self._run_direction_intervention(ns_module, ns_batch, _concept_direction)

        assert_close(
            br_result.pre_intervention_logits,
            ns_result.pre_intervention_logits,
            rtol=1e-4,
            atol=1e-4,
            msg="pre_intervention_logits",
        )

    def test_post_intervention_logits_match(self, _br_module_and_batch, _ns_module_and_batch, _concept_direction):
        """Post-intervention logits should match across backends."""
        br_module, br_batch = _br_module_and_batch
        ns_module, ns_batch = _ns_module_and_batch

        br_result = self._run_direction_intervention(br_module, br_batch, _concept_direction)
        ns_result = self._run_direction_intervention(ns_module, ns_batch, _concept_direction)

        assert_close(
            br_result.post_intervention_logits,
            ns_result.post_intervention_logits,
            rtol=1e-4,
            atol=1e-4,
            msg="post_intervention_logits",
        )

    def test_logit_diff_match(self, _br_module_and_batch, _ns_module_and_batch, _concept_direction):
        """Scalar logit_diff should match across backends."""
        br_module, br_batch = _br_module_and_batch
        ns_module, ns_batch = _ns_module_and_batch

        br_result = self._run_direction_intervention(br_module, br_batch, _concept_direction)
        ns_result = self._run_direction_intervention(ns_module, ns_batch, _concept_direction)

        assert_close(
            br_result.logit_diff,
            ns_result.logit_diff,
            rtol=1e-4,
            atol=1e-4,
            msg="logit_diff",
        )

    def test_intervention_changes_logits(self, _br_module_and_batch, _concept_direction):
        """Verify the intervention actually modifies logits (non-trivial test)."""
        br_module, br_batch = _br_module_and_batch
        result = self._run_direction_intervention(br_module, br_batch, _concept_direction)
        assert not torch.allclose(result.pre_intervention_logits, result.post_intervention_logits, atol=1e-6), (
            "Intervention should change logits"
        )

    def test_scale_factor_affects_magnitude(self, _br_module_and_batch, _ns_module_and_batch, _concept_direction):
        """Larger scale_factor should produce a larger logit delta."""
        br_module, br_batch = _br_module_and_batch
        ns_module, ns_batch = _ns_module_and_batch

        br_result_1x = self._run_direction_intervention(br_module, br_batch, _concept_direction, scale_factor=1.0)
        br_result_5x = self._run_direction_intervention(br_module, br_batch, _concept_direction, scale_factor=5.0)

        ns_result_1x = self._run_direction_intervention(ns_module, ns_batch, _concept_direction, scale_factor=1.0)
        ns_result_5x = self._run_direction_intervention(ns_module, ns_batch, _concept_direction, scale_factor=5.0)

        # Larger scale should yield larger absolute logit delta
        br_delta_1 = (br_result_1x.post_intervention_logits - br_result_1x.pre_intervention_logits).abs().mean()
        br_delta_5 = (br_result_5x.post_intervention_logits - br_result_5x.pre_intervention_logits).abs().mean()
        assert br_delta_5 > br_delta_1, "5× scale should produce larger delta than 1×"

        ns_delta_1 = (ns_result_1x.post_intervention_logits - ns_result_1x.pre_intervention_logits).abs().mean()
        ns_delta_5 = (ns_result_5x.post_intervention_logits - ns_result_5x.pre_intervention_logits).abs().mean()
        assert ns_delta_5 > ns_delta_1, "5× scale should produce larger delta than 1× (NNsight)"

        # Cross-backend delta magnitudes should be close
        assert_close(br_delta_5, ns_delta_5, rtol=1e-3, atol=1e-3, msg="5× scale delta magnitude")
