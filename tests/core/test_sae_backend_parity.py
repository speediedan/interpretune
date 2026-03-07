"""SAE Backend Parity Tests: TransformerBridge ↔ NNsight.

Validates that analysis operations produce equivalent results regardless of
whether the TransformerBridge or NNsight backend is used for SAE analysis.

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
    docs/bridge_ht_divergence_analysis.md for the full analysis.

Fixture mapping (NNsight → Bridge):
    sl_ns_gpt2_logit_diffs_base          → sl_br_gpt2_logit_diffs_base
    sl_ns_gpt2_logit_diffs_sae           → sl_br_gpt2_logit_diffs_sae
    sl_ns_gpt2_logit_diffs_attr_grad     → sl_br_gpt2_logit_diffs_attr_grad
    sl_ns_gpt2_logit_diffs_attr_ablation → sl_br_gpt2_logit_diffs_attr_ablation
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch.testing import assert_close

from interpretune.analysis.core import (
    AnalysisStore,
    base_vs_sae_logit_diffs,
    compute_correct,
)
from tests.runif import RunIf

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_result(request, fixture_key: str) -> AnalysisStore:
    """Retrieve and deepcopy the AnalysisStore result from a fixture."""
    fixture = request.getfixturevalue(fixture_key)
    return deepcopy(fixture.result)


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


# ---------------------------------------------------------------------------
# Bridge bounded-divergence helpers  [REMOVED]
# ---------------------------------------------------------------------------
# Tier 2 parity helpers (_assert_alive_latents_overlap,
# _assert_attribution_shared_keys_correlated) were removed along with the
# Bridge ↔ HookedTransformer Tier 2 test classes.  See the module docstring
# and docs/bridge_ht_divergence_analysis.md for rationale.


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLogitDiffsBaseBackendParity:
    """logit_diffs_base: TransformerBridge ↔ NNsight forward pass (no SAE).

    Simplest parity check — both backends run the HF model's native forward
    pass and compute logit diffs.  Bridge (source of truth) and NNsight
    should produce very close results.
    """

    def test_logit_diffs_match(self, request):
        """Core logit_diffs values should match across backends."""
        br_store = _get_result(request, _BR_BASE)
        ns_store = _get_result(request, _NS_BASE)
        _compare_tensor_lists(br_store.logit_diffs, ns_store.logit_diffs, label="logit_diffs")

    def test_answer_logits_match(self, request):
        """Answer logits should match across backends."""
        br_store = _get_result(request, _BR_BASE)
        ns_store = _get_result(request, _NS_BASE)
        _compare_tensor_lists(br_store.answer_logits, ns_store.answer_logits, label="answer_logits")

    def test_predictions_match(self, request):
        """Model predictions should be identical across backends."""
        br_store = _get_result(request, _BR_BASE)
        ns_store = _get_result(request, _NS_BASE)
        _compare_tensor_lists(br_store.preds, ns_store.preds, label="preds", rtol=0, atol=0)

    def test_labels_match(self, request):
        """Ground truth labels should be identical (dataset-level sanity check)."""
        br_store = _get_result(request, _BR_BASE)
        ns_store = _get_result(request, _NS_BASE)
        _compare_tensor_lists(br_store.orig_labels, ns_store.orig_labels, label="orig_labels", rtol=0, atol=0)
        _compare_tensor_lists(br_store.label_ids, ns_store.label_ids, label="label_ids", rtol=0, atol=0)


class TestLogitDiffsSAEBackendParity:
    """logit_diffs_sae: TransformerBridge ↔ NNsight SAE-spliced forward (``model_cache_forward``).

    Compares SAE-spliced forward pass results including cached activations,
    alive latents, and logit diffs.  Bridge uses TL hook-based SAE splicing;
    NNsight uses thread-interleaved SAE splicing.
    """

    def test_logit_diffs_match(self, request):
        """Logit diffs with SAE should match across backends."""
        br_store = _get_result(request, _BR_SAE)
        ns_store = _get_result(request, _NS_SAE)
        _compare_tensor_lists(br_store.logit_diffs, ns_store.logit_diffs, label="logit_diffs")

    def test_answer_logits_match(self, request):
        """Answer logits with SAE should match across backends."""
        br_store = _get_result(request, _BR_SAE)
        ns_store = _get_result(request, _NS_SAE)
        _compare_tensor_lists(br_store.answer_logits, ns_store.answer_logits, label="answer_logits")

    def test_predictions_match(self, request):
        """Predictions with SAE should match across backends."""
        br_store = _get_result(request, _BR_SAE)
        ns_store = _get_result(request, _NS_SAE)
        _compare_tensor_lists(br_store.preds, ns_store.preds, label="preds", rtol=0, atol=0)

    def test_alive_latents_match(self, request):
        """Alive latent indices should be identical across backends."""
        br_store = _get_result(request, _BR_SAE)
        ns_store = _get_result(request, _NS_SAE)
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
        br_store = _get_result(request, _BR_SAE)
        ns_store = _get_result(request, _NS_SAE)
        # Both stores should have prompts and tokens
        assert br_store.prompts is not None and ns_store.prompts is not None
        assert br_store.tokens is not None and ns_store.tokens is not None
        # Prompt text content should be identical
        assert br_store.prompts == ns_store.prompts

    def test_base_vs_sae_comparison_works(self, monkeypatch, request):
        """``base_vs_sae_logit_diffs`` should work with NNsight backend results."""
        ns_base = _get_result(request, _NS_BASE)
        ns_sae = _get_result(request, _NS_SAE)
        sae_fixture = request.getfixturevalue(_NS_SAE)
        monkeypatch.setattr("tabulate.tabulate", lambda *a, **k: "table")
        base_vs_sae_logit_diffs(
            sae=ns_sae,
            base_ref=ns_base,
            top_k=3,
            tokenizer=sae_fixture.it_session.datamodule.tokenizer,
        )


class TestLogitDiffsAttrGradBackendParity:
    """logit_diffs_attr_grad: TransformerBridge ↔ NNsight gradient attribution (``model_gradient``).

    Highest-risk parity comparison — gradient paths differ between Bridge
    (PyTorch hooks) and NNsight (thread-interleaved tracing).  Tolerances
    are relaxed to accommodate numerical differences in autograd graphs.
    """

    # Relaxed tolerances for gradient comparisons — different autograd graphs
    # between Bridge hooks and NNsight tracing can yield small numerical differences.
    _GRAD_RTOL: float = 5e-3
    _GRAD_ATOL: float = 5e-3

    def test_logit_diffs_match(self, request):
        """Logit diffs from the gradient forward pass should match."""
        br_store = _get_result(request, _BR_GRAD)
        ns_store = _get_result(request, _NS_GRAD)
        _compare_tensor_lists(
            br_store.logit_diffs,
            ns_store.logit_diffs,
            label="logit_diffs",
            rtol=self._GRAD_RTOL,
            atol=self._GRAD_ATOL,
        )

    def test_attribution_values_match(self, request):
        """Gradient-based attribution values should match within tolerance."""
        br_store = _get_result(request, _BR_GRAD)
        ns_store = _get_result(request, _NS_GRAD)
        _compare_dict_tensor_lists(
            br_store.attribution_values,
            ns_store.attribution_values,
            label="attribution_values",
            rtol=self._GRAD_RTOL,
            atol=self._GRAD_ATOL,
        )

    def test_correct_activations_match(self, request):
        """Correct activations (subset where logit_diff > 0) should match."""
        br_store = _get_result(request, _BR_GRAD)
        ns_store = _get_result(request, _NS_GRAD)
        _compare_dict_tensor_lists(
            br_store.correct_activations,
            ns_store.correct_activations,
            label="correct_activations",
            rtol=self._GRAD_RTOL,
            atol=self._GRAD_ATOL,
        )

    def test_alive_latents_match(self, request):
        """Alive latents should be identical (same SAE, same data)."""
        br_store = _get_result(request, _BR_GRAD)
        ns_store = _get_result(request, _NS_GRAD)
        br_alive = br_store.alive_latents
        ns_alive = ns_store.alive_latents
        assert len(br_alive) == len(ns_alive)
        for i, (d_br, d_ns) in enumerate(zip(br_alive, ns_alive)):
            assert set(d_br.keys()) == set(d_ns.keys())
            for hook_name in d_br:
                assert sorted(d_br[hook_name]) == sorted(d_ns[hook_name])


@RunIf(skip_windows=True)  # nnsight interleaver segfaults on Windows during multi-trace ablation
@pytest.mark.usefixtures("cleanup_memory")
class TestLogitDiffsAttrAblationBackendParity:
    """logit_diffs_attr_ablation: TransformerBridge ↔ NNsight ablation (``model_ablation``).

    Compares per-latent ablation results — model runs once per alive latent
    with that latent zeroed out.  Bridge and NNsight should agree closely.
    """

    def test_logit_diffs_match(self, request):
        """Ablation logit diffs should match across backends."""
        br_store = _get_result(request, _BR_ABLATION)
        ns_store = _get_result(request, _NS_ABLATION)
        _compare_tensor_lists(br_store.logit_diffs, ns_store.logit_diffs, label="logit_diffs")

    def test_attribution_values_match(self, request):
        """Per-latent ablation attribution values should match."""
        br_store = _get_result(request, _BR_ABLATION)
        ns_store = _get_result(request, _NS_ABLATION)
        _compare_dict_tensor_lists(
            br_store.attribution_values,
            ns_store.attribution_values,
            label="attribution_values",
        )

    def test_alive_latents_match(self, request):
        """Alive latents should be identical."""
        br_store = _get_result(request, _BR_ABLATION)
        ns_store = _get_result(request, _NS_ABLATION)
        br_alive = br_store.alive_latents
        ns_alive = ns_store.alive_latents
        assert len(br_alive) == len(ns_alive)
        for i, (d_br, d_ns) in enumerate(zip(br_alive, ns_alive)):
            assert set(d_br.keys()) == set(d_ns.keys())
            for hook_name in d_br:
                assert sorted(d_br[hook_name]) == sorted(d_ns[hook_name])

    def test_compute_correct_parity(self, request):
        """compute_correct should return equivalent summaries for both backends."""
        br_fixture = request.getfixturevalue(_BR_ABLATION)
        ns_fixture = request.getfixturevalue(_NS_ABLATION)
        br_summ = compute_correct(deepcopy(br_fixture.result), op="logit_diffs_attr_ablation")
        ns_summ = compute_correct(deepcopy(ns_fixture.result), op="logit_diffs_attr_ablation")
        assert br_summ.total_correct == ns_summ.total_correct
        assert br_summ.percentage_correct == pytest.approx(ns_summ.percentage_correct, abs=1e-4)


class TestBackendParityEdgeCases:
    """Cross-cutting edge-case and structural checks (TransformerBridge ↔ NNsight)."""

    @pytest.mark.parametrize(
        ("br_key", "ns_key", "op_name"),
        [
            pytest.param(_BR_BASE, _NS_BASE, "logit_diffs_base", id="base"),
            pytest.param(_BR_SAE, _NS_SAE, "logit_diffs_sae", id="sae"),
        ],
    )
    def test_result_column_names_match(self, request, br_key, ns_key, op_name):
        """AnalysisStore dataset column names should be identical across backends."""
        br_store = _get_result(request, br_key)
        ns_store = _get_result(request, ns_key)
        assert set(br_store.dataset.column_names) == set(ns_store.dataset.column_names), (
            f"Column name mismatch for {op_name}"
        )

    @pytest.mark.parametrize(
        ("br_key", "ns_key", "op_name"),
        [
            pytest.param(_BR_BASE, _NS_BASE, "logit_diffs_base", id="base"),
            pytest.param(_BR_SAE, _NS_SAE, "logit_diffs_sae", id="sae"),
        ],
    )
    def test_result_row_counts_match(self, request, br_key, ns_key, op_name):
        """AnalysisStore datasets should have the same number of rows."""
        br_store = _get_result(request, br_key)
        ns_store = _get_result(request, ns_key)
        assert len(br_store.dataset) == len(ns_store.dataset), f"Row count mismatch for {op_name}"

    @pytest.mark.parametrize(
        ("br_key", "ns_key"),
        [
            pytest.param(_BR_BASE, _NS_BASE, id="base"),
            pytest.param(_BR_SAE, _NS_SAE, id="sae"),
            pytest.param(_BR_GRAD, _NS_GRAD, id="attr_grad"),
            pytest.param(_BR_ABLATION, _NS_ABLATION, id="attr_ablation", marks=RunIf(skip_windows=True)),
        ],
    )
    def test_answer_logits_dtype_match(self, request, br_key, ns_key):
        """Output tensor dtypes should be consistent across backends."""
        br_store = _get_result(request, br_key)
        ns_store = _get_result(request, ns_key)
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
            pytest.param(_BR_BASE, _NS_BASE, id="base"),
            pytest.param(_BR_SAE, _NS_SAE, id="sae"),
        ],
    )
    def test_answer_logits_shape_match(self, request, br_key, ns_key):
        """Output tensor shapes should match across backends."""
        br_store = _get_result(request, br_key)
        ns_store = _get_result(request, ns_key)
        br_logits = br_store.answer_logits
        ns_logits = ns_store.answer_logits
        assert len(br_logits) == len(ns_logits), "batch count mismatch"
        for i, (t_br, t_ns) in enumerate(zip(br_logits, ns_logits)):
            assert t_br.shape == t_ns.shape, f"shape mismatch in batch {i}: {t_br.shape} vs {t_ns.shape}"

    def test_loss_values_close(self, request):
        """Loss values from logit_diffs_base should be close across backends."""
        br_store = _get_result(request, _BR_BASE)
        ns_store = _get_result(request, _NS_BASE)
        _compare_tensor_lists(br_store.loss, ns_store.loss, label="loss")
