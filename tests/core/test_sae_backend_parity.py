"""SAE Backend Parity Tests: TransformerLens vs NNsight.

Validates that analysis operations produce equivalent results regardless of
whether the TransformerLens or NNsight backend is used for SAE analysis.

Each test compares AnalysisStore results from session-scoped TL and NNsight
fixtures that run the same analysis op on the same GPT-2 model and SAE
configuration, differing only in the backend path.

Fixture mapping (TL → NNsight):
    sl_gpt2_logit_diffs_base          → sl_ns_gpt2_logit_diffs_base
    sl_gpt2_logit_diffs_sae           → sl_ns_gpt2_logit_diffs_sae
    sl_gpt2_logit_diffs_attr_grad     → sl_ns_gpt2_logit_diffs_attr_grad
    sl_gpt2_logit_diffs_attr_ablation → sl_ns_gpt2_logit_diffs_attr_ablation
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

# ---------------------------------------------------------------------------
# Fixture key constants
# ---------------------------------------------------------------------------
_TL_BASE = "get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis"
_TL_SAE = "get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis"
_TL_GRAD = "get_analysis_session__sl_gpt2_logit_diffs_attr_grad__initonly_runanalysis"
_TL_ABLATION = "get_analysis_session__sl_gpt2_logit_diffs_attr_ablation__initonly_runanalysis"

_NS_BASE = "get_analysis_session__sl_ns_gpt2_logit_diffs_base__initonly_runanalysis"
_NS_SAE = "get_analysis_session__sl_ns_gpt2_logit_diffs_sae__initonly_runanalysis"
_NS_GRAD = "get_analysis_session__sl_ns_gpt2_logit_diffs_attr_grad__initonly_runanalysis"
_NS_ABLATION = "get_analysis_session__sl_ns_gpt2_logit_diffs_attr_ablation__initonly_runanalysis"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_result(request, fixture_key: str) -> AnalysisStore:
    """Retrieve and deepcopy the AnalysisStore result from a fixture."""
    fixture = request.getfixturevalue(fixture_key)
    return deepcopy(fixture.result)


def _compare_tensor_lists(
    tl_list: list[torch.Tensor],
    ns_list: list[torch.Tensor],
    *,
    label: str,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    """Compare two lists of tensors element-wise with assert_close."""
    assert len(tl_list) == len(ns_list), f"{label}: batch count mismatch ({len(tl_list)} vs {len(ns_list)})"
    for i, (t_tl, t_ns) in enumerate(zip(tl_list, ns_list)):
        assert_close(t_tl, t_ns, rtol=rtol, atol=atol, msg=f"{label} batch {i}")


def _compare_dict_tensor_lists(
    tl_list: list[dict[str, torch.Tensor]],
    ns_list: list[dict[str, torch.Tensor]],
    *,
    label: str,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    """Compare two lists of dicts mapping hook names → Tensors."""
    assert len(tl_list) == len(ns_list), f"{label}: batch count mismatch ({len(tl_list)} vs {len(ns_list)})"
    for i, (d_tl, d_ns) in enumerate(zip(tl_list, ns_list)):
        assert set(d_tl.keys()) == set(d_ns.keys()), (
            f"{label} batch {i}: hook key mismatch ({d_tl.keys()} vs {d_ns.keys()})"
        )
        for hook_name in d_tl:
            assert_close(
                d_tl[hook_name],
                d_ns[hook_name],
                rtol=rtol,
                atol=atol,
                msg=f"{label} batch {i} hook {hook_name}",
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLogitDiffsBaseBackendParity:
    """logit_diffs_base: basic forward pass with no SAE involvement.

    This is the simplest parity check — both backends run the same HF model
    and compute logit diffs.  Results should be very close.
    """

    def test_logit_diffs_match(self, request):
        """Core logit_diffs values should match across backends."""
        tl_store = _get_result(request, _TL_BASE)
        ns_store = _get_result(request, _NS_BASE)
        _compare_tensor_lists(tl_store.logit_diffs, ns_store.logit_diffs, label="logit_diffs")

    def test_answer_logits_match(self, request):
        """Answer logits should match across backends."""
        tl_store = _get_result(request, _TL_BASE)
        ns_store = _get_result(request, _NS_BASE)
        _compare_tensor_lists(tl_store.answer_logits, ns_store.answer_logits, label="answer_logits")

    def test_predictions_match(self, request):
        """Model predictions should be identical across backends."""
        tl_store = _get_result(request, _TL_BASE)
        ns_store = _get_result(request, _NS_BASE)
        _compare_tensor_lists(tl_store.preds, ns_store.preds, label="preds", rtol=0, atol=0)

    def test_labels_match(self, request):
        """Ground truth labels should be identical (dataset-level sanity check)."""
        tl_store = _get_result(request, _TL_BASE)
        ns_store = _get_result(request, _NS_BASE)
        _compare_tensor_lists(tl_store.orig_labels, ns_store.orig_labels, label="orig_labels", rtol=0, atol=0)
        _compare_tensor_lists(tl_store.label_ids, ns_store.label_ids, label="label_ids", rtol=0, atol=0)


class TestLogitDiffsSAEBackendParity:
    """logit_diffs_sae: forward pass with SAE spliced in (``model_cache_forward``).

    Compares SAE-spliced forward pass results including cached activations,
    alive latents, and logit diffs.
    """

    def test_logit_diffs_match(self, request):
        """Logit diffs with SAE should match across backends."""
        tl_store = _get_result(request, _TL_SAE)
        ns_store = _get_result(request, _NS_SAE)
        _compare_tensor_lists(tl_store.logit_diffs, ns_store.logit_diffs, label="logit_diffs")

    def test_answer_logits_match(self, request):
        """Answer logits with SAE should match across backends."""
        tl_store = _get_result(request, _TL_SAE)
        ns_store = _get_result(request, _NS_SAE)
        _compare_tensor_lists(tl_store.answer_logits, ns_store.answer_logits, label="answer_logits")

    def test_predictions_match(self, request):
        """Predictions with SAE should match across backends."""
        tl_store = _get_result(request, _TL_SAE)
        ns_store = _get_result(request, _NS_SAE)
        _compare_tensor_lists(tl_store.preds, ns_store.preds, label="preds", rtol=0, atol=0)

    def test_alive_latents_match(self, request):
        """Alive latent indices should be identical across backends."""
        tl_store = _get_result(request, _TL_SAE)
        ns_store = _get_result(request, _NS_SAE)
        tl_alive = tl_store.alive_latents
        ns_alive = ns_store.alive_latents
        assert len(tl_alive) == len(ns_alive), (
            f"alive_latents batch count mismatch ({len(tl_alive)} vs {len(ns_alive)})"
        )
        for i, (d_tl, d_ns) in enumerate(zip(tl_alive, ns_alive)):
            assert set(d_tl.keys()) == set(d_ns.keys()), f"alive_latents batch {i}: hook key mismatch"
            for hook_name in d_tl:
                assert sorted(d_tl[hook_name]) == sorted(d_ns[hook_name]), (
                    f"alive_latents batch {i} hook {hook_name}: index mismatch"
                )

    def test_prompts_and_tokens_saved(self, request):
        """Verify prompts/tokens are saved for SAE op (save_prompts=True, save_tokens=True)."""
        tl_store = _get_result(request, _TL_SAE)
        ns_store = _get_result(request, _NS_SAE)
        # Both stores should have prompts and tokens
        assert tl_store.prompts is not None and ns_store.prompts is not None
        assert tl_store.tokens is not None and ns_store.tokens is not None
        # Prompt text content should be identical
        assert tl_store.prompts == ns_store.prompts

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
    """logit_diffs_attr_grad: gradient-based attribution (``model_gradient``).

    This is the highest-risk parity comparison because gradient paths may
    differ between backends.  Tolerances are relaxed to accommodate
    numerical differences in gradient computation.
    """

    # Relaxed tolerances for gradient comparisons — different autograd graphs
    # between TL hooks and NNsight tracing can yield small numerical differences.
    _GRAD_RTOL: float = 5e-3
    _GRAD_ATOL: float = 5e-3

    def test_logit_diffs_match(self, request):
        """Logit diffs from the gradient forward pass should match."""
        tl_store = _get_result(request, _TL_GRAD)
        ns_store = _get_result(request, _NS_GRAD)
        _compare_tensor_lists(
            tl_store.logit_diffs,
            ns_store.logit_diffs,
            label="logit_diffs",
            rtol=self._GRAD_RTOL,
            atol=self._GRAD_ATOL,
        )

    def test_attribution_values_match(self, request):
        """Gradient-based attribution values should match within tolerance."""
        tl_store = _get_result(request, _TL_GRAD)
        ns_store = _get_result(request, _NS_GRAD)
        _compare_dict_tensor_lists(
            tl_store.attribution_values,
            ns_store.attribution_values,
            label="attribution_values",
            rtol=self._GRAD_RTOL,
            atol=self._GRAD_ATOL,
        )

    def test_correct_activations_match(self, request):
        """Correct activations (subset where logit_diff > 0) should match."""
        tl_store = _get_result(request, _TL_GRAD)
        ns_store = _get_result(request, _NS_GRAD)
        _compare_dict_tensor_lists(
            tl_store.correct_activations,
            ns_store.correct_activations,
            label="correct_activations",
            rtol=self._GRAD_RTOL,
            atol=self._GRAD_ATOL,
        )

    def test_alive_latents_match(self, request):
        """Alive latents should be identical (same SAE, same data)."""
        tl_store = _get_result(request, _TL_GRAD)
        ns_store = _get_result(request, _NS_GRAD)
        tl_alive = tl_store.alive_latents
        ns_alive = ns_store.alive_latents
        assert len(tl_alive) == len(ns_alive)
        for i, (d_tl, d_ns) in enumerate(zip(tl_alive, ns_alive)):
            assert set(d_tl.keys()) == set(d_ns.keys())
            for hook_name in d_tl:
                assert sorted(d_tl[hook_name]) == sorted(d_ns[hook_name])


class TestLogitDiffsAttrAblationBackendParity:
    """logit_diffs_attr_ablation: per-latent ablation attribution (``model_ablation``).

    Compares per-latent ablation results which involve running the model once
    per alive latent with that latent zeroed out.
    """

    def test_logit_diffs_match(self, request):
        """Ablation logit diffs should match across backends."""
        tl_store = _get_result(request, _TL_ABLATION)
        ns_store = _get_result(request, _NS_ABLATION)
        _compare_tensor_lists(tl_store.logit_diffs, ns_store.logit_diffs, label="logit_diffs")

    def test_attribution_values_match(self, request):
        """Per-latent ablation attribution values should match."""
        tl_store = _get_result(request, _TL_ABLATION)
        ns_store = _get_result(request, _NS_ABLATION)
        _compare_dict_tensor_lists(
            tl_store.attribution_values,
            ns_store.attribution_values,
            label="attribution_values",
        )

    def test_alive_latents_match(self, request):
        """Alive latents should be identical."""
        tl_store = _get_result(request, _TL_ABLATION)
        ns_store = _get_result(request, _NS_ABLATION)
        tl_alive = tl_store.alive_latents
        ns_alive = ns_store.alive_latents
        assert len(tl_alive) == len(ns_alive)
        for i, (d_tl, d_ns) in enumerate(zip(tl_alive, ns_alive)):
            assert set(d_tl.keys()) == set(d_ns.keys())
            for hook_name in d_tl:
                assert sorted(d_tl[hook_name]) == sorted(d_ns[hook_name])

    def test_compute_correct_parity(self, request):
        """compute_correct should return equivalent summaries for both backends."""
        tl_fixture = request.getfixturevalue(_TL_ABLATION)
        ns_fixture = request.getfixturevalue(_NS_ABLATION)
        tl_summ = compute_correct(deepcopy(tl_fixture.result), op="logit_diffs_attr_ablation")
        ns_summ = compute_correct(deepcopy(ns_fixture.result), op="logit_diffs_attr_ablation")
        assert tl_summ.total_correct == ns_summ.total_correct
        assert tl_summ.percentage_correct == pytest.approx(ns_summ.percentage_correct, abs=1e-4)


class TestBackendParityEdgeCases:
    """Cross-cutting edge-case and structural checks."""

    @pytest.mark.parametrize(
        ("tl_key", "ns_key", "op_name"),
        [
            pytest.param(_TL_BASE, _NS_BASE, "logit_diffs_base", id="base"),
            pytest.param(_TL_SAE, _NS_SAE, "logit_diffs_sae", id="sae"),
        ],
    )
    def test_result_column_names_match(self, request, tl_key, ns_key, op_name):
        """AnalysisStore dataset column names should be identical across backends."""
        tl_store = _get_result(request, tl_key)
        ns_store = _get_result(request, ns_key)
        assert set(tl_store.dataset.column_names) == set(ns_store.dataset.column_names), (
            f"Column name mismatch for {op_name}"
        )

    @pytest.mark.parametrize(
        ("tl_key", "ns_key", "op_name"),
        [
            pytest.param(_TL_BASE, _NS_BASE, "logit_diffs_base", id="base"),
            pytest.param(_TL_SAE, _NS_SAE, "logit_diffs_sae", id="sae"),
        ],
    )
    def test_result_row_counts_match(self, request, tl_key, ns_key, op_name):
        """AnalysisStore datasets should have the same number of rows."""
        tl_store = _get_result(request, tl_key)
        ns_store = _get_result(request, ns_key)
        assert len(tl_store.dataset) == len(ns_store.dataset), f"Row count mismatch for {op_name}"

    @pytest.mark.parametrize(
        ("tl_key", "ns_key"),
        [
            pytest.param(_TL_BASE, _NS_BASE, id="base"),
            pytest.param(_TL_SAE, _NS_SAE, id="sae"),
            pytest.param(_TL_GRAD, _NS_GRAD, id="attr_grad"),
            pytest.param(_TL_ABLATION, _NS_ABLATION, id="attr_ablation"),
        ],
    )
    def test_answer_logits_dtype_match(self, request, tl_key, ns_key):
        """Output tensor dtypes should be consistent across backends."""
        tl_store = _get_result(request, tl_key)
        ns_store = _get_result(request, ns_key)
        # Check first batch answer_logits dtype
        tl_logits = tl_store.answer_logits
        ns_logits = ns_store.answer_logits
        if tl_logits and ns_logits:
            tl_sample = tl_logits[0] if isinstance(tl_logits[0], torch.Tensor) else next(iter(tl_logits[0].values()))
            ns_sample = ns_logits[0] if isinstance(ns_logits[0], torch.Tensor) else next(iter(ns_logits[0].values()))
            # Both should be float32 on CPU
            if isinstance(tl_sample, torch.Tensor) and isinstance(ns_sample, torch.Tensor):
                assert tl_sample.dtype == ns_sample.dtype, "dtype mismatch for answer_logits"

    @pytest.mark.parametrize(
        ("tl_key", "ns_key"),
        [
            pytest.param(_TL_BASE, _NS_BASE, id="base"),
            pytest.param(_TL_SAE, _NS_SAE, id="sae"),
        ],
    )
    def test_answer_logits_shape_match(self, request, tl_key, ns_key):
        """Output tensor shapes should match across backends."""
        tl_store = _get_result(request, tl_key)
        ns_store = _get_result(request, ns_key)
        tl_logits = tl_store.answer_logits
        ns_logits = ns_store.answer_logits
        assert len(tl_logits) == len(ns_logits), "batch count mismatch"
        for i, (t_tl, t_ns) in enumerate(zip(tl_logits, ns_logits)):
            assert t_tl.shape == t_ns.shape, f"shape mismatch in batch {i}: {t_tl.shape} vs {t_ns.shape}"

    def test_loss_values_close(self, request):
        """Loss values from logit_diffs_base should be close across backends."""
        tl_store = _get_result(request, _TL_BASE)
        ns_store = _get_result(request, _NS_BASE)
        _compare_tensor_lists(tl_store.loss, ns_store.loss, label="loss")
