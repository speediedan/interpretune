"""Direct unit tests for the concept-direction streaming aggregator internals.

These pin the CURRENT (pre op-state lifecycle) semantics of the streaming helpers in
``interpretune.analysis.ops.bundled.concept.concept_ops`` so the Phase 2 port onto the declared
op-state seam has a behavioral baseline to diff against. Notably they pin the ``batch_idx == 0``
reset heuristic exactly as implemented today; the Phase 2 lifecycle work replaces that contract
deliberately (and these tests with it).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.bundled.concept.concept_ops import (
    CONCEPT_STREAMING_STATE_FIELDS,
    _accumulate_streaming_group_means,
    _accumulate_streaming_paired_rejection,
    _concept_direction_streaming,
    _drain_streaming_paired_buffers,
    _parse_streaming_per_batch_inputs,
    _resolve_streaming_group_names,
    reset_concept_streaming_state,
)


def _module_without_cfg() -> SimpleNamespace:
    return SimpleNamespace(analysis_cfg=None)


def _batch(states, gids, weights=None, **extra) -> AnalysisBatch:
    payload = {"concept_latent_state": states, "concept_group_id": gids, **extra}
    if weights is not None:
        payload["concept_example_weight"] = weights
    return AnalysisBatch(**payload)


class TestParseStreamingPerBatchInputs:
    def test_tensor_inputs_normalized_to_cpu_float(self):
        states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        gids = torch.tensor([0, 1])
        weights = torch.tensor([0.5, 2.0])
        states_t, gids_t, weights_t, feature_shape = _parse_streaming_per_batch_inputs(
            _module_without_cfg(), _batch(states, gids, weights)
        )
        assert torch.equal(states_t, states)
        assert gids_t.dtype == torch.long and torch.equal(gids_t, gids)
        assert torch.equal(weights_t, weights)
        assert feature_shape == (2,)

    def test_missing_weights_default_to_ones(self):
        _, _, weights_t, _ = _parse_streaming_per_batch_inputs(
            _module_without_cfg(), _batch(torch.ones(3, 2), torch.tensor([0, 0, 1]))
        )
        assert torch.equal(weights_t, torch.ones(3))

    def test_list_inputs_flattened_via_store_rows(self):
        states = [torch.ones(2, 4), torch.full((1, 4), 2.0)]
        gids = [torch.tensor([0, 1]), torch.tensor([0])]
        states_t, gids_t, weights_t, feature_shape = _parse_streaming_per_batch_inputs(
            _module_without_cfg(), _batch(states, gids)
        )
        assert states_t.shape == (3, 4)
        assert gids_t.tolist() == [0, 1, 0]
        assert torch.equal(weights_t, torch.ones(3))
        assert feature_shape == (4,)

    def test_missing_inputs_raise(self):
        with pytest.raises(ValueError, match="streaming mode requires per-batch"):
            _parse_streaming_per_batch_inputs(_module_without_cfg(), AnalysisBatch())


class TestAccumulateStreamingGroupMeans:
    def test_accumulates_weighted_sums_across_batches(self):
        store = SimpleNamespace()
        _accumulate_streaming_group_means(
            torch.tensor([[1.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 1]), torch.tensor([1.0, 3.0]), (2,), store, 0
        )
        _accumulate_streaming_group_means(
            torch.tensor([[3.0, 0.0]]), torch.tensor([0]), torch.tensor([2.0]), (2,), store, 1
        )
        assert torch.allclose(store.concept_running_state_sum_a, torch.tensor([7.0, 0.0]))
        assert torch.allclose(store.concept_running_weight_a, torch.tensor(3.0))
        assert torch.allclose(store.concept_running_state_sum_b, torch.tensor([0.0, 6.0]))
        assert torch.allclose(store.concept_running_weight_b, torch.tensor(3.0))

    def test_batch_zero_resets_existing_state(self):
        # Baseline semantics: batch_idx == 0 discards prior accumulation (the Phase 2 op-state
        # lifecycle replaces this heuristic; this test documents the current contract).
        store = SimpleNamespace(
            concept_running_state_sum_a=torch.tensor([100.0, 100.0]),
            concept_running_weight_a=torch.tensor(50.0),
        )
        _accumulate_streaming_group_means(
            torch.tensor([[1.0, 1.0]]), torch.tensor([0]), torch.tensor([1.0]), (2,), store, 0
        )
        assert torch.allclose(store.concept_running_state_sum_a, torch.tensor([1.0, 1.0]))
        assert torch.allclose(store.concept_running_weight_a, torch.tensor(1.0))

    def test_absent_group_leaves_running_state_untouched(self):
        store = SimpleNamespace()
        _accumulate_streaming_group_means(
            torch.tensor([[1.0, 0.0]]), torch.tensor([0]), torch.tensor([1.0]), (2,), store, 0
        )
        _accumulate_streaming_group_means(
            torch.tensor([[5.0, 5.0]]), torch.tensor([1]), torch.tensor([1.0]), (2,), store, 1
        )
        assert torch.allclose(store.concept_running_state_sum_a, torch.tensor([1.0, 0.0]))
        assert torch.allclose(store.concept_running_state_sum_b, torch.tensor([5.0, 5.0]))


class TestPairedRejectionStreaming:
    def test_matched_pairs_drain_into_residuals(self):
        store = SimpleNamespace()
        a = torch.tensor([[1.0, 1.0]])
        b = torch.tensor([[1.0, 0.0]])
        _accumulate_streaming_paired_rejection(torch.cat([a, b]), torch.tensor([0, 1]), torch.ones(2), (2,), store, 0)
        # residual of ([1,1] rejected on [1,0]) is [0,1]
        assert torch.allclose(store.concept_running_residual_sum, torch.tensor([0.0, 1.0]))
        assert torch.allclose(store.concept_running_pair_weight, torch.tensor(1.0))
        assert store.concept_pending_a_states.shape[0] == 0
        assert store.concept_pending_b_states.shape[0] == 0

    def test_unmatched_rows_wait_for_future_batches(self):
        store = SimpleNamespace()
        _accumulate_streaming_paired_rejection(
            torch.tensor([[1.0, 1.0], [2.0, 2.0]]), torch.tensor([0, 0]), torch.ones(2), (2,), store, 0
        )
        assert getattr(store, "concept_running_residual_sum", None) is None
        assert store.concept_pending_a_states.shape[0] == 2
        _accumulate_streaming_paired_rejection(
            torch.tensor([[1.0, 0.0]]), torch.tensor([1]), torch.ones(1), (2,), store, 1
        )
        # One (a, b) prefix pair matched and drained; second group-a row still pending.
        assert torch.allclose(store.concept_running_residual_sum, torch.tensor([0.0, 1.0]))
        assert store.concept_pending_a_states.shape[0] == 1

    def test_drain_noop_without_pending_buffers(self):
        store = SimpleNamespace()
        _drain_streaming_paired_buffers(store, (2,))
        assert getattr(store, "concept_running_residual_sum", None) is None


class TestResolveStreamingGroupNames:
    def test_explicit_names_win(self):
        batch = _batch(torch.ones(1, 2), torch.tensor([0]), concept_group_a_name="yes", concept_group_b_name="no")
        assert _resolve_streaming_group_names(_module_without_cfg(), batch, torch.tensor([0])) == ("yes", "no")

    def test_names_resolved_from_per_batch_group_names(self):
        batch = _batch(torch.ones(2, 2), torch.tensor([0, 1]), concept_group_name=["alpha", "beta"])
        assert _resolve_streaming_group_names(_module_without_cfg(), batch, torch.tensor([0, 1])) == ("alpha", "beta")

    def test_defaults_without_any_names(self):
        batch = _batch(torch.ones(1, 2), torch.tensor([0]))
        assert _resolve_streaming_group_names(_module_without_cfg(), batch, torch.tensor([0])) == (
            "group_a",
            "group_b",
        )


class TestConceptDirectionStreaming:
    def test_mean_difference_converges_to_weighted_full_batch_result(self):
        store = SimpleNamespace()
        batches = [
            (torch.tensor([[2.0, 0.0], [0.0, 4.0]]), torch.tensor([0, 1]), torch.tensor([1.0, 1.0])),
            (torch.tensor([[4.0, 0.0], [0.0, 8.0]]), torch.tensor([0, 1]), torch.tensor([3.0, 1.0])),
        ]
        for batch_idx, (states, gids, weights) in enumerate(batches):
            result = _concept_direction_streaming(
                _module_without_cfg(), _batch(states, gids, weights), batch_idx, store
            )
        all_states = torch.cat([b[0] for b in batches])
        all_gids = torch.cat([b[1] for b in batches])
        all_weights = torch.cat([b[2] for b in batches])
        mean_a = (all_states[all_gids == 0] * all_weights[all_gids == 0].unsqueeze(-1)).sum(0) / all_weights[
            all_gids == 0
        ].sum()
        mean_b = (all_states[all_gids == 1] * all_weights[all_gids == 1].unsqueeze(-1)).sum(0) / all_weights[
            all_gids == 1
        ].sum()
        expected = mean_a - mean_b
        expected = expected / torch.linalg.vector_norm(expected)
        assert torch.allclose(result.concept_direction, expected)
        assert result.concept_aggregate_output_mode == "streaming"
        assert result.concept_label == "group_a -> group_b"

    def test_single_group_mode(self):
        store = SimpleNamespace()
        result = _concept_direction_streaming(
            _module_without_cfg(),
            _batch(torch.tensor([[3.0, 4.0]]), torch.tensor([0]), concept_direction_mode="single_group"),
            0,
            store,
        )
        assert torch.allclose(result.concept_direction, torch.tensor([0.6, 0.8]))
        assert result.concept_label == "group_a"

    def test_paired_rejection_emits_zero_direction_until_pairs_match(self):
        store = SimpleNamespace()
        result = _concept_direction_streaming(
            _module_without_cfg(),
            _batch(torch.tensor([[1.0, 1.0]]), torch.tensor([0]), concept_direction_mode="paired_rejection"),
            0,
            store,
        )
        assert torch.equal(result.concept_direction, torch.zeros(2))
        result = _concept_direction_streaming(
            _module_without_cfg(),
            _batch(torch.tensor([[1.0, 0.0]]), torch.tensor([1]), concept_direction_mode="paired_rejection"),
            1,
            store,
        )
        assert torch.allclose(result.concept_direction, torch.tensor([0.0, 1.0]))

    def test_missing_group_a_raises(self):
        with pytest.raises(ValueError, match="at least one group A example"):
            _concept_direction_streaming(
                _module_without_cfg(), _batch(torch.ones(1, 2), torch.tensor([1])), 0, SimpleNamespace()
            )

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported concept_direction_mode"):
            _concept_direction_streaming(
                _module_without_cfg(),
                _batch(torch.ones(1, 2), torch.tensor([0]), concept_direction_mode="bogus"),
                0,
                SimpleNamespace(),
            )


class TestResetConceptStreamingState:
    def test_clears_all_contract_fields(self):
        store = SimpleNamespace(**{name: torch.ones(1) for name in CONCEPT_STREAMING_STATE_FIELDS})
        reset_concept_streaming_state(store)
        for name in CONCEPT_STREAMING_STATE_FIELDS:
            assert not hasattr(store, name)

    def test_none_store_is_a_noop(self):
        reset_concept_streaming_state(None)
