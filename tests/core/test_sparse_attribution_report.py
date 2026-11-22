"""Signed checkpoint reports retain their measurement denominator and expose missing attribution."""

from dataclasses import replace
import json

import pytest

from it_examples.utils.sparse_attribution_report import SparseAttributionRow, checkpoint_attribution_report

CHECKPOINT = "a" * 64
BASIS = "transcoders@immutable-revision"


def rows():
    return [
        SparseAttributionRow(
            "positive",
            CHECKPOINT,
            BASIS,
            2.0,
            1,
            [(0, 1, 3), (1, 0, 7)],
            [1.0, 2.0],
            [[-1.0, 0.0], [0.0, 1.0]],
            [0.5, 0.5],
            [0, 1],
            [[1.0, 0.0], [0.0, 2.0]],
            [3.0, 2.0],
            4.0,
            -2.0,
        ),
        SparseAttributionRow(
            "negative",
            CHECKPOINT,
            BASIS,
            -8.0,
            1,
            [(0, 1, 3), (1, 0, 7)],
            [-5.0, 1.0],
            [[-1.0, 0.0], [0.0, -2.0]],
            [-0.5, -0.5],
            [0, 1],
            [[-4.0, 0.0], [1.0, -3.0]],
            [-4.0, -6.0],
            -8.0,
            0.0,
        ),
    ]


def report(measurements=None, **kwargs):
    return checkpoint_attribution_report(
        rows() if measurements is None else measurements,
        checkpoint_sha256=CHECKPOINT,
        basis_id=BASIS,
        model_id="model",
        model_revision="revision",
        target_token_ids=(10, 11),
        **kwargs,
    )


def test_unequal_decisions_use_a_ratio_of_sums_and_keep_opposing_sources():
    actual = report()
    assert actual["sum_abs_decision"] == 10.0
    assert actual["source_shares"] == pytest.approx(
        {"features": 0.7, "errors": 0.3, "embeddings": 0.2, "bias": -0.2, "unassigned": 0.0}
    )
    # Averaging the two item-wise feature shares gives 1.0, an incorrect but plausible answer.
    assert actual["source_shares"]["features"] != (3 / 2 + (-4) / (-8)) / 2
    assert actual["feature_shares_by_layer"] == pytest.approx([0.6, 0.1])
    assert actual["error_shares_by_layer"] == pytest.approx([0.0, 0.3])
    assert actual["top_answer_features"] == [{"layer": 0, "feature": 3, "share": 0.6}]
    assert json.loads(json.dumps(actual, allow_nan=False)) == actual


def test_head_layers_remain_separate_partitions_of_source_edges():
    actual = report()
    assert actual["head_shares"][0] == pytest.approx([0.5, 0.0])
    assert actual["head_shares"][1] == pytest.approx([-0.1, 0.5])
    assert actual["bypass_shares"] == pytest.approx([0.7, 0.8])
    assert actual["complete_head_layers"] is True
    # Each layer partitions 1.2 times the decision because the measured bias contributes -0.2.
    for heads, bypass in zip(actual["head_shares"], actual["bypass_shares"]):
        assert sum(heads) + bypass == pytest.approx(1.2)


def test_unknown_bias_remains_unknown_and_the_remainder_is_visible():
    actual = report([replace(row, bias=None) for row in rows()])
    assert actual["source_shares"]["bias"] is None
    assert actual["source_shares"]["unassigned"] == pytest.approx(-0.2)
    assert actual["source_shares"]["features"] == pytest.approx(0.7)


def test_the_measured_attribution_regime_is_recorded():
    assert report()["attribution_regime"] == {"freeze_attention": True, "freeze_norms": True}
    measurements = [replace(row, freeze_attention=False) for row in rows()]
    assert report(measurements)["attribution_regime"] == {"freeze_attention": False, "freeze_norms": True}


def test_mixed_attribution_regimes_are_refused():
    measurements = rows()
    measurements[1] = replace(measurements[1], freeze_attention=False)
    with pytest.raises(ValueError, match="same attribution regime"):
        report(measurements)


def test_regime_flags_must_be_boolean():
    with pytest.raises(ValueError, match="boolean freeze_attention"):
        report([replace(row, freeze_norms=1) for row in rows()])


def test_partial_head_coverage_is_explicit():
    measurements = [
        replace(row, head_layers=[1], head_edges=[row.head_edges[1]], bypass_edges=[row.bypass_edges[1]])
        for row in rows()
    ]
    actual = report(measurements)
    assert actual["head_layers"] == [1]
    assert actual["complete_head_layers"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_edges", [1.0]),
        ("feature_nodes", [(0, 1, 3), (0, 1, 3)]),
        ("feature_nodes", [(2, 1, 3), (1, 0, 7)]),
        ("feature_nodes", [(0, 2, 3), (1, 0, 7)]),
        ("error_edges", [[0.0], [0.0]]),
        ("embedding_edges", [0.5]),
        ("head_layers", [0, 0]),
        ("head_edges", [[1.0], [2.0]]),
        ("bypass_edges", [3.0]),
        ("decision", 0.0),
        ("decision", float("nan")),
        ("edge_total", float("inf")),
        ("bias", float("nan")),
        ("answer_position", 2),
        ("checkpoint_sha256", "b" * 64),
        ("basis_id", "different-basis"),
    ],
)
def test_incomplete_or_incompatible_measurements_are_rejected(field, value):
    measurements = rows()
    measurements[1] = replace(measurements[1], **{field: value})
    with pytest.raises(ValueError):
        report(measurements)


@pytest.mark.parametrize("field,value", [("bypass_edges", [4.0, 2.0]), ("feature_edges", [2.0, 2.0])])
def test_a_planted_partition_defect_cannot_produce_a_report(field, value):
    measurements = rows()
    measurements[0] = replace(measurements[0], **{field: value})
    with pytest.raises(ValueError, match="partition"):
        report(measurements)


def test_duplicate_rows_and_mixed_bias_evidence_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        report([rows()[0], rows()[0]])
    with pytest.raises(ValueError, match="either measure bias"):
        report([rows()[0], replace(rows()[1], bias=None)])


def test_an_empty_checkpoint_is_not_a_successful_report():
    with pytest.raises(ValueError, match="at least one"):
        report([])


def test_prompt_lengths_can_differ_without_changing_the_source_denominator():
    measurements = rows()
    measurements[1] = replace(
        measurements[1],
        answer_position=0,
        embedding_edges=[-1.0],
        error_edges=[[-1.0], [-2.0]],
        feature_nodes=[(0, 0, 3), (1, 0, 7)],
    )
    actual = report(measurements)
    assert actual["source_shares"]["features"] == pytest.approx(0.7)
    assert actual["top_answer_features"] == [
        {"layer": 0, "feature": 3, "share": 0.6},
        {"layer": 1, "feature": 7, "share": -0.1},
    ]


def test_finite_inputs_cannot_overflow_into_a_nonfinite_json_report():
    measurements = [replace(row, decision=1e308) for row in rows()]
    with pytest.raises(ValueError, match="Out of range float values"):
        report(measurements)
