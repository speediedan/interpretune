"""Checkpoint reporting observes the actual terminal selection, including a veto's rollback and fitted state."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import torch

from it_examples.utils.sparse_attribution_checkpoint import (
    CheckpointMeasurements,
    SparseAttributionReportCallback,
    tensor_state_sha256,
    verify_attribution_checkpoint,
)
from it_examples.utils.sparse_attribution_report import SparseAttributionRow
from interpretune.tuning.core_loop import FTSCoreCallback
from tests.core.test_core_loop_training import TinyModule, _train

pytest.importorskip("finetuning_scheduler.core.tuning_scheduler", reason="checkpoint reporting requires core FTS")


def _checkpoint(path, state, **metadata):
    torch.save({"version": 1, "sharded": False, "include": "all", "model": state, **metadata}, path)
    return path


def test_a_checkpoint_digest_covers_tensor_names_shapes_dtypes_and_values(tmp_path):
    state = {"weight": torch.tensor([[1.0, 2.0]]), "fitted": torch.tensor(3, dtype=torch.int64)}
    path = _checkpoint(tmp_path / "selected.pt", state)
    proof = verify_attribution_checkpoint(path, dict(reversed(list(state.items()))))
    assert len(proof.sha256) == len(proof.state_sha256) == 64
    assert proof.state_keys == ("fitted", "weight")
    for changed in (
        {**state, "weight": state["weight"].flatten()},
        {**state, "weight": state["weight"].double()},
        {**state, "fitted": state["fitted"] + 1},
        {"renamed": state["weight"], "fitted": state["fitted"]},
    ):
        with pytest.raises(ValueError, match="different tensor keys|do not match"):
            verify_attribution_checkpoint(path, changed)


def test_bfloat16_and_noncontiguous_views_are_hashed_without_changing_their_values():
    value = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).T
    assert tensor_state_sha256({"value": value}) == tensor_state_sha256({"value": value.contiguous()})
    assert tensor_state_sha256({"value": value}) != tensor_state_sha256({"value": value + 1})


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"": torch.ones(1)},
        {"not_a_tensor": 1},
        {"meta": torch.empty(2, device="meta")},
        {"sparse": torch.eye(2).to_sparse()},
        {"quantized": torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.qint8)},
    ],
)
def test_partial_or_unsupported_state_is_refused(state):
    with pytest.raises(ValueError, match="nonempty tensor names|dense and materialized"):
        tensor_state_sha256(state)


@pytest.mark.parametrize("metadata", [{"include": "scheduled"}, {"sharded": True}, {"version": 2}])
def test_a_checkpoint_format_cannot_be_assumed(tmp_path, metadata):
    state = {"weight": torch.ones(1)}
    with pytest.raises(ValueError, match="version-1, unsharded.*include='all'"):
        verify_attribution_checkpoint(_checkpoint(tmp_path / "state.pt", state, **metadata), state)


def test_a_sharded_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unsharded file"):
        verify_attribution_checkpoint(tmp_path, {"weight": torch.ones(1)})


class ScalarMeasurements:
    """A measurement stand-in for callback tests, recording the live weights it received after selection.

    This is not circuit-tracer evidence; the public-op CPU test exercises real attribution separately.
    """

    def __init__(self, defect=None):
        self.defect = defect
        self.seen = None

    def __call__(self, module, checkpoint):
        self.seen = {name: value.clone() for name, value in module.state_dict().items()}
        with torch.no_grad():
            decision = float(module.model(torch.ones(1, 2))[0, 0])
        row = SparseAttributionRow(
            row_id="validation-1", checkpoint_sha256=checkpoint.sha256, basis_id="test-basis", decision=decision,
            answer_position=0, feature_nodes=[(0, 0, 0)], feature_edges=[decision / 2], error_edges=[[0.0]],
            embedding_edges=[0.0], head_layers=[0], head_edges=[[decision / 4, 0.0]],
            bypass_edges=[decision / 4], edge_total=decision / 2,
        )  # fmt: skip
        rows = [row]
        if self.defect == "missing-row":
            rows = []
        elif self.defect == "wrong-row":
            rows = [replace(row, row_id="other")]
        elif self.defect == "duplicate-row":
            rows = [row, row]
        elif self.defect == "wrong-checkpoint":
            rows = [replace(row, checkpoint_sha256="0" * 64)]
        elif self.defect == "mutate-fitted":
            module.fitted.add_(1)
        return CheckpointMeasurements(rows, ["validation-1"], "test-basis", "linear", "test", (0, 1))


class DecreasingValidation(TinyModule):
    """Make the second phase's trained weights the policy's best, so a final veto restores a different state."""

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", -float(self.current_epoch))


def _run(tmp_path, collector, *, veto=False, checkpoint_include: Literal["all", "scheduled"] = "all"):
    from finetuning_scheduler.core.checkpoint import BestCheckpoint
    from venngram.tuning import CanonicalView

    module = DecreasingValidation()
    module.register_buffer("fitted", torch.tensor(2.0))
    module.register_parameter("artifact", torch.nn.Parameter(torch.ones(1), requires_grad=False))
    view = CanonicalView(module, roles={"fitted": "fitted", "artifact": "artifact"})
    policy = BestCheckpoint(tmp_path / "training", "val_loss", save_initial=True, include=checkpoint_include)
    seen = []

    def gate(event):
        seen.append((event.final, event.depth, {k: v.clone() for k, v in module.state_dict().items()}))
        return not (veto and event.final)

    fts = FTSCoreCallback(
        {0: {"params": ["model.bias"], "max_transition_epoch": 1}, 1: {"params": ["model.weight"]}},
        tuning_scheduler_kwargs={
            "optimizer": partial(torch.optim.SGD, lr=0.05),
            "checkpoint": policy,
            "gates": [gate],
            "view": view,
        },
    )
    report = SparseAttributionReportCallback(collector, str(tmp_path / "reports"))
    _train(module, max_epochs=2, callbacks=[fts, report])
    assert fts.tuning_scheduler is not None
    return module, fts.tuning_scheduler, policy, seen


@pytest.mark.parametrize("veto", [False, True], ids=["accepted-best", "veto-rollback"])
def test_reporting_runs_after_the_actual_final_selection(tmp_path, veto):
    from finetuning_scheduler.core.checkpoint import load_payload

    collector = ScalarMeasurements()
    module, scheduler, policy, gates = _run(tmp_path, collector, veto=veto)
    saved_path = module.sparse_attribution_report_path
    assert isinstance(saved_path, str)
    report_path = Path(saved_path)
    report = json.loads(report_path.read_text())
    checkpoint = report_path.with_name("checkpoint.pt")
    proof = verify_attribution_checkpoint(checkpoint, scheduler.view.checkpoint_state_dict())
    assert report["checkpoint_sha256"] == proof.sha256
    assert report["checkpoint_state_sha256"] == proof.state_sha256
    assert report["checkpoint_tensor_count"] == 3  # weight, bias and fitted; artifact is intentionally excluded
    assert report["validation_row_ids"] == ["validation-1"] and report["validation_coverage_verified"] is True
    assert report["checkpoint_selection"] == "live_after_terminal_gates"
    assert report["gate_records"][-1]["accept"] is not veto
    assert gates[-1][0] is True
    assert collector.seen is not None
    assert all(torch.equal(collector.seen[k], v) for k, v in module.state_dict().items())
    assert policy.best_path is not None
    best = load_payload(policy.best_path)["model"]
    if veto:
        # The rejected candidate remains the policy's best. Reporting that path would silently name other weights.
        assert not torch.equal(best["model.weight"], module.model.weight)
        assert not torch.equal(gates[-1][2]["model.weight"], collector.seen["model.weight"])
    else:
        assert torch.equal(best["model.weight"], module.model.weight)


@pytest.mark.parametrize("defect", ["missing-row", "wrong-row", "duplicate-row", "wrong-checkpoint", "mutate-fitted"])
def test_bad_coverage_or_mutated_subject_cannot_publish_a_report(tmp_path, defect):
    with pytest.raises(ValueError, match="validation identity set|different checkpoint|do not match"):
        _run(tmp_path, ScalarMeasurements(defect))
    assert not list((tmp_path / "reports").rglob("attribution.json"))


def test_scheduled_only_checkpoints_cannot_claim_complete_model_identity(tmp_path):
    with pytest.raises(ValueError, match="include='all'"):
        _run(tmp_path, ScalarMeasurements(), checkpoint_include="scheduled")
    assert not list((tmp_path / "reports").rglob("attribution.json"))


def test_bad_callback_order_is_refused_before_collection(tmp_path):
    report = SparseAttributionReportCallback(ScalarMeasurements(), str(tmp_path))
    fts = FTSCoreCallback()
    for callbacks in ([report, fts], [report], [fts, fts, report]):
        with pytest.raises(ValueError, match="follow exactly one FTSCoreCallback"):
            report.on_train_start(SimpleNamespace(callbacks=callbacks, dist_run=None))


def test_distributed_collection_is_refused_explicitly(tmp_path):
    report = SparseAttributionReportCallback(ScalarMeasurements(), str(tmp_path))
    with pytest.raises(ValueError, match="single-process unsharded"):
        report.on_train_start(SimpleNamespace(callbacks=[FTSCoreCallback(), report], dist_run=object()))


def test_a_collector_can_be_built_from_run_configuration(tmp_path):
    report = SparseAttributionReportCallback({"class_path": f"{__name__}.ScalarMeasurements"}, str(tmp_path))
    assert isinstance(report.collector, ScalarMeasurements)
