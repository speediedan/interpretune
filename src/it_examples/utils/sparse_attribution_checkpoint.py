"""Bind attribution reports to the actual model selected after the tuning scheduler's terminal gates.

A checkpoint policy's best path can still name a rejected candidate after a veto. The callback saves the live selected
state after FTSCoreCallback ends training, then checks all checkpoint tensors against the view before and after
measurement. A measurement driver must independently check its reconstructed model's logits and basis; this module
proves the selected training state and exact validation coverage, not the driver's reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from it_examples.utils.sparse_attribution_report import SparseAttributionRow, checkpoint_attribution_report


@dataclass(frozen=True)
class VerifiedAttributionCheckpoint:
    """A checkpoint file whose complete tensor state matches the selected training view."""

    path: Path
    sha256: str
    state_sha256: str
    state_keys: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointMeasurements:
    """A driver's measurements and the complete validation identity set it was asked to measure."""

    rows: Sequence[SparseAttributionRow]
    expected_row_ids: Sequence[str]
    basis_id: str
    model_id: str
    model_revision: str
    target_token_ids: tuple[int, int]
    checks: Mapping[str, Any] = field(default_factory=dict)


def tensor_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash every named tensor's key, dtype, shape and bytes, independent of mapping order or device.

    Dense, materialized tensors are required. Unsupported layouts and non-tensor state are refused rather than omitted,
    because a digest of only part of the model would not identify its selected state.
    """
    import torch

    if not state or any(not isinstance(name, str) or not name for name in state):
        raise ValueError("selected state must contain nonempty tensor names")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.is_quantized
            or tensor.device.type == "meta"
        ):
            raise ValueError(f"selected state tensor {name!r} must be dense and materialized")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(memoryview(raw.numpy()))
    return digest.hexdigest()


def verify_attribution_checkpoint(path: str | Path, selected_state: Mapping[str, Any]) -> VerifiedAttributionCheckpoint:
    """Verify a complete, unsharded FTS checkpoint against the actual selected view, without restoring anything.

    Scheduled-only checkpoints cannot prove the unscheduled base and are refused. Callers supply
    ``tuning_scheduler.view.checkpoint_state_dict()`` so fitted tensors are included and artifacts are excluded by
    the same view that wrote the checkpoint. These are trusted local training checkpoints, read with FTS's loader.
    """
    from finetuning_scheduler.core.checkpoint import load_payload

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"attribution checkpoint must be an unsharded file, got {path}")
    with path.open("rb") as stream:
        file_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = load_payload(path)
    if payload.get("version") != 1 or payload.get("sharded") is not False or payload.get("include") != "all":
        raise ValueError("attribution needs a version-1, unsharded FTS checkpoint with include='all'")
    saved = payload.get("model")
    if not isinstance(saved, Mapping) or set(saved) != set(selected_state):
        raise ValueError("checkpoint and selected state have different tensor keys")
    selected_sha = tensor_state_sha256(selected_state)
    if tensor_state_sha256(saved) != selected_sha:
        raise ValueError("checkpoint tensors do not match the selected model state")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != file_sha:
            raise ValueError("attribution checkpoint changed while it was verified")
    return VerifiedAttributionCheckpoint(path, file_sha, selected_sha, tuple(sorted(saved)))


def _complete_validation_coverage(measurements: CheckpointMeasurements) -> list[str]:
    expected = list(measurements.expected_row_ids)
    actual = [row.row_id for row in measurements.rows]
    if not expected or any(not isinstance(row_id, str) or not row_id for row_id in expected):
        raise ValueError("expected validation row ids must be nonempty strings")
    if len(set(expected)) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("measured rows must match the complete expected validation identity set exactly")
    return expected


def _report_summary(report: Mapping[str, Any], destination: Path) -> str:
    source = report["source_shares"]
    heads = [
        (layer, head, value)
        for layer, values in zip(report["head_layers"], report["head_shares"])
        for head, value in enumerate(values)
    ]
    largest = max(heads, key=lambda entry: abs(entry[2]), default=None)
    head = f"largest head L{largest[0]} H{largest[1]} {largest[2]:+.4f}" if largest else "no measured heads"
    features = report["top_answer_features"]
    top = (
        f"top answer feature L{features[0]['layer']} F{features[0]['feature']} {features[0]['share']:+.4f}"
        if features
        else "no active answer features"
    )
    bias = "unmeasured" if source["bias"] is None else f"{source['bias']:+.4f}"
    return (
        f"Selected-checkpoint attribution: {report['items']} validation items; "
        f"feature share {source['features']:+.4f}, error share {source['errors']:+.4f}, "
        f"embedding share {source['embeddings']:+.4f}, bias {bias}, unassigned {source['unassigned']:+.4f}; "
        f"{head}; {top}; {destination}"
    )


class SparseAttributionReportCallback:
    """Save and report the live selected checkpoint after FTSCoreCallback's final restore and gates.

    Args:
        collector: A callable, or a ``class_path`` mapping constructing one. It receives the training module and a
            verified checkpoint and returns CheckpointMeasurements. Its responsibilities include reconstructing the
            measurement model, checking its logits against the selected training model, and identifying the actual
            transcoder basis. A repeated checkpoint hash in its rows is not reconstruction evidence.
        output_dir: Parent for a separate checkpoint/report directory per completed run. The selected checkpoint is
            a full FTS checkpoint with optimizer state. Publication follows successful collection and verification.
        top_k: Answer-position features retained in the aggregate report.

    The callback must follow exactly one FTSCoreCallback. Only single-process, unsharded runs with a complete
    checkpoint view are supported. ``module.sparse_attribution_report_path`` names the JSON after it is published.
    """

    def __init__(
        self,
        collector: Callable[[Any, VerifiedAttributionCheckpoint], CheckpointMeasurements] | Mapping[str, Any],
        output_dir: str = "reports/sparse_attribution",
        top_k: int = 10,
    ) -> None:
        if isinstance(collector, Mapping):
            from interpretune.hub.entrypoints import instantiate_hub_aware_class

            collector = instantiate_hub_aware_class(dict(collector))
        if not callable(collector):
            raise ValueError("attribution collector must be callable")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.collector = collector
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        self._fts: Any = None

    def on_train_start(self, ctx: Any) -> None:
        """Check callback order before training, and refuse distributed collection until it is supported."""
        from interpretune.tuning.core_loop import FTSCoreCallback

        callbacks = list(ctx.callbacks)
        matches = [cb for cb in callbacks if isinstance(cb, FTSCoreCallback)]
        if len(matches) != 1 or self not in callbacks or callbacks.index(matches[0]) >= callbacks.index(self):
            raise ValueError("SparseAttributionReportCallback must follow exactly one FTSCoreCallback")
        if ctx.dist_run is not None:
            raise ValueError("sparse attribution reports currently require a single-process unsharded run")
        self._fts = matches[0]
        ctx.module.sparse_attribution_report_path = None

    def on_train_end(self, ctx: Any) -> None:
        """Save the post-gate subject, measure it, recheck state identity, then atomically publish its report."""
        scheduler = self._fts.tuning_scheduler if self._fts is not None else None
        if scheduler is None or not scheduler.state_dict().get("finished"):
            raise ValueError("attribution reporting requires FTSCoreCallback to finish its terminal gates first")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="selected-", dir=self.output_dir))
        checkpoint = Path(scheduler.save(str(run_dir / "checkpoint.pt")))
        verified = verify_attribution_checkpoint(checkpoint, scheduler.view.checkpoint_state_dict())
        measurements = self.collector(ctx.module, verified)
        expected = _complete_validation_coverage(measurements)
        report = checkpoint_attribution_report(
            measurements.rows,
            checkpoint_sha256=verified.sha256,
            basis_id=measurements.basis_id,
            model_id=measurements.model_id,
            model_revision=measurements.model_revision,
            target_token_ids=measurements.target_token_ids,
            top_k=self.top_k,
        )
        if verify_attribution_checkpoint(checkpoint, scheduler.view.checkpoint_state_dict()) != verified:
            raise ValueError("selected attribution checkpoint changed during measurement")
        report["checkpoint_state_sha256"] = verified.state_sha256
        report["checkpoint_tensor_count"] = len(verified.state_keys)
        report["validation_row_ids"] = expected
        report["validation_coverage_verified"] = True
        report["checkpoint_selection"] = "live_after_terminal_gates"
        report["gate_records"] = scheduler.gate_records
        report["measurement_checks"] = dict(measurements.checks)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        destination = run_dir / "attribution.json"
        pending = run_dir / "attribution.json.tmp"
        pending.write_text(payload, encoding="utf-8")
        pending.replace(destination)
        ctx.module.sparse_attribution_report_path = str(destination)
        from interpretune.utils import rank_zero_info

        rank_zero_info(_report_summary(report, destination))
