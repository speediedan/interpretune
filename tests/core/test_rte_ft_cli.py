"""The RTE phased fine-tuning example (#765) runs from the interpretune CLI alone, on the core loop and on
Lightning, with the same outcome (#847).

The ``speediedan/rte`` component ships the session configuration (``rte_ft.gpt2.core``), the schedule
(``ft_schedules/rte_ft.gpt2.yaml``) and one CLI run configuration per training-loop interface
(``runs/rte_ft.gpt2.{core,lightning}.yaml``): ``FTSCoreCallback`` with a ``PlateauTransition`` and a ``BestCheckpoint``,
and the ``FinetuningScheduler`` callback with ``FTSEarlyStopping`` and ``FTSCheckpoint``. Each run configuration is run
as shipped, plus a recorder callback and CPU-sized limits. Both must thaw the same parameters each epoch, transition in
the same epochs, and keep the same best checkpoint. The run is long enough for one transition by
``max_transition_epoch`` and one by a validation-loss plateau.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip(
    "finetuning_scheduler.core.tuning_scheduler", reason="finetuning-scheduler predates TuningScheduler"
)
pytest.importorskip("lightning")

RTE_COMPONENT_DIR = Path(__file__).resolve().parents[2] / "src" / "it_examples" / "examples" / "rte"
LIMITS = {"max_epochs": 6, "limit_train_batches": 2, "limit_val_batches": 2}
# a large `min_delta` makes the second phase plateau within a CPU-sized run
MIN_DELTA = 2.0


@pytest.fixture
def rte_component(monkeypatch):
    """The revision of the in-tree ``rte`` component, published locally with the cached ``rte_boolq.py``.

    Every reference the run makes names this revision: a cache warmed at a pinned Hub revision (as CI's is) resolves an
    unrevisioned reference to the pin, which need not carry ``rte_ft``.
    """
    from interpretune.hub.components import local_publish
    from tests.rte_component import rte_entrypoint_src

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "1")
    return local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=rte_entrypoint_src())


def _run_config(interface: str, revision: str, tmp_path: Path, record: Path) -> Path:
    """The shipped run configuration for ``interface`` at ``revision``, with its references pinned to that
    revision, a recorder and the CPU-sized plateau threshold."""
    from interpretune.hub.components import resolve_component_file

    shipped = resolve_component_file(f"hub://speediedan/rte@{revision}/runs/rte_ft.gpt2.{interface}.yaml")
    text = shipped.read_text(encoding="utf-8").replace("hub://speediedan/rte/", f"hub://speediedan/rte@{revision}/")
    cfg = yaml.safe_load(text)
    cfg["session_ref"] = f"{cfg['session_ref']}@{revision}"
    if interface == "core":
        callbacks = cfg["run_cfg"]["callbacks"]
        callbacks[0]["init_args"]["tuning_scheduler_kwargs"]["transition"]["init_args"]["min_delta"] = MIN_DELTA
        recorder = "tests.core.rte_ft_record.CoreEpochRecorder"
    else:
        callbacks = cfg["trainer"]["callbacks"]
        callbacks[1]["init_args"]["min_delta"] = MIN_DELTA
        recorder = "tests.core.rte_ft_record.LightningEpochRecorder"
    callbacks.append({"class_path": recorder, "init_args": {"out": str(record)}})
    path = tmp_path / f"rte_ft.gpt2.{interface}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _run(interface: str, revision: str, tmp_path: Path, monkeypatch, bf16_mixed: bool = False) -> dict:
    from interpretune.base.components.cli import core_cli_main, l_cli_main

    work = tmp_path / interface
    work.mkdir()
    monkeypatch.chdir(work)  # checkpoints are written relative to the working directory
    record = work / "record.json"
    args = ["--config", str(_run_config(interface, revision, work, record))]
    args += ["--registered_cfg", json.dumps({"datamodule_cfg": {"dataset_path": str(work / "dataset")}})]
    if interface == "core":
        args += [f"--run_cfg.{k}={v}" for k, v in LIMITS.items()]
        if bf16_mixed:
            args += ["--run_cfg.precision=bf16-mixed", "--run_cfg.accumulate_grad_batches=2"]
        core_cli_main(run_mode="train", args=args)
    else:
        args += [f"--trainer.{k}={v}" for k, v in LIMITS.items()]
        if bf16_mixed:
            args += ["--trainer.precision=bf16-mixed", "--trainer.accumulate_grad_batches=2"]
        l_cli_main(run_mode=True, args=["fit", *args])
    result = json.loads(record.read_text())
    epochs = result["epochs"]
    result["transition_epochs"] = [e["epoch"] for prev, e in zip(epochs, epochs[1:]) if e["depth"] != prev["depth"]]
    return result


def test_the_core_loop_and_lightning_agree_phase_by_phase(rte_component, clean_cli_env, tmp_path, monkeypatch):
    core = _run("core", rte_component, tmp_path, monkeypatch)
    lightning = _run("lightning", rte_component, tmp_path, monkeypatch)
    # phase 0 ends at its `max_transition_epoch` (2); phase 1 on a validation-loss plateau
    assert core["transition_epochs"] == lightning["transition_epochs"] == [2, 5]
    assert core["final_depth"] == lightning["final_depth"] == 2
    assert core["best_depth"] == lightning["best_depth"] == 2
    assert [e["thawed"] for e in core["epochs"]] == [e["thawed"] for e in lightning["epochs"]]
    assert [len(e["thawed"]) for e in core["epochs"]] == [26, 26, 74, 74, 74, 148]
    assert [e["val_loss"] for e in core["epochs"]] == pytest.approx(
        [e["val_loss"] for e in lightning["epochs"]], rel=1e-5
    )
    assert core["best_score"] == pytest.approx(lightning["best_score"], rel=1e-5)


def test_the_hosts_agree_in_bf16_mixed_with_accumulation(rte_component, clean_cli_env, tmp_path, monkeypatch):
    """#697's configuration on both hosts (bf16-mixed, two-step accumulation), which the core loop could not
    express before #836: the same thawed sets and transition epochs, and validation loss within bf16 tolerance
    rather than exact.

    With accumulation over the CPU-sized limit of 2 batches, each epoch is one optimizer step on the two batches' mean
    loss rather than two steps, so the loss trajectory differs from the fp32 run's and the plateau transition lands an
    epoch earlier (4, not 5).
    """
    core = _run("core", rte_component, tmp_path, monkeypatch, bf16_mixed=True)
    lightning = _run("lightning", rte_component, tmp_path, monkeypatch, bf16_mixed=True)
    assert core["transition_epochs"] == lightning["transition_epochs"] == [2, 4]
    assert core["final_depth"] == lightning["final_depth"] == 2
    assert core["best_depth"] == lightning["best_depth"] == 2
    assert [e["thawed"] for e in core["epochs"]] == [e["thawed"] for e in lightning["epochs"]]
    assert [len(e["thawed"]) for e in core["epochs"]] == [26, 26, 74, 74, 148, 148]
    assert [e["val_loss"] for e in core["epochs"]] == pytest.approx(
        [e["val_loss"] for e in lightning["epochs"]], rel=1e-3
    )
