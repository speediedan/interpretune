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
import os
import subprocess
import sys
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


def _run(interface: str, revision: str, tmp_path: Path, monkeypatch) -> dict:
    from interpretune.base.components.cli import core_cli_main, l_cli_main

    work = tmp_path / interface
    work.mkdir()
    monkeypatch.chdir(work)  # checkpoints are written relative to the working directory
    record = work / "record.json"
    args = ["--config", str(_run_config(interface, revision, work, record))]
    args += ["--registered_cfg", json.dumps({"datamodule_cfg": {"dataset_path": str(work / "dataset")}})]
    if interface == "core":
        args += [f"--run_cfg.{k}={v}" for k, v in LIMITS.items()]
        core_cli_main(run_mode="train", args=args)
    else:
        args += [f"--trainer.{k}={v}" for k, v in LIMITS.items()]
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


def _torchrun(revision: str, work: Path, cli: Path, meta_init: bool) -> list[dict]:
    """Each rank's record of the shipped core run, launched through the CLI under ``torchrun`` on two ranks with
    ``placement: fsdp2`` and a plan sharding each GPT-2 block."""
    work.mkdir()
    cfg = _run_config("core", revision, work, work / "record_rank{rank}.json")
    registered: dict = {"datamodule_cfg": {"dataset_path": str(work / "dataset")}}
    if meta_init:
        registered["module_cfg"] = {"init_args": {"hf_from_pretrained_cfg": {"init_args": {"meta_init": True}}}}
    args = [str(cli), "--run_command", "train", "--config", str(cfg), "--registered_cfg", json.dumps(registered)]
    args += [f"--run_cfg.{k}={v}" for k, v in LIMITS.items()]
    args += ["--run_cfg.placement=fsdp2", "--run_cfg.fsdp_plan", json.dumps({"class:GPT2Block": {}})]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", "--no-python", *args]
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    # each rank imports `tests.*` (the recorder), which a subprocess finds only through PYTHONPATH
    repo_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH")) if p)
    proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"torchrun failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return [json.loads((work / f"record_rank{rank}.json").read_text()) for rank in range(2)]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
def test_the_cli_runs_the_phased_core_loop_under_torchrun_with_an_fsdp2_plan(rte_component, clean_cli_env, tmp_path):
    """The shipped core run configuration, launched through the interpretune CLI under ``torchrun`` on two ranks
    with ``placement: fsdp2`` and a plan sharding each GPT-2 block, runs every phase as one process does, and does
    so identically when the model is built on the meta device and loaded from rank 0 (``meta_init``).

    Each rank thaws the same parameters each epoch and transitions in the same epochs as the one-process run above
    (whose thawed counts are asserted there), every parameter ends as a ``DTensor``, and the ranks agree on the
    validation metrics and the best checkpoint, since the loop averages metrics across ranks. The meta-built run's
    validation losses equal the plain run's exactly: its weights, its tied LM head and its buffers all came from rank
    0's copy, through every phase including the last, which trains the embeddings.
    """
    # the console script installed beside this interpreter, launched as the docs show (`torchrun --no-python`)
    cli = Path(sys.executable).with_name("interpretune")
    if not cli.exists():
        pytest.skip(f"no `interpretune` console script beside {sys.executable}")
    plain = _torchrun(rte_component, tmp_path / "plain", cli, meta_init=False)
    meta = _torchrun(rte_component, tmp_path / "meta", cli, meta_init=True)
    for record in (*plain, *meta):
        epochs = record["epochs"]
        transitions = [e["epoch"] for prev, e in zip(epochs, epochs[1:]) if e["depth"] != prev["depth"]]
        assert transitions == [2, 5]
        assert [len(e["thawed"]) for e in epochs] == [26, 26, 74, 74, 74, 148]
        assert record["final_depth"] == record["best_depth"] == 2
        assert record["param_types"] == ["DTensor"]
    for records in (plain, meta):
        assert [e["thawed"] for e in records[0]["epochs"]] == [e["thawed"] for e in records[1]["epochs"]]
        assert [e["val_loss"] for e in records[0]["epochs"]] == [e["val_loss"] for e in records[1]["epochs"]]
        assert records[0]["best_score"] == records[1]["best_score"]
    assert [e["val_loss"] for e in meta[0]["epochs"]] == [e["val_loss"] for e in plain[0]["epochs"]]
