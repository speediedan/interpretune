"""One schedule written in venngram addresses tunes the core/HF, TransformerLens and nnsight compositions of GPT-2,
and each phase thaws the same tensors of the model on all three (#1054).

The ``speediedan/rte`` component ships the schedule (``ft_schedules/rte_ft.gpt2.addresses.yaml``) and one CLI run
configuration per composition (``runs/rte_ft.gpt2.{core.addresses,transformer_lens,nnsight}.yaml``). Each is run as
shipped, plus a recorder and CPU-sized limits, and each epoch's thawed parameters are mapped to the names of the same
tensors in GPT-2 itself: a TransformerLens composition holds the model's tensors under the bridge's paths (and reads
GPT-2's fused query/key/value projection through splits of it), nnsight's under ``hf_model``, the core composition's
under ``model``. The mapped sets must be equal in every phase, and every thawed tensor must change in every epoch it is
thawed (a tensor thawed where no step reaches it matches by name and trains nothing); a difference is the example's
failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.runif import RunIf

pytest.importorskip(
    "finetuning_scheduler.core.tuning_scheduler", reason="finetuning-scheduler predates TuningScheduler"
)
pytest.importorskip("transformer_lens")
pytest.importorskip("nnsight")

RTE_COMPONENT_DIR = Path(__file__).resolve().parents[2] / "src" / "it_examples" / "examples" / "rte"
LIMITS = {"max_epochs": 6, "limit_train_batches": 2, "limit_val_batches": 2}
# a large `min_delta` makes the second phase plateau within a CPU-sized run, as in the core/Lightning comparison
MIN_DELTA = 2.0
RUNS = ("core.addresses", "transformer_lens", "nnsight")


@pytest.fixture
def rte_component(monkeypatch):
    """The revision of the in-tree ``rte`` component, published locally with the cached ``rte_boolq.py``."""
    from interpretune.hub.components import local_publish
    from tests.rte_component import rte_entrypoint_src

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "1")
    return local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=rte_entrypoint_src())


def _run(run: str, revision: str, tmp_path: Path, monkeypatch) -> dict:
    from interpretune.base.components.cli import core_cli_main
    from interpretune.hub.components import resolve_component_file

    work = tmp_path / run
    work.mkdir()
    monkeypatch.chdir(work)  # checkpoints are written relative to the working directory
    record = work / "record.json"
    shipped = resolve_component_file(f"hub://speediedan/rte@{revision}/runs/rte_ft.gpt2.{run}.yaml")
    text = shipped.read_text(encoding="utf-8").replace("hub://speediedan/rte/", f"hub://speediedan/rte@{revision}/")
    cfg = yaml.safe_load(text)
    cfg["session_ref"] = f"{cfg['session_ref']}@{revision}"
    callbacks = cfg["run_cfg"]["callbacks"]
    callbacks[0]["init_args"]["tuning_scheduler_kwargs"]["transition"]["init_args"]["min_delta"] = MIN_DELTA
    callbacks.append(
        {"class_path": "tests.core.rte_ft_record.AddressedEpochRecorder", "init_args": {"out": str(record)}}
    )
    config = work / f"rte_ft.gpt2.{run}.yaml"
    config.write_text(yaml.safe_dump(cfg))
    args = ["--config", str(config)]
    args += ["--registered_cfg", json.dumps({"datamodule_cfg": {"dataset_path": str(work / "dataset")}})]
    args += [f"--run_cfg.{k}={v}" for k, v in LIMITS.items()]
    core_cli_main(run_mode="train", args=args)
    return json.loads(record.read_text())


# a process of its own: it trains GPT-2 three times, once traced by nnsight (whose trace replaces coverage's trace
# function), and the main suite already runs close to a hosted runner's memory
@RunIf(standalone_cpu=True)
def test_one_address_schedule_thaws_the_same_tensors_on_every_composition(
    rte_component, clean_cli_env, tmp_path, monkeypatch
):
    results = {run: _run(run, rte_component, tmp_path, monkeypatch) for run in RUNS}
    # each composition's own losses decide when a phase plateaus, so the thawed sets are compared phase by phase
    by_phase = {}
    for run, result in results.items():
        phases: dict[int, list[str]] = {}
        for epoch in result["epochs"]:
            assert phases.setdefault(epoch["depth"], epoch["wrapped"]) == epoch["wrapped"], (run, epoch["epoch"])
            # every thawed tensor is trained: one thawed where no step reaches it would still match by name
            assert epoch["stepped"] == epoch["wrapped"], (run, epoch["epoch"])
        assert not any(n.startswith("<unwrapped>") for names in phases.values() for n in names), run
        by_phase[run] = phases
    core = by_phase["core.addresses"]
    # phase 0 is the final norm and blocks 10-11's MLPs, phase 1 adds their attention and norms, phase 2 blocks 6-9
    assert {depth: len(names) for depth, names in core.items()} == {0: 10, 1: 26, 2: 74}
    for run in RUNS:
        assert by_phase[run] == core, run
    assert {results[run]["final_depth"] for run in RUNS} == {2}
