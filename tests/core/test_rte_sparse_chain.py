"""The sparsity chain (#698 to #700) as one phased run of the ``speediedan/rte`` component's ``rte_sparse`` variant
(#899), CPU-sized: GPT-2 with random transcoders, run from the shipped run configuration through the interpretune
CLI.

The boundary's graphs come from a test-side stand-in (the research configuration builds them with circuit-tracer on a
GPU); everything else is the component's: the chain's row partition, the penalty and its calibrated normalizer, the
roles (transcoders ``instrument``, enforcement ``fitted``), the payload and fill hooks, and the enforced recovery
phases. Recovery re-thaws in two steps with finetuning-scheduler's per-phase ``refreeze`` (#936).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip(
    "finetuning_scheduler.core.tuning_scheduler", reason="finetuning-scheduler predates TuningScheduler"
)
pytest.importorskip("circuit_tracer", reason="the rte_sparse variant's transcoders are circuit-tracer's")

from tests.core.rte_sparse_stub import EnforcementProbe, StageRecorder  # noqa: E402

RTE_COMPONENT_DIR = Path(__file__).resolve().parents[2] / "src" / "it_examples" / "examples" / "rte"


@pytest.fixture
def rte_component(monkeypatch):
    """The in-tree ``rte`` component, published locally with the warmed ``rte_boolq.py``."""
    from interpretune.hub.components import local_publish
    from tests.rte_component import rte_entrypoint_src

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "1")
    return local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=rte_entrypoint_src())


def test_the_chain_partition_holds_the_graph_items_out_of_training(rte_component):
    import datasets

    from tests.rte_component import rte_entrypoint_module

    splits = rte_entrypoint_module().RTEBoolqChainDataModule.chain_splits(
        datasets.load_dataset("aps/super_glue", "rte")
    )
    sizes = {name: len(split) for name, split in splits.items()}
    assert sizes == {"train": 1860, "validation": 190, "graph": 440, "test": 277}
    idx = {name: set(splits[name]["idx"]) for name in ("train", "validation", "graph")}
    assert not (idx["train"] & idx["graph"]) and not (idx["train"] & idx["validation"])


def test_the_sparse_configurations_shuffle_training_as_the_chain_did(rte_component):
    """Without shuffling the split is read in one fixed order, and with fixed weights and no dropout the seed
    changes nothing (#899): three seeds gave one run."""
    from types import SimpleNamespace

    from torch.utils.data import RandomSampler, SequentialSampler

    from interpretune.hub.components import resolve_component_file
    from tests.rte_component import rte_entrypoint_module

    for model in ("gemma3", "gpt2"):
        shipped = resolve_component_file(f"hub://speediedan/rte@{rte_component}/configs/rte_sparse.{model}.core.yaml")
        datamodule_cfg = yaml.safe_load(shipped.read_text(encoding="utf-8"))["registered_cfg"]["datamodule_cfg"]
        assert datamodule_cfg["dataloader_kwargs"]["shuffle"] is True

    # the setting reaches the training sampler, and the evaluation splits still score every example in order
    data = SimpleNamespace(
        dataset={"train": list(range(8)), "validation": list(range(8))},
        data_collator=None,
        itdm_cfg=SimpleNamespace(dataloader_kwargs={"shuffle": True}, train_batch_size=2, eval_batch_size=2),
    )
    factory = rte_entrypoint_module().RTEBoolqChainDataModule.dataloader_factory
    assert isinstance(factory(data, "train", use_train_batch_size=True).sampler, RandomSampler)
    assert isinstance(factory(data, "validation").sampler, SequentialSampler)


def _run_cli(rte_component, run_dir, recorder=None, resume_from=None):
    """Run the shipped CPU configuration in ``run_dir``, with the test-side graph hook and probes."""
    from interpretune.base.components.cli import core_cli_main
    from interpretune.hub.components import resolve_component_file

    shipped = resolve_component_file(f"hub://speediedan/rte@{rte_component}/runs/rte_sparse.gpt2.core.yaml")
    cfg = yaml.safe_load(
        shipped.read_text(encoding="utf-8").replace("hub://speediedan/rte/", f"hub://speediedan/rte@{rte_component}/")
    )
    cfg["session_ref"] = f"{cfg['session_ref']}@{rte_component}"
    ts_kwargs = cfg["run_cfg"]["callbacks"][0]["init_args"]["tuning_scheduler_kwargs"]
    ts_kwargs["phase_hooks"]["end"] = [{"class_path": "tests.core.rte_sparse_stub.StubGraphs"}]
    ts_kwargs["phase_hooks"]["start"].append({"class_path": "tests.core.rte_sparse_stub.EnforcementProbe"})
    cfg["run_cfg"]["callbacks"].append(
        {"class_path": "tests.core.rte_sparse_stub.StageRecorder", "init_args": recorder or {}}
    )
    if resume_from is not None:
        cfg["run_cfg"]["resume_from"] = str(resume_from)
    (run_dir / "run.yaml").write_text(yaml.safe_dump(cfg))
    EnforcementProbe.seen, StageRecorder.seen = [], []
    args = ["--config", str(run_dir / "run.yaml")]
    args += ["--registered_cfg", json.dumps({"datamodule_cfg": {"dataset_path": str(run_dir.parent / "dataset")}})]
    args += ["--run_cfg.limit_train_batches=2", "--run_cfg.limit_val_batches=2"]
    core_cli_main(run_mode="train", args=args)
    return list(EnforcementProbe.seen), list(StageRecorder.seen)


def test_the_sparse_chain_runs_its_three_stages_and_resumes_after_the_refreeze(
    rte_component, clean_cli_env, tmp_path, monkeypatch
):
    """Sparsity, then a two-step re-thaw (#936): phase 1 refreezes blocks 9 and 10, phase 2 thaws them again, both
    holding the enforcement the sparsity boundary fitted.

    Resumed from the end of phase 1, the run thaws them again and ends where the uninterrupted one did, the equivalence
    finetuning-scheduler#193 makes its own acceptance test.
    """
    from finetuning_scheduler.core.checkpoint import load_payload

    run_dir, resume_dir = tmp_path / "run", tmp_path / "resume"
    run_dir.mkdir(), resume_dir.mkdir()
    monkeypatch.chdir(run_dir)
    probes, epochs = _run_cli(rte_component, run_dir, recorder={"save_epoch": 1, "save_dir": str(resume_dir)})

    # phase 0 starts with nothing to enforce; each recovery phase (after `max_transition_epoch`) enforces the
    # boundary's graphs, which moves the answer logits
    (d0, on0, template0, _), (d1, on1, template1, moved1), (d2, on2, template2, moved2) = probes
    assert (d0, on0, template0) == (0, False, (0, 0))
    assert (d1, d2) == (1, 2) and on1 and on2 and template1[1] > 0 and template2 == template1
    assert moved1 > 0 and moved2 > 0

    stages = [(epoch, depth, blocks) for epoch, depth, blocks, _ in epochs]
    assert stages[:3] == [(0, 0, ["10", "11", "9"]), (1, 1, ["11"]), (2, 2, ["10", "11", "9"])]
    params = [p for *_, p in epochs]
    block9, block11 = "transformer.h.9.mlp.c_fc.weight", "transformer.h.11.mlp.c_fc.weight"
    # phase 1 starts from phase 0's best (its one epoch) and leaves the refrozen blocks there; phase 2 trains them again
    assert torch.equal(params[1][block9], params[0][block9]) and not torch.equal(params[1][block11], params[0][block11])
    assert not torch.equal(params[2][block9], params[1][block9])

    best = load_payload(next((run_dir / "checkpoints" / "rte_sparse.gpt2.core").glob("best-*.pt")))
    assert not any(k.startswith("sparse_tcs.") for k in best["model"])  # instruments are never checkpointed
    assert sum(k.startswith("sparse_enforce_") for k in best["model"]) == 15  # fitted state is
    assert best["statefuls"]["sparse_normalizer"]["value"] > 0

    monkeypatch.chdir(resume_dir)
    _, resumed = _run_cli(rte_component, resume_dir, resume_from=resume_dir / "resume.pt")
    assert [(e, d, b) for e, d, b, _ in resumed] == stages[2:]
    for (*_, expected), (*_, got) in zip(epochs[2:], resumed):
        assert all(torch.equal(expected[n], got[n]) for n in expected)


def test_the_gate_records_an_intermediate_stage_without_judging_it(rte_component, tmp_path):
    """A phase in `record_only_depths` is measured and recorded with the verdict it would have had, and accepted
    unless it ends the run: the first step of the re-thaw is judged through the second (#936)."""
    import hashlib
    from types import SimpleNamespace

    from interpretune.hub.entrypoints import import_snapshot_entrypoint

    entrypoint = import_snapshot_entrypoint("speediedan/rte", "rte_boolq.py", revision=rte_component, what="test")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"rows": {"860": {"errors_share": 0.2, "replacement": 0.5}}}))
    gate = entrypoint.ChainGraphGate(
        str(reference), hashlib.sha256(reference.read_bytes()).hexdigest(), record_only_depths=(1,)
    )

    def judge(depth, final, errors_share):
        rows = {860: {"errors_share": errors_share, "replacement": 0.5}}
        module = SimpleNamespace(sparse_graph_measures={"depth": depth, "rows": rows})
        event = SimpleNamespace(
            depth=depth, final=final, best_depth=depth, tuning_scheduler=SimpleNamespace(model=module)
        )
        return gate(event)

    recorded = judge(1, False, 0.3)  # 10 points over the reference: outside the bounds
    assert recorded.accept and recorded.record["judged"] is False and recorded.record["within_bounds"] is False
    assert not judge(1, True, 0.3).accept  # a run ending on that phase is judged
    assert not judge(2, False, 0.3).accept and not judge(0, False, 0.3).accept
    accepted = judge(2, True, 0.21)
    assert accepted.accept and accepted.record["judged"] and accepted.record["within_bounds"]
