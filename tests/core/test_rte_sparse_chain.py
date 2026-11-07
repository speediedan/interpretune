"""The sparsity chain (#698 to #700) as one phased run of the ``speediedan/rte`` component's ``rte_sparse`` variant
(#899), CPU-sized: GPT-2 with random transcoders, run from the shipped run configuration through the interpretune
CLI.

The boundary's graphs come from a test-side stand-in (the research configuration builds them with circuit-tracer on a
GPU); everything else is the component's: the chain's row partition, the penalty and its calibrated normalizer, the
roles (transcoders ``instrument``, enforcement ``fitted``), the payload and fill hooks, and the enforced recovery
phase.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip(
    "finetuning_scheduler.core.tuning_scheduler", reason="finetuning-scheduler predates TuningScheduler"
)
pytest.importorskip("circuit_tracer", reason="the rte_sparse variant's transcoders are circuit-tracer's")

from tests.core.rte_sparse_stub import EnforcementProbe  # noqa: E402

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


def test_the_sparse_chain_runs_both_phases_from_the_cli(rte_component, clean_cli_env, tmp_path, monkeypatch):
    from finetuning_scheduler.core.checkpoint import load_payload
    from interpretune.base.components.cli import core_cli_main
    from interpretune.hub.components import resolve_component_file

    monkeypatch.chdir(tmp_path)
    shipped = resolve_component_file(f"hub://speediedan/rte@{rte_component}/runs/rte_sparse.gpt2.core.yaml")
    cfg = yaml.safe_load(
        shipped.read_text(encoding="utf-8").replace("hub://speediedan/rte/", f"hub://speediedan/rte@{rte_component}/")
    )
    cfg["session_ref"] = f"{cfg['session_ref']}@{rte_component}"
    ts_kwargs = cfg["run_cfg"]["callbacks"][0]["init_args"]["tuning_scheduler_kwargs"]
    ts_kwargs["phase_hooks"]["end"] = [{"class_path": "tests.core.rte_sparse_stub.StubGraphs"}]
    ts_kwargs["phase_hooks"]["start"].append({"class_path": "tests.core.rte_sparse_stub.EnforcementProbe"})
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(cfg))
    EnforcementProbe.seen = []
    args = ["--config", str(tmp_path / "run.yaml")]
    args += ["--registered_cfg", json.dumps({"datamodule_cfg": {"dataset_path": str(tmp_path / "dataset")}})]
    args += ["--run_cfg.max_epochs=3", "--run_cfg.limit_train_batches=2", "--run_cfg.limit_val_batches=2"]
    core_cli_main(run_mode="train", args=args)

    # phase 0 starts with nothing to enforce; phase 1 (after phase 0's `max_transition_epoch`) starts enforcing the
    # boundary's graphs, which moves the answer logits
    (d0, on0, template0, _), (d1, on1, template1, moved) = EnforcementProbe.seen
    assert (d0, on0, template0) == (0, False, (0, 0))
    assert d1 == 1 and on1 and template1[1] > 0 and moved > 0
    best = load_payload(next((tmp_path / "checkpoints" / "rte_sparse.gpt2.core").glob("best-*.pt")))
    assert not any(k.startswith("sparse_tcs.") for k in best["model"])  # instruments are never checkpointed
    assert sum(k.startswith("sparse_enforce_") for k in best["model"]) == 15  # fitted state is
    assert best["statefuls"]["sparse_normalizer"]["value"] > 0
