"""``AnalysisStoreCallback`` under a multi-process Lightning ``Trainer``: one store from every rank's rows.

Two CPU ranks under ``torch.distributed.run`` (gloo) run ``tests/core/distributed_lightning_store_worker.py``; a
one-process run of the same predict is the reference. Lightning's predict sampler deals the examples out in turn, so a
rank's batch holds other examples than a one-process batch: parity is checked per example, through the ``rank`` and
``batch_idx`` columns each row carries. The refusals that need no second process use stand-in trainers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from interpretune.utils import MisconfigurationException

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("distributed_lightning_store_worker.py")

two_ranks = pytest.mark.skipif(
    sys.platform != "linux" or not dist.is_available(),
    reason="two gloo ranks under torch.distributed.run, on Linux only",
)


def _launch(mode: str, out_dir: Path) -> list[dict]:
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    # the worker imports `tests.*`, which a subprocess finds only through PYTHONPATH
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["IT_TRUST_REMOTE_CODE"] = "1"
    worker = [str(WORKER), "--mode", mode, "--out-dir", str(out_dir)]
    ranks = 1 if mode == "single" else 2
    if ranks == 1:
        cmd = [sys.executable, *worker]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"{mode} run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return [json.loads((out_dir / f"{mode}_rank{rank}.json").read_text(encoding="utf-8")) for rank in range(ranks)]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("lightning_store")
    (single,) = _launch("single", out_dir)
    return single, _launch("ddp", out_dir)


def _per_example(rows: list[dict], world_size: int, batch_size: int, origin=None) -> list[tuple]:
    """Each example's ``(orig_labels, label_ids)``, in the one-process example order.

    A row of rank ``r``'s batch ``b`` holds that rank's local examples ``b * batch_size + j``, and Lightning's predict
    sampler gives rank ``r`` the examples ``r, r + world_size, ...``. ``origin`` overrides where a row came from.
    """
    placed = {}
    for i, row in enumerate(rows):
        rank, batch_idx = origin(i) if origin else (row.get("rank", 0), row.get("batch_idx", i))
        for j, example in enumerate(zip(row["orig_labels"], row["label_ids"])):
            placed[rank + world_size * (batch_idx * batch_size + j)] = example
    return [placed[index] for index in sorted(placed)]


@two_ranks
@pytest.mark.parametrize("rank", [0, 1])
def test_two_ranks_build_one_store_that_holds_every_example_once(runs, rank):
    single, ranks = runs
    ddp = ranks[rank]
    assert ddp["columns"] == sorted(single["columns"] + ["rank", "batch_idx"])
    assert ddp["rows"] == ranks[0]["rows"], "the ranks loaded different stores"
    origins = [(row["batch_idx"], row["rank"]) for row in ddp["rows"]]
    assert origins == sorted(origins) and set(r for _, r in origins) == {0, 1}
    expected = _per_example(single["rows"], 1, single["batch_size"])
    assert len(expected) == 10
    assert _per_example(ddp["rows"], 2, ddp["batch_size"]) == expected
    assert ddp["row_files"] == [], "a rank's shard was left beside the merged store"


@two_ranks
def test_the_per_example_check_tells_the_ranks_apart(runs):
    """Positive control: reading the rows as if they came from the other rank fails the per-example comparison."""
    single, (ddp, _) = runs
    swapped = _per_example(
        ddp["rows"], 2, 2, origin=lambda i: (1 - ddp["rows"][i]["rank"], ddp["rows"][i]["batch_idx"])
    )
    assert swapped != _per_example(single["rows"], 1, single["batch_size"])


@two_ranks
@pytest.mark.parametrize("rank", [0, 1])
def test_two_ranks_report_the_one_process_metric_averages(runs, rank):
    single, ranks = runs
    assert ranks[rank]["analysis_metrics"] == single["analysis_metrics"]
    assert ranks[0]["logged"] == single["logged"], "rank 0's logger did not receive the one-process averages"


@two_ranks
def test_a_rank_s_failed_shard_fails_every_rank_by_name(tmp_path):
    """Every rank raises, naming the failed rank, instead of one waiting on the other, and no store is saved."""
    failed = "RuntimeError: A rank could not close its AnalysisStore shard: rank 1: OSError: disk full"
    for record in _launch("shard_fails", tmp_path):
        assert record["refused"] == failed
        assert not record["store_saved"] and record["row_files"] == []


@two_ranks
def test_a_predict_that_does_not_split_the_examples_is_refused(tmp_path):
    """With Lightning's distributed sampler off, each rank would analyze every example."""
    for record in _launch("unsharded", tmp_path):
        assert "predict sampler makes no shares (SequentialSampler)" in record["refused"]


def _stand_in_trainer(*loaders, world_size=2):
    return SimpleNamespace(world_size=world_size, predict_dataloaders=list(loaders))


def _sharded_loader(num_replicas=2):
    data = list(range(4))
    return DataLoader(data, sampler=DistributedSampler(data, num_replicas=num_replicas, rank=0), batch_size=2)


def _cfg(op_state=None):
    return SimpleNamespace(op=SimpleNamespace(name="op", op_state=op_state))


def test_a_sharded_predict_is_accepted():
    from interpretune.adapters.lightning import AnalysisStoreCallback

    AnalysisStoreCallback._check_distributed_run(_stand_in_trainer(_sharded_loader()), _cfg())


@pytest.mark.parametrize(
    "trainer, match",
    [
        (_stand_in_trainer(_sharded_loader(4)), r"the predict sampler makes 4 shares \(DistributedSampler\)"),
        (
            _stand_in_trainer(DataLoader(list(range(4)), sampler=SequentialSampler(range(4)))),
            r"makes no shares \(SequentialSampler\)",
        ),
        (_stand_in_trainer(_sharded_loader(), _sharded_loader()), "this predict has 2"),
    ],
    ids=["fewer-shares-than-ranks", "not-sharded", "two-dataloaders"],
)
def test_a_predict_the_merged_store_cannot_account_for_is_refused(trainer, match):
    from interpretune.adapters.lightning import AnalysisStoreCallback

    with pytest.raises(MisconfigurationException, match=match):
        AnalysisStoreCallback._check_distributed_run(trainer, _cfg())


def test_an_op_with_cross_batch_state_is_refused_under_several_ranks():
    from interpretune.adapters.lightning import AnalysisStoreCallback

    with pytest.raises(MisconfigurationException, match="op declare cross-batch `op_state`"):
        AnalysisStoreCallback._check_distributed_run(_stand_in_trainer(_sharded_loader()), _cfg(op_state=object()))


def test_an_op_output_that_already_has_a_row_origin_field_is_refused(tmp_path, monkeypatch):
    from datasets import Value

    import interpretune.runners.analysis as analysis
    from interpretune.adapters.lightning import AnalysisStoreCallback
    from tests.core.test_lightning_analysis_step import _analysis_cfg, _lightning_cust_session

    module = _lightning_cust_session().module
    module.analysis_cfg = _analysis_cfg(ignore_manual=True)
    monkeypatch.setattr(AnalysisStoreCallback, "_check_distributed_run", staticmethod(lambda trainer, cfg: None))
    monkeypatch.setattr(module, "_ensure_analysis_setup", lambda: None)
    started = []
    monkeypatch.setattr(module, "on_analysis_start", lambda: started.append(True), raising=False)
    monkeypatch.setattr(analysis, "dataset_features_and_format", lambda m, kw: ({"rank": Value("int32")}, {}, kw))
    trainer = SimpleNamespace(world_size=2, global_rank=0)
    with pytest.raises(MisconfigurationException, match=r"already has \['rank'\]"):
        AnalysisStoreCallback().on_predict_start(trainer, module)
    assert getattr(module, "_it_analysis_row_writer", None) is None
    assert not started, "the analysis start hooks ran for a run that was refused"
