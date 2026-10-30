"""Data-parallel placement for the core loop (``SessionRunnerCfg.placement``): the refusals, the loader rebuild,
and two-rank CPU runs of ``ddp`` and ``fsdp2`` against one process training on their combined batch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, RandomSampler, SubsetRandomSampler

from interpretune.config import AnalysisRunnerCfg, SessionRunnerCfg
from interpretune.runners.distributed import (
    DistributedRun,
    _check_devices,
    _refuse_unshardable,
    _require_fts,
    _resolve_plan,
    distributed_dataloader,
    meta_built,
    resolve_placement,
)
from interpretune.utils import _CT_AVAILABLE, _FTS_AVAILABLE, MisconfigurationException

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("distributed_core_loop_worker.py")
META_WORKER = Path(__file__).with_name("distributed_meta_init_worker.py")
CT_WORKER = Path(__file__).with_name("distributed_ct_worker.py")


def _run(rank: int, world_size: int = 2, placement: str = "ddp") -> DistributedRun:
    return DistributedRun(placement, rank, rank, world_size)


class TestResolvePlacement:
    def test_single_without_a_launcher_has_no_run(self):
        assert resolve_placement("single", env={}) is None

    def test_single_under_several_ranks_is_refused(self):
        with pytest.raises(
            MisconfigurationException, match="WORLD_SIZE=2 ranks, but the runner's placement is 'single'"
        ):
            resolve_placement("single", env={"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"})

    @pytest.mark.parametrize("placement", ["ddp", "fsdp2"])
    def test_data_parallel_with_one_rank_is_refused(self, placement):
        with pytest.raises(MisconfigurationException, match=f"'{placement}' needs at least two ranks.*unset"):
            resolve_placement(placement, env={})

    def test_missing_rank_variables_are_named(self):
        with pytest.raises(MisconfigurationException, match="no RANK or LOCAL_RANK"):
            resolve_placement("ddp", env={"WORLD_SIZE": "2"})

    def test_launcher_environment_is_read(self):
        run = resolve_placement("fsdp2", env={"WORLD_SIZE": "4", "RANK": "3", "LOCAL_RANK": "1"})
        assert run == DistributedRun("fsdp2", 3, 1, 4) and not run.is_rank_zero

    def test_unknown_placement_is_refused(self):
        with pytest.raises(MisconfigurationException, match="Unknown placement 'tp'"):
            resolve_placement("tp", env={})
        with pytest.raises(MisconfigurationException, match="Unknown placement 'tp'"):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement="tp")  # type: ignore[arg-type]

    @pytest.mark.parametrize("placement", ["ddp", "fsdp2"])
    def test_resuming_a_data_parallel_run_is_refused(self, placement):
        """A core-loop checkpoint is one process's state; a sharded run needs distributed checkpointing."""
        with pytest.raises(MisconfigurationException, match=f"supports placement 'single' only, not '{placement}'"):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement=placement, resume_from={})
        SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement="single", resume_from={})

    def test_analysis_runner_refuses_data_parallel(self):
        with pytest.raises(MisconfigurationException, match="supports placement 'single' only, not 'ddp'"):
            AnalysisRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement="ddp")


class TestDistributedDataloader:
    def test_sequential_loader_is_strided_by_rank(self):
        loader = DataLoader(list(range(8)), batch_size=2, collate_fn=lambda xs: xs)
        batches = [list(distributed_dataloader(loader, _run(rank))) for rank in range(2)]
        assert batches == [[[0, 2], [4, 6]], [[1, 3], [5, 7]]]

    def test_rebuild_keeps_loader_settings(self):
        collate = lambda xs: torch.tensor(xs)  # noqa: E731
        loader = DataLoader(list(range(10)), batch_size=3, collate_fn=collate, drop_last=True)
        rebuilt = distributed_dataloader(loader, _run(1))
        assert (rebuilt.batch_size, rebuilt.collate_fn, rebuilt.drop_last) == (3, collate, True)
        assert rebuilt.sampler.num_replicas == 2 and rebuilt.sampler.rank == 1 and not rebuilt.sampler.shuffle

    def test_shuffled_loader_shuffles_every_rank_alike(self):
        loader = DataLoader(list(range(8)), batch_size=4, shuffle=True)
        samplers = [distributed_dataloader(loader, _run(rank)).sampler for rank in range(2)]
        assert all(s.shuffle for s in samplers)
        assert sorted(list(samplers[0]) + list(samplers[1])) == list(range(8))

    def test_non_dataloader_is_refused(self):
        with pytest.raises(MisconfigurationException, match="needs a `DataLoader`, not a list"):
            distributed_dataloader([[1, 2]], _run(0))

    def test_iterable_dataset_is_refused(self):
        class Stream(IterableDataset):
            def __iter__(self):
                return iter(range(4))

        with pytest.raises(MisconfigurationException, match="cannot shard an `IterableDataset` \\(Stream\\)"):
            distributed_dataloader(DataLoader(Stream()), _run(0))

    @pytest.mark.parametrize(
        "sampler",
        [
            SubsetRandomSampler([0, 1, 2]),
            RandomSampler(range(4), replacement=True),
            RandomSampler(range(4), num_samples=2),
        ],
        ids=["subset", "replacement", "num_samples"],
    )
    def test_samplers_a_distributed_sampler_cannot_reproduce_are_refused(self, sampler):
        with pytest.raises(MisconfigurationException, match=f"sampler is a {type(sampler).__name__}"):
            distributed_dataloader(DataLoader(list(range(4)), sampler=sampler), _run(0))


class TestPlacementChecks:
    def test_one_device_per_rank_is_recorded(self):
        assert _check_devices([["cuda:0"], ["cuda:1"]], "ddp") == ("cuda:0", "cuda:1")
        assert _check_devices([["cpu"], ["cpu"]], "fsdp2") == ("cpu", "cpu")

    def test_layer_split_rank_is_refused(self):
        with pytest.raises(MisconfigurationException, match="rank 1 holds cuda:0, cuda:1"):
            _check_devices([["cuda:0"], ["cuda:0", "cuda:1"]], "ddp")

    def test_rank_without_parameters_is_refused(self):
        with pytest.raises(MisconfigurationException, match="rank 0 holds no parameters"):
            _check_devices([[], ["cpu"]], "ddp")

    def test_shared_accelerator_is_refused(self):
        with pytest.raises(MisconfigurationException, match="Ranks 0 and 1 both placed the module on cuda:0"):
            _check_devices([["cuda:0"], ["cuda:0"]], "fsdp2")

    def test_mixed_device_types_are_refused(self):
        with pytest.raises(MisconfigurationException, match="different device types"):
            _check_devices([["cpu"], ["cuda:1"]], "ddp")

    def test_non_contiguous_parameter_is_refused_by_name(self):
        module = nn.Module()
        module.packed = nn.Parameter(torch.zeros(4, 6))
        module.q = nn.Parameter(module.packed.data[:, :3].T)  # a transposed view, as a split of a packed weight
        with pytest.raises(MisconfigurationException, match="non-contiguous parameters, and this module has 1: q"):
            _refuse_unshardable(module)

    @pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
    def test_plan_units_are_ordered_children_first(self):
        """A plan listing a block before the instrument spliced into it still shards the instrument first."""

        class Inner(nn.Linear):
            pass

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner(2, 2)

        model = nn.Sequential(Block(), Block())
        plan = {"class:Block": {}, "class:Inner": {"reshard_after_forward": False}}
        units = _resolve_plan(model, plan, None, "fsdp2")
        assert list(units) == ["0.inner", "1.inner", "0", "1"]
        assert units["0.inner"] == {"reshard_after_forward": False}

    @pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
    def test_plan_keys_select_by_name_regex_and_class_with_defaults(self):
        class Block(nn.Linear):
            pass

        model = nn.Sequential(Block(2, 2), nn.Linear(2, 2), Block(2, 2))
        defaults = {"reshard_after_forward": True}
        # `class:` matches subclasses too (here every Block is an nn.Linear), unlike the retired class-name list
        assert list(_resolve_plan(model, {"class:Linear": None}, defaults, "fsdp2")) == ["0", "1", "2"]
        units = _resolve_plan(model, {"1": {"reshard_after_forward": False}, "[02]": None}, defaults, "fsdp2")
        assert units == {"1": {"reshard_after_forward": False}, "0": defaults, "2": defaults}

    @pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
    def test_fsdp2_needs_a_plan_and_ddp_does_not(self):
        model = nn.Sequential(nn.Linear(2, 2))
        assert _resolve_plan(model, None, None, "ddp") == {}
        assert _resolve_plan(model, {}, None, "fsdp2") == {}
        with pytest.raises(MisconfigurationException, match="Placement 'fsdp2' needs `fsdp_plan`"):
            _resolve_plan(model, None, None, "fsdp2")
        with pytest.raises(MisconfigurationException, match="'class:Block' specified in `fsdp_plan` did not match"):
            _resolve_plan(model, {"class:Linear": {}, "class:Block": {}}, None, "fsdp2")

    def test_plan_without_finetuning_scheduler_is_refused(self, monkeypatch):
        import interpretune.runners.distributed as distributed

        monkeypatch.setattr(distributed, "_FTS_AVAILABLE", False)
        _require_fts({}, None)  # an empty plan shards the module as one unit, with no plan grammar to apply
        with pytest.raises(MisconfigurationException, match="FSDP2 plan, which is not installed"):
            _require_fts({"class:Block": {}}, None)
        with pytest.raises(MisconfigurationException, match="which is not installed"):
            _require_fts(None, {"reshard_after_forward": False})

    def test_meta_built_needs_the_whole_module_on_meta(self):
        assert not meta_built(nn.Linear(2, 2))
        with torch.device("meta"):
            assert meta_built(nn.Linear(2, 2))
            partial = nn.Sequential(nn.Linear(2, 2))
        partial.append(nn.Linear(2, 2))
        with pytest.raises(MisconfigurationException, match="built partly on the meta device: 2 tensors are not"):
            meta_built(partial)

    def test_meta_init_refuses_quantized_models_and_adapters_it_cannot_load(self):
        from interpretune.config.mixins import HFFromPretrainedConfig

        HFFromPretrainedConfig(meta_init=True)
        HFFromPretrainedConfig(meta_init=True, lora_cfg={"r": 8})  # rank 0's copy carries the adapters
        with pytest.raises(MisconfigurationException, match="cannot be combined with\\s+`bitsandbytesconfig`"):
            HFFromPretrainedConfig(meta_init=True, bitsandbytesconfig={"load_in_4bit": True})
        with pytest.raises(MisconfigurationException, match="hold no LoRA adapters"):
            HFFromPretrainedConfig(meta_init=True, meta_init_load="dcp", lora_cfg={"r": 8})
        with pytest.raises(MisconfigurationException, match="must be 'rank0' or 'dcp', not 'stream'"):
            HFFromPretrainedConfig(meta_init=True, meta_init_load="stream")
        with pytest.raises(MisconfigurationException, match="but `meta_init` is not set"):
            HFFromPretrainedConfig(meta_init_load="dcp")

    def test_meta_init_is_refused_when_the_init_did_not_build_on_meta(self):
        from types import SimpleNamespace

        from interpretune.base.modules import BaseITModule
        from interpretune.config.mixins import HFFromPretrainedConfig

        def module(meta_init: bool, built: bool | None):
            stub = SimpleNamespace(
                it_cfg=SimpleNamespace(hf_from_pretrained_cfg=HFFromPretrainedConfig(meta_init=meta_init))
            )
            if built is not None:
                stub._meta_init_built = built
            return stub

        BaseITModule._refuse_unbuilt_meta_init(module(meta_init=False, built=None))
        BaseITModule._refuse_unbuilt_meta_init(module(meta_init=True, built=True))  # the HF from_pretrained meta build
        # a TransformerLens bridge, nnsight or circuit-tracer init (or a deferred one) never runs the meta build
        with pytest.raises(MisconfigurationException, match="built its model without the meta-device build"):
            BaseITModule._refuse_unbuilt_meta_init(module(meta_init=True, built=None))
        no_hf_cfg = SimpleNamespace(it_cfg=SimpleNamespace(hf_from_pretrained_cfg=None))
        BaseITModule._refuse_unbuilt_meta_init(no_hf_cfg)

    def test_plan_under_single_placement_is_refused(self):
        with pytest.raises(MisconfigurationException, match="but the placement is 'single'"):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), fsdp_plan={"class:Block": {}})
        with pytest.raises(MisconfigurationException, match="but the placement is 'single'"):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), fsdp_default_kwargs={"x": 1})


def _launch(placement: str, out_dir: Path, *extra: str) -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    # the worker imports `tests.*` and the test tree's bare modules, which a subprocess finds only through PYTHONPATH
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), str(REPO_ROOT / "tests"), env.get("PYTHONPATH")) if p
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["IT_TRUST_REMOTE_CODE"] = "1"
    worker = [str(WORKER), "--placement", placement, "--out-dir", str(out_dir), *extra]
    if placement == "single":
        cmd = [sys.executable, *worker]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"{placement} run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"


@pytest.fixture(scope="module")
def global_batch_reference(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("dist_reference")
    _launch("single", out_dir)
    return json.loads((out_dir / "single_rank0.json").read_text())


@pytest.fixture(scope="module")
def phased_global_batch_reference(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("dist_phased_reference")
    _launch("single", out_dir, "--phased")
    return json.loads((out_dir / "single_rank0.json").read_text())


# one unit per TransformerBlock, then the module itself as the root unit
BLOCK_PLAN = ("--fsdp-plan", '{"class:TransformerBlock": {}}')


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.parametrize(
    "placement, extra",
    [("ddp", ()), ("fsdp2", BLOCK_PLAN)],
    ids=["ddp", "fsdp2"],
)
def test_two_rank_core_loop_matches_global_batch(placement, extra, global_batch_reference, tmp_path):
    """Two ranks match one process training on their combined batch, step for step.

    The mean of the ranks' losses at each step equals the reference's loss on the global batch only if the loaders
    give each rank its ``DistributedSampler`` share, rank 0's weights reach every rank (the ranks are seeded
    differently, the reference as rank 0) and gradients are averaged across ranks. Every rank's callbacks see the
    reference's epoch metrics: the cross-rank average.
    """
    _launch(placement, tmp_path, *extra)
    records = [json.loads((tmp_path / f"{placement}_rank{rank}.json").read_text()) for rank in range(2)]
    rank_mean = [sum(step) / 2 for step in zip(*(record["losses"] for record in records))]
    assert len(rank_mean) == len(global_batch_reference["losses"]) == 3
    assert rank_mean == pytest.approx(global_batch_reference["losses"], rel=1e-5)
    for record in records:
        assert record["epoch_metrics"][0] == pytest.approx(global_batch_reference["epoch_metrics"][0], rel=1e-5)
        assert record["devices"] == ["cpu", "cpu"]
        assert record["param_types"] == ["DTensor"]
        # the plan's units were sharded, not only the root (whose sharding alone would also make every DTensor)
        n_units = 3 if placement == "fsdp2" else 1
        assert len(record["fsdp_units"]) == n_units and "" in record["fsdp_units"]
    assert global_batch_reference["param_types"] == ["Parameter"]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="phased tuning is finetuning-scheduler's")
def test_two_rank_phased_tuning_under_fsdp2_matches_global_batch(phased_global_batch_reference, tmp_path):
    """Two phases of finetuning-scheduler's ``TuningScheduler`` over a plan-sharded module match one process.

    Phase 0 thaws the last block, phase 1 the rest. The ranks match the reference step for step only if each thaw
    reaches the sharded parameters and the optimizer the tuning scheduler rebuilds at the transition steps them.
    """
    reference = phased_global_batch_reference
    _launch("fsdp2", tmp_path, *BLOCK_PLAN, "--phased")
    records = [json.loads((tmp_path / f"fsdp2_rank{rank}.json").read_text()) for rank in range(2)]
    rank_mean = [sum(step) / 2 for step in zip(*(record["losses"] for record in records))]
    assert len(rank_mean) == len(reference["losses"]) == 6
    assert rank_mean == pytest.approx(reference["losses"], rel=1e-5)
    for record in records:
        assert record["depth"] == reference["depth"] == 1
        assert record["trainable"] == reference["trainable"]
        assert len(record["fsdp_units"]) == 3


def _launch_meta(checkpoint: Path, out_dir: Path, *extra: str) -> list[dict]:
    out_dir.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    worker = [str(META_WORKER), "--checkpoint", str(checkpoint), "--out-dir", str(out_dir), *extra]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"meta-init run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return [json.loads((out_dir / f"rank{rank}.json").read_text()) for rank in range(2)]


@pytest.fixture(scope="module")
def tiny_gpt2_checkpoint(tmp_path_factory):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_gpt2")
    GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=128, n_positions=32)).save_pretrained(path)
    return path


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
def test_a_meta_built_model_is_sharded_then_loaded_from_rank_zero(tiny_gpt2_checkpoint, tmp_path):
    """A model built on the meta device on both ranks and loaded from rank 0's CPU copy computes exactly what
    ``from_pretrained`` does, keeps its LM head tied to the embedding and drops the copy; a copy missing a module
    is refused on every rank, naming the missing keys.

    With LoRA, the copy's adapters load too, and one training step of the adapters over the frozen sharded base matches
    the same step in one process.
    """
    for record in _launch_meta(tiny_gpt2_checkpoint, tmp_path / "loaded"):
        assert record["max_abs_diff"] == 0.0
        assert record["tied"] and record["source_dropped"]
        assert record["param_types"] == ["DTensor"]
    planted = tmp_path / "planted"
    planted.mkdir()
    for record in _launch_meta(tiny_gpt2_checkpoint, planted, "--drop", "ln_f"):
        assert "does not match the meta-built model" in record["refused"]
        assert "transformer.ln_f.bias" in record["refused"]
    # LoRA: rank 0's copy carries the adapters (non-zero B, so they move the output), and every rank gets its values
    for record in _launch_meta(tiny_gpt2_checkpoint, tmp_path / "lora", "--lora", "--train-step"):
        assert record["max_abs_diff"] == 0.0 and record["adapter_effect"] > 0.1
        # one adapter step on both ranks matches one process's, and moves the output
        assert record["step_diff"] < 1e-6 and record["step_effect"] > 0.01
        assert record["trainable"] == ["default"] and record["param_types"] == ["DTensor"]
        assert record["tied"] and record["source_dropped"]


@pytest.fixture(scope="module")
def tiny_llama_checkpoint(tmp_path_factory):
    """A Llama checkpoint split over several safetensors files (with an index), whose rotary frequencies are a non-
    persistent buffer no file holds."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_llama")
    config = LlamaConfig(
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
        bos_token_id=0,
        eos_token_id=1,
    )
    LlamaForCausalLM(config).save_pretrained(path, max_shard_size="20KB")
    return path


def _edited_checkpoint(checkpoint: Path, out: Path, drop: str | None = None, add: str | None = None) -> Path:
    """A copy of a single-file checkpoint with key ``drop`` removed and a key ``add`` added (planted defects)."""
    import shutil

    from safetensors.torch import load_file, save_file

    out.mkdir()
    shutil.copy(checkpoint / "config.json", out / "config.json")
    tensors = load_file(checkpoint / "model.safetensors")
    if drop:
        del tensors[drop]
    if add:
        tensors[add] = torch.zeros(2)
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    return out


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
def test_a_meta_built_model_is_loaded_from_each_ranks_checkpoint_slices(
    tiny_gpt2_checkpoint, tiny_llama_checkpoint, tmp_path
):
    """With ``meta_init_load: dcp``, each rank reads its own slices of the checkpoint files: a single-file GPT-2
    (tied head) and a multi-file Llama (rotary buffers kept from the build) compute exactly what
    ``from_pretrained`` does.

    A key the model tells ``from_pretrained`` to ignore (an old GPT-2 checkpoint's
    attention-mask buffer) is ignored, while a checkpoint missing a parameter or holding a key no model key matches
    is refused on every rank, naming it.
    """
    assert len(list(tiny_llama_checkpoint.glob("*.safetensors"))) > 1
    ignorable = _edited_checkpoint(tiny_gpt2_checkpoint, tmp_path / "ignorable_ckpt", add="transformer.h.0.attn.bias")
    for checkpoint, arch in ((tiny_gpt2_checkpoint, "gpt2"), (tiny_llama_checkpoint, "llama"), (ignorable, "gpt2")):
        for record in _launch_meta(checkpoint, tmp_path / checkpoint.name, "--loader", "dcp", "--arch", arch):
            assert record["max_abs_diff"] == 0.0, checkpoint.name
            assert record["source_dropped"] and record["param_types"] == ["DTensor"]
            assert record["tied"] == (arch == "gpt2")
    missing = _edited_checkpoint(tiny_gpt2_checkpoint, tmp_path / "missing_ckpt", drop="transformer.ln_f.bias")
    for record in _launch_meta(missing, tmp_path / "missing", "--loader", "dcp"):
        assert "do not match the meta-built model" in record["refused"]
        refusal = record["refused"]
        assert "Keys no model key matches: []; parameters no key supplies: ['transformer.ln_f.bias']" in refusal
    unknown = _edited_checkpoint(tiny_gpt2_checkpoint, tmp_path / "unknown_ckpt", add="transformer.h.0.attn.extra")
    for record in _launch_meta(unknown, tmp_path / "unknown", "--loader", "dcp"):
        assert (
            "Keys no model key matches: ['transformer.h.0.attn.extra']; parameters no key supplies: []"
            in (record["refused"])
        )


def _launch_ct(plan: dict, out_dir: Path) -> list[dict]:
    out_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    env["CUDA_VISIBLE_DEVICES"] = ""
    worker = [str(CT_WORKER), "--plan", json.dumps(plan), "--out-dir", str(out_dir)]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"circuit-tracer run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return [json.loads((out_dir / f"rank{rank}.json").read_text()) for rank in range(2)]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
@pytest.mark.skipif(not _CT_AVAILABLE, reason="circuit-tracer is not installed")
def test_a_circuit_tracer_model_is_sharded_with_its_direct_calls_registered(tmp_path):
    """A circuit-tracer replacement model sharded on two ranks gives the logits, the nnsight trace and the
    transcoder activations of one process, once the calls circuit-tracer makes outside the module's forward are
    registered: the Hugging Face module nnsight calls directly, and each transcoder's ``encode`` and activation
    function.

    A plan that leaves the Hugging Face module to the root unit, whose hooks those direct calls bypass, is refused by
    name.
    """
    units = {"class:Gemma2DecoderLayer": {}, "class:SingleLayerTranscoder": {}, "class:JumpReLU": {}}
    for record in _launch_ct({**units, "class:Gemma2ForCausalLM": {}}, tmp_path / "placed"):
        assert record["active"] > 0  # so equal activations are not two empty caches
        assert record["plain_diff"] == record["trace_diff"] == record["activations_diff"] == 0.0
    for record in _launch_ct(units, tmp_path / "no_entry_unit"):
        assert "calls hf_model directly" in record["refused"]
        assert "Add `class:Gemma2ForCausalLM` to `fsdp_plan`" in record["refused"]


@pytest.mark.skipif(not _CT_AVAILABLE, reason="circuit-tracer is not installed")
def test_transcoder_direct_calls_refuse_an_unenumerated_transcoder_type():
    from interpretune.adapters.circuit_tracer.adapter import transcoder_direct_calls

    assert transcoder_direct_calls(nn.Module()) == []  # nothing to shard, nothing to declare
    other = nn.Module()
    other.weight = nn.Parameter(torch.zeros(2))
    with pytest.raises(MisconfigurationException, match="only `SingleLayerTranscoder` sets are"):
        transcoder_direct_calls(other)
