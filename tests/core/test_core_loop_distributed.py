"""Distributed placement for the core loop (``SessionRunnerCfg.placement``): the refusals, the loader rebuild, two-
rank CPU runs of ``ddp`` and ``fsdp2`` against one process training on their combined batch, and of ``tp`` against
one process training on the batch every rank shares."""

from __future__ import annotations

import json
import math
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
    _meta_offloads,
    _packed_groups,
    _refuse_unshardable,
    _require_fts,
    _resolve_plan,
    _resolve_tp_plan,
    distributed_dataloader,
    hf_tp_plan,
    meta_built,
    resolve_placement,
    step_per_tensor,
)
from interpretune.utils import _CT_AVAILABLE, _FTS_AVAILABLE, MisconfigurationException
from tests.runif import RunIf

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("distributed_core_loop_worker.py")
META_WORKER = Path(__file__).with_name("distributed_meta_init_worker.py")
CT_WORKER = Path(__file__).with_name("distributed_ct_worker.py")
LATENT_WORKER = Path(__file__).with_name("distributed_latent_worker.py")
TP_BRIDGE_WORKER = Path(__file__).with_name("distributed_tp_bridge_worker.py")
TP_HF_WORKER = Path(__file__).with_name("distributed_tp_hf_worker.py")
TP_NNSIGHT_WORKER = Path(__file__).with_name("distributed_tp_nnsight_worker.py")
TP_CT_WORKER = Path(__file__).with_name("distributed_tp_ct_worker.py")
TP_ANALYSIS_WORKER = Path(__file__).with_name("distributed_tp_analysis_worker.py")
DDP_ANALYSIS_WORKER = Path(__file__).with_name("distributed_ddp_analysis_worker.py")


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
        with pytest.raises(MisconfigurationException, match="Unknown placement 'pp'"):
            resolve_placement("pp", env={})
        with pytest.raises(MisconfigurationException, match="Unknown placement 'pp'"):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement="pp")  # type: ignore[arg-type]

    def test_tensor_parallel_is_not_data_parallel(self):
        run = resolve_placement("tp", env={"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"})
        assert run == DistributedRun("tp", 1, 1, 2) and not run.data_parallel
        assert _run(0, placement="fsdp2").data_parallel and _run(0, placement="ddp").data_parallel

    def test_plans_must_match_the_placement(self):
        """A tensor-parallel plan means nothing to a data-parallel placement, and an FSDP2 plan nothing to 'tp'."""
        plan = {"model.layers.0.mlp": {"style": "colwise"}}
        with pytest.raises(MisconfigurationException, match="but the placement is 'fsdp2'"):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement="fsdp2", tp_plan=plan)
        with pytest.raises(MisconfigurationException, match="but the placement is 'single'"):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), tp_plan=plan)
        with pytest.raises(MisconfigurationException, match="but the placement is 'tp'"):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement="tp", fsdp_plan={})
        SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement="tp", tp_plan=plan)

    @pytest.mark.parametrize("placement", ["ddp", "fsdp2"])
    def test_resuming_a_data_parallel_run_needs_the_distributed_form(self, placement, tmp_path):
        """A mapping or a `torch.save` file is one process's state; a sharded run resumes from a directory."""
        with pytest.raises(
            MisconfigurationException, match=f"under placement '{placement}' must be a distributed checkpoint"
        ):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement=placement, resume_from={})
        checkpoint_file = tmp_path / "ckpt.pt"
        checkpoint_file.touch()
        with pytest.raises(
            MisconfigurationException, match=f"under placement '{placement}' must be a distributed checkpoint"
        ):
            SessionRunnerCfg(
                module=MagicMock(), datamodule=MagicMock(), placement=placement, resume_from=checkpoint_file
            )
        checkpoint_dir = tmp_path / "ckpt"
        checkpoint_dir.mkdir()
        SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement=placement, resume_from=checkpoint_dir)
        SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement="single", resume_from={})
        with pytest.raises(MisconfigurationException, match="which is a distributed checkpoint, but the"):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement="single", resume_from=checkpoint_dir)

    def test_analysis_runner_refuses_fsdp2(self):
        with pytest.raises(
            MisconfigurationException, match="supports placements 'single', 'tp' and 'ddp', not 'fsdp2'"
        ):
            AnalysisRunnerCfg(module=MagicMock(), datamodule=MagicMock(), placement="fsdp2")

    @pytest.mark.parametrize("placement", ["tp", "ddp"])
    def test_analysis_runner_takes_tp_and_ddp(self, placement):
        session = MagicMock()
        assert AnalysisRunnerCfg(it_session=session, placement=placement).placement == placement


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

    def test_tensor_parallel_ranks_take_the_same_batches(self):
        """Under 'tp' the ranks compute one forward together, so each takes the whole dataset in one shared order,
        even when the loader shuffles and the ranks' own random streams differ."""
        loader = DataLoader(list(range(8)), batch_size=2, shuffle=True)
        per_rank = []
        for rank in range(2):
            torch.manual_seed(rank)
            per_rank.append([batch.tolist() for batch in distributed_dataloader(loader, _run(rank, placement="tp"))])
        assert per_rank[0] == per_rank[1] and sorted(sum(per_rank[0], [])) == list(range(8))

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

        with pytest.raises(MisconfigurationException, match="cannot rebuild an `IterableDataset` \\(Stream\\)"):
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
            _refuse_unshardable(module, "fsdp2")

    @staticmethod
    def _packed(conv1d_layout: bool, sections: tuple[int, ...] = (4, 4, 4)) -> nn.Module:
        """A packed weight and bias with one ``Linear`` view per section, as TransformerLens splits a joint qkv:

        the weight in a ``Conv1D``'s ``[in, out]`` layout (each view a transposed section) or a ``Linear``'s
        ``[out, in]``.
        """
        module = nn.Module()
        module.qkv = nn.Linear(3, sum(sections))
        if conv1d_layout:
            module.qkv.weight = nn.Parameter(torch.randn(3, sum(sections)))
        start = 0
        for name, length in zip("qkv", sections):
            view = nn.Linear(3, length)
            packed = module.qkv.weight
            weight = packed[:, start : start + length].T if conv1d_layout else packed[start : start + length]
            view.weight = nn.Parameter(weight)
            view.bias = nn.Parameter(module.qkv.bias[start : start + length])
            setattr(module, name, view)
            start += length
        return module

    @pytest.mark.parametrize("conv1d_layout", [True, False], ids=["conv1d", "linear"])
    def test_tp_splits_a_packed_weight_with_its_views(self, conv1d_layout):
        """A plan splitting every view of a packed weight by output features splits the packed weight with them, in
        either layout, while a non-contiguous parameter the plan leaves whole is no concern of 'tp'."""
        from torch.distributed.tensor.parallel import ColwiseParallel

        module = self._packed(conv1d_layout)
        module.other = nn.Linear(2, 2)
        module.other.weight = nn.Parameter(torch.zeros(2, 4)[:, :2])  # non-contiguous, but not split
        groups = _packed_groups(module, {n: ColwiseParallel() for n in "qkv"}, world_size=2)
        assert len(groups) == 2  # the weight and the bias
        weight, bias = sorted(groups, key=lambda g: g.packed[1], reverse=True)
        assert weight.packed == (module.qkv, "weight") and bias.packed == (module.qkv, "bias")
        assert weight.dim == (1 if conv1d_layout else 0) and bias.dim == 0
        assert weight.views == [(getattr(module, n), "weight", conv1d_layout) for n in "qkv"]
        assert bias.views == [(getattr(module, n), "bias", False) for n in "qkv"]

    def test_tp_refuses_splits_that_cut_shared_storage(self):
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        def refused(module, styles, match, world_size=2):
            with pytest.raises(MisconfigurationException, match=match):
                _packed_groups(module, styles, world_size)

        colwise = {n: ColwiseParallel() for n in "qkv"}
        refused(self._packed(True), {"q": ColwiseParallel()}, r"only some of the views of qkv.weight.*k.weight")
        refused(self._packed(True), {"qkv": ColwiseParallel()}, r"splits qkv.weight, whose storage")
        refused(self._packed(True), {**colwise, "v": RowwiseParallel()}, r"only with 'colwise'.*v.weight")
        refused(self._packed(True, (4, 2, 2)), colwise, "are not such sections of qkv.weight")
        refused(self._packed(True, (3, 3, 3)), colwise, "have 3 output features each, which 2 ranks cannot split")
        lone = nn.Module()
        lone.proj = nn.Linear(2, 2)
        lone.proj.weight = nn.Parameter(torch.zeros(2, 4)[:, :2])
        refused(lone, {"proj": ColwiseParallel()}, "cannot split the non-contiguous parameter proj.weight")

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

    @pytest.mark.skipif(not _FTS_AVAILABLE, reason="the tensor-parallel plan grammar is finetuning-scheduler's")
    def test_tp_needs_a_plan_whose_keys_select_modules(self):
        model = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 2))
        assert list(_resolve_tp_plan(model, {"0": {"style": "colwise"}, "1": {"style": "rowwise"}})) == ["0", "1"]
        with pytest.raises(MisconfigurationException, match="Placement 'tp' needs `tp_plan`"):
            _resolve_tp_plan(model, None)
        with pytest.raises(MisconfigurationException, match="'class:Block' specified in `tp_plan` did not match"):
            _resolve_tp_plan(model, {"class:Block": {"style": "colwise"}})
        with pytest.raises(MisconfigurationException, match="whose `style` is one of"):
            _resolve_tp_plan(model, {"0": {"style": "diagonal"}})

    @staticmethod
    def _tiny_llama(heads: int = 2, kv_heads: int = 2, hidden: int = 32):
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            num_hidden_layers=2,
            hidden_size=hidden,
            intermediate_size=64,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            vocab_size=128,
        )
        return LlamaForCausalLM(config)

    def test_default_tp_plan_is_the_hf_models_declared_decoder_plan(self):
        """Without a ``tp_plan``, an interpretune module offers the plan transformers declares for its model, keyed
        under the model's path: the decoder layers' entries, with the embedding and LM head left whole."""
        from interpretune.base.modules import BaseITModule

        stub = nn.Module()
        stub.model = self._tiny_llama()
        stub.meta_load_target = lambda: stub.model
        stub.default_tp_plan = lambda world_size: BaseITModule.default_tp_plan(stub, world_size)  # pyright: ignore
        plan = BaseITModule.default_tp_plan(stub, 2)  # pyright: ignore[reportArgumentType]
        assert plan == {
            rf"model\.model\.layers\.\d+\.{part}": {"style": style}
            for part, style in (
                (r"self_attn\.q_proj", "colwise"),
                (r"self_attn\.k_proj", "colwise"),
                (r"self_attn\.v_proj", "colwise"),
                (r"self_attn\.o_proj", "rowwise"),
                (r"mlp\.gate_proj", "colwise"),
                (r"mlp\.up_proj", "colwise"),
                (r"mlp\.down_proj", "rowwise"),
            )
        }
        if _FTS_AVAILABLE:  # every key selects the layers it names, and nothing else
            styles = _resolve_tp_plan(stub, None, 2)
            assert len(styles) == 14 and all(".layers." in name for name in styles)
        # a model declaring no plan (GPT-2) offers none, so the placement asks for one
        from transformers import GPT2Config, GPT2LMHeadModel

        stub.model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=8, n_head=2, vocab_size=16))
        assert BaseITModule.default_tp_plan(stub, 2) is None  # pyright: ignore[reportArgumentType]
        with pytest.raises(MisconfigurationException, match="Module declares no default plan"):
            _resolve_tp_plan(stub, None, 2)

    def test_default_tp_plan_refuses_what_it_cannot_split_faithfully(self):
        with pytest.raises(MisconfigurationException, match=r"num_key_value_heads \(1\) is not divisible by the 2"):
            hf_tp_plan(self._tiny_llama(kv_heads=1), "model", 2)
        with pytest.raises(MisconfigurationException, match=r"num_attention_heads \(3\) is not divisible by the 2"):
            hf_tp_plan(self._tiny_llama(heads=3, kv_heads=3, hidden=48), "model", 2)
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration

        # a multimodal model's heads are its text config's
        text = {"num_hidden_layers": 1, "hidden_size": 24, "intermediate_size": 8, "head_dim": 4, "vocab_size": 16}
        vision = {"num_hidden_layers": 1, "hidden_size": 8, "intermediate_size": 8, "num_attention_heads": 2}
        for heads, match in ((2, None), (3, r"num_attention_heads \(3\)")):
            config = Gemma3Config(
                text_config={**text, "num_attention_heads": heads, "num_key_value_heads": 2}, vision_config=vision
            )
            with torch.device("meta"):
                multimodal = Gemma3ForConditionalGeneration(config)
            if match is None:
                plan = hf_tp_plan(multimodal, "model", 2)  # the norms stay whole, so the plan does not name them
                assert len(plan) == 7 and not any("norm" in key for key in plan)
                continue
            with pytest.raises(MisconfigurationException, match=match):
                hf_tp_plan(multimodal, "model", 2)
        model = self._tiny_llama()
        model._tp_plan = {**model.tp_plan, "model.layers.*.mlp.experts": "local_colwise"}
        with pytest.raises(MisconfigurationException, match="experts the style 'local_colwise'"):
            hf_tp_plan(model, "model", 2)

    def test_tp_plan_without_finetuning_scheduler_is_refused(self, monkeypatch):
        import interpretune.runners.distributed as distributed

        monkeypatch.setattr(distributed, "_FTS_AVAILABLE", False)
        with pytest.raises(MisconfigurationException, match="tensor-parallel plan, which is not installed"):
            _resolve_tp_plan(nn.Linear(2, 2), {"": {"style": "colwise"}})

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

    def test_what_is_attached_beside_a_meta_built_model_holds_its_values(self):
        """The model's checkpoint does not hold it, so it must be built with values, which the placement keeps."""

        def holder(model_device: str, latent_device: str) -> nn.Module:
            root = nn.Module()
            root.model = nn.Linear(2, 2, device=model_device)
            root.latent_models = nn.ModuleDict({"probe": nn.Linear(2, 2, device=latent_device)})
            return root

        assert meta_built(holder("meta", "cpu"))
        assert not meta_built(holder("cpu", "cpu"))
        for model_device in ("meta", "cpu"):
            with pytest.raises(MisconfigurationException, match="2 parameters attached beside the model"):
                meta_built(holder(model_device, "meta"))

    def test_a_meta_build_materializes_on_the_cpu_only_when_every_unit_offloads(self):
        """FSDP2 keeps an offloaded unit's shards on the CPU, so a meta-built module is materialized there when
        every unit offloads, on the rank's device when none does, and a plan offloading only some units is
        refused."""
        from torch.distributed.fsdp import CPUOffloadPolicy

        assert _meta_offloads({"model.layers.0": {}}, {"cpu_offload_policy": {"pin_memory": True}})
        assert not _meta_offloads({"model.layers.0": {}, "model.layers.1": None}, None)
        assert not _meta_offloads({"model.layers.0": {}}, {"reshard_after_forward": True})
        with pytest.raises(MisconfigurationException, match=r"only some units \(model.layers.1\)"):
            _meta_offloads({"model.layers.0": {}, "model.layers.1": {"offload_policy": CPUOffloadPolicy()}}, None)

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
        # a TransformerLens bridge or plain nnsight init (or a deferred one) never runs the meta build
        with pytest.raises(MisconfigurationException, match="built its model without the meta-device build"):
            BaseITModule._refuse_unbuilt_meta_init(module(meta_init=True, built=None))
        no_hf_cfg = SimpleNamespace(it_cfg=SimpleNamespace(hf_from_pretrained_cfg=None))
        BaseITModule._refuse_unbuilt_meta_init(no_hf_cfg)

    def test_tp_optimizers_step_one_tensor_at_a_time(self):
        # CUDA optimizers default to multi-tensor kernels, which refuse a group mixing split and whole parameters
        run = DistributedRun("tp", rank=0, local_rank=0, world_size=2)
        optimizer = torch.optim.AdamW(nn.Linear(2, 2).parameters())
        step_per_tensor([optimizer], run)
        assert optimizer.defaults["foreach"] is False and optimizer.param_groups[0]["foreach"] is False
        optimizer.add_param_group({"params": nn.Linear(2, 2).parameters()})  # a tuning scheduler's next phase
        assert optimizer.param_groups[1]["foreach"] is False
        untouched = torch.optim.AdamW(nn.Linear(2, 2).parameters())
        step_per_tensor([untouched], DistributedRun("fsdp2", rank=0, local_rank=0, world_size=2))
        assert untouched.defaults["foreach"] is None

    @pytest.mark.parametrize("kernel", ["foreach", "fused"])
    def test_tp_refuses_an_explicit_multi_tensor_kernel(self, kernel):
        optimizer = torch.optim.AdamW(nn.Linear(2, 2).parameters(), **{kernel: True})
        with pytest.raises(MisconfigurationException, match=f"`{kernel}=True` \\(defaults\\).*placement 'tp'"):
            step_per_tensor([optimizer], DistributedRun("tp", rank=0, local_rank=0, world_size=2))

    def test_plan_under_single_placement_is_refused(self):
        with pytest.raises(MisconfigurationException, match="but the placement is 'single'"):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), fsdp_plan={"class:Block": {}})
        with pytest.raises(MisconfigurationException, match="but the placement is 'single'"):
            SessionRunnerCfg(module=MagicMock(), datamodule=MagicMock(), fsdp_default_kwargs={"x": 1})


def _launch(placement: str, out_dir: Path, *extra: str, devices: str = "") -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    # the worker imports `tests.*` and the test tree's bare modules, which a subprocess finds only through PYTHONPATH
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), str(REPO_ROOT / "tests"), env.get("PYTHONPATH")) if p
    )
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["IT_TRUST_REMOTE_CODE"] = "1"
    worker = [str(WORKER), "--placement", placement, "--out-dir", str(out_dir), *extra]
    worker += ["--device-type", "cuda"] if devices else []
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
    return json.loads((out_dir / "single_rank0.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def phased_global_batch_reference(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("dist_phased_reference")
    _launch("single", out_dir, "--phased")
    return json.loads((out_dir / "single_rank0.json").read_text(encoding="utf-8"))


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
    records = [json.loads((tmp_path / f"{placement}_rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
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
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
def test_an_fsdp_plan_from_configuration_sets_mixed_precision_and_activation_checkpointing(
    global_batch_reference, tmp_path
):
    """A plan as a run configuration file holds it, where dtypes are names: each unit takes the mixed precision
    policy (bf16 parameters, gradients reduced in fp32), and the blocks the plan lists are checkpointed.

    The losses show the policy computes in bf16: they differ from the fp32 reference's, and stay near them.
    """
    extra = ("--fsdp-plan", '{"class:TransformerBlock": {"act_ckpt": "composable"}}')
    defaults = '{"mp_policy": {"param_dtype": "bfloat16", "reduce_dtype": "float32"}}'
    _launch("fsdp2", tmp_path, *extra, "--fsdp-default-kwargs", defaults)
    records = [json.loads((tmp_path / f"fsdp2_rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    for record in records:
        settings = record["unit_settings"]
        assert len(settings) == len(record["fsdp_units"]) == 3
        assert {s["param_dtype"] for s in settings.values()} == {"torch.bfloat16"}
        assert {s["reduce_dtype"] for s in settings.values()} == {"torch.float32"}
        assert {name for name, s in settings.items() if s["act_ckpt"]} == set(settings) - {""}
    rank_mean = [sum(step) / 2 for step in zip(*(record["losses"] for record in records))]
    assert rank_mean == pytest.approx(global_batch_reference["losses"], rel=2e-2)
    assert rank_mean != pytest.approx(global_batch_reference["losses"], rel=1e-5)


# a trainable intervention adds a learned vector to the first block's output; a block is its own FSDP2 unit and the
# intervention's parameters are the root unit's, so its operand is whole where the block's hooks apply it
INTERVENTION = ("--intervention-site", "model.layers.0")


@pytest.fixture(scope="module")
def intervention_reference(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("dist_intervention_reference")
    _launch("single", out_dir, *INTERVENTION)
    return json.loads((out_dir / "single_rank0.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.parametrize(
    "placement, extra",
    [("ddp", ()), ("fsdp2", BLOCK_PLAN)],
    ids=["ddp", "fsdp2"],
)
def test_two_rank_training_applies_an_intervention_with_its_whole_operand(
    placement, extra, global_batch_reference, intervention_reference, tmp_path
):
    """Two ranks training a trainable intervention match one process on their combined batch, step for step, and
    end with the same operand on both ranks.

    The operand is a vector of the model's width, so a rank that applied only its shard of it would fail on the shape or
    train another value; and the operand must move, or the comparison would pass with it never applied.
    """
    _launch(placement, tmp_path, *extra, *INTERVENTION)
    records = [json.loads((tmp_path / f"{placement}_rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    rank_mean = [sum(step) / 2 for step in zip(*(record["losses"] for record in records))]
    assert rank_mean == pytest.approx(intervention_reference["losses"], rel=1e-5)
    assert rank_mean != pytest.approx(global_batch_reference["losses"], rel=1e-5)  # the intervention has an effect
    assert len(records[0]["intervention_params"]) == 1
    for record in records:
        assert record["intervention_params"] == records[0]["intervention_params"]
        (operand,) = record["intervention_params"].values()
        (expected,) = intervention_reference["intervention_params"].values()
        assert any(expected) and operand == pytest.approx(expected, rel=1e-4, abs=1e-6)


# each block's feed-forward split across the ranks: the first projection by output features, the second by input
# features, so the pair needs one all-reduce and the block's output is whole again on every rank
FFN_TP_PLAN = (
    "--tp-plan",
    json.dumps(
        {
            r"model\.layers\.\d+\.feed_forward\.w1": {"style": "colwise"},
            r"model\.layers\.\d+\.feed_forward\.w2": {"style": "rowwise"},
        }
    ),
)


@pytest.fixture(scope="module")
def shared_batch_reference(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("tp_reference")
    _launch("single", out_dir, "--world-size", "1")
    return json.loads((out_dir / "single_rank0.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the tensor-parallel plan is finetuning-scheduler's")
def test_two_rank_tensor_parallel_matches_one_process_on_the_shared_batch(shared_batch_reference, tmp_path):
    """Two tensor-parallel ranks compute the one-process run on their shared batch, step for step, on every rank.

    The ranks match the reference only if every rank sees the reference's batches in its order, the split layers are
    cut from rank 0's weights (the ranks are seeded differently), and the split feed-forward's partial outputs are
    reduced back to the whole activation. Only the planned layers become ``DTensor``\\ s.
    """
    _launch("tp", tmp_path, *FFN_TP_PLAN)
    records = [json.loads((tmp_path / f"tp_rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    assert len(shared_batch_reference["losses"]) == 3
    for record in records:
        assert record["losses"] == pytest.approx(shared_batch_reference["losses"], rel=1e-5)
        assert record["epoch_metrics"][0] == pytest.approx(shared_batch_reference["epoch_metrics"][0], rel=1e-5)
        assert record["devices"] == ["cpu", "cpu"]
        assert record["param_types"] == ["DTensor", "Parameter"] and record["fsdp_units"] == []
        assert record["split_params"] == [
            f"model.layers.{i}.feed_forward.{w}.{p}"
            for i in range(2)
            for w, p in (("w1", "bias"), ("w1", "weight"), ("w2", "bias"), ("w2", "weight"))
        ]


# a latent model inside each split feed-forward, between its column- and row-split projections
LATENT_SITE = "model.layers.0.feed_forward.gelu"


@pytest.fixture(scope="module")
def latent_shared_batch_reference(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("tp_latent_reference")
    _launch("single", out_dir, "--world-size", "1", "--latent-site", LATENT_SITE)
    return json.loads((out_dir / "single_rank0.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the tensor-parallel plan is finetuning-scheduler's")
def test_two_rank_tp_gathers_a_split_activation_for_a_spliced_latent_model(latent_shared_batch_reference, tmp_path):
    """A latent model spliced inside a split feed-forward reads the whole activation and rewrites each rank's
    shard: the ranks match one process step for step, and train the latent model to the same weights on every rank.

    The first projection keeps its split output as a ``DTensor`` (``use_local_output: false``), which the splice
    gathers; a latent model encoding one rank's shard would fail on its width or compute something else. Splicing at
    a split layer whose output is handed on as a plain local shard is refused, naming the site and the fix.
    """
    plan = json.loads(FFN_TP_PLAN[1])
    plan[r"model\.layers\.\d+\.feed_forward\.w1"]["use_local_output"] = False
    _launch("tp", tmp_path, "--tp-plan", json.dumps(plan), "--latent-site", LATENT_SITE)
    reference = latent_shared_batch_reference
    for rank in range(2):
        record = json.loads((tmp_path / f"tp_rank{rank}.json").read_text(encoding="utf-8"))
        assert record["losses"] == pytest.approx(reference["losses"], rel=1e-5)
        assert record["latent_params"].keys() == reference["latent_params"].keys()
        for name, values in record["latent_params"].items():
            assert values == pytest.approx(reference["latent_params"][name], rel=1e-4, abs=1e-6), name
    _launch("tp", tmp_path, *FFN_TP_PLAN, "--latent-site", "model.layers.0.feed_forward.w1", "--out-name", "local")
    for rank in range(2):
        refused = json.loads((tmp_path / f"local_rank{rank}.json").read_text(encoding="utf-8"))["refused"]
        assert "spliced at model.layers.0.feed_forward.w1" in refused and "use_local_output: false" in refused


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
    records = [json.loads((tmp_path / f"fsdp2_rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    rank_mean = [sum(step) / 2 for step in zip(*(record["losses"] for record in records))]
    assert len(rank_mean) == len(reference["losses"]) == 6
    assert rank_mean == pytest.approx(reference["losses"], rel=1e-5)
    for record in records:
        assert record["depth"] == reference["depth"] == 1
        assert record["trainable"] == reference["trainable"]
        assert len(record["fsdp_units"]) == 3


@pytest.fixture(scope="module", params=["roles", "promote"])
def addressed_reference(request, tmp_path_factory):
    """One process's record, with how the latent model is attached: tunable (``roles``), or frozen and promoted in
    the second phase (``promote``)."""
    out_dir = tmp_path_factory.mktemp(f"dist_addressed_reference_{request.param}")
    _launch("single", out_dir, "--addressed", request.param)
    return request.param, json.loads((out_dir / "single_rank0.json").read_text(encoding="utf-8"))


# the last block's MLP, then the latent model too, then the first block's attention too: what the worker's addresses
# (`layers:-1: & component:mlp.*`, `role:latent:probe`, `layers:0 & component:attn.*`) must select
_PROBE = {f"latent_models.probe.{n}" for n in ("enc.weight", "enc.bias", "dec.weight", "dec.bias")}
_MLP = {f"model.layers.1.feed_forward.{n}.{leaf}" for n in ("w1", "w2") for leaf in ("weight", "bias")}
ADDRESSED_EPOCHS = [_MLP, _MLP | _PROBE, _MLP | _PROBE | {f"model.layers.0.attention.w{n}.weight" for n in "qkvo"}]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="phased tuning and the schedule's addresses are finetuning-scheduler's")
@pytest.mark.parametrize("placement, extra", [("ddp", ()), ("fsdp2", BLOCK_PLAN)], ids=["ddp", "fsdp2"])
def test_two_rank_tuning_by_venngram_address_matches_one_process(placement, extra, addressed_reference, tmp_path):
    """A schedule naming what to tune by layer and component and by role (never by parameter name) thaws the same
    tensors of a sharded composition as of one process, phase by phase, and trains them to the same values, a
    frozen latent model the second phase promotes included."""
    attached, reference = addressed_reference
    _launch(placement, tmp_path, *extra, "--addressed", attached)
    records = [json.loads((tmp_path / f"{placement}_rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    rank_mean = [sum(step) / 2 for step in zip(*(record["losses"] for record in records))]
    assert rank_mean == pytest.approx(reference["losses"], rel=1e-5)
    for record in (reference, *records):
        assert [set(names) for names in record["trainable_by_epoch"]] == ADDRESSED_EPOCHS
        assert record["depth"] == 2
        # the latent model trained: an untrained one would match the reference too, up to its small steps
        assert record["probe_init"] == reference["probe_init"] and record["probe_params"] != record["probe_init"]
        for name, values in reference["probe_params"].items():
            assert record["probe_params"][name] == pytest.approx(values, rel=0, abs=1e-6), name
    assert "" in records[0]["fsdp_units"] and (placement == "ddp" or len(records[0]["fsdp_units"]) == 3)


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the memory profiler and phased tuning are finetuning-scheduler's")
def test_two_rank_fsdp2_memory_profile_covers_each_iteration_per_phase(phased_global_batch_reference, tmp_path):
    """The memory profiler's ``FSDP2MemTracker`` profiles a phased fsdp2 run on every rank, one report entry per
    training iteration, each recording the phase it ran in, without changing the run.

    Each entry spans the whole iteration: the root's module-wise states include the backward pass, which the core
    loop runs outside ``training_step``, so a profile of ``training_step`` alone would hold forward states only. The
    phase comes from ``FTSCoreCallback``, and the tracker is rebuilt over the optimizer the transition installs.
    """
    _launch("fsdp2", tmp_path, *BLOCK_PLAN, "--phased", "--memprof-fsdp")
    records = [json.loads((tmp_path / f"fsdp2_rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    rank_mean = [sum(step) / 2 for step in zip(*(record["losses"] for record in records))]
    assert rank_mean == pytest.approx(phased_global_batch_reference["losses"], rel=1e-5)
    for rank, record in enumerate(records):
        report = json.loads(Path(record["fsdp_mem_report"]).read_text(encoding="utf-8"))
        assert (report["rank"], report["world_size"]) == (rank, 2)
        entries = report["entries"]
        assert [(e["fn_name"], e["iter"], e["phase"]) for e in entries] == [
            ("training_step", i, 0 if i <= 3 else 1) for i in range(1, 7)
        ]
        for entry in entries:
            root_states = set(entry["modules"][next(iter(entry["modules"]))])
            assert {"Peak Forward", "Peak Backward"} <= root_states, entry["iter"]
            assert entry["peak"]["cpu"]["Total"] > 0
        # optimizer state exists from the first step on, so it is in every later entry's peak
        assert all(e["peak"]["cpu"]["OptState"] > 0 for e in entries[1:])


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.parametrize(
    "placement, extra",
    [
        ("ddp", ()),
        ("fsdp2", BLOCK_PLAN),
        pytest.param(
            "tp",
            FFN_TP_PLAN,
            marks=pytest.mark.skipif(not _FTS_AVAILABLE, reason="the tensor-parallel plan is finetuning-scheduler's"),
        ),
    ],
    ids=["ddp", "fsdp2", "tp"],
)
def test_two_rank_resume_matches_the_uninterrupted_run_exactly(placement, extra, tmp_path):
    """A run interrupted at an epoch boundary and resumed from its distributed checkpoint reproduces the
    uninterrupted run's losses exactly: same sharded (or split) weights and optimizer state, same per-rank RNG
    streams, same sampler epoch."""
    checkpoint_dir = tmp_path / "ckpt"
    # shuffled, as the training split is by default: the resumed run must also restore where the sampler's order was
    _launch(placement, tmp_path, *extra, "--shuffle", "--max-epochs", "2", "--out-name", "full")
    _launch(
        placement,
        tmp_path,
        *extra,
        "--shuffle",
        "--max-epochs",
        "2",
        "--out-name",
        "first",
        "--save-checkpoint-dir",
        str(checkpoint_dir),
    )
    _launch(
        placement,
        tmp_path,
        *extra,
        "--shuffle",
        "--max-epochs",
        "2",
        "--out-name",
        "resumed",
        "--resume-from",
        str(checkpoint_dir),
    )
    for rank in range(2):
        full = json.loads((tmp_path / f"full_rank{rank}.json").read_text(encoding="utf-8"))
        first = json.loads((tmp_path / f"first_rank{rank}.json").read_text(encoding="utf-8"))
        resumed = json.loads((tmp_path / f"resumed_rank{rank}.json").read_text(encoding="utf-8"))
        assert first["losses"][:3] + resumed["losses"] == full["losses"]
        assert resumed["epoch_metrics"] == full["epoch_metrics"][1:]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="phased tuning is finetuning-scheduler's")
def test_two_rank_phased_resume_across_the_transition_matches(phased_global_batch_reference, tmp_path):
    """Resuming a phased fsdp2 run from a checkpoint taken at the first epoch's end, so the resumed run is the one
    that crosses the finetuning-scheduler phase transition, still matches the uninterrupted reference:

    the callback's restored state rebuilds the optimizer the checkpoint's optimizer state loads into.
    """
    checkpoint_dir = tmp_path / "ckpt"
    _launch("fsdp2", tmp_path, *BLOCK_PLAN, "--phased", "--out-name", "first",
            "--save-checkpoint-dir", str(checkpoint_dir))  # fmt: skip
    _launch("fsdp2", tmp_path, *BLOCK_PLAN, "--phased", "--out-name", "resumed",
            "--resume-from", str(checkpoint_dir))  # fmt: skip
    reference = phased_global_batch_reference
    records = [json.loads((tmp_path / f"resumed_rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    rank_mean = [sum(step) / 2 for step in zip(*(record["losses"] for record in records))]
    assert rank_mean == pytest.approx(reference["losses"][3:], rel=1e-5)
    for record in records:
        assert record["depth"] == reference["depth"] == 1
        assert record["trainable"] == reference["trainable"]


@pytest.fixture(scope="module")
def phased_shared_batch_reference(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("tp_phased_reference")
    _launch("single", out_dir, "--world-size", "1", "--phased")
    return json.loads((out_dir / "single_rank0.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="phased tuning and the tensor-parallel plan are finetuning-scheduler's")
def test_two_rank_phased_tuning_under_tp_matches_one_process(phased_shared_batch_reference, tmp_path):
    """Two phases of finetuning-scheduler's ``TuningScheduler`` over a tensor-parallel module match one process on
    the shared batch, step for step, on every rank.

    Phase 0 thaws the last block, phase 1 the rest, so phase 1 thaws split (``DTensor``) and whole parameters
    together; the ranks match only if each thaw reaches both kinds and the optimizer rebuilt at the transition steps
    them.
    """
    reference = phased_shared_batch_reference
    _launch("tp", tmp_path, *FFN_TP_PLAN, "--phased")
    for rank in range(2):
        record = json.loads((tmp_path / f"tp_rank{rank}.json").read_text(encoding="utf-8"))
        assert len(record["losses"]) == len(reference["losses"]) == 6
        assert record["losses"] == pytest.approx(reference["losses"], rel=1e-5)
        assert record["depth"] == reference["depth"] == 1
        assert record["trainable"] == reference["trainable"]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="phased tuning and the tensor-parallel plan are finetuning-scheduler's")
def test_two_rank_phased_tp_resume_across_the_transition_matches(phased_shared_batch_reference, tmp_path):
    """A phased tensor-parallel run resumed from the first epoch's checkpoint crosses the phase transition in the
    resumed run and still matches the one-process reference on every rank: split weights and their optimizer state
    load back into the split layout."""
    checkpoint_dir = tmp_path / "ckpt"
    _launch("tp", tmp_path, *FFN_TP_PLAN, "--phased", "--out-name", "first",
            "--save-checkpoint-dir", str(checkpoint_dir))  # fmt: skip
    _launch("tp", tmp_path, *FFN_TP_PLAN, "--phased", "--out-name", "resumed",
            "--resume-from", str(checkpoint_dir))  # fmt: skip
    reference = phased_shared_batch_reference
    for rank in range(2):
        record = json.loads((tmp_path / f"resumed_rank{rank}.json").read_text(encoding="utf-8"))
        assert record["losses"] == pytest.approx(reference["losses"][3:], rel=1e-5)
        assert record["depth"] == reference["depth"] == 1
        assert record["trainable"] == reference["trainable"]


# a transcoder over the first block's split feed-forward: it reads the block's whole input and replaces the reduced
# output, while the second block's split feed-forward keeps training
TRANSCODER = ("--transcoder-site", "model.layers.0.feed_forward")


def _phased_transcoder_records(out_dir: Path, devices: tuple[str, str] = ("", "")) -> tuple[dict, list[dict]]:
    """The one-process reference (on ``devices[0]``) and the two tensor-parallel ranks (on ``devices[1]``) of a
    phased run that tunes the transcoder first, then everything else."""
    _launch("single", out_dir, "--world-size", "1", "--phased", *TRANSCODER, devices=devices[0])
    _launch("tp", out_dir, *FFN_TP_PLAN, "--phased", *TRANSCODER, devices=devices[1])
    reference = json.loads((out_dir / "single_rank0.json").read_text(encoding="utf-8"))
    return reference, [json.loads((out_dir / f"tp_rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]


def _check_phased_transcoder(reference: dict, records: list[dict]) -> None:
    transcoder = sorted(reference["latent_params"])
    assert reference["depth"] == 1 and len(reference["losses"]) == 6
    # positive control: the transcoder is on the loss's path, so tuning moved it (an inert splice would match too)
    moved = [n for n in transcoder if reference["latent_params"][n] != pytest.approx(reference["latent_init"][n])]
    assert moved == transcoder
    # phase 1 thawed the rest: the transcoder and the model are both trainable at the end
    assert {f"latent_models.transcoder.{n}" for n in transcoder} < set(reference["trainable"])
    for record in records:
        assert record["losses"] == pytest.approx(reference["losses"], rel=1e-5)
        assert record["depth"] == reference["depth"] and record["trainable"] == reference["trainable"]
        assert any(".layers.1.feed_forward." in n for n in record["split_params"])
        assert sorted(record["latent_params"]) == transcoder
        for name, values in record["latent_params"].items():
            assert values == pytest.approx(reference["latent_params"][name], rel=1e-4, abs=1e-6), name


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="phased tuning and the tensor-parallel plan are finetuning-scheduler's")
def test_two_rank_tp_tunes_a_spliced_transcoder_in_phases_like_one_process(tmp_path):
    """Two tensor-parallel ranks tune a transcoder spliced over a split feed-forward, then the whole model, and
    match one process step for step, with the same transcoder weights on every rank."""
    _check_phased_transcoder(*_phased_transcoder_records(tmp_path))


@RunIf(min_cuda_gpus=2, min_gpu_mem_gb=1.5)
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="phased tuning and the tensor-parallel plan are finetuning-scheduler's")
def test_two_gpu_tp_transcoder_tuning_matches_one_gpu(tmp_path):
    """On two GPUs, two-phase latent tuning with a spliced transcoder matches the one-GPU run.

    The ranks run over NCCL, each on its own device, and the reference on the first device alone. CUDA optimizers
    default to multi-tensor kernels, which the second phase's group of split and whole parameters would fail (CPU
    optimizers do not, so only this run can catch it). Measured on an RTX 4090 paired with an RTX 2070 SUPER: losses
    within 7e-8 relative, transcoder weights within 1e-8.
    """
    reference, records = _phased_transcoder_records(tmp_path, devices=("0", "0,1"))
    assert [record["devices"] for record in records] == [["cuda:0", "cuda:1"]] * 2
    _check_phased_transcoder(reference, records)


def _launch_meta(checkpoint: Path, out_dir: Path, *extra: str) -> list[dict]:
    out_dir.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    worker = [str(META_WORKER), "--checkpoint", str(checkpoint), "--out-dir", str(out_dir), *extra]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"meta-init run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return [json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]


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


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
@pytest.mark.parametrize("extra", [(), ("--loader", "dcp"), ("--latent-unit",)], ids=["rank0", "dcp", "own-unit"])
def test_a_latent_model_beside_a_meta_built_model_keeps_rank_zeros_values(tiny_gpt2_checkpoint, tmp_path, extra):
    """Materializing a meta-built module discards every parameter's value, and the checkpoint gives back only the
    model's: an attached latent model's are kept, rank 0's on every rank, so the spliced logits match one
    process's."""
    for record in _launch_meta(tiny_gpt2_checkpoint, tmp_path / "out", "--latent", *extra):
        assert record["max_abs_diff"] < 1e-5 and record["adapter_effect"] > 0.1
        assert record["param_types"] == ["DTensor"] and record["tied"] and record["source_dropped"]


@pytest.fixture(scope="module")
def tied_tiny_llama_checkpoint(tmp_path_factory):
    """A Llama checkpoint whose LM head is tied to its embedding, both of which the default plan leaves whole."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tied_tiny_llama")
    config = LlamaConfig(
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=True,
    )
    LlamaForCausalLM(config).save_pretrained(path)
    return path


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the tensor-parallel plan is finetuning-scheduler's")
@pytest.mark.parametrize("loader", ["rank0", "dcp"])
@pytest.mark.parametrize("latent", [False, True], ids=["model", "latent"])
def test_a_meta_built_model_is_split_then_loaded(tied_tiny_llama_checkpoint, tmp_path, loader, latent):
    """A model built on the meta device and split by its default tensor-parallel plan holds about half of the split
    parameters on each rank and computes, and trains, what one process does: the LM head stays tied to the
    embedding the plan leaves whole, and a latent model attached beside the model keeps rank 0's values."""
    extra = ("--arch", "llama", "--placement", "tp", "--loader", loader, "--train-step")
    for record in _launch_meta(
        tied_tiny_llama_checkpoint, tmp_path / "out", *extra, *(("--latent",) if latent else ())
    ):
        assert record["max_abs_diff"] < 1e-5
        assert record["step_diff"] < 1e-5 and record["step_effect"] > 0.01
        assert record["tied"] and record["source_dropped"]
        split = record["split_params"]
        assert len(split) == 14 and all(".layers." in name for name in split)  # q, k, v, o and gate, up, down, twice
        assert record["held_numel"] < 0.75 * record["whole_numel"]
        if latent:
            assert record["adapter_effect"] > 0.1


# the bridge's split q, k and v in every block, by output features, each output gathered so that the attention reads
# every head (TransformerLens's attention reshapes q, k and v by the model's head count)
JOINT_QKV_TP_PLAN = {
    r"blocks\.\d+\._original_component\.attn\.[qkv]\._original_component": {
        "style": "colwise",
        "output_layouts": "replicate",
    }
}


def _launch_tp_bridge(checkpoint: Path, out_dir: Path, plan: dict, *extra: str) -> list[dict]:
    out_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    worker = [str(TP_BRIDGE_WORKER), "--checkpoint", str(checkpoint), "--out-dir", str(out_dir)]
    worker += ["--tp-plan", json.dumps(plan), *extra]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"tp bridge run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return [json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the tensor-parallel plan is finetuning-scheduler's")
def test_two_rank_tp_splits_a_bridge_joint_qkv_and_keeps_its_views_aliased(tiny_gpt2_checkpoint, tmp_path):
    """Splitting a TransformerLens bridge's q, k and v splits its packed ``qkv`` with them: the split bridge
    computes and steps as the whole one does, and views and packed weight still share storage after an SGD step, so
    the packed weight's ``full_tensor()`` is the stepped reference's.

    The control leaves the packed weight whole. Its outputs still match (no forward reads the packed weight), but the
    views stop sharing its storage and it falls behind the step: the silent divergence the split exists to prevent.
    """
    for record in _launch_tp_bridge(tiny_gpt2_checkpoint, tmp_path / "split", JOINT_QKV_TP_PLAN):
        assert record["split_params"] == [
            f"transformer.h.{i}._original_component.attn.{m}._original_component.{p}"
            for i in range(2)
            for m in ("k", "q", "qkv", "v")
            for p in ("bias", "weight")
        ]
        assert record["forward_diff"] == 0.0 and record["packed_diff_before_step"] == 0.0
        assert record["step_effect"] > 0.1 and record["step_diff"] < 1e-5
        assert record["aliased_before_step"] and record["aliased_after_step"]
        assert record["packed_diff_after_step"] < 1e-6
    control = _launch_tp_bridge(tiny_gpt2_checkpoint, tmp_path / "control", JOINT_QKV_TP_PLAN, "--no-packed-split")
    for record in control:
        assert record["step_diff"] < 1e-5 and not record["aliased_after_step"]
        assert record["packed_diff_after_step"] > 1e-3
    # a plan splitting only q would cut it from the packed weight k and v still share: refused on every rank
    only_q = {r"blocks\.\d+\._original_component\.attn\.q\._original_component": {"style": "colwise"}}
    for record in _launch_tp_bridge(tiny_gpt2_checkpoint, tmp_path / "only_q", only_q):
        assert "splits only some of the views of" in record["refused"]


def _launch_tp_hf(checkpoint, out_dir: Path, *extra: str) -> list[dict]:
    out_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    worker = [str(TP_HF_WORKER), "--checkpoint", str(checkpoint), "--out-dir", str(out_dir), *extra]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"tp hf run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return [json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]


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


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the tensor-parallel plan is finetuning-scheduler's")
def test_two_rank_tp_default_plan_splits_hf_attention_heads_and_mlp(tiny_llama_checkpoint, tmp_path):
    """With no ``tp_plan``, a Hugging Face Llama is split by the plan transformers declares for it: each rank
    computes its own attention heads and its own slice of the MLP, and the split model computes and steps as the
    whole one does.

    The embedding and LM head stay whole.
    """
    for record in _launch_tp_hf(tiny_llama_checkpoint, tmp_path / "hf"):
        assert record["split_params"] == sorted(
            f"model.model.layers.{i}.{proj}.weight"
            for i in range(2)
            for proj in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        )
        assert record["forward_diff"] < 1e-5
        assert record["step_effect"] > 0.1 and record["step_diff"] < 1e-4


@pytest.fixture(scope="module")
def tiny_gemma3_checkpoint(tmp_path_factory):
    """A Gemma 3 checkpoint: its attention normalizes the queries and keys of each head."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_gemma3")
    config = Gemma3TextConfig(
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        query_pre_attn_scalar=8,
        vocab_size=128,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
    )
    model = Gemma3ForCausalLM(config)
    with torch.no_grad():  # the norms' own initial weights are zero
        for name, param in model.named_parameters():
            if "norm" in name:
                param.uniform_(-0.5, 0.5)
    model.save_pretrained(path)
    return path


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the tensor-parallel plan is finetuning-scheduler's")
def test_two_rank_tp_sums_the_gradients_of_query_and_key_norms(tiny_gemma3_checkpoint, tmp_path):
    """A query or key norm stays whole and normalizes only the heads its rank computes, so its gradient is summed
    across ranks: over two accumulated backward passes, for norms frozen when the model was placed, with the model's
    own plan and with an explicit one.

    The control, without the sum, gives gradients that are wrong and no error. A plan that leaves attention whole gives
    every rank the whole gradient, which must not be summed. A TransformerLens bridge wraps each component of the model
    in one of its own, so its norms are found without the wrappers. (It runs in the workers only: building a bridge
    replaces transformers' eager Gemma attention for the whole process.)
    """
    attention = [("q_proj", "colwise"), ("k_proj", "colwise"), ("v_proj", "colwise"), ("o_proj", "rowwise")]
    plans: dict[str, dict | None] = {
        "default": None,
        "explicit": {rf"model\.model\.layers\.\d+\.self_attn\.{m}": {"style": s} for m, s in attention},
        "mlp_only": {
            rf"model\.model\.layers\.\d+\.mlp\.{m}": {"style": s}
            for m, s in (("gate_proj", "colwise"), ("up_proj", "colwise"), ("down_proj", "rowwise"))
        },
    }
    bridged = r"model\.blocks\.\d+\._original_component\.self_attn\._original_component"
    plans["bridge"] = {rf"{bridged}\.{m}\._original_component": {"style": s} for m, s in attention}
    n_split = {"default": 14, "explicit": 8, "mlp_only": 6, "bridge": 8}
    for name, plan in plans.items():
        extra = [] if plan is None else ["--tp-plan", json.dumps(plan)]
        extra += ["--bridge"] if name == "bridge" else []
        for record in _launch_tp_hf(tiny_gemma3_checkpoint, tmp_path / name, *extra):
            assert len(record["split_params"]) == n_split[name]
            assert record["frozen_stayed_frozen"]
            assert record["forward_diff"] < 1e-5
            assert len(record["norm_grad_diff"]) == 4 and max(record["norm_grad_diff"].values()) < 1e-4
            assert record["step_effect"] > 0.1 and record["step_diff"] < 1e-4
    bridge = ("--bridge", "--tp-plan", json.dumps(plans["bridge"]))
    for name, extra in (("control", ()), ("control_bridge", bridge)):
        for record in _launch_tp_hf(tiny_gemma3_checkpoint, tmp_path / name, "--no-gradient-sum", *extra):
            assert min(record["norm_grad_diff"].values()) > 1e-2


@pytest.fixture(scope="module")
def tiny_llama_nnsight_checkpoint(tmp_path_factory):
    """A tiny Llama checkpoint with a word-level tokenizer saved beside it, built offline, since nnsight loads the
    tokenizer from the model's path."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_llama_nnsight")
    config = LlamaConfig(
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
    )
    LlamaForCausalLM(config).save_pretrained(path)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t2"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="t0", eos_token="t1", pad_token="t1", unk_token="t2"
    ).save_pretrained(path)
    return path


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the tensor-parallel plan is finetuning-scheduler's")
def test_two_rank_tp_nnsight_composition_default_plan_traces_like_one_process(tiny_llama_nnsight_checkpoint, tmp_path):
    """An ``NNsightModule`` placed under ``tp`` with no ``tp_plan`` is split by its own default plan (the Hugging
    Face model's declared plan, under ``hf_model``), and nnsight's trace on the split model reads, edits and back-
    propagates as on the whole one.

    One edit does not: PyTorch drops a slice assignment along a ``DTensor``'s sharded dimension without an error, so
    slicing an activation inside a split layer (here a row-parallel ``down_proj``'s input) computes the unedited model.
    The last assertion pins that. If it fails, PyTorch has fixed it: drop the caveat on ``SessionRunnerCfg.tp_plan``
    and assert that the edit matches the reference instead.
    """
    out_dir = tmp_path / "nnsight"
    out_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    worker = [str(TP_NNSIGHT_WORKER), "--checkpoint", str(tiny_llama_nnsight_checkpoint), "--out-dir", str(out_dir)]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"tp nnsight run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    for rank in range(2):
        record = json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        assert record["split_params"] == sorted(
            f"hf_model.model.layers.{i}.{proj}.weight"
            for i in range(2)
            for proj in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        )
        assert record["split_activation_type"] == "DTensor"
        diff, effect = record["diff"], record["effect"]
        for reading in ("attn0", "mlp0", "down1_in", "logits", "boundary_slice_edit", "split_scale_edit"):
            assert diff[reading] < 1e-5, (reading, diff[reading])
        assert diff["grad"] < 1e-4
        # each edit moves the reference, so a dropped edit could not pass as a match
        assert min(effect.values()) > 1e-3, effect
        assert record["split_slice_edit_vs_unedited"] < 1e-5


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


def launch_latent_worker(out_dir: Path, *extra: str, placement: str = "fsdp2", devices: str = "") -> list[dict]:
    """Run ``distributed_latent_worker.py`` on two ranks, or as one process with ``placement="single"``
    (``devices`` is ``CUDA_VISIBLE_DEVICES``), and read their records."""
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), str(REPO_ROOT / "tests"), env.get("PYTHONPATH")) if p
    )
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["IT_TRUST_REMOTE_CODE"] = "1"
    worker = [str(LATENT_WORKER), "--out-dir", str(out_dir), "--placement", placement, *extra]
    if placement == "fsdp2":
        # one unit per transformer block; the spliced SAEs stay in the root unit, which is gathered whenever the
        # blocks run, because the bridge calls an SAE's encode and decode directly rather than through its forward
        worker += ["--fsdp-plan", json.dumps({"class:BlockBridge": {}})]
    worker += ["--device-type", "cuda"] if devices else []
    if placement == "single":
        cmd = [sys.executable, *worker]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"latent run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    ranks = 1 if placement == "single" else 2
    return [json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(ranks)]


def check_latent_phases(records: list[dict]) -> None:
    """Phase 0 trains the last block alone; phase 1 adds the spliced SAEs, selected by ``role:latent:sae``."""
    for record in records:
        epochs = record["trainable_by_epoch"]
        assert record["depth"] == 1 and len(epochs) == 4
        saes = set(record["sae_params"])
        assert saes and not saes & set(epochs[0]) and epochs[0] == epochs[1]
        assert set(epochs[2]) == set(epochs[0]) | saes and epochs[2] == epochs[3]
        # every tensor of the last block, its split ones included: a view that loses one leaves it frozen
        assert epochs[0] == record["last_block_params"]
        losses = record["losses"]
        assert all(math.isfinite(x) for x in losses)
        # the same batches each epoch, so each batch's loss falls from its first epoch to its last
        per_epoch = len(losses) // 4
        assert all(losses[-per_epoch + i] < losses[i] for i in range(per_epoch))
    assert records[0]["trainable_by_epoch"] == records[1]["trainable_by_epoch"]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="phased tuning is finetuning-scheduler's")
def test_two_rank_fsdp2_tuning_thaws_spliced_saes_by_latent_role(tmp_path):
    """A phased schedule thaws a latent role after a first phase, on two FSDP2 ranks through the core loop
    (#834)."""
    check_latent_phases(launch_latent_worker(tmp_path))


# each block's MLP split by its hidden features; the SAE reads the activation between the two projections
_SPLIT_MLP = r"model\.blocks\.\d+\._original_component\.mlp\._original_component"
SAE_TP_PLAN = {
    rf"{_SPLIT_MLP}\.fc_in\._original_component": {"style": "colwise", "use_local_output": False},
    rf"{_SPLIT_MLP}\.fc_out\._original_component": {"style": "rowwise"},
}
SPLIT_SAE_HOOK = "blocks.0.mlp.hook_post"


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="phased tuning and the tensor-parallel plan are finetuning-scheduler's")
def test_two_rank_tp_tunes_an_sae_spliced_inside_a_split_layer_like_one_process(tmp_path):
    """An SAE that SAE Lens splices into a bridge inside a split MLP reads the whole activation and rewrites each
    rank's shard, so two tensor-parallel ranks tune it in phases like one process, step for step (#853).

    The first projection keeps its split output as a ``DTensor`` (``use_local_output: false``), which the splice
    gathers; an SAE encoding one rank's shard would fail on its width. With the default local output, the splice is
    refused, naming the hook and the fix.
    """
    sae = ("--sae-hook", SPLIT_SAE_HOOK)
    (reference,) = launch_latent_worker(tmp_path / "single", *sae, placement="single")
    check_latent_phases([reference, reference])
    # positive control: the SAE is on the loss's path, so tuning moved every one of its tensors
    assert all(reference["sae_values"][n] != pytest.approx(reference["sae_init"][n]) for n in reference["sae_init"])
    records = launch_latent_worker(tmp_path / "tp", *sae, "--tp-plan", json.dumps(SAE_TP_PLAN), placement="tp")
    check_latent_phases(records)
    for record in records:
        assert record["losses"] == pytest.approx(reference["losses"], rel=1e-5)
        assert record["trainable_by_epoch"] == reference["trainable_by_epoch"]
        assert any(".mlp._original_component.fc_in." in n for n in record["split_params"])
        assert record["sae_values"].keys() == reference["sae_values"].keys()
        for name, values in record["sae_values"].items():
            assert values == pytest.approx(reference["sae_values"][name], rel=1e-4, abs=1e-6), name
    local = {k: {**v, "use_local_output": True} for k, v in SAE_TP_PLAN.items()}
    for record in launch_latent_worker(tmp_path / "local", *sae, "--tp-plan", json.dumps(local), placement="tp"):
        assert f"spliced at {SPLIT_SAE_HOOK} reads activations of width 10" in record["refused"]
        assert "use_local_output: false" in record["refused"]


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


def _launch_tp_ct(out_dir: Path, *extra: str) -> list[dict]:
    out_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    worker = [str(TP_CT_WORKER), "--out-dir", str(out_dir), *extra]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"tp circuit-tracer run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return [json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the tensor-parallel plan is finetuning-scheduler's")
@pytest.mark.parametrize(
    ("extra", "layers"),
    [
        ((), "model.layers"),
        (("--meta",), "model.layers"),
        (("--arch", "gemma3"), "model.language_model.layers"),
    ],
    ids=["loaded", "meta_init", "gemma3"],
)
def test_two_rank_tp_attributes_like_one_process(tmp_path, extra, layers):
    """A circuit-tracer composition whose base model ``tp`` split by its declared plan gives the graph, and the
    feature interventions, of the whole model on every rank, with ``offload='cpu'`` and with live attention too.

    The calls that read inside a split layer or a split weight are refused before they start. A model built on the meta
    device gives the same, which needs the unembedding circuit-tracer resolved at its build to be refreshed. So does a
    multimodal Gemma 3, whose query and key norms the plan leaves whole.
    """
    for record in _launch_tp_ct(tmp_path / "placed", *extra):
        assert record["split_params"] == sorted(
            f"hf_model.{layers}.{i}.{proj}.weight"
            for i in range(3)
            for proj in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        )
        assert record["n_features"] > 0 and record["same_nodes"] and record["ranks_equal"]
        assert record["diff"].keys() >= {"graph", "offload_cpu", "live", "frozen_edit", "unfrozen_edit"}
        assert max(record["diff"].values()) < 1e-4, record["diff"]
        # each edit moves the reference, so a dropped edit could not pass as a match
        assert min(record["effect"].values()) > 1e-2, record["effect"]
        assert record["refusals"].keys() == {"offload_disk", "head_loadings", "head_conductance"}
        assert all("'tp' has split across ranks" in message for message in record["refusals"].values())
        # one rank's perturbed shard changes the graph, so the comparison above can fail
        assert record["control_differs"]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the tensor-parallel plan is finetuning-scheduler's")
def test_two_rank_tp_refuses_a_composition_whose_base_model_it_cannot_reach(tmp_path):
    for record in _launch_tp_ct(tmp_path / "attribution_only", "--attribution-only"):
        assert "tune_base: true" in record["refused"]


def _launch_ct(plan: dict, out_dir: Path, meta: bool = False) -> list[dict]:
    out_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    env["CUDA_VISIBLE_DEVICES"] = ""
    worker = [str(CT_WORKER), "--plan", json.dumps(plan), "--out-dir", str(out_dir), *(["--meta"] if meta else [])]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"circuit-tracer run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return [json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]


@pytest.fixture(scope="module")
def tiny_mixtral_checkpoints(tmp_path_factory):
    """A tiny Mixtral saved twice: as ``save_pretrained`` writes it (its hub layout, whose expert weights the model's
    conversion mapping merges) and from its own state dict (the model's layout)."""
    from safetensors.torch import save_file
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(0)
    model = MixtralForCausalLM(
        MixtralConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_local_experts=2,
            num_experts_per_tok=2,
            max_position_embeddings=64,
        )
    )
    hub, own = tmp_path_factory.mktemp("mixtral_hub"), tmp_path_factory.mktemp("mixtral_own")
    model.save_pretrained(hub)
    model.config.save_pretrained(own)
    save_file(
        {k: v.contiguous().clone() for k, v in model.state_dict().items()},
        own / "model.safetensors",
        metadata={"format": "pt"},
    )
    return hub, own


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
def test_dcp_loads_a_checkpoint_in_the_models_own_layout(tiny_mixtral_checkpoints, tmp_path):
    """A checkpoint written from ``model.state_dict()`` loads with its own keys, although the model's conversion
    mapping has converters, and computes exactly what ``from_pretrained`` computes from the hub-layout checkpoint.

    The hub-layout checkpoint needs the converters, so it is refused, naming the ones that match a key of it.
    """
    hub, own = tiny_mixtral_checkpoints
    extra = ("--arch", "mixtral", "--reference", str(hub), "--loader", "dcp")
    for record in _launch_meta(own, tmp_path / "own", *extra):
        assert record["max_abs_diff"] == 0.0 and record["param_types"] == ["DTensor"]
    for record in _launch_meta(hub, tmp_path / "hub", *extra):
        assert "needs transformers' weight converters" in record["refused"]
        assert "experts.*.w1.weight" in record["refused"]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
def test_a_model_layout_checkpoint_missing_a_key_is_refused_by_name(tiny_mixtral_checkpoints, tmp_path):
    hub, own = tiny_mixtral_checkpoints
    partial = _edited_checkpoint(own, tmp_path / "partial", drop="model.layers.1.mlp.gate.weight")
    for record in _launch_meta(
        partial, tmp_path / "out", "--arch", "mixtral", "--reference", str(hub), "--loader", "dcp"
    ):
        assert "model.layers.1.mlp.gate.weight" in record["refused"]


class TestModelLayoutDecision:
    """Whether a checkpoint's keys are already the model's own, which its key names alone do not show."""

    @staticmethod
    def _models():
        from transformers import GPT2Config, GPT2LMHeadModel, MixtralConfig, MixtralForCausalLM

        sizes = {"num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2, "num_local_experts": 2}
        with torch.device("meta"):
            mixtral = MixtralForCausalLM(MixtralConfig(vocab_size=64, hidden_size=16, intermediate_size=32, **sizes))
            gpt2 = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=16, n_layer=2, n_head=2))
        return mixtral, gpt2

    def test_the_hub_layouts_own_keys_settle_it(self):
        from interpretune.runners.distributed import _hub_layout_keys, _is_model_layout

        mixtral, _ = self._models()
        own, hub = list(mixtral.state_dict()), sorted(_hub_layout_keys(mixtral))
        assert _is_model_layout(mixtral, own, lambda key: False)
        assert not _is_model_layout(mixtral, hub, lambda key: False)
        assert not _is_model_layout(mixtral, [*own, "extra.weight"], lambda key: False)
        assert _is_model_layout(mixtral, [*own, "extra.weight"], lambda key: key == "extra.weight")  # ignorable

    def test_a_model_whose_hub_layout_has_no_key_of_its_own_is_not_shown_to_be_in_it(self, monkeypatch):
        """Equal key sets cannot tell a model-layout checkpoint from a hub one whose tensors a converter edits in
        place, so such a checkpoint keeps the conversion path (and its refusal), rather than being loaded as it
        is."""
        from interpretune.runners import distributed

        mixtral, gpt2 = self._models()
        assert not distributed._is_model_layout(gpt2, list(gpt2.state_dict()), lambda key: False)
        monkeypatch.setattr(distributed, "_hub_layout_keys", lambda model: set(model.state_dict()))
        assert not distributed._is_model_layout(mixtral, list(mixtral.state_dict()), lambda key: False)

    def test_only_a_transformers_model_states_a_hub_layout(self):
        from interpretune.runners.distributed import _is_model_layout

        assert not _is_model_layout(torch.nn.Linear(2, 2), ["weight", "bias"], lambda key: False)


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
@pytest.mark.skipif(not _CT_AVAILABLE, reason="circuit-tracer is not installed")
def test_a_circuit_tracer_model_is_sharded_with_its_direct_calls_registered(tmp_path):
    """A circuit-tracer replacement model sharded on two ranks gives the logits, the nnsight trace, the transcoder
    activations and the sparse encode and decode of one process, once the calls circuit-tracer makes outside the
    module's forward are registered: the Hugging Face module nnsight calls directly, and each transcoder's methods
    and activation function. A registered method calling another on the same unit (``decode_sparse`` calls
    ``_get_decoder_vectors``) must not reshard the unit under its caller.

    A plan that leaves the Hugging Face module to the root unit, whose hooks those direct calls bypass, is refused by
    name.
    """
    units = {"class:Gemma2DecoderLayer": {}, "class:SingleLayerTranscoder": {}, "class:JumpReLU": {}}
    for record in _launch_ct({**units, "class:Gemma2ForCausalLM": {}}, tmp_path / "placed"):
        assert record["active"] > 0  # so equal activations are not two empty caches
        assert (
            record["plain_diff"] == record["trace_diff"] == record["activations_diff"] == record["sparse_diff"] == 0.0
        )
    for record in _launch_ct(units, tmp_path / "no_entry_unit"):
        assert "calls hf_model directly" in record["refused"]
        assert "Add `class:Gemma2ForCausalLM` to `fsdp_plan`" in record["refused"]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
@pytest.mark.skipif(not _CT_AVAILABLE, reason="circuit-tracer is not installed")
def test_a_meta_built_circuit_tracer_model_loads_from_rank0(tmp_path):
    """A circuit-tracer replacement model built on the meta device (``meta_replacement_model``), sharded on two
    ranks and loaded from rank 0's copy into the Hugging Face module nnsight wraps, gives the logits, the nnsight
    trace, the transcoder activations and the sparse encode and decode of the copy in one process."""
    plan = {f"class:{c}": {} for c in ("Gemma2DecoderLayer", "SingleLayerTranscoder", "JumpReLU", "Gemma2ForCausalLM")}
    for record in _launch_ct(plan, tmp_path / "meta", meta=True):
        assert record["active"] > 0
        assert (
            record["plain_diff"] == record["trace_diff"] == record["activations_diff"] == record["sparse_diff"] == 0.0
        )


@pytest.mark.skipif(not _CT_AVAILABLE, reason="circuit-tracer is not installed")
def test_circuit_tracer_meta_init_refuses_what_the_meta_build_cannot_honor(monkeypatch):
    """The circuit-tracer meta build refuses, by name and all at once, a backend other than nnsight, ``dcp``
    loading, lazily loaded transcoder weights, a composition without ``nnsight_cfg`` (whose Hugging Face module is
    never registered) and a placement of its own; otherwise rank 0 keeps the loaded copy as the meta-init source
    and the module records that it built on the meta device."""
    from types import SimpleNamespace

    from interpretune.adapters.circuit_tracer import adapter
    from interpretune.config import CircuitTracerConfig
    from interpretune.config.mixins import HFFromPretrainedConfig
    from interpretune.runners.distributed import META_INIT_SOURCE

    def stub(load: str, nnsight_cfg: object | None = object()):
        hf_cfg = HFFromPretrainedConfig(meta_init=True, meta_init_load=load)
        it_cfg = SimpleNamespace(hf_from_pretrained_cfg=hf_cfg, model_name_or_path="m", nnsight_cfg=nnsight_cfg)
        return SimpleNamespace(it_cfg=it_cfg)

    build = adapter.BaseCircuitTracerModule._meta_replacement_model
    with pytest.raises(MisconfigurationException, match="cannot do") as refused:
        placement = {"device_map": "a", "remote": 1}
        build(stub("dcp", None), CircuitTracerConfig(backend="transformerlens"), True, placement)
    for name in (
        "'transformerlens'",
        "`meta_init_load: dcp`",
        "`lazy_decoder: false`",
        "`nnsight_cfg`",
        "device_map, remote",
    ):
        assert name in str(refused.value)
    with pytest.raises(MisconfigurationException, match="lazily loaded transcoder weights"):
        build(stub("rank0"), CircuitTracerConfig(backend="nnsight"), False, {})  # lazy_decoder defaults to True

    built, source = object(), torch.nn.Linear(1, 1)
    calls = []
    monkeypatch.setattr(adapter, "meta_replacement_model", lambda *a, **k: calls.append((a, k)) or (built, source))
    monkeypatch.delenv("RANK", raising=False)
    module = stub("rank0")
    cfg = CircuitTracerConfig(backend="nnsight", lazy_decoder=False, transcoder_set="t", model_name="n")
    assert build(module, cfg, False, {"dispatch": True}) is built
    assert calls == [(("n", "t", cfg.dtype), {"load_source": True})]
    assert getattr(module, META_INIT_SOURCE) is source and module._meta_init_built
    monkeypatch.setenv("RANK", "1")
    build(stub("rank0"), cfg, False, {})
    assert calls[-1][1] == {"load_source": False}


def test_an_nnsight_module_loads_meta_weights_into_its_registered_hf_module():
    from types import SimpleNamespace

    from interpretune.adapters.nnsight.adapter import BaseNNsightModule

    hf_model = torch.nn.Linear(1, 1)
    module = SimpleNamespace(_modules={"hf_model": hf_model}, model=object())  # nnsight's wrapper is no torch module
    assert BaseNNsightModule.meta_load_target(module) is hf_model


@pytest.mark.skipif(not _CT_AVAILABLE, reason="circuit-tracer is not installed")
def test_transcoder_direct_calls_refuse_an_unenumerated_transcoder_type():
    from interpretune.adapters.circuit_tracer.adapter import transcoder_direct_calls

    assert transcoder_direct_calls(nn.Module()) == []  # nothing to shard, nothing to declare
    other = nn.Module()
    other.weight = nn.Parameter(torch.zeros(2))
    with pytest.raises(MisconfigurationException, match="only `SingleLayerTranscoder` sets are"):
        transcoder_direct_calls(other)


def _launch_tp_analysis(out_dir: Path) -> list[dict]:
    out_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    worker = [str(TP_ANALYSIS_WORKER), "--out-dir", str(out_dir)]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"tp analysis run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return [json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_tp_analysis_shares_every_step_and_writes_one_store(tmp_path):
    """Under placement 'tp' the analysis runner's ranks run every step together, in one shared (shuffled) order,
    and only rank 0 writes the store, which every rank returns.

    The worker's step makes a collective per batch, so a rank running a step alone would hang the run. A second run with
    the same deterministic fingerprint is served from rank 0's cache: its generator never starts, and the other rank
    must not start it either. J-lens readouts read the unembedding, the final norm and the embedding whole on split
    weights, gathering each once per run.
    """
    records = _launch_tp_analysis(tmp_path / "run")
    assert records[0]["fresh"]["rows"] == records[1]["fresh"]["rows"]
    rows = records[0]["fresh"]["rows"]
    assert sorted(rows["row"]) == sorted(list(range(10)) * 2)  # two epochs over ten rows
    assert rows["row"][:10] != rows["row"][10:], "each epoch reshuffles, alike on both ranks"
    for record in records:
        assert record["fresh"]["steps"] == 8  # four batches per epoch, on every rank
    assert records[0]["fresh"]["store_written"] and records[0]["fresh"]["cache_files"]
    assert not records[1]["fresh"]["store_written"] and not records[1]["fresh"]["cache_files"]
    for record in records:
        assert record["keyed_first"]["steps"] == 4
        assert record["keyed_second"]["steps"] == 0, "a cache hit runs no step on any rank"
        assert record["keyed_second"]["rows"] == record["keyed_first"]["rows"] == records[0]["keyed_first"]["rows"]
        weights = record["weights"]
        assert weights["split_params"] == ["lm_head.weight", "model.embed_tokens.weight"]
        assert weights["lm_head_local_rows"] == 32  # each rank holds half the vocabulary
        assert weights["unembed_whole"] and weights["embedding_whole"] and weights["unembed_type"] == "Tensor"
        assert weights["norm_scale_equal"] and weights["norm_kind"] == "rmsnorm" and weights["readout_equal"]
        assert weights["gathered_once"] and weights["gathers_outside"] == 2 and weights["outside_fresh"]


def test_merged_row_order_interleaves_shares_by_batch():
    """Rows are put back by epoch, then batch index, then their place in the batch, whichever rank yielded them."""
    from interpretune.runners.analysis import merged_row_order

    rank0 = [(0, 0, 1), (0, 2, 2), (1, 0, 1)]  # rank 0's shard rows: b0 | b2, b2 | epoch 1 b0
    rank1 = [(0, 1, 1), (1, 1, 1)]  # rank 1's shard rows (laid after rank 0's): b1 | epoch 1 b1
    assert merged_row_order([rank0, rank1]) == [0, 4, 1, 2, 3, 5]


def test_batch_share_takes_every_nth_batch_without_padding():
    from torch.utils.data import BatchSampler, SequentialSampler

    from interpretune.runners.distributed import BatchShare

    batches = BatchSampler(SequentialSampler(range(10)), batch_size=2, drop_last=False)
    shares = [list(BatchShare(batches, index, 2)) for index in range(2)]
    assert shares == [[[0, 1], [4, 5], [8, 9]], [[2, 3], [6, 7]]]
    assert [len(BatchShare(batches, index, 2)) for index in range(2)] == [3, 2]


def _launch_ddp_analysis(out_dir: Path) -> list[dict]:
    out_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    worker = [str(DDP_ANALYSIS_WORKER), "--out-dir", str(out_dir)]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", *worker]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"ddp analysis run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return [json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]


@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.skipif(not _FTS_AVAILABLE, reason="the FSDP2 plan grammar is finetuning-scheduler's")
def test_two_rank_ddp_analysis_merges_shares_into_the_one_process_store(tmp_path):
    """Under FSDP2 replication ('ddp') each rank analyzes its own batches of the shared sequence, and the shares
    merge into the store one process writes, row for row, which every rank returns.

    The step runs a backward pass through a replicated layer, and the five batches split three and two, so FSDP2's
    gradient reduction (off for the run) would leave a rank waiting forever. An op declaring cross-batch state and a
    deterministic generator-cache key are refused. Each epoch's ``log_analysis`` averages pool every rank's batches, so
    they are the one-process averages on every rank (a rank's own share alone averages three batches or two).
    """
    records = _launch_ddp_analysis(tmp_path / "run")
    reference = records[0]["reference"]
    assert reference["batch_idx"] == [0, 1, 2, 3, 4]
    reference_metrics = records[0]["reference_metrics"]
    assert len(reference_metrics) == 1 and set(reference_metrics[0]) == {"out"}
    reference_out = reference_metrics[0]["out"]
    for record in records:
        assert record["param_types"] == ["DTensor"]
        assert record["ordered"] == reference, "the merged store is the one-process store"
        assert [m["out"] for m in record["ordered_metrics"]] == pytest.approx([reference_out]), "pools every share"
        assert record["shuffled_metrics"] == records[0]["shuffled_metrics"]
        # each epoch sees every example once, in another pairing, and a batch's output sums over its examples
        assert [m["out"] for m in record["shuffled_metrics"]] == pytest.approx([reference_out] * 2)
        assert record["share_dirs_left"] == []
        assert record["shuffled"] == records[0]["shuffled"]
        assert record["shuffled"]["batch_idx"] == [0, 1, 2, 3, 4] * 2
        for epoch in range(2):
            examples = sum(record["shuffled"]["examples"][epoch * 5 : (epoch + 1) * 5], [])
            assert sorted(examples) == list(range(10))
        assert record["shuffled"]["examples"][:5] != record["shuffled"]["examples"][5:]
        assert record["limited"]["batch_idx"] == [0, 1, 2]
        assert "running_mean declare cross-batch `op_state`" in record["refusals"]["op_state"]
        assert "Leave `dataset_fingerprint`" in record["refusals"]["fingerprint"]
