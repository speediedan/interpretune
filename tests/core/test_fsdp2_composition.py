"""FSDP2 can shard an interpretune module (#820): ``fully_shard`` wraps a module by subclassing its composed class
with no build context, which ``ITMeta`` builds as an ordinary subclass."""

from __future__ import annotations

import sys

import pytest
import torch
import torch.distributed as dist


@pytest.mark.skipif(sys.platform != "linux", reason="a single-rank gloo process group, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_fully_shard_wraps_an_interpretune_module(get_it_session__core_cust__initonly, tmp_path):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, fully_shard

    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    module = get_it_session__core_cust__initonly.it_session.module
    composed = type(module)
    dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "store"), 1), rank=0, world_size=1)
    try:
        # the interpretune module as the root unit, as finetuning-scheduler's `shard_model` makes it
        fully_shard(module, mesh=init_device_mesh("cpu", (1,)))
        assert isinstance(module, FSDPModule) and isinstance(module, composed)
        assert type(module).__name__ == f"FSDP{composed.__name__}"
        assert all(type(p).__name__ == "DTensor" for p in module.parameters())
        assert torch.is_tensor(next(module.parameters()).to_local())
    finally:
        dist.destroy_process_group()


def _compositions() -> dict[str, tuple]:
    """Each composition's session configuration, the ``fsdp_plan`` it places with, and the refusal of a plan
    missing the unit its backend calls directly (or of the composition itself)."""
    from tests.core.cfg_aliases import CoreNNsightGPT2, CoreSLBridgeGPT2, CoreSLNNsightGPT2
    from tests.parity_acceptance.test_it_l import CoreCfg
    from tests.parity_acceptance.test_it_tl import TLParityCfg

    blocks = {"class:GPT2Block": None}
    whole_model = {**blocks, "class:GPT2LMHeadModel": None}
    calls_model = "calls hf_model directly.*Add `class:GPT2LMHeadModel` to `fsdp_plan`"
    return {
        "core": (CoreCfg, {"class:TransformerBlock": None}, None),
        "nnsight": (CoreNNsightGPT2, whole_model, (blocks, calls_model)),
        "sae_lens+nnsight": (CoreSLNNsightGPT2, whole_model, (blocks, calls_model)),
        "transformer_lens": (TLParityCfg, {"class:NativeBlock": None}, None),
        "sae_lens+transformer_lens bridge": (
            CoreSLBridgeGPT2,
            None,
            (blocks, r"cannot shard non-contiguous parameters.*blocks\.0\..*attn\.q\..*cannot be placed with 'fsdp2'"),
        ),
    }


@pytest.fixture
def placed_gloo_run(tmp_path):
    """One gloo rank in this process, in which ``place_module`` runs as a rank of a placement does."""
    from interpretune.runners.distributed import DistributedRun, place_module

    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "store"), 1), rank=0, world_size=1)
    yield lambda module, plan: place_module(module, DistributedRun("fsdp2", 0, 0, 1), fsdp_plan=plan)
    dist.destroy_process_group()


def _session_module(cfg, name, tmp_path):
    from tests.configuration import config_modules

    return config_modules(cfg(), name, {}, tmp_path, {}, False).module


PLACED = ["core", "nnsight", "sae_lens+nnsight", "transformer_lens"]
REFUSED = ["nnsight", "sae_lens+nnsight", "sae_lens+transformer_lens bridge"]


@pytest.mark.skipif(sys.platform != "linux", reason="a single-rank gloo process group, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.parametrize("composition", PLACED)
def test_a_composition_places_under_fsdp2_with_its_plan(composition, placed_gloo_run, tmp_path):
    """Placement keeps every parameter's values, sharded: with one rank, each local shard is the whole tensor."""
    from torch.distributed.tensor import DTensor

    cfg, plan, _ = _compositions()[composition]
    module = _session_module(cfg, "fsdp2_placed", tmp_path)
    before = {name: p.detach().clone() for name, p in module.named_parameters()}
    placed_gloo_run(module, plan)
    after = dict(module.named_parameters())
    assert set(after) == set(before)
    for name, param in after.items():
        assert isinstance(param, DTensor), f"{name} was not sharded"
        torch.testing.assert_close(param.full_tensor(), before[name], rtol=0, atol=0, msg=name)


@pytest.mark.skipif(sys.platform != "linux", reason="a single-rank gloo process group, on Linux only")
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.parametrize("composition", REFUSED)
def test_a_composition_that_cannot_place_is_refused_by_name(composition, placed_gloo_run, tmp_path):
    from interpretune.utils import MisconfigurationException

    cfg, _, (plan, match) = _compositions()[composition]
    module = _session_module(cfg, "fsdp2_refused", tmp_path)
    with pytest.raises(MisconfigurationException, match=match):
        placed_gloo_run(module, plan)
