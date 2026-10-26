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
