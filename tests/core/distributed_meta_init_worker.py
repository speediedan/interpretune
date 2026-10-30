"""One rank of the meta-device placement runs (``tests/core/test_core_loop_distributed.py``).

Every rank builds a small GPT-2 from the checkpoint's config on the meta device, and rank 0 also loads the checkpoint
on the CPU, as ``HFFromPretrainedConfig.meta_init`` does. ``place_module`` then shards the model with a block plan and
loads it from rank 0. With ``--drop ln_f``, rank 0's copy lacks the final norm, which must be refused by name.

Writes ``<out_dir>/rank<rank>.json``: the largest difference between the placed model's logits and a plain
``from_pretrained`` copy's on a fixed input, whether the LM head is still tied to the embedding, and the parameter types
after placement, or the refusal's message.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn


class Holder(nn.Module):
    """The part of an interpretune module the placement reads: the model, and rank 0's meta-init source."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # through the root unit, whose pre-forward hook gathers the parameters no block unit holds (the embeddings)
        return self.model(input_ids).logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--drop", default=None, help="a submodule to delete from rank 0's copy (a planted defect)")
    args = parser.parse_args()

    from transformers import GPT2LMHeadModel

    from interpretune.runners.distributed import (
        META_INIT_SOURCE,
        init_process_group,
        place_module,
        resolve_placement,
    )
    from interpretune.utils import MisconfigurationException

    run = resolve_placement("fsdp2")
    assert run is not None
    init_process_group(run)
    reference = GPT2LMHeadModel.from_pretrained(args.checkpoint)
    with torch.device("meta"):
        holder = Holder(GPT2LMHeadModel._from_config(reference.config).eval())  # `from_pretrained` returns eval mode
    if run.rank == 0:
        source = GPT2LMHeadModel.from_pretrained(args.checkpoint)
        if args.drop:
            delattr(source.transformer, args.drop)
        object.__setattr__(holder, META_INIT_SOURCE, source)
    record: dict = {}
    try:
        place_module(holder, run, {"class:GPT2Block": {}})
    except MisconfigurationException as err:
        record["refused"] = str(err)
    else:
        # read before the forward: FSDP2 keeps the root unit's parameters unsharded after a forward until backward
        record["param_types"] = sorted({type(p).__name__ for p in holder.parameters()})
        input_ids = torch.arange(12).reshape(2, 6)
        with torch.no_grad():
            placed = holder(input_ids)
            expected = reference(input_ids).logits
        model = holder.model
        record["max_abs_diff"] = float((placed - expected).abs().max())
        record["tied"] = model.lm_head.weight is model.transformer.wte.weight
        record["source_dropped"] = not hasattr(holder, META_INIT_SOURCE)
    Path(args.out_dir, f"rank{os.environ['RANK']}.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()
