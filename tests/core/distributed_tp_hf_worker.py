"""One rank of the two-rank check of the default tensor-parallel plan for a Hugging Face model.

Each rank loads the same tiny Llama twice: one copy is placed under ``tp`` with no ``tp_plan``, so the plan transformers
declares for the model is used (its decoder layers split the Megatron way: each rank computes its own attention heads
and its own slice of the MLP), and the other stays whole as the reference. Rank 1's copy is perturbed first, so only
the broadcast can make the ranks agree. Both take one SGD step on the same batch.

Writes ``<out_dir>/rank<rank>.json`` with the split parameters and the forward's and the stepped forward's distance
from the reference, or the placement's refusal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


class Holder(nn.Module):
    """Holds the model as ``model`` and declares its default plan, as an interpretune module's ``default_tp_plan``
    does."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits

    def default_tp_plan(self, world_size: int) -> dict:
        from interpretune.runners.distributed import hf_tp_plan

        return hf_tp_plan(self.model, "model", world_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from transformers import LlamaForCausalLM

    from interpretune.runners.distributed import init_process_group, place_module, resolve_placement
    from interpretune.utils import MisconfigurationException

    run = resolve_placement("tp")
    assert run is not None
    init_process_group(run)
    placed = Holder(LlamaForCausalLM.from_pretrained(args.checkpoint))
    reference = Holder(LlamaForCausalLM.from_pretrained(args.checkpoint))
    if run.rank:
        with torch.no_grad():
            for param in placed.parameters():
                param.add_(1.0)
    record: dict = {}
    out = Path(args.out_dir, f"rank{run.rank}.json")
    try:
        place_module(placed, run)
    except MisconfigurationException as err:
        out.write_text(json.dumps({"refused": str(err)}))
        return
    record["split_params"] = sorted(n for n, p in placed.named_parameters() if type(p.data).__name__ == "DTensor")
    input_ids = torch.arange(12).reshape(2, 6) % 128
    with torch.no_grad():
        before = reference(input_ids)
        record["forward_diff"] = float((placed(input_ids) - before).abs().max())
    stepped = {}
    for name, model in (("placed", placed), ("reference", reference)):
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), input_ids[:, 1:].flatten())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            stepped[name] = model(input_ids)
    record["step_diff"] = float((stepped["placed"] - stepped["reference"]).abs().max())
    record["step_effect"] = float((stepped["reference"] - before).abs().max())
    out.write_text(json.dumps(record))


if __name__ == "__main__":
    main()
