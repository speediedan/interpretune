"""One rank of the two-rank tensor-parallel check of a TransformerLens bridge's joint q/k/v.

The bridge's split q, k and v are views of its packed ``qkv`` weight. A plan splitting them must split the packed weight
with them, so that views and packed weight keep sharing storage through training. Each rank boots the same tiny GPT-2
bridge twice: one copy is placed under ``tp`` (its packed weights perturbed off rank 0 first, so only the broadcast can
make the ranks agree), the other stays whole as the reference. Both take one SGD step on the same batch.

Writes ``<out_dir>/rank<rank>.json`` with the split parameters, the forward's and the stepped forward's distance from
the reference, whether each view still shares the packed weight's storage, and whether the packed weight's
``full_tensor()`` is the reference's packed weight after the step.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tp-plan", type=json.loads, required=True, help="the tensor-parallel plan, as JSON")
    parser.add_argument("--no-packed-split", action="store_true", help="leave packed weights whole (a planted defect)")
    args = parser.parse_args()

    from transformer_lens.model_bridge import TransformerBridge

    from interpretune.runners import distributed
    from interpretune.runners.distributed import init_process_group, place_module, resolve_placement
    from interpretune.utils import MisconfigurationException

    if args.no_packed_split:
        distributed._split_packed = lambda group, mesh, rank: None

    run = resolve_placement("tp")
    assert run is not None
    init_process_group(run)
    placed = TransformerBridge.boot_transformers(args.checkpoint, device="cpu")
    reference = TransformerBridge.boot_transformers(args.checkpoint, device="cpu")
    attns = [block._original_component.attn for block in placed.transformer.h]
    if run.rank:
        with torch.no_grad():
            for attn in attns:
                attn.qkv._original_component.weight.add_(1.0)  # the views move with it: they share its storage
    record: dict = {}
    try:
        place_module(placed, run, tp_plan=args.tp_plan)
    except MisconfigurationException as err:
        record["refused"] = str(err)
        Path(args.out_dir, f"rank{run.rank}.json").write_text(json.dumps(record))
        return
    record["split_params"] = sorted(n for n, p in placed.named_parameters() if type(p.data).__name__ == "DTensor")
    input_ids = torch.arange(12).reshape(2, 6) % 128

    def local(param: torch.Tensor) -> torch.Tensor:
        return param.to_local() if hasattr(param, "to_local") else param

    def aliased() -> bool:
        for attn in attns:
            packed = attn.qkv._original_component
            for part in ("weight", "bias"):
                ptr = local(getattr(packed, part)).untyped_storage().data_ptr()
                for view in (attn.q, attn.k, attn.v):
                    if local(getattr(view._original_component, part)).untyped_storage().data_ptr() != ptr:
                        return False
        return True

    def packed_matches_reference() -> float:
        worst = 0.0
        for attn, ref_block in zip(attns, reference.transformer.h):
            ref_packed = ref_block._original_component.attn.qkv._original_component
            for part in ("weight", "bias"):
                full = getattr(attn.qkv._original_component, part)
                full = full.full_tensor() if hasattr(full, "full_tensor") else full
                worst = max(worst, float((full - getattr(ref_packed, part)).abs().max()))
        return worst

    with torch.no_grad():
        before = reference(input_ids)
        record["forward_diff"] = float((placed(input_ids) - before).abs().max())
    record["aliased_before_step"] = aliased()
    record["packed_diff_before_step"] = packed_matches_reference()
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
    record["aliased_after_step"] = aliased()
    # the views were stepped, the packed weight never is (no forward reads it): it moved only through shared storage
    record["packed_diff_after_step"] = packed_matches_reference()
    Path(args.out_dir, f"rank{os.environ['RANK']}.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()
