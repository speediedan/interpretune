"""One rank of the two-rank check of an nnsight composition under placement ``tp`` with its default plan.

Each rank builds the same ``NNsightModule`` on a tiny Llama checkpoint twice. One copy is placed under ``tp`` with no
``tp_plan``, so the plan comes from the module's own ``default_tp_plan``: the plan transformers declares for the Hugging
Face model nnsight wraps, keyed under ``hf_model``, where the module registers it. The other copy stays whole as the
reference. Rank 1's placed copy is perturbed first, so only the broadcast can make the ranks agree.

Readings through nnsight's trace on both copies: an attention output and an MLP output at a layer boundary, the input of
a row-parallel ``down_proj`` (inside the split MLP, so a ``DTensor`` on the placed copy), the logits, three edits (a
slice at a layer boundary, an elementwise scale inside the split MLP, a slice inside it) and the gradient of a layer's
output through nnsight's backward tracer.

Writes ``<out_dir>/rank<rank>.json`` with the split parameters, the type of the split activation, each reading's
largest distance from the reference, each edit's effect on the reference, and the slice edit inside the split MLP's
distance from the reference's UNEDITED logits; or the placement's refusal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nnsight
import torch


def _full(value) -> torch.Tensor:
    return (value.full_tensor() if hasattr(value, "full_tensor") else value).detach().clone()


def build(checkpoint: str):
    from interpretune.adapters.nnsight.adapter import NNsightModule
    from interpretune.adapters.nnsight.config import ITNNsightConfig, NNsightConfig

    cfg = ITNNsightConfig(
        model_name_or_path=checkpoint,
        nnsight_cfg=NNsightConfig(model_name=checkpoint, device_map="cpu", dispatch=True),
    )
    module = NNsightModule(it_cfg=cfg)
    if "hf_model" not in module._modules:
        module.register_tunable_modules()
    return module


def readings(lm, ids: torch.Tensor) -> dict:
    out: dict = {}
    with torch.no_grad():
        with lm.trace(ids):
            attn0 = nnsight.save(lm.model.layers[0].self_attn.output[0])
            mlp0 = nnsight.save(lm.model.layers[0].mlp.output)
            down1_in = nnsight.save(lm.model.layers[1].mlp.down_proj.input)
            logits = nnsight.save(lm.lm_head.output)
        out["split_activation_type"] = type(down1_in).__name__
        out.update(attn0=_full(attn0), mlp0=_full(mlp0), down1_in=_full(down1_in), logits=_full(logits))
        with lm.trace(ids):
            lm.model.layers[0].mlp.output[:, :, :8] = 0
            edited = nnsight.save(lm.lm_head.output)
        out["boundary_slice_edit"] = _full(edited)
        with lm.trace(ids):
            lm.model.layers[1].mlp.down_proj.input = lm.model.layers[1].mlp.down_proj.input * 0.5
            edited = nnsight.save(lm.lm_head.output)
        out["split_scale_edit"] = _full(edited)
        with lm.trace(ids):
            lm.model.layers[1].mlp.down_proj.input[:, :, :8] = 0
            edited = nnsight.save(lm.lm_head.output)
        out["split_slice_edit"] = _full(edited)
    with lm.trace(ids):
        hidden = lm.model.layers[1].output
        loss = lm.lm_head.output.sum()
        with loss.backward():
            grad = nnsight.save(hidden.grad)
    out["grad"] = _full(grad)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from interpretune.runners.distributed import init_process_group, place_module, resolve_placement
    from interpretune.utils import MisconfigurationException

    run = resolve_placement("tp")
    assert run is not None
    init_process_group(run)
    placed, reference = build(args.checkpoint), build(args.checkpoint)
    if run.rank:
        with torch.no_grad():
            for param in placed.parameters():
                param.add_(1.0)
    out = Path(args.out_dir, f"rank{run.rank}.json")
    try:
        place_module(placed, run)
    except MisconfigurationException as err:
        out.write_text(json.dumps({"refused": str(err)}))
        return
    record: dict = {
        "split_params": sorted(n for n, p in placed.named_parameters() if type(p.data).__name__ == "DTensor")
    }
    input_ids = torch.arange(12).reshape(2, 6) % 128
    ref, got = readings(reference.model, input_ids), readings(placed.model, input_ids)
    record["split_activation_type"] = got.pop("split_activation_type")
    ref.pop("split_activation_type")
    record["diff"] = {k: float((got[k] - ref[k]).abs().max()) for k in ref}
    record["effect"] = {k: float((ref[k] - ref["logits"]).abs().max()) for k in ref if k.endswith("_edit")}
    record["split_slice_edit_vs_unedited"] = float((got["split_slice_edit"] - ref["logits"]).abs().max())
    out.write_text(json.dumps(record))


if __name__ == "__main__":
    main()
