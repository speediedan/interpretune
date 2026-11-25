"""One rank of the two-rank check of the default tensor-parallel plan for a Hugging Face model.

Each rank loads the same tiny Llama twice: one copy is placed under ``tp`` with no ``tp_plan``, so the plan transformers
declares for the model is used (its decoder layers split the Megatron way: each rank computes its own attention heads
and its own slice of the MLP), and the other stays whole as the reference. Rank 1's copy is perturbed first, so only
the broadcast can make the ranks agree. Both take one SGD step on the same batch, from a gradient accumulated over two
backward passes.

A checkpoint whose model has query and key norms (Gemma 3) has them frozen when the model is placed and thawed before
the step, as a schedule would. ``--tp-plan`` gives the plan explicitly, and ``--no-gradient-sum`` is the control that
leaves each rank's gradient of those norms its own. ``--bridge`` runs the model inside a TransformerLens bridge.

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
        out = self.model(input_ids)
        return getattr(out, "logits", out)  # a bridge returns the logits

    def default_tp_plan(self, world_size: int) -> dict:
        from interpretune.runners.distributed import hf_tp_plan

        return hf_tp_plan(self.model, "model", world_size)


def load(checkpoint: str, bridge: bool) -> Holder:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    if bridge:
        from transformer_lens.model_bridge import TransformerBridge

        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")  # warmed for offline CI
        model = TransformerBridge.boot_transformers(checkpoint, hf_model=model, tokenizer=tokenizer, device="cpu")
        for param in model.parameters():
            param.requires_grad_(True)
    return Holder(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tp-plan", type=json.loads, default=None)
    parser.add_argument("--bridge", action="store_true")
    parser.add_argument("--no-gradient-sum", action="store_true")
    args = parser.parse_args()

    import interpretune.runners.distributed as distributed
    from interpretune.runners.distributed import init_process_group, place_module, resolve_placement
    from interpretune.utils import MisconfigurationException

    run = resolve_placement("tp")
    assert run is not None
    init_process_group(run)
    if args.no_gradient_sum:
        distributed._sum_partial_gradients = lambda *args: None
    placed, reference = (load(args.checkpoint, args.bridge) for _ in range(2))
    norms = [n for n, _ in placed.named_parameters() if ".q_norm." in n or ".k_norm." in n]
    for name in norms:
        placed.get_parameter(name).requires_grad_(False)
    if run.rank:
        with torch.no_grad():
            for param in placed.parameters():
                param.add_(1.0)
    record: dict = {}
    out = Path(args.out_dir, f"rank{run.rank}.json")
    try:
        place_module(placed, run, tp_plan=args.tp_plan)
    except MisconfigurationException as err:
        out.write_text(json.dumps({"refused": str(err)}), encoding="utf-8")
        return
    record["split_params"] = sorted(n for n, p in placed.named_parameters() if type(p.data).__name__ == "DTensor")
    input_ids = torch.arange(12).reshape(2, 6) % 128
    with torch.no_grad():
        before = reference(input_ids)
        record["forward_diff"] = float((placed(input_ids) - before).abs().max())
    record["frozen_stayed_frozen"] = not any(placed.get_parameter(n).requires_grad for n in norms)
    for norm in norms:
        placed.get_parameter(norm).requires_grad_(True)
    stepped, grads = {}, {}
    for name, model in (("placed", placed), ("reference", reference)):
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        logits = model(input_ids)
        for batch in (logits, model(input_ids.flip(0, 1))):
            nn.functional.cross_entropy(batch[:, :-1].flatten(0, 1), input_ids[:, 1:].flatten()).backward()
        grads[name] = {norm: model.get_parameter(norm).grad for norm in norms}
        optimizer.step()
        with torch.no_grad():
            stepped[name] = model(input_ids)
    record["norm_grad_diff"] = {
        n: float((g - grads["reference"][n]).abs().max() / grads["reference"][n].abs().max())
        for n, g in grads["placed"].items()
    }
    record["step_diff"] = float((stepped["placed"] - stepped["reference"]).abs().max())
    record["step_effect"] = float((stepped["reference"] - before).abs().max())
    out.write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    main()
