"""One rank of the circuit-tracer placement runs (``tests/core/test_core_loop_distributed.py``).

Every rank builds the same tiny circuit-tracer replacement model (``tests/core/circuit_tracer_toy.py``) and records its
logits, an nnsight trace of its LM head and its transcoder activations. A module holding the model's Hugging Face
module declares the calls circuit-tracer makes outside its forward, as the nnsight and circuit-tracer adapters do, and
``place_module`` shards it with the given plan. The same three readings are then taken through the sharded model.

Writes ``<out_dir>/rank<rank>.json``: the largest difference of each reading from the unsharded one and the number of
non-zero transcoder activations, or the placement's refusal.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nnsight
import torch
from torch import nn


class Holder(nn.Module):
    """The part of an nnsight circuit-tracer module the placement reads: the registered Hugging Face module and the
    calls circuit-tracer makes on it and on its transcoders outside this module's forward."""

    def __init__(self, replacement_model) -> None:
        super().__init__()
        self.hf_model = replacement_model._model
        self.transcoders = replacement_model.transcoders._module

    def fsdp_direct_calls(self):
        from interpretune.adapters.circuit_tracer.adapter import transcoder_direct_calls

        return [("hf_model", self.hf_model, ("forward",)), *transcoder_direct_calls(self.transcoders)]


def readings(model, prompt: torch.Tensor) -> dict[str, torch.Tensor]:
    from tests.core.circuit_tracer_toy import only_token_zero_special

    with torch.no_grad():
        plain = model._model(prompt[None]).logits.clone()
        with model.trace(prompt[None]):
            traced = nnsight.save(model.lm_head.output)
    with only_token_zero_special(model):
        _, activations = model.get_activations(prompt)
    return {"plain": plain, "trace": traced.clone(), "activations": activations.clone()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="the FSDP2 plan, as JSON")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from interpretune.runners.distributed import init_process_group, place_module, resolve_placement
    from interpretune.utils import MisconfigurationException
    from tests.core.circuit_tracer_toy import TOY_PROMPT, tiny_gemma2_replacement_model

    run = resolve_placement("fsdp2")
    assert run is not None
    init_process_group(run)
    model = tiny_gemma2_replacement_model()
    # FSDP2 cannot shard a 0-d parameter, and the toy's JumpReLU threshold is one; trained transcoders carry a
    # per-feature threshold, so widen it to one dimension (the same function, by broadcasting)
    for module in model._model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if param.dim() == 0:
                setattr(module, name, nn.Parameter(param.detach().reshape(1).clone()))
    expected = readings(model, TOY_PROMPT)
    record: dict = {"active": int((expected["activations"] != 0).sum())}
    try:
        place_module(Holder(model), run, json.loads(args.plan))
    except MisconfigurationException as err:
        record["refused"] = str(err)
    else:
        placed = readings(model, TOY_PROMPT)
        record.update({f"{k}_diff": float((placed[k] - expected[k]).abs().max()) for k in expected})
    Path(args.out_dir, f"rank{os.environ['RANK']}.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()
