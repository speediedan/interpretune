"""One rank of the meta-device placement runs (``tests/core/test_core_loop_distributed.py``).

Every rank builds a small model (GPT-2, or Llama with ``--arch llama``) from the checkpoint's config on the meta
device, as ``HFFromPretrainedConfig.meta_init`` does. With ``--loader rank0``, rank 0 also loads the checkpoint on the
CPU and ``place_module`` loads the sharded model from it; with ``--drop ln_f``, rank 0's copy lacks the final norm,
which must be refused by name. With ``--loader dcp``, only the parameters are built on meta (buffers keep their values)
and every rank reads its own slices of the checkpoint's safetensors files.

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


def with_adapters(model: nn.Module) -> nn.Module:
    """``model`` with seeded LoRA adapters whose ``B`` is non-zero, so they change the output (PEFT starts it at
    zero).

    The same seed gives every copy the same values, and a meta-built model the same keys.
    """
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn", "c_proj"], lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    if not any(p.is_meta for p in model.parameters()):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(0.0, 0.5)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--drop", default=None, help="a submodule to delete from rank 0's copy (a planted defect)")
    parser.add_argument("--loader", default="rank0", choices=("rank0", "dcp"))
    parser.add_argument("--arch", default="gpt2", choices=("gpt2", "llama"))
    parser.add_argument("--lora", action="store_true", help="LoRA on the model and on rank 0's copy (GPT-2, rank0)")
    args = parser.parse_args()

    from transformers import GPT2LMHeadModel, LlamaForCausalLM

    from interpretune.runners.distributed import (
        META_INIT_CHECKPOINT,
        META_INIT_SOURCE,
        init_process_group,
        params_on_meta,
        place_module,
        resolve_placement,
    )
    from interpretune.utils import MisconfigurationException

    run = resolve_placement("fsdp2")
    assert run is not None
    init_process_group(run)
    archs = {"gpt2": (GPT2LMHeadModel, "GPT2Block"), "llama": (LlamaForCausalLM, "LlamaDecoderLayer")}
    model_class, block = archs[args.arch]
    reference = model_class.from_pretrained(args.checkpoint)
    base_logits = None
    if args.lora:
        with torch.no_grad():
            base_logits = reference(torch.arange(12).reshape(2, 6)).logits
        reference = with_adapters(reference)
    with torch.device("meta") if args.loader == "rank0" else params_on_meta():
        holder = Holder(model_class._from_config(reference.config).eval())  # `from_pretrained` returns eval mode
    if args.lora:
        holder.model = with_adapters(holder.model)
    if args.loader == "dcp":
        object.__setattr__(holder, META_INIT_CHECKPOINT, args.checkpoint)
    elif run.rank == 0:
        source = model_class.from_pretrained(args.checkpoint)
        if args.drop:
            delattr(source.transformer, args.drop)
        object.__setattr__(holder, META_INIT_SOURCE, with_adapters(source) if args.lora else source)
    record: dict = {}
    try:
        place_module(holder, run, {f"class:{block}": {}})
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
        record["tied"] = model.lm_head.weight is model.get_input_embeddings().weight
        record["source_dropped"] = not hasattr(holder, META_INIT_SOURCE) and not hasattr(holder, META_INIT_CHECKPOINT)
        record["trainable"] = sorted({n.split(".")[-2] for n, p in holder.named_parameters() if p.requires_grad})
        if base_logits is not None:  # how far the adapters move the output, so a dropped adapter cannot pass
            record["adapter_effect"] = float((expected - base_logits).abs().max())
    Path(args.out_dir, f"rank{os.environ['RANK']}.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()
