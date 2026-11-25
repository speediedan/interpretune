"""One rank of the meta-device placement runs (``tests/core/test_core_loop_distributed.py``).

Every rank builds a small model (GPT-2, or Llama with ``--arch llama``) from the checkpoint's config on the meta
device, as ``HFFromPretrainedConfig.meta_init`` does. With ``--loader rank0``, rank 0 also loads the checkpoint on the
CPU and ``place_module`` loads the sharded model from it; with ``--drop ln_f``, rank 0's copy lacks the final norm,
which must be refused by name. With ``--latent``, a latent model built with its values is spliced over the first
block's MLP, beside the model the checkpoint holds: each rank builds it with its own values, and rank 0's must be every
rank's after placement. With ``--loader dcp``, only the parameters are built on meta (buffers keep their values)
and every rank reads its own slices of the checkpoint's safetensors files. With ``--placement tp``, the Llama is
split by the plan transformers declares for it instead of sharded (its head and embedding stay whole).

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

    def default_tp_plan(self, world_size: int) -> dict:
        from interpretune.runners.distributed import hf_tp_plan

        return hf_tp_plan(self.model, "model", world_size)

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


class Latent(nn.Module):
    """A small latent model over an MLP's output, seeded, so a wiped or wrongly placed value changes the logits."""

    def __init__(self, width: int, seed: int) -> None:
        super().__init__()
        self.enc, self.dec = nn.Linear(width, 5), nn.Linear(5, width)  # 5 latents: an uneven split on two ranks
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for param in self.parameters():
                param.normal_(0.0, 0.5, generator=generator)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.enc(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def splice(root: nn.Module, seed: int) -> None:
    """Attach the latent model where an interpretune module attaches one: beside its model."""
    from interpretune.tuning.latents import LatentSite, attach_latent_model

    llama = root.model.config.model_type == "llama"
    path = "model.model.layers.0.mlp" if llama else "model.transformer.h.0.mlp"
    latent = Latent(root.model.config.hidden_size, seed)
    attach_latent_model(root, "probe", latent, LatentSite(path, op="replace"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--drop", default=None, help="a submodule to delete from rank 0's copy (a planted defect)")
    parser.add_argument("--loader", default="rank0", choices=("rank0", "dcp"))
    parser.add_argument("--arch", default="gpt2", choices=("gpt2", "llama", "mixtral"))
    parser.add_argument("--reference", default=None, help="a hub-layout checkpoint of the model, to compare with")
    parser.add_argument("--lora", action="store_true", help="LoRA on the model and on rank 0's copy (GPT-2, rank0)")
    parser.add_argument("--latent", action="store_true", help="a latent model with values beside the meta-built model")
    parser.add_argument("--latent-unit", action="store_true", help="make the attached latent model its own FSDP2 unit")
    parser.add_argument("--placement", default="fsdp2", choices=("fsdp2", "tp"))
    parser.add_argument("--train-step", action="store_true", help="one SGD step after loading, against one process")
    args = parser.parse_args()

    from transformers import GPT2LMHeadModel, LlamaForCausalLM, MixtralForCausalLM

    from interpretune.runners.distributed import (
        META_INIT_CHECKPOINT,
        META_INIT_SOURCE,
        init_process_group,
        params_on_meta,
        place_module,
        resolve_placement,
    )
    from interpretune.utils import MisconfigurationException

    run = resolve_placement(args.placement)
    assert run is not None
    init_process_group(run)
    archs = {
        "gpt2": (GPT2LMHeadModel, "GPT2Block"),
        "llama": (LlamaForCausalLM, "LlamaDecoderLayer"),
        "mixtral": (MixtralForCausalLM, "MixtralDecoderLayer"),
    }
    model_class, block = archs[args.arch]
    reference = model_class.from_pretrained(args.reference or args.checkpoint)
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
    expected_holder = Holder(reference)
    if args.latent:
        with torch.no_grad():
            base_logits = reference(torch.arange(12).reshape(2, 6)).logits
        splice(holder, seed=run.rank)  # each rank's own values, of which rank 0's are placed (in the root unit)
        splice(expected_holder, seed=0)
    plan = {f"class:{block}": {}, **({"class:Latent": {}} if args.latent_unit else {})}
    try:
        place_module(holder, run, plan if args.placement == "fsdp2" else None)
    except MisconfigurationException as err:
        record["refused"] = str(err)
    else:
        # read before the forward: FSDP2 keeps the root unit's parameters unsharded after a forward until backward
        record["param_types"] = sorted({type(p).__name__ for p in holder.parameters()})
        # `nn.Parameter(DTensor)` is a DTensor; whole parameters under `tp` are plain ones
        record["split_params"] = sorted(n for n, p in holder.named_parameters() if type(p.data).__name__ == "DTensor")
        record["held_numel"] = sum(getattr(p.data, "_local_tensor", p.data).numel() for p in holder.parameters())
        record["whole_numel"] = sum(p.numel() for p in holder.parameters())
        input_ids = torch.arange(12).reshape(2, 6)
        with torch.no_grad():
            placed = holder(input_ids)
            expected = reference(input_ids).logits
        model = holder.model
        record["max_abs_diff"] = float((placed - expected).abs().max())
        record["tied"] = model.lm_head.weight is model.get_input_embeddings().weight
        record["source_dropped"] = not hasattr(holder, META_INIT_SOURCE) and not hasattr(holder, META_INIT_CHECKPOINT)
        record["trainable"] = sorted({n.split(".")[-2] for n, p in holder.named_parameters() if p.requires_grad})
        if base_logits is not None:  # how far the adapters or the latent model move the output, so neither can drop
            record["adapter_effect"] = float((expected - base_logits).abs().max())
        if args.train_step:
            # one SGD step on the same batch on every rank: FSDP2 averages identical gradients, so the sharded step
            # (frozen base and trainable adapters in the same units) must match one process's
            stepped = {}
            for name, net in (("placed", holder), ("reference", expected_holder)):
                optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=1.0)
                logits = net(input_ids)
                loss = nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), input_ids[:, 1:].flatten())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    stepped[name] = net(input_ids)
            record["step_diff"] = float((stepped["placed"] - stepped["reference"]).abs().max())
            record["step_effect"] = float((stepped["reference"] - expected).abs().max())
    Path(args.out_dir, f"rank{os.environ['RANK']}.json").write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    main()
