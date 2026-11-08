"""One rank of the circuit-tracer placement runs (``tests/core/test_core_loop_distributed.py``).

Every rank builds the same tiny circuit-tracer replacement model (``tests/core/circuit_tracer_toy.py``) and records its
logits, an nnsight trace of its LM head, its transcoder activations and each transcoder's sparse encode and decode (the
calls attribution sets up with). A module holding the model's Hugging Face
module declares the calls circuit-tracer makes outside its forward, as the nnsight and circuit-tracer adapters do, and
``place_module`` shards it with the given plan. The same readings are then taken through the sharded model.

With ``--meta``, the model is instead built on the meta device by ``meta_replacement_model`` (its Hub loaders replaced
by the toy's), rank 0 keeps the toy as the source the placement loads every rank from, and the readings are compared
with the toy's.

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

    def meta_load_target(self) -> nn.Module:
        return self.hf_model


def widen_scalars(model) -> None:
    """FSDP2 cannot shard a 0-d parameter, and the toy's JumpReLU threshold is one; trained transcoders carry a
    per-feature threshold, so widen it to one dimension (the same function, by broadcasting)."""
    for module in model._model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if param.dim() == 0:
                setattr(module, name, nn.Parameter(param.detach().reshape(1).clone()))


def toy_meta_build(rank0: bool):
    """``meta_replacement_model`` over the toy: its config, transcoders (built with a 1-D threshold) and, on rank
    0, the toy itself as the loaded model."""
    import circuit_tracer.utils.hf_utils as hf_utils
    import transformers
    from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
    from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
    from circuit_tracer.transcoder.activation_functions import JumpReLU

    from interpretune.adapters.circuit_tracer.adapter import meta_replacement_model
    from tests.core.circuit_tracer_toy import tiny_gemma2_replacement_model

    toy = tiny_gemma2_replacement_model()
    widen_scalars(toy)

    def transcoders(ref, device, dtype, lazy_encoder, lazy_decoder):
        assert not (lazy_encoder or lazy_decoder)
        layers = {layer: SingleLayerTranscoder(8, 32, JumpReLU(torch.zeros(1), 0.1), layer) for layer in range(3)}
        return TranscoderSet(layers, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out"), {}

    auto_config = transformers.AutoConfig.from_pretrained
    hf_utils.load_transcoder_from_hub = transcoders
    transformers.AutoConfig.from_pretrained = (
        lambda name, **kwargs: toy.config if name == "toy" else auto_config(name, **kwargs)
    )
    NNSightReplacementModel.from_pretrained_and_transcoders = classmethod(lambda cls, *args, **kwargs: toy)
    model, source = meta_replacement_model("toy", "toy", torch.float32, load_source=rank0)
    assert all(p.is_meta for p in model._model.parameters())
    return model, source


def readings(model, prompt: torch.Tensor) -> dict[str, torch.Tensor]:
    from tests.core.circuit_tracer_toy import only_token_zero_special

    with torch.no_grad():
        plain = model._model(prompt[None]).logits.clone()
        with model.trace(prompt[None]):
            traced = nnsight.save(model.lm_head.output)
    with only_token_zero_special(model):
        _, activations = model.get_activations(prompt)
    # attribution's setup path: `decode_sparse` calls `_get_decoder_vectors`, both registered on one unit, so this
    # reading fails if the inner call's return reshards the unit before the outer reads `b_dec`
    transcoders = model.transcoders._module.transcoders
    mlp_in = torch.randn(prompt.shape[0], transcoders[0].W_enc.shape[1], generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        sparse = torch.stack([tc.decode_sparse(tc.encode_sparse(mlp_in)[0], mlp_in)[0] for tc in transcoders])
    return {"plain": plain, "trace": traced.clone(), "activations": activations.clone(), "sparse": sparse.clone()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="the FSDP2 plan, as JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--meta", action="store_true", help="build on the meta device and load from rank 0")
    args = parser.parse_args()

    from interpretune.runners.distributed import META_INIT_SOURCE, init_process_group, place_module, resolve_placement
    from interpretune.utils import MisconfigurationException
    from tests.core.circuit_tracer_toy import TOY_PROMPT, tiny_gemma2_replacement_model

    run = resolve_placement("fsdp2")
    assert run is not None
    init_process_group(run)
    reference = tiny_gemma2_replacement_model()
    widen_scalars(reference)
    expected = readings(reference, TOY_PROMPT)
    if args.meta:
        model, source = toy_meta_build(rank0=run.rank == 0)
        holder = Holder(model)
        if source is not None:
            object.__setattr__(holder, META_INIT_SOURCE, source)
    else:
        model = reference
        holder = Holder(model)
    record: dict = {"active": int((expected["activations"] != 0).sum())}
    try:
        place_module(holder, run, json.loads(args.plan))
    except MisconfigurationException as err:
        record["refused"] = str(err)
    else:
        placed = readings(model, TOY_PROMPT)
        record.update({f"{k}_diff": float((placed[k] - expected[k]).abs().max()) for k in expected})
    Path(args.out_dir, f"rank{os.environ['RANK']}.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()
