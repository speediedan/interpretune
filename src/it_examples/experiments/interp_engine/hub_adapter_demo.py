"""Demonstrate the hub-delivered interp-engine adapter on gemma-3-1b-it.

Run:
    python -m it_examples.experiments.interp_engine.hub_adapter_demo

What it shows, in order:

1. **A third-party adapter arriving through the hub.** The adapter is not bundled in interpretune. It
   is pulled as a component, gated by the trust opt-in, and registers ``Adapter.interp_engine`` into
   ``CompositionRegistry`` on load. Nothing in interpretune knows about interp-engine at build time.
2. **Capture at the tensor a transcoder is actually trained on**, through the adapter's seam.
3. **The hook-name hazard, made concrete.** Asking for the same thing under TransformerLens' legacy
   block-level name is REFUSED with an explanation rather than silently answered with a neighbouring
   tensor.

Point 3 is the one worth the reader's attention. Gemma Scope 2 transcoders declare
``pre_feedforward_layernorm.output`` as their input; SAELens declares the TransformerLens name
``blocks.{i}.hook_mlp_in`` for the same artifacts; and TransformerLens fires that hook on the residual
stream BEFORE the norm. On this model at layer 5 the two tensors share a cosine similarity of 0.088,
which is enough to make a dashboard read as ordinary while encoding activations its transcoder was
never trained on.

CLI rather than a notebook, deliberately: interp-engine's sync free functions refuse inside a running
event loop and Jupyter has one. The async surface works in a notebook; this script uses the sync one
because a script has no loop, which keeps the demonstration free of asyncio noise.
"""

from __future__ import annotations

import argparse
import os

DEFAULT_COMPONENT = "speediedan/it-interp-engine-adapter"
DEFAULT_MODEL = "google/gemma-3-1b-it"
DEFAULT_LAYER = 5
DEFAULT_TEXT = "The capital of France is Paris, and the capital of Germany is Berlin."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--component", default=DEFAULT_COMPONENT, help="hub component repo id")
    parser.add_argument("--revision", default=None, help="pin the component revision (recommended)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import interpretune as it
    from interpretune.hub.adapters import load_hub_adapter, loaded_adapter_module
    from interpretune.protocol import Adapter

    # 1. The component is remote code. Opting in is a deliberate act, and the gate refuses by default
    #    rather than trusting silently -- so this line is the demonstration, not boilerplate.
    os.environ.setdefault("IT_TRUST_REMOTE_CODE", "1")

    print(f"pulling {args.component!r}" + (f" @ {args.revision}" if args.revision else " @ refs/main"))
    it.hub.pull(args.component, revision=args.revision)

    before = set(Adapter.__members__)
    members = load_hub_adapter(args.component)
    added = sorted(set(Adapter.__members__) - before)
    print(f"registered adapters: {[m.name for m in members]}   (new Adapter members: {added})")

    # 2. Wrap a model interpretune already holds. EagerModel takes it in place: no reload, no copy.
    print(f"loading {args.model} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    from interp_engine import EagerModel, run_with_cache  # provided by the component's requires: block

    engine_model = EagerModel(args.model, hf_model=hf_model, tokenizer=tokenizer, dtype="float32")
    tokens = engine_model.to_tokens(args.text)

    # 3. The seam, reached the way an op would reach it.
    seam = loaded_adapter_module(args.component)._Seam

    correct = f"blocks.{args.layer}.mlp.hook_in"
    legacy = f"blocks.{args.layer}.hook_mlp_in"

    point = seam.point_for(correct, engine_model)
    captured = run_with_cache(engine_model, tokens, [point])[point]
    print(f"\ncaptured {correct} -> {point}   shape {tuple(captured.shape)}")

    # Ground truth straight from the module the artifact's own config names.
    grabbed = {}
    block = hf_model.model.layers[args.layer]
    handle = block.pre_feedforward_layernorm.register_forward_hook(
        lambda m, i, o: grabbed.__setitem__("declared", o.detach())
    )
    with torch.no_grad():
        hf_model(**tokenizer(args.text, return_tensors="pt"))
    handle.remove()

    cos = F.cosine_similarity(captured.flatten().float(), grabbed["declared"].flatten().float(), dim=0)
    print(f"agreement with the transcoder's declared input tensor: cos={cos.item():.6f}")

    # 4. The refusal. This is the part that matters for anyone publishing SAE artifacts.
    print(f"\nasking for {legacy!r}, the legacy TransformerLens name for 'the MLP's input':")
    try:
        seam.point_for(legacy, engine_model)
        print("  ... resolved. (Unexpected: this adapter is supposed to refuse.)")
        return 1
    except Exception as exc:  # UnmappableHookError, raised from the component
        print(f"  refused, as intended:\n    {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
