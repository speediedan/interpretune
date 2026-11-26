# Oracle lens for Qwen3.6-27B

WorkspaceBench's oracle lens is a metamodel lens: a LoRA on `Qwen/Qwen3.6-27B` that takes one of the base model's
activations, injected into its own prompt, and writes what the activation encodes as `- ` bullets. A second adapter,
the reconstructor (AR), maps a phrase back to a direction at a layer, so a readout's bullets can be fit to the
activation they came from. The checkpoints are released as `agu18dec/olens_and_ar` (Apache-2.0); this component reads
them through interpretune's op dispatch, from the session's own model.

## What is in it

| file | what it is |
| --- | --- |
| `oracle_lens_runtime.py` | the runtime: the verbalizer prompt and its injection marker, the injection transform, the capture of the base model's activation, the sampled read, the reconstructor and the non-negative fit |
| `oracle_lens_ops.py`, `oracle_lens_ops.yaml` | the `oracle_lens_read` op (alias `oracle_lens`) |

## The contract the checkpoints fix

A call that asks for anything else is refused.

- **Base:** `Qwen/Qwen3.6-27B`. The adapters are mounted on the session's own model with
  `interpretune.utils.peft_mount`, which refuses a session on another base, a session that already runs an adapter
  of its own, and an adapter whose tensors do not all land on the model. No second copy of the base is loaded.
- **Layers:** 20 to 60 in steps of 4, the trained band.
- **Prompt and transform per adapter:** `space` (the default), `rl600` and `student` read the `concepts_raw` prompt
  with the activation injected as $16000 \cdot h / \lVert h \rVert$; `inverter` reads `continuation_raw` with
  $122.32 \cdot h$.
- **Injection marker:** the first character of U+3200 to U+33FF that is a single token appearing once in the fully
  rendered prompt, found on the tokenizer rather than hardcoded.
- **Activation:** the base model's block output at the read layer, captured with every adapter off. The lens is on only
  for the read, and the session's model is left as it was.
- **Sampling is part of the result:** the adapter, the checkpoint revision, the seed, the temperature, top-$p$ and the
  token budget are returned with every readout (`oracle_lens_sampling`).

## Use

Put the component directory on `IT_ANALYSIS_OP_PATHS` before interpretune is imported, and call the op with a
`module` whose model is `Qwen/Qwen3.6-27B` (a core or nnsight composition):

```python
import os

os.environ["IT_ANALYSIS_OP_PATHS"] = "<this directory>"  # before interpretune is imported
import torch
import interpretune as it
from interpretune.analysis.ops.base import AnalysisBatch

ids = tokenizer("Roses are red, the sky is", add_special_tokens=False)["input_ids"]
batch = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
out = it.oracle_lens_read(
    module, AnalysisBatch(), batch, 0,
    oracle_lens_layers=[44, 52], oracle_lens_samples=4, oracle_lens_ar=True,
)
out.oracle_lens_bullets  # [prompt][layer][sample] list of phrases
out.oracle_lens_fve      # [prompt][layer][sample] fraction of the whitened activation the bullets explain
```

`peft` must be installed. The released checkpoint (about 7.5 GB) is fetched from the Hub at a pinned revision on first
use, or read from `oracle_lens_root`.

## Evidence

The runtime is ported from the research code of step 1 of
[interpretune#1202](https://github.com/interpretune/interpretune/issues/1202), which read the lens on the poetry
planning items of WorkspaceBench with the `space` adapter: the committed rhyme word appears in 0.244 of readouts
(0.053 for another item's vector and 0.020 for a rhyme the model did not write), and in 0.440 at layer 52.
The reconstructor reproduces its published per-layer fraction explained on the model's own chat replies (0.135 to
0.358 against the card's 0.120 to 0.341), with the head's LayerNorm non-affine and the hidden state taken after the
final norm, which the checkpoint does not state and step 1 measured.

The op was then run on the real weights (bfloat16 base on two 32 GB GPUs) over 12 of those items, 3 positions and layers
20, 44, 52 and 60, against step 1's `PeftModel` path, each capturing its own activations. All 144 greedy readouts are
the same text (0 of 144 for a shifted pairing), the base's own forward is bit-identical before the mount and after the
last read, and an adapter for another base is refused. In a sampled pass the rhyme word appears in 39 of 144 readouts
(step 1's samples at the same cells: 36), and the reconstructor explains 0.160, 0.356, 0.263 and 0.301 of the activation
at the four layers (step 1: 0.152, 0.346, 0.269 and 0.303).

## Limits

- Tested on one task family (poetry planning); how the lens reads other families is not measured.
- The CPU tests run on a tiny random model of the same architecture: they cover the mechanics and the refusals, not
  read quality, which needs the real weights (two 32 GB GPUs or more).
- The adapters are held in float32, as they are released. Mounted at the base's bfloat16 instead, 4 of 28 greedy
  readouts matched the reference; float32 takes about twice as long per read.
- Compositions: an HF model or an nnsight wrapper of one. A TransformerLens bridge is refused.
