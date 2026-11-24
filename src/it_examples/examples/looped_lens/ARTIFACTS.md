# Looped lens artifacts

None of the shipped remote-capable books needs a trained artifact: their readouts are decoded with the model's own
head. Two artifacts exist; neither is published yet, and both are described here so a published layout can follow.

## The checkpoint head (remote decoding)

A remote read returns states from NDIF and decodes them locally with the checkpoint's final norm gain and unembedding.
They are two tensors of the public Apache-2.0 checkpoint (`thinkingmachines/Inkling-2-Mini`), so the head is derived,
not trained:

| key | shape | dtype |
| --- | --- | --- |
| `model.llm.norm.weight` | (2304,) | bf16 |
| `model.llm.unembed.weight` | (201024, 2304); rows past 200,058 are padding | bf16 |

About 926 MB as safetensors. A published component should derive it at load (one shard of the checkpoint) rather than
ship a copy.

## The M2 translators (`translator:final`, `translator:near`)

Per core visit $(i, p)$ (weight layer $i$ in 6 to 17, pass $p$ in 1 to 3), an affine map of the active block $h$ read
through the frozen head:

$$
T_{i,p}(h) = h + W_i h + b_i + c_{i,p}
$$

$W_i$ and $b_i$ are shared by a layer's three passes; $c_{i,p}$ is a per-pass offset. Two sets, trained on 2,000
pile-10k windows of 512 tokens (64 positions each): `final` minimises the KL divergence to the model's final
distribution, `near` to the coda lens from the end of the same pass.

`translators.pt` (510 MB, `torch.save`):

```text
{
  "final": {"W": float32 (12, 2304, 2304), "b": float32 (12, 2304), "c": float32 (12, 3, 2304)},
  "near":  {"W": ..., "b": ..., "c": ...},
  "log":   [{"window": n, "final": KL, "near": KL}, ...]
}
```

Only lenses that read a translator need it; none of the remote-capable books does. A published copy would be one
safetensors file per set (`translators_final.safetensors` with keys `W`, `b`, `c`) plus the training record, so it can
be loaded without unpickling.
