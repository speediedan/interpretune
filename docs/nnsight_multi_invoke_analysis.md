# NNsight Multi-Invoke Analysis for Ablation Operations

## Summary

Empirical investigation confirming that NNsight's **multi-invoke single-trace** pattern
can replace the current separate-traces-per-ablation-variant approach in
`NNsightModelBackend.fwd_w_hooks_batched`.  Multi-invoke correctly scopes
SAE interventions per invoke and delivers identical results to separate traces,
with a measurable performance improvement (~1.5× on GPT-2-small, expected to
scale better on larger models with higher forward-pass cost).

**Date:** 2026-02-28
**NNsight version:** 0.6.1.dev0+g9340e1259.d20260227
**Test model:** GPT-2-small (openai-community/gpt2) on CPU
**SAE:** gpt2-small-res-jb / blocks.9.attn.hook_z

---

## Background

The ablation analysis op (`definitions.py`) calls `fwd_w_hooks_batched` with one
`hook_config` per alive SAE latent.  Each config applies a different ablation
(zeroing a specific latent) before decoding.  The original implementation ran
each config in a **separate** `model.trace()` context — one physical forward pass
per ablation variant — because the docstring stated *"NNsight's multi-invoke
mechanism … does not scope interventions per invoke"*.

This investigation empirically tested whether that claim was correct, and found
it was **not** — NNsight's Batcher mechanism (narrow / swap) correctly scopes
interventions to each invoke's batch slice.

---

## Key Findings

### 1. `model.lm_head.output` is narrowed per invoke; `model.output.logits` is NOT

| Access pattern           | Single invoke | Multi-invoke (N=3, bs=2) | Narrowed? |
|--------------------------|---------------|--------------------------|-----------|
| `model.lm_head.output`  | [2, 5, 50257] | [2, 5, 50257] per invoke | **Yes**   |
| `model.output.logits`   | [2, 5, 50257] | [4, 5, 50257] per invoke | **No**    |

- `model.lm_head.output` accesses a standard `nn.Linear` envoy → NNsight hooks
  fire and the Batcher narrows the output to the current invoke's slice.
- `model.output.logits` accesses the top-level model's `CausalLMOutputWithCrossAttentions`
  — the `.logits` attribute access happens **after** envoy processing and bypasses
  narrowing.

**Implication:** Multi-invoke code must read logits via `model.lm_head.output`
(or an equivalent envoy-wrapped module), **not** `model.output.logits`.

### 2. Multi-invoke with different SAE interventions matches separate traces exactly

Three SAE intervention variants were tested in a single trace with three input
invokes, and their results compared against separate-trace ground truth:

| Variant       | Separate trace diff | Multi-invoke diff | Match? |
|---------------|--------------------:|------------------:|--------|
| passthrough   | —                   | —                 | ✓      |
| zero_all      | 0.346043            | 0.346043          | ✓      |
| scale_100x    | 99.446716           | 99.446716         | ✓      |

All three multi-invoke results were `torch.allclose(..., atol=1e-5)` identical
to the corresponding separate-trace results.

### 3. Input invokes are isolated; empty invokes are cumulative

- **Input invokes** (`tracer.invoke(**batch)`): Each gets its own batch slice via
  `Batcher.narrow()`.  Modifications via `Batcher.swap()` affect only that slice.
  **Suitable for independent ablation variants.**

- **Empty invokes** (`tracer.invoke()`): Operate on the full stacked batch.
  Modifications from one empty invoke persist to the next.
  **NOT suitable for independent ablation variants.**

### 4. Simple (non-SAE) interventions also scope correctly

Tested zeroing `model.transformer.h[5].output[0][:]` in one invoke while
leaving another invoke clean.  Multi-invoke diff (178.353) exactly matched
separate-trace diff.

### 5. Performance

| Approach       | Time (GPT-2-small, 3 configs) |
|----------------|------------------------------:|
| Separate traces | 344.6 ms                     |
| Multi-invoke    | 230.4 ms                     |
| **Speedup**     | **1.50×**                    |

The speedup comes from:
- One trace setup (AST parse, compilation, thread creation) instead of N.
- One batched forward pass instead of N sequential passes.

For larger models and more ablation variants, the speedup is expected to be
significantly greater since the forward-pass batching dominates.

---

## Mechanism

How NNsight scopes interventions per invoke (from NNsight 0.6 source analysis):

1. **InterleaveTracer** creates one **Mediator** (worker thread) per invoke.
2. When the user accesses an envoy's `.output` or `.input`, the mediator waits
   for the forward hook to fire on that module.
3. The hook fires once for the **entire batch**. Inside `Interleaver.handle()`,
   each mediator processes the hook output sequentially.
4. `Batcher.narrow()` returns only the current invoke's slice
   (`output[start:start+batch_size]`).
5. `Batcher.swap()` writes modifications back to only that slice in the
   original output tensor.
6. After all mediators have processed, the (potentially modified) full batch
   continues through the model.

This means each invoke's intervention code operates on its own slice of the
batch, and modifications are isolated to that slice.

---

## Implementation Impact

### Refactored `fwd_w_hooks_batched`

Changed from N separate traces to one trace with N invokes:

```python
# Before (one trace per config):
for fwd_hooks in hook_configs:
    with model.trace() as tracer:
        with tracer.invoke(**batch):
            # apply hooks + SAE splice
            saved = nnsight.save(model.output.logits)  # full batch (OK for 1 invoke)
    results.append(saved)

# After (one trace, N invokes):
with model.trace() as tracer:
    for fwd_hooks in chunk:
        with tracer.invoke(**batch):
            # apply hooks + SAE splice
            chunk_results.append(nnsight.save(model.lm_head.output))  # narrowed per invoke
```

### `max_invokes_per_trace` chunking

The `max_invokes_per_trace` parameter (already in the protocol) now controls
how many invokes are batched per trace to avoid OOM with large alive-latent
counts. When `None`, all configs are batched in one trace.

### `model.lm_head.output` vs `model.output.logits`

- **Multi-invoke** (`fwd_w_hooks_batched`): uses `model.lm_head.output`
  (narrowed per invoke).
- **Single-invoke** methods: retain `model.output.logits` (narrowing is not
  relevant for single-invoke contexts).

---

## Debug Scripts

- [debug_nnsight_multi_invoke_ablation.py](../debug/debug_nnsight_multi_invoke_ablation.py) — v1 (identified narrowing issues)
- [debug_nnsight_multi_invoke_ablation_v2.py](../debug/debug_nnsight_multi_invoke_ablation_v2.py) — v2 (comprehensive validation)

---

## Open Questions / Future Work

1. **Memory pressure:** With many alive latents (hundreds+), multi-invoke batches
   all inputs in one forward pass (N × batch_size).  `max_invokes_per_trace`
   chunking mitigates this, but the optimal chunk size depends on model size and
   available memory.  An adaptive strategy could be explored.

2. **`model.output.logits` narrowing:** The top-level model output (`CausalLMOutput`)
   is not narrowed per invoke because attribute access on non-module objects bypasses
   envoy hooks.  If NNsight adds support for narrowing structured outputs, the
   distinction between `model.lm_head.output` and `model.output.logits` would
   become unnecessary.

3. **Empty invoke patterns:** Empty invokes could be useful for batch-wide
   observation after per-invoke interventions, but their cumulative nature makes
   them unsuitable for independent ablation variants.
