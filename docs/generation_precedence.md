# Generation flags precedence and semantics

This document describes how Interpretune resolves generation behavior between model-level configuration, configured per-call generation kwargs, and debug-time overrides.

## Precedence

Interpretune currently resolves generation behavior in this order:

1. Model-level defaults stored on `model.generation_config`
2. Configured per-call kwargs in `generative_step_cfg.lm_generation_cfg.generate_kwargs`
3. Debug-time overrides passed into `DebugGeneration._debug_generate(...)`

The important implementation detail is that Interpretune does not try to synthesize HuggingFace generation flags implicitly.

- `generate_kwargs` are passed directly to `it_generate(...)`
- `gen_kwargs_override` mutates that per-call kwargs dict for the debug invocation
- `gen_config_override` writes directly onto `model.generation_config` when the model exposes one

That means configured per-call kwargs remain the preferred place for request-scoped behavior such as `output_logits` or `return_dict_in_generate`, while model-level `generation_config` remains useful for broad defaults.

## Examples

One can set Model-level defaults (which will apply to all generate calls unless overridden):

```python
# Set HF generation defaults in the model's generation_config
model.generation_config.return_dict_in_generate = True
model.generation_config.output_logits = True
```

Or use per-call override (preferred for ad-hoc behavior) to pass direct generation kwargs:

```yaml
lm_generation_cfg:
  generate_kwargs:
    output_logits: true
    return_dict_in_generate: true
```

Remember: per-call kwargs in `generate_kwargs` take precedence over model defaults for the actual `generate(...)` call and are recommended for limited-scope behaviors and debug flows.

## DebugGeneration behavior

`DebugGeneration` adds one more layer: output normalization for debugging utilities.

- It reads `generate_kwargs` from `ph.it_cfg.generative_step_cfg.lm_generation_cfg.generate_kwargs`
- It applies `gen_kwargs_override` on top of those kwargs
- It applies `gen_config_override` directly to `model.generation_config` when available
- It then calls `ph.it_generate(inputs, **gen_kwargs)`

## Output normalization

### Debugging generation outputs with Interpretune's DebugGeneration extension

For legacy TransformerLens `HookedTransformer` generation, raw outputs may still be bare tensors rather than HuggingFace-style `ModelOutput` objects.

Interpretune handles that in `DebugGeneration._normalize_output_to_model_output(...)`.

- If the returned object already exposes one of the requested output attributes, it is normalized into a `ModelOutput` wrapper only when needed.
- If the output is a plain tensor, it is left as a tensor unless debug consumers requested an attribute path that requires normalization.
- This keeps debug helpers compatible with `.sequences`-style expectations without injecting HuggingFace generation semantics into the runtime call path.

In practice, this means the debug layer is responsible for making heterogeneous backend outputs easier to inspect, while the generation call itself stays explicit and backend-driven.

## Backend-specific generation notes

### TransformerBridge (ITLensBridgeConfig)

TransformerBridge delegates generation to the wrapped HuggingFace model, so standard HF generation kwargs (`output_logits`, `return_dict_in_generate`, etc.) work natively. Use `ITLensBridgeConfig` as `tl_cfg` in `SAELensConfig` for the Bridge path; using `ITLensFromPretrainedConfig` with `use_bridge=True` will emit a misconfiguration warning.

### HookedTransformer (ITLensFromPretrainedConfig)

HookedTransformer uses TransformerLens's own `generate()` implementation with `TLensGenerationConfig` fields. Some HF generation flags may not be supported or may behave differently. Set `use_bridge=False` explicitly when targeting HookedTransformer.
