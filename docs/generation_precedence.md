# Generation flags precedence and semantics

This document describes how Interpretune resolves generation settings between model-level configuration and per-call `generate` kwargs.

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

Remember: per-call kwargs in `generate_kwargs` take precedence and are recommended for limited-scope behaviors and debug flows.


### Debugging generation outputs with Interpretune's DebugGeneration extension

 For TransformerLens legacy `HookedTransformer` models, the `generate` method does not accept HF `return_dict_in_generate` semantics by default and often returns bare tensors. Interpretune's debug helpers will wrap tensor outputs into a ModelOutput-like object exposing `sequences` when those are explicitly requested or when DebugGeneration's default output attributes include `sequences`.
 Additionally, the `DebugGeneration` extension normalizes raw tensor outputs into a `ModelOutput` dataclass when appropriate. This normalization occurs when the caller did not specify a `gen_output_attr` (so defaults are applied) or when the `gen_output_attr` itself requests one of the attributes listed in `DebugGeneration.DEFAULT_OUTPUT_ATTRS` (i.e., `sequences`). This preserves behavior for debug utilities that expect `.sequences` while avoiding implicit HF flag injection.
