# Tuning: nnsight

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its Fine-Tuning Scheduler view.

A local, dispatched nnsight composition registers the nnsight model's HF module on the Interpretune module as `hf_model`
(in `register_tunable_modules()`), so the optimizer, checkpoints and the default view, venngram's composed view mounted
at `hf_model`, reach its parameters. nnsight builds the model through its own constructor, so a `lora_cfg` or
`bitsandbytesconfig` in `hf_from_pretrained_cfg` is not applied
([interpretune#942](https://github.com/speediedan/interpretune/issues/942)). A composition
executing remotely (NDIF) has no local weights to tune, and one built with `dispatch=False` keeps its weights on the
meta device; both are refused, each with its reason.

The paths to fuller nnsight tuning, local and remote, are tracked in [interpretune#814](https://github.com/speediedan/interpretune/issues/814). Locally, that covers tuning applied
through nnsight's own tracing, and the remaining local cases. Remotely, it covers work that does not need local weights:
remote models accept persistent changes to values, and ephemeral techniques that tune within a single dispatch may need
no client-side optimizer state at all. The detailed remote design is [interpretune#775](https://github.com/speediedan/interpretune/issues/775).

Concrete example usage for this adapter is to come; it is tracked in [interpretune#812](https://github.com/speediedan/interpretune/issues/812).
