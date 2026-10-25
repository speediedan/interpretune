# Tuning: nnsight

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

A local, dispatched nnsight composition registers the nnsight model's HF module on the interpretune module as `hf_model`
(in `register_tunable_modules()`), so the optimizer, checkpoints and the identity view reach its parameters. A composition
executing remotely (NDIF) has no local weights to tune, and one built with `dispatch=False` keeps its weights on the
meta device; both are refused, each with its reason.

The paths to fuller nnsight tuning, local and remote, are tracked in [interpretune#814](https://github.com/speediedan/interpretune/issues/814). Locally, that covers tuning applied
through nnsight's own tracing, and the remaining local cases. Remotely, it covers work that does not need local weights:
remote models accept persistent changes to values, and ephemeral techniques that tune within a single dispatch may need
no client-side optimizer state at all. The detailed remote design is [interpretune#775](https://github.com/speediedan/interpretune/issues/775).

Concrete example usage for this adapter is to come; it is tracked in [interpretune#812](https://github.com/speediedan/interpretune/issues/812).
