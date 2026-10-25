# Tuning: core and HF

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

The core (HF) composition, and any composition that does not override `build_tunable_view()`, takes
finetuning-scheduler's identity view over the interpretune module: it is trainable, and its physical names are the
module's own (`model.<...>`). PEFT adapter tensors take the `adapter:<name>` role, found through PEFT's tuner-layer API
(see [declaring a view](fts_tunable_views.md#declaring-a-view)); every other parameter is `base` unless
`tunable_roles()` says otherwise.

Concrete example usage for this adapter is to come; it is tracked in [interpretune#812](https://github.com/speediedan/interpretune/issues/812).
