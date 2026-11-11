# Tuning: core and HF

[Tuning](tuning.md) describes how every composition declares and attaches its Fine-Tuning Scheduler view.

The core (HF) composition, and any composition that does not override `build_tunable_view()`, takes venngram's
composed view of the Interpretune module: it is trainable, its physical names are the module's own (`model.<...>`), and
wherever a component map describes the model, a schedule can also address it by layer and component. PEFT adapter
tensors take the `adapter:<name>` role from venngram's overlay maps (see
[declaring a view](tuning.md#declaring-a-view)); every other parameter is `base` unless `tunable_roles()`
says otherwise.

Concrete example usage for this adapter is to come; it is tracked in [interpretune#812](https://github.com/speediedan/interpretune/issues/812).
