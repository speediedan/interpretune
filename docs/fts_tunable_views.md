# Tuning

finetuning-scheduler 2.15 tunes a model through a `TunableView`: the contract between the logical parameter names a
schedule uses and the physical parameters the optimizer, checkpoints and sharding see, plus a lifecycle role for each
tensor. interpretune declares a view for each module composition and attaches it where finetuning-scheduler finds it.
Nothing here requires finetuning-scheduler: when the installed version predates views, nothing is attached.

## What is attached, and when

After model initialization (`BaseITModule._dispatch_model_init`, after `post_auto_model_init` or `model_init`), every
interpretune module carries its view as `fts_tunable_view`. finetuning-scheduler discovers it there unless a `view=` is
passed explicitly. An already-attached view is never replaced, so a hub adapter may attach its own in
`post_auto_model_init`.

**Every view's root is the interpretune module**, so physical names are the module's own (`model.<...>`), the same names
the Lightning callback sees through `pl_module`, and modules registered beside the model (spliced SAEs, interventions)
are reachable. The view's `support()` is the composition's support record: whether it can be tuned, under which naming
scheme, with which role kinds, or why not.

Which view each composition declares, and the details of the backends whose views are not finetuning-scheduler's identity
view, are on [the adapter-specific tuning page](fts_tunable_views_adapters.md).

## Declaring a view

The base `build_tunable_view()` is a declaration in its own right: it returns the identity view, so **a composition
that does not override it is trainable**. A composition declares otherwise by overriding `build_tunable_view()`,
returning its own view or `None` (which declares nothing, so the composition is refused).

Two public hooks extend a composition's view without replacing it. Both are cooperative: an override calls `super()`
and adds to its result, so a backend's view and another mixin's additions compose.

- **`tunable_roles()`** returns roles for the module's parameters and buffers, by name (anything unlisted is `base`).
  Declare an intervention, a run-fitted tensor or a latent dictionary here:

```python
class MyInterventionMixin:
    def tunable_roles(self) -> dict[str, str]:
        roles = super().tunable_roles()
        roles.update({f"steer.{n}": "intervention:steer" for n, _ in self.steer.named_parameters()})
        return roles
```

- **`register_tunable_modules()`** registers on the module any model-owned module the view must reach. It runs after
  whichever init path ran (`post_auto_model_init` or `model_init`) and before the view is attached, so overriding a
  template hook without `super()` never loses a registration.

PEFT adapter tensors are found through PEFT's tuner-layer API rather than by name, so every tuner whose layers hold their
adapters there (LoRA and its variants, IA3 and similar) is covered and the base weight PEFT renames (`base_layer.weight`)
stays `base`. Prompt-learning methods (prompt and prefix tuning) keep their tensors outside tuner layers; those are not
found and stay `base`.

The default optimizer (`optimizer_parameters()`) steps the attached trainable view's root parameters, less those whose
role finetuning-scheduler never optimizes (`fitted` and `instrument`).

## Phased fine-tuning on the core loop

Lightning compositions use finetuning-scheduler's `FinetuningScheduler` callback. On the core loop,
`interpretune.tuning.core_loop.FTSCoreCallback` drives its `FTSController` at the loop's boundaries. It goes in
`SessionRunnerCfg.callbacks`:

```python
from interpretune.config import SessionRunnerCfg
from interpretune.tuning.core_loop import FTSCoreCallback

schedule = {0: {"params": ["role:adapter:default"], "max_transition_epoch": 2}, 1: {"params": ["model.lm_head.weight"]}}
runner_cfg = SessionRunnerCfg(it_session=session, max_epochs=4, callbacks=[FTSCoreCallback(schedule)])
```

The callback builds the controller over the session's module when training starts, so the schedule names parameters
through the module's view, roles included. The controller builds the optimizer from the module's `optimizer_init` and
the lr schedulers from its `lr_scheduler_init`, unless `controller_kwargs` passes `optimizer` or `lr_scheduler`. Each is
installed as the module's own, so the loop steps the phase's optimizer and nothing is stepped twice. Further
`FTSController` arguments go in `controller_kwargs`: transition policies, a checkpoint policy, `max_depth`. Phase hooks,
gates and RNG forking are finetuning-scheduler's, unchanged. The core loop keeps its own `max_epochs`, and the callback
warns when the schedule's final phase asks for more. A `controller_factory` builds the controller instead, for anything
`controller_kwargs` cannot express; its controller must be rooted at the module it is given, and one rooted elsewhere is
refused once it is set up.

- **Metrics.** Transition and checkpoint policies, and epoch-interval lr schedulers, receive the epoch's metrics: the
  core loop's `log()` averages, in which a training and a validation value logged under one name are averaged together.
  Log them under distinct names when a policy monitors one.
- **Metric-stepped lr schedulers** (`ReduceLROnPlateau`) use the epoch interval: step-interval schedulers are stepped
  without metrics. Only `interval`, `frequency` and `monitor` are taken from `pl_lrs_cfg`, so a missing monitored metric
  fails in finetuning-scheduler rather than being skipped as the core loop's `strict=False` would.
- **Resume** through the controller (`callback.controller.save` and `load`). The core loop's `checkpoint_state` holds
  none of the controller's state, so loading one into the callback is refused rather than resuming half a run; routing
  resume through it is [interpretune#809](https://github.com/speediedan/interpretune/issues/809).

## Conformance

finetuning-scheduler ships the view contract's cases; `interpretune.testing.conformance.TuningConformance` runs them
against a composition, selected by what its view declares. A repository subclasses it and sets `target`, as it does for
`ModelBackendConformance` (see [the adapter conformance contract](adapter_conformance_contract.md)):

```python
from interpretune.testing.conformance import ConformanceTarget, TuningConformance


class TestMyAdapterTuning(TuningConformance):
    target = ConformanceTarget(composition=("core", "my_adapter"), session_cfg_factory=build_session_cfg)
```

A view declaring `trainable=True` runs every case that applies to it; a refused composition runs the refusal case; a
run in which no case applies fails. Each case builds a fresh session, because cases change parameters and
`requires_grad`.
