# Tuning

interpretune fine-tunes any composition with [finetuning-scheduler](https://finetuning-scheduler.readthedocs.io):
a schedule thaws parameters phase by phase, on interpretune's core loop or on Lightning.

```python
schedule = {
    0: {"params": ["role:intervention"], "max_transition_epoch": 2},  # a steering vector first
    1: {"params": ["model.transformer.h.11.*"]},  # then the last block, by name
}
```

A schedule can name parameters, as finetuning-scheduler always has, or say **what** to tune: the role a tensor plays
(the model's own weights, an adapter, an intervention, a latent model, an instrument). That is the
[venngram](https://github.com/speediedan/venngram) protocol, which finetuning-scheduler uses to address a model through
its **tunable view**. Its addresses share and compose references to any subset of a model, so:

- **interpretability experiments** tune, freeze and promote adapters, interventions, latent models and instruments by
  role, however many wrappers (TransformerLens, nnsight, SAE Lens) sit around the model;
- **one schedule runs across backends**: each interpretune adapter gives its composition a view reporting the same
  roles, so a schedule written by role runs unchanged on the core/HF, TransformerLens and nnsight compositions;
- **experiments are defined once and shared**: conditions written as addresses depend far less on one model's
  parameter names, so they can be compared and orchestrated over several models, and composed by agents.

```{tip}
**You may not need any of this.** A schedule of parameter names and regexes needs no view and stays the simplest choice
for many uses, above all a schedule you write by hand. finetuning-scheduler's
[tunable views guide](https://finetuning-scheduler.readthedocs.io/en/latest/pytorch/tunable_views.html) explains when
addresses pay off.
```

Today an interpretune composition is addressed by role and by name. Addressing it by layer and architecture-neutral
component (`layers:-4: & component:mlp.*`), as a bare model already is through `venngram.view(model)`, follows in
[#900](https://github.com/speediedan/interpretune/issues/900).

## Running a schedule on the core loop

`FTSCoreCallback` drives finetuning-scheduler's `TuningScheduler` at the core loop's boundaries:

```python
from interpretune.config import SessionRunnerCfg
from interpretune.tuning.core_loop import FTSCoreCallback

runner_cfg = SessionRunnerCfg(it_session=session, max_epochs=4, callbacks=[FTSCoreCallback(schedule)])
```

Lightning compositions use finetuning-scheduler's own `FinetuningScheduler` callback with the same schedule. Both
interfaces see the same view, so a schedule thaws the same tensors on either (the RTE example's tests check this phase
by phase). [Phased fine-tuning on the core loop](#phased-fine-tuning-on-the-core-loop) below has the details.

## Roles

Every tensor has a role. Unlisted tensors are `base`; PEFT adapter tensors are found automatically and become
`adapter:<name>`. A composition declares anything else with `tunable_roles()`:

```python
class MyInterventionMixin:
    def tunable_roles(self) -> dict[str, str]:
        roles = super().tunable_roles()
        roles.update({f"steer.{n}": "intervention:steer" for n, _ in self.steer.named_parameters()})
        return roles
```

Overrides are cooperative: each calls `super()` and adds to its result, so a backend's roles and another mixin's
compose. The role kinds are venngram's (`base`, `adapter`, `intervention`, `latent`, `fitted`, `instrument`); the default
optimizer never steps `fitted` or `instrument` tensors.

PEFT adapters are found through PEFT's tuner-layer API rather than by name, so every tuner whose layers hold their
adapters there (LoRA and its variants, IA3 and similar) is covered, and the base weight PEFT renames
(`base_layer.weight`) stays `base`. Prompt-learning methods (prompt and prefix tuning) keep their tensors outside tuner
layers; those stay `base`.

A model-owned module the view must reach (a dictionary spliced into the model, for example) is registered with
`register_tunable_modules()`. It runs after whichever init path ran and before the view is attached.

## Declaring a view

A composition's view is what `build_tunable_view()` returns. The default is the identity view: physical names, the
roles above, trainable. So **a composition that does not override it can be tuned**, and most never need to.

A composition that cannot be tuned in some configuration declares why, and the refusal is reported by name rather than
the composition being treated as trainable. nnsight's, for instance:

```python
from interpretune.tuning.views import RefusingView, identity_view


def build_tunable_view(self):
    if "hf_model" in self._modules:  # local, dispatched weights
        return identity_view(self, roles=self.tunable_roles())
    return RefusingView(self, "nnsight compositions executing remotely (NDIF) have no local weights to tune")
```

A view with names of its own goes here too: TransformerLens's `TransformerBridgeView` lets a schedule say
`blocks.3.attn.W_Q` where the model stores a fused projection. Returning `None` declares nothing, and the composition is
refused.

Whatever is returned, every view's root is the interpretune module. Physical names are therefore the module's own
(`model.<...>`), the same names the Lightning callback sees through `pl_module`, and modules registered beside the model
(steering vectors, spliced SAEs) are in the view. It is attached as the module's `tunable_view` after model
initialization, where finetuning-scheduler finds it; an attached view is never replaced, so a hub adapter may attach
its own in `post_auto_model_init`.

## Choosing the view in configuration

A run can choose its view in configuration instead of in code:

```yaml
# schedule a composition by its own (physical) parameter names
tunable_view:
  class_path: venngram.tuning.CanonicalView
```

The module configuration's `tunable_view` (a `class_path` and optional `init_args`) is instantiated with the module as
the view's root and takes precedence over `build_tunable_view()`, so how a schedule names parameters is part of the
run's configuration, and of its configuration hash. A view class taking `roles` receives the composition's roles unless
`init_args` sets them; `init_args` cannot set the root. The [TransformerLens](fts_tuning_transformer_lens.md) page shows
both of its choices.

## Phased fine-tuning on the core loop

The callback builds the tuning scheduler over the session's module when training starts, so the schedule names
parameters through the module's view, roles included. The optimizer comes from the module's `optimizer_init` and the lr
schedulers from its `lr_scheduler_init`, unless `tuning_scheduler_kwargs` passes `optimizer` or `lr_scheduler`; each is
installed as the module's own, so the loop steps the phase's optimizer and nothing twice. Further `TuningScheduler`
arguments go in `tuning_scheduler_kwargs` (transition and checkpoint policies, `max_depth`); phase hooks, gates and RNG
forking are finetuning-scheduler's, unchanged. The core loop keeps its own `max_epochs`, and the callback warns when the
final phase asks for more. A `tuning_scheduler_factory` builds the tuning scheduler instead, for anything
`tuning_scheduler_kwargs` cannot express; it must be rooted at the module it is given.

- **Metrics.** Transition and checkpoint policies, and epoch-interval lr schedulers, receive the epoch's metrics: the
  core loop's `log()` averages, in which a training and a validation value logged under one name are averaged together.
  Log them under distinct names when a policy monitors one.
- **Metric-stepped lr schedulers** (`ReduceLROnPlateau`) use the epoch interval: step-interval schedulers are stepped
  without metrics. Only `interval`, `frequency` and `monitor` are taken from `pl_lrs_cfg`, so a missing monitored metric
  fails in finetuning-scheduler rather than being skipped as the core loop's `strict=False` would.
- **Resume** from the core loop's own checkpoint. The callback's `state_dict` is the tuning scheduler's `resume_state`
  (phase, pending decision, gate records, RNG forks, lr schedulers and registered state; the random streams stay with
  the loop's checkpoint), and taking it marks the resume point with the tuning scheduler's checkpoint policy, so the
  best checkpoint and rollback point it refers to are kept. Take `ctx.checkpoint_state()` in a callback's
  `on_train_epoch_end` listed after `FTSCoreCallback`, and pass it (or the path it was saved to) as
  `SessionRunnerCfg.resume_from`. The loop builds the tuning scheduler, loads its resume state, which rebuilds the
  optimizer phase by phase to the saved depth, then loads the module's and that optimizer's state and continues at the
  next epoch.

## Conformance

finetuning-scheduler ships the view contract's cases; `interpretune.testing.conformance.TuningConformance` runs them against a
composition, selected by what its view declares:

```python
from interpretune.testing.conformance import ConformanceTarget, TuningConformance


class TestMyAdapterTuning(TuningConformance):
    target = ConformanceTarget(composition=("core", "my_adapter"), session_cfg_factory=build_session_cfg)
```

A repository subclasses it and sets `target`, as it does for `ModelBackendConformance` (see
[the adapter conformance contract](adapter_conformance_contract.md)). A view declaring `trainable=True` runs every case
that applies to it; a refused composition runs the refusal case; a run in which no case applies fails. Each case builds
a fresh session, because cases change parameters and `requires_grad`.

## What each composition declares

| Composition | View | Roles |
| --- | --- | --- |
| [core / HF](fts_tuning_core_hf.md), and any composition not overriding `build_tunable_view()` (the default) | the identity view: trainable | PEFT adapter tensors as `adapter:<name>` |
| [TransformerLens](fts_tuning_transformer_lens.md) | `TransformerBridgeView`, in TransformerLens names (refused when the bridge holds quantized or packed parameters) | as on its page |
| [sae_lens](fts_tuning_sae_lens.md) with `add_saes_on_init=True` | the backend's view | the spliced SAEs as `latent:sae` |
| [nnsight](fts_tuning_nnsight.md), local and dispatched | the identity view over the HF module, registered as `hf_model` | PEFT adapter tensors as `adapter:<name>` |
| [nnsight](fts_tuning_nnsight.md), remote (NDIF) or not yet dispatched | refused, with the reason; tuning paths tracked in [#814](https://github.com/speediedan/interpretune/issues/814) and [#775](https://github.com/speediedan/interpretune/issues/775) | |
| [circuit-tracer](fts_tuning_circuit_tracer.md) | refused, with the reason; tuning paths tracked in [#813](https://github.com/speediedan/interpretune/issues/813), [#787](https://github.com/speediedan/interpretune/issues/787) and [#709](https://github.com/speediedan/interpretune/issues/709) | |
| [interp-engine (hub adapter)](fts_tuning_interp_engine.md) | interpretune's default identity view until the adapter declares its own; its circuit-tracer composition takes the circuit-tracer refusal | none declared yet |
| a composition whose `build_tunable_view()` returns `None` | refused: never assumed trainable | |
