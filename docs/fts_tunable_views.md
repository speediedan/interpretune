# Tuning interpretune compositions through finetuning-scheduler views

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

| Composition | View | Roles |
| --- | --- | --- |
| core / HF (the default) | finetuning-scheduler's identity view | PEFT adapter tensors as `adapter:<name>` |
| TransformerLens | `TransformerBridgeView`, in TransformerLens names | as below |
| sae_lens with `add_saes_on_init=True` | the backend's view | the spliced SAEs as `latent:sae` |
| nnsight, local and dispatched | finetuning-scheduler's identity view over the HF module, registered as `hf_model` | PEFT adapter tensors as `adapter:<name>` |
| nnsight, remote (NDIF) or not yet dispatched | refused, with the reason | |
| circuit-tracer | refused, with the reason | |
| a composition declaring nothing | refused: never assumed trainable | |

## Declaring a view

A composition declares by overriding `build_tunable_view()`, returning a view or `None` (which declares nothing, so the
composition is refused). Roles are added cooperatively through `_tunable_roles()`, which each mixin extends from
`super()`'s, so a backend's view and another mixin's roles compose:

```python
class MyInterventionMixin:
    def _tunable_roles(self) -> dict[str, str]:
        roles = super()._tunable_roles()
        roles.update({f"steer.{n}": "intervention:steer" for n, _ in self.steer.named_parameters()})
        return roles
```

PEFT adapter tensors are found through PEFT's tuner-layer API rather than by name, so every PEFT tuner is covered and the
base weight PEFT renames (`base_layer.weight`) stays `base`.

## The TransformerLens view

`TransformerBridgeView` names parameters as `tl_named_parameters()` does (`blocks.9.attn.W_Q`). Each name maps to the
parameters its TransformerLens tensor is computed from, found by walking that tensor's autograd graph, so the mapping
follows whatever TransformerLens emits with no table kept here. A name with no parameter behind it (a zero bias
TransformerLens synthesizes for a bias-free projection) is left out, and a schedule naming it is refused.

- **Built on first use.** Constructing the view traces nothing, so a session that never tunes pays nothing.
- **Fused projections.** GPT-2's `q`, `k` and `v` weights are strided views of the fused `qkv` weight. A step through a
  split changes the fused parameter's values, so the fused parameter is thawed with its splits (`coupled`), and the
  default schedule keeps each block whole. finetuning-scheduler refuses a schedule that thaws a block's `W_Q` and `W_K`
  in different phases, because their storage overlaps.
- **Norms.** With `implicit_norm_thaw` (the default), thawing any of a block's parameters thaws the block's norms, and
  thawing an embedding thaws the final norm, as the legacy `TLNamesModelView` does.
- **Copies.** `copy.deepcopy` of a bridge does not preserve the fused-projection aliasing (the splits become independent
  copies), so build a view over the bridge a session actually trains.

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
warns when the schedule's final phase asks for more.

A run is resumed through the controller (`callback.controller.save` and `load`), not yet through the core loop's
`checkpoint_state`.

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
