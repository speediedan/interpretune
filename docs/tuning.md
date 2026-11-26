# Tuning

Every Interpretune composition exposes a **tunable view**: which of its tensors may be tuned, the role each plays, how
they group by depth, and which may be promoted to another role. The view is the tuning interface. Its protocol is
venngram's (`venngram.tuning`), not any one scheduler's, so any scheduler that consumes `venngram.tuning` views can drive
an Interpretune composition.

Today that scheduler is [finetuning-scheduler](https://finetuning-scheduler.readthedocs.io): a schedule thaws parameters
phase by phase, on Interpretune's core loop or on a framework adapter loop like Lightning. The schedule format (phases,
and the per-phase `promote`, `refreeze` and `restore` keys) is finetuning-scheduler's own; the view it reads is not.

```python
schedule = {
    0: {"params": ["role:intervention"], "max_transition_epoch": 2},  # a steering vector first
    1: {"params": ["model.transformer.h.11.*"]},  # then the last block, by name
}
```

A schedule can name parameters, as Fine-Tuning Scheduler always has, or say **what** to tune: the role a tensor plays
(the model's own weights, an adapter, an intervention, a latent model, a loaded artifact), addressed through the model's
**tunable view**.

To enable this model addressing abstraction, Fine-Tuning Scheduler and Interpretune use the
[venngram](https://github.com/speediedan/venngram) protocol. Its addresses can share and compose references to any
subset of a model: from groups of layers to intersecting sets of individual neurons, and from latent space features to
activation sites and attribution graphs. Schedules use the part of it that selects tensors to tune. Addressing by
role means:

- **interpretability experiments** tune, freeze and promote adapters, interventions, latent models and artifacts by
  role, however many wrappers (TransformerLens, nnsight, SAE Lens) sit around the model;
- **one schedule runs across backends**: each Interpretune adapter gives its composition a view reporting the same
  roles, so a schedule written by role runs unchanged on the core/HF, TransformerLens and nnsight compositions;
- **experiments are defined once and shared**: conditions written as addresses depend far less on one model's
  parameter names, so they can be compared and orchestrated over several models, and composed by agents.

```{tip}
**You may not need any of this.** A schedule of parameter names and regexes needs no view and stays the simplest choice
for many uses, above all a schedule you write by hand. Fine-Tuning Scheduler's
[tunable views guide](https://finetuning-scheduler.readthedocs.io/en/latest/pytorch/tunable_views.html) explains when
addresses pay off.
```

An Interpretune composition is addressed by role, by name and, wherever venngram has a component map for its model, by
layer and architecture-neutral component (`layers:-4: & component:mlp.*`), as a bare model is through
`venngram.view(model)`.

## venngram elsewhere in Interpretune

Tuning is one of several places Interpretune consumes venngram:

- **Tuning** (this page): the tunable view protocol and role vocabulary come from `venngram.tuning` and `venngram.roles`,
  and `TuningConformance` runs Fine-Tuning Scheduler's view conformance cases against a composition's view.
- **Interventions**: `interpretune.analysis.venngram` applies a venngram recipe's treatments through the model
  backends' native intervention ops or an analysis backend's feature interventions, and a support record derived from
  each backend's own refuses what it cannot express. [Interpretune Intervention APIs](interpretune_intervention_apis.md) describes the surfaces it runs on, and
  [#905](https://github.com/interpretune/interpretune/issues/905) tracks interventions on venngram-addressed unit sets.
- **Recipes by reference**: `load_recipe` reads a recipe from a venngram document reference (fetched, hash-checked and
  returned with its pinned reference for provenance), a local file or an inline mapping, so an experiment's treatment is
  shared as data rather than code.

## Running a schedule on the core loop

`FTSCoreCallback` drives Fine-Tuning Scheduler's `TuningScheduler` at the core loop's boundaries:

```python
from interpretune.config import SessionRunnerCfg
from interpretune.tuning.core_loop import FTSCoreCallback

runner_cfg = SessionRunnerCfg(it_session=session, max_epochs=4, callbacks=[FTSCoreCallback(schedule)])
```

A composition on Lightning, today's framework adapter loop, uses Fine-Tuning Scheduler's own `FinetuningScheduler`
callback with the same schedule. Both interfaces see the same view, so a schedule thaws the same tensors on either (the
RTE example's tests check this phase by phase). [Phased fine-tuning on the core loop](#phased-fine-tuning-on-the-core-loop) below has the details.

## One schedule across compositions

The `speediedan/rte` component ships one GPT-2 schedule written only in addresses,
`ft_schedules/rte_ft.gpt2.addresses.yaml`, and a run configuration for each composition that runs it:

```bash
interpretune --run_command train --config hub://speediedan/rte/runs/rte_ft.gpt2.core.addresses.yaml
interpretune --run_command train --config hub://speediedan/rte/runs/rte_ft.gpt2.transformer_lens.yaml
interpretune --run_command train --config hub://speediedan/rte/runs/rte_ft.gpt2.nnsight.yaml
```

```yaml
0:
  params:
  - "component:ln_final"
  - "layers:-2: & component:mlp.*"
1:
  params:
  - "layers:-2: & (component:attn.* | component:ln*)"
2:
  params:
  - "layers:-6:-2"
```

Quote each address: `&` and `*` begin YAML anchors and aliases. The three compositions name the same tensor three ways
(`model.transformer.h.10.mlp.c_fc.weight` on the core/HF composition, `hf_model.transformer.h.10.mlp.c_fc.weight` on
nnsight, a bridge path under `blocks.10` on TransformerLens), and each address resolves to the composition's own
names through GPT-2's component map. The example's test runs all three through the CLI and maps every epoch's thawed
tensors back to GPT-2's own names: the sets must be equal in every epoch. Two settings make them so:

- **The TransformerLens composition turns `implicit_norm_thaw` off** (`tunable_view` in its session configuration).
  With the default, thawing any tensor of a block also thaws that block's norms on that composition alone, so phase 0
  would thaw blocks 10 and 11's norms there and in phase 1 everywhere else.
- **A TransformerLens bridge reads some fused tensors through splits it makes of them.** GPT-2's bridge computes
  attention from `q`, `k` and `v` views of the fused `c_attn`, which itself receives no gradient. The view places
  those splits where the fused tensor is, so `component:attn.qkv` thaws the tensors a step reaches
  ([#1107](https://github.com/interpretune/interpretune/issues/1107)), and `TransformerBridgeView.wrapped_names()` maps
  each of its physical names to the wrapped model's name, a split to the tensor it views, for comparisons like this
  one.

The schedule leaves adapters out: a `lora_cfg` is applied on the core/HF and TransformerLens compositions but not on
nnsight, so `role:adapter` would select tensors on two of the three.

## Roles

Every tensor has a role. Unlisted tensors are `base`; PEFT adapter tensors are found by venngram's overlay maps and
become `adapter:<name>`. A composition declares anything else with `tunable_roles()`, by address or by name, as
venngram views take a `roles` mapping: a role mapped to an address is resolved against the view's own names and
structure when the view is built, and a tensor name mapped to a role names one tensor.

```python
from interpretune.tuning.views import merge_roles


class MyInterventionMixin:
    def tunable_roles(self) -> dict[str, str]:
        return merge_roles(super().tunable_roles(), {"intervention:steer": "module:steer"})
```

Overrides are cooperative: each calls `super()` and merges its declarations into the result, so a backend's roles and
another mixin's compose; `merge_roles` joins two addresses given for one role into their union, and a later name entry
replaces an earlier one. An address states no `role` term, since roles are what it assigns, and venngram refuses one
that selects nothing, or two addresses giving one tensor different roles unless a name entry settles it. The roles
Interpretune declares itself are addresses: `module:latent_models.<name>` for an attached latent model,
`module:interventions.<name>` for an intervention, `module:attached_saes.<key>` for a spliced SAE Lens SAE, and a
`name:` pattern for each LoRA adapter of a TransformerLens composition. The address grammar is venngram's; on a
mapped model, `{"adapter:lora": "component:attn.* & leaf:lora_*"}` reads the same on every architecture its map
describes. The role kinds are venngram's (`base`, `adapter`, `intervention`, `latent`, `fitted`, `artifact`); the default
optimizer never steps `fitted` or `artifact` tensors. A `fitted` tensor is computed in the run and checkpointed; an
`artifact` is loaded from a published or saved source, so it is reloaded from that source rather than checkpointed.

venngram publishes an overlay map for each PEFT tuner family whose layers keep their adapters in named containers
(LoRA and DoRA, AdaLoRA, IA3, LoHa, LoKr, OFT, BOFT, VeRA and FourierFT), so each adapter tensor takes the role of its
adapter's name, and the base weight PEFT renames (`base_layer.weight`) stays `base`. Prompt-learning methods (prompt and prefix tuning) keep their tensors outside tuner
layers; those stay `base`.

A model-owned module the view must reach (a dictionary spliced into the model, for example) is registered with
`register_tunable_modules()`. It runs after whichever init path ran and before the view is attached.

## Declaring a view

A composition's view is what `build_tunable_view()` returns. The default is venngram's composed view of the module,
`venngram.view(self, roles=self.tunable_roles())`: the component map published for the model's architecture, mounted
where the module holds the model (`model.`), places every tensor by layer and component, and the overlay maps and
`tunable_roles()` give the roles above. So **a composition that does not override it can be tuned**, by structure as well
as by name:

```python
schedule = {
    0: {"params": ["role:adapter & layers:-4:"]},  # the adapters of the top four blocks
    1: {"params": ["layers:-4: & component:mlp.*"]},  # then those blocks' MLPs
}
```

Where no published map describes the model, the default is the identity view (physical names and the roles above, no
layers or components), and an info-level message says that registering a map (`venngram.maps.register`) or publishing
one to a registry adds them. A composition overrides `build_tunable_view()` only to refuse, or to offer names of its own.

A composition that cannot be tuned in some configuration declares why, rather than being treated as trainable. nnsight's, for instance:

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

Whatever is returned, every view's root is the Interpretune module. Physical names are therefore the module's own
(`model.<...>`), the same names the Lightning callback sees through `pl_module`, and modules registered beside the model
(steering vectors, spliced SAEs) are in the view. It is attached as the module's `tunable_view` after model
initialization, where Fine-Tuning Scheduler finds it; an attached view is never replaced, so a hub adapter may attach
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
`init_args` sets them; `init_args` cannot set the root. The [TransformerLens](tuning_transformer_lens.md) page shows
both of its choices.

## Phased fine-tuning on the core loop

The callback builds the tuning scheduler over the session's module when training starts, so the schedule names
parameters through the module's view, roles included. The optimizer comes from the module's `optimizer_init` and the lr
schedulers from its `lr_scheduler_init`, unless `tuning_scheduler_kwargs` passes `optimizer` or `lr_scheduler`; each is
installed as the module's own, so the loop steps the phase's optimizer and nothing twice. Further `TuningScheduler`
arguments go in `tuning_scheduler_kwargs` (transition and checkpoint policies, `max_depth`); phase hooks, gates and RNG
forking are Fine-Tuning Scheduler's, unchanged. The core loop keeps its own `max_epochs`, and the callback warns when the
final phase asks for more. A `tuning_scheduler_factory` builds the tuning scheduler instead, for anything
`tuning_scheduler_kwargs` cannot express; it must be rooted at the module it is given.

- **Metrics.** Transition and checkpoint policies, and epoch-interval lr schedulers, receive the epoch's metrics: the
  core loop's `log()` averages, in which a training and a validation value logged under one name are averaged together.
  Log them under distinct names when a policy monitors one.
- **Metric-stepped lr schedulers** (`ReduceLROnPlateau`) use the epoch interval: step-interval schedulers are stepped
  without metrics. Only `interval`, `frequency` and `monitor` are taken from `pl_lrs_cfg`, so a missing monitored metric
  fails in Fine-Tuning Scheduler rather than being skipped as the core loop's `strict=False` would.
- **Resume** from the core loop's own checkpoint. The callback's `state_dict` is the tuning scheduler's `resume_state`
  (phase, pending decision, gate records, RNG forks, lr schedulers and registered state; the random streams stay with
  the loop's checkpoint), and taking it marks the resume point with the tuning scheduler's checkpoint policy, so the
  best checkpoint and rollback point it refers to are kept. Take `ctx.checkpoint_state()` in a callback's
  `on_train_epoch_end` listed after `FTSCoreCallback`, and pass it (or the path it was saved to) as
  `SessionRunnerCfg.resume_from`. The loop builds the tuning scheduler, loads its resume state, which rebuilds the
  optimizer phase by phase to the saved depth, then loads the module's and that optimizer's state and continues at the
  next epoch.

A research example of a phased run on the core loop, with run-fitted state, `artifact` transcoders, gates and per-phase
`refreeze` and `restore`, is the [sparsity chain on RTE](tuning_sparsity_chain.md).
A one-phase LoRA run with an auxiliary representation loss and a per-epoch evaluation callback is the
[Knowing-Using Gap example](tuning_knowing_using.md).

## Conformance

Fine-Tuning Scheduler ships the view contract's cases; `interpretune.testing.conformance.TuningConformance` runs them against a
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

An address means the same tensors on every composition of one model, and a test can check that across backends.
`assert_same_selection` (in `interpretune.testing.conformance.tuning`) resolves each address on every view it is given
and compares the selections by the wrapped model's own tensor names: a TransformerLens view's tensors by the names the
bridge recorded for them, and a view mounted at a submodule (`model` on a core/HF composition, `hf_model` on an nnsight
one) without that mount. It fails naming the address and each backend whose selection differs, or on which the address
does not resolve:

```python
from interpretune.testing.conformance.tuning import assert_same_selection

views = {"core/HF": core_view, "nnsight": nnsight_view, "TransformerLens": bridge_view}  # one model, three backends
assert_same_selection(views, ["layers:-3: & component:attn.*", "role:latent & layers:3:"])
```

Interpretune's own tests run it on GPT-2's core/HF, nnsight and TransformerLens compositions, for addresses by layer
and component and for latent models spliced by site address (`tests/core/test_cross_backend_addresses.py`).

## What each composition declares

| Composition | View | Roles |
| --- | --- | --- |
| [core / HF](tuning_core_hf.md), and any composition not overriding `build_tunable_view()` (the default) | venngram's composed view (the identity view where no map describes the model): trainable | PEFT adapter tensors as `adapter:<name>` |
| [TransformerLens](tuning_transformer_lens.md) | `TransformerBridgeView`, in TransformerLens names (refused when the bridge holds quantized or packed parameters) | LoRA tensors from `lora_cfg` as `adapter:<name>` (see [LoRA adapters](tuning_transformer_lens.md#lora-adapters)); the rest as on its page |
| [sae_lens](tuning_sae_lens.md) with `add_saes_on_init=True` | the backend's view | the spliced SAEs as `latent:sae` |
| [nnsight](tuning_nnsight.md), local and dispatched | venngram's composed view over the HF module, registered as `hf_model` | none: nnsight does not apply a `lora_cfg` ([#942](https://github.com/interpretune/interpretune/issues/942)) |
| [nnsight](tuning_nnsight.md), remote (NDIF) or not yet dispatched | refused, with the reason; remote compositions train session-created adapter or intervention tensors in [bounded remote sessions](tuning_nnsight.md#remote-ndif-tuning-in-bounded-sessions) instead; local paths tracked in [#814](https://github.com/interpretune/interpretune/issues/814) | |
| [circuit-tracer](tuning_circuit_tracer.md) | `CircuitTracerView`: the transcoders, with base-model tensors refused unless `tune_base` is set; refused when no replacement model is loaded | transcoders as `artifact`, promotable to `latent:transcoder`; base tuning under a frozen replacement is [#813](https://github.com/interpretune/interpretune/issues/813) |
| [interp-engine (hub adapter)](tuning_interp_engine.md) | Interpretune's default view, which fits `(core, interp_engine)` with no override (measured by the adapter's tuning conformance); its circuit-tracer composition takes the circuit-tracer view | none declared yet |
| a composition whose `build_tunable_view()` returns `None` | refused: never assumed trainable | |

Any composition but circuit-tracer can also carry [latent models](tuning_latent_models.md) (SAEs, transcoders
and other dictionaries) spliced in at a site, which take the role `latent:<name>`, and learned
[interventions](tuning_interventions.md) (steering vectors and other site rewrites), which take the role
`intervention:<name>`. Every interpretune module declares both in its `tunable_roles()`.
