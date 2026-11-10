# Tuning: interventions

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

An *intervention* applies an op (`add`, `scale`, `reject`, `patch`, ...: venngram's op registry) to a site's activations, as [Interpretune Intervention APIs](interpretune_intervention_apis.md) describes. Kinds of intervention differ in the op and in where its operand comes from: a constant or a dose (a zero ablation), a value computed from the model (a concept direction, another run's activation), a [latent model](fts_tuning_latent_models.md)'s reconstruction, or parameters of its own. `interpretune.tuning.interventions` is the seam for the last kind, *trainable* interventions, whose operand a schedule tunes. A steering vector is the `add` op with a learned vector; a learned per-unit gain is `scale`. Any trainable intervention can be attached at a site, on any backend but circuit-tracer (below), and scheduled by role.

## The seam

- **`OpIntervention(op, operand, scale_factor=1.0)`**: applies `op` at every position with the operand's tensor (times `scale_factor`), through the same op math the analysis ops use (`apply_op`, its out-of-place form, so gradients reach the operand). `operand` is a module returning the tensor, or a shape (an `int` for one vector over the site's last axis, or a tuple such as `(n_heads, d_head)` for a per-head operand), for a `LearnedOperand` that starts where the op is the identity (`0` for `add`, `1 / scale_factor` for `scale`), so attaching it changes nothing until it trains. The ops applied so far are `add` and `scale`; other registered ops are refused by name.
- **A steering vector** is `OpIntervention("add", d_model)`: its learned direction is the operand's tensor, `operand.value`.
- **`attach_intervention(root, name, intervention, site, frozen=False)`**: registers the intervention as `interventions.<name>` on the module (so it is in the module's state dict and reachable by its view) and splices it in with a forward hook on the module at `site`, whose output (the first element of a tuple output) becomes `intervention(output)`. `detach_intervention(root, name)` removes it.
- **Sites** are module names under the interpretune module: `model.model.layers.12` on a core module, `hf_model.model.layers.12` on an nnsight module, or a TransformerLens bridge's hook point such as `model.blocks.12.hook_out`. Venngram site addresses (`point:resid_post & layers:12`), one spelling for every backend, replace these once interpretune resolves them ([#864](https://github.com/speediedan/interpretune/issues/864), [#876](https://github.com/speediedan/interpretune/issues/876)).
- **Roles:** an attached intervention's tensors take venngram's role `intervention:<name>`, as a PEFT adapter's take `adapter:<name>`. A schedule selects one intervention with `role:intervention:<name>` and every intervention with `role:intervention`. Every interpretune module declares these roles in its cooperative `tunable_roles()`, which its venngram view reads, so no mixin is needed. A `frozen` intervention is an `instrument` instead: applied in every forward pass and never trained, as a precomputed steering vector is.

Interventions without parameters (an ablation, a patch, a dose of a concept direction) need no role and no seam: the analysis ops apply them during a pass.

**Circuit-tracer compositions refuse trainable interventions for now.** Their view assigns its transcoders' roles itself and admits no other tensors, so an intervention's tensors would have no role. circuit-tracer's own interventions are unaffected: `feature_intervention_forward` and `model_fwd_intervention` on concept directions ([circuit-tracer backend support](circuit_tracer_backend_support.md)) carry no parameters, so they never need one. Lifting the refusal where nothing conflicts needs the view to merge declared roles, site resolution that reaches circuit-tracer's forward on every backend, and attribution that never treats the intervention's tensors as gradient leaves ([#953](https://github.com/speediedan/interpretune/issues/953)).

## From a configuration

Every interpretune module attaches the interventions its configuration declares (`tuning_interventions`, a list of `InterventionCfg`) once its model is built and every module a site may name is registered, before its view is attached. So a steered run needs no module code, on any backend and on either training-loop interface:

```yaml
module_cfg:
  class_path: interpretune.config.module.ITConfig
  init_args:
    tuning_interventions:
      - name: steer
        site: model.model.layers.12  # hf_model.model.layers.12 with nnsight, model.blocks.12.hook_out with TransformerLens
```

`op` is `add` (the default, a steering vector) or `scale`, `shape` defaults to the model's hidden size (`config.hidden_size`, or a TransformerLens model's `cfg.d_model`), and `scale_factor` is a fixed multiplier on the operand. A configured intervention is always tunable: one attached frozen would start at the identity and never change. To hold a trained operand fixed, refreeze it in a later stage (below). To apply a precomputed one, attach it in code with `frozen=True`.

## Attaching them in code

Attach interventions in `register_tunable_modules()`, which runs once the model exists and before the view is built:

```python
from interpretune.tuning.interventions import OpIntervention, attach_intervention


class MySteeredModule(MyModule):
    def register_tunable_modules(self) -> None:
        super().register_tunable_modules()
        d_model = self.model.config.hidden_size
        attach_intervention(self, "steer", OpIntervention("add", d_model), "model.model.layers.12")
```

## Schedules: phases as stages

finetuning-scheduler's per-phase `refreeze` and `restore` keys let each phase be a stage of the experiment, rather than a step of gradually thawing more. Here the steering vector trains alone. Then a second stage holds it fixed and tunes the last four blocks, starting from where the steering stage ended:

```yaml
0:
  params: ["role:intervention:steer"]
  max_transition_epoch: 2
1:
  params: ["role:base & layers:-4:"]
  refreeze: ["role:intervention:steer"]
  restore: none
```

`layers:-4:` needs the layer structure a venngram component map gives the module's view. Where no published map describes the model, the view addresses tensors by name and role only, and a stage names its blocks by regex instead (for example `model\.model\.layers\.(2[4-7])\..*`). `refreeze` and `restore` run on the core loop; the Lightning callback refuses both keys by name.
