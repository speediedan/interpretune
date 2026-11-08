# Tuning: interventions

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

An *intervention* rewrites one site's activations with parameters of its own. A steering vector adds a learned direction to a residual stream; other kinds (a learned ablation, a low-rank edit) rewrite a site in other ways. `interpretune.tuning.interventions` is the seam they share, the counterpart of [latent models](fts_tuning_latent_models.md) for modules that rewrite a site rather than model it through latents. Any intervention can be attached at a site, on any backend but circuit-tracer, and scheduled by role.

## The seam

- **`SteeringVector(width, scale=1.0)`**: the first kind, `x + scale * vector` at every position. The vector starts at zero, so attaching it changes nothing until it trains.
- **`attach_intervention(root, name, intervention, site, frozen=False)`**: registers the intervention as `interventions.<name>` on the module (so it is in the module's state dict and reachable by its view) and splices it in with a forward hook on the module at `site`, whose output (the first element of a tuple output) becomes `intervention(output)`. `detach_intervention(root, name)` removes it.
- **Sites** are module names under the interpretune module: `model.model.layers.12` on a core module, `hf_model.model.layers.12` on an nnsight module, or a TransformerLens bridge's hook point such as `model.blocks.12.hook_out`. Venngram site addresses (`point:resid_post & layers:12`), one spelling for every backend, replace these once interpretune resolves them ([#864](https://github.com/speediedan/interpretune/issues/864), [#876](https://github.com/speediedan/interpretune/issues/876)).
- **Roles:** an attached intervention's tensors take venngram's role `intervention:<name>`, as a PEFT adapter's take `adapter:<name>`. A schedule selects one intervention with `role:intervention:<name>` and every intervention with `role:intervention`. Every interpretune module declares these roles in its cooperative `tunable_roles()`, which its venngram view reads, so no mixin is needed. A `frozen` intervention is an `instrument` instead: applied in every forward pass and never trained, as a precomputed steering vector is.

A circuit-tracer composition refuses interventions: its view assigns its roles itself, so the intervention's tensors would have none.

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

`kind` is `steer` (the only kind a configuration builds), `width` defaults to the model's (`config.hidden_size`, or a TransformerLens model's `cfg.d_model`), and `scale` is the vector's fixed multiplier. A configured vector is always tunable: one attached frozen would start at zero and never change. To hold a trained vector fixed, refreeze it in a later stage (below). To apply a precomputed one, attach it in code with `frozen=True`.

## Attaching them in code

Attach interventions in `register_tunable_modules()`, which runs once the model exists and before the view is built:

```python
from interpretune.tuning.interventions import SteeringVector, attach_intervention


class MySteeredModule(MyModule):
    def register_tunable_modules(self) -> None:
        super().register_tunable_modules()
        width = self.model.config.hidden_size
        attach_intervention(self, "steer", SteeringVector(width), "model.model.layers.12")
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
