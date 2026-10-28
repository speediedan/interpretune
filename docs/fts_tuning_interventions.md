# Tuning: interventions

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

An *intervention* rewrites one site's activations with parameters of its own. A steering vector adds a learned direction to a residual stream; other kinds (a learned ablation, a low-rank edit) rewrite a site in other ways. `interpretune.tuning.interventions` is the seam they share, the counterpart of [latent models](fts_tuning_latent_models.md) for modules that rewrite a site rather than model it through latents. Any intervention can be attached at a site, on any backend, and scheduled by role.

## The seam

- **`SteeringVector(width, scale=1.0)`**: the first kind, `x + scale * vector` at every position. The vector starts at zero, so attaching it changes nothing until it trains. Its kind is `steer`.
- **`attach_intervention(root, name, intervention, site, kind=None, frozen=False)`**: registers the intervention as `interventions.<name>` on the module (so it is in the module's state dict and reachable by its view) and splices it in with a forward hook on the module at `site`, whose output (the first element of a tuple output) becomes `intervention(output)`. Sites are module names under the interpretune module: `model.model.layers.12` on a core module, `hf_model.model.layers.12` on an nnsight module, or a TransformerLens bridge's hook point. `kind` defaults to the intervention's `kind` attribute. `detach_intervention(root, name)` removes it.
- **Roles:** an attached intervention's tensors take the role `intervention:<kind>`, so a schedule selects every steering vector with `role:intervention:steer`. A `frozen` one is an `instrument`: applied in every forward pass and never trained, as a precomputed steering vector is. finetuning-scheduler promotes instruments only to latent roles, so a frozen intervention stays one.

## From a configuration

Every interpretune module attaches the interventions its configuration declares (`tuning_interventions`, a list of `InterventionCfg`) once its model is built and every module a site may name is registered, before its view is attached, and declares their roles. So a steered run needs no module code, on any backend and on either training-loop interface:

```yaml
module_cfg:
  class_path: interpretune.config.module.ITConfig
  init_args:
    tuning_interventions:
      - name: steer
        site: model.model.layers.12  # hf_model.model.layers.12 with nnsight, model.blocks.12.hook_out with TransformerLens
```

`kind` is `steer` (the only kind a configuration builds), `width` defaults to the model's (`config.hidden_size`, or a TransformerLens model's `cfg.d_model`), and `scale` and `frozen` are as above.

## Declaring them in code

Compose `InterventionsMixin` into a module. It adds the attached interventions' roles to the module's cooperative `tunable_roles()`, alongside any other mixin's (`LatentModelsMixin` included). Attach them in `register_tunable_modules()`, which runs once the model exists and before the view is built:

```python
from interpretune.tuning.interventions import InterventionsMixin, SteeringVector


class MySteeredModule(InterventionsMixin, MyModule):
    def register_tunable_modules(self) -> None:
        super().register_tunable_modules()
        width = self.model.config.hidden_size
        self.attach_intervention("steer", SteeringVector(width), "model.model.layers.12")
```

A schedule then trains the steering vector alone first, and thaws blocks by the adapter's own names after it:

```yaml
0:
  params: ["role:intervention:steer"]
  max_transition_epoch: 2
1:
  params: ["model\\.model\\.layers\\.(2[4-7])\\..*"]
```

The [core and HF](fts_tuning_core_hf.md), [TransformerLens](fts_tuning_transformer_lens.md) and [nnsight](fts_tuning_nnsight.md)
Tuning pages each walk through a steered, phased fine-tune of Qwen3-0.6B configured this way.
