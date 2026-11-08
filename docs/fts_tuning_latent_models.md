# Tuning: latent models

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

A *latent model* models a site's activations through latents. Examples:

- a sparse autoencoder reads and rewrites one site;
- a transcoder reads one site (an MLP's input) and rewrites another (that MLP's output);
- a block-sparse featurizer, or any other dictionary, does either.

`interpretune.tuning.latents` fixes only the seam these models share. Any of them can then be attached at a site, on any backend but circuit-tracer, and scheduled by role. It provides the seam, not a recipe: which latent models tune well, and how, is still being measured, so concrete latent tuning examples will follow that work ([interpretune#812](https://github.com/speediedan/interpretune/issues/812)).

## The seam

- **`LatentModel`**: the protocol, an `nn.Module` with `encode(x)` (activations to latents) and `decode(z)` (latents to activations). sae_lens SAEs and circuit-tracer transcoders already satisfy it.
- **`LatentSite(read, write=None, read_from="output", mode="replace_with_error")`**: where the latent model reads (the `read` module's output, or its first input) and which module's output it rewrites (`write`, by default the `read` module). Paths are module names under the interpretune module, so a site is `model.layers.0.mlp` on a core module, `hf_model.model.layers.0.mlp` on an nnsight module, or a TransformerLens bridge's hook point. Venngram site addresses replace these once interpretune resolves them ([#864](https://github.com/speediedan/interpretune/issues/864), [#876](https://github.com/speediedan/interpretune/issues/876)).
- **Modes:**
  - `"replace"`: runs the model on the reconstruction.
  - `"replace_with_error"`: keeps each output's value (the reconstruction plus its detached error, as sae_lens's error term does) while gradients reach the latent model through the reconstruction.
  - `"observe"`: records the latents (`last_latents`) and changes nothing.
- **`attach_latent_model(root, name, latent, site, frozen=False)`**: registers the latent model as `latent_models.<name>` on the module (so it is in the module's state dict and reachable by its view) and splices it in with forward hooks. `detach_latent_model(root, name)` removes it.
- **Roles:** an attached latent model's tensors take venngram's role `latent:<name>`, so a schedule selects them with `role:latent:<name>`. Every interpretune module declares these roles in its cooperative `tunable_roles()`, which its venngram view reads, so no mixin is needed. A `frozen` latent model is an `instrument`: never thawed or checkpointed until a schedule promotes it.

A circuit-tracer composition refuses latent models attached this way: its view assigns its roles itself, and its transcoders are tuned by promoting them. The analysis surface's feature ops (venngram feature targets at an SAE's `hook_sae_acts_post`) reach sae_lens SAEs only, not latent models attached here.

## Attaching them

Attach latent models in `register_tunable_modules()`, which runs once the model exists and before the view is built:

```python
from interpretune.tuning.latents import LatentSite, attach_latent_model


class MyLatentTuningModule(MyModule):
    def register_tunable_modules(self) -> None:
        super().register_tunable_modules()
        attach_latent_model(self, "mlp3", load_transcoder(), LatentSite("model.model.layers.3.mlp", read_from="input"))
```

## Schedules: phases as stages

finetuning-scheduler's per-phase keys let each phase be a stage of the experiment. Here the transcoder trains first; a second stage then holds it fixed (`refreeze`) and tunes the layer it models, from where the first stage ended (`restore: none`):

```yaml
0:
  params: ["role:latent:mlp3"]
  max_transition_epoch: 2
1:
  params: ["role:base & layers:3"]
  refreeze: ["role:latent:mlp3"]
  restore: none
```

Attached with `frozen=True`, the same transcoder stays an `instrument`, applied and never trained, until a later stage promotes it:

```yaml
0:
  params: ["role:base & layers:3"]
  max_transition_epoch: 2
1:
  params: ["role:latent:mlp3"]
  promote: {"role:instrument": "latent:mlp3"}
```

`layers:3` needs the layer structure a venngram component map gives the module's view; without one, name the layer by regex (`model\.model\.layers\.3\..*`). `promote`, `refreeze` and `restore` run on the core loop; the Lightning callback refuses them by name.

sae_lens's own splice (`add_saes_on_init`) keeps registering its SAEs as `attached_saes` with the `latent:sae` role, as [sae_lens](fts_tuning_sae_lens.md) describes. The seam here is for backends without a native splice, and for latent models other than SAEs.
