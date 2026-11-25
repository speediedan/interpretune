# Tuning: latent models

[Tuning](tuning.md) describes how every composition declares and attaches its finetuning-scheduler view.

A *latent model* models a site's activations through latents. Examples:

- a sparse autoencoder reads and rewrites one site;
- a transcoder reads one site (an MLP's input) and rewrites another (that MLP's output);
- a block-sparse featurizer, or any other dictionary, does either.

`interpretune.tuning.latents` fixes only the seam these models share. Any of them can then be attached at a site, on any backend but circuit-tracer, and scheduled by role. It provides the seam, not a recipe: which latent models tune well, and how, is still being measured, so concrete latent tuning examples will follow that work ([interpretune#812](https://github.com/interpretune/interpretune/issues/812)).

## The seam

- **`LatentModel`**: the protocol, an `nn.Module` with `encode(x)` (activations to latents) and `decode(z)` (latents to activations). sae_lens SAEs and circuit-tracer transcoders already satisfy it.
- **`LatentSite(read, write=None, read_from="output", op="replace_with_error")`**: where the latent model reads (`read`; for a module path, the module's output, or its first input with `read_from="input"`), which site it rewrites (`write`, by default the `read` site), and the op applied there with the reconstruction as the operand. `read` and `write` are venngram site addresses, one spelling for every backend: `LatentSite("point:mlp.hook_in & layers:3", write="point:mlp.hook_out & layers:3")` places a transcoder on any composition whose architecture interpretune maps. A module name under the interpretune module also works, but names the site on one composition only: `model.layers.0.mlp` on a core module, `hf_model.model.layers.0.mlp` on an nnsight module, or a TransformerLens bridge's hook point. An address names the activation itself, so it decides whether the module's input or output is read or rewritten, and `read_from` applies only to a module path. Either may instead reference an address book's named set (`@mlp3_in`), with the book's sets passed as `attach_latent_model(..., sets=book_sets(book))`.
- **Ops** (venngram's registry, as every [intervention](tuning_interventions.md) uses):
  - `"replace"`: the output becomes the reconstruction, so the model runs on what the latents capture.
  - `"replace_with_error"`: the straight-through substitution. The output keeps its value (the reconstruction plus its detached error, as sae_lens's error term does) while gradients reach the latent model through the reconstruction. It is one of venngram's core ops, defined with the others on [Interpretune Intervention APIs](interpretune_intervention_apis.md).
  - `None`: no op. The latents are captured (`last_latents`) and the output is unchanged, a read rather than an intervention.
- **`attach_latent_model(root, name, latent, site, frozen=False, release=None)`**: registers the latent model as `latent_models.<name>` on the module (so it is in the module's state dict and reachable by its view) and splices it in with forward hooks. `detach_latent_model(root, name)` removes it.
- **Roles:** an attached latent model's tensors take venngram's role `latent:<name>`, so a schedule selects them with `role:latent:<name>`. Every interpretune module declares these roles in its cooperative `tunable_roles()`, which its venngram view reads, so no mixin is needed. A `frozen` latent model is an `artifact`: loaded from its source, never thawed or checkpointed until a schedule promotes it.
- **Placement:** an attached latent model also takes the layer and component of the module its site rewrites, although it is registered beside the model rather than inside it. An address then selects latent models by depth: `role:latent & layers:12:` thaws the ones spliced at layers 12 and above, and `role:latent & component:mlp` the transcoders that rewrite an MLP. Every interpretune module declares these placements in its cooperative `tunable_placements()`, which its view reads (venngram's `attached`, on every composition but circuit-tracer). A latent model whose site lies outside every module the component map places locates at no layer, with a warning, and so does every latent model of a model no map describes; either is still selected by name and role.

On a circuit-tracer composition another latent model attaches beside the transcoders, at modules of the model circuit-tracer runs, with its role merged into the view's ([Tuning: circuit-tracer](tuning_circuit_tracer.md#interventions-and-other-latent-models-beside-the-transcoders)); the transcoders themselves are tuned by promoting them. The analysis surface's feature ops (venngram feature targets at an SAE's `hook_sae_acts_post`) reach sae_lens SAEs only, not latent models attached here.

## Attaching them

Attach latent models in `register_tunable_modules()`, which runs once the model exists and before the view is built:

```python
from interpretune.tuning.latents import LatentSite, attach_latent_model


class MyLatentTuningModule(MyModule):
    def register_tunable_modules(self) -> None:
        super().register_tunable_modules()
        # a transcoder: it reads layer 3's MLP input and rewrites that MLP's output, on every backend
        site = LatentSite("point:mlp.hook_in & layers:3", write="point:mlp.hook_out & layers:3")
        attach_latent_model(self, "mlp3", load_transcoder(), site)
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

Attached with `frozen=True`, the same transcoder stays an `artifact`, applied and never trained, until a later stage promotes it:

```yaml
0:
  params: ["role:base & layers:3"]
  max_transition_epoch: 2
1:
  params: ["role:latent:mlp3"]
  promote: {"role:artifact": "latent:mlp3"}
```

Splice a latent model the run will tune with `op="replace"`, so the model runs on its reconstruction. Under `"replace_with_error"` the site's value is the original activation whatever the latent model computes, so tuning it against the task loss moves its parameters without changing that loss; the straight-through substitution is for training a latent model to reconstruct its site, with a reconstruction loss of its own. The examples gallery's role-kinds notebook (`tuning/role_kinds_tuning.ipynb`) promotes and tunes a published SAE this way.

`layers:3` needs the layer structure a venngram component map gives the module's view; without one, name the layer by regex (`model\.model\.layers\.3\..*`). `promote`, `refreeze` and `restore` run on the core loop; the Lightning callback refuses them.

sae_lens's own splice (`add_saes_on_init`) keeps registering its SAEs as `attached_saes` with the `latent:sae` role, as [sae_lens](tuning_sae_lens.md) describes. The seam here is for backends without a native splice, and for latent models other than SAEs.

## Pinning a book to the models it was written against

An address book that names latent models (`module:latent_models.sae`, `role: latent:sae`, `features: sae`) selects the same tensors wherever it loads, but it means the same thing only on the base model and the dictionaries it was written against. A pinned book records them in its `requires` (venngram's book schema 8), and is refused on any others, naming each entry that differs.

```python
from interpretune.tuning.latents import LatentSite, attach_latent_model
from interpretune.tuning.pins import check_book_pins, pinned_book

attach_latent_model(
    module, "sae", sae, LatentSite("point:hook_resid_post & layers:12"),
    release={"sae_lens": "gemma-scope-2b-pt-res", "id": "layer_12/width_16k/average_l0_82"},
)
document = pinned_book(document, module)  # the book, with `requires.base_model` and `requires.latent_models`
check_book_pins(book, module)             # venngram.PinMismatchError when a loaded model differs
```

- **The base model** is its id and the commit it was loaded from. A model built from a configuration or loaded from a local path has no commit, so a book cannot be pinned to it.
- **Each latent model** is the `release` it was attached with. A module also reports its spliced SAE Lens SAEs (by their `attached_saes` key, from their configured release and id) and circuit-tracer's transcoders (as `transcoder`, where `transcoder_set` states a revision: `<repository>@<revision>`). `releases=` gives the release of one the module does not hold, such as a catalog's. A latent model with no stated release cannot be pinned and never satisfies a pin.
- **Frames:** with `catalog=`, the version of each frame the catalog declares is pinned with its latent model.
- **`latent_models=`** lists the latent models the book names. The default is every one loaded.

`book_sets(book, module)`, a configured intervention's `book`, and `load_recipe(book, module=module)` check a pinned book when they read it, and refuse one when there is no module to check it against. `export_book(..., pins=book_pins(module))` writes the pins into an exported book. A schedule's `addresses` are read by finetuning-scheduler, which does not check pins, so it refuses a schedule that uses a pinned book's sets.
