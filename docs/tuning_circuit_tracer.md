# Tuning: circuit-tracer

[Tuning](tuning.md) describes how every composition declares and attaches its Fine-Tuning Scheduler view.

A circuit-tracer composition's view (`interpretune.adapters.circuit_tracer.tunable_view.CircuitTracerView`) covers its
transcoders, whichever backend it runs over:

- **Transcoders start as `instrument`.** Every transcoder tensor is a published artifact: never thawed, optimized or
  checkpointed, so a fresh view's default schedule is empty. Promote the tensors you want to adapt to
  `latent:transcoder` through Fine-Tuning Scheduler's promotion, and a schedule can then thaw them. Adapting transcoders
  that drift after fine-tuning ([interpretune#709](https://github.com/speediedan/interpretune/issues/709)) works this way.
- **The base model is refused by name.** The transcoders were trained on the base model's activations, so a schedule
  naming a base-model parameter is refused, saying why. Tuning the base model under a frozen replacement is a separate
  path ([interpretune#813](https://github.com/speediedan/interpretune/issues/813)).
- **Every offload mode keeps the parameters.** circuit-tracer's `offload='disk'` moves each offloaded module to the
  meta device and reloads it with `load_state_dict(assign=True)`, and by default both steps replace its parameter
  objects, which an optimizer built before an attribution would not see. The composition attributes under torch's
  `swap_module_params_on_conversion` instead, set only while `offload='disk'` attributes, so each new tensor is swapped
  into the existing parameter and the optimizer keeps stepping the module's own tensors (and a tied unembedding stays
  tied). Disk offload holds the offloaded weights on disk rather than in host memory, so an attribution needs less host
  memory than under `offload='cpu'`, at the cost of the reads. `offload=None` and `offload='cpu'` never replace a
  parameter.
- **Each transcoder layer is one depth group**, from the last layer; a cross-layer transcoder is one group.

The transcoders are registered on the Interpretune module once: under `model` when the replacement model is a torch
module, under the nnsight model's registered HF module (`hf_model`) when there is one, and as `transcoders` otherwise.
In that last case the base model's parameters are not the module's, so `CircuitTracerConfig.tune_base` refuses the
composition, naming why, rather than tuning nothing of the base.

With the transcoders registered as `transcoders`, every parameter of the module is a transcoder tensor, `instrument`
until a promotion, so there is nothing to optimize: a session whose configuration sets `optimizer_init` gets no
optimizer, with a warning, and analysis sessions run as before.

## Interventions and latent models beside the transcoders

Trainable interventions ([Tuning: interventions](tuning_interventions.md)) and latent models
([Tuning: latent models](tuning_latent_models.md)) attach to a circuit-tracer composition as to any other, with
three rules of its own:

- **Roles are merged, and only a conflict is refused.** The view assigns the transcoders' roles itself. The module's
  `tunable_roles()` (the attached interventions' `intervention:<name>`, the latent models' `latent:<name>`) are declared
  on the bound view, venngram's cooperative-role rule: a declared role on a tensor the view does not assign joins the
  view and is selectable by `role:`, a role equal to the view's own is accepted, and one that differs (a declaration
  giving a transcoder tensor any role but its own) is refused by name. Their tensors form the first depth group. The
  base model is still refused by name unless `tune_base` is set.
- **A site must be in the model circuit-tracer runs.** circuit-tracer runs its replacement model's forward,
  attribution included: on the TransformerLens backend the replacement model itself, on the nnsight backend the HF
  model it wraps (registered as `hf_model`), whose modules' forward hooks fire in its traces. A site elsewhere would
  never fire, so it is refused by name, as is a composition with no torch module behind its replacement model.
- **Attribution sees the intervention and never trains it.** An attached intervention runs in every forward pass
  circuit-tracer makes, so a graph attributes the intervened model. Attribution runs with every thawed tensor frozen,
  the module's own included, so attribution's backward passes never reach an intervention's or latent model's
  `.grad`. As with base tuning ([#813](https://github.com/speediedan/interpretune/issues/813)), a thawed intervention
  changes what the transcoders read: the graph stays complete, since error nodes absorb what the transcoders no longer
  reconstruct, and the cost is interpretability. Attributing with any tensor outside the transcoders thawed warns,
  suggesting a fit measurement (below).

## Lazily loaded weights

With `lazy_encoder` or `lazy_decoder` (the decoder is lazy by default), circuit-tracer re-reads $W_{enc}$ or $W_{dec}$
from safetensors on every access, so those weights are plain tensors that no optimizer can track. Lazy loading is what
makes attribution fit on smaller accelerators, since only the decoder rows of active features are ever resident, so
the view accommodates it rather than refusing it.

Set `CircuitTracerConfig.lazy_delta_rank` to give each lazily loaded weight $W$ a trainable low-rank correction
$\Delta = A B$, zero at initialization. Every read circuit-tracer makes then returns $W + \Delta$: a whole-matrix read
adds $A B$, and a decoder-row read for a feature set $S$ adds $A_S B$, so a sparse read never materializes the full
correction. The corrections are parameters under each transcoder's `lazy_deltas`, with the same `instrument` role as
the rest of it until promoted.

Without a rank, lazily loaded weights are left out of the view, and building the view warns, naming them.

## Measuring drift

`interpretune.adapters.circuit_tracer.drift.TranscoderFitMeter` measures how well a set of transcoders still fits a
model's MLPs, per layer, over chunks of MLP inputs and outputs: the fraction of variance unexplained, the mean number
of active features per position ($L_0$), and the counts of dead and dense features. These are the measures
[interpretune#709](https://github.com/speediedan/interpretune/issues/709) reads before and after fine-tuning.

Concrete example usage for this adapter is to come; it is tracked in [interpretune#812](https://github.com/speediedan/interpretune/issues/812).
