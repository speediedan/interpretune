# Tuning: circuit-tracer

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

A circuit-tracer composition's view (`interpretune.adapters.circuit_tracer.tunable_view.CircuitTracerView`) covers its
transcoders, whichever backend it runs over:

- **Transcoders start as `instrument`.** Every transcoder tensor is a published artifact: never thawed, optimized or
  checkpointed, so a fresh view's default schedule is empty. Promote the tensors you want to adapt to
  `latent:transcoder` through finetuning-scheduler's promotion, and a schedule can then thaw them. Adapting transcoders
  that drift after fine-tuning ([interpretune#709](https://github.com/speediedan/interpretune/issues/709)) works this way.
- **The base model is refused by name.** The transcoders were trained on the base model's activations, so a schedule
  naming a base-model parameter is refused, saying why. Tuning the base model under a frozen replacement is a separate
  path ([interpretune#813](https://github.com/speediedan/interpretune/issues/813)).
- **Disk offload is refused.** circuit-tracer's `offload='disk'` reloads offloaded modules with new parameter objects
  after an attribution, which an optimizer built beforehand would not see. `offload=None` and `offload='cpu'` keep them.
- **Each transcoder layer is one depth group**, from the last layer; a cross-layer transcoder is one group.

The transcoders are registered on the interpretune module once: under `model` when the replacement model is a torch
module, under the nnsight model's registered HF module (`hf_model`) when there is one, and as `transcoders` otherwise.
In that last case the base model's parameters are not the module's, so `CircuitTracerConfig.tune_base` refuses the
composition, naming why, rather than tuning nothing of the base.

With the transcoders registered as `transcoders`, every parameter of the module is a transcoder tensor, `instrument`
until a promotion, so there is nothing to optimize: a session whose configuration sets `optimizer_init` gets no
optimizer, with a warning, and analysis sessions run as before.

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
