# Tuning

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

Which view each composition declares is summarized [at the bottom of this page](#what-each-composition-declares), and
each packaged adapter has its own tuning page with the details of its view.

## Declaring a view

The base `build_tunable_view()` is a declaration in its own right: it returns the identity view, so **a composition
that does not override it is trainable**. A composition declares otherwise by overriding `build_tunable_view()`,
returning its own view or `None` (which declares nothing, so the composition is refused).

Two public hooks extend a composition's view without replacing it. Both are cooperative: an override calls `super()`
and adds to its result, so a backend's view and another mixin's additions compose.

- **`tunable_roles()`** returns roles for the module's parameters and buffers, by name (anything unlisted is `base`).
  Declare an intervention, a run-fitted tensor or a latent dictionary here:

```python
class MyInterventionMixin:
    def tunable_roles(self) -> dict[str, str]:
        roles = super().tunable_roles()
        roles.update({f"steer.{n}": "intervention:steer" for n, _ in self.steer.named_parameters()})
        return roles
```

- **`register_tunable_modules()`** registers on the module any model-owned module the view must reach. It runs after
  whichever init path ran (`post_auto_model_init` or `model_init`) and before the view is attached, so overriding a
  template hook without `super()` never loses a registration.

PEFT adapter tensors are found through PEFT's tuner-layer API rather than by name, so every tuner whose layers hold their
adapters there (LoRA and its variants, IA3 and similar) is covered and the base weight PEFT renames (`base_layer.weight`)
stays `base`. Prompt-learning methods (prompt and prefix tuning) keep their tensors outside tuner layers; those are not
found and stay `base`.

The default optimizer (`optimizer_parameters()`) steps the attached trainable view's root parameters, less those whose
role finetuning-scheduler never optimizes (`fitted` and `instrument`).

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

## What each composition declares

| Composition | View | Roles |
| --- | --- | --- |
| [core / HF](fts_tuning_core_hf.md), and any composition not overriding `build_tunable_view()` (the default) | finetuning-scheduler's identity view: trainable | PEFT adapter tensors as `adapter:<name>` |
| [TransformerLens](fts_tuning_transformer_lens.md) | `TransformerBridgeView`, in TransformerLens names (refused when the bridge holds quantized or packed parameters) | as on its page |
| [sae_lens](fts_tuning_sae_lens.md) with `add_saes_on_init=True` | the backend's view | the spliced SAEs as `latent:sae` |
| [nnsight](fts_tuning_nnsight.md), local and dispatched | finetuning-scheduler's identity view over the HF module, registered as `hf_model` | PEFT adapter tensors as `adapter:<name>` |
| [nnsight](fts_tuning_nnsight.md), remote (NDIF) or not yet dispatched | refused, with the reason; tuning paths tracked in [#814](https://github.com/speediedan/interpretune/issues/814) and [#775](https://github.com/speediedan/interpretune/issues/775) | |
| [circuit-tracer](fts_tuning_circuit_tracer.md) | refused, with the reason; tuning paths tracked in [#813](https://github.com/speediedan/interpretune/issues/813), [#787](https://github.com/speediedan/interpretune/issues/787) and [#709](https://github.com/speediedan/interpretune/issues/709) | |
| [interp-engine (hub adapter)](fts_tuning_interp_engine.md) | interpretune's default identity view until the adapter declares its own; its circuit-tracer composition takes the circuit-tracer refusal | none declared yet |
| a composition whose `build_tunable_view()` returns `None` | refused: never assumed trainable | |
