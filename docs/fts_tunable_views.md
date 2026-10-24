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
| core / HF, and any composition not overriding `build_tunable_view()` (the default) | finetuning-scheduler's identity view: trainable | PEFT adapter tensors as `adapter:<name>` |
| TransformerLens | `TransformerBridgeView`, in TransformerLens names (refused when the bridge holds quantized or packed parameters) | as below |
| sae_lens with `add_saes_on_init=True` | the backend's view | the spliced SAEs as `latent:sae` |
| nnsight, local and dispatched | finetuning-scheduler's identity view over the HF module, registered as `hf_model` | PEFT adapter tensors as `adapter:<name>` |
| nnsight, remote (NDIF) or not yet dispatched | refused, with the reason | |
| circuit-tracer | refused, with the reason | |
| a composition whose `build_tunable_view()` returns `None` | refused: never assumed trainable | |

## Declaring a view

The base `build_tunable_view()` is a declaration in its own right: it returns the identity view, so **a composition
that does not override it is trainable**. A composition declares otherwise by overriding `build_tunable_view()`,
returning its own view or `None` (which declares nothing, so the composition is refused). Roles are added cooperatively through `_tunable_roles()`, which each mixin extends from
`super()`'s, so a backend's view and another mixin's roles compose:

```python
class MyInterventionMixin:
    def _tunable_roles(self) -> dict[str, str]:
        roles = super()._tunable_roles()
        roles.update({f"steer.{n}": "intervention:steer" for n, _ in self.steer.named_parameters()})
        return roles
```

PEFT adapter tensors are found through PEFT's tuner-layer API rather than by name, so every tuner whose layers hold their
adapters there (LoRA and its variants, IA3 and similar) is covered and the base weight PEFT renames (`base_layer.weight`)
stays `base`. Prompt-learning methods (prompt and prefix tuning) keep their tensors outside tuner layers; those are not
found and stay `base`.

The default optimizer (`optimizer_parameters()`) steps the attached trainable view's root parameters, less those whose
role finetuning-scheduler never optimizes (`fitted` and `instrument`). A module the view must reach that the model owns
is registered through the cooperative `_register_tunable_modules()`, which runs after whichever init path ran and before
the view is attached (nnsight registers its HF module there).

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
- **Live tensors.** `logical_named_parameters()` computes the TransformerLens tensors on each call, so they carry the
  parameters' current values.
- **Every traced name is scheduled.** The default schedule orders the unembedding, the blocks and the embeddings; a
  traced name outside those (one an architecture adds outside its blocks) goes in a final group of its own rather than
  being dropped. Norms are found anywhere in a block, including inside its attention.
- **Copies.** `copy.deepcopy` of a bridge does not preserve the fused-projection aliasing (the splits become independent
  copies), so build a view over the bridge a session actually trains.

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
