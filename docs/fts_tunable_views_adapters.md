# Tuning: adapter-specific views

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view. This
page lists what each packaged adapter declares, and the details of the backends whose views are not finetuning-scheduler's
identity view.

## What each composition declares

| Composition | View | Roles |
| --- | --- | --- |
| core / HF, and any composition not overriding `build_tunable_view()` (the default) | finetuning-scheduler's identity view: trainable | PEFT adapter tensors as `adapter:<name>` |
| TransformerLens | `TransformerBridgeView`, in TransformerLens names (refused when the bridge holds quantized or packed parameters) | as below |
| sae_lens with `add_saes_on_init=True` | the backend's view | the spliced SAEs as `latent:sae` |
| nnsight, local and dispatched | finetuning-scheduler's identity view over the HF module, registered as `hf_model` | PEFT adapter tensors as `adapter:<name>` |
| nnsight, remote (NDIF) or not yet dispatched | refused, with the reason | |
| circuit-tracer | refused, with the reason | |
| a composition whose `build_tunable_view()` returns `None` | refused: never assumed trainable | |

## nnsight

A local, dispatched nnsight composition registers the nnsight model's HF module on the interpretune module as `hf_model`
(in `register_tunable_modules()`), so the optimizer, checkpoints and the identity view reach its parameters. A composition
executing remotely (NDIF) has no local weights to tune, and one built with `dispatch=False` keeps its weights on the
meta device; both are refused, each with its reason.

## sae_lens

SAEs spliced at init (`add_saes_on_init=True`) are registered as `attached_saes` and take the `latent:sae` role, so a
schedule can select them by role (`role:latent:sae`). Without splicing, no SAE parameter is the module's and nothing takes
a latent role.

## circuit-tracer

circuit-tracer compositions are refused: the replacement model is frozen at construction and may hold lazily loaded
transcoder weights that are not parameters. The refusal holds for every circuit-tracer composition, whichever backend it
runs over.

## TransformerLens

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
