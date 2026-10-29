# Tuning: TransformerLens

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

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
  thawing an embedding thaws the final norm. Turned off (see [Choosing the naming](#choosing-the-naming)), norms thaw
  only where the schedule names them.
- **Parameters no name reaches.** A parameter no TransformerLens name reaches (a norm TransformerLens does not name,
  such as Gemma 2's `ln1_post`, or GPT-2's fused `qkv`) can be named in a schedule by its own (physical) name, which the
  view resolves to itself. A parameter a TransformerLens name reaches is scheduled by that name.
- **Live tensors.** `logical_named_parameters()` computes the TransformerLens tensors on each call, so they carry the
  parameters' current values.
- **Every traced name is scheduled.** The default schedule orders the unembedding, the blocks and the embeddings; a
  traced name outside those (one an architecture adds outside its blocks) goes in a final group of its own rather than
  being dropped. Norms are found anywhere in a block, including inside its attention.
- **Copies.** `copy.deepcopy` of a bridge does not preserve the fused-projection aliasing (the splits become independent
  copies), so build a view over the bridge a session actually trains.

## Schedules

A schedule for a TransformerLens composition names TransformerLens names, which the same schedule can use across
architectures:

```yaml
0:
  params:
  - blocks.(9|1[0-1]).*
  max_transition_epoch: 2
1:
  params:
  - blocks.([0-8](?!\d)).*
  - (pos_embed|embed|unembed).*
  lr: 1.0e-06
```

finetuning-scheduler reports what it thaws in physical parameters, so a GPT-2 block thawed by name counts 18: its 16
named tensors and the fused `qkv` weight and bias they are views of. Its norms are thawed with it.

## Choosing the naming

The module configuration's `tunable_view` (see [Choosing the view in configuration](fts_tunable_views.md)) selects either
alternative to the default, with no subclass.

To name the module's own parameters (`model.blocks.9._original_component.attn.q.<...>`), select finetuning-scheduler's
identity view:

```yaml
tunable_view:
  class_path: finetuning_scheduler.tunable_view.CanonicalView
```

To keep TransformerLens names but thaw only the norms a schedule names, turn the implicit norm thaw off:

```yaml
tunable_view:
  class_path: interpretune.adapters.transformer_lens.tunable_view.TransformerBridgeView
  init_args:
    implicit_norm_thaw: false
```

A schedule then names norms itself: `blocks.9.ln.*` for the norms TransformerLens names, `ln_final.*` for the final norm,
and the physical name for a norm it does not name. With the schedule above, phase 0 thaws 54 physical parameters with
the implicit thaw on; with it off, and phase 0 naming only `blocks.(9|1[0-1]).(attn|mlp).*`, it thaws 42 (each block's
attention and MLP tensors and the fused `qkv` weight and bias, without the 12 norm parameters).

The Lightning callback and `FTSCoreCallback` both discover the view the module carries, so no strategy adapter is
configured for either.

## Example: a steered, phased fine-tune

`rte_ft.qwen3.transformer_lens` in the `speediedan/rte` component is the [core and HF](fts_tuning_core_hf.md) example
through a `TransformerBridge`: the steering vector sits on the bridge's hook point after block 13
(`model.blocks.13.hook_out`), and the schedule names TransformerLens names:

```yaml
0:
  params:
  - role:intervention:steer
  max_transition_epoch: 2
1:
  params:
  - ln_final.w
  - blocks.2[4-7]\..*
  lr: 1.0e-05
2:
  params:
  - blocks.(1[4-9]|2[0-3])\..*
  lr: 1.0e-05
```

A block's norms thaw with it, Qwen3's `q_norm` and `k_norm` included, which TransformerLens does not name. Run it on
either interface:

```bash
interpretune --run_command train --config hub://speediedan/rte/runs/rte_ft.qwen3.transformer_lens.core.yaml
interpretune --lightning_cli fit --config hub://speediedan/rte/runs/rte_ft.qwen3.transformer_lens.lightning.yaml
```

The component's entrypoint is code from the Hub, so opt in to running it first (`IT_TRUST_REMOTE_CODE=1`; see
[the trust posture](hub_trust_posture.md)). Its losses match the HF run's, epoch for epoch: the view maps the
TransformerLens names to the same parameters the HF schedule names.
