# Tuning: TransformerLens

[Tuning](tuning.md) describes how every composition declares and attaches its Fine-Tuning Scheduler view.

`TransformerBridgeView` names parameters as `tl_named_parameters()` does (`blocks.9.attn.W_Q`). Each name maps to the
parameters its TransformerLens tensor is computed from, found by walking that tensor's autograd graph, so the mapping
follows whatever TransformerLens emits with no table kept here. A name with no parameter behind it (a zero bias
TransformerLens synthesizes for a bias-free projection) is left out, and a schedule naming it is refused.

- **Built on first use.** Constructing the view traces nothing, so a session that never tunes pays nothing.
- **Fused projections.** GPT-2's `q`, `k` and `v` weights are strided views of the fused `qkv` weight. A step through a
  split changes the fused parameter's values, so the fused parameter is thawed with its splits (`coupled`), and the
  default schedule keeps each block whole. Fine-Tuning Scheduler refuses a schedule that thaws a block's `W_Q` and `W_K`
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

Fine-Tuning Scheduler reports what it thaws in physical parameters, so a GPT-2 block thawed by name counts 18: its 16
named tensors and the fused `qkv` weight and bias they are views of. Its norms are thawed with it.

## LoRA adapters

A TransformerLens composition takes PEFT LoRA adapters through the same setting as a core composition,
`hf_from_pretrained_cfg.lora_cfg`:

```yaml
hf_from_pretrained_cfg:
  lora_cfg:
    r: 8
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj"]
    lora_dropout: 0.05
```

- **Where the adapters go.** They are added to the HF model in place (`peft.inject_adapter_in_model`) before the
  `TransformerBridge` wraps it, so each targeted module keeps its path and the bridge runs its LoRA layer. The HF model
  is not wrapped in a `PeftModel`.
- **What trains.** Only the LoRA tensors require gradients after bridging. That includes freezing the parameters the
  bridge adds itself, such as the q, k and v it splits out of GPT-2's fused attention.
- **Roles.** The LoRA tensors take the `adapter:<name>` role (`adapter:default` unless the configuration names the
  adapter). No TransformerLens name reaches them, so a schedule selects them by role:

  ```yaml
  0:
    params:
    - role:adapter
  ```

  The default schedule orders the TransformerLens names, which are base weights, so a LoRA run gives its own schedule.
- **Targets the bridge never calls are refused.** Some bridge components compute from their own copy of a module's
  weights rather than calling the module. GPT-2's fused `c_attn` is one: the bridge splits it into its own q, k and v,
  so a LoRA layer on `c_attn` would never run. After bridging, one single-token forward checks that every LoRA layer
  ran, and the configuration is refused if any did not, naming the module. On GPT-2, target `c_proj` and `c_fc`.
  Where the projections are separate modules (Gemma 3's `q_proj`, `v_proj`, `o_proj` and `down_proj`, for example),
  the bridge calls them.
- **Not with quantization.** A `lora_cfg` together with a `bitsandbytesconfig` is refused at configuration:
  `TransformerBridgeView` refuses a bridge holding quantized parameters. Either setting alone is applied. For QLoRA, use
  a [core](tuning_core_hf.md) or Lightning composition.

## Choosing the naming

The module configuration's `tunable_view` (see [Choosing the view in configuration](tuning.md)) selects either
alternative to the default, with no subclass.

To name the module's own parameters (`model.blocks.9._original_component.attn.q.<...>`), select Fine-Tuning Scheduler's
identity view:

```yaml
tunable_view:
  class_path: venngram.tuning.CanonicalView
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

Concrete example usage for this adapter is to come; it is tracked in [interpretune#812](https://github.com/speediedan/interpretune/issues/812).
