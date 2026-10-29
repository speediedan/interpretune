# Tuning: core and HF

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

The core (HF) composition, and any composition that does not override `build_tunable_view()`, takes
finetuning-scheduler's identity view over the interpretune module: it is trainable, and its physical names are the
module's own (`model.<...>`). PEFT adapter tensors take the `adapter:<name>` role, found through PEFT's tuner-layer API
(see [declaring a view](fts_tunable_views.md#declaring-a-view)); every other parameter is `base` unless
`tunable_roles()` says otherwise.

## Example: a steered, phased fine-tune

The `speediedan/rte` component ships `rte_ft.qwen3.core`, Qwen3-0.6B on RTE in three phases: a steering vector after
decoder layer 13 (declared in the configuration as `tuning_interventions`, see
[interventions](fts_tuning_interventions.md)) trains alone, then the top four layers and the final norm thaw, then
layers 14 to 23, each phase advancing on a validation-loss plateau. Run it on either interface:

```bash
interpretune --run_command train --config hub://speediedan/rte/runs/rte_ft.qwen3.core.yaml
interpretune --lightning_cli fit --config hub://speediedan/rte/runs/rte_ft.qwen3.lightning.yaml
```

The component's entrypoint is code from the Hub, so opt in to running it first (`IT_TRUST_REMOTE_CODE=1`; see
[the trust posture](hub_trust_posture.md)). The schedule, `ft_schedules/rte_ft.qwen3.core.yaml`, names the module's own
parameters:

```yaml
0:
  params:
  - role:intervention:steer
  max_transition_epoch: 2
1:
  params:
  - model.model.norm.weight
  - model.model.layers.2[4-7]\..*
  lr: 1.0e-05
2:
  params:
  - model.model.layers.(1[4-9]|2[0-3])\..*
  lr: 1.0e-05
```

On one 24 GB GPU in fp32, the steering vector alone brings the validation loss to 0.683 in two epochs, and the block
phases bring it to 0.489 (accuracy 0.78), in about five minutes. The same run through
[TransformerLens](fts_tuning_transformer_lens.md) and [nnsight](fts_tuning_nnsight.md) gives the same numbers.
