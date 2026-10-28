# Tuning: nnsight

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

A local, dispatched nnsight composition registers the nnsight model's HF module on the interpretune module as `hf_model`
(in `register_tunable_modules()`), so the optimizer, checkpoints and the identity view reach its parameters. A composition
executing remotely (NDIF) has no local weights to tune, and one built with `dispatch=False` keeps its weights on the
meta device; both are refused, each with its reason.

The paths to fuller nnsight tuning, local and remote, are tracked in [interpretune#814](https://github.com/speediedan/interpretune/issues/814). Locally, that covers tuning applied
through nnsight's own tracing, and the remaining local cases. Remotely, it covers work that does not need local weights:
remote models accept persistent changes to values, and ephemeral techniques that tune within a single dispatch may need
no client-side optimizer state at all. The detailed remote design is [interpretune#775](https://github.com/speediedan/interpretune/issues/775).

## Example: a steered, phased fine-tune

`rte_ft.qwen3.nnsight` in the `speediedan/rte` component is the [core and HF](fts_tuning_core_hf.md) example through a
local, dispatched `LanguageModel`. The HF module nnsight wraps is registered as `hf_model`, so the steering vector sits
on `hf_model.model.layers.13`, and the schedule names the HF module's parameters under `hf_model.` (the identity view's
names):

```yaml
1:
  params:
  - hf_model.model.norm.weight
  - hf_model.model.layers.2[4-7]\..*
  lr: 1.0e-05
```

Run it on either interface:

```bash
interpretune --run_command train --config hub://speediedan/rte/runs/rte_ft.qwen3.nnsight.core.yaml
interpretune --lightning_cli fit --config hub://speediedan/rte/runs/rte_ft.qwen3.nnsight.lightning.yaml
```

The component's entrypoint is code from the Hub, so opt in to running it first (`IT_TRUST_REMOTE_CODE=1`; see
[the trust posture](hub_trust_posture.md)). Its losses match the HF run's, epoch for epoch.
