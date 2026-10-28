# Tuning: sae_lens

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

SAEs spliced at init (`add_saes_on_init=True`) are registered as `attached_saes` and take the `latent:sae` role, so a
schedule can select them by role (`role:latent:sae`). Without splicing, no SAE parameter is the module's and nothing takes
a latent role.

An SAE configured `frozen` (`SAELensFromPretrainedConfig` or `SAELensCustomConfig`) is spliced and applied but takes the
`instrument` role instead: never thawed or checkpointed, unless a phase promotes it (`promote`, on the core loop).

## Example: steering beside a frozen transcoder

`rte_ft.qwen3.sae_lens` in the `speediedan/rte` component is the [TransformerLens](fts_tuning_transformer_lens.md)
example with a transcoder spliced in: the layer-13 transcoder of SAE Lens's `mwhanna-qwen3-0.6b-transcoders-lowl0`
release, at `blocks.13.mlp.hook_in`, with its reconstruction error added back (so the model computes what the
pretrained one does) and configured `frozen`:

```yaml
add_saes_on_init: true
sae_cfgs:
  - class_path: interpretune.adapters.sae_lens.config.SAELensFromPretrainedConfig
    init_args:
      release: mwhanna-qwen3-0.6b-transcoders-lowl0
      sae_id: layer_13
      use_error_term: true
      frozen: true
```

The transcoder is an `instrument`, so the TransformerLens example's schedule runs unchanged: the steering vector first,
then the top and middle blocks, and no phase can thaw the transcoder. Run it on either interface:

```bash
interpretune --run_command train --config hub://speediedan/rte/runs/rte_ft.qwen3.sae_lens.core.yaml
interpretune --lightning_cli fit --config hub://speediedan/rte/runs/rte_ft.qwen3.sae_lens.lightning.yaml
```

The component's entrypoint is code from the Hub, so opt in to running it first (`IT_TRUST_REMOTE_CODE=1`; see
[the trust posture](hub_trust_posture.md)). Training the transcoder too (promoting it to `latent:sae` in a later phase)
is the subject of the [latent models](fts_tuning_latent_models.md) work.
