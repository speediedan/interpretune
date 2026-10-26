# Tuning: sae_lens

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

SAEs spliced at init (`add_saes_on_init=True`) are registered as `attached_saes` and take the `latent:sae` role, so a
schedule can select them by role (`role:latent:sae`). Without splicing, no SAE parameter is the module's and nothing takes
a latent role.

Concrete example usage for this adapter is to come; it is tracked in [interpretune#812](https://github.com/speediedan/interpretune/issues/812).
