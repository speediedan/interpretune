# Tuning: circuit-tracer

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

circuit-tracer compositions are refused: the replacement model is frozen at construction and may hold lazily loaded
transcoder weights that are not parameters. The refusal holds for every circuit-tracer composition, whichever backend it
runs over.

The paths to circuit-tracer tuning are tracked in [interpretune#813](https://github.com/speediedan/interpretune/issues/813): transcoders as tunable latents, with published transcoders as
promotable instruments ([interpretune#787](https://github.com/speediedan/interpretune/issues/787)); adapting transcoders when fine-tuning makes them drift ([interpretune#709](https://github.com/speediedan/interpretune/issues/709)); tuning the base
model under a frozen replacement; and graph-guided fine-tuning ([interpretune#667](https://github.com/speediedan/interpretune/issues/667)).

Concrete example usage for this adapter is to come; it is tracked in [interpretune#812](https://github.com/speediedan/interpretune/issues/812).
