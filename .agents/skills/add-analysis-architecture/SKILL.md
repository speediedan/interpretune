---
name: add-analysis-architecture
description: Add a model architecture to Interpretune's activation-point vocabulary after its venngram component map verifies. Use for a new supported architecture, including the model binding, TransformerLens mapping check, nnsight calibration, and CI registry pin.
---

# Add an Interpretune analysis architecture

Use the `venngram-component-map` skill to find or verify a registry map. Keep the candidate map, its verification
record and its configuration revision. The entry criteria are in `docs/activation_point_vocabulary.md`, under
“Entering and leaving the supported set.”

## Bind and compare against the backends

- Add the exact transformers class to `SUPPORTED_ARCHITECTURES` and a representative meta-device builder to
  `tiny_model` in `tests/core/mapped_architectures.py`. Use the model's actual layer layout. If optional rows vary
  with depth, exercise at least two layouts that differ in where the module exists.
- Bind the map with `bind_component_map` and resolve the model-specific inventory. Check concrete names against
  the modules that exist. An optional point should resolve on a covered layer and raise on an uncovered layer.
- Use `TestEverySupportedMapAgreesWithTransformerLens` in `tests/core/test_component_maps.py` to compare against
  TransformerLens's own adapter built from the tiny configuration. Record any counterpart the adapter genuinely
  lacks in `EXPECTED_WITHOUT_TL_COUNTERPART`, with a reason for each row.
- Exercise the nnsight resolver against a small real forward. Compare its point inventory and calibrated
  `tuple_output` values to direct hooks on the model. Follow `tests/core/test_nnsight_tuple_calibration.py`. Check
  that a planted wrong row causes binding to fail before tracing. Keep configuration-only map and alias checks
  on meta and use small CPU weights only for the checks that observe a forward.

## Pin the verified registry revision

After an authorized registry publication, update the registry revision in `tests/hf_warm_manifest.yaml` and warm
it in a fresh cache. In a clean process with offline mode enabled, run the tests below and check that the strict
lookup serves a verified row from the warmed index. Keep local map registrations out of this check so it fails
when the published row is missing.

```bash
HF_HUB_OFFLINE=1 python -m pytest -v \
  tests/core/test_component_maps.py \
  tests/core/test_activation_points.py \
  tests/core/test_adapters_transformer_lens.py \
  tests/core/test_nnsight_tuple_calibration.py
```

Report the verified registry revision, the tested model configuration and any differences between the backends.
