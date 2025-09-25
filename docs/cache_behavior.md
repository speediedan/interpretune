# Interpretune AnalysisStore caching

This document describes the current AnalysisStore caching behavior (MVP)
and the planned future behavior.

Current behavior (MVP)
- Interpretune ships a toggle `enable_analysisstore_caching()` which is
  disabled by default. Calling it with `True` will emit a warning that the
  feature is planned but not yet implemented for the MVP.
- By default, any temporary files created by AnalysisStore-related
  dataset generation are written to a temporary directory which is
  registered for deletion at process exit.
- This temporary directory behavior uses HuggingFace `datasets` helpers when
  available, and falls back to using the Python `tempfile` module with an
  `atexit` cleanup if not.

Planned future behavior
- When implemented, enabling Interpretune AnalysisStore caching will cause
  generated analysis artifacts (e.g. intermediate generator outputs) to be
  stored persistently under the base analysis cache path used by
  Interpretune: `~/.cache/huggingface/interpretune/<dataset_config>/<dataset_fingerprint>/<module>/...`.
- Persistent AnalysisStore caching will only be available when HF
  `datasets` caching is enabled (via `datasets.enable_caching()`) because
  the implementation is planned to leverage dataset-level fingerprinting
  and cache management.

How to opt-in (future)
- Call `interpretune.analysis.cache.enable_analysisstore_caching(True)`.
  For the MVP this only emits a warning; when implemented it will enable
  persistent caching provided the HF datasets caching is also enabled.

Post-MVP hashing and determinism considerations
- Post-MVP, we plan to inspect the viability of generator-driven
  AnalysisStore dataset cache determinism by implementing a custom
  AnalysisStore dataset hashing function. This custom hash will account
  for the relevant `it_session` and generator transformations so that
  caching can be applied to AnalysisStore operations for a given dataset,
  module, and `it_session` state.
- Currently, without these customizations, generated AnalysisStore
  datasets created by manual or generator-driven closures include
  `it_session` and other runtime variables that lead to inconsistent
  hashing across runs. Implementing a custom hash function that focuses
  on the semantically relevant variables from `it_session` and the
  generator would enable reliable caching in the future.

Example test generator-driven dataset creation:

```python
# interpretune/tests/orchestration.py
    dataset = Dataset.from_generator(
        generator=multi_batch_generator,
        features=features,
        cache_dir=it_session.module.analysis_cfg.output_store.cache_dir,
        split="test",
    ).with_format("interpretune", **it_format_kwargs)
```

Relevant hashing internals (HuggingFace `datasets` BuilderConfig)
- The default `BuilderConfig` name is `"default"` and is defined in
  `datasets/builder.py`.
- When creating a dataset from a generator the builder uses the
  `generator` builder name which results in a `config_id` computed by
  `create_config_id` (see the stack below):

```text
create_config_id (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/datasets/builder.py:196)
_create_builder_config (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/datasets/builder.py:573)
__init__ (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/datasets/builder.py:343)
__init__ (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/datasets/io/generator.py:29)
from_generator (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/datasets/arrow_dataset.py:1178)
save_reload_results_dataset (/home/speediedan/repos/interpretune/tests/orchestration.py:264)
_op_serialization_fixt (/home/speediedan/repos/interpretune/tests/core/conftest.py:86)
test_op_serialization (/home/speediedan/repos/interpretune/tests/core/test_analysis_ops_definitions.py:1178)
```

Concrete hashing example

```python
# config_kwargs_to_add_to_suffix: {'features': {'answer_logits': Array3D(shape=(None, 2, 50257), dtype='float32'), 'prompts': List(Value('string')), 'answer_indices': List(Value('int64')), 'tokens': Array2D(shape=(None, 2), dtype='int64')}, 'gen_kwargs': None, 'generator': <function save_reload_results_dataset.<locals>.multi_batch_generator at 0x7fefcc3136a0>, 'split': 'test'}

# custom_features:  {'answer_logits': Array3D(shape=(None, 2, 50257), dtype='float32'), 'prompts': List(Value('string')), 'answer_indices': List(Value('int64')), 'tokens': Array2D(shape=(None, 2), dtype='int64')}

test_suffix = Hasher.hash(config_kwargs_to_add_to_suffix)  # '126cf650d2c7878a'
# which is updated to this hash with custom_features
m = Hasher()
m.update(test_suffix)
m.update(custom_features)
test_suffix = m.hexdigest()  # '439222ea101f8166'
```

- If the `generator` key is removed from the config the hash becomes
  consistent across runs:

```python
config_kwargs_to_add_to_suffix.pop('generator')
config_kwargs_to_add_to_suffix = {k: config_kwargs_to_add_to_suffix[k] for k in sorted(config_kwargs_to_add_to_suffix)}
test_suffix = Hasher.hash(config_kwargs_to_add_to_suffix)
m = Hasher()
m.update(test_suffix)
m.update(custom_features)
test_suffix = m.hexdigest()
# test_suffix is always '7814d4bb675529bb' given inputs above after popping generator
```
