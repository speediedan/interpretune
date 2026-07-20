# Shared Notebook Experiment Harness

This package holds experiment-agnostic notebook harness infrastructure used by parity notebooks and future notebook-based experiment workflows.

The shared harness owns:

- config loading with `EXTENDS` inheritance
- data-driven model/session registry config
- notebook bootstrap helpers
- the generic papermill launcher
- shared resource/session utilities that are not specific to one experiment family

## Core Files

- `config.py`: layered YAML loading plus shared config-section dataclasses
- `session.py`: model registry, session-surface preset handling, and generic `experiment_session(...)`
- `resource_utils.py`: small shared tensor/path helpers
- `notebook_bootstrap.py`: shared import bootstrap for notebooks
- `nb_experiment_launcher.py`: generic launcher that injects `EXPERIMENT_CONFIG_PATH` into notebooks
- `configs/base.yaml`: shared cross-experiment defaults
- `configs/model_specs.yaml`: model/session registry
- `configs/session_surface_presets.yaml`: reusable notebook debug-session presets

## Launcher Contract

The launcher no longer explodes every config value into individual papermill parameters. Instead it passes:

- `EXPERIMENT_CONFIG_PATH`
- `EXPERIMENT_CONFIG_NAME`
- `EXPERIMENT_NAME`

Experiment notebooks are responsible for resolving the config file and constructing their experiment-specific runtime config from it.

## Config Inheritance

Experiment configs can now use:

```yaml
EXTENDS:
  - ../../configs/base.yaml
  - ./base.yaml
```

Merge behavior is recursive for mappings and replace-on-write for scalars/lists.

## Local Graph Upload

The shared harness can now save attribution graphs under `<work_root>/graph_artifacts/` and upload them into a local
Neuronpedia instance when the config enables both localhost mode and graph upload.

Relevant `NEURONPEDIA` keys in `configs/base.yaml` or an experiment override:

- `use_localhost`: route uploads to the configured local Neuronpedia webapp instead of production
- `upload_local_graphs`: enable shared pipeline graph save/upload hooks
- `local_graph_slug_prefix`: optional slug prefix for easier cleanup and lookup
- `local_db_url`: local PostgreSQL URL used by cleanup and local explanation utilities
- `local_webapp_url`: local Neuronpedia webapp base URL used for upload routing and returned graph links

When `upload_local_graphs` is enabled, the shared pipeline helpers attach the returned graph artifact payload to each
pipeline result under `graph_artifact`.

## Cleanup Utility

Use the cleanup helper to remove locally saved graph JSON files, delete the matching `GraphMetadata` rows, and, when the
local webapp URL plus an API key are supplied, call the localhost graph delete route so the uploaded object-storage
payload is removed before the DB cleanup runs:

```bash
/mnt/cache/speediedan/.venvs/it_latest/bin/python -m tests.nb_experiments.local_graph_cleanup \
  --work-root /path/to/notebook/work_root \
  --local-db-url "$LOCAL_NEURONPEDIA_DB_URL" \
  --local-webapp-url "$LOCAL_NEURONPEDIA_WEBAPP_URL" \
  --local-api-key "$DEV_NEURONPEDIA_API_KEY" \
  --slug-prefix concept-direction-run
```

Add `--dry-run` to preview the matching slugs without deleting anything. Add `--keep-files` when you want to clear the
local Neuronpedia metadata and uploaded payloads but preserve the saved JSON files under `graph_artifacts/`.
