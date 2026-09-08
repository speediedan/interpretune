# Shared Notebook Experiment Harness

This package holds experiment-agnostic notebook harness infrastructure used by parity notebooks and future notebook-based experiment workflows.

The shared harness owns:

- config loading with `EXTENDS` inheritance
- data-driven model/session registry config
- notebook bootstrap helpers
- the generic papermill launcher
- shared resource/session utilities that are not specific to one experiment family

## Running an experiment from outside this repository

The rails live in the core package (`interpretune.utils.notebook_experiments`), so an experiment in
another repository consumes them rather than copying them. Configure it with a table in that
repository's `pyproject.toml`; every key has a default, so a repository that writes no table still
works:

```toml
[tool.interpretune.experiments]
config_dir    = "my_configs"   # default: the directory beside the notebook
output_root   = "artifacts"    # default: <notebook dir>/generated_experiments
harness_paths = ["shared"]     # extra sys.path entries, relative to the repository root
```

**Extending the shared base configs from outside this tree.** `EXTENDS` accepts a
`package.module:resource` form resolved through `importlib.resources`, so a config anywhere can name a
base that ships inside an installed package:

```yaml
EXTENDS: it_examples.experiments.notebook:configs/base.yaml
```

Relative and absolute paths behave exactly as before. A package-resource base is **read-only**, since
resources may live inside a wheel and have no stable location on disk; a config that expects to write
next to its base must name it by path.

**Experiment-specific helpers are supplied, not imported.** The shared harness takes the callables it
needs through `ExperimentHooks`, which the experiment registers (see `concept_direction/__init__.py`).
The harness previously imported them from one particular experiment, which meant importing the shared
rails required that experiment to be installed.

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

Experiment configs can now use a scalar or a list (merged in order), mirroring real usage under
`concept_direction/configs/`:

```yaml
# single parent (sibling file in the same configs/ dir)
EXTENDS: base_gemma3_it_local.yaml

# or multiple parents, e.g. shared cross-experiment defaults + an experiment-family base
EXTENDS:
  - ../../configs/base.yaml
  - base_gemma3_it_local.yaml
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
python -m it_examples.experiments.notebook.local_graph_cleanup \
  --work-root /path/to/notebook/work_root \
  --local-db-url "$LOCAL_NEURONPEDIA_DB_URL" \
  --local-webapp-url "$LOCAL_NEURONPEDIA_WEBAPP_URL" \
  --local-api-key "$DEV_NEURONPEDIA_API_KEY" \
  --slug-prefix concept-direction-run
```

Add `--dry-run` to preview the matching slugs without deleting anything. Add `--keep-files` when you want to clear the
local Neuronpedia metadata and uploaded payloads but preserve the saved JSON files under `graph_artifacts/`.
