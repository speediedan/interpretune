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
  - ../../nb_experiment_harness/configs/base.yaml
  - ./base.yaml
```

Merge behavior is recursive for mappings and replace-on-write for scalars/lists.
