"""Normalization + capture helpers for the loader-equivalence harness (hub design v3 §11.4).

The acceptance instrument for loader unification. The jsonargparse old path was REMOVED after 45/45
namespace-diff equivalence was proven against it; its final captures are frozen as
``tests/core/fixtures/cli_session_baseline_specs.json``, which the harness now holds the unified
loader to. Everything here works on config dataclasses only — deliberately never ``it_session`` —
so equivalence stays checkable on CPU in milliseconds.
"""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Any

from interpretune.session import ITSessionConfig

REPO_ROOT = Path(__file__).parent.parent.parent
CLI_EXPERIMENTS_DIR = REPO_ROOT / "src" / "it_examples" / "config" / "experiments"


def cli_experiment_configs() -> list[Path]:
    """Every CLI experiment config the harness must hold equivalent across the unification."""
    return sorted(CLI_EXPERIMENTS_DIR.rglob("*.yaml"))


def normalize(value: Any) -> Any:
    """Reduce a config object graph to a canonical, comparison- and diff-friendly structure.

    Classes/callables become fully-qualified names (so ``AutoCompConfig``-synthesized classes compare by
    identity-relevant name, not object id); dataclasses become ``{"__class__": fqname, **fields}``;
    tuples/sets/paths/enums/torch dtypes become plain JSON-able values.
    """
    import torch

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {"__class__": f"{type(value).__module__}.{type(value).__qualname__}"}
        for f in dataclasses.fields(value):
            out[f.name] = normalize(getattr(value, f.name))
        return out
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{getattr(value, '__module__', '?')}.{value.__qualname__}"
    if isinstance(value, enum.Enum):
        return f"{type(value).__qualname__}.{value.name}"
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(normalize(v) for v in value)
    return value


def session_spec(session_cfg: ITSessionConfig) -> dict:
    """The comparison payload: everything the loader must reproduce, nothing it may not."""
    return {
        "adapter_ctx": [a.name for a in session_cfg.adapter_ctx],
        "datamodule_cls": normalize(session_cfg.datamodule_cls),
        "module_cls": normalize(session_cfg.module_cls),
        "datamodule_cfg": normalize(session_cfg.datamodule_cfg),
        "module_cfg": normalize(session_cfg.module_cfg),
    }


EXAMPLES_DIR = REPO_ROOT / "src" / "it_examples" / "examples"


def example_configuration_files() -> list[Path]:
    """Every examples/ configuration indexed by a component manifest (the hub-side harness leg)."""
    from it_examples.example_module_registry import iter_component_manifests

    files = []
    for component_dir, manifest in iter_component_manifests(EXAMPLES_DIR):
        for rel in (manifest.get("module", {}).get("configs") or {}).values():
            files.append(component_dir / rel)
    return sorted(files)


def capture_registered_cfg_via_factories(config_path: Path):
    """Instantiate one examples/ configuration through the CURRENT factory path (the one merge site).

    This is the baseline the unified loader must match for hub-shaped bodies — including
    ``AutoCompConfig`` ``make_dataclass`` synthesis, which happens inside the config constructors the
    factories invoke and which none of the CLI configs exercises.
    """
    from interpretune.registry import ModuleRegistry
    from it_examples.example_module_registry import example_register_func, load_config_file

    registry = ModuleRegistry()
    key, body = load_config_file(config_path)
    example_register_func(registry)(key, body)
    return key, registry.get(key)


def registered_spec(registered_cfg) -> dict:
    """Comparison payload for the factory path (mirrors ``session_spec`` for hub-shaped bodies)."""
    return {
        "datamodule_cls": normalize(registered_cfg.datamodule_cls),
        "module_cls": normalize(registered_cfg.module_cls),
        "datamodule_cfg": normalize(registered_cfg.datamodule_cfg),
        "module_cfg": normalize(registered_cfg.module_cfg),
    }
