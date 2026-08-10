"""Old-path capture + normalization for the 4a loader-equivalence harness (hub design v3 §11.4).

The acceptance instrument for loader unification: for every CLI experiment config, the session
configuration produced by the CURRENT jsonargparse path and by the unified declarative loader must be
namespace-identical before the old path is removed. This module implements the old-path capture and the
normalization; the comparison test lives in ``tests/core/test_loader_equivalence.py`` and gains its
new-path leg when ``load_session_cfg`` lands.

The capture instantiates ``session_cfg`` (the dataclasses, including any ``AutoCompConfig``
``make_dataclass`` synthesis their ``__post_init__`` performs) but deliberately NOT ``it_session`` —
session instantiation loads models, and config equivalence must be checkable on CPU in milliseconds.
"""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Any

from jsonargparse import ArgumentParser

from interpretune.config.shared import ITSharedConfig
from interpretune.session import ITSessionConfig

REPO_ROOT = Path(__file__).parent.parent.parent
CLI_EXPERIMENTS_DIR = REPO_ROOT / "src" / "it_examples" / "config" / "experiments"


def cli_experiment_configs() -> list[Path]:
    """Every CLI experiment config the harness must hold equivalent across the unification."""
    return sorted(CLI_EXPERIMENTS_DIR.rglob("*.yaml"))


def capture_session_cfg_via_cli(config_files: list[Path], defaults_dir: Path | None = None) -> ITSessionConfig:
    """Instantiate ``session_cfg`` exactly as the current CLI does, without touching ``it_session``.

    Mirrors ``ITSessionMixin.add_arguments_to_parser``'s session surface — ``add_class_arguments`` with
    ``sub_configs=True`` plus the per-field ``ITSharedConfig`` links — and ``core_cli_main``'s
    ``default_config_files`` composition, minus the ``ITSession``/runner registrations whose
    instantiation loads models.
    """
    import yaml as _yaml

    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(ITSessionConfig, "session_cfg", instantiate=True, sub_configs=True)
    for attr in ITSharedConfig.__dataclass_fields__:
        parser.link_arguments(
            f"session_cfg.datamodule_cfg.init_args.{attr}", f"session_cfg.module_cfg.init_args.{attr}"
        )
    # The harness's contract is the SESSION subtree — trainer/runner keys remain jsonargparse/Lightning
    # shim territory after unification, so they are out of comparison scope by design, not convenience.
    merged: dict[str, Any] = {}
    for cfg in [*(sorted(Path(defaults_dir).glob("*.yaml")) if defaults_dir else []), *config_files]:
        body = _yaml.safe_load(Path(cfg).read_text(encoding="utf-8")) or {}
        if "session_cfg" in body:
            merged.update({"session_cfg": {**merged.get("session_cfg", {}), **body["session_cfg"]}})
    config = parser.parse_object(merged)
    return parser.instantiate_classes(config).session_cfg


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
