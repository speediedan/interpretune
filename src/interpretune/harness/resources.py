"""Small resource helpers shared by notebook experiment pipelines."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch


def create_work_root(base_dir: str | None, experiment_name: str, *, prefix: str = "nb_experiment") -> Path:
    """Create the experiment work root, namespaced by experiment name."""
    if base_dir:
        work_root = Path(base_dir).expanduser().resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        return work_root
    return Path(tempfile.mkdtemp(prefix=f"{prefix}_{experiment_name}_"))


def tensor_to_cpu(value: torch.Tensor) -> torch.Tensor:
    """Move a tensor to CPU, detaching it from any device context."""
    return value.detach().cpu().to(torch.float32)


def feature_ids_to_tuples(feature_ids: Any) -> list[tuple[int, ...]]:
    """Normalize feature ids into (layer, index) tuples."""
    return [tuple(feature.tolist()) for feature in feature_ids]


def scalar_tensor_list(values: list[float] | tuple[float, ...], *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a float tensor from a list of scalars."""
    return torch.tensor(list(values), dtype=dtype)


def phase_run_name(experiment_name: str, phase_name: str) -> str:
    """The run name for one pipeline phase of an experiment."""
    return f"{experiment_name}_{phase_name}"
