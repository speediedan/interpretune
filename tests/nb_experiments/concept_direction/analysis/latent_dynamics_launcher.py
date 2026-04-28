#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml  # type: ignore[import-untyped]


def _bootstrap_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "tests").exists():
            repo_root = candidate
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            return repo_root
    raise RuntimeError(f"Unable to locate repo root from {start}")


_bootstrap_repo_root(Path(__file__).resolve())

from tests.nb_experiments.config import load_experiment_config  # noqa: E402


NOTEBOOK_PATH = Path(__file__).resolve().with_name("concept_direction_latent_dynamics_analysis.ipynb")
DEFAULT_CONFIG_DIR = NOTEBOOK_PATH.parent / "configs"
DEFAULT_CONCEPT_DIRECTION_CONFIG_DIR = NOTEBOOK_PATH.parent.parent / "configs"
DEFAULT_OUTPUT_DIR = Path("/tmp/it_concept_direction_experiments/analysis")


@dataclass(frozen=True)
class LatentDynamicsExecutionPlan:
    config_path: Path
    resolved_config: dict[str, Any]
    output_path: Path
    source_config_path: Path
    resolved_config_path: Path
    parameters: dict[str, Any]
    timeout: int
    kernel_name: str | None
    prepare_only: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and execute the latent-dynamics notebook via papermill.",
    )
    parser.add_argument(
        "config",
        help="Config file path or config basename relative to the latent-dynamics configs directory.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help=(
            "Directory containing latent-dynamics configs. "
            "Defaults to tests/nb_experiments/concept_direction/analysis/configs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory where executed notebooks will be written. "
            "Defaults to /tmp/it_concept_direction_experiments/analysis."
        ),
    )
    parser.add_argument(
        "--projection-method",
        default="umap",
        help="Notebook PROJECTION_METHOD override. Defaults to umap.",
    )
    parser.add_argument(
        "--stage-top-n",
        type=int,
        help="Optional STAGE_TOP_N override passed to papermill.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Per-notebook timeout in seconds.",
    )
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help="Optional kernel override passed to papermill.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Render the parameterized notebook without executing it.",
    )
    return parser.parse_args(argv)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_config_path(raw_value: str, config_dir: Path) -> Path:
    raw_path = Path(raw_value).expanduser()
    candidates = [raw_path, config_dir / raw_value, DEFAULT_CONCEPT_DIRECTION_CONFIG_DIR / raw_value]
    if raw_path.suffix == "":
        candidates.extend(
            [
                config_dir / f"{raw_value}.yaml",
                config_dir / f"{raw_value}.yml",
                DEFAULT_CONCEPT_DIRECTION_CONFIG_DIR / f"{raw_value}.yaml",
                DEFAULT_CONCEPT_DIRECTION_CONFIG_DIR / f"{raw_value}.yml",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve config path from {raw_value!r}")


def build_papermill_parameters(
    config_path: Path,
    *,
    projection_method: str,
    stage_top_n: int | None,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "EXPERIMENT_CONFIG_PATH": str(config_path.resolve()),
        "PROJECTION_METHOD": projection_method,
    }
    if stage_top_n is not None:
        parameters["STAGE_TOP_N"] = int(stage_top_n)
    return parameters


def build_execution_plan(
    config_path: Path,
    *,
    output_dir: Path,
    projection_method: str,
    stage_top_n: int | None,
    timeout: int,
    kernel_name: str | None,
    prepare_only: bool,
) -> LatentDynamicsExecutionPlan:
    resolved_config = load_experiment_config(config_path)
    experiment_name = str(resolved_config.get("EXPERIMENT_NAME", config_path.stem))
    stamp = _timestamp()
    output_path = output_dir / f"{experiment_name}_{stamp}.ipynb"
    source_config_path = output_dir / f"{experiment_name}_{stamp}.source.yaml"
    resolved_config_path = output_dir / f"{experiment_name}_{stamp}.resolved.yaml"
    parameters = build_papermill_parameters(
        config_path,
        projection_method=projection_method,
        stage_top_n=stage_top_n,
    )
    return LatentDynamicsExecutionPlan(
        config_path=config_path,
        resolved_config=resolved_config,
        output_path=output_path,
        source_config_path=source_config_path,
        resolved_config_path=resolved_config_path,
        parameters=parameters,
        timeout=timeout,
        kernel_name=kernel_name,
        prepare_only=prepare_only,
    )


def execute_plan(plan: LatentDynamicsExecutionPlan) -> Path:
    import papermill as pm

    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running latent-dynamics config {plan.config_path.stem} -> {plan.output_path.name}")
    pm.execute_notebook(
        input_path=str(NOTEBOOK_PATH),
        output_path=str(plan.output_path),
        parameters=plan.parameters,
        timeout=plan.timeout,
        log_output=True,
        cwd=str(NOTEBOOK_PATH.parent),
        prepare_only=plan.prepare_only,
        kernel_name=plan.kernel_name,
    )
    shutil.copy2(plan.config_path, plan.source_config_path)
    plan.resolved_config_path.write_text(
        yaml.safe_dump(plan.resolved_config, sort_keys=False),
        encoding="utf-8",
    )
    return plan.output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_dir = Path(args.config_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config_path = resolve_config_path(str(args.config), config_dir)
    plan = build_execution_plan(
        config_path,
        output_dir=output_dir,
        projection_method=str(args.projection_method),
        stage_top_n=args.stage_top_n,
        timeout=int(args.timeout),
        kernel_name=args.kernel_name,
        prepare_only=bool(args.prepare_only),
    )
    execute_plan(plan)
    print(plan.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
