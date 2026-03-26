#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_NOTEBOOK = ROOT / "concept_direction_experiment_harness.ipynb"
DEFAULT_CONFIG_DIR = ROOT / "configs"
DEFAULT_OUTPUT_DIR = ROOT / "generated_experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and execute parameterized concept-direction notebooks via papermill.",
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="Config file paths or config basenames relative to the configs directory.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing flat YAML experiment configs.",
    )
    parser.add_argument(
        "--notebook",
        default=str(DEFAULT_NOTEBOOK),
        help="Notebook harness to execute.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where executed notebooks will be written.",
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
        help="Render parameterized notebooks without executing them.",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run every YAML file under the config directory.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_config_path(raw_value: str, config_dir: Path) -> Path:
    raw_path = Path(raw_value)
    candidates = [raw_path, config_dir / raw_value]
    if raw_path.suffix == "":
        candidates.extend([config_dir / f"{raw_value}.yaml", config_dir / f"{raw_value}.yml"])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve config path from '{raw_value}'")


def discover_config_paths(args: argparse.Namespace, config_dir: Path) -> list[Path]:
    if args.all_configs:
        return sorted(config_dir.glob("*.y*ml"))
    if not args.configs:
        raise SystemExit("No configs provided. Pass one or more config files or use --all-configs.")
    return [_resolve_config_path(raw_value, config_dir) for raw_value in args.configs]


def load_flat_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} must parse to a mapping.")
    for key, value in payload.items():
        if isinstance(value, dict):
            raise ValueError(f"Config {path} is not flat: key '{key}' contains a nested mapping.")
    return payload


def execute_config(
    notebook_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    timeout: int,
    kernel_name: str | None,
    prepare_only: bool,
) -> Path:
    import papermill as pm

    parameters = load_flat_yaml(config_path)
    config_name = config_path.stem
    stamp = _timestamp()
    output_path = output_dir / f"{config_name}_{stamp}.ipynb"
    archived_config = output_dir / f"{config_name}_{stamp}.yaml"

    parameters.setdefault("EXPERIMENT_CONFIG_NAME", config_name)
    parameters.setdefault("EXPERIMENT_NAME", config_name)

    print(f"Running config {config_name} -> {output_path.name}")
    pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        parameters=parameters,
        timeout=timeout,
        log_output=True,
        cwd=str(notebook_path.parent),
        prepare_only=prepare_only,
        kernel_name=kernel_name,
    )
    shutil.copy2(config_path, archived_config)
    return output_path


def main() -> int:
    args = parse_args()
    config_dir = Path(args.config_dir).resolve()
    notebook_path = Path(args.notebook).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_paths = discover_config_paths(args, config_dir)
    if not config_paths:
        raise SystemExit(f"No config files found in {config_dir}")

    outputs: list[Path] = []
    for config_path in config_paths:
        output_path = execute_config(
            notebook_path,
            config_path,
            output_dir,
            timeout=args.timeout,
            kernel_name=args.kernel_name,
            prepare_only=args.prepare_only,
        )
        outputs.append(output_path)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print("Completed notebook executions:")
    for output_path in outputs:
        print(f"  - {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())