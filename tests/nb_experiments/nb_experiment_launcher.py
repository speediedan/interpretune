#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml  # type: ignore[import-untyped]

from tests.nb_experiments.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and execute parameterized experiment notebooks via papermill.",
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="Config file paths or config basenames relative to the configs directory.",
    )
    parser.add_argument(
        "--config-dir",
        help="Directory containing experiment configs. Defaults to '<notebook parent>/configs'.",
    )
    parser.add_argument(
        "--notebook",
        required=True,
        help="Notebook harness to execute.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory where executed notebooks will be written. Defaults to '<notebook parent>/generated_experiments'."
        ),
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
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining configs if one fails.",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run every YAML file under the config directory.",
    )
    parser.add_argument(
        "--config-pattern",
        help=(
            "Optional regular expression applied to config filenames. "
            "If provided without explicit configs, all configs are discovered first and then filtered."
        ),
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
    use_all_configs = args.all_configs or bool(args.config_pattern and not args.configs)
    if use_all_configs:
        config_paths = sorted(config_dir.glob("*.y*ml"))
    elif args.configs:
        config_paths = [_resolve_config_path(raw_value, config_dir) for raw_value in args.configs]
    else:
        raise SystemExit("No configs provided. Pass one or more config files or use --all-configs.")

    if args.config_pattern:
        pattern = re.compile(args.config_pattern)
        config_paths = [path for path in config_paths if pattern.search(path.name)]
    return config_paths


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

    resolved_config = load_experiment_config(config_path)
    config_name = config_path.stem
    stamp = _timestamp()
    output_path = output_dir / f"{config_name}_{stamp}.ipynb"
    source_config_path = output_dir / f"{config_name}_{stamp}.source.yaml"
    resolved_config_path = output_dir / f"{config_name}_{stamp}.resolved.yaml"

    parameters: dict[str, Any] = {
        "EXPERIMENT_CONFIG_PATH": str(config_path.resolve()),
        "EXPERIMENT_CONFIG_NAME": str(resolved_config.get("EXPERIMENT_CONFIG_NAME", config_name)),
        "EXPERIMENT_NAME": str(resolved_config.get("EXPERIMENT_NAME", config_name)),
    }

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
    shutil.copy2(config_path, source_config_path)
    resolved_config_path.write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    notebook_path = Path(args.notebook).resolve()
    config_dir = Path(args.config_dir).resolve() if args.config_dir else notebook_path.parent / "configs"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else notebook_path.parent / "generated_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_paths = discover_config_paths(args, config_dir)
    if not config_paths:
        raise SystemExit(f"No config files found in {config_dir}")

    outputs: list[Path] = []
    failures: list[tuple[Path, str]] = []
    for config_path in config_paths:
        try:
            output_path = execute_config(
                notebook_path,
                config_path,
                output_dir,
                timeout=args.timeout,
                kernel_name=args.kernel_name,
                prepare_only=args.prepare_only,
            )
            outputs.append(output_path)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((config_path, str(exc)))
            print(f"ERROR: Config {config_path.name} failed: {exc}")
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print("Completed notebook executions:")
    for output_path in outputs:
        print(f"  - {output_path}")
    if failures:
        print(f"\nFailed configs ({len(failures)}):")
        for config_path, err_msg in failures:
            print(f"  - {config_path.name}: {err_msg[:200]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
