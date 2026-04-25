#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml  # type: ignore[import-untyped]


def _bootstrap_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "tests").exists():
            repo_root = candidate
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            return repo_root
    raise RuntimeError(f"Unable to locate repo root from {start}")


_BOOTSTRAP_REPO_ROOT = _bootstrap_repo_root(Path(__file__).resolve())

from tests.nb_experiments.concept_direction.analysis.concept_direction_analysis import (  # noqa: E402
    ARTIFACT_GENERATION_ENV,
    PRESERVE_ARTIFACT_DIR_ENV,
    PRESERVE_ARTIFACTS_ENV,
    DEFAULT_CONCEPT_DIRECTION_ARTIFACT_ROOT,
)


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "tests").exists():
            return candidate
    raise RuntimeError(f"Unable to locate repo root from {start}")


REPO_ROOT = _BOOTSTRAP_REPO_ROOT
ANALYSIS_DIR = Path(__file__).resolve().parent
CONCEPT_DIRECTION_DIR = ANALYSIS_DIR.parent
DEFAULT_NOTEBOOK_PATH = CONCEPT_DIRECTION_DIR / "concept_direction_template.ipynb"
DEFAULT_CONFIG_DIR = CONCEPT_DIRECTION_DIR / "configs"
DEFAULT_NOTEBOOK_OUTPUT_DIR = CONCEPT_DIRECTION_DIR / "generated_experiments"
DEFAULT_ANALYSIS_NOTEBOOK = ANALYSIS_DIR / "concept_direction_analysis.ipynb"
DEFAULT_ANALYSIS_OUTPUT_DIR = ANALYSIS_DIR / "generated_experiments"
DEFAULT_EXPERIMENT_SET_PATH = ANALYSIS_DIR / "default_analysis_experiment_set.yaml"
DEFAULT_NB_LAUNCHER = REPO_ROOT / "tests" / "nb_experiments" / "nb_experiment_launcher.py"


def build_generation_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class AnalysisExperimentSet:
    reference_tests: tuple[str, ...]
    notebook_configs: tuple[str, ...]
    notebook_path: Path
    config_dir: Path
    notebook_output_dir: Path
    analysis_notebook: Path
    analysis_output_dir: Path
    analysis_output_stem: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate concept-direction reference reports and notebook artifacts, then execute the analysis notebook."
        ),
    )
    parser.add_argument(
        "--experiment-set",
        default=str(DEFAULT_EXPERIMENT_SET_PATH),
        help="Path to the analysis experiment-set manifest.",
    )
    parser.add_argument(
        "--artifact-root",
        default=DEFAULT_CONCEPT_DIRECTION_ARTIFACT_ROOT,
        help="Artifact root used for preserved reports and notebook pipeline state payloads.",
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Reuse the current artifact root instead of archiving it and regenerating reports.",
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
        help="Kernel name passed through to papermill and nbconvert execution.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Render notebook experiments without executing them. The analysis notebook still executes unless skipped.",
    )
    parser.add_argument(
        "--skip-analysis-notebook",
        action="store_true",
        help="Regenerate reports and notebook artifacts without executing the final analysis notebook.",
    )
    return parser.parse_args(argv)


def _resolve_manifest_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def load_experiment_set(path: str | Path) -> AnalysisExperimentSet:
    manifest_path = _resolve_manifest_path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment-set manifest must parse to a mapping: {manifest_path}")

    def _resolve_path(key: str, default: Path) -> Path:
        raw_value = payload.get(key)
        if raw_value is None:
            return default
        path = Path(str(raw_value)).expanduser()
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        return path

    reference_tests = tuple(str(item) for item in payload.get("reference_tests", ()))
    notebook_configs = tuple(str(item) for item in payload.get("notebook_configs", ()))
    if not reference_tests:
        raise ValueError(f"Experiment-set manifest must define at least one reference test: {manifest_path}")
    if not notebook_configs:
        raise ValueError(f"Experiment-set manifest must define at least one notebook config: {manifest_path}")

    return AnalysisExperimentSet(
        reference_tests=reference_tests,
        notebook_configs=notebook_configs,
        notebook_path=_resolve_path("notebook_path", DEFAULT_NOTEBOOK_PATH),
        config_dir=_resolve_path("config_dir", DEFAULT_CONFIG_DIR),
        notebook_output_dir=_resolve_path("notebook_output_dir", DEFAULT_NOTEBOOK_OUTPUT_DIR),
        analysis_notebook=_resolve_path("analysis_notebook", DEFAULT_ANALYSIS_NOTEBOOK),
        analysis_output_dir=_resolve_path("analysis_output_dir", DEFAULT_ANALYSIS_OUTPUT_DIR),
        analysis_output_stem=str(payload.get("analysis_output_stem", DEFAULT_ANALYSIS_NOTEBOOK.stem)),
    )


def archive_existing_artifact_root(artifact_root: Path, *, generation: str) -> Path | None:
    if not artifact_root.exists() or not any(artifact_root.iterdir()):
        return None
    archive_path = artifact_root.with_name(f"{artifact_root.name}_{generation}")
    artifact_root.rename(archive_path)
    return archive_path


def build_run_environment(artifact_root: Path, *, generation: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env[PRESERVE_ARTIFACTS_ENV] = "1"
    env[PRESERVE_ARTIFACT_DIR_ENV] = str(artifact_root)
    env["IT_RUN_OPTIONAL_TESTS"] = "1"
    if generation is None:
        env.pop(ARTIFACT_GENERATION_ENV, None)
    else:
        env[ARTIFACT_GENERATION_ENV] = generation
    return env


def _run_subprocess(command: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)


def run_reference_tests(experiment_set: AnalysisExperimentSet, *, env: dict[str, str]) -> None:
    command = [sys.executable, "-m", "pytest", "-v", "-ra", *experiment_set.reference_tests]
    _run_subprocess(command, env=env)


def run_notebook_experiments(
    experiment_set: AnalysisExperimentSet,
    *,
    env: dict[str, str],
    timeout: int,
    kernel_name: str,
    prepare_only: bool,
) -> None:
    command = [
        sys.executable,
        str(DEFAULT_NB_LAUNCHER),
        "--notebook",
        str(experiment_set.notebook_path),
        "--config-dir",
        str(experiment_set.config_dir),
        "--output-dir",
        str(experiment_set.notebook_output_dir),
        "--timeout",
        str(timeout),
        "--kernel-name",
        kernel_name,
        *experiment_set.notebook_configs,
    ]
    if prepare_only:
        command.append("--prepare-only")
    _run_subprocess(command, env=env)


def run_analysis_notebook(
    experiment_set: AnalysisExperimentSet,
    *,
    env: dict[str, str],
    generation: str | None,
    timeout: int,
    kernel_name: str,
) -> Path:
    experiment_set.analysis_output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = experiment_set.analysis_output_stem
    if generation is not None:
        output_stem = f"{output_stem}_{generation}"
    command = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout",
        str(timeout),
        "--ExecutePreprocessor.kernel_name",
        kernel_name,
        "--output",
        f"{output_stem}.ipynb",
        "--output-dir",
        str(experiment_set.analysis_output_dir),
        str(experiment_set.analysis_notebook),
    ]
    _run_subprocess(command, env=env)
    return experiment_set.analysis_output_dir / f"{output_stem}.ipynb"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_set = load_experiment_set(args.experiment_set)
    artifact_root = Path(args.artifact_root).expanduser().resolve()

    generation = None if args.use_existing else build_generation_token()
    if args.use_existing:
        if not artifact_root.exists():
            raise FileNotFoundError(f"Artifact root does not exist for --use-existing: {artifact_root}")
    else:
        assert generation is not None
        archived_path = archive_existing_artifact_root(artifact_root, generation=generation)
        if archived_path is not None:
            print(f"Archived existing artifact root to {archived_path}")
        artifact_root.mkdir(parents=True, exist_ok=True)

    env = build_run_environment(artifact_root, generation=generation)

    if not args.use_existing:
        run_reference_tests(experiment_set, env=env)
        run_notebook_experiments(
            experiment_set,
            env=env,
            timeout=int(args.timeout),
            kernel_name=str(args.kernel_name),
            prepare_only=bool(args.prepare_only),
        )

    if args.skip_analysis_notebook:
        return 0

    output_path = run_analysis_notebook(
        experiment_set,
        env=env,
        generation=generation,
        timeout=int(args.timeout),
        kernel_name=str(args.kernel_name),
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
