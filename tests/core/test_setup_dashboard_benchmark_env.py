"""The benchmark env setup builds a separate transformer-lens 3.x venv for the preserved-baseline legs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_setup_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "setup_dashboard_benchmark_env", REPO_ROOT / "scripts" / "setup_dashboard_benchmark_env.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _setup(module: ModuleType, tmp_path: Path, *extra: str):
    args = module.build_parser().parse_args(["--log-dir", str(tmp_path / "logs"), "--yes", "--dry-run", *extra])
    setup = module.Setup(args)
    setup.repo_paths = {"interpretune": tmp_path / "interpretune", "sae_dashboard": tmp_path / "SAEDashboard"}
    return setup


def test_build_command_targets_the_requested_venv(tmp_path: Path) -> None:
    module = _load_setup_module()
    cmd = _setup(module, tmp_path)._build_it_env_cmd("it_bench_baseline", tmp_path / "venvs")
    assert "--target-env-name=it_bench_baseline" in cmd
    assert f"--venv-dir={tmp_path / 'venvs'}" in cmd


def test_baseline_venv_is_planned_with_a_config_free_transformer_lens_3_pin(tmp_path: Path) -> None:
    """Dry run: nothing executes, and the pin bypasses the project's own transformer-lens override."""
    module = _load_setup_module()
    setup = _setup(module, tmp_path)
    venv_path = tmp_path / "venvs" / "it_bench"

    baseline = setup.build_baseline_env(venv_path)

    assert baseline == tmp_path / "venvs" / "it_bench_baseline"
    assert setup.actions_taken == [], "a dry run must not record executed actions"
    assert module.BASELINE_TRANSFORMER_LENS.split(".")[0] == "3"
    log = setup.log_path.read_text(encoding="utf-8")
    assert (
        f"--no-config --python {baseline / 'bin' / 'python'} transformer-lens=={module.BASELINE_TRANSFORMER_LENS}"
        in log
    )
