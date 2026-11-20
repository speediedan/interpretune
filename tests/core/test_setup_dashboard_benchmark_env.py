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


def _baseline_tree_needing(tmp_path: Path, dist: str) -> tuple[Path, Path]:
    """A preserved-baseline worktree whose poetry deps declare ``dist``, and a venv whose interpreter lacks it."""
    tree = tmp_path / "baseline" / "SAEDashboard-7886eaa"
    tree.mkdir(parents=True)
    (tree / "pyproject.toml").write_text(
        f'[tool.poetry.dependencies]\npython = "^3.10"\n"{dist}" = "^3.10.0"\n', encoding="utf-8"
    )
    venv = tmp_path / "venvs" / "it_bench_baseline"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    return tree.parent, venv


def test_a_venv_not_built_here_is_told_which_baseline_dependencies_it_lacks(tmp_path: Path) -> None:
    """Under --skip-env-build nothing is installed, but a baseline dependency absent from the venv is named, so the
    detached-legacy legs do not fail on it later with nothing pointing back at the setup."""
    module = _load_setup_module()
    dist = "it-no-such-baseline-dist"
    worktrees, venv = _baseline_tree_needing(tmp_path, dist)
    args = module.build_parser().parse_args(
        ["--log-dir", str(tmp_path / "logs"), "--yes", "--skip-env-build", "--worktrees-dir", str(worktrees)]
    )
    setup = module.Setup(args)
    setup.repo_paths = {"interpretune": tmp_path / "interpretune"}

    setup.build_baseline_env(venv.parent / "it_bench")

    assert setup.actions_taken == []
    assert any(dist in w and "ModuleNotFoundError" in w for w in setup.warnings), setup.warnings


def test_baseline_dependencies_are_installed_with_uv_not_pip(tmp_path: Path, monkeypatch) -> None:
    """The benchmark venvs are uv-built; a pip install there would leave a second owner for the package."""
    module = _load_setup_module()
    dist = "it-no-such-baseline-dist"
    worktrees, venv = _baseline_tree_needing(tmp_path, dist)
    args = module.build_parser().parse_args(
        ["--log-dir", str(tmp_path / "logs"), "--yes", "--worktrees-dir", str(worktrees)]
    )
    setup = module.Setup(args)
    setup.repo_paths = {"interpretune": tmp_path / "interpretune"}
    real_run = module.subprocess.run
    installs: list[list[str]] = []

    def run(cmd, *a, **kw):
        if "install" in cmd:
            installs.append(list(cmd))
            return module.subprocess.CompletedProcess(cmd, 0)
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(module.subprocess, "run", run)
    setup._install_baseline_only_deps(venv)

    assert len(installs) == 1, installs
    cmd = installs[0]
    assert cmd[:3] == ["uv", "pip", "install"] and "pip" not in cmd[3:]
    assert cmd[cmd.index("--python") + 1] == str(venv / "bin" / "python")
    assert f"{dist}>=3.10.0" in cmd
