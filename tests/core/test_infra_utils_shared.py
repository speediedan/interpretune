"""The shared ``scripts/infra_utils.sh`` under each project's configuration.

The file is byte-identical with finetuning-scheduler's copy, so what differs per project comes from the
``infra_utils_project.sh`` it sources at its end. These run the shared file in a scratch ``scripts/`` directory with
interpretune's project file, with a finetuning-scheduler-shaped one, and with none, so a reconciliation that serves
only one project fails here.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# On a Windows runner `bash` resolves to the WSL launcher stub, which prints an install notice instead of running.
pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="bash helper")

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _tree(tmp_path: Path, project: str | None) -> Path:
    """A scratch repository with the shared file and, when given, a project file of that content."""
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(_SCRIPTS / "infra_utils.sh", scripts / "infra_utils.sh")
    if project == "interpretune":
        shutil.copy(_SCRIPTS / "infra_utils_project.sh", scripts / "infra_utils_project.sh")
    elif project is not None:
        (scripts / "infra_utils_project.sh").write_text(project)
    return scripts


def _run(scripts: Path, body: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    script = f'set -eo pipefail\nsource "{scripts}/infra_utils.sh"\n{body}\n'
    base = {"PATH": "/usr/bin:/bin", "HOME": "/home/nobody"}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**base, **(env or {})})


FTS_PROJECT = "INFRA_VENV_BASE_VAR=FTS_VENV_BASE\n"


@pytest.mark.parametrize(
    "project, env, expected",
    [
        pytest.param("interpretune", {"IT_VENV_BASE": "/v/it"}, "/v/it/env", id="interpretune-variable"),
        pytest.param(FTS_PROJECT, {"FTS_VENV_BASE": "/v/fts"}, "/v/fts/env", id="fts-variable"),
        pytest.param("interpretune", {"FTS_VENV_BASE": "/v/fts"}, "/home/nobody/.venvs/env", id="other-projects-var"),
        pytest.param(None, {"IT_VENV_BASE": "/v/it"}, "/home/nobody/.venvs/env", id="no-project-file"),
    ],
)
def test_the_venv_base_is_the_projects_own_variable(tmp_path, project, env, expected):
    out = _run(_tree(tmp_path, project), 'determine_venv_path "" env', env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == expected


def test_an_explicit_venv_dir_wins(tmp_path):
    out = _run(_tree(tmp_path, "interpretune"), "determine_venv_path /explicit env", {"IT_VENV_BASE": "/v/it"})
    assert out.stdout.strip() == "/explicit/env"


def test_sourcing_without_a_project_file_succeeds_under_set_e(tmp_path):
    """The project file is optional: sourcing must not return the status of a failed existence test."""
    out = _run(_tree(tmp_path, None), "echo sourced")
    assert out.returncode == 0 and out.stdout.strip() == "sourced", out.stderr


def test_interpretunes_helpers_come_from_its_project_file(tmp_path):
    present = _run(_tree(tmp_path / "a", "interpretune"), "type resource_debug_enabled log_shell_resource_snapshot")
    assert present.returncode == 0, present.stderr
    absent = _run(_tree(tmp_path / "b", FTS_PROJECT), "type resource_debug_enabled")
    assert absent.returncode != 0


@pytest.mark.parametrize(
    "lines, expected",
    [
        pytest.param("2.11.0.dev1\ncu130\nnightly\n", "2.11.0.dev1|cu130|nightly", id="nightly"),
        pytest.param("# header\n\n2.10.0\ncpu\ntest\n", "2.10.0|cpu|test", id="comments-and-blanks"),
        pytest.param("2.10.0\ncu128\nstable\n", "2.10.0|cu128|stable", id="stable"),
        pytest.param("", "||", id="empty"),
    ],
)
def test_the_torch_pre_config_reads_from_the_default_path(tmp_path, lines, expected):
    scripts = _tree(tmp_path, "interpretune")
    (tmp_path / "requirements" / "ci").mkdir(parents=True)
    (tmp_path / "requirements" / "ci" / "torch-pre.txt").write_text(lines)
    out = _run(scripts, 'read_torch_pre_config; echo "${TORCH_PRE_VERSION}|${TORCH_PRE_CUDA}|${TORCH_PRE_CHANNEL}"')
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == expected


def test_an_explicit_torch_pre_path_and_a_missing_file(tmp_path):
    scripts = _tree(tmp_path, "interpretune")
    other = tmp_path / "elsewhere.txt"
    other.write_text("2.9.0\ncu126\ntest\n")
    out = _run(scripts, f'read_torch_pre_config "{other}"; echo "$TORCH_PRE_VERSION"')
    assert out.stdout.strip() == "2.9.0"
    missing = _run(scripts, 'read_torch_pre_config; echo "[$TORCH_PRE_VERSION]"')
    assert missing.returncode == 0 and missing.stdout.strip() == "[]"


def test_an_unknown_torch_channel_is_refused(tmp_path):
    scripts = _tree(tmp_path, "interpretune")
    bad = tmp_path / "bad.txt"
    bad.write_text("2.9.0\ncu126\nbeta\n")
    out = _run(scripts, f'read_torch_pre_config "{bad}" || echo "refused $?"')
    assert out.stdout.strip() == "refused 1"
    assert "Invalid torch channel 'beta'" in out.stderr and str(bad) in out.stderr


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param("nightly cu130", "https://download.pytorch.org/whl/nightly/cu130", id="nightly"),
        pytest.param("test", "https://download.pytorch.org/whl/test/cpu", id="test-defaults-to-cpu"),
        pytest.param("stable cu128", "https://download.pytorch.org/whl/cu128", id="stable"),
    ],
)
def test_the_torch_index_url(tmp_path, args, expected):
    out = _run(_tree(tmp_path, None), f"get_torch_index_url {args}")
    assert out.returncode == 0 and out.stdout.strip() == expected, out.stderr


def test_an_unknown_index_channel_is_refused(tmp_path):
    out = _run(_tree(tmp_path, None), "get_torch_index_url beta cu128 || echo refused")
    assert out.stdout.strip() == "refused" and "Unknown torch channel: beta" in out.stderr
