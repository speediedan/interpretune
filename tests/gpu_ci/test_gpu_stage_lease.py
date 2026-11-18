"""The two-stage GPU lease steps (CPU): what each stage asks the host's lease tool for, what it exports to later
steps, and the fallbacks when the tool or the lease directory is absent.

The host's lease tool is private infrastructure, so a stub stands in for it: it records its arguments and, for
``--hold``, writes the grant file a real hold writes.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

# The pipeline steps run only in the Linux job container, and the fallback path needs util-linux (flock, setsid).
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the GPU pipeline's steps run in a Linux container")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".azure-pipelines" / "scripts" / "gpu_stage_lease.sh"
CHECK = ROOT / ".azure-pipelines" / "scripts" / "check_gpu_phases.sh"
DEVICES = "GPU-big:24564,GPU-small:8192"

STUB = """#!/usr/bin/env bash
echo "$*" >> "{log}"
pidfile=""; gpus=all
while [[ $# -gt 0 ]]; do
  case "$1" in --pidfile) pidfile=$2; shift 2 ;; --gpus) gpus=$2; shift 2 ;; *) shift ;; esac
done
if [[ -n "$pidfile" ]]; then
  echo 12345 > "$pidfile"
  [[ "$gpus" == 1 ]] && echo GPU-small > "$pidfile.device" || echo all > "$pidfile.device"
fi
exit {rc}
"""


def _run(tmp_path: Path, *args: str, tool_rc: int = 0, with_tool: bool = True, with_dir: bool = True, **env):
    leases = tmp_path / "leases"
    if with_dir:
        leases.mkdir(exist_ok=True)
    tool = tmp_path / "tool.sh"
    log = tmp_path / "tool.log"
    if with_tool:
        tool.write_text(STUB.format(log=log, rc=tool_rc))
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    full_env = {
        **os.environ,
        "GPU_STAGE_LEASE_DIR": str(leases),
        "GPU_STAGE_LEASE_TOOL": str(tool),
        "GPU_STAGE_LEASE_PIDDIR": str(tmp_path),
        "GPU_LEASE_DEVICES": DEVICES,
        "BUILD_BUILDID": "42",
        "IT_GPU_LEASE_WAIT": "30",
        "IT_GPU_LEASE_ON_TIMEOUT": "fail",
        **env,
    }
    proc = subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, env=full_env, capture_output=True, text=True)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls


def _vars(proc: subprocess.CompletedProcess) -> dict[str, str]:
    prefix = "##vso[task.setvariable variable="
    return dict(line[len(prefix) :].split("]", 1) for line in proc.stdout.splitlines() if line.startswith(prefix))


def test_the_small_stage_asks_for_one_device_sized_to_the_smallest_card(tmp_path):
    proc, calls = _run(tmp_path, "acquire", "small")
    assert proc.returncode == 0, proc.stderr
    # one hold through the shared CI function, for one device and not cpu-heavy: its tests fit the smallest card, and
    # waiting for cpu-heavy behind a local CPU suite once failed a gate on idle GPUs (build 2101)
    assert len(calls) == 1 and calls[0].startswith("--hold"), calls
    assert "--gpus 1 --min-vram 8" in calls[0] and "--project azure-it-42" in calls[0], calls
    assert "--cpu-heavy" not in calls[0], calls
    assert _vars(proc) == {
        "IT_GPU_LEASE_HELD": "1",
        "IT_GPU_TWO_STAGE": "1",
        "IT_GPU_DEVICE": "GPU-small",
        "IT_GPU_SMALL_MAX_GB": "8",
    }


def test_a_one_stage_host_takes_the_whole_server_even_with_the_tool(tmp_path):
    """The any-vendor template sets IT_GPU_ONE_STAGE on an AMD agent, whose identical cards make placement moot."""
    proc, calls = _run(tmp_path, "acquire", "small", IT_GPU_ONE_STAGE="1")
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 1 and "--gpus all --cpu-heavy" in calls[0], calls
    assert _vars(proc) == {"IT_GPU_LEASE_HELD": "1", "IT_GPU_TWO_STAGE": "0", "IT_GPU_DEVICE": "all"}
    # the later stage is then a no-op, and the lease is held under the large stage's pidfile for "release all"
    later, later_calls = _run(tmp_path, "acquire", "large", IT_GPU_TWO_STAGE="0")
    assert later.returncode == 0 and later_calls == calls
    assert (tmp_path / "azp_gpu_large.pid").exists()


def test_the_large_stage_asks_for_the_whole_server(tmp_path):
    proc, calls = _run(tmp_path, "acquire", "large", IT_GPU_TWO_STAGE="1")
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 1 and calls[0].startswith("--hold"), calls
    assert "--gpus all --cpu-heavy" in calls[0], calls
    assert _vars(proc) == {"IT_GPU_LARGE_HELD": "1", "IT_GPU_DEVICE": "all"}


def test_without_the_tool_the_job_falls_back_to_one_whole_server_stage(tmp_path):
    proc, calls = _run(tmp_path, "acquire", "small", with_tool=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "not mounted" in proc.stdout and not calls
    assert _vars(proc) == {"IT_GPU_LEASE_HELD": "1", "IT_GPU_TWO_STAGE": "0", "IT_GPU_DEVICE": "all"}
    # the plain whole-server locks were taken, in order, through the job's existing flock helper
    assert (tmp_path / "leases" / "gpu.holder").exists() and (tmp_path / "leases" / "cpu-heavy.holder").exists()
    # the pipeline's own step order: "release small" must leave the one whole-server stage held for the later phases
    _run(tmp_path, "release", "small", with_tool=False)
    assert (tmp_path / "leases" / "gpu.holder").exists() and (tmp_path / "leases" / "cpu-heavy.holder").exists()
    proc, _ = _run(tmp_path, "acquire", "large", with_tool=False, IT_GPU_TWO_STAGE="0")
    assert proc.returncode == 0 and "one-stage run" in proc.stdout
    _run(tmp_path, "release", "all", with_tool=False)
    assert not (tmp_path / "leases" / "gpu.holder").exists()


def test_without_the_lease_directory_the_host_does_not_use_leases(tmp_path):
    proc, calls = _run(tmp_path, "acquire", "small", with_dir=False)
    assert proc.returncode == 0 and "does not use GPU leases" in proc.stdout and not calls
    assert _vars(proc)["IT_GPU_TWO_STAGE"] == "0"


@pytest.mark.parametrize("stage", ["small", "large"])
def test_a_gate_whose_lease_times_out_fails(tmp_path, stage):
    proc, _ = _run(tmp_path, "acquire", stage, tool_rc=75, IT_GPU_TWO_STAGE="1")
    assert proc.returncode == 1 and "cancel" not in proc.stdout.lower(), proc.stdout


def test_the_small_stage_takes_cpu_heavy_again_when_asked(tmp_path):
    proc, calls = _run(tmp_path, "acquire", "small", IT_GPU_SMALL_CPU_HEAVY="1")
    assert proc.returncode == 0 and "--gpus 1 --min-vram 8 --cpu-heavy" in calls[0], calls


@pytest.mark.parametrize("stage", ["small", "large"])
def test_a_gate_waits_again_while_its_lease_budget_lasts_then_fails_naming_the_holders(tmp_path, stage):
    """A wait that times out with budget left is retried (as many waits as fit the budget), not failed."""
    proc, calls = _run(
        tmp_path, "acquire", stage, tool_rc=75, IT_GPU_TWO_STAGE="1", IT_GPU_LEASE_WAIT="30", IT_GPU_LEASE_BUDGET="90"
    )
    holds = [c for c in calls if c.startswith("--hold")]
    assert proc.returncode == 1, proc.stdout
    assert len(holds) == 3 and all("--timeout 30" in c for c in holds), calls
    assert proc.stdout.count("waiting again") == 2 and "through 3 wait(s)" in proc.stdout, proc.stdout
    assert calls.count("--status") == 3, "each timed-out wait logs who holds the leases"
    assert "cancel" not in proc.stdout.lower()


ONCE_STUB = """#!/usr/bin/env bash
echo "$*" >> "{log}"
pidfile=""
while [[ $# -gt 0 ]]; do case "$1" in --pidfile) pidfile=$2; shift 2 ;; *) shift ;; esac; done
[[ -n "$pidfile" ]] || exit 0
[[ -e "{flag}" ]] || {{ : > "{flag}"; exit 75; }}
echo 12345 > "$pidfile"; echo GPU-small > "$pidfile.device"
"""


def test_a_gate_held_on_its_second_wait_proceeds(tmp_path):
    """The tool times out once, then grants: the stage holds its lease after one retry."""
    proc, _ = _run(tmp_path, "acquire", "small", IT_GPU_LEASE_WAIT="30", IT_GPU_LEASE_BUDGET="90")  # writes the stub
    (tmp_path / "tool.sh").write_text(ONCE_STUB.format(log=tmp_path / "once.log", flag=tmp_path / "timed_out"))
    env = {
        **os.environ,
        "GPU_STAGE_LEASE_DIR": str(tmp_path / "leases"),
        "GPU_STAGE_LEASE_TOOL": str(tmp_path / "tool.sh"),
        "GPU_STAGE_LEASE_PIDDIR": str(tmp_path / "second"),
        "GPU_LEASE_DEVICES": DEVICES,
        "BUILD_BUILDID": "42",
        "IT_GPU_LEASE_WAIT": "30",
        "IT_GPU_LEASE_BUDGET": "90",
        "IT_GPU_LEASE_ON_TIMEOUT": "fail",
    }
    (tmp_path / "second").mkdir()
    proc = subprocess.run(["bash", str(SCRIPT), "acquire", "small"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "waiting again" in proc.stdout and _vars(proc)["IT_GPU_LEASE_HELD"] == "1", proc.stdout


def test_a_gate_from_an_older_plan_waits_once(tmp_path):
    """Without IT_GPU_LEASE_BUDGET a gate keeps its old policy: one wait per stage, then fail."""
    proc, calls = _run(tmp_path, "acquire", "small", tool_rc=75, IT_GPU_LEASE_WAIT="30")
    assert proc.returncode == 1 and len([c for c in calls if c.startswith("--hold")]) == 1, calls


def _phase_counts(tmp_path: Path, **states: str) -> dict:
    counts = tmp_path / "gpu_phase_counts"
    counts.mkdir(exist_ok=True)
    for name, state in states.items():
        (counts / name.replace("__", ".")).write_text(state + "\n")
    return {**os.environ, "AGENT_TEMPDIRECTORY": str(tmp_path)}


def test_a_full_run_fails_when_a_phase_ran_nothing_in_every_class(tmp_path):
    env = _phase_counts(
        tmp_path,
        cuda__small="ran",
        cuda__large="ran",
        standalone__small="empty",
        standalone__large="empty",
        profile_ci__small="ran",
        profile_ci__large="empty",
    )
    proc = subprocess.run(
        ["bash", str(CHECK)], env={**env, "IT_GPU_SELECTION_MODE": "full"}, capture_output=True, text=True
    )
    assert proc.returncode == 1 and "phase standalone ran no test" in proc.stderr, proc.stderr
    selected = subprocess.run(
        ["bash", str(CHECK)], env={**env, "IT_GPU_SELECTION_MODE": "selected"}, capture_output=True
    )
    assert selected.returncode == 0


def test_a_full_run_passes_when_every_phase_ran_in_some_class(tmp_path):
    env = _phase_counts(
        tmp_path,
        cuda__small="ran",
        cuda__large="empty",
        standalone__small="empty",
        standalone__large="ran",
        profile_ci__all="ran",
    )
    proc = subprocess.run(
        ["bash", str(CHECK)], env={**env, "IT_GPU_SELECTION_MODE": "full"}, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def _events(tmp_path: Path) -> list[dict]:
    import json

    path = tmp_path / "telemetry" / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_a_stage_records_its_wait_its_grant_with_devices_and_its_release(tmp_path):
    telemetry = {"IT_TELEMETRY_DIR": str(tmp_path / "telemetry")}
    assert _run(tmp_path, "acquire", "small", **telemetry)[0].returncode == 0
    assert _run(tmp_path, "release", "small", **telemetry)[0].returncode == 0
    events = _events(tmp_path)
    assert [e["action"] for e in events] == ["wait", "acquired", "released"]
    acquired = events[1]
    assert (acquired["stage"], acquired["grant"], acquired["devices"], acquired["mode"]) == (
        "small",
        "GPU-small",
        "GPU-small",
        "tool",
    )
    assert all(isinstance(e["epoch"], int) for e in events)


def test_a_whole_server_grant_records_every_device_it_covers(tmp_path):
    proc, _ = _run(tmp_path, "acquire", "large", IT_GPU_TWO_STAGE="1", IT_TELEMETRY_DIR=str(tmp_path / "telemetry"))
    assert proc.returncode == 0, proc.stderr
    acquired = [e for e in _events(tmp_path) if e["action"] == "acquired"]
    assert acquired and acquired[0]["grant"] == "all" and acquired[0]["devices"] == "GPU-big;GPU-small"


def test_a_timed_out_wait_is_recorded_and_no_release_is_invented(tmp_path):
    telemetry = {"IT_TELEMETRY_DIR": str(tmp_path / "telemetry")}
    assert _run(tmp_path, "acquire", "small", tool_rc=75, **telemetry)[0].returncode == 1
    _run(tmp_path, "release", "small", **telemetry)
    assert [e["action"] for e in _events(tmp_path)] == ["wait", "timeout"]
