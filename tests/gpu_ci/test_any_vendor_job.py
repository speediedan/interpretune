"""The any-vendor GPU job (#703), on CPU: what the in-container exec helper forwards, and the pipeline invariants a
YAML edit could break silently (every step inside the container, digest-pinned images both pipelines agree on, the
admission demands, and AMD's one-stage lease)."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PIPELINES = ROOT / ".azure-pipelines"
CTR = PIPELINES / "scripts" / "ctr.sh"
TEMPLATE = PIPELINES / "templates" / "gpu-job.yml"
DIGEST = re.compile(r"^speediedan/interpretune:[\w.-]+@sha256:[0-9a-f]{64}$")


def _load(name: str) -> dict:
    return yaml.safe_load((PIPELINES / name).read_text())


def _template_use(pipeline: dict) -> dict:
    (job,) = pipeline["jobs"]
    assert job["template"] == "templates/gpu-job.yml"
    return job["parameters"]


@pytest.mark.skipif(sys.platform != "linux", reason="the GPU pipeline's steps run on a Linux agent")
class TestCtr:
    @staticmethod
    def _run(tmp_path: Path, **env):
        fake = tmp_path / "docker"
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        base = {k: v for k, v in os.environ.items() if not k.startswith("IT_")}
        full = {**base, "PATH": f"{tmp_path}:{os.environ['PATH']}", **env}
        return subprocess.run(["bash", str(CTR), "echo hi"], env=full, capture_output=True, text=True)

    def test_forwards_it_variables_and_the_named_ones_by_name(self, tmp_path):
        proc = self._run(
            tmp_path, IT_CTR="ctr", IT_GPU_DEVICE="all", HF_TOKEN="secret", BUILD_BUILDID="7", AGENT_TEMPDIRECTORY="/x"
        )
        assert proc.returncode == 0, proc.stderr
        args = proc.stdout.splitlines()
        forwarded = {args[i + 1] for i, a in enumerate(args) if a == "-e"}
        assert {"IT_GPU_DEVICE", "HF_TOKEN", "BUILD_BUILDID"} <= forwarded
        # a host path and the helper's own variable stay on the host; values never appear on the command line
        assert "AGENT_TEMPDIRECTORY" not in forwarded and "IT_CTR" not in forwarded
        assert "secret" not in proc.stdout
        assert args[0] == "exec" and args[-4:] == ["ctr", "bash", "-c", "echo hi"]

    def test_refuses_without_the_container_name(self, tmp_path):
        proc = self._run(tmp_path)
        assert proc.returncode != 0 and "IT_CTR" in proc.stderr


@pytest.mark.parametrize("name", ["gpu-tests.yml", "readme-flow-gpu.yml"])
def test_every_step_runs_inside_the_container(name):
    steps = _template_use(_load(name))["steps"]
    assert steps
    for step in steps:
        assert step["bash"].startswith("bash .azure-pipelines/scripts/ctr.sh "), step["bash"]


def test_both_pipelines_pin_the_same_images_by_digest():
    images = [
        next(p["default"] for p in _load(name)["parameters"] if p["name"] == "images")
        for name in ("gpu-tests.yml", "readme-flow-gpu.yml")
    ]
    assert images[0] == images[1]
    assert set(images[0]) == {"nvidia", "amd"}
    for vendor, image in images[0].items():
        assert DIGEST.match(image), (vendor, image)
    assert "-rocm" in images[0]["amd"] and "-rocm" not in images[0]["nvidia"]


def test_admission():
    tests, readme = _template_use(_load("gpu-tests.yml")), _template_use(_load("readme-flow-gpu.yml"))
    # gate builds run on either agent; the README flow installs CUDA torch, so it demands NVIDIA, as a release job
    assert (tests["vendor"], tests["jobClass"]) == ("${{ parameters.vendor }}", "ci-any-vendor")
    assert (readme["vendor"], readme["jobClass"]) == ("nvidia", "ci-release")
    template = TEMPLATE.read_text()
    for demand in ("- GPU_VENDOR\n", "- ${{ parameters.jobClass }}", "- ${{ parameters.project }}"):
        assert demand in template
    assert "default: ci-it" in template


def test_amd_takes_one_whole_server_stage():
    template = TEMPLATE.read_text()
    amd = next(line for line in template.splitlines() if line.strip().startswith("amd)"))
    nvidia = next(line for line in template.splitlines() if line.strip().startswith("nvidia)"))
    assert "one_stage=1" in amd and "/dev/kfd" in amd
    assert "one_stage=0" in nvidia


def test_every_gpu_phase_runs_while_its_stage_holds_its_lease():
    """One phase's failure must not hide the others (it once hid standalone and profiling on the AMD agent)."""
    steps = _template_use(_load("gpu-tests.yml"))["steps"]
    phases = [s for s in steps if s["displayName"].startswith("Testing: ") and "extended" not in s["displayName"]]
    assert len(phases) == 6
    for step in phases:
        held = "IT_GPU_LARGE_HELD" if "(large)" in step["displayName"] else "IT_GPU_LEASE_HELD"
        assert step["condition"].startswith("and(succeededOrFailed(),") and held in step["condition"], step
