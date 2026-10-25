"""CPU tests for the phase collection manifest and the ROCm triage buckets (#794).

The planted failures below are sanitized: invented one-line shapes of each bucket, not copied
logs. They pin the classifier's contract (bucket names, first-match order, ``unknown`` for a
miss) without needing AMD hardware, which is the point: the triage paths are exercised now,
and ROCm execution itself stays explicitly pending.
"""

from __future__ import annotations

import subprocess

import pytest

from tests.gpu_ci.phase_manifest import (
    COLLECT_DIRS,
    PHASES,
    classify_failure,
    classify_run,
    collect_phase,
    env_pins,
    git_dirty_paths,
    junit_failures,
    parse_nodeids,
)


def test_parse_nodeids_reads_collect_only_output():
    output = "tests/a.py::test_one\ntests/a.py::TestB::test_two[param]\n\n2 tests collected in 0.01s\n"
    assert parse_nodeids(output) == ["tests/a.py::test_one", "tests/a.py::TestB::test_two[param]"]


def test_phases_cover_the_ci_gpu_phases():
    # gpu_phase.sh runs cuda / standalone / profile_ci; a renamed phase would strand the manifest.
    assert {"cuda", "standalone", "profile_ci"} <= set(PHASES)


@pytest.mark.parametrize(
    "text, bucket",
    [
        ("ModuleNotFoundError: No module named 'flash_attn'", "cuda-only-dependency"),
        ("OSError: libcudart.so.12: cannot open shared object file: No such file", "cuda-only-dependency"),
        ("RuntimeError: hipErrorNotYetImplemented: operation not yet implemented", "rocm-kernel-gap"),
        ("RuntimeError: miopenStatusNotImplemented: not implemented for this miopen build", "rocm-kernel-gap"),
        ("RuntimeError: NCCL is not available on this device", "backend-gap"),
        ("AssertionError: backend capability LATENT_MODELS is not supported here", "backend-gap"),
        ("FileNotFoundError: 'nvidia-smi' not found on PATH", "hip-compatible-cuda-interface"),
        ("AssertionError: expected an NVIDIA device name, got 'AMD Instinct'", "hip-compatible-cuda-interface"),
        ("torch.AcceleratorError: CUDA error: device-side assert triggered", "device-fault"),
        ("torch.AcceleratorError: CUDA error: unspecified launch failure (hipErrorLaunchFailure)", "device-fault"),
        (
            "HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016",
            "device-fault",
        ),
        ("RuntimeError: CUDA error: an illegal memory access was encountered", "device-fault"),
        (
            "memprofiler.memory_stats['npp_diff']['allocated_bytes.all.peak']\nAssertionError: Scalars are not close!",
            "resource-calibration",
        ),
        (
            "AssertionError: Mismatched elements: 4 / 10 (40.0%)\nGreatest absolute difference: 0.031",
            "numerics-tolerance",
        ),
        ("AssertionError: Tensor-likes are not close!", "numerics-tolerance"),
        ("AssertionError: assert 1 == 2", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_failure_planted_shapes(text, bucket):
    result = classify_failure(text)
    assert result["bucket"] == bucket
    assert (result["evidence"] != "") == (bucket != "unknown")


def test_classify_failure_prefers_the_dependency_bucket_over_a_kernel_mention():
    # A missing CUDA-only library whose error also mentions HIP must not read as a kernel gap:
    # the fix is the dependency, and the bucket order is what says so.
    text = "OSError: libcuda.so: cannot open shared object file (hip runtime present)"
    assert classify_failure(text)["bucket"] == "cuda-only-dependency"


def test_classify_failure_reads_a_kernel_gap_before_the_fault_it_raises():
    # An unimplemented HIP op that also faults the launch is a kernel gap: the gap is the fix.
    text = "hipErrorNotYetImplemented: operation not yet implemented\nCUDA error: unspecified launch failure"
    assert classify_failure(text)["bucket"] == "rocm-kernel-gap"


def test_classify_run_attributes_a_sticky_context_cascade_to_the_first_fault():
    fault = "torch.AcceleratorError: CUDA error: device-side assert triggered"
    failures = [
        ("tests/a.py::test_ok_numerics", "AssertionError: Greatest absolute difference: 0.5"),
        ("tests/b.py::test_first_fault", fault),
        ("tests/c.py::test_after", fault),
        ("tests/d.py::test_numerics_after", "AssertionError: Tensor-likes are not close!"),
        ("tests/e.py::test_after_again", f"failed on setup with {fault}"),
    ]
    run = classify_run(failures)
    assert run["first_fault"] == "tests/b.py::test_first_fault"
    buckets = [f["bucket"] for f in run["failures"]]
    assert buckets == [
        "numerics-tolerance",
        "device-fault",
        "device-fault-cascade",
        "numerics-tolerance",
        "device-fault-cascade",
    ]
    assert {f["first_fault"] for f in run["failures"] if f["bucket"] == "device-fault-cascade"} == {
        "tests/b.py::test_first_fault"
    }
    assert run["counts"] == {"numerics-tolerance": 2, "device-fault": 1, "device-fault-cascade": 2}


def test_classify_run_without_a_fault_has_no_first_fault():
    run = classify_run([("tests/a.py::test_x", "AssertionError: assert 1 == 2")])
    assert run["first_fault"] is None and run["counts"] == {"unknown": 1}


def test_junit_failures_reads_failures_and_errors_in_file_order(tmp_path):
    junit = tmp_path / "phase.junit.xml"
    junit.write_text(
        '<testsuites><testsuite name="pytest">'
        '<testcase classname="tests.a" name="test_pass"/>'
        '<testcase classname="tests.b" name="test_fails"><failure message="device-side assert triggered">tb</failure>'
        "</testcase>"
        '<testcase classname="tests.c" name="test_errors"><error message="failed on setup">setup tb</error></testcase>'
        '<testcase classname="tests.d" name="test_skip"><skipped message="no"/></testcase>'
        "</testsuite></testsuites>"
    )
    failures = junit_failures(junit)
    assert [n for n, _ in failures] == ["tests.b::test_fails", "tests.c::test_errors"]
    assert failures[0][1] == "device-side assert triggered\ntb"


def test_env_pins_name_the_source_and_torch():
    pins = env_pins()
    assert pins["git_sha"] and pins["git_sha"] != "unknown"
    assert "torch" in pins and "python" in pins
    assert pins["git_dirty"] == bool(pins["git_dirty_paths"])


def test_git_dirty_paths_sees_modified_and_untracked_collection_files(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    for d in COLLECT_DIRS:
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "test_kept.py").write_text("def test_kept():\n    pass\n")
    (tmp_path / "README.md").write_text("outside the collection roots\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    assert git_dirty_paths(tmp_path) == []
    (tmp_path / "tests" / "test_kept.py").write_text("def test_kept():\n    assert True\n")
    (tmp_path / "tests" / "test_new.py").write_text("def test_new():\n    pass\n")
    (tmp_path / "README.md").write_text("changed, but no collection root\n")
    assert git_dirty_paths(tmp_path) == ["tests/test_kept.py", "tests/test_new.py"]


def test_git_dirty_paths_is_none_outside_git(tmp_path):
    assert git_dirty_paths(tmp_path) is None


def test_collect_phase_narrows_and_reports_collected_beside_phase():
    entry = collect_phase("cuda", ["-k", "test_the_probe_reads_back_a_known_allocation"])
    assert entry["phase"] == "cuda"
    assert entry["collected"] == len(entry["nodeids"]) == 1
    assert entry["executed"] == 0
    assert entry["extra_args"] == ["-k", "test_the_probe_reads_back_a_known_allocation"]
    assert "test_the_probe_reads_back_a_known_allocation" in entry["nodeids"][0]


def test_collect_phase_rejects_an_unknown_phase():
    with pytest.raises(ValueError, match="unknown phase"):
        collect_phase("not-a-phase")
