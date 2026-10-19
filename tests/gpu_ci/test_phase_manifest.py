"""CPU tests for the phase collection manifest and the ROCm triage buckets (#794).

The planted failures below are sanitized: invented one-line shapes of each bucket, not copied
logs. They pin the classifier's contract (bucket names, first-match order, ``unknown`` for a
miss) without needing AMD hardware, which is the point: the triage paths are exercised now,
and ROCm execution itself stays explicitly pending.
"""

from __future__ import annotations

import pytest

from tests.gpu_ci.phase_manifest import PHASES, classify_failure, collect_phase, env_pins, parse_nodeids


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


def test_env_pins_name_the_source_and_torch():
    pins = env_pins()
    assert pins["git_sha"] and pins["git_sha"] != "unknown"
    assert "torch" in pins and "python" in pins


def test_collect_phase_narrows_and_reports_collected_beside_phase():
    entry = collect_phase("cuda", ["-k", "test_the_probe_reads_back_a_known_allocation"])
    assert entry["phase"] == "cuda"
    assert entry["collected"] == len(entry["nodeids"]) == 1
    assert entry["executed"] == 0
    assert "test_the_probe_reads_back_a_known_allocation" in entry["nodeids"][0]


def test_collect_phase_rejects_an_unknown_phase():
    with pytest.raises(ValueError, match="unknown phase"):
        collect_phase("not-a-phase")
