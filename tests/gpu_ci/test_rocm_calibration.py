"""CPU tests for the ROCm calibrations measured on the first dual-vendor parity runs (#794).

The memory-profiler expectations take a footprint's ROCm overrides only on a ROCm build, and the conformance numerics
take their ROCm tolerances only for tensors on a ROCm device. Both are exercised here without AMD hardware, by faking
the build where needed.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.testing.conformance import cases
from tests import results


def _footprint(**rocm):
    footprint = {
        "src": "cuda",
        "phase": "test",
        "expected_mem": {"allocated_bytes.all.peak": 10, "reserved_bytes.all.peak": 100},
    }
    if rocm:
        footprint[results.ROCM_MEM_KEY] = rocm
    return {"alias": footprint}


@pytest.mark.parametrize("hip", [None, "7.15.26333"], ids=["cuda", "rocm"])
def test_rocm_overrides_apply_only_on_a_rocm_build(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    _, expected = results.mem_results(_footprint(**{"reserved_bytes.all.peak": 120}), "alias")["expected_memstats"]
    assert expected["reserved_bytes.all.peak"] == (120 if hip else 100)
    assert expected["allocated_bytes.all.peak"] == 10  # a stat with no override keeps the CUDA expectation


def test_a_footprint_without_overrides_is_unchanged_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.15.26333", raising=False)
    _, expected = results.mem_results(_footprint(), "alias")["expected_memstats"]
    assert expected == {"allocated_bytes.all.peak": 10, "reserved_bytes.all.peak": 100}


def test_the_measured_rocm_override_is_recorded():
    footprints = results.load_memory_footprint_results()
    assert footprints["test_l_profiling.test_cuda_32"][results.ROCM_MEM_KEY] == {"reserved_bytes.all.peak": 689963008}


def test_cpu_tensors_never_take_the_rocm_tolerances(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.15.26333", raising=False)
    assert not cases._on_rocm_device(torch.zeros(1))


@pytest.mark.parametrize(
    "device_type, hip, expected", [("cuda", "7.15", True), ("cpu", "7.15", False), ("cuda", None, False)]
)
def test_rocm_is_decided_by_the_suite_device(monkeypatch, device_type, hip, expected):
    """The commuting control compares logits already on the CPU, so the suite's device decides, not the
    tensor's."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    assert cases._suite_on_rocm(SimpleNamespace(inputs=SimpleNamespace(device_type=device_type))) is expected


class TestMostlyClose:
    """The bounded-fraction comparison the batched-hooks case uses on ROCm."""

    @staticmethod
    def _pair(n_beyond: int, size: int = 10_000, delta: float = 0.03):
        ref = torch.ones(size)
        got = ref.clone()
        got[:n_beyond] += delta
        return got, ref

    def test_a_bounded_fraction_passes(self):
        got, ref = self._pair(n_beyond=10)  # 1e-3 of the elements
        cases._assert_mostly_close(got, ref, what="x", max_fraction=1.5e-3, cap=6e-2)

    def test_too_many_beyond_the_tolerance_fails_with_the_numbers(self):
        got, ref = self._pair(n_beyond=20)
        with pytest.raises(AssertionError, match=r"20 of 10000 elements \(2\.00e-03\)"):
            cases._assert_mostly_close(got, ref, what="x", max_fraction=1.5e-3, cap=6e-2)

    def test_one_element_beyond_the_cap_fails(self):
        got, ref = self._pair(n_beyond=1, delta=0.1)
        with pytest.raises(AssertionError, match="cap 6.0e-02"):
            cases._assert_mostly_close(got, ref, what="x", max_fraction=1.5e-3, cap=6e-2)
