"""Tests for the resource-debug summary parser."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_summary_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "resource_debug_summary.py"
    spec = importlib.util.spec_from_file_location("resource_debug_summary", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_resource_log_and_build_summary(tmp_path):
    module = _load_summary_module()
    log_path = tmp_path / "resource.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "[INFO] interpretune.utils.logging: The following columns don't have a corresponding argument "
                    "in `NoneType.forward` and have been ignored."
                ),
                (
                    "[shell_resource_debug] coverage-phase:start:base pytest: context=shell rss_gb=1.00 "
                    "vms_gb=2.00 cuda_available=true cuda_device_count=1 cuda_gpu0_total_gb=40.00 "
                    "cuda_gpu0_current_allocated_gb=3.00 cuda_gpu0_current_reserved_gb=4.00 "
                    "cuda_gpu0_peak_allocated_gb=5.00 cuda_gpu0_peak_reserved_gb=6.00"
                ),
                (
                    "[fixture_resource_debug] it_session_fixture:ct_nnsight:setup:end: context=fixture "
                    "kind=it_session_fixture key=ct_nnsight scope=function lifecycle=setup_end phase=setup "
                    "rss_gb=2.00 vms_gb=3.00 cuda_available=true cuda_device_count=1 "
                    "cuda_gpu0_total_gb=40.00 cuda_gpu0_current_allocated_gb=7.00 "
                    "cuda_gpu0_current_reserved_gb=8.00 cuda_gpu0_peak_allocated_gb=9.00 "
                    "cuda_gpu0_peak_reserved_gb=10.00 delta_cuda_gpu0_current_reserved_gb=2.50 "
                    "delta_cuda_gpu0_peak_reserved_gb=3.50"
                ),
            ]
        ),
        encoding="utf-8",
    )

    entries = module.parse_resource_log(log_path)
    payload = module.build_summary_payload(entries)
    summary_text = module.build_summary_text(entries)

    assert len(entries) == 2
    assert payload["gpu_summary"][0]["gpu_id"] == 0
    assert payload["gpu_summary"][0]["peak_reserved_pct"] == 25.0
    assert payload["fixture_summary"][0]["key"] == "ct_nnsight"
    assert payload["fixture_summary"][0]["setup_reserved_delta_gb"] == 2.5
    assert "GPU Peak Usage" in summary_text
    assert "Fixture VRAM Estimates" in summary_text
    assert "ct_nnsight" in summary_text
