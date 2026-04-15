from __future__ import annotations

import shutil
from pathlib import Path

from tests.parity_analysis.oqi_debug_session_ablation import _build_notebook_cfg


def test_build_notebook_cfg_preserves_debug_session_surface_preset() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "concept_direction_approach_parity"
        / "archived_cfgs"
        / "gemma3_4b_it_local_oqi_reasoning_single_fs_di_60.yaml"
    )
    cfg, should_cleanup = _build_notebook_cfg(config_path)
    try:
        assert cfg.debug_session_surface_preset == "parity_surface"
        assert cfg.session_kwargs["debug_session_surface_preset"] == "parity_surface"
        assert cfg.session_kwargs["force_device"] is None
    finally:
        if should_cleanup:
            shutil.rmtree(cfg.work_root, ignore_errors=True)
