from __future__ import annotations

import shutil
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from tests.nb_experiments.concept_direction.analysis.oqi_debug_session_ablation import _build_notebook_cfg


def test_build_notebook_cfg_preserves_debug_session_surface_preset(tmp_path: Path) -> None:
    base_config_path = Path(__file__).resolve().parents[1] / "configs" / "base_oqi_reasoning_oh.yaml"
    config_path = tmp_path / "oqi_debug_surface.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "EXTENDS": str(base_config_path),
                "EXPERIMENT": {
                    "concept_pair_config_path": str(
                        Path(__file__).resolve().parents[1] / "configs" / "cp_ohio_entities_gemma_it.yaml"
                    )
                },
                "SESSION": {"debug_session_surface_preset": "parity_surface"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg, should_cleanup = _build_notebook_cfg(config_path)
    try:
        assert cfg.debug_session_surface_preset == "parity_surface"
        assert cfg.session_kwargs["debug_session_surface_preset"] == "parity_surface"
        assert cfg.session_kwargs["force_device"] is None
    finally:
        if should_cleanup:
            shutil.rmtree(cfg.work_root, ignore_errors=True)
