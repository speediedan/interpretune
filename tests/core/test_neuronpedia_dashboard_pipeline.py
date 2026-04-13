from __future__ import annotations

from pathlib import Path

from interpretune.utils.neuronpedia_dashboard_pipeline import completed_layers_from_logs


def test_completed_layers_from_logs_collects_done_markers(tmp_path: Path) -> None:
    primary_log = tmp_path / "run.log"
    secondary_log = tmp_path / "run.resume.log"
    primary_log.write_text(
        "START layer=23 sae_path=foo\nDONE layer=23 sae_path=foo time=2026-04-06T11:00:00\n",
        encoding="utf-8",
    )
    secondary_log.write_text(
        "DONE layer=24 sae_path=bar time=2026-04-06T12:00:00\nFAIL layer=25 sae_path=baz\n",
        encoding="utf-8",
    )

    completed = completed_layers_from_logs(primary_log, secondary_log)

    assert completed == {23, 24}
