from __future__ import annotations

import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from tests.nb_experiments.concept_direction.analysis import concept_direction_analysis as analysis
from tests.nb_experiments.concept_direction.analysis import concept_direction_analysis_launcher as launcher


def _write_versioned_report(
    artifact_dir: Path,
    artifact_name: str,
    base_name: str,
    generation: str,
    *,
    payload: dict[str, object] | None = None,
) -> Path:
    target_dir = artifact_dir / artifact_name
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / analysis.build_versioned_artifact_file_name(base_name, generation=generation)
    report_payload = dict(payload or {})
    report_payload[analysis.ARTIFACT_METADATA_KEY] = {
        "artifact_name": artifact_name,
        "artifact_kind": "test_payload",
        "generation": generation,
        "generated_at": generation,
    }
    path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return path


def test_load_experiment_set_resolves_relative_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "analysis_set.yaml"
    notebook_path = tmp_path / "concept_direction_template.ipynb"
    config_dir = tmp_path / "configs"
    analysis_notebook = tmp_path / "concept_direction_analysis.ipynb"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "reference_tests": ["tests/example.py::test_demo"],
                "notebook_configs": ["orange.yaml"],
                "notebook_path": notebook_path.name,
                "config_dir": config_dir.name,
                "analysis_notebook": analysis_notebook.name,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    notebook_path.write_text("{}", encoding="utf-8")
    config_dir.mkdir()
    analysis_notebook.write_text("{}", encoding="utf-8")

    experiment_set = launcher.load_experiment_set(manifest_path)

    assert experiment_set.reference_tests == ("tests/example.py::test_demo",)
    assert experiment_set.notebook_configs == ("orange.yaml",)
    assert experiment_set.notebook_path == notebook_path.resolve()
    assert experiment_set.config_dir == config_dir.resolve()
    assert experiment_set.analysis_notebook == analysis_notebook.resolve()


def test_archive_existing_artifact_root_renames_non_empty_directory(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "demo.txt").write_text("payload", encoding="utf-8")

    archive_path = launcher.archive_existing_artifact_root(artifact_root, generation="20260424_190000")

    assert archive_path == tmp_path / "artifacts_20260424_190000"
    assert archive_path is not None and archive_path.exists()
    assert not artifact_root.exists()


def test_build_run_environment_sets_artifact_generation(tmp_path: Path) -> None:
    env = launcher.build_run_environment(tmp_path, generation="20260424_190000")

    assert env[launcher.PRESERVE_ARTIFACTS_ENV] == "1"
    assert env[launcher.PRESERVE_ARTIFACT_DIR_ENV] == str(tmp_path)
    assert env[launcher.ARTIFACT_GENERATION_ENV] == "20260424_190000"


def test_save_concept_direction_parity_report_writes_versioned_metadata(tmp_path: Path) -> None:
    output_path = analysis.save_concept_direction_parity_report(
        {"status": "ok"},
        artifact_name="gemma3_1b_it_orange",
        artifact_dir=tmp_path,
        generation="20260424_180000",
    )

    assert output_path.name == "concept_direction_parity_report_20260424_180000.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload[analysis.ARTIFACT_METADATA_KEY] == {
        "artifact_name": "gemma3_1b_it_orange",
        "artifact_kind": "parity_report",
        "base_name": analysis.PARITY_REPORT_FILE_BASENAME,
        "file_name": "concept_direction_parity_report_20260424_180000.json",
        "generation": "20260424_180000",
        "generated_at": "20260424_180000",
    }


def test_discover_concept_direction_artifacts_selects_latest_by_generation_not_write_order(tmp_path: Path) -> None:
    newest_report = _write_versioned_report(
        tmp_path,
        "gemma3_1b_it_orange",
        analysis.PARITY_REPORT_FILE_BASENAME,
        "20260424_180001",
        payload={"report": "new"},
    )
    older_report = _write_versioned_report(
        tmp_path,
        "gemma3_1b_it_orange",
        analysis.PARITY_REPORT_FILE_BASENAME,
        "20260424_170001",
        payload={"report": "old"},
    )

    discovered = analysis.discover_concept_direction_artifacts(artifact_dir=tmp_path)

    assert discovered["artifact_root"] == tmp_path.resolve()
    assert discovered["report_paths"] == [newest_report]
    assert older_report not in discovered["report_paths"]


def test_discover_concept_direction_artifacts_filters_by_generation(tmp_path: Path) -> None:
    first_generation = "20260424_180001"
    second_generation = "20260424_180101"
    _write_versioned_report(
        tmp_path,
        "gemma3_1b_it_orange",
        analysis.PARITY_REPORT_FILE_BASENAME,
        first_generation,
    )
    matching_report = _write_versioned_report(
        tmp_path,
        "gemma3_1b_it_orange",
        analysis.PARITY_REPORT_FILE_BASENAME,
        second_generation,
    )
    matching_pipeline = _write_versioned_report(
        tmp_path,
        "gemma3_1b_it_orange_notebook_debug",
        analysis.PIPELINE_STATE_FILE_BASENAME,
        second_generation,
    )

    discovered = analysis.discover_concept_direction_artifacts(
        artifact_dir=tmp_path,
        generation=second_generation,
    )

    assert discovered["artifact_generation"] == second_generation
    assert discovered["report_paths"] == [matching_report]
    assert discovered["notebook_pipeline_state_paths"] == [matching_pipeline]
