from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

from tests.nb_experiments.concept_direction.analysis import latent_dynamics_launcher


def test_parse_args_defaults_to_umap() -> None:
    args = latent_dynamics_launcher.parse_args(["orange_config.yaml"])

    assert args.config == "orange_config.yaml"
    assert args.projection_method == "umap"
    assert args.stage_top_n is None
    assert args.kernel_name == "python3"
    assert args.prepare_only is False


def test_build_execution_plan_uses_experiment_name_and_optional_stage_top_n(tmp_path: Path) -> None:
    config_path = tmp_path / "orange.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "EXPERIMENT_NAME": "latent_surface_demo",
                "LATENT_DYNAMICS": {"stage_top_n": 5},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    plan = latent_dynamics_launcher.build_execution_plan(
        config_path,
        output_dir=tmp_path,
        projection_method="pca",
        stage_top_n=9,
        timeout=123,
        kernel_name="python3",
        prepare_only=True,
    )

    assert plan.output_path.parent == tmp_path
    assert plan.output_path.name.startswith("latent_surface_demo_")
    assert plan.output_path.suffix == ".ipynb"
    assert plan.source_config_path.name.startswith("latent_surface_demo_")
    assert plan.resolved_config_path.name.startswith("latent_surface_demo_")
    assert plan.parameters == {
        "EXPERIMENT_CONFIG_PATH": str(config_path.resolve()),
        "PROJECTION_METHOD": "pca",
        "STAGE_TOP_N": 9,
    }
    assert plan.timeout == 123
    assert plan.prepare_only is True


def test_build_papermill_parameters_omits_stage_top_n_when_not_provided(tmp_path: Path) -> None:
    config_path = tmp_path / "orange.yaml"
    config_path.write_text("EXPERIMENT_NAME: latent_surface_demo\n", encoding="utf-8")

    parameters = latent_dynamics_launcher.build_papermill_parameters(
        config_path,
        projection_method="umap",
        stage_top_n=None,
    )

    assert parameters == {
        "EXPERIMENT_CONFIG_PATH": str(config_path.resolve()),
        "PROJECTION_METHOD": "umap",
    }
