# pyright: reportMissingTypeStubs=false
from __future__ import annotations

import json
import logging
import importlib.util
import sys
import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from typing import Any, cast

import pytest
import torch
import yaml
from datasets import Dataset

import interpretune.utils.neuronpedia_dashboard_pipeline as dashboard_pipeline
from interpretune.utils.neuronpedia_dashboard_pipeline import (
    NeuronpediaDashboardPipelineConfig,
    completed_layers_from_logs,
)


def _load_dashboard_launcher_module() -> ModuleType:
    script_path = Path(__file__).parents[2] / "scripts" / "launch_neuronpedia_dashboard_pipeline.py"
    spec = importlib.util.spec_from_file_location("test_dashboard_pipeline_launcher", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_dashboard_profile_module() -> ModuleType:
    script_path = Path(__file__).parents[2] / "scripts" / "profile_neuronpedia_dashboard_generation.py"
    spec = importlib.util.spec_from_file_location("test_dashboard_profile", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_completed_layers_from_logs_collects_timestamped_done_markers(tmp_path: Path) -> None:
    log_path = tmp_path / "run.resume.log"
    log_path.write_text(
        (
            "2026-04-28 18:07:26,740 INFO DONE layer=0 elapsed_seconds=773.4\n"
            "2026-04-28 18:08:27,033 INFO Diagnostics reason=nonzero-exit-1 pid=123\n"
        ),
        encoding="utf-8",
    )

    completed = completed_layers_from_logs(log_path)

    assert completed == {0}


def test_dashboard_log_contains_oom_scans_from_offset(tmp_path: Path) -> None:
    log_path = tmp_path / "run.gpu1.log"
    log_path.write_text("old CUDA out of memory\n", encoding="utf-8")
    offset = log_path.stat().st_size
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("new healthy batch\n")

    assert dashboard_pipeline.dashboard_log_contains_oom(log_path)
    assert not dashboard_pipeline.dashboard_log_contains_oom(log_path, start_offset=offset)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("torch.OutOfMemoryError: cuda allocator failed\n")

    assert dashboard_pipeline.dashboard_log_contains_oom(log_path, start_offset=offset)


def test_profile_parser_detects_columnar_and_resource_batch_completion() -> None:
    profile_module = _load_dashboard_profile_module()

    batch_events, resource_events, runner_perf_events = profile_module.parse_runner_events(
        [
            "Output written to /tmp/run/layer_9/foo/batch-0.json\n",
            (
                "[runner_resource] stage=post_batch_1 pid=123 rss_gib=5.50 cuda_allocated_gib=1.25 "
                "cuda_reserved_gib=2.50 cuda_max_allocated_gib=3.75\n"
            ),
            '[runner_perf] event=batch_total batch=1 wall_s=29.174150 process_io_delta={"write_bytes":17}\n',
            (
                "2026-05-26 09:51:01,312 [runner_perf] event=columnar_output_summary feature_count=1024 "
                "artifact_dir=/tmp/run/layer_9/foo/batch-2.columnar manifest_path=/tmp/run/manifest.json\n"
            ),
            "Columnar output written to /tmp/run/layer_9/foo/batch-3.columnar/manifest.json\n",
        ],
        started_monotonic=time.monotonic(),
    )

    assert {event.batch_num for event in batch_events} == {0, 1, 2, 3}
    assert [event.stage for event in resource_events] == ["post_batch_1"]
    assert resource_events[0].cuda_max_allocated_gib == pytest.approx(3.75)
    assert [event.event for event in runner_perf_events] == ["batch_total", "columnar_output_summary"]
    assert runner_perf_events[0].fields["batch"] == 1
    assert runner_perf_events[0].fields["wall_s"] == pytest.approx(29.17415)
    assert runner_perf_events[0].fields["process_io_delta"] == {"write_bytes": 17}
    assert runner_perf_events[0].raw_line.endswith(
        '[runner_perf] event=batch_total batch=1 wall_s=29.174150 process_io_delta={"write_bytes":17}'
    )
    assert runner_perf_events[1].raw_line.startswith("2026-05-26 09:51:01,312 [runner_perf]")


def test_profile_command_appends_dashboard_extra_args(tmp_path: Path) -> None:
    profile_module = _load_dashboard_profile_module()

    command = profile_module.build_dashboard_command(
        profile_module.ProfileConfig("profile", 1024, 512),
        layer=9,
        python_executable="python",
        run_root=tmp_path / "runs",
        pretokenized_dataset_path=tmp_path / "prompts",
        primary_acts_batch_size=128,
        cuda_visible_devices="0",
        dashboard_extra_args=(
            "--runner-dashboard-output-format=columnar",
            "--runner-sequence-selection-backend=lazy_gpu",
        ),
    )

    assert command[-2:] == [
        "--runner-dashboard-output-format=columnar",
        "--runner-sequence-selection-backend=lazy_gpu",
    ]


def test_profile_command_uses_pretokenized_context_size_for_n_tokens_in_prompt(tmp_path: Path) -> None:
    profile_module = _load_dashboard_profile_module()
    prompts_path = tmp_path / "prompts"
    prompts_path.mkdir()
    (prompts_path / "sae_lens.json").write_text('{"context_size": 319}\n', encoding="utf-8")

    command = profile_module.build_dashboard_command(
        profile_module.ProfileConfig("profile", 1024, 512),
        layer=9,
        python_executable="python",
        run_root=tmp_path / "runs",
        pretokenized_dataset_path=prompts_path,
        primary_acts_batch_size=128,
        cuda_visible_devices="0",
        dashboard_extra_args=(),
    )

    token_arg_index = command.index("--n-tokens-in-prompt")
    assert command[token_arg_index + 1] == "319"


def test_profile_command_uses_dashboard_extra_arg_prompt_overrides(tmp_path: Path) -> None:
    profile_module = _load_dashboard_profile_module()

    command = profile_module.build_dashboard_command(
        profile_module.ProfileConfig("profile", 1024, 512),
        layer=9,
        python_executable="python",
        run_root=tmp_path / "runs",
        pretokenized_dataset_path=None,
        primary_acts_batch_size=128,
        cuda_visible_devices="0",
        dashboard_extra_args=("--n-prompts-total=2490", "--n-tokens-in-prompt=319"),
    )

    prompts_arg_index = command.index("--n-prompts-total")
    token_arg_index = command.index("--n-tokens-in-prompt")

    assert command[prompts_arg_index + 1] == "2490"
    assert command[token_arg_index + 1] == "319"
    assert command.count("--n-prompts-total") == 1
    assert command.count("--n-tokens-in-prompt") == 1


def test_profile_preset_phase3_legacy_rte_smoke_applies_baseline_args() -> None:
    profile_module = _load_dashboard_profile_module()
    args = profile_module.build_parser().parse_args(["--preset", "phase3-legacy-rte-smoke"])

    preset = profile_module.apply_profile_preset(args)

    assert preset is not None
    assert preset.name == "phase3-legacy-rte-smoke"
    assert args.target_batches == 2
    assert args.summary_warmup_batches == 0
    assert args.prompts_pretokenized_dataset_path is None
    assert args.dashboard_extra_arg[0] == "--run-name-suffix=phase3-legacy-rte-smoke"
    assert (
        f"--prompts-huggingface-dataset-path={profile_module.DEFAULT_PHASE3_RTE_TEXT_DATASET}"
        in args.dashboard_extra_arg
    )
    assert "--prompts-dataset-mode=load_dataset" in args.dashboard_extra_arg
    assert f"--n-prompts-total={profile_module.DEFAULT_PHASE3_PROMPTS_TOTAL}" in args.dashboard_extra_arg
    assert f"--n-tokens-in-prompt={profile_module.DEFAULT_PHASE3_RTE_TOKENS_IN_PROMPT}" in args.dashboard_extra_arg
    assert any(arg.startswith("--saedashboard-repo-root=") for arg in args.dashboard_extra_arg)
    assert any(arg.startswith("--saelens-repo-root=") for arg in args.dashboard_extra_arg)
    assert any(arg.startswith("--neuronpedia-utils-root=") for arg in args.dashboard_extra_arg)
    assert "--runner-implementation=legacy" in args.dashboard_extra_arg


def test_profile_preset_preserves_explicit_pretokenized_dataset_path(tmp_path: Path) -> None:
    profile_module = _load_dashboard_profile_module()
    prompts_path = tmp_path / "prompts"
    args = profile_module.build_parser().parse_args(
        [
            "--preset",
            "phase3-lazy-rte-reduced",
            "--prompts-pretokenized-dataset-path",
            str(prompts_path),
        ]
    )

    preset = profile_module.apply_profile_preset(args)

    assert preset is not None
    assert args.prompts_pretokenized_dataset_path == prompts_path


def test_profile_preset_lazy_rte_uses_upstream_owned_pretokenized_cache() -> None:
    profile_module = _load_dashboard_profile_module()

    lazy_rte = profile_module.PROFILE_PRESETS["phase3-lazy-rte-pretokenized-reduced"]

    assert lazy_rte.pretokenized_dataset_path == profile_module.DEFAULT_PHASE3_RTE_LAZY_PRETOKENIZED_DATASET
    assert lazy_rte.pretokenized_dataset_path.name == "gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts"


def test_profile_dashboard_run_name_uses_extra_suffix() -> None:
    profile_module = _load_dashboard_profile_module()

    assert profile_module.dashboard_run_name([]) == profile_module.DEFAULT_RUN_NAME
    assert profile_module.dashboard_run_name(["--run-name-suffix=arrow-default"]) == (
        f"{profile_module.DEFAULT_RUN_NAME}_arrow-default"
    )
    assert profile_module.dashboard_run_name(["--run-name-suffix", "arrow-default"]) == (
        f"{profile_module.DEFAULT_RUN_NAME}_arrow-default"
    )
    assert (
        profile_module.dashboard_run_name(
            [
                "--neuronpedia-source-set-id=gemmascope-2-transcoder-262k",
                "--run-name-suffix=monology-smoke",
            ]
        )
        == "gemma-3-1b-it_gemmascope-2-transcoder-262k_monology-smoke"
    )
    assert (
        profile_module.dashboard_run_name(
            [
                "--model-name=gemma-3-4b-it",
                "--neuronpedia-source-set-id=gemmascope-2-transcoder-262k",
                "--run-name-suffix=monology-smoke",
            ]
        )
        == "gemma-3-4b-it_gemmascope-2-transcoder-262k_monology-smoke"
    )


def test_profile_dashboard_pipeline_log_path_uses_effective_layer_overrides(tmp_path: Path) -> None:
    profile_module = _load_dashboard_profile_module()
    run_root = tmp_path / "runs"

    default_path = profile_module.dashboard_pipeline_log_path(run_root, [], default_layer=2)
    overridden_path = profile_module.dashboard_pipeline_log_path(
        run_root,
        [
            "--run-name-suffix=phase3-legacy-rte-pretokenized-reduced-l9-b3-20260521",
            "--start-layer=9",
            "--end-layer=9",
        ],
        default_layer=2,
    )

    assert default_path == run_root / profile_module.DEFAULT_RUN_NAME / "run.resume-2-2.log"
    assert overridden_path == (
        run_root
        / f"{profile_module.DEFAULT_RUN_NAME}_phase3-legacy-rte-pretokenized-reduced-l9-b3-20260521"
        / "run.resume-9-9.log"
    )


def test_phase3_profile_preset_pairs_share_reduced_shapes() -> None:
    profile_module = _load_dashboard_profile_module()

    legacy = profile_module.PROFILE_PRESETS["phase3-legacy-monology-reduced"]
    lazy = profile_module.PROFILE_PRESETS["phase3-lazy-monology-reduced"]

    assert legacy.config.n_features_per_batch == lazy.config.n_features_per_batch == 1024
    assert legacy.config.n_prompts_in_forward_pass == lazy.config.n_prompts_in_forward_pass == 256
    assert legacy.target_batches == lazy.target_batches == 4
    assert legacy.pretokenized_dataset_path is None
    assert lazy.pretokenized_dataset_path is None
    assert "--runner-implementation=legacy" in legacy.dashboard_extra_args
    assert "--runner-dashboard-output-format=columnar" in lazy.dashboard_extra_args
    assert "--prompts-dataset-mode=load_dataset" in legacy.dashboard_extra_args
    assert "--prompts-dataset-mode=load_dataset" in lazy.dashboard_extra_args
    assert f"--n-prompts-total={profile_module.DEFAULT_PHASE3_PROMPTS_TOTAL}" in legacy.dashboard_extra_args
    assert f"--n-prompts-total={profile_module.DEFAULT_PHASE3_PROMPTS_TOTAL}" in lazy.dashboard_extra_args
    assert (
        f"--n-tokens-in-prompt={profile_module.DEFAULT_PHASE3_MONOLOGY_TOKENS_IN_PROMPT}" in legacy.dashboard_extra_args
    )
    assert (
        f"--n-tokens-in-prompt={profile_module.DEFAULT_PHASE3_MONOLOGY_TOKENS_IN_PROMPT}" in lazy.dashboard_extra_args
    )
    assert f"--neuronpedia-source-set-id={profile_module.DEFAULT_MONOLOGY_SOURCE_SET_ID}" in legacy.dashboard_extra_args
    assert f"--neuronpedia-source-set-id={profile_module.DEFAULT_MONOLOGY_SOURCE_SET_ID}" in lazy.dashboard_extra_args


def test_phase3_pretokenized_reduced_presets_use_final_prompt_artifacts() -> None:
    profile_module = _load_dashboard_profile_module()

    legacy_rte = profile_module.PROFILE_PRESETS["phase3-legacy-rte-pretokenized-reduced"]
    lazy_rte = profile_module.PROFILE_PRESETS["phase3-lazy-rte-pretokenized-reduced"]
    legacy_monology = profile_module.PROFILE_PRESETS["phase3-legacy-monology-pretokenized-reduced"]
    lazy_monology = profile_module.PROFILE_PRESETS["phase3-lazy-monology-pretokenized-reduced"]

    assert legacy_rte.config.n_features_per_batch == lazy_rte.config.n_features_per_batch == 512
    assert legacy_rte.config.n_prompts_in_forward_pass == lazy_rte.config.n_prompts_in_forward_pass == 128
    assert legacy_rte.target_batches == lazy_rte.target_batches == 4
    assert legacy_rte.pretokenized_dataset_path is None
    assert lazy_rte.pretokenized_dataset_path == profile_module.DEFAULT_PHASE3_RTE_LAZY_PRETOKENIZED_DATASET
    assert "--no-runner-use-cached-activations" in legacy_rte.dashboard_extra_args
    assert "--no-runner-use-cached-activations" in lazy_rte.dashboard_extra_args
    assert (
        f"--prompts-huggingface-dataset-path={profile_module.DEFAULT_PHASE3_RTE_LEGACY_PRETOKENIZED_DATASET}"
        in legacy_rte.dashboard_extra_args
    )
    assert "--prompts-dataset-mode=legacy_jsonl" in legacy_rte.dashboard_extra_args
    assert "--prompts-dataset-mode=load_from_disk" in lazy_rte.dashboard_extra_args

    assert legacy_monology.config.n_features_per_batch == lazy_monology.config.n_features_per_batch == 1024
    assert legacy_monology.config.n_prompts_in_forward_pass == lazy_monology.config.n_prompts_in_forward_pass == 256
    assert legacy_monology.target_batches == lazy_monology.target_batches == 4
    assert legacy_monology.pretokenized_dataset_path is None
    assert lazy_monology.pretokenized_dataset_path == profile_module.DEFAULT_PHASE3_MONOLOGY_PRETOKENIZED_DATASET
    assert "--no-runner-use-cached-activations" in legacy_monology.dashboard_extra_args
    assert "--no-runner-use-cached-activations" in lazy_monology.dashboard_extra_args
    assert (
        f"--prompts-huggingface-dataset-path={profile_module.DEFAULT_PHASE3_MONOLOGY_LEGACY_PRETOKENIZED_DATASET}"
        in legacy_monology.dashboard_extra_args
    )
    assert "--prompts-dataset-mode=legacy_jsonl" in legacy_monology.dashboard_extra_args
    assert "--prompts-dataset-mode=load_from_disk" in lazy_monology.dashboard_extra_args


def test_phase4_current_legacy_presets_omit_detached_repo_overrides() -> None:
    profile_module = _load_dashboard_profile_module()

    current_legacy_rte = profile_module.PROFILE_PRESETS["phase4-current-legacy-rte-pretokenized-reduced"]
    current_legacy_monology = profile_module.PROFILE_PRESETS["phase4-current-legacy-monology-pretokenized-reduced"]

    for preset in (current_legacy_rte, current_legacy_monology):
        assert "--runner-implementation=legacy" in preset.dashboard_extra_args
        assert "--no-runner-use-cached-activations" in preset.dashboard_extra_args
        assert "--legacy-export-bundle-contract=preserved_baseline" in preset.dashboard_extra_args
        assert "--runner-rolling-coefficient-num-threads=4" in preset.dashboard_extra_args
        assert "--prompts-dataset-mode=legacy_jsonl" in preset.dashboard_extra_args
        assert not any(
            arg.startswith("--runner-logits-histogram-compatibility=") for arg in preset.dashboard_extra_args
        )
        assert not any(arg.startswith("--runner-legacy-compatibility=") for arg in preset.dashboard_extra_args)
        assert not any(arg.startswith("--saedashboard-repo-root=") for arg in preset.dashboard_extra_args)
        assert not any(arg.startswith("--saelens-repo-root=") for arg in preset.dashboard_extra_args)
        assert not any(arg.startswith("--neuronpedia-utils-root=") for arg in preset.dashboard_extra_args)

    assert current_legacy_rte.config.n_features_per_batch == 512
    assert current_legacy_rte.config.n_prompts_in_forward_pass == 128
    assert (
        f"--prompts-huggingface-dataset-path={profile_module.DEFAULT_PHASE3_RTE_LEGACY_PRETOKENIZED_DATASET}"
        in current_legacy_rte.dashboard_extra_args
    )
    assert (
        f"--prompts-shared-tokens-file={profile_module.DEFAULT_PHASE4_RTE_SHARED_TOKENS_FILE}"
        in current_legacy_rte.dashboard_extra_args
    )
    assert "--runner-logits-histogram-backend=object" in current_legacy_rte.dashboard_extra_args
    assert current_legacy_monology.config.n_features_per_batch == 1024
    assert current_legacy_monology.config.n_prompts_in_forward_pass == 256
    assert (
        f"--prompts-huggingface-dataset-path={profile_module.DEFAULT_PHASE3_MONOLOGY_LEGACY_PRETOKENIZED_DATASET}"
        in current_legacy_monology.dashboard_extra_args
    )
    assert (
        f"--prompts-shared-tokens-file={profile_module.DEFAULT_PHASE4_MONOLOGY_SHARED_TOKENS_FILE}"
        in current_legacy_monology.dashboard_extra_args
    )
    assert "--runner-logits-histogram-backend=object" in current_legacy_monology.dashboard_extra_args


def test_phase4_import_tolerance_fixture_records_preserved_baseline_contract() -> None:
    fixture_path = (
        Path(__file__).parents[1] / "fixtures" / "neuronpedia_dashboard_phase4" / "import_tolerance_baselines.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert fixture["schema_version"] == 1
    assert fixture["preserved_baseline_lineage"] == {
        "saedashboard": "7886eaa",
        "saelens": "3eea6552",
        "neuronpedia": "5a33f17",
        "interpretune": "fc69a14",
    }
    assert fixture["comparison_tolerances"]["detached_baseline_vs_current_legacy"] == {
        "activation_rows_max_abs_delta": 0,
        "generation_throughput_relative_tolerance": 0.10,
        "import_wall_relative_tolerance": 0.15,
    }

    rte = fixture["scenarios"]["rte_reduced"]
    monology = fixture["scenarios"]["monology_reduced"]

    assert rte["prompt_contract"]["mode"] == "legacy_jsonl"
    assert monology["prompt_contract"]["mode"] == "legacy_jsonl"
    assert (
        rte["legacy_contract"]
        == monology["legacy_contract"]
        == {
            "runner_implementation": "legacy",
            "dashboard_output_format": "legacy_json",
            "sequence_selection_backend": "legacy",
        }
    )
    assert rte["preserved_baseline_result"]["imported_activation_rows"] == 65129
    assert monology["preserved_baseline_result"]["imported_activation_rows"] == 137331


def test_profile_command_defaults_to_load_from_disk_mode_when_prompt_cache_present(tmp_path: Path) -> None:
    profile_module = _load_dashboard_profile_module()

    command = profile_module.build_dashboard_command(
        profile_module.ProfileConfig("profile", 1024, 512),
        layer=9,
        python_executable="python",
        run_root=tmp_path / "runs",
        pretokenized_dataset_path=tmp_path / "prompts",
        primary_acts_batch_size=128,
        cuda_visible_devices="0",
        dashboard_extra_args=(),
    )

    mode_index = command.index("--prompts-dataset-mode")
    assert command[mode_index + 1] == "load_from_disk"


def test_profile_command_allows_explicit_prompt_dataset_mode_override(tmp_path: Path) -> None:
    profile_module = _load_dashboard_profile_module()

    command = profile_module.build_dashboard_command(
        profile_module.ProfileConfig("profile", 1024, 512),
        layer=9,
        python_executable="python",
        run_root=tmp_path / "runs",
        pretokenized_dataset_path=tmp_path / "prompts",
        primary_acts_batch_size=128,
        cuda_visible_devices="0",
        dashboard_extra_args=("--prompts-dataset-mode=legacy_jsonl",),
    )

    mode_index = command.index("--prompts-dataset-mode")
    assert command[mode_index + 1] == "legacy_jsonl"


def test_normalized_prompt_bucket_ceilings_use_quantiles_when_not_explicit(tmp_path: Path) -> None:
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        n_tokens_in_prompt=128,
    )

    ceilings = dashboard_pipeline._normalized_prompt_bucket_ceilings(
        config,
        [40, 50, 60, 64, 64, 64, 65, 70, 80, 90, 110, 120],
    )

    assert ceilings == (60, 64, 80, 120)


def test_profile_import_stage_legacy_includes_conversion_in_activation_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_module = _load_dashboard_profile_module()
    run_root = tmp_path / "runs"
    extra_args = ("--run-name-suffix=legacy-import-profile", "--runner-dashboard-output-format=legacy_json")
    command = profile_module.build_dashboard_command(
        profile_module.ProfileConfig("profile", 2048, 512),
        layer=9,
        python_executable="python",
        run_root=run_root,
        pretokenized_dataset_path=None,
        primary_acts_batch_size=128,
        cuda_visible_devices="0",
        dashboard_extra_args=extra_args,
    )
    output_dir = run_root / profile_module.dashboard_run_name(extra_args) / "layer_9"
    output_dir.mkdir(parents=True)

    monkeypatch.setattr(profile_module, "resolve_local_neuronpedia_db_url", lambda _: "postgres://local")
    monkeypatch.setattr(
        profile_module.dashboard_pipeline,
        "convert_dashboard_output",
        lambda config, *, layer_num, output_dir, logger=None: tmp_path / "export_root",
    )
    monkeypatch.setattr(
        profile_module.dashboard_pipeline,
        "import_neuronpedia_export_bundle_local_db",
        lambda *args, **kwargs: SimpleNamespace(
            imported_row_counts={"Activation": 259607, "Neuron": 8192},
            table_load_seconds={"Activation": 13.24},
            table_import_seconds={"Activation": 42.88},
            table_import_substage_seconds={
                "Activation": {"copy_write": 0.37, "copy_stream_close": 5.59, "insert_from_stage": 15.41}
            },
        ),
    )
    monotonic_values = iter([0.0, 1.0, 11.0, 51.0])
    monkeypatch.setattr(profile_module.time, "monotonic", lambda: next(monotonic_values))

    import_profile = profile_module.profile_import_stage(command, local_db_url=None)

    assert import_profile.mode == "legacy_json"
    assert import_profile.conversion_seconds == pytest.approx(10.0)
    assert import_profile.activation_table_load_seconds == pytest.approx(13.24)
    assert import_profile.activation_load_seconds == pytest.approx(23.24)
    assert import_profile.activation_import_seconds == pytest.approx(42.88)
    assert import_profile.wall_seconds == pytest.approx(51.0)
    assert import_profile.local_db_source_id == "export_root"
    assert import_profile.imported_activation_rows == 259607
    assert import_profile.imported_neuron_rows == 8192
    assert import_profile.imported_row_counts == {"Activation": 259607, "Neuron": 8192}


def test_profile_import_stage_legacy_uses_preserved_baseline_import_preferences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_module = _load_dashboard_profile_module()
    run_root = tmp_path / "runs"
    extra_args = (
        "--run-name-suffix=legacy-contract-profile",
        "--runner-dashboard-output-format=legacy_json",
        "--legacy-export-bundle-contract=preserved_baseline",
    )
    command = profile_module.build_dashboard_command(
        profile_module.ProfileConfig("profile", 2048, 512),
        layer=9,
        python_executable="python",
        run_root=run_root,
        pretokenized_dataset_path=None,
        primary_acts_batch_size=128,
        cuda_visible_devices="0",
        dashboard_extra_args=extra_args,
    )
    output_dir = run_root / profile_module.dashboard_run_name(extra_args) / "layer_9"
    output_dir.mkdir(parents=True)

    monkeypatch.setattr(profile_module, "resolve_local_neuronpedia_db_url", lambda _: "postgres://local")
    monkeypatch.setattr(
        profile_module.dashboard_pipeline,
        "convert_dashboard_output",
        lambda config, *, layer_num, output_dir, logger=None: tmp_path / "export_root",
    )
    captured_import_kwargs: dict[str, object] = {}

    def _fake_import_bundle(*args, **kwargs):
        captured_import_kwargs.update(kwargs)
        return SimpleNamespace(
            imported_row_counts={"Activation": 137331, "Neuron": 0},
            table_load_seconds={"Activation": 9.65},
            table_import_seconds={"Activation": 31.34},
            table_import_substage_seconds={},
        )

    monkeypatch.setattr(
        profile_module.dashboard_pipeline,
        "import_neuronpedia_export_bundle_local_db",
        _fake_import_bundle,
    )
    monotonic_values = iter([0.0, 1.0, 17.28, 58.89])
    monkeypatch.setattr(profile_module.time, "monotonic", lambda: next(monotonic_values))

    import_profile = profile_module.profile_import_stage(command, local_db_url=None)

    assert captured_import_kwargs["prefer_arrow_for_tables"] == ()
    assert captured_import_kwargs["prefer_copy_for_tables"] == ()
    assert import_profile.imported_row_counts == {"Activation": 137331, "Neuron": 0}


def test_profile_import_stage_columnar_skips_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_module = _load_dashboard_profile_module()
    run_root = tmp_path / "runs"
    extra_args = ("--run-name-suffix=columnar-import-profile", "--runner-dashboard-output-format=columnar")
    command = profile_module.build_dashboard_command(
        profile_module.ProfileConfig("profile", 2048, 512),
        layer=9,
        python_executable="python",
        run_root=run_root,
        pretokenized_dataset_path=None,
        primary_acts_batch_size=128,
        cuda_visible_devices="0",
        dashboard_extra_args=extra_args,
    )
    output_dir = run_root / profile_module.dashboard_run_name(extra_args) / "layer_9"
    output_dir.mkdir(parents=True)
    (output_dir / "batch-0.json").write_text(json.dumps({"sae_id_suffix": "columnar-import-profile"}), encoding="utf-8")

    monkeypatch.setattr(profile_module, "resolve_local_neuronpedia_db_url", lambda _: "postgres://local")
    monkeypatch.setattr(
        profile_module.dashboard_pipeline,
        "convert_dashboard_output",
        lambda *args, **kwargs: pytest.fail("columnar import profiling should not convert legacy output"),
    )
    monkeypatch.setattr(profile_module, "uuid4", lambda: SimpleNamespace(hex="feedfacecafebeef1234"))
    captured_import_kwargs: dict[str, object] = {}

    def _fake_import_columnar_dashboard_output(*args, **kwargs):
        captured_import_kwargs.update(kwargs)
        return SimpleNamespace(
            imported_row_counts={"Activation": 259605, "Neuron": 8192},
            table_load_seconds={"Activation": 2.34},
            table_import_seconds={"Activation": 43.87},
            table_import_substage_seconds={
                "Activation": {"copy_write": 3.46, "copy_stream_close": 2.09, "insert_from_stage": 34.84}
            },
        )

    monkeypatch.setattr(
        profile_module.dashboard_pipeline,
        "import_columnar_dashboard_output",
        _fake_import_columnar_dashboard_output,
    )
    monotonic_values = iter([0.0, 50.2])
    monkeypatch.setattr(profile_module.time, "monotonic", lambda: next(monotonic_values))

    import_profile = profile_module.profile_import_stage(command, local_db_url=None)

    assert import_profile.mode == "columnar"
    assert captured_import_kwargs["activation_use_stage_table"] is False
    assert captured_import_kwargs["source_id_override"] == (
        "9-gemmascope-2-transcoder-262k-rte__columnar-import-profile__profile-import-feedfacecafe"
    )
    assert captured_import_kwargs["activation_id_prefix"] == (
        "9-gemmascope-2-transcoder-262k-rte__columnar-import-profile__profile-import-feedfacecafe-activation"
    )
    assert import_profile.conversion_seconds == pytest.approx(0.0)
    assert import_profile.activation_table_load_seconds == pytest.approx(2.34)
    assert import_profile.activation_load_seconds == pytest.approx(2.34)
    assert import_profile.activation_import_seconds == pytest.approx(43.87)
    assert import_profile.wall_seconds == pytest.approx(50.2)
    assert import_profile.local_db_source_id == (
        "9-gemmascope-2-transcoder-262k-rte__columnar-import-profile__profile-import-feedfacecafe"
    )
    assert import_profile.imported_activation_rows == 259605
    assert import_profile.imported_neuron_rows == 8192
    assert import_profile.imported_row_counts == {"Activation": 259605, "Neuron": 8192}


def test_build_dashboard_command_enables_activation_copy_rows_for_columnar_profile_import(
    tmp_path: Path,
) -> None:
    profile_module = _load_dashboard_profile_module()

    command = profile_module.build_dashboard_command(
        profile_module.ProfileConfig("profile", 2048, 512),
        layer=9,
        python_executable="python",
        run_root=tmp_path / "runs",
        pretokenized_dataset_path=None,
        primary_acts_batch_size=128,
        cuda_visible_devices="0",
        dashboard_extra_args=("--runner-dashboard-output-format=columnar",),
        profile_import_stage=True,
    )

    assert "--runner-emit-activation-copy-rows" in command


def test_import_columnar_dashboard_output_can_disable_activation_stage_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        dashboard_pipeline,
        "_resolve_source_id",
        lambda output_dir, layer_num, source_set_id: f"{layer_num}-{source_set_id}",
    )
    monkeypatch.setattr(
        dashboard_pipeline,
        "_build_columnar_token_decoder",
        lambda config: (lambda token_ids: [str(token_id) for token_id in token_ids], 0),
    )

    def _fake_import_saedashboard_columnar_bundle_local_db(columnar_root: Path, **kwargs: object) -> object:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            imported_row_counts={"Activation": 3, "Neuron": 2},
            table_load_seconds={"Activation": 0.1},
            table_import_seconds={"Activation": 0.2},
            table_import_substage_seconds={"Activation": {"copy_write": 0.05}},
        )

    monkeypatch.setattr(
        dashboard_pipeline,
        "import_saedashboard_columnar_bundle_local_db",
        _fake_import_saedashboard_columnar_bundle_local_db,
    )

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=9,
        end_layer=9,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        local_db_url="postgres://local",
        local_db_import_chunk_size=4096,
        runner_dashboard_output_format="columnar",
    )

    summary = dashboard_pipeline.import_columnar_dashboard_output(
        config,
        layer_num=9,
        output_dir=tmp_path / "layer_9",
        activation_use_stage_table=False,
        source_id_override="9-gemmascope-2-transcoder-262k-rte__profile-import-deadbeef",
        activation_id_prefix="9-gemmascope-2-transcoder-262k-rte__profile-import-deadbeef-activation",
    )

    assert captured_kwargs["activation_use_stage_table"] is False
    assert captured_kwargs["chunk_size"] == 4096
    assert captured_kwargs["source_id"] == "9-gemmascope-2-transcoder-262k-rte__profile-import-deadbeef"
    assert captured_kwargs["activation_id_prefix"] == (
        "9-gemmascope-2-transcoder-262k-rte__profile-import-deadbeef-activation"
    )
    assert summary.imported_row_counts == {"Activation": 3, "Neuron": 2}


class _FakeHookPointChoices(str, Enum):
    hook_resid_post = "hook_resid_post"


def test_convert_dashboard_output_passes_model_metadata(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "layer_0"
    dashboard_leaf_dir = output_dir / "leaf"
    dashboard_leaf_dir.mkdir(parents=True)
    (dashboard_leaf_dir / "batch-0.json").write_text(json.dumps({"sae_id_suffix": ""}), encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeModule:
        HOOK_POINT_TYPE_CHOICES = _FakeHookPointChoices

        @staticmethod
        def main(ctx, **params):
            captured["ctx"] = ctx
            captured["params"] = params
            export_dir = (
                Path(str(params["export_root"]))
                / str(params["model_name"])
                / f"{params['layer_num']}-{params['neuronpedia_source_set_id']}"
            )
            export_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dashboard_pipeline, "_load_converter_module", lambda _: FakeModule)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-4b-it",
        model_layers=34,
        sae_set="gemmascope-2-transcoder-16k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-16k",
        neuronpedia_source_set_description="Transcoder - 16k",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/mwhanna/gemma-scope-2-4b-it",
        hf_weights_repo_id="mwhanna/gemma-scope-2-4b-it",
        hf_weights_path_template="transcoder/layer_{layer}/weights.safetensors",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="monology/pile-uncopyrighted",
        start_layer=0,
        end_layer=0,
        sae_path_template="sae/layer_{layer}",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
    )

    export_root = dashboard_pipeline.convert_dashboard_output(
        config,
        layer_num=0,
        output_dir=output_dir,
        logger=logging.getLogger(__name__),
    )

    assert export_root == tmp_path / "exports" / "gemma-3-4b-it" / "0-gemmascope-2-transcoder-16k"
    assert captured["params"] == {
        "saedashboard_output_dir": str(dashboard_leaf_dir),
        "export_root": str(tmp_path / "exports"),
        "creator_name": "Google DeepMind",
        "release_id": "gemma-scope-2",
        "release_title": "Gemma Scope 2",
        "url": "https://huggingface.co/mwhanna/gemma-scope-2-4b-it",
        "model_name": "gemma-3-4b-it",
        "model_layers": 34,
        "neuronpedia_source_set_id": "gemmascope-2-transcoder-16k",
        "neuronpedia_source_set_description": "Transcoder - 16k",
        "hf_weights_repo_id": "mwhanna/gemma-scope-2-4b-it",
        "hf_weights_path": "transcoder/layer_0/weights.safetensors",
        "hook_point": _FakeHookPointChoices.hook_resid_post,
        "layer_num": 0,
        "prompts_huggingface_dataset_path": "monology/pile-uncopyrighted#dataset_mode=load_dataset",
        "n_prompts_total": 24576,
        "n_tokens_in_prompt": 128,
        "zero_out_bos_token": False,
    }


def test_pipeline_config_default_export_root_prefers_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEURONPEDIA_EXPORT_ROOT", str(tmp_path / "cache_exports"))

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-4b-it",
        model_layers=34,
        sae_set="gemmascope-2-transcoder-16k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-16k",
        neuronpedia_source_set_description="Transcoder - 16k",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/mwhanna/gemma-scope-2-4b-it",
        hf_weights_repo_id="mwhanna/gemma-scope-2-4b-it",
        hf_weights_path_template="transcoder/layer_{layer}/weights.safetensors",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="monology/pile-uncopyrighted",
        start_layer=0,
        end_layer=0,
        sae_path_template="sae/layer_{layer}",
        run_root=tmp_path / "runs",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
    )

    assert config.export_root == tmp_path / "cache_exports"


def test_build_generation_env_sets_neuronpedia_export_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IT_NP_CACHE", raising=False)
    monkeypatch.delenv("NEURONPEDIA_EXPORT_ROOT", raising=False)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-4b-it",
        model_layers=34,
        sae_set="gemmascope-2-transcoder-16k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-16k",
        neuronpedia_source_set_description="Transcoder - 16k",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/mwhanna/gemma-scope-2-4b-it",
        hf_weights_repo_id="mwhanna/gemma-scope-2-4b-it",
        hf_weights_path_template="transcoder/layer_{layer}/weights.safetensors",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="monology/pile-uncopyrighted",
        start_layer=0,
        end_layer=0,
        sae_path_template="sae/layer_{layer}",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
    )

    env = dashboard_pipeline._build_generation_env(config)

    assert env["IT_NP_CACHE"] == str(dashboard_pipeline.DEFAULT_IT_NP_CACHE)
    assert env["NEURONPEDIA_EXPORT_ROOT"] == str(config.export_root)


def test_layer_runner_command_includes_bridge_and_custom_dataset_args(tmp_path: Path) -> None:
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_huggingface_dataset_config_name="rte",
        prompts_huggingface_dataset_split="train",
        prompts_pretokenized_dataset_path=tmp_path / "pretokenized_prompts",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        dataset_streaming=False,
        n_features_per_batch=512,
        primary_acts_batch_size=128,
        start_batch=2,
        end_batch=3,
        model_wrapper="bridge",
        bridge_enable_compatibility_mode=True,
        bridge_compatibility_mode_kwargs={"no_processing": True},
        runner_log_resource_snapshots=True,
        runner_log_hook_aliases=True,
        runner_profile_rolling_substages=True,
        runner_cleanup_each_minibatch=True,
        runner_rolling_coefficient_num_threads=4,
        runner_converter_input_artifact_dir=tmp_path / "converter_inputs",
        runner_feature_statistics_backend="arrow",
        runner_logits_histogram_backend="arrow",
        runner_activation_histogram_backend="polars",
        runner_defer_component_construction=True,
        runner_sequence_selection_backend="lazy_gpu",
        runner_prompt_bucket_schedule_file=tmp_path / "selected_bucket_configs.json",
    )

    command = dashboard_pipeline._layer_runner_command(config, layer_num=0, output_dir=tmp_path / "layer_0")

    assert "--model-wrapper=bridge" in command
    assert f"--prompt-dataset-path={tmp_path / 'pretokenized_prompts'}" in command
    assert "--prompt-dataset-mode=load_from_disk" in command
    assert "--prompt-dataset-name=rte" in command
    assert "--prompt-dataset-split=train" in command
    assert not any(part.startswith("--prompt-builder") for part in command)
    assert f"--shared-tokens-file={tmp_path / 'pretokenized_prompts' / 'tokens_24576.pt'}" in command
    assert "--primary-acts-batch-size=128" in command
    assert "--start-batch=2" in command
    assert "--end-batch=3" in command
    assert "--no-dataset-streaming" in command
    assert "--bridge-enable-compatibility-mode" in command
    assert '--bridge-compatibility-mode-kwargs-json={"no_processing":true}' in command
    assert "--log-resource-snapshots" in command
    assert "--log-hook-aliases" in command
    assert "--profile-rolling-substages" in command
    assert "--cleanup-each-minibatch" in command
    assert "--rolling-coefficient-num-threads=4" in command
    assert f"--converter-input-artifact-dir={tmp_path / 'converter_inputs'}" in command


def test_layer_runner_command_uses_explicit_shared_tokens_file(tmp_path: Path) -> None:
    explicit_tokens_path = tmp_path / "baseline_tokens.pt"
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_huggingface_dataset_config_name="rte",
        prompts_huggingface_dataset_split="train",
        prompts_pretokenized_dataset_path=tmp_path / "pretokenized_prompts",
        prompts_shared_tokens_file=explicit_tokens_path,
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        dataset_streaming=False,
        n_features_per_batch=512,
    )

    command = dashboard_pipeline._layer_runner_command(config, layer_num=0, output_dir=tmp_path / "layer_0")

    assert f"--prompt-dataset-path={tmp_path / 'pretokenized_prompts'}" in command
    assert "--prompt-dataset-mode=load_from_disk" in command
    assert f"--shared-tokens-file={explicit_tokens_path}" in command
    assert f"--shared-tokens-file={tmp_path / 'pretokenized_prompts' / 'tokens_24576.pt'}" not in command


def test_layer_runner_command_can_target_legacy_runner(tmp_path: Path) -> None:
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_huggingface_dataset_config_name="rte",
        prompts_huggingface_dataset_split="train",
        prompts_pretokenized_dataset_path=tmp_path / "pretokenized_prompts",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path / "baseline_saedashboard",
        saelens_repo_root=tmp_path / "baseline_saelens",
        neuronpedia_utils_root=tmp_path / "baseline_neuronpedia_utils",
        interpretune_env_file=None,
        dataset_streaming=False,
        n_features_per_batch=128,
        n_prompts_in_forward_pass=32,
        start_batch=0,
        end_batch=1,
        model_wrapper="bridge",
        bridge_enable_compatibility_mode=True,
        runner_log_resource_snapshots=True,
        runner_log_hook_aliases=True,
        runner_log_performance=True,
        runner_profile_rolling_substages=True,
        runner_shuffle_tokens=False,
        runner_implementation="legacy",
        prompts_shared_tokens_file=tmp_path / "baseline_tokens.pt",
        runner_cleanup_each_minibatch=True,
        runner_rolling_coefficient_num_threads=4,
        runner_feature_statistics_backend="arrow",
        runner_logits_histogram_backend="arrow",
        runner_activation_histogram_backend="polars",
        runner_defer_component_construction=True,
        runner_sequence_selection_backend="lazy_gpu",
        runner_dashboard_output_format="columnar",
        runner_use_cached_activations=False,
    )

    command = dashboard_pipeline._layer_runner_command(config, layer_num=0, output_dir=tmp_path / "layer_0")

    assert dashboard_pipeline._resolve_runner_dashboard_output_format(config) == "legacy_json"
    assert "--prompt-dataset-path=aps/super_glue" in command
    assert "--prompt-dataset-mode=load_dataset" in command
    assert "--prompt-dataset-name=rte" in command
    assert "--prompt-dataset-split=train" in command
    assert "--n-features-per-batch=128" in command
    assert "--n-prompts-in-forward-pass=32" in command
    assert "--end-batch=1" in command
    assert "--no-shuffle-tokens" in command
    assert "--log-resource-snapshots" in command
    assert "--log-hook-aliases" in command
    assert "--log-performance" in command
    assert "--profile-rolling-substages" in command
    assert "--cleanup-each-minibatch" in command
    assert "--correlation-accumulation-device=auto" in command
    assert "--rolling-coefficient-num-threads=4" in command
    assert "--logits-histogram-backend=arrow" in command
    assert not any(arg.startswith("--logits-histogram-compatibility=") for arg in command)
    assert not any(arg.startswith("--legacy-compatibility=") for arg in command)
    assert "--no-use-cached-activations" in command
    assert "--sequence-selection-backend=legacy" in command
    assert "--dashboard-output-format=legacy_json" in command
    assert f"--shared-tokens-file={tmp_path / 'baseline_tokens.pt'}" in command
    unsupported_legacy_flags = (
        "--model-wrapper=bridge",
        f"--prompt-dataset-path={tmp_path / 'pretokenized_prompts'}",
        "--prompt-dataset-mode=load_from_disk",
        f"--pretokenized-dataset-path={tmp_path / 'pretokenized_prompts'}",
        f"--shared-tokens-file={tmp_path / 'pretokenized_prompts' / 'tokens_24576.pt'}",
        "--feature-statistics-backend=arrow",
        "--activation-histogram-backend=polars",
        "--defer-component-construction",
        "--sequence-selection-backend=lazy_gpu",
        "--dashboard-output-format=columnar",
        "--columnar-artifact-format=parquet",
    )
    assert not any(flag in command for flag in unsupported_legacy_flags)


def test_layer_runner_command_rejects_legacy_with_load_from_disk_mode(tmp_path: Path) -> None:
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_dataset_mode="load_from_disk",
        prompts_pretokenized_dataset_path=tmp_path / "pretokenized_prompts",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path / "baseline_saedashboard",
        saelens_repo_root=tmp_path / "baseline_saelens",
        neuronpedia_utils_root=tmp_path / "baseline_neuronpedia_utils",
        interpretune_env_file=None,
        runner_implementation="legacy",
    )

    with pytest.raises(ValueError, match="legacy does not accept prompts_dataset_mode='load_from_disk'"):
        dashboard_pipeline._layer_runner_command(config, layer_num=0, output_dir=tmp_path / "layer_0")


def test_layer_runner_command_uses_absolute_legacy_jsonl_path_for_current_runner(tmp_path: Path) -> None:
    legacy_dataset_dir = tmp_path / "legacy_pretok_export"
    legacy_dataset_dir.mkdir()
    (legacy_dataset_dir / "train.jsonl").write_text('{"input_ids": [1, 2, 3]}\n', encoding="utf-8")
    (legacy_dataset_dir / "sae_lens.json").write_text('{"context_size": 319}\n', encoding="utf-8")

    saedashboard_repo_root = tmp_path / "baseline_saedashboard"
    saedashboard_repo_root.mkdir()

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path=str(legacy_dataset_dir),
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=saedashboard_repo_root,
        saelens_repo_root=tmp_path / "baseline_saelens",
        neuronpedia_utils_root=tmp_path / "baseline_neuronpedia_utils",
        interpretune_env_file=None,
        n_features_per_batch=128,
        n_prompts_in_forward_pass=32,
        runner_implementation="legacy",
    )

    command = dashboard_pipeline._layer_runner_command(config, layer_num=0, output_dir=tmp_path / "layer_0")

    assert "--prompt-dataset-mode=legacy_jsonl" in command

    assert f"--prompt-dataset-path={legacy_dataset_dir}" in command
    assert not any(saedashboard_repo_root.glob("it-legacy-local-*"))


def test_layer_runner_command_uses_legacy_dataset_flag_for_detached_baseline_runner(tmp_path: Path) -> None:
    legacy_dataset_dir = tmp_path / "legacy_pretok_export"
    legacy_dataset_dir.mkdir()
    (legacy_dataset_dir / "train.jsonl").write_text('{"input_ids": [1, 2, 3]}\n', encoding="utf-8")
    (legacy_dataset_dir / "sae_lens.json").write_text('{"context_size": 319}\n', encoding="utf-8")

    saedashboard_repo_root = tmp_path / "baseline_saedashboard"
    runner_path = saedashboard_repo_root / "sae_dashboard" / "neuronpedia"
    runner_path.mkdir(parents=True)
    (runner_path / "neuronpedia_runner.py").write_text(
        'parser.add_argument("--dataset-path", required=True)\n',
        encoding="utf-8",
    )

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path=str(legacy_dataset_dir),
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=saedashboard_repo_root,
        saelens_repo_root=tmp_path / "baseline_saelens",
        neuronpedia_utils_root=tmp_path / "baseline_neuronpedia_utils",
        interpretune_env_file=None,
        n_features_per_batch=128,
        n_prompts_in_forward_pass=32,
        runner_shuffle_tokens=False,
        runner_implementation="legacy",
    )

    command = dashboard_pipeline._layer_runner_command(config, layer_num=0, output_dir=tmp_path / "layer_0")

    assert "--prompt-dataset-path=legacy_prompt_dataset" not in command
    assert "--prompt-dataset-mode=legacy_jsonl" not in command
    assert "--no-shuffle-tokens" in command
    assert "--sequence-selection-backend=legacy" not in command
    assert "--dashboard-output-format=legacy_json" not in command
    dataset_flag = next(part for part in command if part.startswith("--dataset-path="))
    materialized_dataset_path = dataset_flag.split("=", maxsplit=1)[1]
    assert materialized_dataset_path != str(legacy_dataset_dir)

    alias_path = saedashboard_repo_root / materialized_dataset_path
    assert alias_path.exists() or alias_path.is_symlink()


def test_parse_args_accepts_runner_shuffle_toggle() -> None:
    args = dashboard_pipeline._parse_args(["--no-runner-shuffle-tokens"])

    assert args.runner_shuffle_tokens is False


def test_prompts_dataset_identifier_records_resolved_dataset_mode(tmp_path: Path) -> None:
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_huggingface_dataset_config_name="rte",
        prompts_huggingface_dataset_split="train",
        prompts_pretokenized_dataset_path=tmp_path / "pretokenized_prompts",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
    )

    assert "#dataset_mode=load_from_disk" in config.prompts_dataset_identifier


def test_layer_runner_command_can_enable_runner_auto_prompt_bucket_schedule(tmp_path: Path) -> None:
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_huggingface_dataset_config_name="rte",
        prompts_huggingface_dataset_split="train",
        prompts_pretokenized_dataset_path=tmp_path / "pretokenized_prompts",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        model_wrapper="bridge",
        runner_auto_prompt_bucket_schedule=True,
        runner_prompt_bucket_ceilings=(64, 128, 192, 256),
        runner_prompt_bucket_scale_limit=4.0,
        runner_prompt_primary_acts_scale_limit=4.0,
        runner_prompt_batch_size_round_to=8,
        deduplicate_shared_prompt_tokens=False,
        strict_shared_prompt_count=True,
    )

    command = dashboard_pipeline._layer_runner_command(config, layer_num=0, output_dir=tmp_path / "layer_0")

    assert "--auto-prompt-bucket-schedule" in command
    assert "--prompt-bucket-ceilings=64,128,192,256" in command
    assert "--prompt-bucket-scale-limit=4.0" in command
    assert "--prompt-primary-acts-scale-limit=4.0" in command
    assert "--prompt-batch-size-round-to=8" in command
    assert "--no-deduplicate-shared-prompt-tokens" in command
    assert "--strict-shared-prompt-count" in command


def test_layer_runner_command_disables_activation_copy_rows_for_generation_only_by_default(tmp_path: Path) -> None:
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        import_to_local_db=False,
        runner_feature_statistics_backend="arrow",
        runner_logits_histogram_backend="arrow",
        runner_activation_histogram_backend="polars",
        runner_defer_component_construction=True,
        runner_sequence_selection_backend="lazy_gpu",
    )

    command = dashboard_pipeline._layer_runner_command(config, layer_num=0, output_dir=tmp_path / "layer_0")

    assert "--dashboard-output-format=columnar" in command
    assert "--columnar-artifact-format=parquet" in command
    assert "--no-columnar-emit-activation-copy-rows" in command
    assert not any(part.startswith("--columnar-activation-copy-model-id=") for part in command)


def test_convert_dashboard_output_uses_structured_dataset_identifier(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "layer_0"
    dashboard_leaf_dir = output_dir / "leaf"
    dashboard_leaf_dir.mkdir(parents=True)
    (dashboard_leaf_dir / "batch-0.json").write_text(json.dumps({"sae_id_suffix": ""}), encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeModule:
        HOOK_POINT_TYPE_CHOICES = _FakeHookPointChoices

        @staticmethod
        def main(ctx, **params):
            captured["params"] = params
            export_dir = (
                Path(str(params["export_root"]))
                / str(params["model_name"])
                / f"{params['layer_num']}-{params['neuronpedia_source_set_id']}"
            )
            export_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dashboard_pipeline, "_load_converter_module", lambda _: FakeModule)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_huggingface_dataset_config_name="rte",
        prompts_huggingface_dataset_split="train",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
    )

    dashboard_pipeline.convert_dashboard_output(
        config,
        layer_num=0,
        output_dir=output_dir,
        logger=logging.getLogger(__name__),
    )

    params = captured["params"]
    assert isinstance(params, dict)
    assert params["prompts_huggingface_dataset_path"] == "aps/super_glue:rte[train]#dataset_mode=load_dataset"


def test_pipeline_config_coerces_optional_log_paths_to_path_objects(tmp_path: Path) -> None:
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=25,
        end_layer=25,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        existing_log_path=cast(Any, str(tmp_path / "run.resume-0-25.log")),
        pipeline_log_path=cast(Any, str(tmp_path / "run.resume-25-25.log")),
    )

    assert config.existing_log_path == tmp_path / "run.resume-0-25.log"
    assert config.pipeline_log_path == tmp_path / "run.resume-25-25.log"


def test_layer_runner_command_includes_clt_local_loader_args(tmp_path: Path) -> None:
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemma-scope-2-1b-it-clt-all",
        neuronpedia_source_set_id="gemmascope-2-clt-262k-rte",
        neuronpedia_source_set_description="CLT - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="clt/width_262k_l0_medium_affine/params_layer_{layer}.safetensors",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_huggingface_dataset_config_name="rte",
        prompts_huggingface_dataset_split="train",
        start_layer=0,
        end_layer=0,
        sae_path_template="/tmp/local_clt_dir",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        use_clt=True,
        clt_dtype="float32",
        clt_weights_filename="params_layer_0.safetensors",
        hf_model_path="google/gemma-3-1b-it",
        model_wrapper="bridge",
    )

    command = dashboard_pipeline._layer_runner_command(config, layer_num=7, output_dir=tmp_path / "layer_7")

    assert "--from-local-sae" in command
    assert "--use-clt" in command
    assert "--clt-layer-idx=7" in command
    assert "--clt-dtype=float32" in command
    assert "--clt-weights-filename=params_layer_0.safetensors" in command
    assert "--hf-model-path=google/gemma-3-1b-it" in command
    assert "--sae-path=/tmp/local_clt_dir" in command


def test_layer_runner_command_includes_np_sae_id_suffix(tmp_path: Path) -> None:
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=7,
        end_layer=7,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        run_name_suffix="phase1-pr-clean",
    )

    command = dashboard_pipeline._layer_runner_command(config, layer_num=7, output_dir=tmp_path / "layer_7")

    assert "--np-sae-id-suffix=phase1-pr-clean" in command


def test_find_existing_export_root_uses_single_matching_bundle(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    model_dir = export_root / "gemma-3-1b-it"
    expected = model_dir / "7-gemmascope-2-transcoder-262k-rte__hook_mlp_in"
    expected.mkdir(parents=True)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=7,
        end_layer=7,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=export_root,
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
    )

    assert dashboard_pipeline._find_existing_export_root(config, layer_num=7) == expected


def test_find_existing_export_root_prefers_matching_run_name_suffix(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    model_dir = export_root / "gemma-3-1b-it"
    (model_dir / "7-gemmascope-2-transcoder-262k-rte__older-lineage").mkdir(parents=True)
    expected = model_dir / "7-gemmascope-2-transcoder-262k-rte__phase1-pr-final-lineage"
    expected.mkdir(parents=True)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=7,
        end_layer=7,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=export_root,
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        run_name_suffix="phase1-pr-final-lineage",
    )

    assert dashboard_pipeline._find_existing_export_root(config, layer_num=7) == expected


def test_resolve_source_id_uses_run_name_suffix_for_columnar_output(tmp_path: Path) -> None:
    output_dir = (
        tmp_path / "runs" / "gemma-3-1b-it_gemmascope-2-transcoder-262k-rte_phase1-pr-clean-steadystate" / "layer_7"
    )
    columnar_leaf = output_dir / "google_gemma-leaf" / "batch-0.columnar"
    columnar_leaf.mkdir(parents=True)

    assert dashboard_pipeline._resolve_source_id(output_dir, 7, "gemmascope-2-transcoder-262k-rte") == (
        "7-gemmascope-2-transcoder-262k-rte__phase1-pr-clean-steadystate"
    )


def test_ensure_shared_prompt_tokens_file_materializes_unique_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pretokenized_prompts"
    Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
            ]
        }
    ).save_to_disk(dataset_path)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_pretokenized_dataset_path=dataset_path,
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        n_prompts_total=3,
        n_tokens_in_prompt=4,
    )

    shared_tokens_file = dashboard_pipeline._ensure_shared_prompt_tokens_file(
        config,
        logger=logging.getLogger(__name__),
    )

    assert shared_tokens_file == dataset_path / "tokens_3.pt"
    assert shared_tokens_file is not None and shared_tokens_file.exists()
    tokens = torch.load(shared_tokens_file)
    assert tokens.shape == (3, 4)
    assert tokens.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    metadata = json.loads((dataset_path / "tokens_3.metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_rows"] == 4
    assert metadata["deduplicate"] is True
    assert metadata["requested_prompts"] == 3
    assert metadata["source_dataset_path"] == str(dataset_path)
    assert metadata["tensor_shape"] == [3, 4]
    assert metadata["tokens_per_prompt"] == 4
    assert metadata["unique_rows"] == 3
    assert metadata["effective_length_min"] == 4
    assert metadata["effective_length_max"] == 4
    assert metadata["effective_length_mean"] == 4.0
    assert metadata["bucket_ceilings"] == [4]
    assert metadata["effective_lengths_file"] == str(dataset_path / "tokens_3.effective_lengths.pt")
    assert metadata["bucket_manifest"] == str(dataset_path / "tokens_3.buckets.json")

    bucket_manifest = json.loads((dataset_path / "tokens_3.buckets.json").read_text(encoding="utf-8"))
    assert bucket_manifest["requested_prompts"] == 3
    assert bucket_manifest["source_dataset_path"] == str(dataset_path)
    assert bucket_manifest["bucket_ceilings"] == [4]
    assert bucket_manifest["buckets"] == [
        {
            "bucket_label": "(0, 4]",
            "effective_length_max": 4,
            "effective_length_mean": 4.0,
            "effective_length_min": 4,
            "effective_lengths_file": str(dataset_path / "tokens_3.bucket_leq_4.effective_lengths.pt"),
            "lower_exclusive": 0,
            "metadata_file": str(dataset_path / "tokens_3.bucket_leq_4.metadata.json"),
            "prompt_count": 3,
            "tokens_file": str(dataset_path / "tokens_3.bucket_leq_4.pt"),
            "upper_inclusive": 4,
        }
    ]


def test_ensure_shared_prompt_tokens_file_strict_count_errors_on_shortfall(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pretokenized_prompts"
    Dataset.from_dict({"input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]]}).save_to_disk(dataset_path)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_pretokenized_dataset_path=dataset_path,
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        n_prompts_total=3,
        n_tokens_in_prompt=4,
        strict_shared_prompt_count=True,
    )

    with pytest.raises(ValueError, match="did not satisfy the requested prompt count"):
        dashboard_pipeline._ensure_shared_prompt_tokens_file(config, logger=logging.getLogger(__name__))


def test_ensure_shared_prompt_tokens_file_can_preserve_duplicate_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pretokenized_prompts"
    Dataset.from_dict({"input_ids": [[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8]]}).save_to_disk(dataset_path)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_pretokenized_dataset_path=dataset_path,
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        n_prompts_total=3,
        n_tokens_in_prompt=4,
        deduplicate_shared_prompt_tokens=False,
        strict_shared_prompt_count=True,
    )

    shared_tokens_file = dashboard_pipeline._ensure_shared_prompt_tokens_file(
        config,
        logger=logging.getLogger(__name__),
    )

    assert shared_tokens_file is not None
    assert torch.load(shared_tokens_file).tolist() == [[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8]]
    metadata = json.loads(shared_tokens_file.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["tensor_shape"] == [3, 4]
    assert metadata["unique_rows"] == 2
    assert metadata["deduplicate"] is False


def test_run_dashboard_pipeline_import_only_uses_existing_export_bundle(tmp_path: Path, monkeypatch) -> None:
    export_root = tmp_path / "exports"
    existing_bundle = export_root / "gemma-3-1b-it" / "0-gemmascope-2-transcoder-262k-rte"
    existing_bundle.mkdir(parents=True)
    imported_paths: list[Path] = []

    monkeypatch.setattr(
        dashboard_pipeline,
        "check_local_neuronpedia_services",
        lambda **_: SimpleNamespace(
            db_available=True,
            webapp_available=True,
            db_error=None,
            webapp_error=None,
            db_url_redacted="postgres://postgres:***@127.0.0.1:5433/postgres",
        ),
    )

    def _fake_import(
        bundle_path: Path,
        local_db_url: str,
        *,
        prefer_arrow_for_tables: tuple[str, ...] = (),
        prefer_copy_for_tables: tuple[str, ...] = (),
    ) -> SimpleNamespace:
        assert prefer_arrow_for_tables == ("Activation",)
        assert prefer_copy_for_tables == ("Activation",)
        imported_paths.append(bundle_path)
        return SimpleNamespace(imported_row_counts={"Source": 1, "Feature": 2})

    monkeypatch.setattr(
        dashboard_pipeline,
        "import_neuronpedia_export_bundle_local_db",
        _fake_import,
    )

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError("generation subprocess should not start in import-only mode")

    monkeypatch.setattr(dashboard_pipeline.subprocess, "Popen", _unexpected_popen)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=export_root,
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        import_only_local_db=True,
    )

    results = dashboard_pipeline.run_dashboard_pipeline(config)

    assert imported_paths == [existing_bundle]
    assert len(results) == 1
    assert results[0].export_root == existing_bundle
    assert results[0].import_summary is not None
    assert results[0].import_summary.imported_row_counts == {"Source": 1, "Feature": 2}
    assert not results[0].skipped


def test_run_dashboard_pipeline_import_only_uses_preserved_baseline_legacy_import_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    export_root = tmp_path / "exports"
    existing_bundle = export_root / "gemma-3-1b-it" / "0-gemmascope-2-transcoder-262k-rte"
    existing_bundle.mkdir(parents=True)
    (existing_bundle / "release.jsonl.gz").write_bytes(b"{}\n")
    imported_paths: list[Path] = []

    monkeypatch.setattr(
        dashboard_pipeline,
        "check_local_neuronpedia_services",
        lambda **_: SimpleNamespace(
            db_available=True,
            webapp_available=True,
            db_error=None,
            webapp_error=None,
            db_url_redacted="postgres://postgres:***@127.0.0.1:5433/postgres",
        ),
    )

    def _fake_import(
        bundle_path: Path,
        local_db_url: str,
        *,
        prefer_arrow_for_tables: tuple[str, ...] = (),
        prefer_copy_for_tables: tuple[str, ...] = (),
    ) -> SimpleNamespace:
        assert prefer_arrow_for_tables == ()
        assert prefer_copy_for_tables == ()
        imported_paths.append(bundle_path)
        return SimpleNamespace(imported_row_counts={"Source": 1, "Feature": 2})

    monkeypatch.setattr(
        dashboard_pipeline,
        "import_neuronpedia_export_bundle_local_db",
        _fake_import,
    )

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError("generation subprocess should not start in import-only mode")

    monkeypatch.setattr(dashboard_pipeline.subprocess, "Popen", _unexpected_popen)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=export_root,
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        import_only_local_db=True,
        legacy_export_bundle_contract="preserved_baseline",
    )

    results = dashboard_pipeline.run_dashboard_pipeline(config)

    assert imported_paths == [existing_bundle]
    assert len(results) == 1
    assert results[0].export_root == existing_bundle
    assert results[0].import_summary is not None
    assert results[0].import_summary.imported_row_counts == {"Source": 1, "Feature": 2}
    assert not results[0].skipped


def test_build_columnar_token_decoder_falls_back_to_slow_tokenizer(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    pretokenized_path = tmp_path / "pretokenized"
    pretokenized_path.mkdir()
    (pretokenized_path / "sae_lens.json").write_text(
        json.dumps({"tokenizer_name": "google/gemma-3-1b-it"}),
        encoding="utf-8",
    )

    class DummyTokenizer:
        pad_token_id = 7

        def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
            return [f"tok-{token_id}" for token_id in token_ids]

    def _fake_from_pretrained(name: str, *args, **kwargs):
        calls.append({"name": name, "kwargs": dict(kwargs)})
        if not kwargs:
            raise ValueError(
                "Couldn't instantiate the backend tokenizer from one of: tokenizer.json, slow tokenizer, or conversion"
            )
        assert kwargs == {"use_fast": False}
        return DummyTokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _fake_from_pretrained)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        prompts_pretokenized_dataset_path=pretokenized_path,
    )

    decode_token_ids, pad_token_id = dashboard_pipeline._build_columnar_token_decoder(config)

    assert decode_token_ids([1, 2, 3]) == ["tok-1", "tok-2", "tok-3"]
    assert pad_token_id == 7
    assert calls == [
        {"name": "google/gemma-3-1b-it", "kwargs": {}},
        {"name": "google/gemma-3-1b-it", "kwargs": {"use_fast": False}},
    ]


def test_build_columnar_token_decoder_uses_release_owner_when_metadata_missing(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    pretokenized_path = tmp_path / "pretokenized"
    pretokenized_path.mkdir()
    (pretokenized_path / "sae_lens.json").write_text(json.dumps({"context_size": 128}), encoding="utf-8")

    class DummyTokenizer:
        pad_token_id = 0

        def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
            return [f"tok-{token_id}" for token_id in token_ids]

    def _fake_from_pretrained(name: str, *args, **kwargs):
        calls.append({"name": name, "kwargs": dict(kwargs)})
        assert not kwargs
        assert name == "google/gemma-3-1b-it"
        return DummyTokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _fake_from_pretrained)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="monology/pile-uncopyrighted",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        prompts_pretokenized_dataset_path=pretokenized_path,
    )

    decode_token_ids, pad_token_id = dashboard_pipeline._build_columnar_token_decoder(config)

    assert decode_token_ids([1, 2]) == ["tok-1", "tok-2"]
    assert pad_token_id == 0
    assert calls == [{"name": "google/gemma-3-1b-it", "kwargs": {}}]


def test_run_dashboard_pipeline_import_only_uses_existing_columnar_output(tmp_path: Path, monkeypatch) -> None:
    imported_layers: list[tuple[int, Path]] = []

    columnar_root = (
        tmp_path
        / "runs"
        / "gemma-3-1b-it_gemmascope-2-transcoder-262k-rte"
        / "layer_0"
        / "google_gemma-3-1b-it_gemma-scope-2-1b-it-transcoders-all_blocks.0.hook_mlp_in_262144"
        / "batch-0.columnar"
    )
    columnar_root.mkdir(parents=True)

    monkeypatch.setattr(
        dashboard_pipeline,
        "check_local_neuronpedia_services",
        lambda **_: SimpleNamespace(
            db_available=True,
            webapp_available=True,
            db_error=None,
            webapp_error=None,
            db_url_redacted="postgres://postgres:***@127.0.0.1:5433/postgres",
        ),
    )

    def _fake_import_columnar(
        config: dashboard_pipeline.NeuronpediaDashboardPipelineConfig,
        *,
        layer_num: int,
        output_dir: Path,
    ) -> SimpleNamespace:
        imported_layers.append((layer_num, output_dir))
        return SimpleNamespace(imported_row_counts={"Source": 1, "Neuron": 2, "Activation": 3})

    def _unexpected_legacy_import(*args, **kwargs):
        raise AssertionError("columnar import-only runs should not invoke legacy export bundle import")

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError("generation subprocess should not start in import-only mode")

    monkeypatch.setattr(dashboard_pipeline, "import_columnar_dashboard_output", _fake_import_columnar)
    monkeypatch.setattr(dashboard_pipeline, "import_neuronpedia_export_bundle_local_db", _unexpected_legacy_import)
    monkeypatch.setattr(dashboard_pipeline.subprocess, "Popen", _unexpected_popen)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        import_only_local_db=True,
        runner_feature_statistics_backend="arrow",
        runner_logits_histogram_backend="arrow",
        runner_activation_histogram_backend="polars",
        runner_defer_component_construction=True,
        runner_sequence_selection_backend="lazy_gpu",
    )

    results = dashboard_pipeline.run_dashboard_pipeline(config)

    expected_output_dir = config.output_dir_for_layer(0)
    assert imported_layers == [(0, expected_output_dir)]
    assert len(results) == 1
    assert results[0].export_root == expected_output_dir
    assert results[0].import_summary is not None
    assert results[0].import_summary.imported_row_counts == {"Source": 1, "Neuron": 2, "Activation": 3}
    assert not results[0].skipped


def test_run_dashboard_pipeline_rejects_import_only_with_skip_import(tmp_path: Path) -> None:
    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        import_to_local_db=False,
        import_only_local_db=True,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        dashboard_pipeline.run_dashboard_pipeline(config)


def test_load_dashboard_pipeline_config_payload_supports_extends(tmp_path: Path) -> None:
    base_config_path = tmp_path / "base.yaml"
    child_config_path = tmp_path / "child.yaml"
    base_config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {
                    "model_name": "gemma-3-1b-it",
                    "model_layers": 26,
                    "sae_set": "gemma-scope-2-1b-it-transcoders-all",
                    "neuronpedia_source_set_id": "gemmascope-2-transcoder-262k-rte",
                    "neuronpedia_source_set_description": "Transcoder - 262k (RTE)",
                    "creator_name": "Google DeepMind",
                    "release_id": "gemma-scope-2",
                    "release_title": "Gemma Scope 2",
                    "release_url": "https://huggingface.co/google/gemma-scope-2-1b-it",
                    "hf_weights_repo_id": "google/gemma-scope-2-1b-it",
                    "hf_weights_path_template": "transcoder_all/layer_{layer}_width_262k_l0_small_affine",
                    "hook_point": "hook_mlp_in",
                    "prompts_huggingface_dataset_path": "aps/super_glue",
                    "prompts_huggingface_dataset_config_name": "rte",
                    "prompts_huggingface_dataset_split": "train",
                    "start_layer": 0,
                    "end_layer": 25,
                    "sae_path_template": "layer_{layer}_width_262k_l0_small_affine",
                    "n_features_per_batch": 512,
                    "bridge_compatibility_mode_kwargs": {"no_processing": True},
                },
                "launcher": {
                    "background": True,
                    "env": {"HF_HOME": "/tmp/hf_home"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    child_config_path.write_text(
        yaml.safe_dump(
            {
                "EXTENDS": str(base_config_path),
                "pipeline": {
                    "n_prompts_in_forward_pass": 256,
                    "primary_acts_batch_size": 64,
                },
                "launcher": {
                    "log_path": "/tmp/dashboard.launch.log",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    payload = dashboard_pipeline.load_dashboard_pipeline_config_payload(child_config_path)

    assert payload["pipeline"]["model_name"] == "gemma-3-1b-it"
    assert payload["pipeline"]["n_features_per_batch"] == 512
    assert payload["pipeline"]["n_prompts_in_forward_pass"] == 256
    assert payload["pipeline"]["primary_acts_batch_size"] == 64
    assert payload["launcher"]["background"] is True
    assert payload["launcher"]["env"]["HF_HOME"] == "/tmp/hf_home"
    assert payload["launcher"]["log_path"] == "/tmp/dashboard.launch.log"


def test_build_dashboard_pipeline_config_uses_yaml_config_and_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "dashboard.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {
                    "model_name": "gemma-3-1b-it",
                    "model_layers": 26,
                    "sae_set": "gemma-scope-2-1b-it-transcoders-all",
                    "neuronpedia_source_set_id": "gemmascope-2-transcoder-262k-rte",
                    "neuronpedia_source_set_description": "Transcoder - 262k (RTE)",
                    "creator_name": "Google DeepMind",
                    "release_id": "gemma-scope-2",
                    "release_title": "Gemma Scope 2",
                    "release_url": "https://huggingface.co/google/gemma-scope-2-1b-it",
                    "hf_weights_repo_id": "google/gemma-scope-2-1b-it",
                    "hf_weights_path_template": "transcoder_all/layer_{layer}_width_262k_l0_small_affine",
                    "hook_point": "hook_mlp_in",
                    "prompts_huggingface_dataset_path": "aps/super_glue",
                    "prompts_huggingface_dataset_config_name": "rte",
                    "prompts_huggingface_dataset_split": "train",
                    "start_layer": 0,
                    "end_layer": 25,
                    "sae_path_template": "layer_{layer}_width_262k_l0_small_affine",
                    "n_features_per_batch": 512,
                    "n_prompts_in_forward_pass": 256,
                    "primary_acts_batch_size": 64,
                    "model_wrapper": "bridge",
                    "bridge_enable_compatibility_mode": True,
                    "dataset_streaming": False,
                    "bridge_compatibility_mode_kwargs": {"no_processing": True},
                    "skip_local_db_import": True,
                },
                "launcher": {
                    "background": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    args = dashboard_pipeline._parse_args(
        [
            "--config",
            str(config_path),
            "--prompts-shared-tokens-file",
            str(tmp_path / "baseline_tokens.pt"),
            "--n-features-per-batch",
            "1024",
            "--n-prompts-in-forward-pass",
            "128",
            "--run-name-suffix",
            "context319-full-prompts",
            "--bridge-compatibility-mode-kwargs-json",
            '{"no_processing": false, "alias_debug": true}',
            "--no-bridge-enable-compatibility-mode",
            "--no-deduplicate-shared-prompt-tokens",
            "--strict-shared-prompt-count",
            "--runner-prompt-bucket-schedule-file",
            str(tmp_path / "selected_bucket_configs.json"),
            "--runner-auto-prompt-bucket-schedule",
            "--runner-prompt-bucket-ceilings",
            "64,128,192,256",
            "--runner-prompt-bucket-scale-limit",
            "3.5",
            "--runner-prompt-primary-acts-scale-limit",
            "4.5",
            "--runner-prompt-batch-size-round-to",
            "16",
            "--runner-dashboard-output-format",
            "columnar",
            "--legacy-export-bundle-contract",
            "preserved_baseline",
            "--local-db-import-chunk-size",
            "4096",
            "--runner-profile-rolling-substages",
            "--runner-cleanup-each-minibatch",
            "--runner-correlation-accumulation-device",
            "cpu",
            "--runner-rolling-coefficient-num-threads",
            "4",
        ]
    )

    config = dashboard_pipeline._build_dashboard_pipeline_config(args)

    assert config.model_name == "gemma-3-1b-it"
    assert config.n_features_per_batch == 1024
    assert config.n_prompts_in_forward_pass == 128
    assert config.primary_acts_batch_size == 64
    assert config.model_wrapper == "bridge"
    assert config.run_name == "gemma-3-1b-it_gemmascope-2-transcoder-262k-rte_context319-full-prompts"
    assert config.prompts_shared_tokens_file == tmp_path / "baseline_tokens.pt"
    assert config.shared_prompt_tokens_file == tmp_path / "baseline_tokens.pt"
    assert config.bridge_enable_compatibility_mode is False
    assert config.bridge_compatibility_mode_kwargs == {"no_processing": False, "alias_debug": True}
    assert config.import_to_local_db is False
    assert config.dataset_streaming is False
    assert config.deduplicate_shared_prompt_tokens is False
    assert config.strict_shared_prompt_count is True
    assert config.runner_prompt_bucket_schedule_file == tmp_path / "selected_bucket_configs.json"
    assert config.runner_auto_prompt_bucket_schedule is True
    assert config.runner_prompt_bucket_ceilings == (64, 128, 192, 256)
    assert config.runner_prompt_bucket_scale_limit == 3.5
    assert config.runner_prompt_primary_acts_scale_limit == 4.5
    assert config.runner_prompt_batch_size_round_to == 16
    assert config.runner_dashboard_output_format == "columnar"
    assert config.legacy_export_bundle_contract == "preserved_baseline"
    assert config.runner_columnar_artifact_format == "parquet"
    assert config.runner_emit_activation_copy_rows is None
    assert config.local_db_import_chunk_size == 4096
    assert config.runner_profile_rolling_substages is True
    assert config.runner_cleanup_each_minibatch is True
    assert config.runner_correlation_accumulation_device == "cpu"
    assert config.runner_rolling_coefficient_num_threads == 4


def test_run_dashboard_pipeline_skips_conversion_for_columnar_generation_only(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    test_logger = logging.getLogger("test_run_dashboard_pipeline_skips_conversion_for_columnar_generation_only")
    test_logger.handlers.clear()
    test_logger.propagate = True
    test_logger.setLevel(logging.INFO)
    launched_commands: list[list[str]] = []

    monkeypatch.setattr(dashboard_pipeline, "_configure_logger", lambda _: test_logger)
    monkeypatch.setattr(dashboard_pipeline, "_build_generation_env", lambda _: {})
    monkeypatch.setattr(dashboard_pipeline, "_monitor_process", lambda *args, **kwargs: 0)

    def _fake_popen(command, *args, **kwargs):
        launched_commands.append(command)
        output_arg = next(part for part in command if part.startswith("--output-dir="))
        fake_leaf_dir = Path(output_arg.split("=", 1)[1]) / "model_source"
        fake_leaf_dir.mkdir(parents=True)
        (fake_leaf_dir / "batch-0.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(pid=12345)

    def _unexpected_convert(*args, **kwargs):
        raise AssertionError("columnar generation-only runs should not invoke the legacy converter")

    monkeypatch.setattr(dashboard_pipeline.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dashboard_pipeline, "convert_dashboard_output", _unexpected_convert)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        pipeline_log_path=tmp_path / "run.log",
        import_to_local_db=False,
        runner_feature_statistics_backend="arrow",
        runner_logits_histogram_backend="arrow",
        runner_activation_histogram_backend="polars",
        runner_defer_component_construction=True,
        runner_sequence_selection_backend="lazy_gpu",
    )

    caplog.set_level(logging.INFO, logger=test_logger.name)

    results = dashboard_pipeline.run_dashboard_pipeline(config)

    assert len(results) == 1
    assert results[0].export_root is None
    assert not results[0].skipped
    assert launched_commands
    assert "--dashboard-output-format=columnar" in launched_commands[0]
    assert "--correlation-accumulation-device=auto" in launched_commands[0]
    assert "Skipping legacy Neuronpedia conversion for columnar dashboard output" in caplog.text


def test_run_dashboard_pipeline_skips_conversion_for_legacy_generation_only(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    test_logger = logging.getLogger("test_run_dashboard_pipeline_skips_conversion_for_legacy_generation_only")
    test_logger.handlers.clear()
    test_logger.propagate = True
    test_logger.setLevel(logging.INFO)
    launched_commands: list[list[str]] = []

    monkeypatch.setattr(dashboard_pipeline, "_configure_logger", lambda _: test_logger)
    monkeypatch.setattr(dashboard_pipeline, "_build_generation_env", lambda _: {})
    monkeypatch.setattr(dashboard_pipeline, "_monitor_process", lambda *args, **kwargs: 0)

    def _fake_popen(command, *args, **kwargs):
        launched_commands.append(command)
        output_arg = next(part for part in command if part.startswith("--output-dir="))
        fake_leaf_dir = Path(output_arg.split("=", 1)[1]) / "model_source"
        fake_leaf_dir.mkdir(parents=True)
        (fake_leaf_dir / "batch-0.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(pid=12345)

    def _unexpected_convert(*args, **kwargs):
        raise AssertionError("legacy generation-only runs should not invoke conversion")

    monkeypatch.setattr(dashboard_pipeline.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dashboard_pipeline, "convert_dashboard_output", _unexpected_convert)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        pipeline_log_path=tmp_path / "run.log",
        import_to_local_db=False,
        runner_implementation="legacy",
    )

    caplog.set_level(logging.INFO, logger=test_logger.name)

    results = dashboard_pipeline.run_dashboard_pipeline(config)

    assert len(results) == 1
    assert results[0].export_root is None
    assert not results[0].skipped
    assert launched_commands
    assert "--sequence-selection-backend=legacy" in launched_commands[0]
    assert "--dashboard-output-format=legacy_json" in launched_commands[0]
    assert "Skipping Neuronpedia conversion for legacy generation-only" in caplog.text


def test_convert_dashboard_output_filters_new_kwargs_for_legacy_converter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dashboard_leaf_dir = tmp_path / "runs" / "layer_2" / "model_source"
    dashboard_leaf_dir.mkdir(parents=True)
    (dashboard_leaf_dir / "batch-0.json").write_text('{"sae_id_suffix":"smoke"}', encoding="utf-8")
    seen: dict[str, object] = {}

    def _fake_main(
        ctx,
        *,
        saedashboard_output_dir,
        creator_name,
        release_id,
        release_title,
        url,
        model_name,
        neuronpedia_source_set_id,
        neuronpedia_source_set_description,
        hf_weights_repo_id,
        hf_weights_path,
        hook_point,
        layer_num,
        prompts_huggingface_dataset_path,
        n_prompts_total,
        n_tokens_in_prompt,
        zero_out_bos_token,
    ):
        seen.update(ctx.params)
        export_dir = Path(fake_module.OUTPUT_DIR) / model_name / f"{layer_num}-{neuronpedia_source_set_id}__smoke"
        export_dir.mkdir(parents=True)
        assert Path(saedashboard_output_dir) == dashboard_leaf_dir
        assert creator_name == "Google DeepMind"
        assert release_id == "gemma-scope-2"
        assert release_title == "Gemma Scope 2"
        assert url == "https://huggingface.co/google/gemma-scope-2-1b-it"
        assert hf_weights_repo_id == "google/gemma-scope-2-1b-it"
        assert hf_weights_path == "transcoder_all/layer_2_width_262k_l0_small_affine"
        assert hook_point == "hook_mlp_in"
        assert prompts_huggingface_dataset_path == "aps/super_glue#dataset_mode=load_dataset"
        assert n_prompts_total == 8
        assert n_tokens_in_prompt == 32
        assert zero_out_bos_token is False

    fake_module = SimpleNamespace(
        OUTPUT_DIR=tmp_path / "wrong-export-root",
        HOOK_POINT_TYPE_CHOICES=lambda value: value,
        main=_fake_main,
        CONVERSION_DEBUG_CALLBACK=None,
    )
    monkeypatch.setattr(dashboard_pipeline, "_load_converter_module", lambda _: fake_module)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_mlp_in",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=2,
        end_layer=2,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        pipeline_log_path=tmp_path / "run.log",
        n_prompts_total=8,
        n_tokens_in_prompt=32,
    )

    export_root = dashboard_pipeline.convert_dashboard_output(config, layer_num=2, output_dir=dashboard_leaf_dir.parent)

    assert export_root == tmp_path / "exports" / "gemma-3-1b-it" / "2-gemmascope-2-transcoder-262k-rte__smoke"
    assert fake_module.OUTPUT_DIR == str(tmp_path / "exports")
    assert "emit_arrow" not in seen
    assert "export_root" not in seen
    assert "model_layers" not in seen


def test_convert_dashboard_output_sets_preserved_baseline_contract_for_legacy_converter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dashboard_leaf_dir = tmp_path / "runs" / "layer_2" / "model_source"
    dashboard_leaf_dir.mkdir(parents=True)
    (dashboard_leaf_dir / "batch-0.json").write_text('{"sae_id_suffix":"smoke"}', encoding="utf-8")
    seen: dict[str, object] = {}

    def _fake_main(ctx, **kwargs):
        seen.update(ctx.params)
        export_dir = Path(fake_module.OUTPUT_DIR) / kwargs["model_name"] / "2-gemmascope-2-transcoder-262k-rte__smoke"
        export_dir.mkdir(parents=True)

    fake_module = SimpleNamespace(
        OUTPUT_DIR=tmp_path / "wrong-export-root",
        HOOK_POINT_TYPE_CHOICES=lambda value: value,
        main=_fake_main,
        CONVERSION_DEBUG_CALLBACK=None,
    )
    monkeypatch.setattr(dashboard_pipeline, "_load_converter_module", lambda _: fake_module)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_mlp_in",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=2,
        end_layer=2,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        pipeline_log_path=tmp_path / "run.log",
        legacy_export_bundle_contract="preserved_baseline",
    )

    export_root = dashboard_pipeline.convert_dashboard_output(config, layer_num=2, output_dir=dashboard_leaf_dir.parent)

    assert export_root == tmp_path / "exports" / "gemma-3-1b-it" / "2-gemmascope-2-transcoder-262k-rte__smoke"
    assert seen["emit_arrow"] is False


def test_legacy_generation_converts_when_import_requested_but_db_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    test_logger = logging.getLogger("test_legacy_generation_converts_when_import_requested_but_db_unavailable")
    test_logger.handlers.clear()
    test_logger.propagate = True
    test_logger.setLevel(logging.INFO)
    converted_layers: list[tuple[int, Path]] = []

    monkeypatch.setattr(dashboard_pipeline, "_configure_logger", lambda _: test_logger)
    monkeypatch.setattr(dashboard_pipeline, "_build_generation_env", lambda _: {})
    monkeypatch.setattr(dashboard_pipeline, "_monitor_process", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        dashboard_pipeline,
        "check_local_neuronpedia_services",
        lambda **_: SimpleNamespace(
            db_available=False,
            webapp_available=False,
            db_error="no local db",
            webapp_error="no local webapp",
            db_url_redacted="postgres://postgres:***@127.0.0.1:5433/postgres",
        ),
    )

    def _fake_popen(command, *args, **kwargs):
        output_arg = next(part for part in command if part.startswith("--output-dir="))
        fake_leaf_dir = Path(output_arg.split("=", 1)[1]) / "model_source"
        fake_leaf_dir.mkdir(parents=True)
        (fake_leaf_dir / "batch-0.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(pid=12345)

    def _fake_convert(
        config: NeuronpediaDashboardPipelineConfig,
        *,
        layer_num: int,
        output_dir: Path,
        logger: logging.Logger | None = None,
    ) -> Path:
        converted_layers.append((layer_num, output_dir))
        export_root = config.export_root / config.model_name / f"{layer_num}-{config.neuronpedia_source_set_id}"
        export_root.mkdir(parents=True)
        return export_root

    monkeypatch.setattr(dashboard_pipeline.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dashboard_pipeline, "convert_dashboard_output", _fake_convert)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        pipeline_log_path=tmp_path / "run.log",
        import_to_local_db=True,
        runner_implementation="legacy",
    )

    results = dashboard_pipeline.run_dashboard_pipeline(config)

    assert converted_layers == [(0, tmp_path / "runs" / config.run_name / "layer_0")]
    assert results[0].export_root == tmp_path / "exports" / "gemma-3-1b-it" / "0-gemmascope-2-transcoder-262k-rte"
    assert results[0].import_summary is None


def test_run_dashboard_pipeline_imports_columnar_output_directly(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    test_logger = logging.getLogger("test_run_dashboard_pipeline_imports_columnar_output_directly")
    test_logger.handlers.clear()
    test_logger.propagate = True
    test_logger.setLevel(logging.INFO)
    launched_commands: list[list[str]] = []
    imported_layers: list[tuple[int, Path]] = []

    monkeypatch.setattr(dashboard_pipeline, "_configure_logger", lambda _: test_logger)
    monkeypatch.setattr(dashboard_pipeline, "_build_generation_env", lambda _: {})
    monkeypatch.setattr(dashboard_pipeline, "_monitor_process", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        dashboard_pipeline,
        "check_local_neuronpedia_services",
        lambda **_: SimpleNamespace(
            db_available=True,
            webapp_available=True,
            db_error=None,
            webapp_error=None,
            db_url_redacted="postgres://postgres:***@127.0.0.1:5433/postgres",
        ),
    )

    def _fake_popen(command, *args, **kwargs):
        launched_commands.append(command)
        return SimpleNamespace(pid=12345)

    def _unexpected_convert(*args, **kwargs):
        raise AssertionError("columnar local DB imports should not invoke the legacy converter")

    def _fake_import_columnar(
        config: NeuronpediaDashboardPipelineConfig,
        *,
        layer_num: int,
        output_dir: Path,
    ) -> SimpleNamespace:
        imported_layers.append((layer_num, output_dir))
        return SimpleNamespace(imported_row_counts={"Source": 1, "Neuron": 2, "Activation": 3})

    monkeypatch.setattr(dashboard_pipeline.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dashboard_pipeline, "convert_dashboard_output", _unexpected_convert)
    monkeypatch.setattr(dashboard_pipeline, "import_columnar_dashboard_output", _fake_import_columnar)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        pipeline_log_path=tmp_path / "run.log",
        local_db_url="postgres://postgres:postgres@127.0.0.1:5433/postgres",
        runner_feature_statistics_backend="arrow",
        runner_logits_histogram_backend="arrow",
        runner_activation_histogram_backend="polars",
        runner_defer_component_construction=True,
        runner_sequence_selection_backend="lazy_gpu",
    )

    caplog.set_level(logging.INFO, logger=test_logger.name)

    results = dashboard_pipeline.run_dashboard_pipeline(config)

    expected_output_dir = config.output_dir_for_layer(0)
    assert len(results) == 1
    assert results[0].export_root == expected_output_dir
    assert results[0].import_summary is not None
    assert results[0].import_summary.imported_row_counts == {"Source": 1, "Neuron": 2, "Activation": 3}
    assert imported_layers == [(0, expected_output_dir)]
    assert launched_commands
    assert "--dashboard-output-format=columnar" in launched_commands[0]
    assert "Imported columnar layer=0 into local DB" in caplog.text


def test_run_dashboard_pipeline_warns_when_requested_range_is_already_complete(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    existing_log_path = tmp_path / "run.log"
    existing_log_path.write_text(
        "DONE layer=0 elapsed_seconds=1.0\nDONE layer=1 elapsed_seconds=1.0\n",
        encoding="utf-8",
    )
    pipeline_log_path = tmp_path / "run.resume-0-1.log"

    test_logger = logging.getLogger("test_run_dashboard_pipeline_warns_when_requested_range_is_already_complete")
    test_logger.handlers.clear()
    test_logger.propagate = True
    test_logger.setLevel(logging.INFO)

    monkeypatch.setattr(dashboard_pipeline, "_configure_logger", lambda _: test_logger)
    monkeypatch.setattr(dashboard_pipeline, "_build_generation_env", lambda _: {})
    monkeypatch.setattr(dashboard_pipeline, "_ensure_shared_prompt_tokens_file", lambda *args, **kwargs: None)

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError("generation subprocess should not start when the requested range is already complete")

    monkeypatch.setattr(dashboard_pipeline.subprocess, "Popen", _unexpected_popen)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=1,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        existing_log_path=existing_log_path,
        pipeline_log_path=pipeline_log_path,
        import_to_local_db=False,
    )

    caplog.set_level(logging.INFO, logger=test_logger.name)

    results = dashboard_pipeline.run_dashboard_pipeline(config)

    assert len(results) == 2
    assert all(result.skipped for result in results)
    assert "Requested layer range 0-1 is already complete in existing logs" in caplog.text
    assert "Use --run-name-suffix or --run-root for a fresh run lineage" in caplog.text


def test_run_dashboard_pipeline_defers_shared_prompt_prep_to_runner_for_auto_schedule(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    existing_log_path = tmp_path / "run.log"
    existing_log_path.write_text("DONE layer=0 elapsed_seconds=1.0\n", encoding="utf-8")
    pipeline_log_path = tmp_path / "run.resume-0-0.log"

    test_logger = logging.getLogger("test_run_dashboard_pipeline_defers_shared_prompt_prep_to_runner_for_auto_schedule")
    test_logger.handlers.clear()
    test_logger.propagate = True
    test_logger.setLevel(logging.INFO)

    monkeypatch.setattr(dashboard_pipeline, "_configure_logger", lambda _: test_logger)
    monkeypatch.setattr(dashboard_pipeline, "_build_generation_env", lambda _: {})

    def _unexpected_shared_prompt_prep(*args, **kwargs):
        raise AssertionError("shared prompt tokens should be prepared by the runner for the auto-schedule path")

    monkeypatch.setattr(dashboard_pipeline, "_ensure_shared_prompt_tokens_file", _unexpected_shared_prompt_prep)

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError("generation subprocess should not start when the requested range is already complete")

    monkeypatch.setattr(dashboard_pipeline.subprocess, "Popen", _unexpected_popen)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        prompts_huggingface_dataset_config_name="rte",
        prompts_huggingface_dataset_split="train",
        prompts_pretokenized_dataset_path=tmp_path / "pretokenized_prompts",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        existing_log_path=existing_log_path,
        pipeline_log_path=pipeline_log_path,
        import_to_local_db=False,
        runner_auto_prompt_bucket_schedule=True,
    )

    caplog.set_level(logging.INFO, logger=test_logger.name)

    results = dashboard_pipeline.run_dashboard_pipeline(config)

    assert len(results) == 1
    assert results[0].skipped
    assert "Deferring shared prompt token/effective-length preparation to the runner" in caplog.text


def test_load_dashboard_launcher_settings_defaults_log_path_from_pipeline_config(tmp_path: Path) -> None:
    config_path = tmp_path / "dashboard.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {
                    "model_name": "gemma-3-1b-it",
                    "model_layers": 26,
                    "sae_set": "gemma-scope-2-1b-it-transcoders-all",
                    "neuronpedia_source_set_id": "gemmascope-2-transcoder-262k-rte",
                    "neuronpedia_source_set_description": "Transcoder - 262k (RTE)",
                    "creator_name": "Google DeepMind",
                    "release_id": "gemma-scope-2",
                    "release_title": "Gemma Scope 2",
                    "release_url": "https://huggingface.co/google/gemma-scope-2-1b-it",
                    "hf_weights_repo_id": "google/gemma-scope-2-1b-it",
                    "hf_weights_path_template": "transcoder_all/layer_{layer}_width_262k_l0_small_affine",
                    "hook_point": "hook_mlp_in",
                    "prompts_huggingface_dataset_path": "aps/super_glue",
                    "start_layer": 0,
                    "end_layer": 25,
                    "sae_path_template": "layer_{layer}_width_262k_l0_small_affine",
                    "run_root": str(tmp_path / "runs"),
                },
                "launcher": {
                    "background": True,
                    "env": {"HF_HOME": "/tmp/hf_home"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = dashboard_pipeline._build_dashboard_pipeline_config(
        dashboard_pipeline._parse_args(["--config", str(config_path)])
    )

    launcher_settings = dashboard_pipeline.load_dashboard_launcher_settings(config_path, pipeline_config=config)

    assert launcher_settings["background"] is True
    assert launcher_settings["env"] == {"HF_HOME": "/tmp/hf_home"}
    assert launcher_settings["log_path"] is not None
    assert launcher_settings["log_path"].parent == config.run_directory
    assert launcher_settings["log_path"].name.startswith("launcher.")


def test_load_dashboard_launcher_settings_accepts_worker_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "dashboard.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {
                    "model_name": "gemma-3-1b-it",
                    "model_layers": 26,
                    "sae_set": "gemma-scope-2-1b-it-transcoders-all",
                    "neuronpedia_source_set_id": "gemmascope-2-transcoder-262k-rte",
                    "neuronpedia_source_set_description": "Transcoder - 262k (RTE)",
                    "creator_name": "Google DeepMind",
                    "release_id": "gemma-scope-2",
                    "release_title": "Gemma Scope 2",
                    "release_url": "https://huggingface.co/google/gemma-scope-2-1b-it",
                    "hf_weights_repo_id": "google/gemma-scope-2-1b-it",
                    "hf_weights_path_template": "transcoder_all/layer_{layer}_width_262k_l0_small_affine",
                    "hook_point": "hook_mlp_in",
                    "prompts_huggingface_dataset_path": "aps/super_glue",
                    "start_layer": 0,
                    "end_layer": 25,
                    "sae_path_template": "layer_{layer}_width_262k_l0_small_affine",
                    "run_root": str(tmp_path / "runs"),
                },
                "launcher": {
                    "monitor": True,
                    "monitor_heartbeat_seconds": 7,
                    "workers": [
                        {
                            "id": "gpu1",
                            "cuda_visible_devices": "1",
                            "start_layer": 3,
                            "n_features_per_batch": 256,
                            "n_prompts_in_forward_pass": 64,
                            "primary_acts_batch_size": 16,
                        },
                        {
                            "id": "gpu0",
                            "cuda_visible_devices": "0",
                            "n_features_per_batch": 1024,
                            "n_prompts_in_forward_pass": 256,
                            "primary_acts_batch_size": 64,
                        },
                    ],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    launcher_settings = dashboard_pipeline.load_dashboard_launcher_settings(config_path)

    assert launcher_settings["monitor"] is True
    assert launcher_settings["monitor_heartbeat_seconds"] == 7
    assert launcher_settings["workers"] == [
        {
            "id": "gpu1",
            "cuda_visible_devices": "1",
            "start_layer": 3,
            "n_features_per_batch": 256,
            "n_prompts_in_forward_pass": 64,
            "primary_acts_batch_size": 16,
        },
        {
            "id": "gpu0",
            "cuda_visible_devices": "0",
            "n_features_per_batch": 1024,
            "n_prompts_in_forward_pass": 256,
            "primary_acts_batch_size": 64,
        },
    ]


def test_launcher_worker_args_use_worker_log_and_locking(tmp_path: Path) -> None:
    launcher_module = _load_dashboard_launcher_module()
    worker_args = launcher_module._worker_passthrough_args(
        ["--run-name-suffix", "context319-full-prompts"],
        {
            "id": "gpu1",
            "cuda_visible_devices": "1",
            "start_layer": 3,
            "n_features_per_batch": 256,
            "n_prompts_in_forward_pass": 64,
            "primary_acts_batch_size": 16,
        },
    )

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=25,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        run_name_suffix="context319-full-prompts",
        worker_id="gpu1",
        enable_layer_locks=True,
        cuda_visible_devices="1",
        n_features_per_batch=256,
        n_prompts_in_forward_pass=64,
        primary_acts_batch_size=16,
    )

    assert worker_args == [
        "--run-name-suffix",
        "context319-full-prompts",
        "--worker-id",
        "gpu1",
        "--enable-layer-locks",
        "--cuda-visible-devices",
        "1",
        "--start-layer",
        "3",
        "--n-features-per-batch",
        "256",
        "--n-prompts-in-forward-pass",
        "64",
        "--primary-acts-batch-size",
        "16",
    ]
    assert config.pipeline_log_path == config.run_directory / "run.gpu1.resume-0-25.log"


def test_launcher_oom_reduction_halves_primary_then_prompt_batch() -> None:
    launcher_module = _load_dashboard_launcher_module()
    worker = {
        "id": "gpu0",
        "cuda_visible_devices": "0",
        "n_features_per_batch": 1024,
        "n_prompts_in_forward_pass": 512,
        "primary_acts_batch_size": 128,
    }

    first_restart = launcher_module._worker_with_oom_reduction(worker, 1)
    assert first_restart is not None
    assert first_restart["primary_acts_batch_size"] == 64
    assert first_restart["n_prompts_in_forward_pass"] == 512

    second_restart = launcher_module._worker_with_oom_reduction(first_restart, 2)
    assert second_restart is not None
    assert second_restart["primary_acts_batch_size"] == 64
    assert second_restart["n_prompts_in_forward_pass"] == 256

    assert launcher_module._worker_with_oom_reduction(second_restart, 3) is None


def test_launcher_log_exit_code_scans_from_offset(tmp_path: Path) -> None:
    launcher_module = _load_dashboard_launcher_module()
    log_path = tmp_path / "launcher.gpu0.log"
    log_path.write_text("old Diagnostics reason=nonzero-exit-1 pid=123\n", encoding="utf-8")
    offset = log_path.stat().st_size

    assert launcher_module._launcher_log_exit_code(log_path) == 1
    assert launcher_module._launcher_log_exit_code(log_path, start_offset=offset) is None

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("RuntimeError: Dashboard generation failed for layer 4 with exit code -9\n")

    assert launcher_module._launcher_log_exit_code(log_path, start_offset=offset) == -9


def test_monitor_treats_initial_sigkill_exit_as_oom_like_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher_module = _load_dashboard_launcher_module()
    worker = {
        "id": "gpu0",
        "cuda_visible_devices": "0",
        "start_layer": 4,
        "n_features_per_batch": 1024,
        "n_prompts_in_forward_pass": 512,
        "primary_acts_batch_size": 256,
    }
    pipeline_log_path = tmp_path / "run.gpu0.resume-4-25.log"
    pipeline_log_path.write_text("healthy batch\n", encoding="utf-8")
    launcher_log_path = tmp_path / "launcher.gpu0.20260503_140904.log"
    launcher_log_path.write_text(
        "RuntimeError: Dashboard generation failed for layer 4 with exit code -9\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "launcher.workers.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "worker_id": "gpu0",
                    "pid": 1706986,
                    "pipeline_log": str(pipeline_log_path),
                    "launcher_log": str(launcher_log_path),
                    "worker": worker,
                }
            ]
        ),
        encoding="utf-8",
    )
    worker_config = SimpleNamespace(
        worker_id="gpu0",
        pipeline_log_path=pipeline_log_path,
        run_directory=tmp_path,
        start_layer=4,
        end_layer=25,
        cuda_visible_devices="0",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=512,
        primary_acts_batch_size=256,
    )
    restarted_workers: list[dict[str, Any]] = []

    monkeypatch.setattr(launcher_module, "_build_pipeline_config", lambda *args, **kwargs: worker_config)
    monkeypatch.setattr(launcher_module, "_pid_is_running", lambda pid: False)

    def _fake_restart_worker(**kwargs):
        state = kwargs["state"]
        restarted_workers.append(dict(state["worker"]))
        state["completed"] = True
        state["pid"] = 1707999
        return SimpleNamespace(pid=1707999)

    monkeypatch.setattr(launcher_module, "_restart_worker", _fake_restart_worker)

    result = launcher_module._run_monitor_loop(
        config_path=tmp_path / "dashboard.yaml",
        passthrough_args=[],
        manifest_path=manifest_path,
        heartbeat_seconds=1,
        repo_root=tmp_path,
        env={},
    )

    captured = capsys.readouterr().out
    assert result == 0
    assert restarted_workers == [
        {
            "id": "gpu0",
            "cuda_visible_devices": "0",
            "start_layer": 4,
            "n_features_per_batch": 1024,
            "n_prompts_in_forward_pass": 512,
            "primary_acts_batch_size": 128,
        }
    ]
    assert "MONITOR_OOM_LIKE_EXIT worker=gpu0 pid=1706986 return_code=-9" in captured
    assert "MONITOR_RESTART worker=gpu0 pid=1707999 oom_count=1" in captured


def test_monitor_reports_initial_worker_nonzero_exit_from_launcher_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher_module = _load_dashboard_launcher_module()
    worker = {
        "id": "gpu0",
        "cuda_visible_devices": "0",
        "start_layer": 4,
        "n_features_per_batch": 1024,
        "n_prompts_in_forward_pass": 512,
        "primary_acts_batch_size": 256,
    }
    pipeline_log_path = tmp_path / "run.gpu0.resume-4-25.log"
    pipeline_log_path.write_text("healthy batch\n", encoding="utf-8")
    launcher_log_path = tmp_path / "launcher.gpu0.20260503_140904.log"
    launcher_log_path.write_text(
        "RuntimeError: Dashboard generation failed for layer 4 with exit code 1\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "launcher.workers.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "worker_id": "gpu0",
                    "pid": 1706986,
                    "pipeline_log": str(pipeline_log_path),
                    "launcher_log": str(launcher_log_path),
                    "worker": worker,
                }
            ]
        ),
        encoding="utf-8",
    )
    worker_config = SimpleNamespace(
        worker_id="gpu0",
        pipeline_log_path=pipeline_log_path,
        run_directory=tmp_path,
        start_layer=4,
        end_layer=25,
        cuda_visible_devices="0",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=512,
        primary_acts_batch_size=256,
    )

    monkeypatch.setattr(launcher_module, "_build_pipeline_config", lambda *args, **kwargs: worker_config)
    monkeypatch.setattr(launcher_module, "_pid_is_running", lambda pid: False)
    monkeypatch.setattr(
        launcher_module,
        "_restart_worker",
        lambda **kwargs: pytest.fail("non-OOM launcher exits should not restart"),
    )

    result = launcher_module._run_monitor_loop(
        config_path=tmp_path / "dashboard.yaml",
        passthrough_args=[],
        manifest_path=manifest_path,
        heartbeat_seconds=1,
        repo_root=tmp_path,
        env={},
    )

    captured = capsys.readouterr().out
    assert result == 0
    assert "MONITOR_WORKER_FAILED_WITHOUT_OOM worker=gpu0 pid=1706986 return_code=1" in captured


def test_run_dashboard_pipeline_skips_layer_with_live_lock(tmp_path: Path, monkeypatch) -> None:
    test_logger = logging.getLogger("test_run_dashboard_pipeline_skips_layer_with_live_lock")
    test_logger.handlers.clear()
    test_logger.propagate = True
    test_logger.setLevel(logging.INFO)

    monkeypatch.setattr(dashboard_pipeline, "_configure_logger", lambda _: test_logger)
    monkeypatch.setattr(dashboard_pipeline, "_build_generation_env", lambda _: {})
    monkeypatch.setattr(dashboard_pipeline, "_ensure_shared_prompt_tokens_file", lambda *args, **kwargs: None)

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError("generation subprocess should not start for a locked layer")

    monkeypatch.setattr(dashboard_pipeline.subprocess, "Popen", _unexpected_popen)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        worker_id="gpu1",
        enable_layer_locks=True,
        import_to_local_db=False,
    )
    lock_path = config.layer_lock_path(0)
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({"pid": dashboard_pipeline.os.getpid(), "worker_id": "gpu0"}), encoding="utf-8")

    results = dashboard_pipeline.run_dashboard_pipeline(config)

    assert len(results) == 1
    assert results[0].skipped
    assert lock_path.exists()


def test_run_dashboard_pipeline_rejects_partial_resume_with_mismatched_feature_batch_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    test_logger = logging.getLogger(
        "test_run_dashboard_pipeline_rejects_partial_resume_with_mismatched_feature_batch_size"
    )
    test_logger.handlers.clear()
    test_logger.propagate = True
    test_logger.setLevel(logging.INFO)

    monkeypatch.setattr(dashboard_pipeline, "_configure_logger", lambda _: test_logger)
    monkeypatch.setattr(dashboard_pipeline, "_build_generation_env", lambda _: {})
    monkeypatch.setattr(dashboard_pipeline, "_ensure_shared_prompt_tokens_file", lambda *args, **kwargs: None)

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError("generation subprocess should not start for an incompatible partial layer resume")

    monkeypatch.setattr(dashboard_pipeline.subprocess, "Popen", _unexpected_popen)

    config = NeuronpediaDashboardPipelineConfig(
        model_name="gemma-3-1b-it",
        model_layers=26,
        sae_set="gemmascope-2-transcoder-262k",
        neuronpedia_source_set_id="gemmascope-2-transcoder-262k-rte",
        neuronpedia_source_set_description="Transcoder - 262k - RTE",
        creator_name="Google DeepMind",
        release_id="gemma-scope-2",
        release_title="Gemma Scope 2",
        release_url="https://huggingface.co/google/gemma-scope-2-1b-it",
        hf_weights_repo_id="google/gemma-scope-2-1b-it",
        hf_weights_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        hook_point="hook_resid_post",
        prompts_huggingface_dataset_path="aps/super_glue",
        start_layer=0,
        end_layer=0,
        sae_path_template="transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        run_root=tmp_path / "runs",
        export_root=tmp_path / "exports",
        saedashboard_repo_root=tmp_path,
        saelens_repo_root=tmp_path,
        neuronpedia_utils_root=tmp_path,
        interpretune_env_file=None,
        import_to_local_db=False,
        archive_partial_dirs=False,
        n_features_per_batch=256,
    )

    output_dir = config.output_dir_for_layer(0)
    output_dir.mkdir(parents=True)
    (output_dir / "batch-0.json").write_text(json.dumps({"features": []}), encoding="utf-8")
    (output_dir / dashboard_pipeline.RUN_SETTINGS_FILE).write_text(
        json.dumps({"n_features_at_a_time": 1024}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="n_features_per_batch=1024"):
        dashboard_pipeline.run_dashboard_pipeline(config)
