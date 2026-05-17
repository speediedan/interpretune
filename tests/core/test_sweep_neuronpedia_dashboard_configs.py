from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest


def _load_sweep_module() -> ModuleType:
    script_path = Path(__file__).parents[2] / "scripts" / "sweep_neuronpedia_dashboard_configs.py"
    spec = importlib.util.spec_from_file_location("test_dashboard_sweep", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_bucket_seed_config_scales_prompt_and_primary_acts_for_shorter_context() -> None:
    sweep_module = _load_sweep_module()

    config = sweep_module.build_bucket_seed_config(
        label_prefix="gpu0-pilot",
        n_features_per_batch=1024,
        base_tokens_in_prompt=319,
        base_n_prompts_in_forward_pass=256,
        base_primary_acts_batch_size=64,
        bucket_ceiling=128,
        bucket_prompt_count=640,
        prompt_scale_limit=4.0,
        primary_acts_scale_limit=4.0,
        round_to=8,
    )

    assert config.prompt_bucket_ceiling == 128
    assert config.n_features_per_batch == 1024
    assert config.n_prompts_in_forward_pass == 608
    assert config.primary_acts_batch_size == 152
    assert config.label == "gpu0-pilot-bucket128-1024x608-acts152"


def test_build_same_layer_fixed_baseline_config_uses_base_shape() -> None:
    sweep_module = _load_sweep_module()

    config = sweep_module.build_same_layer_fixed_baseline_config(
        Namespace(
            base_n_features_per_batch=1024,
            base_n_prompts_in_forward_pass=256,
            base_primary_acts_batch_size=64,
        )
    )

    assert config.label == "same-layer-fixed-1024x256-acts64"
    assert config.n_features_per_batch == 1024
    assert config.n_prompts_in_forward_pass == 256
    assert config.primary_acts_batch_size == 64
    assert config.prompt_bucket_ceiling is None


def test_prefer_exact_primary_acts_partition_snaps_to_nearby_divisible_shapes() -> None:
    sweep_module = _load_sweep_module()

    assert sweep_module.prefer_exact_primary_acts_partition(
        n_prompts_in_forward_pass=536,
        primary_acts_batch_size=256,
        max_prompt_count=543,
    ) == (512, 256)
    assert sweep_module.prefer_exact_primary_acts_partition(
        n_prompts_in_forward_pass=632,
        primary_acts_batch_size=152,
        max_prompt_count=1450,
    ) == (608, 152)
    assert sweep_module.prefer_exact_primary_acts_partition(
        n_prompts_in_forward_pass=400,
        primary_acts_batch_size=104,
        max_prompt_count=406,
    ) == (400, 100)
    assert sweep_module.prefer_exact_primary_acts_partition(
        n_prompts_in_forward_pass=256,
        primary_acts_batch_size=176,
        max_prompt_count=256,
    ) == (256, 176)


def test_candidate_gain_is_worthwhile_rejects_large_vram_growth_for_tiny_gain() -> None:
    sweep_module = _load_sweep_module()

    baseline = sweep_module.SweepResult(
        label="baseline",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=256,
        status="target_reached",
        reason="completed_3_batches",
        exit_code=0,
        started_at_utc="2026-05-04T17:00:00+00:00",
        ended_at_utc="2026-05-04T17:03:00+00:00",
        elapsed_seconds=180.0,
        throughput_features_per_min=800.0,
        max_gpu_process_used_mib=6000,
    )
    candidate = sweep_module.SweepResult(
        label="candidate",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=512,
        status="target_reached",
        reason="completed_3_batches",
        exit_code=0,
        started_at_utc="2026-05-04T17:04:00+00:00",
        ended_at_utc="2026-05-04T17:07:00+00:00",
        elapsed_seconds=180.0,
        throughput_features_per_min=808.0,
        max_gpu_process_used_mib=12000,
    )

    assert not sweep_module.candidate_gain_is_worthwhile(
        baseline,
        candidate,
        min_throughput_gain_fraction=0.03,
        max_vram_growth_per_gain=8.0,
    )


def test_build_command_includes_bucket_and_primary_acts_args(tmp_path: Path) -> None:
    sweep_module = _load_sweep_module()
    config = sweep_module.SweepConfig(
        label="bucket128",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=632,
        primary_acts_batch_size=152,
        prompt_bucket_ceiling=128,
    )

    command = sweep_module.build_command(
        config,
        layer=9,
        python_executable="/mnt/cache/speediedan/.venvs/it_latest/bin/python",
        run_root=tmp_path / "runs",
        pretokenized_dataset_path=tmp_path / "pretokenized",
    )

    assert "--prompt-bucket-ceiling" in command
    assert "128" in command
    assert "--runner-log-performance" in command
    assert "--no-runner-use-cached-activations" in command
    assert "--primary-acts-batch-size" in command
    assert "152" in command
    assert "--n-prompts-in-forward-pass" in command
    assert "632" in command


def test_build_command_can_enable_runner_activation_cache(tmp_path: Path) -> None:
    sweep_module = _load_sweep_module()
    config = sweep_module.SweepConfig(
        label="bucket128-cache-on",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=632,
        primary_acts_batch_size=152,
        prompt_bucket_ceiling=128,
    )

    command = sweep_module.build_command(
        config,
        layer=9,
        python_executable="/mnt/cache/speediedan/.venvs/it_latest/bin/python",
        run_root=tmp_path / "runs",
        pretokenized_dataset_path=tmp_path / "pretokenized",
        runner_use_cached_activations=True,
    )

    assert "--runner-use-cached-activations" in command
    assert "--no-runner-use-cached-activations" not in command


def test_resolve_pretokenized_dataset_path_infers_parent_from_bucket_manifest(tmp_path: Path) -> None:
    sweep_module = _load_sweep_module()
    manifest_path = tmp_path / "tokens_2490.buckets.json"

    inferred = sweep_module.resolve_pretokenized_dataset_path(
        bucket_manifest_path=manifest_path,
        pretokenized_dataset_path=None,
    )

    assert inferred == tmp_path


def test_resolve_pretokenized_dataset_path_prefers_explicit_value(tmp_path: Path) -> None:
    sweep_module = _load_sweep_module()
    manifest_path = tmp_path / "tokens_2490.buckets.json"
    explicit_path = tmp_path / "explicit"

    resolved = sweep_module.resolve_pretokenized_dataset_path(
        bucket_manifest_path=manifest_path,
        pretokenized_dataset_path=explicit_path,
    )

    assert resolved == explicit_path


def test_build_more_conservative_bucket_candidate_halves_acts_before_prompts() -> None:
    sweep_module = _load_sweep_module()
    config = sweep_module.SweepConfig(
        label="bucket128",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=632,
        primary_acts_batch_size=152,
        prompt_bucket_ceiling=128,
    )

    first = sweep_module.build_more_conservative_bucket_candidate(config, round_to=8)
    assert first is not None
    assert first.primary_acts_batch_size == 72
    assert first.n_prompts_in_forward_pass == 632

    second = sweep_module.build_more_conservative_bucket_candidate(
        sweep_module.SweepConfig(
            label="bucket128-smallacts",
            n_features_per_batch=1024,
            n_prompts_in_forward_pass=632,
            primary_acts_batch_size=8,
            prompt_bucket_ceiling=128,
        ),
        round_to=8,
    )
    assert second is not None
    assert second.n_prompts_in_forward_pass == 312
    assert second.primary_acts_batch_size == 8


def test_build_more_aggressive_bucket_candidate_caps_to_bucket_prompt_count() -> None:
    sweep_module = _load_sweep_module()
    config = sweep_module.SweepConfig(
        label="bucket64",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=248,
        primary_acts_batch_size=120,
        prompt_bucket_ceiling=64,
    )

    candidate = sweep_module.build_more_aggressive_bucket_candidate(
        config,
        bucket_prompt_count=256,
        growth_factor=1.5,
        round_to=8,
    )

    assert candidate is not None
    assert candidate.n_prompts_in_forward_pass == 256
    assert candidate.primary_acts_batch_size == 176
    assert candidate.prompt_bucket_ceiling == 64


def test_run_bucket_pilot_keeps_under_budget_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sweep_module = _load_sweep_module()
    seed_config = sweep_module.SweepConfig(
        label="bucket128-seed",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=608,
        primary_acts_batch_size=152,
        prompt_bucket_ceiling=128,
    )
    aggressive_config = sweep_module.SweepConfig(
        label="bucket128-aggressive",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=896,
        primary_acts_batch_size=224,
        prompt_bucket_ceiling=128,
    )
    probe_results = iter(
        [
            sweep_module.SweepResult(
                label=seed_config.label,
                n_features_per_batch=seed_config.n_features_per_batch,
                n_prompts_in_forward_pass=seed_config.n_prompts_in_forward_pass,
                status="target_reached",
                reason="completed_2_batches",
                exit_code=0,
                started_at_utc="2026-05-05T03:00:00+00:00",
                ended_at_utc="2026-05-05T03:02:00+00:00",
                elapsed_seconds=120.0,
                throughput_features_per_min=900.0,
                max_gpu_process_used_mib=9400,
            ),
            sweep_module.SweepResult(
                label=aggressive_config.label,
                n_features_per_batch=aggressive_config.n_features_per_batch,
                n_prompts_in_forward_pass=aggressive_config.n_prompts_in_forward_pass,
                status="target_reached",
                reason="completed_2_batches",
                exit_code=0,
                started_at_utc="2026-05-05T03:03:00+00:00",
                ended_at_utc="2026-05-05T03:05:00+00:00",
                elapsed_seconds=120.0,
                throughput_features_per_min=1100.0,
                max_gpu_process_used_mib=11700,
            ),
        ]
    )

    monkeypatch.setattr(
        sweep_module,
        "load_bucket_manifest",
        lambda _path: {"buckets": [{"upper_inclusive": 128, "prompt_count": 1450}]},
    )
    monkeypatch.setattr(sweep_module, "build_bucket_seed_config", lambda **_kwargs: seed_config)
    monkeypatch.setattr(
        sweep_module,
        "build_more_aggressive_bucket_candidate",
        lambda config, **_kwargs: aggressive_config if config == seed_config else None,
    )
    monkeypatch.setattr(
        sweep_module,
        "run_probe",
        lambda _config, *, args, session_dir, env: next(probe_results),
    )
    monkeypatch.setattr(sweep_module, "write_results", lambda *_args, **_kwargs: None)

    args = Namespace(
        bucket_manifest_path=tmp_path / "manifest.json",
        bucket_ceiling=[],
        layer=9,
        base_n_features_per_batch=1024,
        base_tokens_in_prompt=319,
        base_n_prompts_in_forward_pass=256,
        base_primary_acts_batch_size=64,
        prompt_scale_limit=4.0,
        primary_acts_scale_limit=4.0,
        batch_size_round_to=8,
        max_probe_attempts_per_bucket=4,
        probe_growth_factor=1.5,
        min_throughput_gain_fraction=0.03,
        max_vram_growth_per_gain=8.0,
        target_max_gpu_process_used_mib=10000,
        runner_use_cached_activations=False,
        same_layer_fixed_baseline=False,
    )

    sweep_module.run_bucket_pilot(args, session_dir=tmp_path, env={"CUDA_VISIBLE_DEVICES": "0"})

    selected_payload = json.loads((tmp_path / "selected_bucket_configs.json").read_text(encoding="utf-8"))
    assert selected_payload["target_max_gpu_process_used_mib"] == 10000
    assert selected_payload["same_layer_fixed_baseline"] is None
    selected_bucket = selected_payload["buckets"][0]
    assert selected_bucket["selected_config"]["label"] == seed_config.label
    assert selected_bucket["selected_result"]["label"] == seed_config.label


def test_run_bucket_pilot_can_persist_same_layer_fixed_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep_module = _load_sweep_module()
    fixed_config = sweep_module.SweepConfig(
        label="same-layer-fixed-1024x256-acts64",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=256,
        primary_acts_batch_size=64,
    )
    seed_config = sweep_module.SweepConfig(
        label="bucket128-seed",
        n_features_per_batch=1024,
        n_prompts_in_forward_pass=608,
        primary_acts_batch_size=152,
        prompt_bucket_ceiling=128,
    )
    probe_results = iter(
        [
            sweep_module.SweepResult(
                label=fixed_config.label,
                n_features_per_batch=fixed_config.n_features_per_batch,
                n_prompts_in_forward_pass=fixed_config.n_prompts_in_forward_pass,
                status="target_reached",
                reason="completed_2_batches",
                exit_code=0,
                started_at_utc="2026-05-06T00:00:00+00:00",
                ended_at_utc="2026-05-06T00:02:00+00:00",
                elapsed_seconds=120.0,
                throughput_features_per_min=600.0,
                max_gpu_process_used_mib=9800,
            ),
            sweep_module.SweepResult(
                label=seed_config.label,
                n_features_per_batch=seed_config.n_features_per_batch,
                n_prompts_in_forward_pass=seed_config.n_prompts_in_forward_pass,
                status="target_reached",
                reason="completed_2_batches",
                exit_code=0,
                started_at_utc="2026-05-06T00:03:00+00:00",
                ended_at_utc="2026-05-06T00:05:00+00:00",
                elapsed_seconds=120.0,
                throughput_features_per_min=450.0,
                max_gpu_process_used_mib=9400,
            ),
        ]
    )

    monkeypatch.setattr(
        sweep_module,
        "load_bucket_manifest",
        lambda _path: {"buckets": [{"upper_inclusive": 128, "prompt_count": 1450}]},
    )
    monkeypatch.setattr(sweep_module, "build_same_layer_fixed_baseline_config", lambda _args: fixed_config)
    monkeypatch.setattr(sweep_module, "build_bucket_seed_config", lambda **_kwargs: seed_config)
    monkeypatch.setattr(sweep_module, "build_more_aggressive_bucket_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sweep_module, "run_probe", lambda _config, *, args, session_dir, env: next(probe_results))
    monkeypatch.setattr(sweep_module, "write_results", lambda *_args, **_kwargs: None)

    args = Namespace(
        bucket_manifest_path=tmp_path / "manifest.json",
        bucket_ceiling=[],
        layer=9,
        base_n_features_per_batch=1024,
        base_tokens_in_prompt=319,
        base_n_prompts_in_forward_pass=256,
        base_primary_acts_batch_size=64,
        prompt_scale_limit=4.0,
        primary_acts_scale_limit=4.0,
        batch_size_round_to=8,
        max_probe_attempts_per_bucket=4,
        probe_growth_factor=1.5,
        min_throughput_gain_fraction=0.03,
        max_vram_growth_per_gain=8.0,
        target_max_gpu_process_used_mib=10000,
        runner_use_cached_activations=False,
        same_layer_fixed_baseline=True,
    )

    sweep_module.run_bucket_pilot(args, session_dir=tmp_path, env={"CUDA_VISIBLE_DEVICES": "0"})

    selected_payload = json.loads((tmp_path / "selected_bucket_configs.json").read_text(encoding="utf-8"))
    assert selected_payload["same_layer_fixed_baseline"]["config"]["label"] == fixed_config.label
    assert selected_payload["same_layer_fixed_baseline"]["result"]["label"] == fixed_config.label
    assert selected_payload["buckets"][0]["selected_result"]["label"] == seed_config.label


def test_select_bucket_entries_filters_requested_ceilings() -> None:
    sweep_module = _load_sweep_module()
    manifest = {
        "buckets": [
            {"upper_inclusive": 64, "prompt_count": 200},
            {"upper_inclusive": 128, "prompt_count": 640},
            {"upper_inclusive": 319, "prompt_count": 2490},
        ]
    }

    selected = sweep_module.select_bucket_entries(manifest, [64, 319])

    assert [bucket["upper_inclusive"] for bucket in selected] == [64, 319]


def test_parse_prefixed_kv_line_decodes_runner_perf_scalars_and_json() -> None:
    sweep_module = _load_sweep_module()

    fields = sweep_module.parse_prefixed_kv_line(
        '[runner_perf] event=get_feature_data_summary wall_s=1.250000 prompt_count=543 cpu={"load1":1.5}',
        sweep_module.RUNNER_PERF_PREFIX,
    )

    assert fields == {
        "event": "get_feature_data_summary",
        "wall_s": pytest.approx(1.25),
        "prompt_count": 543,
        "cpu": {"load1": pytest.approx(1.5)},
    }


def test_summarize_feature_data_metrics_aggregates_completed_batches() -> None:
    sweep_module = _load_sweep_module()

    metrics = sweep_module.summarize_feature_data_metrics(
        [
            {
                "model_forward_passes": 4,
                "total_forward_wall_s": 2.0,
                "get_feature_data_wall_s": 8.5,
                "peak_rss_gib": 5.5,
                "peak_cuda_allocated_gib": 3.0,
                "peak_cuda_reserved_gib": 4.0,
            },
            {
                "model_forward_passes": 6,
                "total_forward_wall_s": 3.0,
                "get_feature_data_wall_s": 10.0,
                "peak_rss_gib": 5.8,
                "peak_cuda_allocated_gib": 3.2,
                "peak_cuda_reserved_gib": 4.1,
            },
            {
                "model_forward_passes": 99,
                "total_forward_wall_s": 99.0,
                "get_feature_data_wall_s": 99.0,
                "peak_rss_gib": 99.0,
                "peak_cuda_allocated_gib": 99.0,
                "peak_cuda_reserved_gib": 99.0,
            },
        ],
        [8.0, 10.0, 12.0],
        [16.0, 20.0],
        completed_batch_count=2,
    )

    assert metrics.total_model_forward_passes == 10
    assert metrics.total_model_forward_wall_s == pytest.approx(5.0)
    assert metrics.avg_model_forward_wall_s == pytest.approx(0.5)
    assert metrics.total_get_feature_data_wall_s == pytest.approx(18.0)
    assert metrics.avg_get_feature_data_wall_s == pytest.approx(9.0)
    assert metrics.avg_get_feature_data_share_of_batch == pytest.approx(0.5)
    assert metrics.peak_get_feature_data_rss_gib == pytest.approx(5.8)
    assert metrics.peak_get_feature_data_cuda_allocated_gib == pytest.approx(3.2)
    assert metrics.peak_get_feature_data_cuda_reserved_gib == pytest.approx(4.1)
