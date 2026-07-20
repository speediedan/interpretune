from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import interpretune.utils.neuronpedia_dashboard_pipeline as dashboard_pipeline
from interpretune.utils.neuronpedia_dashboard_pipeline import (
    NeuronpediaDashboardPipelineConfig,
    completed_layers_from_logs,
)


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
        OUTPUT_DIR = ""

        @staticmethod
        def main(ctx, **params):
            captured["ctx"] = ctx
            captured["params"] = params
            export_dir = (
                Path(FakeModule.OUTPUT_DIR)
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

    export_root = dashboard_pipeline.convert_dashboard_output(config, layer_num=0, output_dir=output_dir)

    assert export_root == tmp_path / "exports" / "gemma-3-4b-it" / "0-gemmascope-2-transcoder-16k"
    assert Path(FakeModule.OUTPUT_DIR) == config.export_root
    assert captured["params"] == {
        "saedashboard_output_dir": str(dashboard_leaf_dir),
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
        "prompts_huggingface_dataset_path": "monology/pile-uncopyrighted",
        "n_prompts_total": 24576,
        "n_tokens_in_prompt": 128,
        "zero_out_bos_token": False,
    }
