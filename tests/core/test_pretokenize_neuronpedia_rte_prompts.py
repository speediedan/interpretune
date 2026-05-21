from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
import pytest
from datasets import Dataset, load_dataset
from sae_lens.config import PretokenizeRunnerConfig
from transformers.tokenization_utils_base import BatchEncoding


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "pretokenize_dashboard_dataset.py"
SPEC = importlib.util.spec_from_file_location("pretokenize_dashboard_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
pretokenize_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pretokenize_script
SPEC.loader.exec_module(pretokenize_script)

RTE_MODULE = importlib.import_module("it_examples.utils.dashboard_pretokenization_rte")


class RecordingTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __init__(self, responses: list[torch.Tensor] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = responses or []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "return_tensors": return_tensors,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return torch.tensor([[1, 10, 11, 12]], dtype=torch.long)


def test_load_rte_pretokenization_settings_uses_gemma_prompt_config() -> None:
    args = pretokenize_script.parse_args(
        [
            "--dataset-path",
            "aps/super_glue",
            "--dataset-name",
            "rte",
            "--tokenizer-name",
            "google/gemma-3-1b-it",
            "--output-dir",
            "/tmp/rte_tokens",
            "--custom-dataset-module",
            "it_examples.utils.dashboard_pretokenization_rte",
        ]
    )
    settings = RTE_MODULE.load_custom_pretokenization_settings(args)

    assert settings.task_name == "rte"
    assert settings.text_fields == ("premise", "hypothesis")
    assert settings.prompt_config_class_path == "it_examples.example_prompt_configs.RTEBoolqGemmaPromptConfig"
    assert settings.prompt_cfg.ctx_question_join == "Does the previous passage imply that "


def test_build_rte_boolq_task_prompt_matches_datamodule_template() -> None:
    args = pretokenize_script.parse_args(
        [
            "--dataset-path",
            "aps/super_glue",
            "--dataset-name",
            "rte",
            "--tokenizer-name",
            "google/gemma-3-1b-it",
            "--output-dir",
            "/tmp/rte_tokens",
            "--custom-dataset-module",
            "it_examples.utils.dashboard_pretokenization_rte",
        ]
    )
    settings = RTE_MODULE.load_custom_pretokenization_settings(args)

    prompt = RTE_MODULE.build_rte_boolq_task_prompt(
        {"premise": "The sky is blue. ", "hypothesis": "the sky is blue"},
        prompt_cfg=settings.prompt_cfg,
        text_fields=settings.text_fields,
    )

    assert prompt == (
        "The sky is blue. Does the previous passage imply that "
        "the sky is blue? Answer with only one word, either Yes or No."
    )


def test_pretokenize_rte_dataset_uses_apply_chat_template_with_generation_prompt() -> None:
    args = pretokenize_script.parse_args(
        [
            "--dataset-path",
            "aps/super_glue",
            "--dataset-name",
            "rte",
            "--tokenizer-name",
            "google/gemma-3-1b-it",
            "--output-dir",
            "/tmp/rte_tokens",
            "--custom-dataset-module",
            "it_examples.utils.dashboard_pretokenization_rte",
        ]
    )
    settings = RTE_MODULE.load_custom_pretokenization_settings(args)
    tokenizer = RecordingTokenizer()
    dataset = Dataset.from_dict({"premise": ["Premise."], "hypothesis": ["Hypothesis"], "label": [0]})
    cfg = PretokenizeRunnerConfig(
        tokenizer_name="fake-tokenizer",
        dataset_path="aps/super_glue",
        dataset_name="rte",
        split="train",
        num_proc=1,
        context_size=4,
        column_name="text",
        shuffle=False,
        streaming=False,
        pretokenize_batch_size=1,
        begin_batch_token="bos",
        begin_sequence_token=None,
        sequence_separator_token="bos",
        disable_concat_sequences=True,
    )

    result = RTE_MODULE.pretokenize_custom_dataset(dataset, tokenizer, cfg, settings)
    tokenized = result.tokenized_dataset

    assert len(tokenized) == 1
    assert tokenized[0]["input_ids"].tolist() == [1, 10, 11, 12]
    assert tokenizer.calls == [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Premise.Does the previous passage imply that Hypothesis? "
                    "Answer with only one word, either Yes or No.",
                }
            ],
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
    ]


def test_pretokenize_rte_dataset_max_prompt_pad_keeps_every_prompt() -> None:
    args = pretokenize_script.parse_args(
        [
            "--dataset-path",
            "aps/super_glue",
            "--dataset-name",
            "rte",
            "--tokenizer-name",
            "google/gemma-3-1b-it",
            "--output-dir",
            "/tmp/rte_tokens",
            "--custom-dataset-module",
            "it_examples.utils.dashboard_pretokenization_rte",
        ]
    )
    settings = RTE_MODULE.load_custom_pretokenization_settings(args)
    tokenizer = RecordingTokenizer(
        responses=[
            torch.tensor([[1, 10, 11]], dtype=torch.long),
            torch.tensor([[1, 10, 11, 12, 13]], dtype=torch.long),
        ]
    )
    dataset = Dataset.from_dict(
        {
            "premise": ["Premise A.", "Premise B."],
            "hypothesis": ["Hypothesis A", "Hypothesis B"],
            "label": [0, 1],
        }
    )
    cfg = PretokenizeRunnerConfig(
        tokenizer_name="fake-tokenizer",
        dataset_path="aps/super_glue",
        dataset_name="rte",
        split="train",
        num_proc=1,
        context_size=16,
        column_name="text",
        shuffle=False,
        streaming=False,
        pretokenize_batch_size=1,
        begin_batch_token="bos",
        begin_sequence_token=None,
        sequence_separator_token="bos",
        disable_concat_sequences=True,
    )

    settings = settings.__class__(**{**settings.__dict__, "windowing_mode": "max-prompt-pad"})

    result = RTE_MODULE.pretokenize_custom_dataset(dataset, tokenizer, cfg, settings)

    assert result.effective_context_size == 5
    assert result.prompt_lengths == (3, 5)
    assert len(result.tokenized_dataset) == 2
    assert result.tokenized_dataset[0]["input_ids"].tolist() == [1, 10, 11, 0, 0]
    assert result.tokenized_dataset[0]["attention_mask"].tolist() == [1, 1, 1, 0, 0]
    assert result.tokenized_dataset[1]["input_ids"].tolist() == [1, 10, 11, 12, 13]


def test_pad_to_context_size_refuses_truncation() -> None:
    with pytest.raises(ValueError, match="does not truncate"):
        RTE_MODULE._pad_to_context_size(torch.tensor([1, 2, 3]), context_size=2, pad_token_id=0)


def test_as_1d_token_tensor_accepts_batch_encoding() -> None:
    tokens = RTE_MODULE._as_1d_token_tensor(BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}))

    assert tokens.tolist() == [1, 2, 3]


def test_default_harness_materializes_streaming_tokenized_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset.from_dict({"text": ["alpha", "beta", "gamma"]}).to_iterable_dataset()

    class DummyTokenizer:
        model_max_length = 0

    args = pretokenize_script.parse_args(
        [
            "--dataset-path",
            "monology/pile-uncopyrighted",
            "--tokenizer-name",
            "google/gemma-3-1b-it",
            "--output-dir",
            "/tmp/monology_tokens",
            "--streaming",
            "--use-chat-formatting",
            "--max-tokenized-rows",
            "2",
        ]
    )

    monkeypatch.setattr(pretokenize_script, "load_dashboard_dataset", lambda cfg, max_rows: dataset)
    monkeypatch.setattr(pretokenize_script.AutoTokenizer, "from_pretrained", lambda _: DummyTokenizer())

    def fake_pretokenize_dataset(dataset, tokenizer, cfg):
        return Dataset.from_dict({"input_ids": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]}).to_iterable_dataset()

    monkeypatch.setattr(pretokenize_script, "pretokenize_dataset", fake_pretokenize_dataset)

    result, cfg, metadata = pretokenize_script.run_dashboard_pretokenization(args)

    assert isinstance(result.tokenized_dataset, Dataset)
    assert len(result.tokenized_dataset) == 2
    assert result.tokenized_dataset[0]["input_ids"].tolist() == [1, 2, 3, 4]
    assert cfg.use_chat_formatting is True
    assert "custom" not in metadata


def test_persist_legacy_load_dataset_directory_writes_jsonl_export(tmp_path: Path) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [4, 5, 0]],
            "attention_mask": [[1, 1, 1], [1, 1, 0]],
        }
    ).with_format("torch")
    output_dir = tmp_path / "legacy_tokens"
    metadata = {"tokenizer_name": "google/gemma-3-1b-it", "pad_token_id": 0}

    pretokenize_script.persist_legacy_load_dataset_directory(
        dataset,
        output_dir=output_dir,
        split="train",
        metadata=metadata,
        force=False,
    )

    loaded = load_dataset(str(output_dir), split="train")

    assert loaded[0]["input_ids"] == [1, 2, 3]
    assert loaded[1]["attention_mask"] == [1, 1, 0]
    assert json.loads((output_dir / "sae_lens.json").read_text(encoding="utf-8")) == metadata
