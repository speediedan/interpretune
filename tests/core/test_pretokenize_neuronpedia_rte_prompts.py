from __future__ import annotations

import importlib
from typing import Any

import torch
import pytest
from datasets import Dataset
from sae_lens.config import PretokenizeRunnerConfig
from sae_dashboard.neuronpedia import prompt_pretokenization as pretokenize_script
from transformers.tokenization_utils_base import BatchEncoding


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
    assert settings.windowing_mode == "max-prompt-pad"


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
    assert result.windowing_mode == "max-prompt-pad"
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
    assert result.windowing_mode == "max-prompt-pad"
    assert len(result.tokenized_dataset) == 2
    assert result.tokenized_dataset[0]["input_ids"].tolist() == [1, 10, 11, 0, 0]
    assert result.tokenized_dataset[0]["attention_mask"].tolist() == [1, 1, 1, 0, 0]
    assert result.tokenized_dataset[1]["input_ids"].tolist() == [1, 10, 11, 12, 13]


def test_pretokenize_rte_dataset_fixed_context_pad_uses_requested_context() -> None:
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
            "--windowing-mode",
            "fixed-context-pad",
        ]
    )
    settings = RTE_MODULE.load_custom_pretokenization_settings(args)
    tokenizer = RecordingTokenizer(
        responses=[
            torch.tensor([[1, 10, 11]], dtype=torch.long),
            torch.tensor([[1, 10]], dtype=torch.long),
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
        context_size=6,
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

    assert result.effective_context_size == 6
    assert result.prompt_lengths == (3, 2)
    assert result.windowing_mode == "fixed-context-pad"
    assert result.tokenized_dataset[0]["attention_mask"].tolist() == [1, 1, 1, 0, 0, 0]
    assert result.tokenized_dataset[1]["input_ids"].tolist() == [1, 10, 0, 0, 0, 0]


def test_pretokenize_rte_dataset_filter_truncate_uses_upstream_packed_windowing() -> None:
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
            "--windowing-mode",
            "filter-truncate",
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
        context_size=4,
        column_name="text",
        shuffle=False,
        streaming=False,
        pretokenize_batch_size=1,
        begin_batch_token=None,
        begin_sequence_token=None,
        sequence_separator_token=None,
        disable_concat_sequences=True,
    )

    result = RTE_MODULE.pretokenize_custom_dataset(dataset, tokenizer, cfg, settings)

    assert result.effective_context_size == 4
    assert result.prompt_lengths is None
    assert result.windowing_mode == "filter-truncate"
    assert len(result.tokenized_dataset) == 1
    assert result.tokenized_dataset[0]["input_ids"].tolist() == [1, 10, 11, 12]


def test_build_custom_metadata_keeps_only_rte_specific_fields() -> None:
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

    metadata = RTE_MODULE.build_custom_metadata(
        args,
        settings,
        pretokenize_script.PretokenizationResult(tokenized_dataset=Dataset.from_dict({"input_ids": [[1, 2, 3]]})),
        RecordingTokenizer(),
        PretokenizeRunnerConfig(
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
            begin_batch_token=None,
            begin_sequence_token=None,
            sequence_separator_token=None,
            disable_concat_sequences=True,
        ),
    )

    assert metadata == {
        "prompt_renderer": "apply_chat_template_fn",
        "prompt_config_class": settings.prompt_config_class_path,
        "experiment_config": str(settings.experiment_config),
        "chat_template_add_generation_prompt": True,
        "task_name": "rte",
        "text_fields": ["premise", "hypothesis"],
    }


def test_pad_to_context_size_refuses_truncation() -> None:
    with pytest.raises(ValueError, match="does not truncate"):
        pretokenize_script.pad_to_context_size(torch.tensor([1, 2, 3]), context_size=2, pad_token_id=0)


def test_as_1d_token_tensor_accepts_batch_encoding() -> None:
    tokens = pretokenize_script.as_1d_token_tensor(
        BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)})
    )

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
