from __future__ import annotations

import argparse
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml  # pyright: ignore[reportMissingTypeStubs]
from datasets import Dataset, IterableDataset
from sae_lens.config import PretokenizeRunnerConfig
from sae_lens.pretokenize_runner import get_special_token_from_cfg
from sae_lens.tokenization_and_batching import concat_and_batch_sequences
from transformers import PreTrainedTokenizerBase

from interpretune.utils.dashboard_pretokenization import (
    PretokenizationResult,
    ensure_torch_dataset_format,
    materialize_tokenized_dataset,
)
from it_examples.experiments.rte_boolq import TASK_TEXT_FIELD_MAP


DEFAULT_EXPERIMENT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "it_examples/config/experiments/rte_boolq/gemma3/1b_it_lightning_ct_ns_zs_test.yaml"
)
WINDOWING_MODES = ("max-prompt-pad", "fixed-context-pad", "concatenate", "long-only")


@dataclass(frozen=True)
class RTEPretokenizationSettings:
    prompt_cfg: Any
    task_name: str
    text_fields: tuple[str, str]
    prompt_config_class_path: str
    experiment_config: Path
    windowing_mode: str


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument(
        "--windowing-mode",
        choices=WINDOWING_MODES,
        default="max-prompt-pad",
        help=(
            "Custom RTE prompt windowing policy. max-prompt-pad pads every rendered prompt to the longest rendered "
            "prompt without truncation."
        ),
    )


def load_custom_pretokenization_settings(args: argparse.Namespace) -> RTEPretokenizationSettings:
    payload = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))
    datamodule_args = payload["session_cfg"]["datamodule_cfg"]["init_args"]
    prompt_cfg_payload = datamodule_args["prompt_cfg"]
    prompt_config_class_path = prompt_cfg_payload["class_path"]
    prompt_config_cls = _import_from_class_path(prompt_config_class_path)
    prompt_cfg = prompt_config_cls(**(prompt_cfg_payload.get("init_args") or {}))
    task_name = datamodule_args.get("task_name", "rte")
    text_fields = TASK_TEXT_FIELD_MAP[task_name]
    return RTEPretokenizationSettings(
        prompt_cfg=prompt_cfg,
        task_name=task_name,
        text_fields=(text_fields[0], text_fields[1]),
        prompt_config_class_path=prompt_config_class_path,
        experiment_config=args.experiment_config,
        windowing_mode=args.windowing_mode,
    )


def pretokenize_custom_dataset(
    dataset: Dataset | IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
    settings: RTEPretokenizationSettings | None,
) -> PretokenizationResult:
    if settings is None:
        raise ValueError("RTE custom pretokenization requires settings.")

    if settings.windowing_mode in {"max-prompt-pad", "fixed-context-pad"}:
        prompt_tokens = list(_iter_dataset_chat_template_tokens(dataset, tokenizer=tokenizer, settings=settings))
        prompt_lengths = tuple(int(tokens.numel()) for tokens in prompt_tokens)
        if not prompt_tokens:
            raise ValueError("RTE dataset did not yield any prompts to pretokenize.")
        effective_context_size = (
            max(prompt_lengths) if settings.windowing_mode == "max-prompt-pad" else cfg.context_size
        )
        pad_token_id = _resolve_pad_token_id(tokenizer)
        tokenized_dataset = Dataset.from_dict(
            {
                "input_ids": [
                    _pad_to_context_size(
                        tokens,
                        context_size=effective_context_size,
                        pad_token_id=pad_token_id,
                    ).tolist()
                    for tokens in prompt_tokens
                ],
                "attention_mask": [
                    _attention_mask(length=int(tokens.numel()), context_size=effective_context_size)
                    for tokens in prompt_tokens
                ],
            }
        )
        return PretokenizationResult(
            tokenized_dataset=ensure_torch_dataset_format(tokenized_dataset),
            effective_context_size=effective_context_size,
            prompt_lengths=prompt_lengths,
        )

    disable_concat_sequences = settings.windowing_mode == "long-only"

    def process_examples(examples: dict[str, list[Any]]) -> dict[str, list[torch.Tensor]]:
        return {
            "input_ids": list(
                concat_and_batch_sequences(
                    tokens_iterator=iter(
                        _iter_chat_template_tokens(
                            examples,
                            tokenizer=tokenizer,
                            settings=settings,
                        )
                    ),
                    context_size=cfg.context_size,
                    begin_batch_token_id=get_special_token_from_cfg(cfg.begin_batch_token, tokenizer),
                    begin_sequence_token_id=get_special_token_from_cfg(cfg.begin_sequence_token, tokenizer),
                    sequence_separator_token_id=get_special_token_from_cfg(
                        cfg.sequence_separator_token,
                        tokenizer,
                    ),
                    disable_concat_sequences=disable_concat_sequences,
                )
            )
        }

    tokenized_dataset = _map_dataset(dataset, cfg=cfg, process_examples=process_examples)
    if cfg.shuffle:
        tokenized_dataset = tokenized_dataset.shuffle(seed=cfg.seed)
    materialized_dataset = materialize_tokenized_dataset(tokenized_dataset)
    return PretokenizationResult(
        tokenized_dataset=materialized_dataset,
        effective_context_size=cfg.context_size,
        prompt_lengths=None,
    )


def build_custom_metadata(
    args: argparse.Namespace,
    settings: RTEPretokenizationSettings | None,
    result: PretokenizationResult,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
) -> dict[str, Any]:
    if settings is None:
        return {}
    prompt_lengths = result.prompt_lengths or ()
    return {
        "prompt_renderer": "apply_chat_template_fn",
        "prompt_config_class": settings.prompt_config_class_path,
        "experiment_config": str(settings.experiment_config),
        "chat_template_add_generation_prompt": True,
        "task_name": settings.task_name,
        "text_fields": list(settings.text_fields),
        "rows": len(result.tokenized_dataset),
        "windowing_mode": settings.windowing_mode,
        "effective_context_size": result.effective_context_size,
        "prompt_length_min": min(prompt_lengths) if prompt_lengths else None,
        "prompt_length_max": max(prompt_lengths) if prompt_lengths else None,
        "prompt_length_mean": (sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else None),
        "pad_token_id": tokenizer.pad_token_id,
        "truncation_policy": "error",
        "disable_concat_sequences": (settings.windowing_mode == "long-only" or cfg.disable_concat_sequences),
    }


def build_rte_boolq_task_prompt(
    example: dict[str, Any],
    *,
    prompt_cfg: Any,
    text_fields: tuple[str, str],
) -> str:
    field1 = str(example[text_fields[0]])
    field2 = str(example[text_fields[1]])
    if prompt_cfg.cust_task_prompt:
        return (
            prompt_cfg.cust_task_prompt["context"]
            + "\n"
            + field1
            + "\n"
            + prompt_cfg.cust_task_prompt["question"]
            + "\n"
            + field2
        )
    return field1 + prompt_cfg.ctx_question_join + field2 + prompt_cfg.question_suffix


def _import_from_class_path(class_path: str) -> type[Any]:
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Invalid class path: {class_path}")
    return getattr(importlib.import_module(module_name), class_name)


def _iter_chat_template_tokens(
    examples: dict[str, list[Any]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    settings: RTEPretokenizationSettings,
):
    for field1, field2 in zip(examples[settings.text_fields[0]], examples[settings.text_fields[1]]):
        task_prompt = build_rte_boolq_task_prompt(
            {settings.text_fields[0]: field1, settings.text_fields[1]: field2},
            prompt_cfg=settings.prompt_cfg,
            text_fields=settings.text_fields,
        )
        tokens = settings.prompt_cfg.apply_chat_template_fn(
            tokenizer,
            task_prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        yield _as_1d_token_tensor(tokens)


def _iter_dataset_chat_template_tokens(
    dataset: Dataset | IterableDataset,
    *,
    tokenizer: PreTrainedTokenizerBase,
    settings: RTEPretokenizationSettings,
):
    for row in dataset:
        row_dict = dict(row)
        task_prompt = build_rte_boolq_task_prompt(
            row_dict,
            prompt_cfg=settings.prompt_cfg,
            text_fields=settings.text_fields,
        )
        tokens = settings.prompt_cfg.apply_chat_template_fn(
            tokenizer,
            task_prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        yield _as_1d_token_tensor(tokens)


def _as_1d_token_tensor(tokens: Any) -> torch.Tensor:
    if isinstance(tokens, Mapping) and "input_ids" in tokens:
        tokens = tokens["input_ids"]
    if isinstance(tokens, torch.Tensor):
        return tokens[0] if tokens.ndim == 2 else tokens
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    return token_tensor[0] if token_tensor.ndim == 2 else token_tensor


def _resolve_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for full-prompt dashboard pretokenization.")
    return int(pad_token_id)


def _pad_to_context_size(
    tokens: torch.Tensor,
    *,
    context_size: int,
    pad_token_id: int,
) -> torch.Tensor:
    if tokens.numel() > context_size:
        raise ValueError(
            f"Tokenized prompt length {tokens.numel()} exceeds context_size={context_size}; "
            "full-prompt dashboard pretokenization does not truncate examples."
        )
    if tokens.numel() == context_size:
        return tokens.to(dtype=torch.long).cpu()
    padding = torch.full(
        (context_size - tokens.numel(),),
        pad_token_id,
        dtype=torch.long,
        device=tokens.device,
    )
    return torch.cat([tokens.to(dtype=torch.long), padding]).cpu()


def _attention_mask(*, length: int, context_size: int) -> list[int]:
    return [1] * length + [0] * (context_size - length)


def _map_dataset(
    dataset: Dataset | IterableDataset,
    *,
    cfg: PretokenizeRunnerConfig,
    process_examples: Any,
):
    if cfg.streaming:
        if cfg.num_proc > 1:
            raise ValueError("num_proc must be 1 when streaming is True")
        return dataset.map(
            process_examples,
            batched=True,
            batch_size=cfg.pretokenize_batch_size,
            remove_columns=dataset.column_names,
        )
    return dataset.map(
        process_examples,
        batched=True,
        batch_size=cfg.pretokenize_batch_size,
        num_proc=cfg.num_proc,
        remove_columns=dataset.column_names,
    )
