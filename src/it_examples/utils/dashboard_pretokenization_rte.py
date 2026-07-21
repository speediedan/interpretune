from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # pyright: ignore[reportMissingTypeStubs]
from datasets import Dataset, IterableDataset
from sae_lens.config import PretokenizeRunnerConfig
from transformers import PreTrainedTokenizerBase

from sae_dashboard.neuronpedia.prompt_pretokenization import (
    PretokenizationResult,
    WindowingMode,
    as_1d_token_tensor,
    pretokenize_prompt_token_sequences,
    set_default_windowing_mode,
)
from it_examples.experiments.rte_boolq import TASK_TEXT_FIELD_MAP


DEFAULT_EXPERIMENT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "it_examples/config/experiments/rte_boolq/gemma3/1b_it_lightning_ct_ns_zs_test.yaml"
)


@dataclass(frozen=True)
class RTEPretokenizationSettings:
    prompt_cfg: Any
    task_name: str
    text_fields: tuple[str, str]
    prompt_config_class_path: str
    experiment_config: Path
    windowing_mode: WindowingMode


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    set_default_windowing_mode(parser, default="max-prompt-pad")


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
        windowing_mode=cast(WindowingMode, args.windowing_mode),
    )


def pretokenize_custom_dataset(
    dataset: Dataset | IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
    settings: RTEPretokenizationSettings | None,
) -> PretokenizationResult:
    if settings is None:
        raise ValueError("RTE custom pretokenization requires settings.")
    return pretokenize_prompt_token_sequences(
        _iter_dataset_chat_template_tokens(dataset, tokenizer=tokenizer, settings=settings),
        tokenizer=tokenizer,
        cfg=cfg,
        windowing_mode=settings.windowing_mode,
    )


def build_custom_metadata(
    args: argparse.Namespace,
    settings: RTEPretokenizationSettings | None,
    result: PretokenizationResult,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
) -> dict[str, Any]:
    del args, result, tokenizer, cfg
    if settings is None:
        return {}
    return {
        "prompt_renderer": "apply_chat_template_fn",
        "prompt_config_class": settings.prompt_config_class_path,
        "experiment_config": str(settings.experiment_config),
        "chat_template_add_generation_prompt": True,
        "task_name": settings.task_name,
        "text_fields": list(settings.text_fields),
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
        yield as_1d_token_tensor(tokens)


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
        yield as_1d_token_tensor(tokens)
