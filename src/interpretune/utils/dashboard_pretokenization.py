from __future__ import annotations

import argparse
import importlib
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any, Protocol

import torch
from datasets import Dataset, IterableDataset
from sae_lens.config import PretokenizeRunnerConfig
from sae_lens.pretokenize_runner import PretokenizedDatasetMetadata, metadata_from_config
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class PretokenizationResult:
    tokenized_dataset: Dataset
    effective_context_size: int | None = None
    prompt_lengths: tuple[int, ...] | None = None


class DashboardPretokenizationModule(Protocol):
    def configure_parser(self, parser: argparse.ArgumentParser) -> None: ...

    def load_custom_pretokenization_settings(self, args: argparse.Namespace) -> Any | None: ...

    def pretokenize_custom_dataset(
        self,
        dataset: Dataset | IterableDataset,
        tokenizer: PreTrainedTokenizerBase,
        cfg: PretokenizeRunnerConfig,
        settings: Any | None,
    ) -> PretokenizationResult: ...

    def build_custom_metadata(
        self,
        args: argparse.Namespace,
        settings: Any | None,
        result: PretokenizationResult,
        tokenizer: PreTrainedTokenizerBase,
        cfg: PretokenizeRunnerConfig,
    ) -> dict[str, Any]: ...


def load_custom_pretokenization_module(module_path: str) -> ModuleType:
    return importlib.import_module(module_path)


def maybe_configure_custom_parser(module: ModuleType | None, parser: argparse.ArgumentParser) -> None:
    configure_parser = getattr(module, "configure_parser", None)
    if callable(configure_parser):
        configure_parser(parser)


def maybe_load_custom_pretokenization_settings(module: ModuleType | None, args: argparse.Namespace) -> Any | None:
    load_settings = getattr(module, "load_custom_pretokenization_settings", None)
    if callable(load_settings):
        return load_settings(args)
    return None


def maybe_build_custom_metadata(
    module: ModuleType | None,
    *,
    args: argparse.Namespace,
    settings: Any | None,
    result: PretokenizationResult,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
) -> dict[str, Any]:
    build_metadata = getattr(module, "build_custom_metadata", None)
    if callable(build_metadata):
        return build_metadata(args, settings, result, tokenizer, cfg)
    return {}


def build_dashboard_metadata(
    cfg: PretokenizeRunnerConfig,
    *,
    custom_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = asdict(metadata_from_config(cfg))
    if custom_metadata:
        metadata["custom"] = custom_metadata
    return metadata


def materialize_tokenized_dataset(
    tokenized_dataset: Dataset | IterableDataset,
    *,
    max_tokenized_rows: int | None = None,
) -> Dataset:
    if isinstance(tokenized_dataset, Dataset):
        if max_tokenized_rows is not None and len(tokenized_dataset) > max_tokenized_rows:
            tokenized_dataset = tokenized_dataset.select(range(max_tokenized_rows))
        return ensure_torch_dataset_format(tokenized_dataset)

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(tokenized_dataset):
        if max_tokenized_rows is not None and index >= max_tokenized_rows:
            break
        rows.append({key: _to_python_value(value) for key, value in row.items()})

    if not rows:
        return Dataset.from_dict({"input_ids": []})

    materialized = Dataset.from_list(rows)
    return ensure_torch_dataset_format(materialized)


def ensure_torch_dataset_format(dataset: Dataset) -> Dataset:
    tensor_columns = [
        column_name for column_name in ("input_ids", "tokens", "attention_mask") if column_name in dataset.column_names
    ]
    if tensor_columns:
        dataset.set_format(type="torch", columns=tensor_columns)
    return dataset


def _to_python_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_python_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python_value(inner_value) for inner_value in value]
    return value


def metadata_as_upstream_type(metadata: dict[str, Any]) -> PretokenizedDatasetMetadata:
    upstream_metadata = {key: value for key, value in metadata.items() if key != "custom"}
    return PretokenizedDatasetMetadata(**upstream_metadata)
