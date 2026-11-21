"""Prompt analysis: a task-neutral component for analyses of free-form prompts.

The module side needs no entrypoint class: the configurations name the adapter modules interpretune ships (for
example ``CircuitTracerModule``) and compose their configuration sections through ``NeutralSessionMixin``. This file
supplies what the core does not: a prompt configuration that carries the prompts themselves, and a datamodule that
serves them as the test and predict splits. There is no dataset, no label and no task head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import datasets
from torch.utils.data import DataLoader

from interpretune import ChatTemplatePromptConfig, ITDataModule, PromptConfig, sanitize_input_name


@dataclass(kw_only=True)
class PromptSetConfig(PromptConfig):
    """The prompts a session analyzes, spelled as-is (base models).

    ``prompts`` may be empty in a configuration and supplied by the caller before the datamodule is set up, which is
    how the model-only configurations are meant to be used. An experiment variant names its own prompts.
    """

    prompts: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class ChatTemplatePromptSetConfig(ChatTemplatePromptConfig, PromptSetConfig):
    """The prompts a session analyzes, wrapped in the tokenizer's chat template (instruction-tuned models)."""


def _no_prompts() -> ValueError:
    return ValueError(
        "No prompts to analyze: set `datamodule_cfg.prompt_cfg.prompts`, or choose an experiment variant that names "
        "its own."
    )


def _refusing_dataloader(split: str):
    def _loader(self) -> DataLoader:
        raise NotImplementedError(
            f"Prompt analysis has no {split} split: its prompts carry no labels to train or validate against. Use "
            "`test_dataloader()` or `predict_dataloader()`."
        )

    _loader.__name__ = f"{split}_dataloader"
    return _loader


class PromptDataModule(ITDataModule):
    """Serves the configured prompts, tokenized, as the ``test`` split.

    The dataset is built in memory at ``setup`` rather than saved under ``dataset_path``: that path is keyed by task
    name, so a saved copy would outlive a change to the prompts and be served in their place. A model-only
    configuration sets up with no prompts, since a session sets up its datamodule even when the caller passes prompts
    to the model directly; asking that datamodule for a split is what is refused.
    """

    train_dataloader = _refusing_dataloader("train")
    val_dataloader = _refusing_dataloader("validation")

    def prepare_data(self, target_model=None) -> None:
        """Nothing to download or cache."""

    def setup(self, stage: str | None = None, module=None, *args, **kwargs) -> None:
        if module is not None:
            self._module = module
        self.dataset = datasets.DatasetDict({"test": self.encode_prompts(self.prompts)}) if self.prompts else None

    @property
    def prompts(self) -> list[str]:
        prompt_cfg = self.itdm_cfg.prompt_cfg
        if not isinstance(prompt_cfg, PromptSetConfig):
            raise TypeError(
                f"`PromptDataModule` reads its prompts from a `PromptSetConfig`, got {type(prompt_cfg).__name__}. Set "
                "`datamodule_cfg.prompt_cfg.class_path` to `prompt_analysis.PromptSetConfig` or "
                "`prompt_analysis.ChatTemplatePromptSetConfig`."
            )
        return prompt_cfg.prompts

    def encode_prompts(self, prompts: list[str]) -> datasets.Dataset:
        """Tokenize ``prompts`` through the configured prompt template."""
        if not prompts:
            raise _no_prompts()
        prompt_cfg = self.itdm_cfg.prompt_cfg
        if (bind_tokenizer := getattr(prompt_cfg, "bind_tokenizer", None)) is not None:
            bind_tokenizer(self.tokenizer)
        sequences = [
            prompt_cfg.model_chat_template_fn(
                task_prompt=p, tokenization_pattern=self.itdm_cfg.cust_tokenization_pattern
            )
            for p in prompts
        ]
        features = self.tokenizer(sequences, padding="longest", padding_side=self.tokenizer.padding_side)
        features = sanitize_input_name(self.tokenizer.model_input_names, features)
        return datasets.Dataset.from_dict({**features, "sequences": sequences}).select_columns(
            list(self.tokenizer.model_input_names)
        )

    def _eval_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise _no_prompts()
        prompts: Any = self.dataset["test"]  # a datasets.Dataset, which DataLoader indexes like a torch Dataset
        return DataLoader(
            prompts,
            collate_fn=self.data_collator,
            batch_size=self.itdm_cfg.eval_batch_size,
            **{**self.itdm_cfg.dataloader_kwargs, "shuffle": False, "drop_last": False},
        )

    def test_dataloader(self) -> DataLoader:
        return self._eval_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self._eval_dataloader()
