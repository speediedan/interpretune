from typing import Any, Tuple, List
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from interpretune.protocol import StrOrPath
from interpretune.config import ITSerializableCfg, ITSharedConfig
from interpretune.utils import rank_zero_warn, rank_zero_debug


log = logging.getLogger(__name__)

################################################################################
# ITDatamodule Configuration Dataclasses
################################################################################


@dataclass(kw_only=True)
class PromptConfig(ITSerializableCfg):
    cust_task_prompt: dict[str, Any] = field(default_factory=dict)

    def model_chat_template_fn(self, task_prompt: str, tokenization_pattern: str | None = None) -> str:
        return task_prompt.strip()


@dataclass(kw_only=True)
class ChatTemplatePromptConfig(PromptConfig):
    """Chat-template-first prompt construction (hub design §11.5): delegate to the tokenizer's native template.

    The default prompt path: most models need ZERO published prompt artifacts because the tokenizer
    already carries the chat template. Per-model token-spelling dataclasses become the exception
    (template-less base models, research-specific formatting) and are published as ``promptconfigs``
    components. The owning datamodule binds its tokenizer via :meth:`bind_tokenizer`; unbound (or for
    a tokenizer without a chat template) construction falls back to the plain stripped prompt.
    """

    add_generation_prompt: bool = True

    def __post_init__(self) -> None:
        self._tokenizer: Any = None

    def bind_tokenizer(self, tokenizer: Any) -> None:
        """Bind the datamodule's tokenizer so template construction can delegate to it."""
        self._tokenizer = tokenizer

    def build_messages(self, task_prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": task_prompt.strip()}]

    def apply_chat_template_fn(
        self,
        tokenizer: Any,
        task_prompt: str,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        return_tensors: str | None = None,
    ) -> Any:
        """Explicit-tokenizer template application (the pretokenization consumers' seam)."""
        return tokenizer.apply_chat_template(
            self.build_messages(task_prompt),
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
        )

    def model_chat_template_fn(self, task_prompt: str, tokenization_pattern: str | None = None) -> str:
        if self._tokenizer is None or not getattr(self._tokenizer, "chat_template", None):
            return task_prompt.strip()
        return self._tokenizer.apply_chat_template(
            self.build_messages(task_prompt), tokenize=False, add_generation_prompt=self.add_generation_prompt
        )


@dataclass(kw_only=True)
class TokenizationConfig(ITSerializableCfg):
    tokenizers_parallelism: bool = True
    local_fast_tokenizer_path: str | None = None
    cust_tokenization_pattern: str | None = None
    special_tokens_dict: dict[str, Any] = field(default_factory=dict)
    max_seq_length: int = 2048  # TODO: force this to be set rather than allowing a default?


@dataclass(kw_only=True)
class DatasetProcessingConfig(ITSerializableCfg):
    remove_unused_columns: bool = True
    text_fields: Tuple | None = None
    dataset_path: StrOrPath | None = None
    enable_datasets_cache: bool | None = False  # disable caching unless explicitly set to improve reproducibility
    data_collator_cfg: dict[str, Any] = field(default_factory=dict)
    signature_columns: List | None = field(default_factory=list)
    prepare_data_map_cfg: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ITDataModuleConfig(ITSharedConfig, TokenizationConfig, DatasetProcessingConfig):
    # See NOTE [Interpretune Dataclass-Oriented Configuration]
    train_batch_size: int = 32
    eval_batch_size: int = 32
    dataloader_kwargs: dict[str, Any] = field(default_factory=dict)
    # note that for prompt_cfg, we:
    #   1. use (data)classes to minimize special character yaml parsing complications (can override w/ diff init_args)
    #   2. do not provide a default dataclass to avoid current dataclass subclass limitations
    prompt_cfg: PromptConfig = field(default_factory=PromptConfig)

    def __post_init__(self) -> None:
        # TODO: validate prompt_cfg validity
        self.dataloader_kwargs = {
            "num_workers": self.dataloader_kwargs.get("num_workers", 0),
            "pin_memory": self.dataloader_kwargs.get("pin_memory", False),
        }
        if not self.data_collator_cfg:
            self.data_collator_cfg = {"collator_class": "transformers.DataCollatorWithPadding"}

        # Use pathlib for cross-platform path handling and sanitize task name for Windows compatibility
        sanitized_task_name = self.task_name.replace(":", "_").replace("|", "_")
        rank_zero_debug(f"[DATAMODULE_CONFIG] Sanitized task name: '{sanitized_task_name}'")
        hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")

        if hf_datasets_cache:
            cache_home = Path(hf_datasets_cache)
            rank_zero_debug(f"[DATAMODULE_CONFIG] Using HF_DATASETS_CACHE: {cache_home}")
        else:
            # Use Path.home() for cross-platform home directory detection
            cache_home = Path.home() / ".cache" / "huggingface" / "datasets"
            rank_zero_debug(f"[DATAMODULE_CONFIG] Using default cache path: {cache_home}")

        default_dataset_save_path = cache_home / sanitized_task_name
        rank_zero_debug(f"[DATAMODULE_CONFIG] Default dataset path: {default_dataset_save_path}")

        # Ensure proper platform-specific path separators
        if self.dataset_path is None:
            self.dataset_path = default_dataset_save_path.resolve()
        else:
            # Convert existing path to use proper separators
            self.dataset_path = Path(self.dataset_path).resolve()
        rank_zero_debug(f"[DATAMODULE_CONFIG] Final dataset_path: {self.dataset_path}")

    def _cross_validate(self, it_cfg: ITSerializableCfg) -> None:
        # inspect tokenizer, tokenizer_name, model_name_or_path here, updating datamodule config before instantiation
        # if a value is missing in the datamodule config but present in the module config
        # we first inspect to see if we have a fallback `model_name_or_path`
        for dm_fallback_attr in ["tokenizer", "tokenizer_name", "model_name_or_path"]:
            if getattr(self, dm_fallback_attr) is None and getattr(it_cfg, dm_fallback_attr, None) is not None:
                rank_zero_warn(
                    f"Since no datamodule `{dm_fallback_attr}` was provided, attempting to use fallback"
                    f" configuration, setting `{dm_fallback_attr}` to {str(getattr(it_cfg, dm_fallback_attr))[:20]}."
                )
                setattr(self, dm_fallback_attr, getattr(it_cfg, dm_fallback_attr))
