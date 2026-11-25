from typing import Any, Mapping, Tuple, List
import hashlib
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
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
    """Base prompt construction: the task prompt is used as-is.

    Subclasses override :meth:`model_chat_template_fn` to wrap it for models that expect a chat format.
    """

    cust_task_prompt: dict[str, Any] = field(default_factory=dict)

    def model_chat_template_fn(self, task_prompt: str, tokenization_pattern: str | None = None) -> str:
        """Return the prompt unchanged (stripped).

        The identity case for non-chat models.
        """
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
    #: Extra keyword arguments for the tokenizer's ``apply_chat_template``, which a template reads as variables (for
    #: example ``{"enable_thinking": False}`` for Qwen3, whose template otherwise lets the model open a reasoning block
    #: at the answer position).
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._tokenizer: Any = None

    def bind_tokenizer(self, tokenizer: Any) -> None:
        """Bind the datamodule's tokenizer so template construction can delegate to it."""
        self._tokenizer = tokenizer

    def __getstate__(self) -> dict[str, Any]:
        """Leave the bound tokenizer out of a pickled config.

        The binding is runtime state, not configuration. A Lightning datamodule's hyperparameters hold its config, so
        a checkpoint would otherwise pickle the tokenizer, and Lightning, which loads checkpoints with
        ``weights_only=True``, refuses it (restoring a best checkpoint at a phase transition included). The owning
        datamodule binds its tokenizer again when it prepares data.
        """
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        return state

    def __deepcopy__(self, memo: dict[int, Any]) -> "ChatTemplatePromptConfig":
        """Copy the configuration, sharing the bound tokenizer rather than copying it."""
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            setattr(clone, name, value if name == "_tokenizer" else deepcopy(value, memo))
        return clone

    def build_messages(self, task_prompt: str) -> list[dict[str, str]]:
        """Wrap the task prompt as a single user message.

        Override for multi-turn or system prompts.
        """
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
            **self.chat_template_kwargs,
        )

    def model_chat_template_fn(self, task_prompt: str, tokenization_pattern: str | None = None) -> str:
        """Apply the tokenizer's native chat template, falling back to the bare prompt if it has none.

        The fallback matters: a model without a chat template must still produce a usable prompt rather
        than an error, so the same config works across chat and non-chat checkpoints.
        """
        if self._tokenizer is None or not getattr(self._tokenizer, "chat_template", None):
            return task_prompt.strip()
        return self._tokenizer.apply_chat_template(
            self.build_messages(task_prompt),
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
            **self.chat_template_kwargs,
        )


@dataclass(kw_only=True)
class TokenizationConfig(ITSerializableCfg):
    """Tokenizer selection and behavior: parallelism, special tokens, sequence length."""

    tokenizers_parallelism: bool = True
    local_fast_tokenizer_path: str | None = None
    cust_tokenization_pattern: str | None = None
    special_tokens_dict: dict[str, Any] = field(default_factory=dict)
    max_seq_length: int = 2048  # TODO: force this to be set rather than allowing a default?


@dataclass(kw_only=True)
class DatasetProcessingConfig(ITSerializableCfg):
    """Dataset loading and preprocessing: source, text fields, collation, and caching.

    Caching defaults to DISABLED for reproducibility -- a cached mapped dataset silently survives a
    change to the preprocessing that produced it.
    """

    remove_unused_columns: bool = True
    text_fields: Tuple | None = None
    dataset_path: StrOrPath | None = None
    enable_datasets_cache: bool | None = False  # disable caching unless explicitly set to improve reproducibility
    data_collator_cfg: dict[str, Any] = field(default_factory=dict)
    signature_columns: List | None = field(default_factory=list)
    prepare_data_map_cfg: dict[str, Any] = field(default_factory=dict)


def hf_datasets_cache_root() -> Path:
    """The directory ``datasets`` caches under, resolved from the environment at call time.

    Same precedence as ``datasets.config``: ``HF_DATASETS_CACHE``, else ``$HF_HOME/datasets``, else
    ``$XDG_CACHE_HOME/huggingface/datasets`` (``~/.cache`` when unset), so a task dataset lands on the filesystem the
    user pointed Hugging Face at. Read here rather than imported from ``datasets.config``, whose values freeze when
    ``datasets`` is first imported and would ignore an environment set afterwards.
    """
    if datasets_cache := os.environ.get("HF_DATASETS_CACHE"):
        return Path(datasets_cache)
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home).expanduser() / "datasets"
    xdg_cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg_cache).expanduser() / "huggingface" / "datasets"


#: :class:`ITDataModuleConfig` fields that cannot change a prepared dataset's rows, so they are left out of its key.
#: The tokenizer is identified separately, and ``signature_columns`` is filled in by preparation itself.
_UNKEYED_FIELDS = frozenset(
    {
        "data_collator_cfg",
        "dataloader_kwargs",
        "dataset_path",
        "defer_model_init",
        "enable_datasets_cache",
        "eval_batch_size",
        "os_env_model_auth_key",
        "signature_columns",
        "tokenizer",
        "tokenizers_parallelism",
        "train_batch_size",
    }
)


def _key_material(value: Any) -> Any:
    """``value`` as JSON-serializable data that is the same in every process.

    A value whose only text form is its ``repr`` contributes its class alone, since a ``repr`` may carry a memory
    address, and the ranks of one run must agree on the key.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        declared = {f.name: _key_material(getattr(value, f.name)) for f in fields(value)}
        return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "fields": declared}
    if isinstance(value, Mapping):
        return {str(k): _key_material(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_key_material(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted(json.dumps(_key_material(v), sort_keys=True) for v in value)
    named = value if hasattr(value, "__qualname__") else type(value)
    return f"{named.__module__}.{named.__qualname__}"


@dataclass(kw_only=True)
class ITDataModuleConfig(ITSharedConfig, TokenizationConfig, DatasetProcessingConfig):
    """The datamodule-side configuration: batch sizes and dataloading, over tokenization and dataset processing.

    Composed from the grouped configs rather than declared flat -- see NOTE [Interpretune
    Dataclass-Oriented Configuration].
    """

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
        # defaults for the two worker settings, under every key the user declared (``shuffle``, ``drop_last``, ...)
        self.dataloader_kwargs = {"num_workers": 0, "pin_memory": False, **self.dataloader_kwargs}
        if not self.data_collator_cfg:
            self.data_collator_cfg = {"collator_class": "transformers.DataCollatorWithPadding"}

        # Use pathlib for cross-platform path handling and sanitize task name for Windows compatibility
        sanitized_task_name = self.task_name.replace(":", "_").replace("|", "_")
        rank_zero_debug(f"[DATAMODULE_CONFIG] Sanitized task name: '{sanitized_task_name}'")
        cache_home = hf_datasets_cache_root()
        rank_zero_debug(f"[DATAMODULE_CONFIG] Using datasets cache root: {cache_home}")

        default_dataset_save_path = cache_home / sanitized_task_name
        rank_zero_debug(f"[DATAMODULE_CONFIG] Default dataset path: {default_dataset_save_path}")

        # Ensure proper platform-specific path separators
        explicit_dataset_path = self.dataset_path is not None
        if self.dataset_path is None:
            self.dataset_path = default_dataset_save_path.resolve()
        else:
            # Convert existing path to use proper separators
            self.dataset_path = Path(self.dataset_path).resolve()
        rank_zero_debug(f"[DATAMODULE_CONFIG] Final dataset_path: {self.dataset_path}")
        # the default directory, and the path last derived from it: ``dataset_path`` is the default while it is the
        # latter, so a path set explicitly, here or on the config afterwards, is never rekeyed
        self._default_dataset_path = default_dataset_save_path.resolve()
        self._derived_dataset_path = self._default_dataset_path if not explicit_dataset_path else None

    def prepared_dataset_identity(self, tokenizer: Any) -> dict[str, Any]:
        """What determines a prepared dataset's rows: this configuration's own fields and ``tokenizer``."""
        identity = {f.name: _key_material(getattr(self, f.name)) for f in fields(self) if f.name not in _UNKEYED_FIELDS}
        identity["config_class"] = f"{type(self).__module__}.{type(self).__qualname__}"
        identity["tokenizer_identity"] = [
            type(tokenizer).__name__,
            getattr(tokenizer, "name_or_path", ""),
            len(tokenizer) if hasattr(tokenizer, "__len__") else None,
        ]
        return identity

    def key_default_dataset_path(self, tokenizer: Any, **extra: Any) -> None:
        """Key the default ``dataset_path`` by what determines the dataset, not by the task alone.

        Sessions of one task differ in model, tokenizer and prompt configuration, and each writes and then memory-maps
        its prepared splits, so in one directory they read each other's rows. The datamodule calls this once it has
        configured its tokenizer, and again if the configuration is reused with another. ``extra`` adds what a
        datamodule's preparation depends on beyond its configuration. An explicitly set path is left alone.
        """
        if self._derived_dataset_path is None or self.dataset_path != self._derived_dataset_path:
            return
        identity = {**self.prepared_dataset_identity(tokenizer), **{k: _key_material(v) for k, v in extra.items()}}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        default = self._default_dataset_path
        self.dataset_path = self._derived_dataset_path = default.with_name(f"{default.name}__{key}")
        rank_zero_debug(f"[DATAMODULE_CONFIG] Keyed default dataset_path: {self.dataset_path}")

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
