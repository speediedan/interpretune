import os
import inspect
from typing import Optional, Callable, Any
from functools import reduce
import logging

import torch
import datasets
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase

from interpretune.utils.logging import rank_zero_info, rank_zero_warn
from interpretune.utils.import_utils import _import_class, _LIGHTNING_AVAILABLE
from interpretune.base.config.datamodule import ITDataModuleConfig

log = logging.getLogger(__name__)


################################################################################
# ITDatamodule Definition
################################################################################

# TODO: move overridden datamodule hooks to a separate class (maybe a separate module, maybe this one)
# TODO: switch ITDataModule from ABC to concrete class and rely upon Protocol?
class ITDataModule:

    tokenization_func: Callable

    def __init__(
        self,
        itdm_cfg: ITDataModuleConfig,
    ):
        r"""Initialize the ``LightningDataModule`` designed for both the RTE or BoolQ SuperGLUE Hugging Face
        datasets.

        Args:
            itdm_cfg (:class:`~ITDataModuleConfig`): Configuration for the datamodule.
        """
        # See NOTE [Interpretune Dataclass-Oriented Configuration]
        super().__init__()
        # module handle
        self._module = None
        self.itdm_cfg = itdm_cfg
        if self.itdm_cfg.enable_datasets_cache:  # explicitly toggle datasets caching
            datasets.enable_caching()
        else:
            datasets.disable_caching()
        if hasattr(self, 'save_hyperparameters'):
            self.save_hyperparameters()
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if self.itdm_cfg.tokenizers_parallelism else "false"
        self.tokenizer = self.configure_tokenizer()
        collator_kwargs = self.itdm_cfg.data_collator_cfg.get('collator_kwargs', None) or {}
        collator_class = _import_class(self.itdm_cfg.data_collator_cfg["collator_class"])
        self.data_collator = collator_class(self.tokenizer, **collator_kwargs)

    @property
    def module(self) -> Optional[torch.nn.Module]:
        try:
            module = getattr(self, "_module", None) or reduce(getattr, "trainer.model".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find module reference (has it been attached yet?): {ae}")
            module = None
        return module

    def _hook_output_handler(self, hook_name: str, output: Any) -> None:
        rank_zero_warn(f"Output received for hook `{hook_name}` which is not yet supported.")

    # @abstractmethod
    def prepare_data(self, target_model: Optional[torch.nn.Module] = None) -> None:
        """Load the SuperGLUE dataset."""

    def setup(self, stage: Optional[str] = None, module: Optional[torch.nn.Module] = None, *args, **kwargs) -> None:
        """Setup our dataset splits for training/validation."""
        # stage is optional for raw pytorch support
        # attaching module handle to datamodule is optional. It can be convenient to align data prep witha  model using
        # signature inspection
        self.dataset = datasets.load_from_disk(self.itdm_cfg.dataset_path)
        self._module = module

    def configure_tokenizer(self) -> PreTrainedTokenizerBase:
        access_token = os.environ[self.itdm_cfg.os_env_model_auth_key.upper()] if self.itdm_cfg.os_env_model_auth_key \
              else None
        ### tokenizer config precedence: preconfigured > local fast path > pretrained tokenizer name -> model name
        if self.itdm_cfg.tokenizer:
            tokenizer = self.itdm_cfg.tokenizer
        elif self.itdm_cfg.local_fast_tokenizer_path:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(self.itdm_cfg.local_fast_tokenizer_path,
                                                                **self.itdm_cfg.tokenizer_kwargs)
        elif self.itdm_cfg.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                self.itdm_cfg.tokenizer_name, token=access_token, **self.itdm_cfg.tokenizer_kwargs
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.itdm_cfg.model_name_or_path, token=access_token, **self.itdm_cfg.tokenizer_kwargs
            )
        # TODO: reconsider whether adding of special tokens be excluded for the local fast tokenizer path
        _ = tokenizer.add_special_tokens(self.itdm_cfg.special_tokens_dict)
        if self.itdm_cfg.tokenizer_id_overrides:
            for k, v in self.itdm_cfg.tokenizer_id_overrides.items():
                setattr(tokenizer, k, v)
        return tokenizer

    # adapted from HF native trainer
    def _set_signature_columns_if_needed(self, target_model: torch.nn.Module) -> None:
        if len(self.itdm_cfg.signature_columns) == 0:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(target_model.forward)
            self.itdm_cfg.signature_columns = list(signature.parameters.keys())

    # adapted from HF native trainer
    # note for raw pytorch we require a target_model (vs getting it from the trainer in the lightning version)
    def _remove_unused_columns(self, dataset: "datasets.Dataset", target_model: Optional[torch.nn.Module] = None,
                               description: Optional[str] = None) -> Dataset:
            if not self.itdm_cfg.remove_unused_columns:
                return dataset
            if not self.itdm_cfg.signature_columns:
                if not target_model:
                    target_model = self.module.model
                self._set_signature_columns_if_needed(target_model)
            ignored_columns = list(set(dataset.column_names) - set(self.itdm_cfg.signature_columns))
            if len(ignored_columns) > 0:
                dset_description = "" if description is None else f"in the {description} set"
                target_name = f"`{target_model.__class__.__name__}.forward`"
                rank_zero_info(
                    f"The following columns {dset_description} don't have a corresponding argument in {target_name} and"
                    f" have been ignored: {', '.join(ignored_columns)}. If {', '.join(ignored_columns)} are not"
                    f" expected by {target_name}, you can safely ignore this message."
                )
            return dataset.remove_columns(ignored_columns)

    def on_train_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune train session steps."""

    def on_validation_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune train session steps."""

    def on_test_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune train session steps."""

    def on_predict_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune train session steps."""

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import LightningDataModule
    class ITLightningDataModule(ITDataModule, LightningDataModule):
        ...
else:
    ITLightningDataModule = object
