import os
import inspect
from typing import Any, cast
from functools import reduce
import logging

import torch
import datasets
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from interpretune.utils import rank_zero_info, rank_zero_warn, _import_class, rank_zero_debug
from interpretune.config import ITDataModuleConfig
from interpretune.protocol import SaveHyperparametersProtocol, ITModuleProtocol


log = logging.getLogger(__name__)


################################################################################
# ITDatamodule Definition
################################################################################


# TODO: move core datamodule logic to a separate mixin and compose it with supported hooks analogous to BaseITModule
#       and ITModule
class ITDataModule:
    def __init__(self, itdm_cfg: ITDataModuleConfig, *args, **kwargs):
        r"""

        Args:
            itdm_cfg (:class:`~ITDataModuleConfig`): Configuration for the datamodule.
        """
        # See NOTE [Interpretune Dataclass-Oriented Configuration]
        super().__init__(*args, **kwargs)
        # module handle
        self._module = None
        self.itdm_cfg = itdm_cfg
        if self.itdm_cfg.enable_datasets_cache:  # explicitly toggle datasets caching
            datasets.enable_caching()
        else:
            datasets.disable_caching()
        if hasattr(self, "save_hyperparameters"):
            cast(SaveHyperparametersProtocol, self).save_hyperparameters()
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if self.itdm_cfg.tokenizers_parallelism else "false"
        self.tokenizer = self.configure_tokenizer()
        rank_zero_debug(
            f"[ITDataModule.__init__] Tokenizer initialized: {self.tokenizer.__class__.__name__}, "
            f"padding_side={getattr(self.tokenizer, 'padding_side', None)}, "
            f"model_input_names={getattr(self.tokenizer, 'model_input_names', None)}"
        )
        collator_kwargs = self.itdm_cfg.data_collator_cfg.get("collator_kwargs", None) or {}
        collator_class = _import_class(self.itdm_cfg.data_collator_cfg["collator_class"])
        self.data_collator = collator_class(self.tokenizer, **collator_kwargs)

    @property
    def module(self) -> "ITModuleProtocol | None":
        try:
            module = getattr(self, "_module", None) or reduce(getattr, "trainer.model".split("."), self)
            # Ensure we return the correct type. If a module-like object is attached but isn't a
            # torch.nn.Module, warn the user that we'll cast it for downstream typing assumptions.
            if module is not None and not isinstance(module, torch.nn.Module):
                rank_zero_warn(
                    f"Attached module is not an instance of torch.nn.Module (type={type(module)}); "
                    "casting to torch.nn.Module for downstream compatibility."
                )
            # Cast to the protocol type for static type checkers; runtime cast is a no-op.
            return cast("ITModuleProtocol | None", module)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find module reference (has it been attached yet?): {ae}")
            return None

    def _hook_output_handler(self, hook_name: str, output: Any) -> None:
        rank_zero_warn(f"Output received for hook `{hook_name}` which is not yet supported.")

    def prepare_data(self, target_model: torch.nn.Module | None = None) -> None:
        """Load the SuperGLUE dataset."""

    def setup(self, stage: str | None = None, module: torch.nn.Module | None = None, *args, **kwargs) -> None:
        """Setup our dataset splits for training/validation."""
        # stage is optional for raw pytorch support
        # attaching module handle to datamodule is optional. It can be convenient to align data prep with a  model using
        # signature inspection
        assert self.itdm_cfg.dataset_path is not None, "Dataset path is not configured"
        # Use os.fspath to support both str and PathLike  cleanly
        self.dataset = datasets.load_from_disk(os.fspath(self.itdm_cfg.dataset_path))
        if module is not None:
            self._module = module

    def configure_tokenizer(self) -> PreTrainedTokenizerBase:
        access_token = (
            os.environ[self.itdm_cfg.os_env_model_auth_key.upper()] if self.itdm_cfg.os_env_model_auth_key else None
        )
        ### tokenizer config precedence: pre-configured > pretrained tokenizer name -> model name
        if self.itdm_cfg.tokenizer:
            tokenizer = self.itdm_cfg.tokenizer
            rank_zero_debug(f"[configure_tokenizer] Using pre-configured tokenizer: {tokenizer.__class__.__name__}")
        elif self.itdm_cfg.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                self.itdm_cfg.tokenizer_name, token=access_token, **self.itdm_cfg.tokenizer_kwargs
            )
            rank_zero_debug(f"[configure_tokenizer] Loaded tokenizer from name: {self.itdm_cfg.tokenizer_name}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.itdm_cfg.model_name_or_path, token=access_token, **self.itdm_cfg.tokenizer_kwargs
            )
            rank_zero_debug(
                f"[configure_tokenizer] Loaded tokenizer from model_name_or_path: {self.itdm_cfg.model_name_or_path}"
            )
        _ = tokenizer.add_special_tokens(self.itdm_cfg.special_tokens_dict)
        if self.itdm_cfg.tokenizer_id_overrides:
            for k, v in self.itdm_cfg.tokenizer_id_overrides.items():
                setattr(tokenizer, k, v)
                rank_zero_debug(f"[configure_tokenizer] Overriding tokenizer attribute: {k}={v}")
        rank_zero_debug(
            f"[configure_tokenizer] Final tokenizer config: "
            f"padding_side={getattr(tokenizer, 'padding_side', None)}, "
            f"model_input_names={getattr(tokenizer, 'model_input_names', None)}"
        )
        return tokenizer

    # adapted from HF native trainer
    def _set_signature_columns_if_needed(self, target_model: torch.nn.Module) -> None:
        if not self.itdm_cfg.signature_columns or len(self.itdm_cfg.signature_columns) == 0:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(target_model.forward)
            self.itdm_cfg.signature_columns = list(signature.parameters.keys())

    # adapted from HF native trainer
    # note for raw pytorch we require a target_model (vs getting it from the trainer)
    def _remove_unused_columns(
        self, dataset: "datasets.Dataset", target_model: torch.nn.Module | None = None, description: str | None = None
    ) -> Dataset:
        if not self.itdm_cfg.remove_unused_columns:
            return dataset
        if not self.itdm_cfg.signature_columns:
            if not target_model:
                assert self.module is not None, "No target model provided and module is not set"
                target_model = self.module.model
            assert target_model is not None, "Target model should be available at this point"
            self._set_signature_columns_if_needed(target_model)

        # Ensure signature_columns is not None after setup
        signature_columns = self.itdm_cfg.signature_columns
        if signature_columns is None:
            signature_columns = []

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        # sort for deterministic fingerprinting (datasets caching relies on deterministic column order)
        ignored_columns.sort()
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            target_name = f"`{target_model.__class__.__name__}.forward`"
            rank_zero_info(
                f"The following columns {dset_description} don't have a corresponding argument in {target_name} and"
                f" have been ignored: {', '.join(ignored_columns)}. If {', '.join(ignored_columns)} are not"
                f" expected by {target_name}, you can safely ignore this message."
            )
        return dataset.remove_columns(ignored_columns)

    def __repr__(self):
        if self._module:
            module_str = getattr(self._module, "_orig_module_name", self._module.__class__.__name__)
        else:
            module_str = "No module yet attached."
        tokenizer_str = self.tokenizer.__class__.__name__ if self.tokenizer else "No tokenizer yet defined"
        repr_string = [f"Attached module: {module_str}"]
        repr_string += [f"Attached tokenizer: {tokenizer_str}"]
        return self.__class__.__name__ + "(" + ", ".join(repr_string) + ")"

    def on_train_end(self) -> Any | None:
        """Optionally execute some post-interpretune train session steps."""

    def on_validation_end(self) -> Any | None:
        """Optionally execute some post-interpretune train session steps."""

    def on_test_end(self) -> Any | None:
        """Optionally execute some post-interpretune train session steps."""

    def on_predict_end(self) -> Any | None:
        """Optionally execute some post-interpretune train session steps."""

    def on_analysis_end(self) -> Any | None:
        """Optionally execute relevant post-phase steps."""
