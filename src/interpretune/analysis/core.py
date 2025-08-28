from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Optional, Any, Callable, Sequence, Union, List, Dict, Type
from types import MappingProxyType
import os
from pathlib import Path
from copy import deepcopy

import torch
from torch._dynamo.utils import is_namedtuple_cls, namedtuple_fields
import pandas as pd
import plotly.express as px
from tabulate import tabulate
from transformers import PreTrainedTokenizerBase
from sae_lens.config import HfDataset
from datasets import Features, Array2D, Value, Array3D, load_dataset, Column
from datasets import Sequence as DatasetsSequence

from interpretune.protocol import (
    AnalysisStoreProtocol,
    DefaultAnalysisBatchProtocol,
    BaseAnalysisBatchProtocol,
    NamesFilter,
    SAEFqn,
    AnalysisCfgProtocol,
    StrOrPath,
)
from interpretune.analysis.ops.base import AnalysisOp, OpSchema
from interpretune.analysis.ops.dispatcher import DISPATCHER
from interpretune.utils import rank_zero_warn, DEFAULT_DECODE_KWARGS


class SAEAnalysisDict(dict):
    """Dictionary for SAE-specific data where values must be torch.Tensor or list[torch.Tensor]."""

    def __setitem__(self, key: str, value: torch.Tensor | Sequence[torch.Tensor] | None) -> None:
        if value is None:
            super().__setitem__(key, None)  # type: ignore[arg-type]  # SAEAnalysisDict allows None values for empty SAE data
            return
        # Handle namedtuple-like torch.return_types.<op> objects
        if not isinstance(value, (torch.Tensor, list)):
            if is_namedtuple_cls(type(value)):
                # TODO: decide whether to keep these values as quasi-named tuples or continue converting to
                # lists (we would need to serialize these quasi namedtuples later if we keep them namedtuples here)
                # Extract tensor fields from the namedtuple
                tensor_fields = []
                for field_name in namedtuple_fields(type(value)):
                    field_value = getattr(value, field_name)
                    if isinstance(field_value, torch.Tensor):
                        tensor_fields.append(field_value)
                if tensor_fields:
                    # Store the first tensor field or a list of all tensor fields
                    value = tensor_fields[0] if len(tensor_fields) == 1 else tensor_fields
                else:
                    raise TypeError(f"Namedtuple {type(value).__name__} does not contain any tensor fields")
            else:
                raise TypeError(
                    "Values must be torch.Tensor, list[torch.Tensor], or namedtuple-like containing tensors"
                )

        # TODO: at which point in the pipeline should we remove batches with None or empty list values?
        #       to maintain batch alignment, we keep None valued batches for now and skip them in operations that join
        #       batches
        # TODO: verify that we don't need to validate len(v) > 0 for tensors where ndim > 0
        if isinstance(value, list) and not all(isinstance(v, torch.Tensor) for v in value if v is not None):
            raise TypeError("All list elements must be torch.Tensor")
        super().__setitem__(key, value)

    @property
    def shapes(self) -> dict[str, torch.Size | list[torch.Size]]:
        """Return shapes for each tensor or list of tensors in the dictionary.

        Returns:
            Dictionary mapping SAE names to either single tensor shapes or lists of tensor shapes
        """
        shapes = {}
        for sae, values in self.items():
            if isinstance(values, torch.Tensor):
                shapes[sae] = values.shape
            elif isinstance(values, list):
                shapes[sae] = [t.shape for t in values]
        return shapes

    def batch_join(
        self, across_saes: bool = False, join_fn: Callable = torch.cat
    ) -> SAEAnalysisDict | list[torch.Tensor]:
        """Join field values either by SAE or across SAEs.

        Args:
            join_across_saes: If True, joins values across SAEs for each batch.
                                If False, joins batches for each SAE separately.
            join_fn: Function to use for joining (default: torch.cat)

        Returns:
            If join_across_saes=True: List of tensors, one per batch, with values joined across SAEs
            If join_across_saes=False: SAEAnalysisDict with batches joined for each SAE
        """
        if across_saes:
            # Get number of batches from first SAE's values
            num_batches = len(next(iter(self.values())))

            # For each batch, collect and join tensors from all SAEs
            result = []
            for batch_idx in range(num_batches):
                batch_tensors = []
                for sae_values in self.values():
                    if sae_values[batch_idx] is not None:
                        batch_tensors.append(sae_values[batch_idx])
                if batch_tensors:  # Only join if there are non-None tensors
                    result.append(join_fn(batch_tensors))
                else:
                    result.append(None)
            return result
        else:
            # Join batches for each SAE separately
            result = SAEAnalysisDict()
            for k, v in self.items():
                # Filter out None values before joining
                valid_batches = [batch for batch in v if batch is not None]
                if valid_batches:  # Only join if there are valid batches
                    result[k] = join_fn(valid_batches, dim=0)
                else:
                    result[k] = None
            return result

    def apply_op_by_sae(self, operation: Callable | str, *args, **kwargs) -> SAEAnalysisDict:
        """Apply an operation to each tensor value while preserving SAE keys.

        Args:
            operation: Either callable or string name of torch.Tensor method
            *args, **kwargs: Additional arguments passed to the operation

        Returns:
            SAEAnalysisDict: New dictionary mapping SAE names to operated tensor values

        Examples:
            # Apply mean
            my_dict.batch_join().apply_op_by_sae('mean', dim=0)

            # Apply custom function
            my_dict.batch_join().apply_op_by_sae(torch.mean, dim=0)
        """
        result = SAEAnalysisDict()

        for k, v in self.items():
            if isinstance(operation, str):
                result[k] = getattr(v, operation)(*args, **kwargs)
            else:
                result[k] = operation(v, *args, **kwargs)

        return result


def get_module_dims(module) -> tuple[int, int, int, int, int]:
    """Extract key dimensions from module state.

    Args:
        module: Module instance containing batch information

    Returns:
        Tuple containing:
            batch_size: Current batch size
            max_answer_tokens: Maximum number of answer tokens
            num_classes: Number of output classes
            vocab_size: Size of the vocabulary
            max_seq_len: Maximum sequence length
    """
    # TODO: in the future, this will depend on our configured dataloader (e.g. for train modes etc.)
    batch_size = module.datamodule.itdm_cfg.eval_batch_size
    # TODO: currently only support a single new token, but this will be configurable
    max_answer_tokens = module.it_cfg.generative_step_cfg.lm_generation_cfg.max_new_tokens
    # TODO: abstract this so it's not tied to entailment use case but num_classes
    num_classes = module.it_cfg.num_labels or len(module.it_cfg.entailment_mapping)
    vocab_size = module.model.tokenizer.vocab_size
    max_seq_len = module.model.tokenizer.model_max_length
    # TODO: decide if we want to expose a dict of both latent hook names and dimensions here (not just d_sae)
    # d_saes = tuple(h.cfg.d_sae for h in module.sae_handles) if getattr(module, 'sae_handles', None) else None
    # TODO: max_seq_len not currently used, could be provided for downstream usage
    #       should be same as module.model.cfg.n_ctx, assert this here?
    # max_seq_len = module.model.tokenizer.model_max_length
    return batch_size, max_answer_tokens, num_classes, vocab_size, max_seq_len


def get_filtered_sae_hook_keys(handle, names_filter: Callable[[str], bool]) -> list[str]:
    """Get filtered hook keys based on names_filter.

    Args:
        handle: SAE handle containing hook configuration
        names_filter: Function to filter hook names

    Returns:
        List of filtered hook keys
    """
    return [
        f"{handle.cfg.metadata.hook_name}.{key}"
        for key in handle.hook_dict.keys()
        if names_filter(f"{handle.cfg.metadata.hook_name}.{key}")
    ]


def _check_names_filter_available(module, col_name: str, col_cfg) -> bool:
    """Check if names_filter is available for column configuration.

    Args:
        module: The module being analyzed
        col_name: Name of the column being processed
        col_cfg: Column configuration object

    Returns:
        True if processing should continue, False if column should be skipped

    Raises:
        ValueError: If names_filter is required but not available
    """
    no_filter = (
        not hasattr(module, "analysis_cfg")
        or not hasattr(module.analysis_cfg, "names_filter")
        or module.analysis_cfg.names_filter is None
    )
    if no_filter:
        if col_cfg.required:
            raise ValueError(f"Column '{col_name}' requires names_filter, but module.analysis_cfg.names_filter is None")
        return False  # Skip this field if not required
    return True


def schema_to_features(
    module,
    op: str | AnalysisOp | None = None,
    schema: OpSchema | None = None,
    default_dtype: str = "float32",
    int_dtype: str = "int64",
) -> Features:
    """Convert an operation schema or direct schema to features for Dataset.from_generator.

    Args:
        module: The module being analyzed
        op: An optional AnalysisOp to get the schema from
        schema: An optional direct schema to use instead of op.output_schema

    Returns:
        A features dict compatible with Dataset.from_generator
    """
    from interpretune.analysis.ops.base import AnalysisOpLike

    if isinstance(op, str):
        op = DISPATCHER.get_op(op)  # type: ignore[assignment]  # DISPATCHER.get_op can return callable or AnalysisOp
    if schema is None and op is not None and hasattr(op, "output_schema"):
        schema = op.output_schema  # type: ignore[attr-defined]  # AnalysisOp has output_schema attribute
    elif schema is None and module is not None and hasattr(module, "analysis_cfg"):
        has_output_schema = hasattr(module.analysis_cfg, "output_schema")
        if has_output_schema:
            schema = module.analysis_cfg.output_schema

    # Handle the case where schema is an OpWrapper or AnalysisOp instead of OpSchema
    if isinstance(schema, AnalysisOpLike):
        schema = schema.output_schema

    if not schema:
        return Features()  # Return empty Features if no schema available

    batch_size, max_answer_tokens, num_classes, vocab_size, max_seq_len = get_module_dims(module)
    features_dict = {}

    # Map dimension variables to their actual values
    dim_vars = {
        "batch_size": batch_size,
        "max_answer_tokens": max_answer_tokens,
        "num_classes": num_classes,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
    }

    for col_name, col_cfg in schema.items():
        if col_cfg.intermediate_only:
            continue

        dtype = col_cfg.array_dtype or col_cfg.datasets_dtype

        # Handle per-SAE hook fields first
        if col_cfg.per_sae_hook:
            # Check if names_filter is available
            if not _check_names_filter_available(module, col_name, col_cfg):
                features_dict[col_name] = {}  # Set to empty dict instead of skipping
                continue

            sae_features = {}
            for handle in module.sae_handles:
                for hook_key in get_filtered_sae_hook_keys(handle, module.analysis_cfg.names_filter):
                    if col_cfg.non_tensor and col_cfg.sequence_type:
                        sae_features[hook_key] = DatasetsSequence(Value(dtype=dtype))
                    else:
                        sae_features[hook_key] = Array2D(
                            shape=(None, handle.cfg.d_sae), dtype=handle.cfg.dtype or default_dtype
                        )
            features_dict[col_name] = sae_features
            continue

        # Handle per-latent fields
        if col_cfg.per_latent:
            # Check if names_filter is available
            if not _check_names_filter_available(module, col_name, col_cfg):
                features_dict[col_name] = {}  # Set to empty dict instead of skipping
                continue

            sae_features = {}
            for handle in module.sae_handles:
                for hook_key in get_filtered_sae_hook_keys(handle, module.analysis_cfg.names_filter):
                    # Determine base feature based on ColCfg properties
                    base_feature = Value(dtype=dtype)  # Default to simple Value type
                    if col_cfg.array_shape:
                        shape = list(col_cfg.array_shape)
                        shape = [dim_vars.get(dim, dim) if isinstance(dim, str) else dim for dim in shape]
                        if len(shape) == 2:
                            base_feature = Array2D(shape=tuple(shape), dtype=dtype)
                        elif len(shape) == 3:
                            base_feature = Array3D(shape=tuple(shape), dtype=dtype)
                    elif col_cfg.sequence_type:
                        base_feature = DatasetsSequence(Value(dtype=dtype))

                    sae_features[hook_key] = {
                        "latents": DatasetsSequence(Value(int_dtype)),
                        "per_latent": DatasetsSequence(base_feature),
                    }
            features_dict[col_name] = sae_features
            continue

        # Handle sequence types
        if col_cfg.sequence_type:
            features_dict[col_name] = DatasetsSequence(Value(dtype=dtype))
            continue

        # Handle explicit array shapes
        if col_cfg.array_shape:
            shape = list(col_cfg.array_shape)
            shape = [dim_vars.get(dim, dim) if isinstance(dim, str) else dim for dim in shape]

            if len(shape) == 2:
                features_dict[col_name] = Array2D(shape=tuple(shape), dtype=dtype)
            elif len(shape) == 3:
                features_dict[col_name] = Array3D(shape=tuple(shape), dtype=dtype)
            continue

        # Handle scalar values (no array_shape and not sequence_type)
        features_dict[col_name] = Value(dtype=dtype)

    return Features(features_dict)


class AnalysisStore:
    def __init__(
        self,
        # dataset: can be a path or a loaded Hugging Face dataset
        dataset: HfDataset | StrOrPath | None = None,
        op_output_dataset_path: str | None = None,
        cache_dir: str | None = None,
        streaming: bool = False,
        split: str = "validation",
        it_format_kwargs: dict | None = None,
        stack_batches: bool = False,  # Controls tensor stacking behavior
        protocol_cls: Type[BaseAnalysisBatchProtocol] = DefaultAnalysisBatchProtocol,
    ) -> None:
        self.stack_batches = stack_batches
        self.cache_dir = cache_dir
        self.streaming = streaming
        self.op_output_dataset_path = op_output_dataset_path
        self.split = split
        self.it_format_kwargs = it_format_kwargs
        self._protocol_cls = protocol_cls

        load_dataset_kwargs = dict(split=split, streaming=streaming)

        if isinstance(dataset, (str, Path)):
            dataset_path = os.path.abspath(str(dataset))
            if self.op_output_dataset_path is not None:
                # Only consider conflict if raw paths match
                if str(dataset) == self.op_output_dataset_path:
                    raise ValueError("The dataset path and op_output_dataset_path must not overlap.")
            self.dataset = load_dataset(path=dataset_path, **load_dataset_kwargs)  # type: ignore[misc]  # load_dataset API compatible with these args
        else:
            self.dataset = dataset

        if self.dataset is not None:
            if self.it_format_kwargs is not None:
                # Set custom format options if provided
                self.dataset.set_format(type="interpretune", **self.it_format_kwargs)  # type: ignore[attr-defined]  # datasets API supports set_format
            else:
                # check current format and set it to interpretune if not already set
                if (
                    not hasattr(self.dataset, "format")
                    or getattr(self.dataset, "format", {}).get("type") != "interpretune"
                ):  # type: ignore[attr-defined]  # datasets format attribute
                    # Set default format as our custom analysis formatter
                    self.dataset.set_format(type="interpretune")  # type: ignore[attr-defined]  # datasets API supports set_format

    def _format_columns(self, cols_to_fetch: list[str], indices: Optional[Union[int, slice, list[int]]] = None) -> dict:
        """Internal helper to format specified columns into tensors with proper shape reconstruction.

        Args:
            cols_to_fetch: List of column names to fetch and format
            indices: Optional index, slice or list of indices to select. If None, fetch all rows.
        """
        # Let the dataset handle the formatting using our registered ITAnalysisFormatter
        assert self.dataset is not None, "Dataset should be loaded before formatting columns"
        self.dataset.set_format(type="interpretune", columns=cols_to_fetch)  # type: ignore[attr-defined]  # datasets API supports set_format

        # Handle different types of indexing to get examples
        if indices is None:
            examples = list(self.dataset)  # type: ignore[arg-type]  # datasets are iterable
        elif isinstance(indices, int):
            examples = [self.dataset[indices]]  # type: ignore[index]  # datasets support integer indexing
        elif isinstance(indices, slice):
            start = indices.start or 0
            stop = indices.stop or len(self.dataset)  # type: ignore[arg-type]  # datasets have len
            step = indices.step or 1
            examples = [self.dataset[i] for i in range(start, stop, step)]  # type: ignore[index]  # datasets support integer indexing
        else:
            examples = [self.dataset[idx] for idx in indices]  # type: ignore[index]  # datasets support integer indexing

        # Extract requested columns
        result = {col: [ex[col] for ex in examples] for col in cols_to_fetch}

        # Return single item instead of list for integer indices
        if isinstance(indices, int):
            result = {k: v[0] for k, v in result.items()}

        return result

    @property
    def save_dir(self) -> Path:
        """Directory where datasets will be saved."""
        if self.op_output_dataset_path is None:
            raise ValueError("op_output_dataset_path must be set to save datasets")
        return Path(self.op_output_dataset_path)

    @save_dir.setter
    def save_dir(self, path: StrOrPath) -> None:
        """Set the directory where datasets will be saved."""
        self.op_output_dataset_path = Path(path)

    def _load_dataset(self, dataset: HfDataset | StrOrPath | None) -> None:
        """Load a dataset from a path or existing dataset object."""
        load_dataset_kwargs = dict(
            split=self.split,
            streaming=self.streaming,
        )

        if isinstance(dataset, (str, Path)):
            dataset_path = os.path.abspath(str(dataset))
            if self.op_output_dataset_path:
                # Compare raw paths to detect overlap
                if str(dataset) == self.op_output_dataset_path:
                    raise ValueError("The dataset path and op_output_dataset_path must not overlap.")
            self.dataset = load_dataset(path=dataset_path, **load_dataset_kwargs)  # type: ignore[misc]  # load_dataset API compatible with these args
        else:
            self.dataset = dataset

        # Set default tensor format
        if self.dataset is not None:
            self.dataset.set_format(type="interpretune")  # type: ignore[attr-defined]  # datasets API supports set_format

    def reset_dataset(self) -> None:
        """Reset the dataset."""
        # TODO: decide on appropriate reloading/clearing behavior
        if hasattr(self, "dataset"):
            # Reload dataset from disk if available
            try:
                self._load_dataset(self.save_dir)
            except Exception:
                self.dataset = None

    def _is_tensor_seq(self, data) -> bool:
        """Check if data is a Datasets Column, sequence of torch.Tensors or a single torch.Tensor."""
        return (
            (isinstance(data, Column) and isinstance(data[0], torch.Tensor))
            or isinstance(data, torch.Tensor)
            or (isinstance(data, list) and all(isinstance(x, torch.Tensor) for x in data))
        )

    def __getitem__(self, key: Union[str, List[str], int, slice]) -> Union[List, Dict]:
        """Enable direct column/row access similar to HF Dataset.

        Args:
            key: Column name(s) or row indices

        Returns:
            Selected columns or rows. For tensor data:
            - If stack_batches=False (default): Returns list of individual tensors
            - If stack_batches=True: Returns stacked tensor
        """
        if isinstance(key, str):
            # Single column access
            assert self.dataset is not None, "Dataset should be loaded before accessing columns"
            data = self.dataset[key]  # type: ignore[index]  # datasets support string column indexing

            # now that Datasets >= 4.0 has introduced the Column class, we need to refactor AnalysisStore
            # methods accordingly (since we are pre-MVP, we can start with the Datasets 4.0 API w/o bc concerns)
            # TODO: For the time-being, we will return the underlying tensor data for columns directly, but
            # a future refactor (or subclass of Column) that allows our stacking behavior might make more sense
            if self._is_tensor_seq(data):
                if self.stack_batches:
                    return torch.stack([t for t in data])  # type: ignore[return-value]  # stacked tensor is valid return for tensor sequences
                if hasattr(data, "__getitem__") and len(data) > 0 and hasattr(data[0], "dim") and data[0].dim() > 1:  # type: ignore[index,attr-defined]  # tensor sequence access
                    return [t for t in data]  # type: ignore[return-value]  # list of tensors is valid return
                # Split 1D tensors into scalar tensors
                return [torch.tensor(x) for x in data]  # type: ignore[return-value]  # list of tensors is valid return
            return data  # type: ignore[return-value]  # raw data return is valid
        elif isinstance(key, list) and all(isinstance(k, str) for k in key):
            # Multiple column access
            result = {}
            for col in key:
                assert self.dataset is not None, "Dataset should be loaded before accessing columns"
                data = self.dataset[col]  # type: ignore[index]  # datasets support string column indexing
                # If tensor data, optionally stack individual tensors
                if self._is_tensor_seq(data):
                    if self.stack_batches:
                        result[col] = torch.stack([t for t in data])  # type: ignore[misc]  # torch.stack accepts tensor sequences
                    elif (
                        hasattr(data, "__getitem__") and len(data) > 0 and hasattr(data[0], "dim") and data[0].dim() > 1
                    ):  # type: ignore[index,attr-defined]  # tensor sequence access
                        result[col] = [t for t in data]
                    else:
                        # Split 1D tensors into scalar tensors
                        result[col] = [torch.tensor(x) for x in data]
                else:
                    result[col] = data
            return result
        else:
            # Row access delegates to dataset
            assert self.dataset is not None, "Dataset should be loaded before accessing rows"
            return self.dataset[key]  # type: ignore[index]  # datasets support various indexing types

    def select_columns(self, column_names: List[str]) -> "AnalysisStore":
        """Select a subset of columns.

        Args:
            column_names: List of column names to select

        Returns:
            New AnalysisStore with only selected columns
        """
        assert self.dataset is not None, "Dataset should be loaded before selecting columns"
        self.dataset = self.dataset.select_columns(column_names)  # type: ignore[attr-defined]  # datasets support select_columns
        return self

    def __getattr__(self, name: str) -> Any:
        """Allow accessing dataset columns and attributes.

        Args:
            name: Name of column or dataset attribute to fetch

        Returns:
            Column data if name exists in protocol annotations and dataset,
            otherwise returns dataset attribute.

        Raises:
            AttributeError: If name doesn't exist in protocol annotations or dataset attributes
        """
        # First check if it's a protocol-defined column
        if name in self._protocol_cls.__annotations__:
            # Check if the column actually exists in the dataset
            if (
                self.dataset is not None
                and hasattr(self.dataset, "column_names")
                and name in getattr(self.dataset, "column_names", [])
            ):  # type: ignore[attr-defined]  # datasets have column_names
                return self[name]
            # Return None if column isn't available yet
            return None

        # If not, try to get attribute from dataset
        if hasattr(self.dataset, name):
            attr = getattr(self.dataset, name)
            # Handle callable attributes (e.g. dataset methods)
            if callable(attr):

                def _caller(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    # If the result is a Dataset, ensure it maintains the interpretune format
                    if hasattr(result, "set_format"):
                        result.set_format(type="interpretune")  # type: ignore[attr-defined]  # datasets support set_format
                    return result

                return _caller
            return attr

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def by_sae(self, field_name: str, stack_latents: bool = True) -> SAEAnalysisDict:
        """Transform batch-oriented field values into per-SAE lists of batch values.

        Args:
            field_name: Name of the field to process (e.g. 'correct_activations', 'preds')
            stack_latents: Whether to stack latent values using torch.stack for nested dictionary fields

        Returns:
            SAEAnalysisDict: Dictionary mapping SAE names to lists of batch values.
            For nested dictionary fields, latent values within each batch are stacked if stack_latents=True.

        Raises:
            TypeError: If the values are not dictionaries and thus cannot be transformed into an SAEAnalysisDict
        """
        values = self.__getattr__(field_name)
        assert values, f"No values found for field {field_name}"
        if not isinstance(values[0], dict):
            raise TypeError(
                f"Values for field {field_name} must be dictionaries to be transformed into an SAEAnalysisDict"
            )
        result = SAEAnalysisDict()
        sae_names = values[0].keys()
        for sae in sae_names:
            if isinstance(values[0][sae], dict) and stack_latents:
                # Stack latent tensors for each batch
                batch_tensors = []
                for batch in values:
                    latent_tensors = [t for t in batch[sae].values()]
                    batch_tensors.append(torch.stack(latent_tensors) if latent_tensors else None)
                result[sae] = batch_tensors  # type: ignore[assignment]  # SAEAnalysisDict accepts list of tensor batches
            else:
                # Handle both non-nested and non-stacked nested cases
                result[sae] = [  # type: ignore[assignment]  # SAEAnalysisDict handles mixed tensor/None lists
                    None if isinstance(batch[sae], list) and not batch[sae] else batch[sae] for batch in values
                ]
        return result

    def calc_activation_summary(self) -> ActivationSumm:
        """Calculate per-SAE activation summary from analysis cache.

        Computes mean activations and number of non-zero activations per SAE based on activation data
        stored in AnalysisStore. The cache must contain 'correct_activations' data.

        Returns:
            ActivationSumm: Container with:
                - mean_activation: Mean activation values per SAE
                - num_samples_active: Number of non-zero activations per SAE

        Raises:
            ValueError: If no 'correct_activations' data is present in analysis_store.
        """
        if not self.correct_activations:
            raise ValueError("Analysis cache requires 'correct_activations' data to calculate per-SAE activation stats")

        sae_data = self.by_sae("correct_activations").batch_join()
        assert isinstance(sae_data, SAEAnalysisDict), "batch_join should return SAEAnalysisDict"
        mean_activation = sae_data.apply_op_by_sae(operation="mean", dim=0)
        num_samples_active = sae_data.apply_op_by_sae(operation=torch.count_nonzero, dim=0)

        return ActivationSumm(mean_activation=mean_activation, num_samples_active=num_samples_active)

    def calculate_latent_metrics(
        self,
        pred_summ: PredSumm,
        activation_summary: ActivationSumm | None = None,
        filter_by_correct: bool = True,
        run_name: str | None = None,
    ) -> LatentMetrics:
        """Calculate latent metrics from analysis cache.

        Args:
            pred_summ: Prediction summary containing model predictions
            activation_summary: Optional summary of activation statistics. If None, will be calculated.
            filter_by_correct: If True, only use examples with correct predictions. Default True.
            run_name: Optional name for this analysis run

        Returns:
            LatentMetrics object containing computed metrics
        """
        if activation_summary is None:
            activation_summary = self.calc_activation_summary()

        total_examples = len(torch.cat(self.orig_labels))

        correct_mask = None
        if filter_by_correct:
            if pred_summ.batch_predictions is not None:
                correct_mask = torch.cat(
                    [(labels == preds) for labels, preds in zip(self.orig_labels, pred_summ.batch_predictions)]
                )
            else:
                correct_mask = torch.cat([(diffs > 0) for diffs in self.logit_diffs])
            total_examples = correct_mask.sum()

        assert isinstance(activation_summary.num_samples_active, SAEAnalysisDict), (
            "num_samples_active should be SAEAnalysisDict"
        )
        proportion_samples_active = activation_summary.num_samples_active.apply_op_by_sae(
            operation=torch.div, other=total_examples
        )

        attribution_values = self.by_sae("attribution_values").batch_join()
        assert isinstance(attribution_values, SAEAnalysisDict), "batch_join should return SAEAnalysisDict"

        if filter_by_correct:
            per_example_latent_effects = attribution_values.apply_op_by_sae(
                operation=lambda x, mask: x[mask], mask=correct_mask
            )
        else:
            per_example_latent_effects = attribution_values  # type: ignore[assignment]  # attribution_values is SAEAnalysisDict

        total_effect = per_example_latent_effects.apply_op_by_sae(operation=torch.sum, dim=0)
        # TODO: make mean effect normalized by num samples active?
        mean_effect = per_example_latent_effects.apply_op_by_sae(operation=torch.mean, dim=0)

        return LatentMetrics(
            total_effect=total_effect,
            mean_effect=mean_effect,
            proportion_samples_active=proportion_samples_active,
            mean_activation=activation_summary.mean_activation,
            num_samples_active=activation_summary.num_samples_active,
            run_name=run_name,
        )

    def plot_latent_effects(self, per_batch: bool | None = False, title_prefix="Latent effects of"):
        """Plot Latent Effects aggregated or per-batch.

        Args:
            per_batch: If True, plot effects per batch. If False, aggregate across all batches. Defaults to True.
            title_prefix: Optional string prefix for plot titles. Defaults to "Latent effects of".

        Returns:
            None - displays plots in notebook
        """
        if per_batch:
            # Plot per batch
            for i, batch in enumerate(self.attribution_values):
                for act_name, attribution_values in batch.items():
                    len_alive = len(self.alive_latents[i][act_name])
                    px.line(
                        attribution_values.mean(dim=0).cpu().numpy(),
                        title=f"{title_prefix} {act_name} latent effect on logit diff of batch {i} ({len_alive} alive)",
                        labels={"index": "Latent", "value": "Latent effect on logit diff"},
                        template="ggplot2",
                        width=1000,
                    ).update_layout(showlegend=False).show()
        else:
            # Aggregate across all batches using SAEAnalysisDict operations
            stacked_values = self.by_sae("attribution_values").batch_join()
            assert isinstance(stacked_values, SAEAnalysisDict), "batch_join should return SAEAnalysisDict"
            len_alives = {
                act_name: len({latent for batch in self.alive_latents for latent in batch.get(act_name, [])})
                for act_name in stacked_values.keys()
            }

            mean_effects = stacked_values.apply_op_by_sae(operation="mean", dim=0)

            for act_name, effects in mean_effects.items():
                px.line(
                    effects.cpu().numpy(),
                    title=(
                        f"{title_prefix} ({act_name}) Latent effect on logit diff "
                        f"(aggregated, {len_alives[act_name]} alive)"
                    ),
                    labels={"index": "Latent", "value": "Latent effect on logit diff"},
                    template="ggplot2",
                    width=1000,
                ).update_layout(showlegend=False).show()

    def __deepcopy__(self, memo):
        """Deep copy the AnalysisStore, using memo to avoid recursion and handling non-deepcopyable attributes.

        Special care is taken to avoid triggering __getattr__ recursion on MagicMock or similar objects.
        """
        if id(self) in memo:
            return memo[id(self)]
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            try:
                if k == "op_output_dataset_path" and v:
                    # Append _<objectid> to the path string
                    new_path = f"{v}_{id(result)}"
                    setattr(result, k, new_path)
                else:
                    setattr(result, k, deepcopy(v, memo))
            except Exception:
                rank_zero_warn(f"Failed to deepcopy attribute {k} of type {type(v).__name__}. Skipping.")
        return result


def default_sae_id_factory_fn(layer: int, prefix_pat: str = "blocks", suffix_pat: str = "hook_z") -> str:
    return ".".join([prefix_pat, str(layer), suffix_pat])


def default_sae_hook_match_fn(
    in_name: str,
    layers: int | Sequence[int] | None = None,
    hook_point_suffix: str = "hook_sae_acts_post",
    hook_point_prefix: str = "blocks",
) -> bool:
    suffix_matched = in_name.endswith(f"{hook_point_suffix}")
    if suffix_matched and layers is not None:
        if isinstance(layers, int):
            layers = [layers]
        return any(in_name.startswith(f"{hook_point_prefix}.{layer}.") for layer in layers)
    return suffix_matched


@dataclass
class SAEAnalysisTargets:
    """Encapsulation of SAE IDs and specific hooks involved in an SAE-mediated analysis, along with helper
    functions for explicit or pattern based matching by name and/or layer."""

    sae_fqns: list[SAEFqn] = field(default_factory=list)
    sae_release: str = "gpt2-small-hook-z-kk"
    target_sae_ids: Sequence[str] = field(default_factory=list)  # explicit sae_id list
    sae_id_factory_fn: Callable = default_sae_id_factory_fn  # function to generate sae_ids based on layer
    target_layers: list[int] = field(default_factory=list)
    # don't want to overload "hook_pattern" term, used with or without target layers to filter desired hooks
    sae_hook_match_fn: Callable = default_sae_hook_match_fn

    def __post_init__(self):
        self.validate_sae_fqns()

    def validate_sae_fqns(self):
        # Validate provided sae_fqns or generate them based on target_sae_ids or target_layers
        if self.sae_fqns:
            new_sae_fqns = []
            for s in self.sae_fqns:
                if isinstance(s, SAEFqn):
                    new_sae_fqns.append(s)
                elif isinstance(s, tuple) and len(s) == 2:
                    new_sae_fqns.append(SAEFqn(release=s[0], sae_id=s[1]))
                else:
                    raise TypeError("All elements in sae_fqns must be instances of SAEFqn or 2-tuples")
            self.sae_fqns = tuple(new_sae_fqns)  # type: ignore[misc]  # assignment to dataclass field with SAEFqn tuple
        else:
            if self.target_sae_ids:
                self.sae_fqns = tuple(SAEFqn(release=self.sae_release, sae_id=s) for s in self.target_sae_ids)  # type: ignore[misc]  # assignment to dataclass field
            elif self.target_layers:
                self.sae_fqns = tuple(  # type: ignore[misc]  # assignment to dataclass field
                    SAEFqn(release=self.sae_release, sae_id=self.sae_id_factory_fn(layer))
                    for layer in self.target_layers
                )
            else:
                rank_zero_warn("SAEFqns could not be resolved based on provided configuration")
                self.sae_fqns = tuple()  # type: ignore[misc]  # assignment to dataclass field
        return self.sae_fqns


@dataclass(kw_only=True)
class BaseMetrics:
    """Base class for all latent metrics containers."""

    custom_repr: dict[str, str] = field(default_factory=dict)
    run_name: str | None = None

    def __post_init__(self):
        self._set_field_repr()
        # Validation of metric dictionaries
        metric_dicts = [
            value
            for value in self.__dict__.values()
            if isinstance(value, dict) and value not in (self.custom_repr, self._field_repr)
        ]

        if metric_dicts:
            reference_keys = set(metric_dicts[0].keys())
            if not all(set(d.keys()) == reference_keys for d in metric_dicts):
                raise ValueError("All hook dictionaries must have the same keys")

    def get_field_name(self, field: str) -> str:
        """Get display name for a metric field."""
        return self._field_repr.get(field, field)

    def _set_field_repr(self, default_field_repr: dict | None = None) -> None:
        """Update field representations with defaults and custom values."""
        # Initialize field_repr if not already done
        if not hasattr(self, "_field_repr"):
            self._field_repr = {}
        default_values = default_field_repr or {}
        self._field_repr.update({**default_values, **self.custom_repr})

    def get_field_names(self, dict_only: bool = False) -> dict[str, str]:
        """Get all non-protected field names and their representations.

        Args:
            dict_only: If True, return only fields that are dictionaries

        Returns:
            Dict mapping field names to their display representations
        """
        return {
            f: r
            for f, r in self._field_repr.items()
            if f != "custom_repr" and (not dict_only or isinstance(getattr(self, f), dict))
        }


@dataclass(kw_only=True)
class ActivationSumm(BaseMetrics):
    """Container for activation summary metrics."""

    mean_activation: dict[str, torch.Tensor]
    num_samples_active: dict[str, torch.Tensor]

    def __post_init__(self):
        _default_field_repr = MappingProxyType(
            {"mean_activation": "Mean Activation", "num_samples_active": "Number Active"}
        )
        self._set_field_repr(_default_field_repr)  # type: ignore[arg-type]  # MappingProxyType is compatible with dict
        super().__post_init__()


@dataclass(kw_only=True)
class LatentMetrics(ActivationSumm):
    """Container for latent analysis metrics.

    Each metric maps sae names to tensors of latent-level statistics.
    """

    total_effect: dict[str, torch.Tensor]
    mean_effect: dict[str, torch.Tensor]
    proportion_samples_active: dict[str, torch.Tensor]

    def __post_init__(self):
        _default_field_repr = MappingProxyType(
            {
                "total_effect": "Total Effect",
                "mean_effect": "Mean Effect",
                "proportion_samples_active": "Proportion Active",
            }
        )
        self._set_field_repr(_default_field_repr)  # type: ignore[arg-type]  # MappingProxyType is compatible with dict
        super().__post_init__()

    # TODO: decompose this function into smaller, more testable parts
    def create_attribution_tables(
        self,
        sort_by: str = "total_effect",
        top_k: int = 10,
        filter_type: Literal["positive", "negative", "both"] = "both",
        per_sae: bool | None = False,
    ) -> dict[str, str]:
        """Creates formatted tables of attribution metrics.

        Args:
            metrics: Instance of LatentMetrics or subclass containing attribution data
            sort_by: Attribute name from metrics instance to sort by
            top_k: Number of top entries to include
            filter_type: Which values to include ('positive', 'negative', or 'both')
            per_hook: Whether to create separate tables per hook
        """
        # Validate sort_by attribute exists
        if not hasattr(self, sort_by):
            valid_attrs = [attr for attr in dir(self) if not attr.startswith("_")]
            raise ValueError(f"Invalid sort_by field '{sort_by}'. Must be one of: {valid_attrs}")

        sort_metric = getattr(self, sort_by)
        tables = {}
        hooks = list(sort_metric.keys()) if per_sae else ["all"]

        # Get metric names and their display representations
        metric_names = self.get_field_names(dict_only=True)

        for hook in hooks:
            for sign in ["positive", "negative"] if filter_type == "both" else [filter_type]:
                largest = sign == "positive"
                if per_sae:
                    values = sort_metric[hook]
                    topk_values, indices = torch.topk(values, min(top_k, len(values)), largest=largest)
                    table_data = []
                    for idx, val in zip(indices, topk_values):
                        if (largest and val > 0) or (not largest and val < 0):
                            row = {
                                "Hook": hook,
                                "Latent Index": idx.item(),
                            }
                            for metric_attr, display_name in metric_names.items():
                                metric_values = getattr(self, metric_attr)
                                row[display_name] = f"{float(metric_values[hook][idx]):.4f}"
                            table_data.append(row)
                else:
                    # Combine values from all hooks
                    all_values = []
                    for h in sort_metric.keys():
                        values = sort_metric[h]
                        topk_values, indices = torch.topk(values, min(top_k, len(values)), largest=largest)
                        for idx, val in zip(indices, topk_values):
                            if (largest and val > 0) or (not largest and val < 0):
                                all_values.append((h, idx, val))
                    # Sort combined values
                    all_values.sort(key=lambda x: x[2], reverse=largest)
                    table_data = []
                    for h, idx, _ in all_values[:top_k]:
                        row = {
                            "Hook": h,
                            "Latent Index": idx.item(),
                        }
                        for metric_attr, display_name in metric_names.items():
                            metric_values = getattr(self, metric_attr)
                            row[display_name] = f"{float(metric_values[h][idx]):.4f}"
                        table_data.append(row)

                if table_data:
                    title = f"Top {top_k} {sign} {sort_by} "
                    title += f"for {hook}" if per_sae else "across all hooks"
                    tables[title] = tabulate(table_data, headers="keys", tablefmt="pipe")

        return tables


class PredSumm(NamedTuple):
    total_correct: int
    percentage_correct: float
    batch_predictions: list | None


def latent_metrics_scatter(
    metrics1: LatentMetrics,
    metrics2: LatentMetrics,
    metric_field: str = "total_effect",
    label1: str = "Metrics 1",
    label2: str = "Metrics 2",
    width: int = 800,
    height: int = 600,
) -> None:
    """Create scatter plots comparing two sets of LatentMetrics.

    Args:
        metrics1: First LatentMetrics to compare
        metrics2: Second LatentMetrics to compare
        metric_field: Name of metric field to compare (default: 'total_effect')
        label1: Label for first metrics set
        label2: Label for second metrics set
        width: Plot width in pixels
        height: Plot height in pixels
    """
    if not hasattr(metrics1, metric_field) or not hasattr(metrics2, metric_field):
        raise ValueError(f"Metric field '{metric_field}' not found in one or both metrics")

    metrics1_data = getattr(metrics1, metric_field)
    metrics2_data = getattr(metrics2, metric_field)

    for hook_name in metrics1_data.keys():
        df = pd.DataFrame(
            {
                label1: metrics1_data[hook_name].numpy(),
                label2: metrics2_data[hook_name].numpy(),
                "Latent": torch.arange(metrics2_data[hook_name].size(0)).numpy(),
            }
        )

        px.scatter(
            df,
            x=label1,
            y=label2,
            hover_data=["Latent"],
            title=f"{label2} vs {label1} {metric_field} for {hook_name}",
            template="ggplot2",
            width=width,
            height=height,
        ).add_shape(
            type="line",
            x0=metrics2_data[hook_name].min(),
            x1=metrics2_data[hook_name].max(),
            y0=metrics2_data[hook_name].min(),
            y1=metrics2_data[hook_name].max(),
            line=dict(color="red", width=2, dash="dash"),
        ).show()


def base_vs_sae_logit_diffs(
    sae: AnalysisStoreProtocol,
    base_ref: AnalysisStoreProtocol,
    tokenizer: PreTrainedTokenizerBase,
    top_k: int = 10,
    max_prompt_width: int = 80,
) -> None:
    """Display a table comparing reference vs SAE logit differences.

    Args:
        sae: Analysis cache from clean with SAE run
        no_sae_ref: Analysis cache from clean without SAE reference run
        tokenizer: Tokenizer for decoding labels
        top_k: Number of top samples to show
        max_prompt_width: Maximum width for prompt column
    """
    translated_labels = [tokenizer.batch_decode(label_ids, **DEFAULT_DECODE_KWARGS) for label_ids in base_ref.label_ids]

    df = pd.DataFrame(
        {
            "prompt": sae.prompts,
            "correct_answer": translated_labels,
            "clean_logit_diff": base_ref.logit_diffs,
            "sae_logit_diff": sae.logit_diffs,
        }
    )
    df = df.explode(["prompt", "correct_answer", "clean_logit_diff", "sae_logit_diff"])
    df["sample_id"] = range(len(df))
    df = df[["sample_id", "prompt", "correct_answer", "clean_logit_diff", "sae_logit_diff"]]
    df = df[df.clean_logit_diff > 0].sort_values(by="clean_logit_diff", ascending=False)  # type: ignore[misc]  # pandas DataFrame sort_values method

    max_samples = min(top_k, len(df))
    df = df.head(max_samples)

    print(
        tabulate(
            df,
            headers=["Sample ID", "Prompt", "Answer", "Clean Logit Diff", "SAE Logit Diff"],
            maxcolwidths=[None, max_prompt_width, None, None, None],
            tablefmt="grid",
            numalign="left",
            floatfmt="+.3f",
            showindex="never",
        )
    )


# TODO: convert compute_correct to an AnalysisOp?  Assumes sae usage, need to clarify req
def compute_correct(
    analysis_obj: AnalysisStoreProtocol | AnalysisCfgProtocol, op: str | AnalysisOp | None = None
) -> PredSumm:
    """Compute correct prediction statistics for a given analysis mode."""
    # Handle input type and get op and analysis_store
    if hasattr(analysis_obj, "output_store") and hasattr(analysis_obj, "op"):
        analysis_store = analysis_obj.output_store
        op = analysis_obj.op  # type: ignore[assignment]  # analysis objects have op attribute
    else:
        # TODO: we can assume this is None if not provided, we should really change compute_correct to have a per-latent
        #       data structured overload
        # if op is None:
        #    raise ValueError("op argument required when passing AnalysisStore type objects")
        analysis_store = analysis_obj

    if isinstance(op, str):
        op = DISPATCHER.get_op(op)  # type: ignore[assignment]  # DISPATCHER.get_op can return AnalysisOp

    # TODO: this is another location where we should be conditioning behavior on op functionality, not name
    if hasattr(op, "ctx_key") and op.ctx_key == "logit_diffs_attr_ablation":  # type: ignore[attr-defined]  # AnalysisOp has ctx_key
        batch_preds = [b.mode(dim=0).values.cpu() for b in analysis_store.by_sae("preds").batch_join(across_saes=True)]  # type: ignore[attr-defined]  # analysis_store has by_sae method
    else:
        batch_preds = analysis_store.preds  # type: ignore[attr-defined]  # analysis_store has preds attribute
    correct_statuses = [
        (labels == preds).nonzero().unique().size(0)
        for labels, preds in zip(analysis_store.orig_labels, batch_preds)  # type: ignore[attr-defined]  # analysis_store has orig_labels
    ]
    total_correct = sum(correct_statuses)
    percentage_correct = total_correct / (len(torch.cat(analysis_store.orig_labels))) * 100  # type: ignore[attr-defined]  # analysis_store has orig_labels
    return PredSumm(
        total_correct,
        percentage_correct,
        batch_preds if hasattr(op, "ctx_key") and op.ctx_key == "logit_diffs_attr_ablation" else None,  # type: ignore[attr-defined]  # AnalysisOp has ctx_key
    )


def resolve_names_filter(names_filter: NamesFilter | None) -> Callable[[str], bool]:
    # similar to logic in `transformer_lens.hook_points.get_caching_hooks` but accessible to other functions
    if names_filter is None:
        names_filter = lambda name: True
    elif isinstance(names_filter, str):
        filter_str = names_filter
        names_filter = lambda name: name == filter_str
    elif isinstance(names_filter, list):
        filter_list = names_filter
        names_filter = lambda name: name in filter_list
    elif callable(names_filter):
        names_filter = names_filter
    else:
        raise ValueError("names_filter must be a string, list of strings, or function")
    assert callable(names_filter)
    return names_filter


def _make_simple_cache_hook(cache_dict: dict, is_backward: bool = False) -> Callable:
    """Create a hook function that caches activations.

    Args:
        cache_dict: Dictionary to store cached activations
        is_backward: Whether this is a backward hook. Default False.

    Returns:
        Callable: Hook function that caches activations
    """

    def cache_hook(act, hook):
        assert hook.name is not None
        hook_name = hook.name
        if is_backward:
            hook_name += "_grad"
        cache_dict[hook_name] = act.detach()

    return cache_hook


# def validate_analysis_order(self):
#     """Validates and potentially reorders analysis operations to ensure dependencies are met.

#     Currently ensures that logit_diffs.sae comes before ablation if ablation is enabled.
#     """
#     # Check if ablation analysis is requested
#     if it.logit_diffs_attr_ablation in self.analysis_ops:
#         # Ensure logit_diffs.sae is included and comes before ablation
#         if it.logit_diffs_sae not in self.analysis_ops:
#             print("Note: Adding logit_diffs.sae op since it is required for ablation")
#             self.analysis_ops = tuple([it.logit_diffs_sae] + list(self.analysis_ops))
#         # Sort ops to ensure logit_diffs.sae comes before ablation
#         sorted_ops = sorted(self.analysis_ops,
#                           key=lambda x: (x != it.logit_diffs_sae,
#                                        x != it.logit_diffs_attr_ablation))
#         if sorted_ops != list(self.analysis_ops):
#             print("Note: Re-ordering analysis ops to ensure logit_diffs.sae runs before ablation")
#             self.analysis_ops = tuple(sorted_ops)
