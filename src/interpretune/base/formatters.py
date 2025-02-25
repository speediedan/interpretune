from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Any, Dict, Sequence, Union, Mapping, Optional
import torch
import numpy as np
from datasets.formatting import TensorFormatter
import pyarrow as pa
from contextlib import contextmanager

from interpretune.base.contract.analysis import ColCfg


class ITAnalysisFormatter(TensorFormatter[Mapping, "torch.Tensor", Mapping]):
    """Formatter for Interpretune analysis data.

    Handles both tensor and non-tensor SAE fields appropriately.
    """
    # TODO: abstract custom sae-hook transformation (per_latent and per_hook), nontensor cfg, and dyn_dim functionality
    #       into a separate interface with explicit hook points into standard TorchFormatter logic
    def __init__(self, features=None, **format_kwargs):
        super().__init__(features=features)
        # Extract column config and rebuild ColCfg objects from dicts
        col_cfg = format_kwargs.pop('col_cfg', {})
        col_cfg = {k: ColCfg.from_dict(v) for k, v in col_cfg.items()}

        self.dyn_dims = {
            k: cfg.dyn_dim for k, cfg in col_cfg.items()
            if cfg.dyn_dim is not None
        }
        self.non_tensor_fields = {
            k for k, cfg in col_cfg.items() if cfg.non_tensor
        }
        self.per_latent_fields = {
            k for k, cfg in col_cfg.items() if cfg.per_latent
        }
        self.torch_tensor_kwargs = format_kwargs
        self._field_context = []

    @contextmanager
    def field_context(self, field_info: Union[tuple[Optional[str], dict], Optional[str]]):
        """Context manager to track the current field being processed."""
        if isinstance(field_info, str) or field_info is None:
            field_name = field_info
            # Direct lookup of dynamic dimension
            dyn_dim = self.dyn_dims.get(field_name)
            field_info = (field_name, {'dyn_dim': dyn_dim} if dyn_dim is not None else {})
        self._field_context.append(field_info)
        try:
            yield
        finally:
            self._field_context.pop()

    def _is_field_non_tensor(self, field_name: Optional[str]) -> bool:
        """Check if current field or any parent field in context is marked as non-tensor."""
        # Check current field
        if field_name is not None and field_name in self.non_tensor_fields:
            return True
        # Check parent context fields - first element of each tuple is field name
        return any(context[0] in self.non_tensor_fields for context in self._field_context if context[0] is not None)

    def _consolidate(self, column):
        """Special consolidation logic for analysis tensors."""
        # import torch #TODO: this formatter depends on torch for now so imported at top

        # Handle lists of tensors with same shape/dtype  # TODO: make consolidation configurable
        if isinstance(column, list) and column:
            if all(isinstance(x, torch.Tensor) and x.shape == column[0].shape and x.dtype == column[0].dtype
                   for x in column):
                return torch.stack(column)
        return column

    def _tensorize(self, value: Any, field_name: Optional[str] = None) -> Any:
        """Convert values to tensors unless they belong to non-tensor fields."""
        if isinstance(value, (str, bytes, type(None))):
            return value
        elif isinstance(value, (np.character, np.ndarray)) and np.issubdtype(value.dtype, np.character):
            return value.tolist()

        # Use helper method to check non-tensor fields
        if self._is_field_non_tensor(field_name):
            value = value.tolist() if isinstance(value, np.ndarray) else value
            return value

        is_per_latent = (field_name in self.per_latent_fields or
                       any(ctx[0] in self.per_latent_fields for ctx in self._field_context if ctx[0] is not None))

        if isinstance(value, dict):
            # Handle per_latent reconstruction at hook level
            if is_per_latent and set(value.keys()) == {'latents', 'per_latent'}:
                # Transform latents and per_latent lists into key-value pairs
                latents = value['latents']
                per_latent_values = value['per_latent']
                if latents is not None and per_latent_values is not None:
                    if len(latents) != len(per_latent_values):
                        raise ValueError(f"Mismatch in latents ({len(latents)}) and values ({len(per_latent_values)})")
                    return {
                        int(k): self._tensorize(v, field_name)
                        for k, v in zip(latents, per_latent_values)
                    }
                return value  # Return original if no valid data

            # Regular dict processing
            return {k: self._tensorize(v, k) for k, v in value.items()}

        # Handle potential tensor data
        if isinstance(value, (list, tuple)) and is_per_latent:
            # Process lists within per_latent fields directly rather than treating as regular lists
            return value
        # TODO: consider pruning this logic as part of making TorchFormatter customization features explicit functions
        elif isinstance(value, list):
            # Keep lists as lists for non-tensor fields
            if field_name in self.non_tensor_fields or any(f in self.non_tensor_fields for f in self._field_context):
                return value
            # Special handling for lists that should remain lists
            if any(isinstance(x, dict) for x in value):
                return [self._tensorize(x, field_name) for x in value]
            if any(isinstance(x, list) for x in value):
                return [self._tensorize(x, field_name) for x in value]

        default_dtype = {}
        if isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.integer):
            default_dtype = {"dtype": torch.int64}
            if value.dtype in [np.uint16, np.uint32]:
                value = value.astype(np.int64)
        elif isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.floating):
            default_dtype = {"dtype": torch.float32}

        try:
            tensor = torch.tensor(value, **{**default_dtype, **self.torch_tensor_kwargs})
            # Direct dict lookup for dynamic dimension
            dyn_dim = self.dyn_dims.get(field_name)
            if dyn_dim is not None and tensor.dim() > dyn_dim:
                # Reverse the dimension swap
                dims = list(range(tensor.dim()))
                dims[0], dims[dyn_dim] = dims[dyn_dim], dims[0]
                tensor = tensor.permute(*dims)
            return tensor
        except (ValueError, TypeError):
            # If tensor conversion fails, return as-is
            return value

    def _recursive_tensorize(self, data_struct: Any) -> Any:
        """Recursively convert data structures to tensors.

        Args:
            data_struct: The data structure to convert
        """
        # import torch #TODO: this formatter depends on torch for now so imported at top

        # Support for torch tensors and other array-like objects
        if hasattr(data_struct, "__array__") and not isinstance(data_struct, torch.Tensor):
            data_struct = data_struct.__array__()

        # Handle nested numpy arrays and special types
        if isinstance(data_struct, np.ndarray):
            if data_struct.dtype == object:
                return self._consolidate([self._recursive_tensorize(substruct) for substruct in data_struct])
        elif isinstance(data_struct, (list, tuple)):
            return self._consolidate([self._recursive_tensorize(substruct) for substruct in data_struct])
        elif isinstance(data_struct, dict):
            # For nested dicts, pass the current field as parent_field
            result = {}
            for k, v in data_struct.items():
                col_dict = self.dyn_dims.get(k, {})
                with self.field_context((k, col_dict)):
                    result[k] = self._recursive_tensorize(v)
            if any(ctx[0] in self.per_latent_fields for ctx in self._field_context if ctx[0] is not None):
                # For nested dicts in per_latent fields, handle the hook level reconstruction
                if all(isinstance(v, dict) and set(v.keys()) == {'latents', 'per_latent'} for v in result.values()):
                    return {
                        hook_name: self._tensorize(hook_data, self._field_context[-1][0])
                        for hook_name, hook_data in result.items()
                    }
            return result

        return self._tensorize(data_struct)

    def recursive_tensorize(self, data_struct: dict) -> Dict:
        """Apply tensorize recursively while preserving structure."""
        with self.field_context(None):
            return self._recursive_tensorize(data_struct)

    def format_row(self, pa_table: "pa.Table") -> Mapping:
        """Format a single row from Arrow table."""
        row = self.numpy_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        # Pass None as current_field since we're at the root level
        return self.recursive_tensorize(row)

    def format_column(self, pa_table: "pa.Table") -> Union[torch.Tensor, Sequence]:
        """Format a column from Arrow table."""
        column = self.numpy_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        col_name = pa_table.column_names[0]
        col_dict = self.dyn_dims.get(col_name, {})
        # Pass the column name and its format dict as current_field
        with self.field_context((col_name, col_dict)):
            column = self._recursive_tensorize(column)
        return self._consolidate(column)

    def format_batch(self, pa_table: "pa.Table") -> Mapping:
        """Format a batch from Arrow table."""
        batch = self.numpy_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        # Pass None as current_field since we're at the root level
        batch = self.recursive_tensorize(batch)
        for column_name in batch:
            batch[column_name] = self._consolidate(batch[column_name])
        return batch
