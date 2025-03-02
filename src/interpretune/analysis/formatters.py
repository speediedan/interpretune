from __future__ import annotations
from typing import Any, Sequence, Union, Optional
import torch
import numpy as np
from datasets.formatting import TorchFormatter
import pyarrow as pa
from contextlib import contextmanager

from interpretune.analysis import ColCfg

class OpSchemaExt:
    """Provides operation schema extensions for tensor processing."""

    def __init__(self, col_cfg: dict = None, **kwargs):
        super().__init__(**kwargs)  # Allow proper multiple inheritance
        col_cfg = col_cfg or {}
        col_cfg = {k: ColCfg.from_dict(v) for k, v in col_cfg.items()}
        self.dyn_dims = {k: cfg.dyn_dim for k, cfg in col_cfg.items() if cfg.dyn_dim is not None}
        self.non_tensor_fields = {k for k, cfg in col_cfg.items() if cfg.non_tensor}
        self.per_latent_fields = {k for k, cfg in col_cfg.items() if cfg.per_latent}
        self._field_context = []

    @contextmanager
    def field_context(self, field_info: Union[tuple[Optional[str], dict], Optional[str]]):
        """Context manager to track the current field being processed."""
        if isinstance(field_info, str) or field_info is None:
            field_name = field_info
            dyn_dim = self.dyn_dims.get(field_name)
            field_info = (field_name, {'dyn_dim': dyn_dim} if dyn_dim is not None else {})
        self._field_context.append(field_info)
        try:
            yield
        finally:
            self._field_context.pop()

    def is_field_non_tensor(self, field_name: Optional[str]) -> bool:
        """Check if current field or any parent field in context is marked as non-tensor."""
        if field_name is not None and field_name in self.non_tensor_fields:
            return True
        return any(context[0] in self.non_tensor_fields for context in self._field_context if context[0] is not None)

    def is_field_per_latent(self, field_name: Optional[str]) -> bool:
        """Check if current field or any parent field in context is marked as per_latent."""
        if field_name is not None and field_name in self.per_latent_fields:
            return True
        return any(context[0] in self.per_latent_fields for context in self._field_context if context[0] is not None)

    def handle_per_latent_dict(self, value: dict, tensorize_fn) -> dict:
        """Transform per_latent dictionary structure into key-value pairs."""
        if set(value.keys()) == {'latents', 'per_latent'}:
            latents = value['latents']
            per_latent_values = value['per_latent']
            if latents is not None and per_latent_values is not None:
                if len(latents) != len(per_latent_values):
                    raise ValueError(f"Mismatch in latents ({len(latents)}) and values ({len(per_latent_values)})")
                return {int(k): tensorize_fn(v) for k, v in zip(latents, per_latent_values)}
        return value

    def apply_dynamic_dimension(self, tensor: torch.Tensor, field_name: Optional[str]) -> torch.Tensor:
        """Apply dynamic dimension transformation if configured."""
        dyn_dim = self.dyn_dims.get(field_name)
        if dyn_dim is not None and tensor.dim() > dyn_dim:
            dims = list(range(tensor.dim()))
            dims[0], dims[dyn_dim] = dims[dyn_dim], dims[0]
            return tensor.permute(*dims)
        return tensor


class ITAnalysisFormatter(OpSchemaExt, TorchFormatter):
    """Formatter for Interpretune analysis operations that extends TorchFormatter with operation schema
    extensions."""

    def __init__(self, features=None, **format_kwargs):
        col_cfg = format_kwargs.pop('col_cfg', {})
        super().__init__(col_cfg=col_cfg, features=features, **format_kwargs)

    def _tensorize(self, value: Any, field_name: Optional[str] = None) -> Any:
        """Enhanced tensorization with support for non-tensor fields, per-latent transformations and non-zero
        dynamic dimensions."""
        if isinstance(value, (str, bytes, type(None))):
            return value
        elif isinstance(value, (np.character, np.ndarray)) and np.issubdtype(value.dtype, np.character):
            return value.tolist()

        if self.is_field_non_tensor(field_name):
            return value.tolist() if isinstance(value, np.ndarray) else value

        if isinstance(value, dict):
            if self.is_field_per_latent(field_name):
                return self.handle_per_latent_dict(value, lambda v: self._tensorize(v, field_name))
            return {k: self._tensorize(v, k) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            if self.is_field_per_latent(field_name):
                return value
            if self.is_field_non_tensor(field_name):
                return value
            if any(isinstance(x, (dict, list)) for x in value):
                return [self._tensorize(x, field_name) for x in value]

        tensor = TorchFormatter._tensorize(self, value)
        if isinstance(tensor, torch.Tensor):
            return self.apply_dynamic_dimension(tensor, field_name)
        return tensor

    def _recursive_tensorize(self, data_struct: Any) -> Any:
        if hasattr(data_struct, "__array__") and not isinstance(data_struct, torch.Tensor):
            data_struct = data_struct.__array__()

        if isinstance(data_struct, np.ndarray):
            if data_struct.dtype == object:
                return self._consolidate([self._recursive_tensorize(substruct) for substruct in data_struct])
        elif isinstance(data_struct, (list, tuple)):
            return self._consolidate([self._recursive_tensorize(substruct) for substruct in data_struct])
        elif isinstance(data_struct, dict):
            result = {}
            for k, v in data_struct.items():
                col_dict = self.dyn_dims.get(k, {})
                with self.field_context((k, col_dict)):
                    result[k] = self._recursive_tensorize(v)

            current_field = self._field_context[-1][0] if self._field_context else None
            if self.is_field_per_latent(current_field):
                if all(isinstance(v, dict) and set(v.keys()) == {'latents', 'per_latent'} for v in result.values()):
                    return {
                        hook_name: self._tensorize(hook_data, current_field)
                        for hook_name, hook_data in result.items()
                    }
            return result

        return self._tensorize(data_struct)

    def format_column(self, pa_table: "pa.Table") -> Union[torch.Tensor, Sequence]:
        """Format a column with enhanced tensorization."""
        column = self.numpy_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        col_name = pa_table.column_names[0]
        col_dict = self.dyn_dims.get(col_name, {})
        with self.field_context((col_name, col_dict)):
            column = self._recursive_tensorize(column)
        return self._consolidate(column)

    # we inherit `format_row`` and `format_batch`` from `TorchFormatter`
