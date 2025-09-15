# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional
from functools import partial

import torch

from datasets import Array2D, Array3D, Value, Sequence as DatasetsSequence

from interpretune.analysis.core import (
    get_module_dims,
    schema_to_features,
    get_filtered_sae_hook_keys,
    _check_names_filter_available,
)


def _generate_cache_data(module, field, cfg, num_batches, dim_vars, is_grad_cache=False):
    """Generate cache data for activation cache field.

    Args:
        module: The module being analyzed
        field: Name of the field being generated
        cfg: Configuration for the field
        num_batches: Number of batches to generate
        dim_vars: Dictionary of module dimensions, including actual batch shapes
        is_grad_cache: Whether to include gradient entries with _grad suffix

    Returns:
        List of ActivationCache objects or None if generation not possible
    """
    from transformer_lens import ActivationCache

    # Check if we have what we need to generate cache data
    if (
        not hasattr(module, "sae_handles")
        or len(module.sae_handles) == 0
        or not _check_names_filter_available(module, field, cfg)
    ):
        if cfg.required:
            raise ValueError(f"Field '{field}' requires module.sae_handles, but it is missing or empty")
        return None  # Skip this field if not required

    batch_size = dim_vars["batch_size"]
    max_seq_len = dim_vars["max_seq_len"]

    # Use the actual sequence length from the batch if available
    seq_len = dim_vars.get("target_seq_len", max_seq_len)
    # Use minimum of max_seq_len and actual sequence length
    seq_len = min(seq_len, max_seq_len)

    # Generate cache data for each batch
    batch_data_list = []

    for _ in range(num_batches):
        cache_dict = {}
        # Populate the cache dict with tensors for each hook
        for handle in module.sae_handles:
            d_sae = handle.cfg.d_sae  # Get SAE dimension from the handle
            for hook_key in get_filtered_sae_hook_keys(handle, module.analysis_cfg.names_filter):
                # Create random tensor with shape [batch_size, seq_len, d_sae]
                tensor = torch.randn(batch_size, seq_len, d_sae)
                cache_dict[hook_key] = tensor

                # For grad_cache, also add entries with _grad suffix
                if is_grad_cache:
                    grad_tensor = torch.randn(batch_size, seq_len, d_sae)  # Different random values for grad
                    cache_dict[f"{hook_key}_grad"] = grad_tensor

        # Wrap the dictionary in an ActivationCache object
        batch_data = ActivationCache(cache_dict, module.model)
        batch_data_list.append(batch_data)

    return batch_data_list


def _generate_per_sae_hook_data(module, field, cfg, num_batches, dim_vars, predefined_indices=True):
    """Generate data for fields with per_sae_hook=True.

    Args:
        module: The module being analyzed
        field: Name of the field being generated
        cfg: Configuration for the field
        num_batches: Number of batches to generate
        dim_vars: Dictionary of module dimensions
        predefined_indices: Whether to use predefined indices (3,4,5) instead of random ones

    Returns:
        List of dictionaries mapping hook paths to appropriate values
    """
    # Check if we have what we need to generate hook data
    if (
        not hasattr(module, "sae_handles")
        or len(module.sae_handles) == 0
        or not _check_names_filter_available(module, field, cfg)
    ):
        if cfg.required:
            raise ValueError(f"Field '{field}' requires module.sae_handles, but it is missing or empty")
        return None  # Skip this field if not required

    # Not using batch_size, but it's available if needed by future code
    # batch_size = dim_vars['batch_size']
    batch_data_list = []

    # Generate data for each batch
    for _ in range(num_batches):
        hook_dict = {}

        # For each SAE handle, get the filtered hook keys
        for handle in module.sae_handles:
            d_sae = handle.cfg.d_sae  # Total number of latents

            for hook_key in get_filtered_sae_hook_keys(handle, module.analysis_cfg.names_filter):
                if cfg.non_tensor and cfg.sequence_type:
                    # Generate a sequence of integers (e.g., alive_latents)
                    # we hardcode to 3 for now to reduce overhead and randomness but might make configurable
                    # in the future
                    num_alive = 3
                    if predefined_indices:
                        latent_indices = [3, 4, 5]  # Use predefined indices [3, 4, 5]
                    else:
                        # Use random indices
                        latent_indices = torch.randperm(d_sae)[:num_alive].sort()[0].tolist()
                    hook_dict[hook_key] = latent_indices
                elif not cfg.non_tensor:
                    # Generate tensor data (e.g., correct_activations)
                    # Use the actual sequence length if available
                    seq_len = dim_vars.get("target_seq_len", dim_vars["max_seq_len"])
                    seq_len = min(seq_len, dim_vars["max_seq_len"])

                    # For tensor fields, create appropriate shape based on field requirements
                    if cfg.sequence_type:
                        # Create sequence of activations across all latents
                        tensor = torch.randn(d_sae)  # Shape: [d_sae]
                    else:
                        # Create 2D tensor with activations across sequence and latents
                        tensor = torch.randn(seq_len, d_sae)  # Shape: [seq_len, d_sae]

                    hook_dict[hook_key] = tensor.tolist()

        batch_data_list.append(hook_dict)

    return batch_data_list


def _generate_per_latent_data(module, field, cfg, num_batches, dim_vars, predefined_indices=True):
    """Generate data for fields with per_latent=True.

    Args:
        module: The module being analyzed
        field: Name of the field being generated
        cfg: Configuration for the field
        num_batches: Number of batches to generate
        dim_vars: Dictionary of module dimensions
        predefined_indices: Whether to use predefined indices (3,4,5) instead of random ones

    Returns:
        List of dictionaries with proper per_latent structure
    """
    # Check if we have what we need to generate latent data
    if (
        not hasattr(module, "sae_handles")
        or len(module.sae_handles) == 0
        or not _check_names_filter_available(module, field, cfg)
    ):
        if cfg.required:
            raise ValueError(f"Field '{field}' requires module.sae_handles, but it is missing or empty")
        return None  # Skip this field if not required

    # These are available but not currently used - keeping for future use
    # batch_size = dim_vars['batch_size']
    # max_answer_tokens = dim_vars['max_answer_tokens']
    num_classes = dim_vars["num_classes"]
    batch_data_list = []
    dtype = cfg.array_dtype or cfg.datasets_dtype

    # Generate data for each batch
    for _ in range(num_batches):
        hook_dict = {}

        # For each SAE handle, get the filtered hook keys
        for handle in module.sae_handles:
            d_sae = handle.cfg.d_sae  # Total number of latents

            for hook_key in get_filtered_sae_hook_keys(handle, module.analysis_cfg.names_filter):
                # Generate a number of latents to include
                num_latents = 3  # Hardcoded for now, can be made configurable in the future
                if predefined_indices:
                    latent_indices = [3, 4, 5]  # Use predefined indices [3, 4, 5]
                else:
                    # Random indices
                    latent_indices = torch.randperm(d_sae)[:num_latents].sort()[0].tolist()

                # Determine base feature shape based on ColCfg properties
                if cfg.array_shape:
                    shape = list(cfg.array_shape)
                    shape = [dim_vars.get(dim, dim) if isinstance(dim, str) else dim for dim in shape]

                    # Create per-latent data with appropriate shape
                    per_latent_data = []
                    for _ in range(num_latents):
                        if len(shape) == 2:
                            tensor = (
                                torch.randn(*shape)
                                if dtype.startswith("float")
                                else torch.randint(0, num_classes, shape)
                            )
                            per_latent_data.append(tensor.tolist())
                        elif len(shape) == 3:
                            tensor = (
                                torch.randn(*shape)
                                if dtype.startswith("float")
                                else torch.randint(0, num_classes, shape)
                            )
                            per_latent_data.append(tensor.tolist())
                elif cfg.sequence_type:
                    # Simple sequence per latent
                    per_latent_data = []
                    for _ in range(num_latents):
                        seq_len = torch.randint(1, 10, (1,)).item()
                        if dtype.startswith("float"):
                            tensor = torch.randn(seq_len)
                        else:
                            tensor = torch.randint(0, num_classes, (seq_len,))
                        per_latent_data.append(tensor.tolist())
                else:
                    # Single value per latent
                    per_latent_data = []
                    for _ in range(num_latents):
                        if dtype.startswith("float"):
                            val = float(torch.randn(1).item())
                        else:
                            val = int(torch.randint(0, num_classes, (1,)).item())
                        per_latent_data.append(val)

                # Create the per_latent structure
                hook_dict[hook_key] = {"latents": latent_indices, "per_latent": per_latent_data}

        batch_data_list.append(hook_dict)

    return batch_data_list


# Define mapping from field names to generator functions
INTERMEDIATE_FIELD_GENERATORS = {
    "cache": _generate_cache_data,
    "grad_cache": partial(_generate_cache_data, is_grad_cache=True),
    # More field generators can be added here as needed
}


def should_process_field(field, cfg, required_only, override_req_cols=None):
    """Determine if a field should be processed based on required status and overrides.

    Args:
        field: Name of the field
        cfg: Configuration object for the field
        required_only: If True, only required fields will be processed unless overridden
        override_req_cols: Tuple of field names to override the required_only behavior

    Returns:
        Boolean indicating whether the field should be processed
    """
    # If override_req_cols is None, initialize as empty tuple
    if override_req_cols is None:
        override_req_cols = tuple()

    field_is_required = getattr(cfg, "required", False)

    # Field should be processed if:
    # - required_only=False (process all fields) OR
    # - field is required OR
    # - field is in override_req_cols
    return not required_only or field_is_required or field in override_req_cols


def _infer_high_value_from_cfg(cfg, dim_vars):
    """Infer appropriate high value for random integer generation based on ColCfg.

    Args:
        cfg: ColCfg object for the field
        dim_vars: Dictionary of dimension variables

    Returns:
        Appropriate high value for torch.randint
    """
    # Default fallback
    default_high = 2

    # Check if we have array_shape that might contain dimension variables
    if cfg.array_shape:
        for dim in cfg.array_shape:
            if isinstance(dim, str) and dim in dim_vars:
                # If a dimension is a vocab_size or num_classes, use that as high value
                if dim in ["vocab_size", "num_classes"]:
                    return dim_vars[dim]
    return dim_vars.get("num_classes", default_high)


def gen_or_validate_input_data(
    module,
    input_schema,
    batch_shapes=None,
    input_data=None,
    num_batches=1,
    required_only=True,
    override_req_cols: Optional[tuple] = None,
    predefined_indices=True,
):
    """Generate or validate input data based on an input schema.

    Args:
        module: The module to use for schema conversion
        input_schema: Schema defining input data requirements
        batch_shapes: Optional dictionary containing shape information from test batches
        input_data: Existing input data to validate/extend (optional)
        num_batches: Number of batches to generate (default: 1)
        required_only: If True, only generate data for required fields (default: True)
        override_req_cols: Tuple of field names to override the required_only behavior.
                          When required_only=True, these fields will be generated even if not required.
                          When required_only=False, these fields represent the only non-required fields to process.
        predefined_indices: Whether to use predefined indices for per_latent and per_sae_hook data
                           to ensure alignment between fields that need to reference same indices

    Returns:
        Tuple of (regular_data, intermediate_data) where:
            - regular_data: Dictionary of validated/generated regular data
            - intermediate_data: Dictionary of intermediate_only data that should not be serialized
    """
    # Get dims and map vars
    batch_size, max_answer_tokens, num_classes, vocab_size, max_seq_len = get_module_dims(module)
    dim_vars = {
        "batch_size": batch_size,
        "max_answer_tokens": max_answer_tokens,
        "num_classes": num_classes,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
    }

    # Update with actual batch shapes if available
    if batch_shapes:
        dim_vars.update(batch_shapes)

    # .feature_dict exposes the raw mapping
    input_features = schema_to_features(module=module, schema=input_schema)

    # if input_data present, make a copy
    input_data = input_data.copy() if input_data else {}
    intermediate_data = {}  # Separate dict for intermediate_only values

    # Process all fields from input_schema
    for field, cfg in input_schema.items():
        if getattr(cfg, "connected_obj", None) != "analysis_store":
            continue

        # Check if this field should be processed
        if not should_process_field(field, cfg, required_only, override_req_cols):
            continue

        # Check if this is an intermediate_only field with a registered generator
        if getattr(cfg, "intermediate_only", False) and field in INTERMEDIATE_FIELD_GENERATORS:
            if field in input_data:
                # Just validate batch count if data already exists
                if len(input_data[field]) != num_batches:
                    raise ValueError(
                        f"Input field '{field}' has {len(input_data[field])} batches but expected {num_batches}"
                    )
                # Move to intermediate_data
                intermediate_data[field] = input_data[field]
                del input_data[field]
            else:
                # Generate data using the appropriate generator function
                generator_fn = INTERMEDIATE_FIELD_GENERATORS[field]
                batch_data_list = generator_fn(module, field, cfg, num_batches, dim_vars)

                if batch_data_list:
                    intermediate_data[field] = batch_data_list

            # Skip regular field processing for this field
            continue

        # Handle per_sae_hook fields
        elif getattr(cfg, "per_sae_hook", False):
            if field in input_data:
                # Just validate batch count if data already exists
                if len(input_data[field]) != num_batches:
                    raise ValueError(
                        f"Input field '{field}' has {len(input_data[field])} batches but expected {num_batches}"
                    )
            else:
                # Generate data for per_sae_hook fields
                batch_data_list = _generate_per_sae_hook_data(
                    module, field, cfg, num_batches, dim_vars, predefined_indices
                )

                if batch_data_list:
                    input_data[field] = batch_data_list

            # Continue to next field since we've handled this one
            continue

        # Handle per_latent fields
        elif getattr(cfg, "per_latent", False):
            if field in input_data:
                # Just validate batch count if data already exists
                if len(input_data[field]) != num_batches:
                    raise ValueError(
                        f"Input field '{field}' has {len(input_data[field])} batches but expected {num_batches}"
                    )
            else:
                # Generate data for per_latent fields
                batch_data_list = _generate_per_latent_data(
                    module, field, cfg, num_batches, dim_vars, predefined_indices
                )

                if batch_data_list:
                    input_data[field] = batch_data_list

            # Continue to next field since we've handled this one
            continue

        # For regular fields, get the feature from input_features if it exists
        if field not in input_features:
            continue

        feature = input_features[field]

        # Determine expected shape
        if isinstance(feature, (Array2D, Array3D)):
            exp_shape = feature.shape
        elif isinstance(feature, DatasetsSequence):
            inner = feature.feature
            if isinstance(inner, (Array2D, Array3D)):
                exp_shape = (batch_size,) + inner.shape
            else:
                exp_shape = (batch_size,)
        elif isinstance(feature, Value):
            exp_shape = ()
        else:
            # unknown feature type, skip
            continue

        # Handle dyn_dim properly by setting the dynamic dimension to its ceiling value and swapping dimensions
        if cfg.array_shape is not None and cfg.dyn_dim is not None:
            # Start with the base shape
            shape = list(cfg.array_shape)
            # Replace dimension variables with actual values
            shape = [dim_vars.get(dim, dim) if isinstance(dim, str) else dim for dim in shape]

            # Get the dynamic dimension ceiling value
            dyn_dim_value = 10  # Default value
            if cfg.dyn_dim_ceil is not None and cfg.dyn_dim_ceil in dim_vars:
                dyn_dim_value = dim_vars[cfg.dyn_dim_ceil]

            # Find the null position (where the dynamic dimension will be inserted)
            null_pos = None
            for i, dim in enumerate(shape):
                if dim is None:
                    null_pos = i
                    shape[i] = dyn_dim_value  # Replace null with the ceiling value
                    break

            # If null_pos is found and different from dyn_dim, swap them
            if null_pos is not None and null_pos != cfg.dyn_dim:
                # Swap the dimensions to mimic the pre-serialization shape
                # (the null position is always at index 0 in this context)
                shape[null_pos], shape[cfg.dyn_dim] = shape[cfg.dyn_dim], shape[null_pos]

            # This is the expected shape for the original tensor (not serialized version)
            exp_shape = tuple(shape)

        # If we already have data for this field
        if field in input_data:
            # Make sure we have the right number of batches
            if len(input_data[field]) != num_batches:
                raise ValueError(
                    f"Input field '{field}' has {len(input_data[field])} batches but expected {num_batches}"
                )

            # Validate each batch
            for batch_idx, batch_data in enumerate(input_data[field]):
                arr = torch.tensor(batch_data)
                if tuple(arr.shape) != exp_shape:
                    raise ValueError(
                        f"Input field '{field}' batch {batch_idx} has shape {tuple(arr.shape)} but expected {exp_shape}"
                    )
        else:
            # Generate default tensor/list of correct shape for each batch
            batch_data_list = []
            for _ in range(num_batches):
                if isinstance(feature, (Array2D, Array3D)):
                    # Determine appropriate high value based on ColCfg and feature shape
                    if feature.dtype.startswith("float"):
                        tensor = torch.randn(*exp_shape)
                    else:
                        # Infer appropriate high value based on context
                        high = _infer_high_value_from_cfg(cfg, dim_vars)
                        tensor = torch.randint(0, high, exp_shape)
                    batch_data_list.append(tensor.tolist())
                elif isinstance(feature, DatasetsSequence):
                    inner = feature.feature
                    if isinstance(inner, (Array2D, Array3D)):
                        shape = exp_shape
                        if inner.dtype.startswith("float"):
                            tensor = torch.randn(*shape)
                        else:
                            high = _infer_high_value_from_cfg(cfg, dim_vars)
                            tensor = torch.randint(0, high, shape)
                    else:
                        # Value sequence
                        if inner.dtype.startswith("float"):
                            tensor = torch.randn(batch_size)
                        else:
                            high = _infer_high_value_from_cfg(cfg, dim_vars)
                            tensor = torch.randint(0, high, (batch_size,))
                    batch_data_list.append(tensor.tolist())
                elif isinstance(feature, Value):
                    # scalar default
                    tensor = torch.tensor(0, dtype=getattr(torch, feature.dtype))
                    batch_data_list.append(tensor.tolist())
                else:
                    continue

            input_data[field] = batch_data_list

    return input_data, intermediate_data
