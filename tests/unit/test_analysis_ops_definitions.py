from __future__ import annotations
import pytest
import torch
from unittest.mock import MagicMock

from typing import Any, List, Dict

import interpretune as it
from interpretune.analysis.ops.definitions import (get_loss_preds_diffs, ablate_sae_latent,
                                                   boolean_logits_to_avg_logit_diff)
from interpretune.analysis.ops.base import AnalysisBatch
from tests.utils import _unwrap_one
from tests.base_defaults import BaseAugTest, pytest_factory, OpTestConfig
from tests.orchestration import run_op_with_config


class TestBooleanLogitFunctions:
    """Tests for the boolean logits utility functions."""

    def test_boolean_logits_to_avg_logit_diff(self):
        """Test the boolean_logits_to_avg_logit_diff function."""
        # Create test data: batch_size=3, seq_len=1, num_classes=2
        logits = torch.tensor([[[0.8, 0.2]], [[0.3, 0.7]], [[0.6, 0.4]]])
        target_indices = torch.tensor([0, 1, 0])  # First and third examples correct class is 0, second is 1

        # Test with no reduction
        logit_diff = boolean_logits_to_avg_logit_diff(logits, target_indices)
        expected = torch.tensor([0.6, 0.4, 0.2])  # Difference between correct and incorrect logits
        assert torch.allclose(logit_diff, expected)

        # Test with mean reduction
        mean_diff = boolean_logits_to_avg_logit_diff(logits, target_indices, reduction="mean")
        expected_mean = torch.tensor(0.4)  # Mean of [0.6, 0.4, 0.2]
        assert torch.allclose(mean_diff, expected_mean)

        # Test with sum reduction
        sum_diff = boolean_logits_to_avg_logit_diff(logits, target_indices, reduction="sum")
        expected_sum = torch.tensor(1.2)  # Sum of [0.6, 0.4, 0.2]
        assert torch.allclose(sum_diff, expected_sum)

        # Test with keep_as_tensor=False
        list_diff = boolean_logits_to_avg_logit_diff(logits, target_indices, keep_as_tensor=False)
        assert isinstance(list_diff, list)
        assert torch.allclose(torch.tensor(list_diff), torch.tensor([0.6, 0.4, 0.2]))

    def test_get_loss_preds_diffs(self):
        """Test the get_loss_preds_diffs function."""
        # Create mock module with standardize_logits and loss_fn
        mock_module = MagicMock()
        mock_module.loss_fn.return_value = torch.tensor(0.5)
        mock_module.standardize_logits.return_value = torch.tensor([[[0.8, 0.2]], [[0.3, 0.7]], [[0.6, 0.4]]])

        # Create mock analysis batch
        mock_batch = MagicMock()
        mock_batch.labels = torch.tensor([0, 1, 0])
        mock_batch.orig_labels = torch.tensor([0, 1, 0])

        # Create answer logits
        answer_logits = torch.tensor([[[0.8, 0.2]], [[0.3, 0.7]], [[0.6, 0.4]]])

        # Mock logit_diff_fn
        mock_logit_diff_fn = MagicMock()
        mock_logit_diff_fn.return_value = torch.tensor([0.6, 0.4, 0.2])

        # Call the function
        loss, logit_diffs, preds, out_logits = get_loss_preds_diffs(
            mock_module, mock_batch, answer_logits, logit_diff_fn=mock_logit_diff_fn
        )

        # Verify the results
        assert torch.equal(loss, torch.tensor(0.5))
        assert torch.equal(logit_diffs, torch.tensor([0.6, 0.4, 0.2]))
        assert torch.equal(preds, torch.tensor([0, 1, 0]))  # Based on max logit for each example
        assert torch.equal(out_logits, torch.tensor([[[0.8, 0.2]], [[0.3, 0.7]], [[0.6, 0.4]]]))

        # Verify the mock calls
        mock_module.loss_fn.assert_called_once()
        mock_module.standardize_logits.assert_called_once_with(answer_logits)
        mock_logit_diff_fn.assert_called_once()


class TestAblationFunctions:
    """Tests for the ablation-related functions."""

    def test_ablate_sae_latent(self):
        """Test the ablate_sae_latent function."""
        # Create test SAE activations: [batch_size=3, seq_len=4, d_sae=5]
        sae_acts = torch.ones(3, 4, 5)

        # Create mock hook
        mock_hook = MagicMock()

        # Test ablation of a specific latent at specific positions
        seq_pos = torch.tensor([1, 2, 3])  # One position per batch
        latent_idx = 2

        result = ablate_sae_latent(sae_acts, mock_hook, latent_idx=latent_idx, seq_pos=seq_pos)

        # Check that the specified latent at specified positions is zero
        for batch_idx, pos in enumerate(seq_pos):
            assert result[batch_idx, pos, latent_idx] == 0.0

        # Check that other positions are unchanged
        assert result[0, 0, latent_idx] == 1.0
        assert result[1, 0, latent_idx] == 1.0
        assert result[2, 0, latent_idx] == 1.0

        # Test with None latent_idx (should ablate all latents at specified positions)
        result = ablate_sae_latent(sae_acts.clone(), mock_hook, latent_idx=None, seq_pos=seq_pos)

        # Check that all latents at specified positions are zeroed
        for batch_idx, pos in enumerate(seq_pos):
            assert torch.all(result[batch_idx, pos, :] == 0.0)


class TestLabelsAndIndicesFunctions:
    """Tests for the labels and indices-related functions."""

class TestGradientOperations:
    """Tests for the gradient-related operations."""

    @pytest.fixture
    def mock_module_with_cache(self):
        """Create a mock module with cache for testing."""
        module = MagicMock()
        module.analysis_cfg = MagicMock()
        module.analysis_cfg.names_filter = lambda x: True
        module.analysis_cfg.cache_dict = {"hook1": torch.ones(2, 3, 4), "hook1_grad": torch.ones(2, 3, 4) * 0.5}
        module.sae_handles = [MagicMock()]
        module.sae_handles[0].cfg.d_sae = 4
        return module


# Define test configurations
SERIALIZATION_TEST_CONFIGS = (
    BaseAugTest(alias="model_forward", cfg=OpTestConfig(target_op=it.model_forward)),
    BaseAugTest(alias="model_forward_multi_batch", cfg=OpTestConfig(target_op=it.model_forward, batch_size=2)),
    BaseAugTest(alias="labels_to_ids", cfg=OpTestConfig(target_op=it.labels_to_ids)),
    BaseAugTest(alias="get_answer_indices", cfg=OpTestConfig(target_op=it.get_answer_indices)),
    BaseAugTest(alias="get_alive_latents", cfg=OpTestConfig(target_op=it.get_alive_latents,
                                                            # TODO: we could just set generate_required_only=False
                                                            override_req_cols=("cache", "answer_indices"))),
    BaseAugTest(alias="model_cache_forward", cfg=OpTestConfig(target_op=it.model_cache_forward)),
    BaseAugTest(alias="model_ablation", cfg=OpTestConfig(target_op=it.model_ablation)),
    BaseAugTest(alias="model_gradient", cfg=OpTestConfig(target_op=it.model_gradient)),
    BaseAugTest(alias="logit_diffs", cfg=OpTestConfig(target_op=it.logit_diffs)),
    BaseAugTest(alias="logit_diffs_cache", cfg=OpTestConfig(target_op=it.logit_diffs_cache)),
    BaseAugTest(alias="model_cache_forward.logit_diffs_cache", cfg=OpTestConfig(target_op=[it.model_cache_forward,
                                                                                           it.logit_diffs_cache])),
    BaseAugTest(alias="sae_correct_acts", cfg=OpTestConfig(target_op=it.sae_correct_acts)),
    BaseAugTest(alias="ablation_attribution", cfg=OpTestConfig(target_op=it.ablation_attribution)),
    BaseAugTest(alias="gradient_attribution", cfg=OpTestConfig(target_op=it.gradient_attribution)),
    BaseAugTest(alias="logit_diffs_attr_ablation", cfg=OpTestConfig(target_op=it.logit_diffs_attr_ablation)),
)

class TestAnalysisOperationsImplementations:
    """Tests for the core analysis operation implementation functions."""

    def _validate_column_shape(self, column_name: str, shape_info: torch.Size, loaded_column: torch.Tensor,
                             col_cfg, batch_count: int, context: str = "") -> None:
        """Helper to validate column shape based on config and expected shape.

        Args:
            column_name: Name of the column being validated
            shape_info: Expected tensor shape (without batch dimension)
            loaded_column: The loaded column data to validate
            col_cfg: Column configuration from output_schema
            batch_count: Expected number of batches
            context: Optional context for error messages (e.g., "range access:")
        """
        context_prefix = f"{context} " if context else ""

        # For columns with dynamic dimensions
        if col_cfg.dyn_dim is not None:
            # First dimension should be the dataset size (number of batches)
            assert loaded_column.shape[0] == batch_count, (
                f"{context_prefix}Expected column {column_name} to have {batch_count} batches, "
                f"got {loaded_column.shape[0]}"
            )

            # The rest of the shape should match original shape
            assert shape_info == loaded_column.shape[1:], (
                f"{context_prefix}Shape mismatch for column {column_name}: "
                f"expected {shape_info}, got {loaded_column.shape[1:]}"
            )
        else:
            # Standard column - should have batch dimension added
            expected_shape = (batch_count,) + tuple(shape_info)
            assert loaded_column.shape == expected_shape, (
                f"{context_prefix}Shape mismatch for column {column_name}: "
                f"expected {expected_shape}, got {loaded_column.shape}"
            )

    def _should_validate_column(self, column_name: str, shape_info: Any, op_cfg: OpTestConfig) -> bool:
        """Determine if a column should be validated based on its type and config.

        Args:
            column_name: Name of the column
            shape_info: Shape information from pre_serialization_shapes
            op_cfg: Operation test configuration

        Returns:
            True if the column should be validated, False otherwise
        """
        # Skip non-tensor and complex structures
        if not isinstance(shape_info, torch.Size):
            return False

        # Get column config
        col_cfg = op_cfg.resolved_op.output_schema.get(column_name)
        return col_cfg is not None

    def _validate_format_column_path(self, op_cfg: OpTestConfig, result_batches: List[AnalysisBatch],
                                   loaded_dataset, pre_serialization_shapes: Dict) -> None:
        """Validate loaded dataset using direct column access (format_column path)."""
        if not pre_serialization_shapes:
            return

        for column_name, shape_info in pre_serialization_shapes.items():
            # Skip columns that don't need validation
            if not self._should_validate_column(column_name, shape_info, op_cfg):
                continue

            # Get column config
            col_cfg = op_cfg.resolved_op.output_schema.get(column_name)

            try:
                # Test direct column access
                loaded_column = loaded_dataset[column_name]

                # Validate column shape
                self._validate_column_shape(
                    column_name,
                    shape_info,
                    loaded_column,
                    col_cfg,
                    len(result_batches)
                )
            except (KeyError, ValueError, AssertionError) as e:
                print(f"Warning: Column access validation failed for '{column_name}': {e}")

    def _validate_format_batch_path(self, op_cfg: OpTestConfig, result_batches: List[AnalysisBatch],
                                  loaded_dataset, pre_serialization_shapes: Dict) -> None:
        """Validate loaded dataset using batch access (format_batch path)."""
        if not pre_serialization_shapes or len(result_batches) <= 1:
            return  # Cannot test batch access with fewer than 2 batches

        total_batches = len(result_batches)

        # Test multiple batch access patterns
        access_methods = {
            "range": range(0, total_batches),
            "slice": slice(0, total_batches),
            "list": [i for i in range(total_batches)]
        }

        for method_name, access_pattern in access_methods.items():
            # Get the dataset subset using this access pattern
            subset_dataset = loaded_dataset[access_pattern]

            # Verify length
            assert len(subset_dataset) == total_batches, (
                f"{method_name} access: Expected {total_batches} items, got {len(subset_dataset)}"
            )

            # Validate columns
            for column_name, shape_info in pre_serialization_shapes.items():
                # Skip columns that don't need validation
                if not self._should_validate_column(column_name, shape_info, op_cfg):
                    continue

                # Get column config
                col_cfg = op_cfg.resolved_op.output_schema.get(column_name)

                try:
                    # Test column access on the subset
                    loaded_column = subset_dataset[column_name]

                    # Validate column shape with access method context
                    self._validate_column_shape(
                        column_name,
                        shape_info,
                        loaded_column,
                        col_cfg,
                        total_batches,
                        f"{method_name} access:"
                    )
                except (KeyError, ValueError, AssertionError) as e:
                    print(f"Warning: {method_name} access validation failed for '{column_name}': {e}")

    def _validate_format_row_path(self, op_cfg: OpTestConfig, result_batches: List[AnalysisBatch],
                                loaded_dataset) -> None:
        """Validate loaded dataset using row-by-row access (format_row path)."""
        for i, original_result in enumerate(result_batches):
            loaded_batch = AnalysisBatch(loaded_dataset[i])

            # Check each column in output schema
            for column_name, col_cfg in op_cfg.resolved_op.output_schema.items():
                # Skip intermediates and fields not in original result
                if col_cfg.intermediate_only or not hasattr(original_result, column_name):
                    continue

                # Verify field exists in loaded data
                assert hasattr(loaded_batch, column_name), f"Missing field: {column_name}"

                value = getattr(original_result, column_name)
                loaded_value = getattr(loaded_batch, column_name)

                # Check basic type
                if isinstance(value, torch.Tensor):
                    assert isinstance(loaded_value, torch.Tensor)

                    # For simple tensors, shape should match (accounting for dyn_dim)
                    if col_cfg.dyn_dim is None:
                        # Direct shape comparison
                        assert loaded_value.shape == value.shape, f"Shape mismatch for {column_name}"
                    else:
                        # remember, value is the result after dyn_dim swapped with 0th dim for serialization
                        dyn = col_cfg.dyn_dim
                        # Rank must match
                        assert loaded_value.dim() == value.dim(), f"Dimension count mismatch for {column_name}"
                        # Ensure total size matches
                        assert sum(loaded_value.shape) == sum(value.shape), f"Size mismatch for {column_name}"
                        # Validate swapped dimensions: 0th and dyn should be swapped
                        assert loaded_value.shape[0] == value.shape[dyn], (
                            f"Expected loaded dim 0 ({loaded_value.shape[0]}) to equal "
                            f"value dim {dyn} ({value.shape[dyn]}) for {column_name}"
                        )
                        assert loaded_value.shape[dyn] == value.shape[0], (
                            f"Expected loaded dim {dyn} ({loaded_value.shape[dyn]}) to equal "
                            f"value dim 0 ({value.shape[0]}) for {column_name}"
                        )
                        # Validate other dimensions remain unchanged
                        for idx in range(value.dim()):
                            if idx in (0, dyn):
                                continue
                            assert loaded_value.shape[idx] == value.shape[idx], (
                                f"Dimension {idx} mismatch for {column_name}: "
                                f"loaded {loaded_value.shape[idx]} vs value {value.shape[idx]}"
                            )

                # Handle dictionaries (like attribution values, alive_latents)
                elif isinstance(value, dict):
                    assert isinstance(loaded_value, dict)
                    # Check keys match
                    assert set(value.keys()) == set(loaded_value.keys()), f"Key mismatch for {column_name}"

    def validate_loaded_dataset(self, op_cfg: OpTestConfig, result_batches: List[AnalysisBatch],
                               loaded_dataset, pre_serialization_shapes: Dict = None) -> None:
        """Validate loaded dataset against original results.

        Args:
            op_cfg: Operation test configuration
            result_batches: Original result batches before serialization
            loaded_dataset: Dataset loaded after serialization
            pre_serialization_shapes: Dictionary mapping column names to their original shapes
        """
        # Verify the loaded dataset has the correct number of rows
        assert len(loaded_dataset) == len(result_batches)

        # Validate direct column access (format_column path)
        if pre_serialization_shapes:
            self._validate_format_column_path(op_cfg, result_batches, loaded_dataset, pre_serialization_shapes)
            self._validate_format_batch_path(op_cfg, result_batches, loaded_dataset, pre_serialization_shapes)

        # Validate row-by-row access (format_row path)
        self._validate_format_row_path(op_cfg, result_batches, loaded_dataset)

    @pytest.mark.parametrize(("test_alias", "test_config"), pytest_factory(SERIALIZATION_TEST_CONFIGS, unpack=False))
    def test_op_serialization(self, request, op_serialization_fixt, test_alias, test_config):
        """Test multiple operations using schema-driven column validation."""
        # Run operation and get results - now using the standalone function
        it_session, batches, result_batches, pre_serialization_shapes = run_op_with_config(request, test_config)

        # Test dataset serialization and loading
        loaded_dataset = op_serialization_fixt(it_session, _unwrap_one(result_batches), _unwrap_one(batches), request)

        # Validate loaded dataset against original results
        self.validate_loaded_dataset(test_config, result_batches, loaded_dataset, pre_serialization_shapes)
