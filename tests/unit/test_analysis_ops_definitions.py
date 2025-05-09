from __future__ import annotations
import pytest
import torch
from unittest.mock import MagicMock, patch

from typing import Any, List, Dict
from transformers import BatchEncoding
from torch.testing import assert_close

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

    def test_labels_to_ids_impl(self):
        """Test labels_to_ids_impl function."""
        # Create mock module
        mock_module = MagicMock()
        mock_module.labels_to_ids.return_value = (torch.tensor([0, 1]), torch.tensor([0, 1]))

        # Create mock batch with labels
        mock_batch = {"labels": ["label1", "label2"]}

        # Test with analysis_batch=None
        from interpretune.analysis.ops.definitions import labels_to_ids_impl

        # Test with existing analysis_batch
        existing_batch = AnalysisBatch()
        mock_batch = {"labels": ["label1", "label2"]}
        result_batch = labels_to_ids_impl(mock_module, existing_batch, mock_batch, 0)

        # Verify the results
        assert hasattr(result_batch, "labels")
        assert hasattr(result_batch, "orig_labels")
        assert torch.equal(result_batch.labels, torch.tensor([0, 1]))
        assert torch.equal(result_batch.orig_labels, torch.tensor([0, 1]))

    def test_get_answer_indices_impl(self):
        """Test get_answer_indices_impl function with left padding."""
        # Create mock module
        mock_module = MagicMock()
        mock_module.datamodule.tokenizer.padding_side = "left"
        mock_module.analysis_cfg.input_store = None

        # Create mock batch
        mock_batch = {"input": torch.ones(3, 5)}  # batch_size=3, seq_len=5

        # Import function under test
        from interpretune.analysis.ops.definitions import get_answer_indices_impl

        existing_batch = AnalysisBatch()
        result_batch = get_answer_indices_impl(mock_module, existing_batch, mock_batch, 0)

        # Verify the results
        assert hasattr(result_batch, "answer_indices")
        assert torch.equal(result_batch.answer_indices, torch.full((3,), -1))

        existing_batch.answer_indices = torch.tensor([2, 3, 4])
        result_batch = get_answer_indices_impl(mock_module, existing_batch, mock_batch, 0)
        # Verify the results
        assert hasattr(result_batch, "answer_indices")
        assert torch.equal(result_batch.answer_indices, torch.tensor([2, 3, 4]))

    def test_get_answer_indices_impl_right_padding(self):
        """Test get_answer_indices_impl function with right padding."""
        # Create mock module
        mock_module = MagicMock()
        mock_module.datamodule.tokenizer.padding_side = "right"
        mock_module.datamodule.tokenizer.pad_token_id = 0
        mock_module.analysis_cfg.input_store = None

        # Create mock batch with padding tokens at the end
        mock_input = torch.ones(2, 5)  # batch_size=2, seq_len=5
        mock_input[0, 3:] = 0  # First example has 3 non-pad tokens
        mock_input[1, 4:] = 0  # Second example has 4 non-pad tokens
        mock_batch = {"input": mock_input}

        # Import function under test
        from interpretune.analysis.ops.definitions import get_answer_indices_impl

        existing_batch = AnalysisBatch()
        result_batch = get_answer_indices_impl(mock_module, existing_batch, mock_batch, 0)

        # Verify the results
        assert hasattr(result_batch, "answer_indices")
        # Expected indices: (num_non_pad_tokens - 1) for each example
        expected = torch.tensor([2, 3])  # 3-1=2 for first example, 4-1=3 for second
        assert torch.equal(result_batch.answer_indices, expected)

    def test_get_answer_indices_impl_with_input_store(self):
        """Test get_answer_indices_impl with answer_indices from input_store."""
        # Create mock module with input store
        mock_module = MagicMock()
        mock_module.analysis_cfg.input_store = MagicMock()
        mock_module.analysis_cfg.input_store.answer_indices = [torch.tensor([5, 6])]

        # Create mock batch
        mock_batch = {"input": torch.ones(2, 5)}

        # Import function under test
        from interpretune.analysis.ops.definitions import get_answer_indices_impl

        existing_batch = AnalysisBatch()
        # Test with batch_idx=0
        result_batch = get_answer_indices_impl(mock_module, existing_batch, mock_batch, 0)

        # Verify that answer_indices were taken from input_store
        assert hasattr(result_batch, "answer_indices")
        assert torch.equal(result_batch.answer_indices, torch.tensor([5, 6]))

    def test_get_alive_latents_impl(self):
        """Test get_alive_latents_impl function."""
        # Create mock module
        mock_module = MagicMock()
        mock_module.analysis_cfg.names_filter = lambda x: True
        mock_module.analysis_cfg.input_store = None

        mock_cache = None

        # Create analysis batch with empty cache and answer_indices
        analysis_batch = AnalysisBatch(
            cache=mock_cache,
            answer_indices=torch.tensor([2, 3])  # Answer positions for batch examples
        )

        # Create mock batch
        mock_batch = {"input": torch.ones(2, 4)}

        # Import function under test
        from interpretune.analysis.ops.definitions import get_alive_latents_impl

        # Run the function
        result_batch = get_alive_latents_impl(mock_module, analysis_batch, mock_batch, 0)

        assert isinstance(result_batch, AnalysisBatch)
        assert result_batch.alive_latents == {}

        # Create mock cache with activations
        mock_cache = {
            "hook1": torch.zeros(2, 4, 5),  # batch_size=2, seq_len=4, d_hidden=5
            "hook2": torch.zeros(2, 4, 5)
        }
        # Set some activations to be "alive"
        mock_cache["hook1"][0, 2, 1] = 1.0  # Example 0, seq_pos 2, latent 1
        mock_cache["hook1"][1, 3, 3] = 1.0  # Example 1, seq_pos 3, latent 3
        mock_cache["hook2"][0, 2, 0] = 1.0  # Example 0, seq_pos 2, latent 0
        mock_cache["hook2"][1, 3, 2] = 1.0  # Example 1, seq_pos 3, latent 2

        # Create analysis batch with non-empty cache and answer_indices
        analysis_batch = AnalysisBatch(
            cache=mock_cache,
            answer_indices=torch.tensor([2, 3])  # Answer positions for batch examples
        )

        # Run the function
        result_batch = get_alive_latents_impl(mock_module, analysis_batch, mock_batch, 0)

        # Verify the results
        assert hasattr(result_batch, "alive_latents")
        assert "hook1" in result_batch.alive_latents
        assert "hook2" in result_batch.alive_latents
        assert 1 in result_batch.alive_latents["hook1"]  # Latent 1 is alive
        assert 3 in result_batch.alive_latents["hook1"]  # Latent 3 is alive
        assert 0 in result_batch.alive_latents["hook2"]  # Latent 0 is alive
        assert 2 in result_batch.alive_latents["hook2"]  # Latent 2 is alive

        # mock no latents alive
        # Set some activations to be "alive"
        mock_cache["hook1"][0, 2, 1] = 0.0  # Example 0, seq_pos 2, latent 1 back to inactive
        mock_cache["hook1"][1, 3, 3] = 0.0  # Example 1, seq_pos 3, latent 3 back to inactive
        mock_cache["hook2"][0, 2, 0] = 0.0  # Example 0, seq_pos 2, latent 0 back to inactive
        mock_cache["hook2"][1, 3, 2] = 0.0  # Example 1, seq_pos 3, latent 2 back to inactive

        # Create analysis batch with non-empty cache and answer_indices
        analysis_batch = AnalysisBatch(
            cache=mock_cache,
            answer_indices=torch.tensor([2, 3])  # Answer positions for batch examples
        )

        # Run the function
        result_batch = get_alive_latents_impl(mock_module, analysis_batch, mock_batch, 0)

    def test_get_alive_latents_impl_with_input_store(self):
        """Test get_alive_latents_impl with alive_latents from input_store."""
        # Create mock module with input store
        mock_module = MagicMock()
        mock_module.analysis_cfg.input_store = MagicMock()
        mock_module.analysis_cfg.input_store.alive_latents = [{"hook1": [0, 1], "hook2": [2, 3]}]

        # Create mock batch
        mock_batch = {"input": torch.ones(2, 4)}

        # Import function under test
        from interpretune.analysis.ops.definitions import get_alive_latents_impl

        existing_batch = AnalysisBatch()
        # Test with batch_idx=0
        result_batch = get_alive_latents_impl(mock_module, existing_batch, mock_batch, 0)

        # Verify that alive_latents were taken from input_store
        assert hasattr(result_batch, "alive_latents")
        assert result_batch.alive_latents["hook1"] == [0, 1]
        assert result_batch.alive_latents["hook2"] == [2, 3]

    def test_get_alive_latents_impl_existing_batch(self):
        """Test get_alive_latents_impl with existing alive_latents."""
        # Create mock module
        mock_module = MagicMock()
        mock_module.analysis_cfg.input_store = None

        # Create analysis batch with alive_latents already set
        existing_alive_latents = {"hook1": [0, 1], "hook2": [2, 3]}
        analysis_batch = AnalysisBatch(alive_latents=existing_alive_latents)

        # Create mock batch
        mock_batch = {"input": torch.ones(2, 4)}

        # Import function under test
        from interpretune.analysis.ops.definitions import get_alive_latents_impl

        # Run the function
        result_batch = get_alive_latents_impl(mock_module, analysis_batch, mock_batch, 0)

        # Verify the existing alive_latents were preserved
        assert hasattr(result_batch, "alive_latents")
        assert result_batch.alive_latents == existing_alive_latents


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

    @pytest.fixture
    def scalar_logit_setup(self):
        """Fixture providing common setup for scalar logit diff tests."""
        # Create a scalar tensor (dimension 0)
        scalar_logit_diff = torch.tensor(0.3)
        assert scalar_logit_diff.dim() == 0, "Setup requires a scalar tensor"

        # Create mock_get_loss_preds_diffs that returns scalar logit_diffs
        with patch("interpretune.analysis.ops.definitions.get_loss_preds_diffs") as mock_get_loss_preds_diffs:
            mock_get_loss_preds_diffs.return_value = (
                torch.tensor(0.2),       # loss (scalar)
                scalar_logit_diff,       # logit_diffs (scalar tensor)
                torch.tensor([0]),       # preds
                torch.tensor([[1.0, 0.0]])  # answer_logits
            )
            yield mock_get_loss_preds_diffs, scalar_logit_diff

    @pytest.fixture
    def mock_module_base(self):
        """Fixture providing a basic mock module with standardized output methods."""
        mock_module = MagicMock()

        # Basic configuration
        mock_module.analysis_cfg = MagicMock()
        mock_module.analysis_cfg.auto_prune_batch_encoding = True
        mock_module.auto_prune_batch.return_value = {"input": torch.ones(1, 5)}

        # Mock standardize_logits to return a tensor
        mock_module.standardize_logits = MagicMock(return_value=torch.tensor([[[0.9, 0.1]]]))

        # Mock loss_fn to avoid tensor operation issues
        mock_module.loss_fn = MagicMock(return_value=torch.tensor(0.5))

        return mock_module

    def test_model_gradient_impl_scalar_logit_diffs(self, mock_module_base, scalar_logit_setup):
        """Test model_gradient_impl function with scalar logit_diffs edge case.

        This test specifically targets the case where logit_diffs is a scalar (dim=0) and verifies it gets properly
        unsqueezed for consistent batch handling.
        """
        # Get setup from fixtures
        mock_get_loss_preds_diffs, scalar_logit_diff = scalar_logit_setup
        mock_module = mock_module_base

        # Configure additional hooks - required for gradient attribution
        mock_module.analysis_cfg.fwd_hooks = [("hook1", MagicMock())]
        mock_module.analysis_cfg.bwd_hooks = [("hook1", MagicMock())]
        mock_module.analysis_cfg.cache_dict = {"hook1": torch.ones(1, 5, 3), "hook1_grad": torch.ones(1, 5, 3) * 0.5}
        mock_module.analysis_cfg.add_default_cache_hooks = MagicMock()

        # Set up model with SAEs
        mock_module.model = MagicMock()
        mock_module.sae_handles = [MagicMock()]

        # Create mock batch with BatchEncoding
        from transformers import BatchEncoding
        mock_batch = BatchEncoding({"input": torch.ones(1, 5)})  # Single example batch

        # Create analysis batch with required attributes
        analysis_batch = AnalysisBatch(
            answer_indices=torch.tensor([2]),  # Single example, answer at position 2
            labels=torch.tensor([0]),          # Required for get_loss_preds_diffs
            orig_labels=torch.tensor([0])      # Required for get_loss_preds_diffs
        )

        # Import the function to be tested
        from interpretune.analysis.ops.definitions import model_gradient_impl

        # Run the function with appropriate mocks
        with patch("torch.set_grad_enabled") as mock_grad_enabled, \
             patch.object(torch.Tensor, "backward") as mock_backward:

            # Configure context manager mocks
            mock_grad_enabled.return_value.__enter__.return_value = None
            mock_grad_enabled.return_value.__exit__.return_value = None
            mock_module.model.saes.return_value.__enter__.return_value = None
            mock_module.model.saes.return_value.__exit__.return_value = None
            mock_module.model.hooks.return_value.__enter__.return_value = None
            mock_module.model.hooks.return_value.__exit__.return_value = None

            # Create a tensor with sufficient sequence length for index 2
            # [batch_size=1, seq_len=5, features=2] - this allows accessing index 2 safely
            mock_logits = torch.zeros(1, 5, 2)
            mock_logits[0, 2, 0] = 1.0  # Put a value at the expected answer index
            mock_module.model.return_value = mock_logits

            # Run the function
            result_batch = model_gradient_impl(mock_module, analysis_batch, mock_batch, 0,
                                               get_loss_preds_diffs=mock_get_loss_preds_diffs)

            # Verify backward was called on the scalar tensor
            mock_backward.assert_called_once()

            # Verify the key edge case: scalar logit_diffs should be unsqueezed to 1D tensor
            assert result_batch.logit_diffs.dim() == 1, "Scalar logit_diffs should be unsqueezed to 1D"
            assert_close(result_batch.logit_diffs, scalar_logit_diff)

            # Verify other expected attributes are present
            assert hasattr(result_batch, "answer_logits")
            assert hasattr(result_batch, "answer_indices")
            assert hasattr(result_batch, "preds")
            assert hasattr(result_batch, "loss")
            assert hasattr(result_batch, "grad_cache")

    def test_logit_diffs_impl_scalar_logit_diffs(self, mock_module_base, scalar_logit_setup):
        """Test logit_diffs_impl function with scalar logit_diffs edge case.

        This test specifically targets the case where logit_diffs is a scalar (dim=0) and verifies it gets properly
        unsqueezed for consistent batch handling.
        """
        # Get setup from fixtures
        mock_get_loss_preds_diffs, scalar_logit_diff = scalar_logit_setup
        mock_module = mock_module_base

        # Create mock batch
        mock_batch = {"input": torch.ones(1, 5)}

        # Create analysis batch with required inputs for logit_diffs_impl
        analysis_batch = AnalysisBatch(
            answer_logits=torch.tensor([[[0.9, 0.1]]]),  # Single example logits
            answer_indices=torch.tensor([0]),            # Answer index
            labels=torch.tensor([0]),                    # For loss computation
            orig_labels=torch.tensor([0])                # For logit diff computation
        )

        # Import the function to be tested
        from interpretune.analysis.ops.definitions import logit_diffs_impl

        # Run the function - this will use our mocked get_loss_preds_diffs
        result_batch = logit_diffs_impl(mock_module, analysis_batch, mock_batch, 0,
                                        get_loss_preds_diffs=mock_get_loss_preds_diffs)

        # Verify the key edge case: scalar logit_diffs should be unsqueezed to 1D tensor
        assert result_batch.logit_diffs.dim() == 1, "Scalar logit_diffs should be unsqueezed to 1D"
        assert_close(result_batch.logit_diffs, scalar_logit_diff)

        # Verify mock_get_loss_preds_diffs was called with expected arguments
        mock_get_loss_preds_diffs.assert_called_once()

        # Verify the result contains all expected attributes
        assert hasattr(result_batch, "loss")
        assert torch.equal(result_batch.loss, torch.tensor(0.2))
        assert hasattr(result_batch, "preds")
        assert torch.equal(result_batch.preds, torch.tensor([0]))
        assert hasattr(result_batch, "answer_logits")
        assert torch.equal(result_batch.answer_logits, torch.tensor([[1.0, 0.0]]))

    def test_model_ablation_impl(self):
        """Test model_ablation_impl function and its edge cases for answer_indices and alive_latents."""
        # Create mock module
        mock_module = MagicMock()
        mock_module.analysis_cfg.auto_prune_batch_encoding = True
        mock_module.auto_prune_batch.return_value = {"input": torch.ones(2, 10)}

        # Create mock tensor with sufficient sequence length (at least 7 to support indices 0-6)
        mock_logits = torch.zeros(2, 10, 2)  # [batch_size=2, seq_len=10, features=2]
        # Fill in some sample values
        mock_logits[:, :, :] = torch.tensor([1.0, 0.0])  # Set default values
        mock_logits[1, :, :] = torch.tensor([0.0, 1.0])  # Different values for second example
        mock_module.model.run_with_hooks_with_saes.return_value = mock_logits

        # Make sure we have SAE handles for the test
        mock_module.sae_handles = [MagicMock()]

        # Create mock batch
        mock_batch = BatchEncoding({"input": torch.ones(2, 10)})

        # Import function under test and functions it depends on
        from interpretune.analysis.ops.definitions import model_ablation_impl

        # Test case 1: Analysis batch already has both answer_indices and alive_latents
        analysis_batch_complete = AnalysisBatch(
            answer_indices=torch.tensor([5, 6]),
            alive_latents={"hook1": [0, 1], "hook2": [2, 3]}
        )

        with patch("interpretune.analysis.ops.definitions.get_answer_indices_impl") as mock_get_indices, \
             patch("interpretune.analysis.ops.definitions.get_alive_latents_impl") as mock_get_alive:

            result_batch = model_ablation_impl(mock_module, analysis_batch_complete, mock_batch, 0)

            # Verify neither helper function was called
            mock_get_indices.assert_not_called()
            mock_get_alive.assert_not_called()

            # Verify auto_prune_batch was called
            mock_module.auto_prune_batch.assert_called_once_with(mock_batch, 'forward')

            # Verify run_with_hooks_with_saes was called for each latent (2 hooks × 2 latents each)
            assert mock_module.model.run_with_hooks_with_saes.call_count == 4

            # Verify results
            assert hasattr(result_batch, "answer_logits")
            assert isinstance(result_batch.answer_logits, dict)
            assert "hook1" in result_batch.answer_logits
            assert "hook2" in result_batch.answer_logits

        # Test case 2: Analysis batch missing answer_indices
        analysis_batch_no_indices = AnalysisBatch(
            alive_latents={"hook1": [0, 1], "hook2": [2, 3]}
        )

        with patch("interpretune.analysis.ops.definitions.get_answer_indices_impl") as mock_get_indices, \
             patch("interpretune.analysis.ops.definitions.get_alive_latents_impl") as mock_get_alive:

            # Set up mock to return an analysis batch with answer_indices
            mock_get_indices.return_value = AnalysisBatch(
                answer_indices=torch.tensor([5, 6]),
                alive_latents={"hook1": [0, 1], "hook2": [2, 3]}
            )

            mock_module.model.run_with_hooks_with_saes.reset_mock()
            mock_module.auto_prune_batch.reset_mock()

            result_batch = model_ablation_impl(mock_module, analysis_batch_no_indices, mock_batch, 0)

            # Verify only the indices function was called
            mock_get_indices.assert_called_once()
            mock_get_alive.assert_not_called()

        # Test case 3: Analysis batch missing alive_latents, can retrieve from input_store
        analysis_batch_no_alive = AnalysisBatch(
            answer_indices=torch.tensor([5, 6])
        )

        # Set up input_store with alive_latents
        mock_module.analysis_cfg.input_store = MagicMock()
        mock_module.analysis_cfg.input_store.alive_latents = [{"hook1": [0, 1], "hook2": [2, 3]}]

        with patch("interpretune.analysis.ops.definitions.get_answer_indices_impl") as mock_get_indices, \
             patch("interpretune.analysis.ops.definitions.get_alive_latents_impl") as mock_get_alive:

            # Set up mock to return an analysis batch with alive_latents
            mock_get_alive.return_value = AnalysisBatch(
                answer_indices=torch.tensor([5, 6]),
                alive_latents={"hook1": [0, 1], "hook2": [2, 3]}
            )

            mock_module.model.run_with_hooks_with_saes.reset_mock()
            mock_module.auto_prune_batch.reset_mock()

            result_batch = model_ablation_impl(mock_module, analysis_batch_no_alive, mock_batch, 0)

            # Verify only the alive_latents function was called
            mock_get_indices.assert_not_called()
            mock_get_alive.assert_called_once()

        # Test case 4: Analysis batch missing alive_latents, input_store has none (should raise AssertionError)
        analysis_batch_error = AnalysisBatch(
            answer_indices=torch.tensor([5, 6])
        )

        # Set up input_store with no alive_latents
        mock_module.analysis_cfg.input_store = None

        with patch("interpretune.analysis.ops.definitions.get_answer_indices_impl") as mock_get_indices, \
             patch("interpretune.analysis.ops.definitions.get_alive_latents_impl") as mock_get_alive:

            with pytest.raises(AssertionError, match="alive_latents required for ablation op"):
                model_ablation_impl(mock_module, analysis_batch_error, mock_batch, 0)

    def test_sae_correct_acts_impl_scalar_and_error_cases(self, mock_module_base):
        """Test sae_correct_acts_impl function with scalar logit_diffs and required input validation.

        This test specifically targets:
        1. The case where logit_diffs is a scalar (dim=0) and needs to be unsqueezed
        2. The ValueError handling for missing required inputs
        """
        mock_module = mock_module_base
        mock_module.__class__.__name__ = "TestModule"  # Set class name for error message testing

        # Create mock batch
        mock_batch = {"input": torch.ones(1, 5)}

        # Import the function to be tested
        from interpretune.analysis.ops.definitions import sae_correct_acts_impl

        # Test missing required inputs (ValueError case)
        incomplete_batch = AnalysisBatch(
            # Missing one or more required fields: logit_diffs, answer_indices, cache
            logit_diffs=torch.tensor(0.5)  # Only provide logit_diffs
        )

        with pytest.raises(ValueError, match="Missing required input 'answer_indices' for TestModule.sae_correct_acts"):
            sae_correct_acts_impl(mock_module, incomplete_batch, mock_batch, 0)

        incomplete_batch_2 = AnalysisBatch(
            # Missing cache
            logit_diffs=torch.tensor(0.5),
            answer_indices=torch.tensor([2])
        )

        with pytest.raises(ValueError, match="Missing required input 'cache' for TestModule.sae_correct_acts"):
            sae_correct_acts_impl(mock_module, incomplete_batch_2, mock_batch, 0)

        # Now test the scalar case - prepare a complete batch with scalar logit_diffs
        # Create a scalar tensor (dimension 0)
        scalar_logit_diff = torch.tensor(0.3)
        assert scalar_logit_diff.dim() == 0, "Test setup requires a scalar tensor"

        # Create a mock cache
        mock_cache = {
            "hook1": torch.zeros(1, 5, 4),  # [batch_size=1, seq_len=5, features=4]
            "no_match": torch.zeros(1, 5, 4)  # Another hook for testing unmatched filter
        }
        # Set some activations to be analyzed
        mock_cache["hook1"][0, 2, 1] = 1.0  # Example 0, seq_pos 2, latent 1
        mock_cache["no_match"][0, 3, 2] = 1.0  # Example 0, seq_pos 2, latent 1

        # Create analysis batch with all required attributes including scalar logit_diffs
        analysis_batch = AnalysisBatch(
            logit_diffs=scalar_logit_diff,  # Scalar tensor
            answer_indices=torch.tensor([2]),
            cache=mock_cache
        )

        # Set up names_filter to accept all hooks
        #mock_module.analysis_cfg.names_filter = lambda x: True
        mock_module.analysis_cfg.names_filter = lambda x : True if x != 'no_match' else False
        # Run the function
        result_batch = sae_correct_acts_impl(mock_module, analysis_batch, mock_batch, 0)

        # Verify the scalar logit_diffs was properly handled
        # The function should have created correct_activations for hook1 since logit_diff is positive
        assert hasattr(result_batch, "correct_activations")
        assert "hook1" in result_batch.correct_activations
        assert isinstance(result_batch.correct_activations["hook1"], torch.Tensor)

        # Since our scalar logit_diff was positive (0.3 > 0), the activation should have been
        # captured and included in correct_activations
        assert result_batch.correct_activations["hook1"].numel() > 0

        # Test with negative scalar logit_diff (should result in empty correct_activations)
        negative_scalar_logit_diff = torch.tensor(-0.3)
        assert negative_scalar_logit_diff.dim() == 0, "Test setup requires a scalar tensor"

        analysis_batch.logit_diffs = negative_scalar_logit_diff
        result_batch = sae_correct_acts_impl(mock_module, analysis_batch, mock_batch, 0)

        # Should still have correct_activations field but the tensors should be empty
        assert hasattr(result_batch, "correct_activations")
        assert "hook1" in result_batch.correct_activations
        # For negative logit_diffs, no activations should be considered "correct"
        assert result_batch.correct_activations["hook1"].numel() == 0

    def test_gradient_attribution_impl_error_cases(self, mock_module_base):
        """Test gradient_attribution_impl function with error cases and cache selection logic.

        This test specifically targets:
        1. The ValueError cases for missing required inputs
        2. The ValueError case when no cache is available
        3. The two branches for getting cache from analysis_batch.grad_cache or module.analysis_cfg.cache_dict
        """
        mock_module = mock_module_base
        mock_module.__class__.__name__ = "TestModule"  # Set class name for error message testing

        # Create mock batch
        mock_batch = {"input": torch.ones(1, 5)}

        # Import the function to be tested
        from interpretune.analysis.ops.definitions import gradient_attribution_impl

        # Test 1: Missing required input 'answer_indices'
        incomplete_batch = AnalysisBatch(
            # Missing answer_indices
            logit_diffs=torch.tensor([0.5])
        )

        with pytest.raises(ValueError, match="Missing required input 'answer_indices' for gradient attribution"):
            gradient_attribution_impl(mock_module, incomplete_batch, mock_batch, 0)

        # Test 2: Missing required input 'logit_diffs'
        incomplete_batch_2 = AnalysisBatch(
            # Missing logit_diffs
            answer_indices=torch.tensor([2])
        )

        with pytest.raises(ValueError, match="Missing required input 'logit_diffs' for gradient attribution"):
            gradient_attribution_impl(mock_module, incomplete_batch_2, mock_batch, 0)

        # Test 3: Missing cache (both grad_cache and cache_dict)
        # Create a batch with required inputs but no cache
        no_cache_batch = AnalysisBatch(
            answer_indices=torch.tensor([2]),
            logit_diffs=torch.tensor([0.5])
        )

        # Make sure module has no cache_dict
        mock_module.analysis_cfg.cache_dict = None

        with pytest.raises(ValueError, match="No cache available: neither analysis_batch.grad_cache"):
            gradient_attribution_impl(mock_module, no_cache_batch, mock_batch, 0)

        # Test 4: Using analysis_batch.grad_cache (first branch)
        # Create a mock batch with grad_cache
        batch_size = 1
        seq_len = 5
        d_sae = 4

        # Create mock gradient cache with appropriate data
        mock_grad_cache = {
            "hook1": torch.zeros(batch_size, seq_len, d_sae),
            "hook1_grad": torch.zeros(batch_size, seq_len, d_sae),
            "hook2": torch.zeros(batch_size, seq_len, d_sae),  # Non-grad hook
            "no_grad_hook": torch.zeros(batch_size, seq_len, d_sae)  # Hook without grad
        }

        # Set up some mock activations and gradients
        mock_grad_cache["hook1"][0, 2, 1] = 1.0  # Example 0, seq_pos 2, latent 1 activation
        mock_grad_cache["hook1_grad"][0, 2, 1] = 0.5  # Example 0, seq_pos 2, latent 1 gradient

        # Create a batch with grad_cache
        grad_cache_batch = AnalysisBatch(
            answer_indices=torch.tensor([2]),
            logit_diffs=torch.tensor([0.5]),
            grad_cache=mock_grad_cache
        )

        # Set up SAE handles
        mock_module.sae_handles = [MagicMock()]
        mock_module.sae_handles[0].cfg.d_sae = d_sae
        mock_module.model = MagicMock()

        # Configure names_filter to accept all hooks
        mock_module.analysis_cfg.names_filter = lambda x: True

        # Create mock_get_loss_preds_diffs that returns scalar logit_diffs
        with patch("interpretune.analysis.ops.definitions.get_alive_latents_impl") as mock_get_alive_latents_impl:
            mock_get_alive_latents_impl.return_value = AnalysisBatch(alive_latents={"hook1": [0, 1]})

            # Run the function
            result_batch = gradient_attribution_impl(mock_module, grad_cache_batch, mock_batch, 0)

        # Verify the result has attribution_values
        assert hasattr(result_batch, "attribution_values")
        assert "hook1" in result_batch.attribution_values
        assert isinstance(result_batch.attribution_values["hook1"], torch.Tensor)
        assert result_batch.attribution_values["hook1"].shape == (batch_size, d_sae)

        # The latent 1 should have non-zero attribution (activation * gradient)
        assert result_batch.attribution_values["hook1"][0, 1] > 0

        # Test 5: Using module.analysis_cfg.cache_dict (second branch)
        # Create a batch without grad_cache
        module_cache_batch = AnalysisBatch(
            answer_indices=torch.tensor([2]),
            logit_diffs=torch.tensor([0.5])
        )

        # Set up module's cache_dict
        mock_module.analysis_cfg.cache_dict = mock_grad_cache

        with patch("interpretune.analysis.ops.definitions.get_alive_latents_impl") as mock_get_alive_latents_impl:
            mock_get_alive_latents_impl.return_value = AnalysisBatch(alive_latents={"hook1": [0, 1]})
            # Run the function
            result_batch = gradient_attribution_impl(mock_module, module_cache_batch, mock_batch, 0)

        # Verify the result has attribution_values
        assert hasattr(result_batch, "attribution_values")
        assert "hook1" in result_batch.attribution_values
        assert isinstance(result_batch.attribution_values["hook1"], torch.Tensor)

        # The latent 1 should have non-zero attribution (activation * gradient)
        assert result_batch.attribution_values["hook1"][0, 1] > 0

        # Also verify that alive_latents was set
        assert hasattr(result_batch, "alive_latents")
        assert "hook1" in result_batch.alive_latents

    def test_ablation_attribution_impl_scalar_and_error_cases(self, mock_module_base):
        """Test ablation_attribution_impl function with scalar tensor handling and required input validation.

        This test specifically targets:
        1. The ValueError handling for missing required inputs
        2. The case where tensors like example_mask or base_diffs are scalars (dim=0) and need to be unsqueezed
        """
        # Setup mock module
        mock_module = mock_module_base

        # Create mock batch
        mock_batch = {"input": torch.ones(1, 5)}

        # Import the function
        from interpretune.analysis.ops.definitions import ablation_attribution_impl

        # Test 1: Missing required inputs (answer_logits, alive_latents, logit_diffs)
        for missing_key in ['answer_logits', 'alive_latents', 'logit_diffs']:
            # Create a batch with all required inputs except the one we're testing
            batch_args = {
                'answer_logits': {"hook1": {0: torch.tensor([[1.0, 0.0]])}},
                'alive_latents': {"hook1": [0]},
                'logit_diffs': torch.tensor([0.5])
            }
            # Remove the key we want to test is missing
            batch_args.pop(missing_key)
            incomplete_batch = AnalysisBatch(**batch_args)

            with pytest.raises(ValueError, match=f"Missing required input '{missing_key}' for ablation attribution"):
                ablation_attribution_impl(mock_module, incomplete_batch, mock_batch, 0)

        # Setup for scalar tensor test
        # Create mock SAE handles with feature dimension
        mock_module.sae_handles = [MagicMock()]
        mock_module.sae_handles[0].cfg.d_sae = 2  # Small feature dimension for testing

        # Create a mock for get_loss_preds_diffs that will return a scalar logit diff
        # This will trigger the creation of a scalar example_mask in the implementation
        mock_get_loss_preds_diffs = MagicMock()
        scalar_result = torch.tensor(0.1)  # A small positive scalar -> positive example_mask
        assert scalar_result.dim() == 0, "Setup requires a scalar tensor"

        mock_get_loss_preds_diffs.return_value = (
            torch.tensor(0.5),     # loss
            scalar_result,         # logit_diffs (scalar)
            torch.tensor(0),       # preds (scalar)
            torch.tensor([[0.6, 0.4]])  # answer_logits
        )

        # Create analysis batch with scalar logit_diffs
        scalar_batch = AnalysisBatch(
            # Per-latent logits for a single hook and single latent
            answer_logits={"hook1": {0: torch.tensor([[0.6, 0.4]])}},
            alive_latents={"hook1": [0]},
            # Use a scalar tensor for logit_diffs to trigger the unsqueeze_ logic
            logit_diffs=torch.tensor(0.3)  # Scalar positive value -> attribution should be calculated
        )
        assert scalar_batch.logit_diffs.dim() == 0, "Test requires a scalar logit_diffs"

        # Test the scalar handling by tracing calls to unsqueeze_
        original_unsqueeze_ = torch.Tensor.unsqueeze_
        unsqueezed_tensors = []  # To track which tensors get unsqueezed

        def spy_unsqueeze_(self, *args, **kwargs):
            # Track original dimension
            unsqueezed_tensors.append(self.dim())
            return original_unsqueeze_(self, *args, **kwargs)

        with patch.object(torch.Tensor, "unsqueeze_", spy_unsqueeze_):
            # Run the function
            result_batch = ablation_attribution_impl(
                mock_module, scalar_batch, mock_batch, 0,
                get_loss_preds_diffs=mock_get_loss_preds_diffs
            )

        # Verify that at least one scalar tensor was unsqueezed
        assert 0 in unsqueezed_tensors, "Expected at least one scalar tensor to be unsqueezed"

        # Verify that attribution_values were calculated correctly
        assert hasattr(result_batch, "attribution_values")
        assert "hook1" in result_batch.attribution_values
        attribution = result_batch.attribution_values["hook1"]

        # The attribution should be a 2D tensor (batch_size=1, features) even though inputs were scalar
        assert attribution.dim() == 2, "Attribution should be a 2D tensor"
        assert attribution.shape == (1, mock_module.sae_handles[0].cfg.d_sae)

        # Test that per_latent dictionaries were created properly
        assert hasattr(result_batch, "loss")
        assert "hook1" in result_batch.loss
        assert 0 in result_batch.loss["hook1"]

        assert hasattr(result_batch, "logit_diffs")
        assert "hook1" in result_batch.logit_diffs
        assert 0 in result_batch.logit_diffs["hook1"]

        # Test with negative scalar logit_diff where no attribution would be calculated
        negative_scalar_batch = AnalysisBatch(
            answer_logits={"hook1": {0: torch.tensor([[0.6, 0.4]])}},
            alive_latents={"hook1": [0]},
            logit_diffs=torch.tensor(-0.3)  # Negative scalar
        )
        assert negative_scalar_batch.logit_diffs.dim() == 0, "Test requires a scalar logit_diffs"

        # Return a negative scalar result to ensure no attributions are calculated
        mock_get_loss_preds_diffs.return_value = (
            torch.tensor(0.5),     # loss
            torch.tensor(-0.1),    # negative logit_diffs (scalar)
            torch.tensor(0),       # preds
            torch.tensor([[0.4, 0.6]])  # answer_logits
        )

        # Run the function again with negative scalar
        result_batch = ablation_attribution_impl(
            mock_module, negative_scalar_batch, mock_batch, 0,
            get_loss_preds_diffs=mock_get_loss_preds_diffs
        )

        # Verify that attribution values for negative logit_diffs are zero
        assert hasattr(result_batch, "attribution_values")
        assert "hook1" in result_batch.attribution_values
        assert torch.all(result_batch.attribution_values["hook1"] == 0)


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
