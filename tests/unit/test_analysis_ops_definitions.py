from __future__ import annotations
import pytest
import torch
from unittest.mock import MagicMock, patch
from transformers import BatchEncoding

from interpretune.analysis.ops.definitions import (
    get_loss_preds_diffs,
    ablate_sae_latent,
    gradient_attribution_impl
)
from interpretune.analysis.ops.base import AnalysisBatch


class TestBooleanLogitFunctions:
    """Tests for the boolean logits utility functions."""

    # def test_boolean_logits_to_avg_logit_diff(self):
    #     """Test the boolean_logits_to_avg_logit_diff function."""
    #     # Create test data: batch_size=3, seq_len=1, num_classes=2
    #     logits = torch.tensor([[[0.8, 0.2]], [[0.3, 0.7]], [[0.6, 0.4]]])
    #     target_indices = torch.tensor([0, 1, 0])  # First and third examples correct class is 0, second is 1

    #     # Test with no reduction
    #     logit_diff = boolean_logits_to_avg_logit_diff(logits, target_indices)
    #     expected = torch.tensor([0.6, 0.4, 0.2])  # Difference between correct and incorrect logits
    #     assert torch.allclose(logit_diff, expected)

    #     # Test with mean reduction
    #     mean_diff = boolean_logits_to_avg_logit_diff(logits, target_indices, reduction="mean")
    #     expected_mean = torch.tensor(0.4)  # Mean of [0.6, 0.4, 0.2]
    #     assert torch.allclose(mean_diff, expected_mean)

    #     # Test with sum reduction
    #     sum_diff = boolean_logits_to_avg_logit_diff(logits, target_indices, reduction="sum")
    #     expected_sum = torch.tensor(1.2)  # Sum of [0.6, 0.4, 0.2]
    #     assert torch.allclose(sum_diff, expected_sum)

    #     # Test with keep_as_tensor=False
    #     list_diff = boolean_logits_to_avg_logit_diff(logits, target_indices, keep_as_tensor=False)
    #     assert isinstance(list_diff, list)
    #     assert list_diff == [0.6, 0.4, 0.2]

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

    # def test_labels_to_ids_impl(self):
    #     """Test the labels_to_ids_impl function."""
    #     # Create mock module
    #     mock_module = MagicMock()
    #     mock_module.labels_to_ids.return_value = (torch.tensor([0, 1]), torch.tensor([0, 1]))

    #     # Create mock batch
    #     mock_batch = BatchEncoding({"input": torch.tensor([[1, 2], [3, 4]]), "labels": ["label1", "label2"]})

    #     # Test with None analysis_batch
    #     result = labels_to_ids_impl(mock_module, None, mock_batch, 0)

    #     # Check that an AnalysisBatch was created with labels
    #     assert isinstance(result, AnalysisBatch)
    #     assert torch.equal(result.labels, torch.tensor([0, 1]))
    #     assert torch.equal(result.orig_labels, torch.tensor([0, 1]))

    #     # Test with existing analysis_batch
    #     existing_batch = AnalysisBatch()
    #     result = labels_to_ids_impl(mock_module, existing_batch, mock_batch, 0)

    #     # Check that the existing batch was updated
    #     assert result is existing_batch
    #     assert torch.equal(result.labels, torch.tensor([0, 1]))
    #     assert torch.equal(result.orig_labels, torch.tensor([0, 1]))

    #     # Test without labels in batch
    #     mock_batch_no_labels = BatchEncoding({"input": torch.tensor([[1, 2], [3, 4]])})
    #     result = labels_to_ids_impl(mock_module, None, mock_batch_no_labels, 0)

    #     # Should return empty AnalysisBatch
    #     assert isinstance(result, AnalysisBatch)
    #     assert not hasattr(result, "labels")


class TestAnalysisOperationsImplementations:
    """Tests for the core analysis operation implementation functions."""

    @pytest.fixture
    def mock_module(self):
        """Create a mock module for testing."""
        module = MagicMock()
        module.analysis_cfg = MagicMock()
        module.analysis_cfg.names_filter = lambda x: True
        module.datamodule = MagicMock()
        module.model = MagicMock()
        return module

    # def test_get_answer_indices_impl(self, mock_module):
    #     """Test the get_answer_indices_impl function."""
    #     # Create mock batch
    #     mock_batch = BatchEncoding({"input": torch.tensor([[1, 2, 3], [4, 5, 6]])})

    #     # Test with None analysis_batch and left padding tokenizer
    #     mock_module.datamodule.tokenizer.padding_side = "left"

    #     result = get_answer_indices_impl(mock_module, None, mock_batch, 0)

    #     # Should return indices pointing to the last token position (-1)
    #     assert isinstance(result, AnalysisBatch)
    #     assert torch.equal(result.answer_indices, torch.full((2,), -1))

    #     # Test with None analysis_batch and right padding tokenizer
    #     mock_module.datamodule.tokenizer.padding_side = "right"
    #     mock_module.datamodule.tokenizer.pad_token_id = 0  # Assume 0 is pad token
    #     mock_batch = BatchEncoding({"input": torch.tensor([[1, 2, 0], [4, 5, 0]])})

    #     result = get_answer_indices_impl(mock_module, None, mock_batch, 0)

    #     # Should compute indices based on non-padding tokens
    #     assert isinstance(result, AnalysisBatch)
    #     assert torch.equal(result.answer_indices, torch.tensor([1, 1]))  # 2 tokens each, so indices should be 1

    #     # Test with existing answer_indices in input_store
    #     mock_module.analysis_cfg.input_store = MagicMock()
    #     mock_module.analysis_cfg.input_store.answer_indices = [torch.tensor([2, 2])]

    #     result = get_answer_indices_impl(mock_module, None, mock_batch, 0)

    #     # Should use the indices from input_store
    #     assert torch.equal(result.answer_indices, torch.tensor([2, 2]))

    # def test_model_forward_impl(self, mock_module):
    #     """Test the model_forward_impl function."""
    #     # Create mock batch
    #     mock_batch = BatchEncoding({"input": torch.tensor([[1, 2, 3], [4, 5, 6]])})

    #     # Set up model to return tensor
    #     mock_module.return_value = torch.tensor([[[0.8, 0.2]], [[0.3, 0.7]]])

    #     # Test with None analysis_batch
    #     with patch('interpretune.analysis.ops.definitions.get_answer_indices_impl',
    #                return_value=AnalysisBatch(answer_indices=torch.tensor([2, 2]))):
    #         result = model_forward_impl(mock_module, None, mock_batch, 0)

    #     # Should return batch with answer logits
    #     assert isinstance(result, AnalysisBatch)
    #     assert torch.equal(result.answer_logits, torch.tensor([[[0.8, 0.2]], [[0.3, 0.7]]]))

    #     # Test with existing analysis_batch
    #     existing_batch = AnalysisBatch()
    #     with patch('interpretune.analysis.ops.definitions.get_answer_indices_impl',
    #                return_value=AnalysisBatch(answer_indices=torch.tensor([2, 2]))):
    #         result = model_forward_impl(mock_module, existing_batch, mock_batch, 0)

    #     # Should update the existing batch
    #     assert result is existing_batch
    #     assert torch.equal(result.answer_logits, torch.tensor([[[0.8, 0.2]], [[0.3, 0.7]]]))


# Additional test classes for other operation implementations would follow similar patterns
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

    def test_gradient_attribution_impl(self, mock_module_with_cache):
        """Test the gradient_attribution_impl function."""
        # Create mock batch
        mock_batch = BatchEncoding({"input": torch.tensor([[1, 2, 3], [4, 5, 6]])})

        # Create mock analysis batch with required fields
        mock_analysis_batch = AnalysisBatch(
            answer_indices=torch.tensor([1, 1]),
            logit_diffs=torch.tensor([0.6, -0.2])  # First example positive, second negative
        )

        # Mock get_alive_latents_impl to return mock alive_latents
        with patch('interpretune.analysis.ops.definitions.get_alive_latents_impl',
                  return_value=AnalysisBatch(alive_latents={"hook1": [0, 1, 2]})):
            result = gradient_attribution_impl(mock_module_with_cache, mock_analysis_batch, mock_batch, 0)

        # Check that attribution values and correct activations were computed
        assert "attribution_values" in result
        assert "correct_activations" in result
        assert "hook1" in result.attribution_values
        assert "hook1" in result.correct_activations

        # Check attribution shape
        assert result.attribution_values["hook1"].shape == torch.Size([2, 4])  # batch_size x d_sae

        # First example has positive logit diff, so should have correct activations
        assert result.correct_activations["hook1"].shape == torch.Size([1, 4])  # 1 positive example x d_sae
