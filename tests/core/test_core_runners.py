from __future__ import annotations
import pytest
import torch
from unittest.mock import MagicMock

from interpretune.runners.core import (
    core_train_loop,
    core_test_loop
)


class TestCoreRunner:
    """Tests for the core runner module."""

    @pytest.fixture
    def mock_module(self):
        """Create a mock module for testing."""
        module = MagicMock()
        module.model = MagicMock()
        module.training_step.return_value = torch.tensor(0.5)
        module.validation_step.return_value = None
        module.test_step.return_value = None
        module.optimizers = [MagicMock()]
        return module

    @pytest.fixture
    def mock_datamodule(self):
        """Create a mock datamodule for testing."""
        datamodule = MagicMock()
        train_batch = {"input": torch.tensor([[1, 2], [3, 4]]), "labels": torch.tensor([0, 1])}
        val_batch = {"input": torch.tensor([[5, 6], [7, 8]]), "labels": torch.tensor([1, 0])}

        # Create mock dataloaders
        train_dataloader = MagicMock()
        train_dataloader.__iter__.return_value = [train_batch]

        val_dataloader = MagicMock()
        val_dataloader.__iter__.return_value = [val_batch]

        # Assign dataloaders to datamodule
        datamodule.train_dataloader.return_value = train_dataloader
        datamodule.val_dataloader.return_value = val_dataloader
        datamodule.test_dataloader.return_value = val_dataloader  # Use val data for test too

        return datamodule

    def test_core_test_loop(self, mock_module, mock_datamodule):
        """Test the core_test_loop function."""
        # Call the test loop
        core_test_loop(
            module=mock_module,
            datamodule=mock_datamodule,
            limit_test_batches=1
        )

        # Verify model was set to eval mode
        mock_module.model.eval.assert_called_once()

        # Verify test_step was called
        mock_module.test_step.assert_called_once()

        # Verify datamodule was used correctly
        mock_datamodule.test_dataloader.assert_called_once()

class TestCoreRunnerEdgeCases:
    """Tests for edge cases in the core runner."""

    @pytest.fixture
    def mock_module(self):
        """Create a mock module for testing."""
        module = MagicMock()
        module.model = MagicMock()
        module.optimizers = [MagicMock()]
        return module

    @pytest.fixture
    def mock_datamodule_empty(self):
        """Create a mock datamodule with empty dataloaders."""
        datamodule = MagicMock()

        # Create empty dataloaders
        empty_dataloader = MagicMock()
        empty_dataloader.__iter__.return_value = []

        datamodule.train_dataloader.return_value = empty_dataloader
        datamodule.val_dataloader.return_value = empty_dataloader
        datamodule.test_dataloader.return_value = empty_dataloader

        return datamodule

    @pytest.fixture
    def mock_datamodule_no_val(self):
        """Create a mock datamodule without validation dataloader."""
        datamodule = MagicMock()
        train_batch = {"input": torch.tensor([[1, 2], [3, 4]]), "labels": torch.tensor([0, 1])}

        # Create train dataloader
        train_dataloader = MagicMock()
        train_dataloader.__iter__.return_value = [train_batch]

        datamodule.train_dataloader.return_value = train_dataloader
        datamodule.val_dataloader.return_value = None

        return datamodule

    def test_train_loop_with_empty_dataloaders(self, mock_module, mock_datamodule_empty):
        """Test the train loop with empty dataloaders."""
        # This should not raise exceptions
        core_train_loop(
            module=mock_module,
            datamodule=mock_datamodule_empty,
            limit_train_batches=1,
            limit_val_batches=1,
            max_epochs=1
        )

        # Verify epoch hooks were still called
        mock_module.on_train_epoch_start.assert_called_once()
        mock_module.on_train_epoch_end.assert_called_once()

        # No steps should be run
        mock_module.training_step.assert_not_called()
        mock_module.validation_step.assert_not_called()

    def test_train_loop_without_validation(self, mock_module, mock_datamodule_no_val):
        """Test the train loop without a validation dataloader."""
        # This should not raise exceptions
        core_train_loop(
            module=mock_module,
            datamodule=mock_datamodule_no_val,
            limit_train_batches=1,
            limit_val_batches=1,
            max_epochs=1
        )

        # Training should proceed normally
        mock_module.training_step.assert_called_once()

        # No validation step should be run
        mock_module.validation_step.assert_not_called()

        # Model should not switch to eval mode for validation
        assert mock_module.model.eval.call_count == 0
