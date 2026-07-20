from __future__ import annotations
import pytest
import torch
from unittest.mock import MagicMock

from interpretune.runners.core import core_train_loop, core_test_loop
from interpretune.base import _call_itmodule_hook


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
        core_test_loop(module=mock_module, datamodule=mock_datamodule, limit_test_batches=1)

        # Verify model was set to eval mode
        mock_module.model.eval.assert_called_once()

        # Verify test_step was called
        mock_module.test_step.assert_called_once()

        # Verify datamodule was used correctly
        mock_datamodule.test_dataloader.assert_called_once()


class TestCallITModuleHook:
    """Tests for _call_itmodule_hook optional hook handling."""

    def test_optional_hook_missing_returns_none(self):
        """When optional=True and hook is missing, _call_itmodule_hook returns None."""
        module = MagicMock(spec=[])  # spec=[] means no attributes
        result = _call_itmodule_hook(module, hook_name="nonexistent_hook", hook_msg="test", optional=True)
        assert result is None

    def test_optional_hook_present_is_called(self):
        """When optional=True and hook exists, it is called normally."""
        module = MagicMock()
        module.my_hook.return_value = "result"
        result = _call_itmodule_hook(module, hook_name="my_hook", hook_msg="test", optional=True)
        module.my_hook.assert_called_once()
        assert result == "result"

    def test_required_hook_missing_raises(self):
        """When optional=False (default) and hook is missing, raises AttributeError."""
        module = MagicMock(spec=[])
        with pytest.raises(AttributeError, match="nonexistent_hook"):
            _call_itmodule_hook(module, hook_name="nonexistent_hook", hook_msg="test")

    def test_required_hook_present_is_called(self):
        """When optional=False (default) and hook exists, it is called normally."""
        module = MagicMock()
        module.my_hook.return_value = "result"
        assert _call_itmodule_hook(module, hook_name="my_hook", hook_msg="test") == "result"
        module.my_hook.assert_called_once()

    def test_hook_receives_kwargs(self):
        """Hook receives additional kwargs passed through _call_itmodule_hook."""
        module = MagicMock()
        _call_itmodule_hook(module, hook_name="my_hook", hook_msg="test", batch={"x": 1}, batch_idx=0)
        module.my_hook.assert_called_once_with(batch={"x": 1}, batch_idx=0)


class TestCoreTestLoopOptionalHooks:
    """Tests for core_test_loop with modules missing optional hooks."""

    @pytest.fixture
    def mock_module_no_hooks(self):
        """Module without optional hooks (on_test_batch_start, on_test_epoch_end)."""
        module = MagicMock()
        module.model = MagicMock()
        module.test_step.return_value = None
        module.global_step = 0
        module._it_state = MagicMock()
        module.device = torch.device("cpu")
        module.dtype = torch.float32
        # Remove optional hooks
        del module.on_test_batch_start
        del module.on_test_epoch_end
        return module

    @pytest.fixture
    def mock_datamodule(self):
        datamodule = MagicMock()
        test_batch = {"input": torch.tensor([[1, 2]]), "labels": torch.tensor([0])}
        test_dataloader = MagicMock()
        test_dataloader.__iter__.return_value = [test_batch]
        datamodule.test_dataloader.return_value = test_dataloader
        return datamodule

    def test_test_loop_succeeds_without_optional_hooks(self, mock_module_no_hooks, mock_datamodule):
        """core_test_loop completes without error when module lacks optional hooks."""
        core_test_loop(module=mock_module_no_hooks, datamodule=mock_datamodule, limit_test_batches=1)
        mock_module_no_hooks.test_step.assert_called_once()


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
            max_epochs=1,
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
            max_epochs=1,
        )

        # Training should proceed normally
        mock_module.training_step.assert_called_once()

        # No validation step should be run
        mock_module.validation_step.assert_not_called()

        # Model should not switch to eval mode for validation
        assert mock_module.model.eval.call_count == 0


class TestCoreLogMetricAccumulation:
    """Tests for CoreHelperAttributes.log/log_dict and core_test_loop metric reporting."""

    @pytest.fixture
    def mock_module_with_metrics(self):
        """Module that accumulates metrics via _logged_metrics."""
        module = MagicMock()
        module.model = MagicMock()
        module.global_step = 0
        module._it_state = MagicMock()
        module.device = torch.device("cpu")
        module.dtype = torch.float32
        module._logged_metrics = {}

        def log_side_effect(name, value, *args, **kwargs):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            module._logged_metrics.setdefault(name, []).append(value)

        def log_dict_side_effect(metric_dict, *args, **kwargs):
            for k, v in metric_dict.items():
                log_side_effect(k, v)

        module.log = MagicMock(side_effect=log_side_effect)
        module.log_dict = MagicMock(side_effect=log_dict_side_effect)

        def test_step_side_effect(batch, batch_idx):
            module.log("accuracy", 0.85)
            module.log("loss", torch.tensor(0.35))
            return None

        module.test_step = MagicMock(side_effect=test_step_side_effect)
        del module.on_test_batch_start
        del module.on_test_epoch_end
        return module

    @pytest.fixture
    def mock_datamodule_multi_batch(self):
        datamodule = MagicMock()
        batch1 = {"input": torch.tensor([[1, 2]]), "labels": torch.tensor([0])}
        batch2 = {"input": torch.tensor([[3, 4]]), "labels": torch.tensor([1])}
        test_dataloader = MagicMock()
        test_dataloader.__iter__.return_value = [batch1, batch2]
        datamodule.test_dataloader.return_value = test_dataloader
        return datamodule

    def test_metrics_accumulated_across_batches(self, mock_module_with_metrics, mock_datamodule_multi_batch):
        """Log() calls accumulate values in _logged_metrics across batches."""
        core_test_loop(module=mock_module_with_metrics, datamodule=mock_datamodule_multi_batch, limit_test_batches=2)
        # _logged_metrics should be cleared after epoch-end reporting
        assert mock_module_with_metrics._logged_metrics == {}
        # test_step was called twice (two batches)
        assert mock_module_with_metrics.test_step.call_count == 2

    def test_metrics_printed_at_epoch_end(self, mock_module_with_metrics, mock_datamodule_multi_batch, capsys):
        """core_test_loop prints averaged metrics at epoch end."""
        core_test_loop(module=mock_module_with_metrics, datamodule=mock_datamodule_multi_batch, limit_test_batches=2)
        captured = capsys.readouterr()
        assert "Test epoch end:" in captured.out
        assert "'accuracy'" in captured.out
        assert "'loss'" in captured.out

    def test_no_metrics_no_output(self, capsys):
        """When no metrics are logged, no epoch-end output is printed."""
        module = MagicMock()
        module.model = MagicMock()
        module.global_step = 0
        module._it_state = MagicMock()
        module.device = torch.device("cpu")
        module.dtype = torch.float32
        module._logged_metrics = {}
        module.test_step.return_value = None
        del module.on_test_batch_start
        del module.on_test_epoch_end

        datamodule = MagicMock()
        batch = {"input": torch.tensor([[1, 2]]), "labels": torch.tensor([0])}
        dl = MagicMock()
        dl.__iter__.return_value = [batch]
        datamodule.test_dataloader.return_value = dl

        core_test_loop(module=module, datamodule=datamodule, limit_test_batches=1)
        captured = capsys.readouterr()
        assert "Test epoch end:" not in captured.out

    def test_log_dict_accumulates_multiple_keys(self, mock_module_with_metrics, mock_datamodule_multi_batch):
        """log_dict dispatches to log for each key, accumulating properly."""

        # Replace test_step to use log_dict
        def test_step_with_dict(batch, batch_idx):
            mock_module_with_metrics.log_dict({"f1": 0.9, "precision": 0.88})
            return None

        mock_module_with_metrics.test_step = MagicMock(side_effect=test_step_with_dict)

        core_test_loop(module=mock_module_with_metrics, datamodule=mock_datamodule_multi_batch, limit_test_batches=2)
        # Metrics cleared after printing
        assert mock_module_with_metrics._logged_metrics == {}
