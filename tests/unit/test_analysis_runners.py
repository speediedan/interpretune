from __future__ import annotations
import pytest
import torch
from unittest.mock import MagicMock

from interpretune.runners.analysis import analysis_store_generator
import interpretune as it
from interpretune.config import AnalysisCfg
from interpretune.analysis.core import AnalysisStore
from tests.orchestration import run_analysis_operation
from interpretune.analysis.ops.base import AnalysisBatch


class TestAnalysisRunner:
    """Tests for the analysis runner module."""

    @pytest.mark.parametrize(
        "session_fixture, analysis_cfgs",
        [
            pytest.param("get_it_session__sl_gpt2_analysis__setup", (AnalysisCfg(output_schema=it.sae_correct_acts),)),
            pytest.param("get_analysis_session__sl_gpt2_logit_diffs_sae__setup_runanalysis", None),
        ],
        ids=[#"api_generated_step",
             "manual_step",
             "analysis_store_fixt"],
    )
    def test_basic_runner_mode_parity(self, request, session_fixture, analysis_cfgs):
        fixture = request.getfixturevalue(session_fixture)
        it_session, test_cfg = fixture.it_session, fixture.test_cfg()
        if (analysis_result := getattr(fixture, "result", None)) is None:
            if analysis_cfgs is not None:
                test_cfg.analysis_cfgs = analysis_cfgs
            analysis_result = run_analysis_operation(it_session, use_run_cfg=False, test_cfg=test_cfg)

        # Common validation for both fixture types
        assert isinstance(analysis_result, AnalysisStore)

    @pytest.fixture
    def mock_analysis_module(self):
        """Create a mock module for analysis testing."""
        module = MagicMock()
        module.analysis_step.return_value = [AnalysisBatch(logit_diffs=torch.tensor([0.5]))]
        return module

    @pytest.fixture
    def mock_analysis_datamodule(self):
        """Create a mock datamodule for analysis testing."""
        datamodule = MagicMock()
        batch = {"input": torch.tensor([[1, 2], [3, 4]]), "labels": torch.tensor([0, 1])}
        dataloader = MagicMock()
        dataloader.__iter__.return_value = [batch]
        datamodule.test_dataloader.return_value = dataloader
        return datamodule

    def test_analysis_store_generator(self, mock_analysis_module, mock_analysis_datamodule):
        """Test the analysis_store_generator function."""
        # Call the generator
        generator = analysis_store_generator(
            module=mock_analysis_module,
            datamodule=mock_analysis_datamodule,
            limit_analysis_batches=1,
            step_fn="analysis_step",
            max_epochs=1
        )

        # Consume the generator
        results = list(generator)

        # Verify that the generator yielded the expected results
        assert len(results) == 1
        assert isinstance(results[0], AnalysisBatch)
        assert torch.equal(results[0].logit_diffs, torch.tensor([0.5]))

        # Verify that the module was called as expected
        mock_analysis_module.analysis_step.assert_called_once()
        mock_analysis_module.on_analysis_epoch_end.assert_called_once()

        # Verify that the datamodule was used correctly
        mock_analysis_datamodule.test_dataloader.assert_called_once()

    # def test_core_analysis_loop_with_op(self, mock_analysis_module, mock_analysis_datamodule):
    #     """Test the core_analysis_loop with an operation set in the module configuration."""
    #     # Set up module with operation in analysis_cfg
    #     mock_analysis_module.analysis_cfg.op = "logit_diffs_base"
    #     mock_analysis_module.analysis_cfg.output_schema = {"logit_diffs": {"datasets_dtype": "float32"}}

    #     # Mock schema_to_features to return test features
    #     with patch("interpretune.analysis.schema_to_features") as mock_schema_to_features:
    #         mock_schema_to_features.return_value = {"logit_diffs": torch.ones(2)}

    #         # Call the analysis loop with mocked generator
    #         with patch("interpretune.runners.analysis.analysis_store_generator") as mock_generator:
    #             mock_generator.return_value = [
    #                 AnalysisBatch(logit_diffs=torch.tensor([0.5])),
    #                 AnalysisBatch(logit_diffs=torch.tensor([0.7])),
    #             ]

    #             # Run the analysis loop
    #             core_analysis_loop(
    #                 module=mock_analysis_module,
    #                 datamodule=mock_analysis_datamodule,
    #                 limit_analysis_batches=1,
    #                 max_epochs=1
    #             )

    #     # Verify hooks were called
    #     mock_analysis_module.on_analysis_start.assert_called_once()

    #     # Verify schema features were generated
    #     mock_schema_to_features.assert_called_once()

    # def test_core_analysis_loop_with_output_schema(self, mock_analysis_module, mock_analysis_datamodule):
    #     """Test the core_analysis_loop with output_schema set in the module configuration."""
    #     # Set up module with output_schema in analysis_cfg
    #     mock_analysis_module.analysis_cfg.op = None
    #     mock_analysis_module.analysis_cfg.output_schema = {"logit_diffs": {"datasets_dtype": "float32"}}

    #     # Mock schema_to_features to return test features
    #     with patch("interpretune.analysis.schema_to_features") as mock_schema_to_features:
    #         mock_schema_to_features.return_value = {"logit_diffs": torch.ones(2)}

    #         # Call the analysis loop with mocked generator
    #         with patch("interpretune.runners.analysis.analysis_store_generator") as mock_generator:
    #             mock_generator.return_value = [
    #                 AnalysisBatch(logit_diffs=torch.tensor([0.5])),
    #                 AnalysisBatch(logit_diffs=torch.tensor([0.7])),
    #             ]

    #             # Run the analysis loop
    #             core_analysis_loop(
    #                 module=mock_analysis_module,
    #                 datamodule=mock_analysis_datamodule,
    #                 limit_analysis_batches=1,
    #                 max_epochs=1
    #             )

    #     # Verify hooks were called
    #     mock_analysis_module.on_analysis_start.assert_called_once()

    #     # Verify schema features were generated
    #     mock_schema_to_features.assert_called_once()


class TestAnalysisRunnerWithAnalysisStore:
    """Tests for the analysis runner with actual analysis store integration."""

    @pytest.fixture
    def mock_module_with_store(self):
        """Create a mock module with analysis store capabilities."""
        module = MagicMock()

        # Configure the analysis_cfg with save_batch method
        module.analysis_cfg = MagicMock()

        # Create batches for testing
        batch1 = AnalysisBatch(
            logit_diffs=torch.tensor([0.5, 0.6]),
            labels=torch.tensor([0, 1])
        )
        batch2 = AnalysisBatch(
            logit_diffs=torch.tensor([0.7, 0.3]),
            labels=torch.tensor([1, 0])
        )

        # Configure analysis_step to return different batches
        def mock_analysis_step(*args, **kwargs):
            yield batch1
            yield batch2

        module.analysis_step = mock_analysis_step

        return module

    @pytest.fixture
    def mock_datamodule_multi_batch(self):
        """Create a mock datamodule with multiple batches."""
        datamodule = MagicMock()
        batch1 = {"input": torch.tensor([[1, 2], [3, 4]]), "labels": torch.tensor([0, 1])}
        batch2 = {"input": torch.tensor([[5, 6], [7, 8]]), "labels": torch.tensor([1, 0])}
        dataloader = MagicMock()
        dataloader.__iter__.return_value = [batch1, batch2]
        datamodule.test_dataloader.return_value = dataloader
        return datamodule

    def test_analysis_store_generator_multiple_batches(self, mock_module_with_store, mock_datamodule_multi_batch):
        """Test the analysis_store_generator with multiple batches."""
        # Call the generator
        generator = analysis_store_generator(
            module=mock_module_with_store,
            datamodule=mock_datamodule_multi_batch,
            limit_analysis_batches=2,
            step_fn="analysis_step",
            max_epochs=1
        )

        # Consume the generator
        results = list(generator)

        # Verify that the generator yielded the expected number of results
        # Each batch from analysis_step yields 2 batches, and we have 2 input batches
        assert len(results) == 4

        # Check properties of the batches
        assert torch.equal(results[0].logit_diffs, torch.tensor([0.5, 0.6]))
        assert torch.equal(results[1].logit_diffs, torch.tensor([0.7, 0.3]))
        assert torch.equal(results[2].logit_diffs, torch.tensor([0.5, 0.6]))
        assert torch.equal(results[3].logit_diffs, torch.tensor([0.7, 0.3]))
