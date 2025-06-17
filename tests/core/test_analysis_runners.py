from __future__ import annotations
import pytest
import torch
from unittest.mock import MagicMock, patch

from interpretune.runners.analysis import analysis_store_generator, generate_analysis_dataset, maybe_init_analysis_cfg
import interpretune as it
from interpretune.config import AnalysisCfg
from interpretune.base import _call_itmodule_hook
from interpretune.analysis.core import AnalysisStore
from tests.orchestration import run_analysis_operation
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.config.runner import AnalysisRunnerCfg


class TestAnalysisRunner:
    """Tests for the analysis runner module."""

    @pytest.mark.parametrize(
        "session_fixture, test_cfg_override_kwargs",
        [
            pytest.param("get_it_session__sl_gpt2_analysis__setup",
                         {'analysis_cfgs': [AnalysisCfg(output_schema=it.sae_correct_acts)]}),
            # we need to set ignore_manual=True at both the analysis_cfg and the test_cfg levels since we always want
            # test_cfg to override nested configs (analysis_cfg here) but also want to leverage an existing
            # fixture test_cfg in this case.
            pytest.param("get_it_session__sl_gpt2_analysis__setup",
                         {'analysis_cfgs': [AnalysisCfg(target_op=it.model_forward, ignore_manual=True)],
                          'ignore_manual': True}),
            pytest.param("get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis", {}),
        ],
        ids=["manual_step",
             "api_generated_step_with_op",
             "analysis_store_fixt"],
    )
    def test_basic_runner_mode_parity(self, request, session_fixture, test_cfg_override_kwargs):
        fixture = request.getfixturevalue(session_fixture)
        it_session, test_cfg = fixture.it_session, fixture.test_cfg()
        # Update test_cfg with valid attributes from test_cfg_override_kwargs
        for key, value in test_cfg_override_kwargs.items():
            if hasattr(test_cfg, key):
                setattr(test_cfg, key, value)
        if (analysis_result := getattr(fixture, "result", None)) is None:
            analysis_result = run_analysis_operation(it_session, use_run_cfg=False, test_cfg=test_cfg)

        # Common validation for both fixture types
        assert isinstance(analysis_result, AnalysisStore)

        # For the op-based test, verify the generated analysis step was created and used
        if test_cfg_override_kwargs.get("analysis_configs") and \
            hasattr(test_cfg_override_kwargs["analysis_configs"][0], "target_op") and \
                test_cfg_override_kwargs["analysis_configs"][0].target_op is not None:
            # Verify that the dynamically generated analysis step method exists
            assert hasattr(it_session.module, "_generated_analysis_step")
            # Verify the analysis_cfg op was properly set
            assert it_session.module.analysis_cfg.op is not None
            assert it_session.module.analysis_cfg.op.name == "model_forward"

    @pytest.fixture
    def mock_analysis_module(self, tmp_path):
        """Create a mock module for analysis testing."""
        # Use a normal MagicMock without spec for flexibility
        module = MagicMock()
        # Add analysis_step method
        module.analysis_step = MagicMock(return_value=[AnalysisBatch(logit_diffs=torch.tensor([0.5]))])

        # Create a real Path object in the temporary test directory
        # This ensures any operations on core_log_dir will be confined to the pytest-managed tmp_path
        core_log_dir = tmp_path / "core_log_dir"
        core_log_dir.mkdir(exist_ok=True)
        module.core_log_dir = core_log_dir

        # Set name attributes
        module.name = "mock_module"
        module.__class__.__name__ = "MockAnalysisModule"

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

    def test_maybe_init_analysis_cfg(self, mock_analysis_module):
        """Test the maybe_init_analysis_cfg function directly."""
        analysis_cfg = AnalysisCfg(name="test_analysis_cfg_direct", target_op=it.model_forward, ignore_manual=True)

        # Test with extra kwargs that should be filtered
        extra_kwargs = {"cache_dir": "/tmp/test", "unknown_param": "value"}
        result_kwargs = maybe_init_analysis_cfg(mock_analysis_module, analysis_cfg, **extra_kwargs)

        # Verify that module.analysis_cfg is set
        assert mock_analysis_module.analysis_cfg == analysis_cfg

        # Verify that known params are removed from kwargs
        assert "cache_dir" not in result_kwargs
        # But unknown params remain
        assert "unknown_param" in result_kwargs

    @patch("datasets.Dataset.from_generator")
    def test_generate_analysis_dataset_error(self, mock_from_generator, mock_analysis_module, mock_analysis_datamodule):
        """Test error handling in generate_analysis_dataset."""
        # Set up to simulate an error during dataset generation
        mock_from_generator.side_effect = ValueError("Test error")

        # Prepare module with needed attributes
        mock_analysis_module.analysis_cfg = MagicMock()
        mock_analysis_module.analysis_cfg.output_store = MagicMock()
        mock_analysis_module.analysis_cfg.output_store.cache_dir = "/tmp/test"

        # Mock handle_exception_with_debug_dump to avoid actual debug dumping
        with patch("interpretune.runners.analysis.handle_exception_with_debug_dump") as mock_handle_error:
            # Call the function and expect an error
            with pytest.raises(ValueError, match="Test error"):
                generate_analysis_dataset(
                    module=mock_analysis_module,
                    features={},
                    it_format_kwargs={},
                    gen_kwargs={"module": mock_analysis_module, "datamodule": mock_analysis_datamodule}
                )

            # Verify error handling was called
            mock_handle_error.assert_called_once()

    @patch.object(it.runners.analysis.AnalysisRunner, "__init__", return_value=None)
    def test_run_method_partialmethod(self, mock_init, request):
        """Test the _run method and analysis partialmethod in AnalysisRunner."""
        # Use a real session fixture to avoid pickling issues with MagicMock
        fixture = request.getfixturevalue("get_it_session__sl_gpt2_analysis__setup")
        it_session, _ = fixture.it_session, fixture.test_cfg()

        # Create a runner with the real session
        run_cfg_kwargs = {"it_session": it_session}
        run_cfg = AnalysisRunnerCfg(**run_cfg_kwargs)

        # Create a mock dataset to return from generate_analysis_dataset
        mock_dataset = MagicMock()
        mock_output_store = MagicMock()

        # Create runner with patched init
        runner = it.runners.analysis.AnalysisRunner.__new__(it.runners.analysis.AnalysisRunner)
        runner.phase = None
        runner.run_cfg = run_cfg
        runner.it_session_end = MagicMock()

        # Add a default analysis_cfg with an op to module to prevent the AttributeError
        if not hasattr(it_session.module, 'analysis_cfg') or it_session.module.analysis_cfg is None:
            it_session.module.analysis_cfg = MagicMock()
            it_session.module.analysis_cfg.op = MagicMock()

        # Setup output_store on the mock analysis_cfg
        it_session.module.analysis_cfg.output_store = mock_output_store
        mock_output_store.dataset = mock_dataset

        # Patch generate_analysis_dataset to avoid pickling MagicMock
        with patch("interpretune.runners.analysis.generate_analysis_dataset", return_value=mock_dataset):
            # Also patch save_to_disk on the dataset to avoid actual file operations
            with patch.object(mock_dataset, "save_to_disk"):
                # Patch the core analysis loop which is the first level function called
                with patch("interpretune.runners.analysis.core_analysis_loop") as mock_loop:
                    mock_loop.return_value = "test_result"

                    # Test _run method directly
                    result = runner._run(phase="analysis", loop_fn=mock_loop, step_fn="test_step")

                    # Verify phase was set to the correct AllPhases enum
                    from interpretune.protocol import AllPhases
                    assert runner.phase == AllPhases.analysis
                    assert result == "test_result"

                    # Test the analysis partialmethod
                    # For this test we need to use our own implementation to avoid the pickling issues
                    def direct_core_analysis_loop(*args, **kwargs):
                        # Run analysis start hooks
                        _call_itmodule_hook(it_session.module, hook_name="on_analysis_start",
                                           hook_msg="Running analysis start hooks")
                        # Run analysis end hooks
                        _call_itmodule_hook(it_session.module, hook_name="on_analysis_end",
                                           hook_msg="Running analysis end hooks")
                        return "test_result"

                    # Patch the core_analysis_loop with our direct implementation
                    with patch("interpretune.runners.analysis.core_analysis_loop",
                               side_effect=direct_core_analysis_loop):
                        result = runner.analysis(step_fn="test_step")
                        #

                    # Verify it_session_end was called both times
                    assert runner.it_session_end.call_count == 2

    @patch.object(it.runners.analysis.AnalysisRunner, "__init__", return_value=None)
    def test_run_analysis_with_single_config(self, mock_init):
        """Test run_analysis with a single analysis configuration."""
        # Create a runner instance directly
        runner = it.runners.analysis.AnalysisRunner.__new__(it.runners.analysis.AnalysisRunner)

        # Set up necessary attributes
        mock_module = MagicMock()
        mock_run_cfg = MagicMock()
        mock_run_cfg.module = mock_module
        mock_run_cfg.datamodule = MagicMock()
        mock_run_cfg._processed_analysis_cfgs = [AnalysisCfg(name="test_analysis", target_op=it.model_forward)]
        mock_run_cfg.cache_dir = None
        mock_run_cfg.op_output_dataset_path = None
        mock_run_cfg.sae_analysis_targets = None
        mock_run_cfg.ignore_manual = False
        runner.run_cfg = mock_run_cfg

        # Mock _run_analysis_cfg to return a test result
        with patch.object(runner, "_run_analysis_cfg") as mock_run_cfg:
            mock_run_cfg.return_value = "test_result"

            # Patch the init_analysis_cfgs function
            with patch("interpretune.runners.analysis.init_analysis_cfgs") as mock_init_cfgs:
                # Run analysis with overrides
                result = runner.run_analysis(
                    analysis_cfgs=AnalysisCfg(target_op=it.model_forward),
                    cache_dir="/tmp/test_cache",
                    op_output_dataset_path="/tmp/test_output"
                )

                # Verify the result is directly returned (not in a dict)
                assert result == "test_result"

                # Verify config updates were applied
                assert runner.run_cfg.cache_dir == "/tmp/test_cache"
                assert runner.run_cfg.op_output_dataset_path == "/tmp/test_output"

                # Verify that init_analysis_cfgs was called with the right parameters
                mock_init_cfgs.assert_called_once()

    @patch.object(it.runners.analysis.AnalysisRunner, "__init__", return_value=None)
    def test_run_analysis_with_multiple_configs(self, mock_init):
        """Test run_analysis with multiple analysis configurations."""
        # Create a runner instance directly
        runner = it.runners.analysis.AnalysisRunner.__new__(it.runners.analysis.AnalysisRunner)

        # Set up necessary attributes
        mock_module = MagicMock()
        mock_run_cfg = MagicMock()
        mock_run_cfg.module = mock_module
        mock_run_cfg.datamodule = MagicMock()
        mock_run_cfg._processed_analysis_cfgs = [
            AnalysisCfg(name="analysis1", target_op=it.model_forward),
            AnalysisCfg(name="analysis2", target_op=it.logit_diffs_base)
        ]
        mock_run_cfg.cache_dir = None
        mock_run_cfg.op_output_dataset_path = None
        mock_run_cfg.sae_analysis_targets = None
        mock_run_cfg.ignore_manual = False
        runner.run_cfg = mock_run_cfg
        runner.analysis_results = {}

        # Mock _run_analysis_cfg to return different results based on config
        def side_effect(cfg):
            return f"result_for_{cfg.name}"

        with patch.object(runner, "_run_analysis_cfg", side_effect=side_effect):
            # Patch the init_analysis_cfgs function
            with patch("interpretune.runners.analysis.init_analysis_cfgs"):
                # Run analysis
                results = runner.run_analysis()

                # Verify results are returned as a dictionary
                assert isinstance(results, dict)
                assert len(results) == 2
                assert results["analysis1"] == "result_for_analysis1"
                assert results["analysis2"] == "result_for_analysis2"

    @patch.object(it.runners.analysis.AnalysisRunner, "__init__", return_value=None)
    def test_run_analysis_with_no_configs(self, mock_init):
        """Test run_analysis with no analysis configurations."""
        # Create a runner instance directly
        runner = it.runners.analysis.AnalysisRunner.__new__(it.runners.analysis.AnalysisRunner)

        # Set up necessary attributes
        mock_module = MagicMock()
        mock_run_cfg = MagicMock()
        mock_run_cfg.module = mock_module
        mock_run_cfg.datamodule = MagicMock()
        mock_run_cfg._processed_analysis_cfgs = []
        runner.run_cfg = mock_run_cfg
        runner.analysis_results = {}

        # Patch the init_analysis_cfgs function
        with patch("interpretune.runners.analysis.init_analysis_cfgs"):
            # Expect ValueError when no configs are provided
            with pytest.raises(ValueError, match="No analysis configurations provided"):
                runner.run_analysis()

    def test_dataset_features_and_format_without_schema(self, mock_analysis_module):
        """Test dataset_features_and_format when no schema is defined (manual analysis step without schema)."""
        # Set up a mock analysis_cfg without op or output_schema
        mock_analysis_module.analysis_cfg = MagicMock()
        # Explicitly set op and output_schema to None to simulate manual analysis step without schema
        mock_analysis_module.analysis_cfg.op = None
        mock_analysis_module.analysis_cfg.output_schema = None

        # Call the function being tested
        from interpretune.runners.analysis import dataset_features_and_format
        features, it_format_kwargs, kwargs = dataset_features_and_format(mock_analysis_module, {})

        # Verify that features is an empty dict
        assert features == {}

        # Verify the format kwargs have an empty col_cfg
        assert it_format_kwargs == {"col_cfg": {}}

        # Verify that kwargs contains expected values
        assert "schema_source" in kwargs
        assert kwargs["schema_source"] == {}
        assert "serializable_col_cfg" in kwargs
        assert kwargs["serializable_col_cfg"] == {}

    @patch("interpretune.runners.analysis.rank_zero_warn")
    def test_analysis_runner_it_init_branches(self, mock_warn):
        """Test all branches of AnalysisRunner.it_init when phase='analysis'."""

        # Common setup
        mock_module = MagicMock()
        mock_run_cfg = MagicMock()
        mock_run_cfg.module = mock_module

        # Case 1: phase='analysis', no analysis_step, has analysis_cfg with op
        runner1 = it.runners.analysis.AnalysisRunner.__new__(it.runners.analysis.AnalysisRunner)
        runner1.phase = 'analysis'
        runner1.run_cfg = mock_run_cfg

        # Remove analysis_step attribute
        del mock_module.analysis_step

        # Set up analysis_cfg with op
        mock_module.analysis_cfg = MagicMock()
        mock_module.analysis_cfg.op = MagicMock()

        # Create and patch apply method
        mock_apply = MagicMock()
        mock_module.analysis_cfg.apply = mock_apply

        # Run it_init
        with patch("interpretune.runners.SessionRunner.it_init"):
            runner1.it_init()

        # Verify apply was called
        mock_apply.assert_called_once_with(mock_module)
        mock_warn.assert_not_called()

        # Reset mocks
        mock_warn.reset_mock()
        mock_apply.reset_mock()

        # Case 2: phase='analysis', has generated analysis_step, has analysis_cfg with op
        runner2 = it.runners.analysis.AnalysisRunner.__new__(it.runners.analysis.AnalysisRunner)
        runner2.phase = 'analysis'
        runner2.run_cfg = mock_run_cfg

        # Add analysis_step and set _generated_analysis_step flag
        mock_module.analysis_step = MagicMock()
        mock_module._generated_analysis_step = True

        # Run it_init
        with patch("interpretune.runners.SessionRunner.it_init"):
            runner2.it_init()

        # Verify apply was called (should regenerate the step)
        mock_apply.assert_called_once_with(mock_module)
        mock_warn.assert_not_called()

        # Reset mocks
        mock_warn.reset_mock()
        mock_apply.reset_mock()

        # Case 3: phase='analysis', no analysis_step, no analysis_cfg with op
        runner3 = it.runners.analysis.AnalysisRunner.__new__(it.runners.analysis.AnalysisRunner)
        runner3.phase = 'analysis'
        runner3.run_cfg = mock_run_cfg

        # Remove analysis_step
        del mock_module.analysis_step

        # Remove op from analysis_cfg
        del mock_module.analysis_cfg.op

        # Run it_init
        with patch("interpretune.runners.SessionRunner.it_init"):
            runner3.it_init()

        # Verify warning was issued and apply was not called
        mock_warn.assert_called_once()
        warning_msg = mock_warn.call_args[0][0]
        assert "has no analysis_step method" in warning_msg
        mock_apply.assert_not_called()

        # Reset mocks
        mock_warn.reset_mock()

        # Case 4: phase='analysis', has regular (non-generated) analysis_step
        runner4 = it.runners.analysis.AnalysisRunner.__new__(it.runners.analysis.AnalysisRunner)
        runner4.phase = 'analysis'
        runner4.run_cfg = mock_run_cfg

        # Add analysis_step but no _generated_analysis_step flag
        mock_module.analysis_step = MagicMock()
        if hasattr(mock_module, '_generated_analysis_step'):
            delattr(mock_module, '_generated_analysis_step')

        # Run it_init
        with patch("interpretune.runners.SessionRunner.it_init"):
            runner4.it_init()

        # Verify neither apply nor warning was called
        mock_apply.assert_not_called()
        mock_warn.assert_not_called()

        # Case 5: phase != 'analysis'
        runner5 = it.runners.analysis.AnalysisRunner.__new__(it.runners.analysis.AnalysisRunner)
        runner5.phase = 'train'  # Not 'analysis'
        runner5.run_cfg = mock_run_cfg

        # Remove analysis_step to make sure the check would trigger if it ran
        del mock_module.analysis_step

        # Run it_init
        with patch("interpretune.runners.SessionRunner.it_init"):
            runner5.it_init()

        # Verify neither apply nor warning was called - branch not entered
        mock_apply.assert_not_called()
        mock_warn.assert_not_called()
