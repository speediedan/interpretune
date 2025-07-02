"""Tests for centralized parameter handling in analysis operations."""
import torch
from unittest.mock import MagicMock, patch

from interpretune.analysis.ops.base import (
    AnalysisOp, CompositeAnalysisOp, OpSchema, ColCfg, AnalysisBatch,
    DEFAULT_OP_PARAMS, DEFAULT_OP_PARAM_NAMES, build_call_args
)


class TestCentralizedParams:
    """Test centralized parameter handling functionality."""

    def test_default_op_params_constants(self):
        """Test that default parameter constants are properly defined."""
        # Check that constants exist and have expected structure
        assert isinstance(DEFAULT_OP_PARAMS, dict)
        assert isinstance(DEFAULT_OP_PARAM_NAMES, frozenset)

        # Check expected parameter names
        expected_params = {'module', 'analysis_batch', 'batch', 'batch_idx'}
        assert DEFAULT_OP_PARAM_NAMES == expected_params

        # Check that all default values are None
        for param_name in expected_params:
            assert param_name in DEFAULT_OP_PARAMS
            assert DEFAULT_OP_PARAMS[param_name] is None

    def test_build_call_args(self):
        """Test the build_call_args utility function."""
        module = MagicMock()
        analysis_batch = AnalysisBatch()
        batch = {"input": torch.tensor([1, 2, 3])}
        batch_idx = 5
        impl_params = {"threshold": 0.5, "mode": "test"}
        extra_kwargs = {"debug": True}

        # Test with all parameters
        result = build_call_args(
            module, analysis_batch, batch, batch_idx,
            impl_params=impl_params, **extra_kwargs
        )

        # Check that all default parameters are included
        assert result["module"] is module
        assert result["analysis_batch"] is analysis_batch
        assert result["batch"] is batch
        assert result["batch_idx"] == batch_idx

        # Check that impl_params are included
        assert result["threshold"] == 0.5
        assert result["mode"] == "test"

        # Check that extra kwargs are included
        assert result["debug"] is True

    def test_build_call_args_without_impl_params(self):
        """Test build_call_args when impl_params is None."""
        module = MagicMock()
        analysis_batch = AnalysisBatch()
        batch = {}
        batch_idx = 0

        result = build_call_args(module, analysis_batch, batch, batch_idx)

        # Should have all default parameters
        assert len([k for k in result.keys() if k in DEFAULT_OP_PARAM_NAMES]) == 4
        assert result["module"] is module
        assert result["analysis_batch"] is analysis_batch

    def test_build_call_args_kwargs_override_impl_params(self):
        """Test that kwargs can override impl_params in build_call_args."""
        impl_params = {"threshold": 0.5}
        kwargs = {"threshold": 0.8}

        result = build_call_args(
            None, None, None, None,
            impl_params=impl_params, **kwargs
        )

        # kwargs should override impl_params
        assert result["threshold"] == 0.8

    def test_analysis_op_uses_centralized_params(self):
        """Test that AnalysisOp properly uses centralized parameter handling."""
        def test_impl(module, analysis_batch, batch, batch_idx, custom_param=None):
            result = analysis_batch or AnalysisBatch()
            result.output = torch.tensor([custom_param or 0])
            return result

        op = AnalysisOp(
            name="test_centralized",
            description="Test centralized params",
            output_schema=OpSchema({"output": ColCfg(datasets_dtype="float32")}),
            impl_params={"custom_param": 42}
        )
        op._impl = test_impl

        # Call the operation
        result = op(MagicMock(), AnalysisBatch(), {}, 0)

        # Should have used the impl_params
        assert torch.equal(result.output, torch.tensor([42]))

    def test_composite_op_uses_centralized_params(self):
        """Test that CompositeAnalysisOp uses centralized parameter handling."""
        def impl1(module, analysis_batch, batch, batch_idx, value1=None):
            result = analysis_batch or AnalysisBatch()
            result.step1 = torch.tensor([value1 or 1])
            return result

        def impl2(module, analysis_batch, batch, batch_idx, value2=None):
            result = analysis_batch or AnalysisBatch()
            result.step2 = torch.tensor([value2 or 2])
            return result

        op1 = AnalysisOp(
            name="step1",
            description="First step",
            output_schema=OpSchema({"step1": ColCfg(datasets_dtype="float32")}),
            impl_params={"value1": 10}
        )
        op1._impl = impl1

        op2 = AnalysisOp(
            name="step2",
            description="Second step",
            output_schema=OpSchema({"step2": ColCfg(datasets_dtype="float32")}),
            impl_params={"value2": 20}
        )
        op2._impl = impl2

        # Mock the compilation function to avoid circular imports in test
        with patch('interpretune.analysis.ops.compiler.schema_compiler.jit_compile_composition_schema') as mock_compile:
            # Mock the compilation to return empty schemas
            mock_compile.return_value = (OpSchema({}), OpSchema({}))

            composite = CompositeAnalysisOp([op1, op2], name="test_composite")

            # Call the composite operation
            result = composite(MagicMock(), AnalysisBatch(), {}, 0)

            # Both operations should have executed with their impl_params
            assert torch.equal(result.step1, torch.tensor([10]))
            assert torch.equal(result.step2, torch.tensor([20]))

    def test_parameter_precedence(self):
        """Test parameter precedence: kwargs > impl_params > defaults."""
        def test_impl(module, analysis_batch, batch, batch_idx, param=None):
            result = analysis_batch or AnalysisBatch()
            result.param_value = torch.tensor([param])
            return result

        op = AnalysisOp(
            name="test_precedence",
            description="Test parameter precedence",
            output_schema=OpSchema({"param_value": ColCfg(datasets_dtype="float32")}),
            impl_params={"param": 100}
        )
        op._impl = test_impl

        # Test that kwargs override impl_params
        result = op(MagicMock(), AnalysisBatch(), {}, 0, param=200)
        assert torch.equal(result.param_value, torch.tensor([200]))

        # Test that impl_params are used when no kwargs override
        result = op(MagicMock(), AnalysisBatch(), {}, 0)
        assert torch.equal(result.param_value, torch.tensor([100]))

    def test_impl_params_renamed_from_impl_args(self):
        """Test that the parameter is now called impl_params instead of impl_args."""
        op = AnalysisOp(
            name="test_rename",
            description="Test impl_params rename",
            output_schema=OpSchema({"output": ColCfg(datasets_dtype="float32")}),
            impl_params={"test_param": "test_value"}
        )

        # Should have impl_params attribute
        assert hasattr(op, 'impl_params')
        assert op.impl_params == {"test_param": "test_value"}

        # Should not have the old impl_args attribute
        assert not hasattr(op, 'impl_args')
