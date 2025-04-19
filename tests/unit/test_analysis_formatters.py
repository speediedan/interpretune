from __future__ import annotations
import pytest
import torch
import numpy as np

from interpretune.analysis.formatters import OpSchemaExt, ITAnalysisFormatter


class TestOpSchemaExt:
    """Tests for the OpSchemaExt class."""

    def test_init_with_col_cfg(self):
        """Test initialization with column configuration."""
        col_cfg = {
            'field1': {'datasets_dtype': 'float32', 'dyn_dim': 1},
            'field2': {'datasets_dtype': 'int64', 'non_tensor': True},
            'field3': {'datasets_dtype': 'float32', 'per_latent': True}
        }

        schema_ext = OpSchemaExt(col_cfg=col_cfg)

        assert schema_ext.dyn_dims == {'field1': 1}
        assert schema_ext.non_tensor_fields == {'field2'}
        assert schema_ext.per_latent_fields == {'field3'}
        assert schema_ext._field_context == []

    def test_field_context_with_string(self):
        """Test field_context context manager with string field name."""
        col_cfg = {
            'field1': {'datasets_dtype': 'float32', 'dyn_dim': 1},
        }

        schema_ext = OpSchemaExt(col_cfg=col_cfg)

        with schema_ext.field_context('field1'):
            assert len(schema_ext._field_context) == 1
            assert schema_ext._field_context[0][0] == 'field1'
            assert schema_ext._field_context[0][1] == {'dyn_dim': 1}

        assert schema_ext._field_context == []

    def test_field_context_with_tuple(self):
        """Test field_context context manager with tuple field info."""
        schema_ext = OpSchemaExt()

        with schema_ext.field_context(('custom_field', {'custom_key': 'value'})):
            assert len(schema_ext._field_context) == 1
            assert schema_ext._field_context[0][0] == 'custom_field'
            assert schema_ext._field_context[0][1] == {'custom_key': 'value'}

        assert schema_ext._field_context == []

    def test_field_context_nested(self):
        """Test nested field_context context managers."""
        schema_ext = OpSchemaExt()

        with schema_ext.field_context('field1'):
            assert len(schema_ext._field_context) == 1
            with schema_ext.field_context('field2'):
                assert len(schema_ext._field_context) == 2
            assert len(schema_ext._field_context) == 1

        assert schema_ext._field_context == []

    def test_is_field_non_tensor(self):
        """Test is_field_non_tensor method."""
        col_cfg = {
            'non_tensor_field': {'datasets_dtype': 'str', 'non_tensor': True},
            'tensor_field': {'datasets_dtype': 'float32', 'non_tensor': False}
        }

        schema_ext = OpSchemaExt(col_cfg=col_cfg)

        assert schema_ext.is_field_non_tensor('non_tensor_field') is True
        assert schema_ext.is_field_non_tensor('tensor_field') is False
        assert schema_ext.is_field_non_tensor('unknown_field') is False

        # Test with field context
        with schema_ext.field_context('non_tensor_field'):
            assert schema_ext.is_field_non_tensor('tensor_field') is True

    def test_is_field_per_latent(self):
        """Test is_field_per_latent method."""
        col_cfg = {
            'per_latent_field': {'datasets_dtype': 'float32', 'per_latent': True},
            'normal_field': {'datasets_dtype': 'float32', 'per_latent': False}
        }

        schema_ext = OpSchemaExt(col_cfg=col_cfg)

        assert schema_ext.is_field_per_latent('per_latent_field') is True
        assert schema_ext.is_field_per_latent('normal_field') is False
        assert schema_ext.is_field_per_latent('unknown_field') is False

        # Test with field context
        with schema_ext.field_context('per_latent_field'):
            assert schema_ext.is_field_per_latent('normal_field') is True

    def test_handle_per_latent_dict(self):
        """Test handle_per_latent_dict method."""
        schema_ext = OpSchemaExt()

        # Valid per_latent dict
        test_dict = {
            'latents': [0, 1, 2],
            'per_latent': ['value1', 'value2', 'value3']
        }

        result = schema_ext.handle_per_latent_dict(test_dict, lambda x: f"processed_{x}")

        assert result == {0: 'processed_value1', 1: 'processed_value2', 2: 'processed_value3'}

        # Test with mismatched lengths
        invalid_dict = {
            'latents': [0, 1, 2],
            'per_latent': ['value1', 'value2']
        }

        with pytest.raises(ValueError, match="Mismatch in latents"):
            schema_ext.handle_per_latent_dict(invalid_dict, lambda x: x)

        # Test with non-per_latent dict structure
        normal_dict = {'key1': 'value1', 'key2': 'value2'}

        assert schema_ext.handle_per_latent_dict(normal_dict, lambda x: x) == normal_dict

    # def test_apply_dynamic_dimension(self):
    #     """Test apply_dynamic_dimension method."""
    #     col_cfg = {
    #         'dyn_field': {'datasets_dtype': 'float32', 'dyn_dim': 1},
    #         'normal_field': {'datasets_dtype': 'float32'}
    #     }

    #     schema_ext = OpSchemaExt(col_cfg=col_cfg)

    #     # Create a 3D tensor [2, 3, 4]
    #     tensor = torch.randn(2, 3, 4)

    #     # Apply dynamic dimension to dyn_field (should swap dims 0 and 1)
    #     result = schema_ext.apply_dynamic_dimension(tensor, 'dyn_field')
    #     assert result.shape == torch.Size([3, 2, 4])
    #     assert torch.allclose(result[0], tensor[0])

    #     # Test with field that doesn't have dyn_dim
    #     result = schema_ext.apply_dynamic_dimension(tensor, 'normal_field')
    #     assert result.shape == torch.Size([2, 3, 4])
    #     assert torch.equal(result, tensor)

    #     # Test with smaller tensor than dyn_dim
    #     small_tensor = torch.randn(2)
    #     result = schema_ext.apply_dynamic_dimension(small_tensor, 'dyn_field')
    #     assert result.shape == torch.Size([2])
    #     assert torch.equal(result, small_tensor)


class TestITAnalysisFormatter:
    """Tests for the ITAnalysisFormatter class."""

    def test_init(self):
        """Test initialization."""
        col_cfg = {
            'field1': {'datasets_dtype': 'float32', 'dyn_dim': 1},
            'field2': {'datasets_dtype': 'int64', 'non_tensor': True}
        }

        formatter = ITAnalysisFormatter(features=None, col_cfg=col_cfg)

        assert formatter.dyn_dims == {'field1': 1}
        assert formatter.non_tensor_fields == {'field2'}

    def test_tensorize_simple_types(self):
        """Test _tensorize with simple types."""
        formatter = ITAnalysisFormatter()

        # String should remain unchanged
        assert formatter._tensorize("test_string") == "test_string"

        # None should remain unchanged
        assert formatter._tensorize(None) is None

        # numpy string array should be converted to list
        np_str_array = np.array(["test1", "test2"])
        assert formatter._tensorize(np_str_array) == ["test1", "test2"]

    def test_tensorize_non_tensor_fields(self):
        """Test _tensorize with non_tensor fields."""
        col_cfg = {
            'non_tensor_field': {'datasets_dtype': 'float32', 'non_tensor': True}
        }

        formatter = ITAnalysisFormatter(col_cfg=col_cfg)

        # NumPy array with non_tensor field should be converted to list
        np_array = np.array([1, 2, 3])
        result = formatter._tensorize(np_array, 'non_tensor_field')
        assert isinstance(result, list)
        assert result == [1, 2, 3]

        # Regular value with non_tensor field should be left unchanged
        value = "test_value"
        result = formatter._tensorize(value, 'non_tensor_field')
        assert result == "test_value"

    def test_tensorize_dict(self):
        """Test _tensorize with dictionaries."""
        formatter = ITAnalysisFormatter()

        # Regular dict
        test_dict = {'key1': np.array([1, 2]), 'key2': np.array([3, 4])}
        result = formatter._tensorize(test_dict)

        assert isinstance(result, dict)
        assert all(isinstance(v, torch.Tensor) for v in result.values())
        assert torch.equal(result['key1'], torch.tensor([1, 2]))

        # Test with per_latent dict
        col_cfg = {
            'per_latent_field': {'datasets_dtype': 'float32', 'per_latent': True}
        }

        formatter = ITAnalysisFormatter(col_cfg=col_cfg)

        per_latent_dict = {
            'latents': [0, 1],
            'per_latent': [np.array([1, 2]), np.array([3, 4])]
        }

        with formatter.field_context('per_latent_field'):
            result = formatter._tensorize(per_latent_dict)

        assert isinstance(result, dict)
        assert 0 in result and 1 in result
        assert torch.equal(result[0], torch.tensor([1, 2]))
        assert torch.equal(result[1], torch.tensor([3, 4]))
