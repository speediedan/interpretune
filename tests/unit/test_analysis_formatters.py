from __future__ import annotations
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from interpretune.analysis.formatters import OpSchemaExt, ITAnalysisFormatter
from interpretune.analysis import ColCfg
from interpretune.analysis.core import schema_to_features, OpSchema
from tests.data_generation import _generate_per_latent_data
from tests.utils import InOutComp

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

    def test_apply_dynamic_dimension(self):
        """Test apply_dynamic_dimension method."""
        col_cfg = {
            'dyn_field': {'datasets_dtype': 'float32', 'dyn_dim': 1},
            'normal_field': {'datasets_dtype': 'float32'}
        }

        schema_ext = OpSchemaExt(col_cfg=col_cfg)
        # Mock the features attribute to simulate feature schema info
        schema_ext.features = {
            'dyn_field': MagicMock(shape=(3, 4)),
            'normal_field': MagicMock(shape=(2, 3, 4))
        }

        # Create a 3D tensor [2, 3, 4]
        tensor = torch.randn(2, 3, 4)

        # Apply dynamic dimension to dyn_field (should swap dims 0 and 1 of the original field, keeping the first
        # dataset example dim unchanged)
        result = schema_ext.apply_dynamic_dimension(tensor, 'dyn_field')
        assert result.shape == torch.Size([2, 4, 3])
        # Verify dims 1 and 2 swapped for the first element
        assert torch.allclose(result[0], tensor[0].transpose(0, 1))

        # Test with field that doesn't have dyn_dim
        result = schema_ext.apply_dynamic_dimension(tensor, 'normal_field')
        assert result.shape == torch.Size([2, 3, 4])
        assert torch.equal(result, tensor)

        # Test with smaller tensor than dyn_dim
        small_tensor = torch.randn(2)
        result = schema_ext.apply_dynamic_dimension(small_tensor, 'dyn_field')
        assert result.shape == torch.Size([2])
        assert torch.equal(result, small_tensor)

        # Test with tensor shape having one less dimension raises error
        with pytest.raises(ValueError, match="Tensor dimension length mismatch"):
            all_examples_tensor = torch.randn(5, 2, 3, 4)  # Batch dimension added
            result = schema_ext.apply_dynamic_dimension(all_examples_tensor, 'dyn_field')


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

    def test_tensorize_lists_and_tuples(self):
        """Test _tensorize with lists and tuples."""
        # Create formatter with per_latent and non_tensor configurations
        col_cfg = {
            'per_latent_field': {'datasets_dtype': 'float32', 'per_latent': True},
            'non_tensor_field': {'datasets_dtype': 'float32', 'non_tensor': True},
            'normal_field': {'datasets_dtype': 'float32'},
            #'str_field': {'datasets_dtype': 'str', 'non_tensor': True}
        }
        formatter = ITAnalysisFormatter(col_cfg=col_cfg)

        # Test per_latent field with list
        list_value = [1, 2, 3]
        # TODO: make documentation more explicit about the fact that per_latent fields have a special structure
        # that currently does not support the non_tensor option and probably add a warning if the user tries to use it
        # TODO: add support for non_tensor per_latent fields
        with formatter.field_context('per_latent_field'):
            result = formatter._tensorize(list_value, 'per_latent_field')
        assert result == list_value  # Should return unchanged

        # Test non_tensor field with list
        with formatter.field_context('non_tensor_field'):
            result = formatter._tensorize(list_value, 'non_tensor_field')
        assert result == list_value  # Should return unchanged

        # Test list with dict/list elements
        complex_list = [{'a': 1}, [1, 2, 3]]
        result = formatter._tensorize(complex_list, 'normal_field')
        assert isinstance(result, list)
        assert isinstance(result[0]['a'], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

        # test that when upstream TorchFormatter._tensorize returns a non-tensor, we return it as is rather than attempt
        # to apply dynamic dimensions
        simple_list = [1, 2, 3]
        with patch('datasets.formatting.TorchFormatter._tensorize', return_value=simple_list):
            with formatter.field_context('normal_field'):
                result = formatter._tensorize(simple_list, 'normal_field')
        assert isinstance(result, list)

    def test_tensorize_per_latent(self, request, initialized_analysis_cfg):
        """Test _tensorize with lists and tuples."""
        fixture = request.getfixturevalue("get_it_session__sl_gpt2_analysis__setup")
        it_session, dim_vars = initialized_analysis_cfg(fixture)

        shared_defaults = {'datasets_dtype': 'float32', 'per_latent': True}
        arr_2d, arr_3d = ['batch_size', 'vocab_size'], ['batch_size', 'max_answer_tokens', 'num_classes']
        test_schema = {
            'per_latent_field_seq': ColCfg.from_dict(shared_defaults),
            'per_latent_field_2d': ColCfg.from_dict(shared_defaults | {'sequence_type': False, 'array_shape': arr_2d}),
            'per_latent_field_3d': ColCfg.from_dict(shared_defaults | {'sequence_type': False, 'array_shape': arr_3d}),
        }
        test_input_schema = OpSchema(test_schema)
        input_features = schema_to_features(module=it_session.module, schema=test_input_schema)

        serializable_col_cfg = {k: v.to_dict() for k, v in test_input_schema.items()}
        it_format_kwargs = dict(col_cfg=serializable_col_cfg, features=input_features)
        formatter = ITAnalysisFormatter(**it_format_kwargs)

        input_output = {}
        for field_name, col_cfg in test_input_schema.items():
            sample_value = _generate_per_latent_data(it_session.module, cfg=col_cfg, field=field_name, num_batches=1,
                                                     dim_vars=dim_vars)
            with formatter.field_context(field_name):
                input_output[field_name] = InOutComp(sample_value, formatter._recursive_tensorize(sample_value))

        # assert expected shared latent structure
        for key, value in input_output.items():
            assert isinstance(value.output, list)
            assert isinstance(value.output[0], dict)
            assert isinstance(value.output[0]['blocks.9.attn.hook_z.hook_sae_acts_post'], dict)

        def sample_input_output(field: str, hook_name: str='blocks.9.attn.hook_z.hook_sae_acts_post'):
            """Get the input and output for a specific field."""
            input_data, output_data = input_output[field]
            return input_data[0][hook_name]['per_latent'][0], output_data[0][hook_name][3]

        # assert expected variable latent structure
        seq_input_sample, seq_output_sample = sample_input_output('per_latent_field_seq')
        assert isinstance(seq_output_sample, torch.Tensor)
        assert seq_output_sample.size(0) == len(seq_input_sample)

        array_2d_input_sample, array_2d_output_sample = sample_input_output('per_latent_field_2d')
        assert isinstance(array_2d_output_sample, torch.Tensor)
        assert array_2d_output_sample.size(0) == len(array_2d_input_sample) == dim_vars['batch_size']
        assert array_2d_output_sample.size(1) == len(array_2d_input_sample[0]) == dim_vars['vocab_size']

        array_3d_input_sample, array_3d_output_sample = sample_input_output('per_latent_field_3d')
        assert isinstance(array_3d_output_sample, torch.Tensor)
        assert array_3d_output_sample.size(0) == len(array_3d_input_sample) == dim_vars['batch_size']
        assert array_3d_output_sample.size(1) == len(array_3d_input_sample[0]) == dim_vars['max_answer_tokens']
        assert array_3d_output_sample.size(2) == len(array_3d_input_sample[0][0]) == dim_vars['num_classes']

    def test_tensorize_parent_call(self):
        """Test that parent TorchFormatter._tensorize is called."""
        formatter = ITAnalysisFormatter()

        # Test with a numpy array that should be converted to tensor
        np_array = np.array([1.0, 2.0, 3.0])

        # Mock the parent method to verify it's called
        with patch('datasets.formatting.TorchFormatter._tensorize',
                   return_value=torch.tensor([1.0, 2.0, 3.0])) as mock_tensorize:
            result = formatter._tensorize(np_array, 'test_field')

            # Verify the parent method was called
            mock_tensorize.assert_called_once()
            assert torch.is_tensor(result)

    def test_recursive_tensorize_numpy_array(self):
        """Test _recursive_tensorize with numpy arrays."""
        formatter = ITAnalysisFormatter()

        # Test with regular numpy array
        np_array = np.array([1, 2, 3])
        result = formatter._recursive_tensorize(np_array)
        assert torch.is_tensor(result)
        assert torch.equal(result, torch.tensor([1, 2, 3]))

        # Test with object dtype numpy array
        obj_array = np.array([np.array([1, 2]), np.array([3, 4])], dtype=object)
        result = formatter._recursive_tensorize(obj_array)
        assert torch.is_tensor(result)
        assert result.shape == (2, 2)
        assert torch.equal(result, torch.tensor([[1, 2], [3, 4]]))
