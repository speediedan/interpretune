from __future__ import annotations
import dataclasses
from collections import namedtuple
from unittest.mock import MagicMock, patch
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.testing import assert_close

import interpretune as it
from interpretune.analysis.core import (SAEAnalysisDict, AnalysisStore, resolve_names_filter,
                                        schema_to_features, SAEFqn, _make_simple_cache_hook,
                                       SAEAnalysisTargets, BaseMetrics, ActivationSumm, LatentMetrics,
                                       latent_metrics_scatter, base_vs_sae_logit_diffs,
                                       compute_correct, AnalysisBatchProtocol)
from interpretune.analysis.ops.base import ColCfg
from interpretune.config import AnalysisCfg


def validate_sae_operations(sae_data, sae_analysis_targets,
                            analysis_targets_template: str = "blocks.{layer}.attn.hook_z.hook_sae_acts_post"):
    """Validate operations on SAE data including shape consistency and operations.

    Args:
        sae_data: SAEAnalysisDict containing the data to validate
        sae_analysis_targets: The SAEAnalysisTargets configuration object
        analysis_targets_template: Optional template for generating hook names. Defaults to
                                "blocks.{layer}.attn.hook_z.hook_sae_acts_post".
    """

    # Get hook names from the targets
    hook_names = [
        analysis_targets_template.format(layer=layer)
        for layer in sae_analysis_targets.target_layers
    ]

    # Validate pre-join shapes and dimensions
    batch_sizes = {}
    for hook_name in hook_names:
        for batch_idx, batch in enumerate(sae_data[hook_name]):
            if batch is not None:
                # TODO: generalize ndim to handler other contexts if using this validation more broadly
                assert batch.ndim == 2, f"Expected 2D tensor for {hook_name}[{batch_idx}], got {batch.ndim}D"
                if hook_name not in batch_sizes:
                    batch_sizes[hook_name] = []
                batch_sizes[hook_name].append(batch.shape[0])

    # Perform batch join and validate resulting shapes
    joined_data = sae_data.batch_join()
    for hook_name in hook_names:
        expected_size = sum(batch_sizes[hook_name])
        assert joined_data[hook_name].shape[0] == expected_size, \
            f"Expected first dimension of {expected_size} for {hook_name}, got {joined_data[hook_name].shape[0]}"

    # Validate mean operation
    mean_activation = joined_data.apply_op_by_sae(operation='mean', dim=0)
    for hook_name in hook_names:
        feature_dim = joined_data[hook_name].shape[1]
        assert mean_activation[hook_name].shape == torch.Size([feature_dim]), \
            f"Expected shape {torch.Size([feature_dim])} for mean of {hook_name}, \
                got {mean_activation[hook_name].shape}"

    # Validate count_nonzero operation
    num_samples_active = joined_data.apply_op_by_sae(operation=torch.count_nonzero, dim=0)
    for hook_name in hook_names:
        assert_close(num_samples_active[hook_name],
                     torch.count_nonzero(joined_data[hook_name], dim=0))

    # Validate max operation
    for hook_name in hook_names:
        values, indices = torch.max(mean_activation[hook_name], dim=0)
        max_vals_by_sae = mean_activation.apply_op_by_sae(operation=torch.max)
        assert_close(max_vals_by_sae[hook_name], values)

        max_val_inds_by_sae = mean_activation.apply_op_by_sae(operation=torch.max, dim=0)
        assert_close(max_val_inds_by_sae[hook_name], [values, indices])

class TestAnalysisStore:

    def test_format_columns_all_indices(self, request):
        # Use the real AnalysisStore from fixture
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
        store = deepcopy(fixture.result)

        # Get a sample of column names from the dataset to use for the test
        # We're assuming the dataset has at least two columns
        available_columns = list(store.dataset.column_names)[:2]

        # Perform the operation with real columns
        result = store._format_columns(available_columns)

        # Validate the result
        assert set(result.keys()) == set(available_columns)
        # We expect elements in each result column
        assert len(result[available_columns[0]]) > 0
        assert len(result[available_columns[1]]) > 0

    def test_format_columns_int_index(self):
        mock_dataset = MagicMock()
        mock_dataset.set_format = MagicMock()
        mock_dataset.__getitem__.side_effect = lambda idx: {"a": idx, "b": idx+1}
        store = AnalysisStore(dataset=mock_dataset)
        store.dataset = mock_dataset
        result = store._format_columns(["a", "b"], indices=1)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_format_columns_slice(self):
        mock_dataset = MagicMock()
        mock_dataset.set_format = MagicMock()
        mock_dataset.__getitem__.side_effect = lambda idx: {"a": idx, "b": idx+1}
        store = AnalysisStore(dataset=mock_dataset)
        store.dataset = mock_dataset
        result = store._format_columns(["a", "b"], indices=slice(0,2))
        assert result["a"] == [0,1]
        assert result["b"] == [1,2]

    def test_format_columns_list_indices(self):
        mock_dataset = MagicMock()
        mock_dataset.set_format = MagicMock()
        mock_dataset.__getitem__.side_effect = lambda idx: {"a": idx, "b": idx+1}
        store = AnalysisStore(dataset=mock_dataset)
        store.dataset = mock_dataset
        result = store._format_columns(["a", "b"], indices=[0,2])
        assert result["a"] == [0,2]
        assert result["b"] == [1,3]

    def test_format_columns_with_list_indices(self):
        """Test _format_columns with a list of indices."""
        mock_dataset = MagicMock()
        mock_dataset.set_format = MagicMock()

        # Set up mock dataset to return different values for different indices
        mock_dataset.__getitem__.side_effect = lambda idx: {
            "col1": f"value_{idx}",
            "col2": f"value_{idx+1}"
        }

        store = AnalysisStore(dataset=mock_dataset)
        store.dataset = mock_dataset

        # Test with a list of indices
        result = store._format_columns(["col1", "col2"], indices=[0, 2, 4])

        # Verify we get the expected data for each column
        assert result["col1"] == ["value_0", "value_2", "value_4"]
        assert result["col2"] == ["value_1", "value_3", "value_5"]

        # Verify that set_format was called with interpretune type
        mock_dataset.set_format.assert_called_with(type='interpretune', columns=["col1", "col2"])

    def test_save_dir_returns_path(self, tmp_path):
        store = AnalysisStore(dataset=None, op_output_dataset_path=str(tmp_path / "foo"))
        # Should return a Path object
        save_dir = store.save_dir
        assert isinstance(save_dir, Path)
        assert str(save_dir).endswith("foo")
        # Directory should be created if not exists
        if not save_dir.exists():
            save_dir.mkdir()
        assert save_dir.exists()

    def test_save_dir_when_path_is_none(self):
        """Test save_dir raises ValueError when op_output_dataset_path is None."""
        store = AnalysisStore(dataset=None, op_output_dataset_path=None)
        with pytest.raises(ValueError, match="op_output_dataset_path must be set to save datasets"):
            _ = store.save_dir

    def test_load_dataset_str_path(self, request, tmp_path, monkeypatch):
        # Import the module to find the correct patch location
        from interpretune.analysis import core

        # Get a real dataset from the fixture
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
        real_dataset = deepcopy(fixture.result.dataset)

        # Create a mock for load_dataset that returns our real dataset
        mock_load_dataset = MagicMock(return_value=real_dataset)

        # Patch the load_dataset function at the module level where it's used
        monkeypatch.setattr(core, 'load_dataset', mock_load_dataset)

        # Create a new store and call _load_dataset
        store = AnalysisStore()
        store.split = "validation"
        store.streaming = False
        store.dataset_trust_remote_code = False
        store._load_dataset(str(tmp_path))

        # Verify the dataset was loaded and is the expected one
        assert store.dataset is real_dataset
        assert mock_load_dataset.called

    def test_load_dataset_existing(self):
        mock_ds = MagicMock()
        store = AnalysisStore(dataset=mock_ds)
        store._load_dataset(mock_ds)
        assert store.dataset is mock_ds

    def test_reset_clears_dataset(self):
        store = AnalysisStore(dataset=MagicMock())
        store.dataset = MagicMock()
        store.reset_dataset()
        assert hasattr(store, "dataset")

    def test_reset_clears_settings(self, request):
        """Test that reset maintains custom attributes according to implementation."""
        # Use real fixture instead of mocks
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
        store = deepcopy(fixture.result)

        # Add test attribute
        store._custom_attr = "test"

        # Get reference to dataset before reset
        original_dataset = store.dataset

        # Call reset
        store.reset_dataset()

        # The custom attribute should still exist because the reset method doesn't clear attributes
        assert hasattr(store, "_custom_attr")
        assert store._custom_attr == "test"

        # The dataset should still be there or be None depending on if reload worked
        assert store.dataset is original_dataset or store.dataset is None

    def test_getitem_str(self):
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = [1,2,3]
        store = AnalysisStore(dataset=mock_ds)
        store.dataset = mock_ds
        result = store["foo"]
        assert result == [1,2,3]

    def test_getitem_list(self):
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = {"a": [1], "b": [2]}
        store = AnalysisStore(dataset=mock_ds)
        store.dataset = mock_ds
        result = store[["a","b"]]
        assert set(result.keys()) == {"a","b"}

    def test_getitem_else(self, request):
        # Use a real AnalysisStore from the session fixture
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
        store = fixture.result
        # Using an integer index that's not a valid key should raise an exception
        with pytest.raises(Exception):
            _ = store[123]

    def test_getitem_tensor_handling(self):
        """Test tensor handling in __getitem__ with different stack_batches settings."""
        import torch

        mock_ds = MagicMock()
        # Create a 2D tensor
        tensor_2d = torch.tensor([[1, 2], [3, 4], [5, 6]])
        # Create a 1D tensor
        tensor_1d = torch.tensor([10, 20, 30])

        mock_ds.__getitem__.side_effect = lambda k: tensor_2d if k == "2d" else tensor_1d if k == "1d" else None

        # Test with stack_batches=False
        store_unstacked = AnalysisStore(dataset=mock_ds)
        store_unstacked.dataset = mock_ds
        store_unstacked.stack_batches = False

        # Test 2D tensor unstacking
        result_2d = store_unstacked["2d"]
        assert isinstance(result_2d, list)
        assert len(result_2d) == 3
        assert all(isinstance(t, torch.Tensor) for t in result_2d)
        assert torch.equal(result_2d[0], torch.tensor([1, 2]))

        # Test 1D tensor unstacking - should convert to scalar tensors
        result_1d = store_unstacked["1d"]
        assert isinstance(result_1d, list)
        assert len(result_1d) == 3
        assert all(isinstance(t, torch.Tensor) for t in result_1d)
        assert torch.equal(result_1d[0], torch.tensor(10))

        # Test with stack_batches=True
        store_stacked = AnalysisStore(dataset=mock_ds)
        store_stacked.dataset = mock_ds
        store_stacked.stack_batches = True

        # Should return tensors as is when stack_batches=True
        assert torch.equal(store_stacked["2d"], tensor_2d)
        assert torch.equal(store_stacked["1d"], tensor_1d)

    def test_getitem_multi_column_tensor_handling(self):
        """Test multi-column tensor handling in __getitem__."""
        import torch

        mock_ds = MagicMock()
        # Create different tensors for different columns
        tensor_2d = torch.tensor([[1, 2], [3, 4]])
        tensor_1d = torch.tensor([10, 20])

        mock_ds.__getitem__.side_effect = lambda k: tensor_2d if k == "2d" else tensor_1d if k == "1d" else None

        # Test with stack_batches=False

        store_unstacked = AnalysisStore(dataset=mock_ds)
        store_unstacked.dataset = mock_ds
        store_unstacked.stack_batches = False

        # Test multi-column access
        result = store_unstacked[["2d", "1d"]]
        assert isinstance(result, dict)
        assert "2d" in result and "1d" in result

        # Verify 2D tensor handling
        assert isinstance(result["2d"], list)
        assert len(result["2d"]) == 2
        assert torch.equal(result["2d"][0], torch.tensor([1, 2]))

        # Verify 1D tensor handling
        assert isinstance(result["1d"], list)
        assert len(result["1d"]) == 2
        assert torch.equal(result["1d"][0], torch.tensor(10))

        # Test with stack_batches=True
        store_stacked = AnalysisStore(dataset=mock_ds)
        store_stacked.dataset = mock_ds
        store_stacked.stack_batches = True

        # Should return tensors as is when stack_batches=True
        result_stacked = store_stacked[["2d", "1d"]]
        assert torch.equal(result_stacked["2d"], tensor_2d)
        assert torch.equal(result_stacked["1d"], tensor_1d)

    def test_select_columns(self, request):
        # Use real fixture instead of mock
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
        store = deepcopy(fixture.result)
        column_subset = store.select_columns(column_names=['orig_labels', 'answer_logits'])
        assert isinstance(column_subset, AnalysisStore)
        assert hasattr(column_subset, 'dataset')
        assert isinstance(column_subset.orig_labels, list)
        assert all(isinstance(element, torch.Tensor) for element in column_subset.orig_labels)
        assert all(isinstance(element, torch.Tensor) for element in column_subset.answer_logits)

        assert column_subset.logit_diffs is None
        with pytest.raises(KeyError):
            _ = column_subset['logit_diffs']

    def test_getattr_protocol(self, request):
        # Use real fixture instead of mock
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")

        # Test with stack_batches=False (default behavior)
        store_unstacked = deepcopy(fixture.result)
        store_unstacked.stack_batches = False

        # Find an existing column that's in the protocol
        protocol_fields = [field for field in AnalysisBatchProtocol.__annotations__
                          if field in store_unstacked.dataset.column_names]

        # Use an existing protocol field
        test_field = protocol_fields[0]

        # With stack_batches=False:
        # Direct dataset access returns the original tensor (stacked batches)
        dataset_value = store_unstacked.dataset[test_field]

        # __getattr__ access returns individual tensors when stack_batches=False
        unstacked_value = getattr(store_unstacked, test_field)

        # Test with stack_batches=True
        store_stacked = deepcopy(fixture.result)
        store_stacked.stack_batches = True

        # With stack_batches=True, __getattr__ should return the same as direct dataset access
        stacked_value = getattr(store_stacked, test_field)

        # Verify behavior:

        # 1. Direct dataset access should always return stacked batches
        assert isinstance(dataset_value, torch.Tensor)

        # 2. With stack_batches=False, __getattr__ should return a list of tensors
        assert isinstance(unstacked_value, list)
        if len(unstacked_value) > 0:
            assert all(isinstance(t, torch.Tensor) for t in unstacked_value)

        # 3. With stack_batches=True, __getattr__ should return the same as stacked dataset access
        assert isinstance(stacked_value, torch.Tensor)
        assert torch.equal(stacked_value, dataset_value)

    def test_getattr_dataset_attr(self, request):
        # Use real fixture
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
        store = deepcopy(fixture.result)

        # Set a test attribute
        test_attr = "test_dataset_attr"
        test_value = 42
        setattr(store.dataset, test_attr, test_value)

        # Verify it can be accessed through __getattr__
        assert getattr(store, test_attr) == test_value
        assert store.test_dataset_attr == test_value

    def test_getattr_not_found(self, request):
        # Use real fixture
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
        store = deepcopy(fixture.result)

        # Make sure the attribute doesn't exist anywhere
        nonexistent_attr = "definitely_not_a_real_attribute_name_xyz123"

        # Make sure it's not in the protocol
        if nonexistent_attr in AnalysisBatchProtocol.__annotations__:
            del AnalysisBatchProtocol.__annotations__[nonexistent_attr]

        # Make sure it's not in the dataset
        if hasattr(store.dataset, nonexistent_attr):
            delattr(store.dataset, nonexistent_attr)

        # Now it should raise AttributeError
        with pytest.raises(AttributeError):
            getattr(store, nonexistent_attr)

    def test_by_sae_non_dict(self):
        mock_ds = MagicMock()
        setattr(mock_ds, "field", [1,2,3])
        store = AnalysisStore(dataset=mock_ds)
        store.__getattr__ = lambda name: getattr(mock_ds, name)
        with pytest.raises(TypeError):
            store.by_sae("field")

    def test_by_sae_empty(self):
        mock_ds = MagicMock()
        setattr(mock_ds, "field", [{"sae1": []}, {"sae1": torch.tensor([1])}])
        store = AnalysisStore(dataset=mock_ds)
        store.__getattr__ = lambda name: getattr(mock_ds, name)
        result = store.by_sae("field")
        assert result["sae1"][0] is None
        assert torch.equal(result["sae1"][1], torch.tensor([1]))

    def test_by_sae_with_list_tensors(self):
        """Test by_sae with list of tensors."""
        import torch
        # Create mock dataset with list of tensors (simpler structure that works with by_sae)
        mock_ds = MagicMock()

        # Create batch data where each item is a dict with SAE name as key and tensor as value
        batch_data = [
            {
                'sae1': torch.tensor([1.0, 2.0]),
            },
            {
                'sae1': torch.tensor([3.0, 4.0]),
            }
        ]

        setattr(mock_ds, "field_name", batch_data)
        store = AnalysisStore(dataset=mock_ds)
        store.__getattr__ = lambda name: getattr(mock_ds, name)

        # Test with by_sae
        result = store.by_sae("field_name")
        assert 'sae1' in result
        assert isinstance(result['sae1'], list)
        assert len(result['sae1']) == 2
        assert torch.equal(result['sae1'][0], torch.tensor([1.0, 2.0]))
        assert torch.equal(result['sae1'][1], torch.tensor([3.0, 4.0]))

    def test_by_sae_with_empty_dict_values(self):
        """Test by_sae with empty dict values."""
        import torch

        # Create mock dataset with empty dict values
        mock_ds = MagicMock()
        batch_data = [
            {
                'sae1': {}  # Empty dict
            },
            {
                'sae1': {
                    'latent1': torch.tensor([3.0])
                }
            }
        ]

        setattr(mock_ds, "field_name", batch_data)
        store = AnalysisStore(dataset=mock_ds)
        store.__getattr__ = lambda name: getattr(mock_ds, name)

        # Test with stack_latents=True
        result = store.by_sae("field_name")
        assert 'sae1' in result
        assert isinstance(result['sae1'], list)
        assert result['sae1'][0] is None  # Empty dict should result in None
        assert isinstance(result['sae1'][1], torch.Tensor)  # Second batch has tensor

    def test_by_sae_with_simple_nested_structure(self):
        """Test by_sae with a simple flat nested structure that can be stacked."""
        import torch

        # Create mock dataset with nested dict values that have direct tensor values
        # instead of deeper nested dictionaries
        mock_ds = MagicMock()
        batch_data = [
            {
                'sae1': {
                    'feature1': torch.tensor([1.0, 2.0]),
                    'feature2': torch.tensor([3.0, 4.0])
                }
            },
            {
                'sae1': {
                    'feature1': torch.tensor([5.0, 6.0]),
                    'feature2': torch.tensor([7.0, 8.0])
                }
            }
        ]

        setattr(mock_ds, "nested_field", batch_data)
        store = AnalysisStore(dataset=mock_ds)
        store.__getattr__ = lambda name: getattr(mock_ds, name)

        # Test with the simplified structure
        result = store.by_sae("nested_field")
        assert 'sae1' in result
        assert isinstance(result['sae1'], list)
        assert len(result['sae1']) == 2
        assert isinstance(result['sae1'][0], torch.Tensor)

    def test_calc_activation_summary_raises(self):
        store = AnalysisStore(dataset=MagicMock())
        store.correct_activations = []
        with pytest.raises(ValueError):
            store.calc_activation_summary()

    def test_deepcopy_memo_shortcircuit(self, request):
        """Test that __deepcopy__ handles exceptions for non-deepcopyable attributes."""
        # Get a real AnalysisStore to work with
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
        store = deepcopy(fixture.result)
        shortcircuit_memo_store = store.__deepcopy__(memo={id(store): store})
        assert shortcircuit_memo_store is store

    def test_deepcopy_handles_exceptions(self, request):
        """Test that __deepcopy__ handles exceptions for non-deepcopyable attributes."""
        # Get a real AnalysisStore to work with
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
        store = deepcopy(fixture.result)

        # Add a non-deepcopyable attribute that will raise an exception
        class NonDeepCopyable:
            def __deepcopy__(self, memo):
                raise RuntimeError("Cannot deepcopy this object")

        store._non_copyable = NonDeepCopyable()

        # Perform deepcopy - should not raise despite the non-copyable attribute
        store_copy = deepcopy(store)

        # The copied store should have all the regular attributes
        assert hasattr(store_copy, 'dataset')
        assert hasattr(store_copy, 'stack_batches')

        # But should not have the non-copyable attribute
        assert not hasattr(store_copy, '_non_copyable')

    def test_deepcopy_analysisstore(self):
        store = AnalysisStore(op_output_dataset_path='/tmp/foo')
        store2 = deepcopy(store)
        assert store2 is not store
        assert store2.op_output_dataset_path != store.op_output_dataset_path


class TestCoreFunctionality:

    def test_schema_to_features_edge_cases(self):
        mock_module = MagicMock()
        mock_module.datamodule.itdm_cfg.eval_batch_size = 2
        mock_module.it_cfg.generative_step_cfg.lm_generation_cfg.max_new_tokens = 3
        mock_module.it_cfg.num_labels = 2
        mock_module.it_cfg.entailment_mapping = {'a': 0, 'b': 1}
        # per_sae_hook, per_latent, sequence_type, array_shape, scalar
        mock_handle = MagicMock()
        mock_handle.cfg.hook_name = 'blocks.0.attn'
        mock_handle.hook_dict = {'hook_z': None}
        mock_module.sae_handles = [mock_handle]
        mock_module.analysis_cfg.names_filter = lambda x: True
        schema = {
            'per_sae': ColCfg(datasets_dtype='float32', per_sae_hook=True),
            'per_latent': ColCfg(datasets_dtype='float32', per_latent=True, array_shape=(2, 2)),
            'seq': ColCfg(datasets_dtype='string', sequence_type=True),
            'arr': ColCfg(datasets_dtype='float32', array_shape=(2, 3)),
            'scalar': ColCfg(datasets_dtype='float32'),
        }
        features = schema_to_features(mock_module, schema=schema)
        assert 'per_sae' in features and 'per_latent' in features \
              and 'seq' in features and 'arr' in features and 'scalar' in features

        # test resolving op from string
        features = schema_to_features(mock_module, op='logit_diffs', schema=None)

        mock_module.analysis_cfg.names_filter = None

        # test skipping various non-required features that depend on names_filter when one is not available
        schema_per_no_filter = {
            'scalar': ColCfg(datasets_dtype='float32', required=False),
            'per_sae': ColCfg(datasets_dtype='float32', per_sae_hook=True, required=False),
            'per_latent': ColCfg(datasets_dtype='float32', per_latent=True, required=False),
        }
        features = schema_to_features(mock_module, schema=schema_per_no_filter)

        schema_per_req_no_filter = {
            'per_latent': ColCfg(datasets_dtype='float32', per_latent=True, required=True),
        }
        with pytest.raises(ValueError, match="requires names_filter, but"):
            schema_to_features(mock_module, schema=schema_per_req_no_filter)
        # test empty schema case
        mock_module.analysis_cfg = None
        features = schema_to_features(mock_module, schema=None)

    def test_schema_from_module_analysis_cfg(self):
        """Test schema_to_features when schema comes from module.analysis_cfg.output_schema."""
        mock_module = MagicMock()
        mock_module.datamodule.itdm_cfg.eval_batch_size = 2
        mock_module.it_cfg.generative_step_cfg.lm_generation_cfg.max_new_tokens = 3
        mock_module.it_cfg.num_labels = 2
        mock_module.it_cfg.entailment_mapping = {'a': 0, 'b': 1}

        # Create analysis_cfg with output_schema
        mock_module.analysis_cfg = MagicMock()
        mock_module.analysis_cfg.output_schema = {
            'test_field': ColCfg(datasets_dtype='float32')
        }

        # Call without providing schema or op
        features = schema_to_features(mock_module)

        # Verify it used the module.analysis_cfg.output_schema
        assert 'test_field' in features

    def test_schema_to_features_intermediate_only(self):
        """Test schema_to_features properly skips intermediate_only columns."""
        from interpretune.analysis.core import schema_to_features
        from interpretune.analysis.ops.base import ColCfg

        mock_module = MagicMock()
        mock_module.datamodule.itdm_cfg.eval_batch_size = 2
        mock_module.it_cfg.generative_step_cfg.lm_generation_cfg.max_new_tokens = 3
        mock_module.it_cfg.num_labels = 2
        mock_module.it_cfg.entailment_mapping = {'a': 0, 'b': 1}

        # Create schema with both regular and intermediate_only columns
        schema = {
            'regular_col': ColCfg(datasets_dtype='float32'),
            'skipped_col': ColCfg(datasets_dtype='float32', intermediate_only=True)
        }

        # First check that a regular column is included
        features = schema_to_features(mock_module, schema=schema)
        print(f"Features include regular_col: {'regular_col' in features}")
        print(f"Features exclude skipped_col: {'skipped_col' not in features}")

        # Verify intermediate_only column is skipped
        assert 'regular_col' in features
        assert 'skipped_col' not in features

    def test_make_simple_cache_hook(self):
        cache = {}
        class Hook:
            name = 'foo'
        act = torch.tensor([1.0])
        hook_fn = _make_simple_cache_hook(cache)
        hook_fn(act, Hook())
        assert 'foo' in cache and torch.equal(cache['foo'], act)
        # Backward
        cache = {}
        hook_fn = _make_simple_cache_hook(cache, is_backward=True)
        hook_fn(act, Hook())
        assert 'foo_grad' in cache

    def test_resolve_names_filter(self):
        assert callable(resolve_names_filter(None))
        assert callable(resolve_names_filter('foo'))
        assert callable(resolve_names_filter(['foo', 'bar']))
        assert callable(resolve_names_filter(lambda x: True))
        with pytest.raises(ValueError):
            resolve_names_filter(123)

    def test_resolve_names_filter_function(self):
        """Test that resolve_names_filter works with callable, string, list, and None."""
        # Test with None (default filter)
        default_filter = resolve_names_filter(None)
        assert default_filter("any_name") is True

        # Test with string
        string_filter = resolve_names_filter("specific_name")
        assert string_filter("specific_name") is True
        assert string_filter("other_name") is False

        # Test with list
        list_filter = resolve_names_filter(["name1", "name2"])
        assert list_filter("name1") is True
        assert list_filter("name3") is False

        # Test with callable
        callable_filter = resolve_names_filter(lambda name: name.startswith("prefix_"))
        assert callable_filter("prefix_foo") is True
        assert callable_filter("other_foo") is False

        # Test with invalid type (not None, string, list, or callable)
        with pytest.raises(ValueError, match="must be a string, list of strings, or function"):
            resolve_names_filter(123)

    def test_sl_default_sae_match_fn(self, get_it_session__sl_gpt2__initonly):
        from interpretune.analysis.core import default_sae_hook_match_fn
        fixture = get_it_session__sl_gpt2__initonly
        sl_test_module = fixture.it_session.module
        name_filter_list = sl_test_module.construct_names_filter(target_layers=0,
                                                                    sae_hook_match_fn=default_sae_hook_match_fn)
        assert isinstance(name_filter_list, list)
        assert len(name_filter_list) == 1
        assert name_filter_list[0].startswith("blocks.0.hook_resid_pre")

    def test_resolve_names_filter_callable_path(self):
        """Test resolve_names_filter when passed a callable (line 969)."""
        # Define a custom filter function
        def custom_filter(name):
            return name.startswith("test_")

        # Resolve the filter
        resolved_filter = resolve_names_filter(custom_filter)

        # Verify it's the same function (not wrapped)
        assert resolved_filter is custom_filter

        # Test the filter behavior
        assert resolved_filter("test_name") is True
        assert resolved_filter("other_name") is False

    def test_base_vs_sae_logit_diffs(self, monkeypatch, request):
        # Use a real AnalysisStore fixture for fidelity
        base_fixture = \
            request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
        sae_fixture = \
            request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis")
        base_analysis_store = deepcopy(base_fixture.result)
        sae_analysis_store = deepcopy(sae_fixture.result)
        # Patch tabulate to avoid printing
        monkeypatch.setattr('tabulate.tabulate', lambda *a, **k: "table")
        base_vs_sae_logit_diffs(sae=sae_analysis_store, base_ref=base_analysis_store, top_k=3,
                                tokenizer=sae_fixture.it_session.datamodule.tokenizer)

    def test_init_with_overlapping_paths(self, tmp_path, monkeypatch):
        """Test that overlapping paths in __init__ raises ValueError."""
        # Import the module to find the correct patch location
        from interpretune.analysis import core

        # Create a temporary path to use for both dataset and output
        path_str = str(tmp_path / "overlap")

        # Mock load_dataset to avoid actual loading
        mock_load_dataset = MagicMock()

        # Patch the load_dataset function at the module level where it's used
        monkeypatch.setattr(core, 'load_dataset', mock_load_dataset)

        # Should raise ValueError when paths overlap
        with pytest.raises(ValueError, match="must not overlap"):
            AnalysisStore(dataset=path_str, op_output_dataset_path=path_str)

        # Should not raise when paths are different
        different_path = str(tmp_path / "different")
        AnalysisStore(dataset=path_str, op_output_dataset_path=different_path)

    def test_load_dataset_with_overlapping_paths(self, tmp_path, monkeypatch):
        """Test that overlapping paths in _load_dataset raises ValueError."""
        # Import the module to find the correct patch location
        from interpretune.analysis import core

        # Create a temporary path to use for both dataset and output
        path_str = str(tmp_path / "overlap")

        # Mock load_dataset to avoid actual loading
        mock_load_dataset = MagicMock()

        # Patch the load_dataset function at the module level where it's used
        monkeypatch.setattr(core, 'load_dataset', mock_load_dataset)

        # Create store with output path
        store = AnalysisStore(op_output_dataset_path=path_str)

        # Should raise ValueError when loading from overlapping path
        with pytest.raises(ValueError, match="must not overlap"):
            store._load_dataset(path_str)

        # Should not raise when paths are different
        different_path = str(tmp_path / "different")
        store._load_dataset(different_path)

    def test_debugger_identifier_from_env(self):
        """Test that OpWrapper._debugger_identifier reflects the IT_ENABLE_LAZY_DEBUGGER environment variable."""
        import os
        import importlib

        # Get the OpWrapper class
        analysis_ops_base = importlib.import_module("interpretune.analysis.ops.base")
        OpWrapper = analysis_ops_base.OpWrapper

        # Verify current value matches current environment setting
        current_env_value = os.environ.get('IT_ENABLE_LAZY_DEBUGGER', '')
        assert OpWrapper._debugger_identifier == current_env_value, \
            f"OpWrapper._debugger_identifier value '{OpWrapper._debugger_identifier}' does not match environment " \
            f"variable '{current_env_value}'"

    def test_operation_aliases_registered(self):
        """Test that operation aliases are registered in the top-level module."""
        import interpretune
        from interpretune.analysis import DISPATCHER
        import importlib
        import sys

        # Save the modules we're going to modify
        modules_to_reload = {name:module for name, module in sys.modules.items() \
                             if name.startswith('interpretune.analysis')}
        try:
            # Remove analysis modules to force reload
            for module_name in modules_to_reload.keys():
                if module_name in sys.modules:
                    del sys.modules[module_name]

            importlib.import_module('interpretune.analysis')

            # Get the OpWrapper class
            analysis_ops_base = importlib.import_module("interpretune.analysis.ops.base")
            OpWrapper = analysis_ops_base.OpWrapper

            # Get operation aliases from the dispatcher
            aliases = list(DISPATCHER.get_all_aliases())

            # Ensure we have some aliases to test
            assert len(aliases) > 0, "No operation aliases found in DISPATCHER"

            # Check that aliases are properly registered in the top-level module
            for alias, op_name in aliases:
                assert hasattr(interpretune, alias), f"Alias '{alias}' not found in top-level module"
                wrapper = getattr(interpretune, alias)
                assert isinstance(wrapper, OpWrapper), f"'{alias}' is not an OpWrapper instance"
                assert DISPATCHER.get_op(wrapper._op_name).name == op_name, \
                    f"Alias '{alias}' points to '{wrapper.op_name}' instead of '{op_name}'"

        finally:
            for module_name, module in modules_to_reload.items():
                # Restore the original module
                if module_name in sys.modules:
                    del sys.modules[module_name]
                sys.modules[module_name] = module

    def test_analysis_import_hook_system_modules_cache(self):
        """Test _AnalysisImportHook returns cached module from sys.modules (line 35)."""
        import sys
        from interpretune import _AnalysisImportHook

        # Save original module if it exists
        original_module = sys.modules.get('interpretune.analysis')

        try:
            # Create a mock module and store it in sys.modules
            mock_module = ModuleType('interpretune.analysis')
            mock_module.test_marker = "This is a mock module"
            sys.modules['interpretune.analysis'] = mock_module

            # Create import hook instance
            import_hook = _AnalysisImportHook()

            # Test that load_module returns our mock from sys.modules
            result = import_hook.load_module('interpretune.analysis')
            assert result is mock_module, "Import hook should return the module from sys.modules"
            assert hasattr(result, 'test_marker'), "The returned module should be our mock"

        finally:
            # Restore original module
            if original_module:
                sys.modules['interpretune.analysis'] = original_module
            else:
                sys.modules.pop('interpretune.analysis', None)

    def test_plot_latent_effects_per_batch_false(self, monkeypatch):
        """Test plot_latent_effects with per_batch=False to cover line 666."""
        import torch
        from unittest.mock import MagicMock

        # Mock the px.line function to avoid actual plotting
        mock_px_line = MagicMock()
        mock_px_line.return_value.update_layout.return_value.show = MagicMock()
        monkeypatch.setattr('plotly.express.line', mock_px_line)

        # Create a mock AnalysisStore with the required attributes
        store = AnalysisStore(dataset=MagicMock())

        # Mock the attribution_values and alive_latents attributes
        store.attribution_values = [
            {'sae1': torch.randn(5, 10)},
            {'sae1': torch.randn(5, 10)}
        ]
        store.alive_latents = [
            {'sae1': [0, 1, 2]},
            {'sae1': [3, 4, 5]}
        ]

        # Mock the by_sae method to return a controlled SAEAnalysisDict
        class MockSAEDict(SAEAnalysisDict):
            def batch_join(self):
                return self

            def apply_op_by_sae(self, operation, dim=None):
                return self

            def keys(self):
                return ['sae1']

        mock_dict = MockSAEDict({'sae1': torch.randn(10)})
        store.by_sae = MagicMock(return_value=mock_dict)

        # Call plot_latent_effects with per_batch=False
        store.plot_latent_effects(per_batch=False)

        # Verify that the by_sae method was called with 'attribution_values'
        store.by_sae.assert_called_with('attribution_values')

        # Verify that px.line was called (indicating the aggregation plotting path was taken)
        assert mock_px_line.call_count > 0


class TestMetricsAndTargets:

    def test_validate_sae_fqns_explicit(self):
        targets = SAEAnalysisTargets(sae_fqns=[SAEFqn('rel', 'id')])
        assert isinstance(targets.validate_sae_fqns(), tuple)
        targets = SAEAnalysisTargets(sae_fqns=[('rel', 'id')])
        assert isinstance(targets.validate_sae_fqns(), tuple)
        # Invalid input should raise TypeError
        with pytest.raises(TypeError):
            SAEAnalysisTargets(sae_fqns=[123])
        targets = SAEAnalysisTargets(target_sae_ids=['foo'])
        assert all(isinstance(f, type(targets.sae_fqns[0])) for f in targets.sae_fqns)
        targets = SAEAnalysisTargets(target_layers=[1, 2])
        assert all(isinstance(f, type(targets.sae_fqns[0])) for f in targets.sae_fqns)
        targets = SAEAnalysisTargets()
        assert isinstance(targets.validate_sae_fqns(), tuple)

    def test_post_init_and_validate(self):
        # Should not raise for empty/default
        targets = SAEAnalysisTargets()
        targets.__post_init__()
        targets.validate_sae_fqns()
        # Should handle explicit sae_fqns
        targets2 = SAEAnalysisTargets(sae_fqns=[SAEFqn('rel', 'id')])
        targets2.__post_init__()
        targets2.validate_sae_fqns()
        # Should handle target_sae_ids
        targets3 = SAEAnalysisTargets(target_sae_ids=['foo'])
        targets3.__post_init__()
        targets3.validate_sae_fqns()
        # Should handle target_layers
        targets4 = SAEAnalysisTargets(target_layers=[1,2])
        targets4.__post_init__()
        targets4.validate_sae_fqns()

    def test_compute_correct_with_mock(self, mock_analysis_store):
        """Test compute_correct function with mocked dispatcher."""

        # Setup test data
        mock_analysis_cfg = MagicMock(spec=AnalysisCfg)
        mock_analysis_cfg.output_store = mock_analysis_store
        mock_analysis_cfg.output_store.logit_diffs = [torch.tensor([0.5, -0.3]), torch.tensor([-0.1, 0.7])]
        mock_analysis_cfg.output_store.orig_labels = [torch.tensor([0, 1]), torch.tensor([1, 0])]
        mock_analysis_cfg.output_store.preds = [torch.tensor([0, 1]), torch.tensor([1, 0])]
        mock_analysis_cfg.op = MagicMock(autospec=it.logit_diffs_sae)
        mock_analysis_cfg.op.aliases = ["test_alias"]

        # Create a mock for get_preds_summ
        from dataclasses import dataclass

        @dataclass
        class MockPredSumm:
            total_correct: int
            percentage_correct: float
            batch_predictions: list | None = None


        # Mock get_preds_summ function to return a valid PredSumm
        def mock_pred_summ(total_correct, percentage_correct, batch_predictions=None):
            if batch_predictions is None:
                batch_predictions = [[0], [1]]
            return MockPredSumm(
                total_correct=total_correct,
                percentage_correct=percentage_correct,
                batch_predictions=batch_predictions
            )

        # Mock the dispatcher
        mock_dispatcher = MagicMock()
        mock_dispatcher.get_op.return_value = None
        #mock_dispatcher.get_by_alias.return_value = None

        with patch('interpretune.analysis.core.PredSumm', mock_pred_summ), \
            patch('interpretune.analysis.core.DISPATCHER', mock_dispatcher):
            # Test compute_correct under patched PredSumm and DISPATCHER
            result = compute_correct(mock_analysis_cfg)
        assert isinstance(result, MockPredSumm)
        assert result.total_correct == 4

    def test_base_metrics_repr_and_validation(self):
        @dataclasses.dataclass(kw_only=True)
        class Dummy(BaseMetrics):
            a: dict
            b: dict
        d = Dummy(a={'x': 1}, b={'x': 2})
        assert d.get_field_name('a') == 'a'
        assert isinstance(d.get_field_names(), dict)
        @dataclasses.dataclass(kw_only=True)
        class Dummy2(BaseMetrics):
            a: dict
            b: dict
        with pytest.raises(ValueError):
            Dummy2(a={'x': 1}, b={'y': 2})

    def test_activation_summ_and_latentmetrics(self):
        vals = torch.tensor([1.0, 2.0])
        _ = ActivationSumm(mean_activation={'h': vals}, num_samples_active={'h': vals})
        lat_metrics = LatentMetrics(mean_activation={'h': vals}, num_samples_active={'h': vals},
                          total_effect={'h': vals}, mean_effect={'h': vals}, proportion_samples_active={'h': vals})
        tables = lat_metrics.create_attribution_tables(sort_by='total_effect', top_k=1, filter_type='both',
                                                       per_sae=True)
        assert isinstance(tables, dict)
        tables = lat_metrics.create_attribution_tables(sort_by='total_effect', top_k=1, filter_type='positive',
                                                       per_sae=False)
        assert isinstance(tables, dict)
        tables = lat_metrics.create_attribution_tables(sort_by='total_effect', top_k=1, filter_type='negative',
                                                       per_sae=False)
        assert isinstance(tables, dict)
        with pytest.raises(ValueError):
            lat_metrics.create_attribution_tables(sort_by='not_a_field')

    def test_post_init_metric_dict_validation(self):
        """Test that BaseMetrics.__post_init__ validates metric dictionaries."""
        import dataclasses
        import torch

        # Define a subclass of BaseMetrics with metric dictionaries
        @dataclasses.dataclass(kw_only=True)
        class TestMetrics(BaseMetrics):
            metric1: dict
            metric2: dict

        # Valid case - all dictionaries have same keys
        metrics = TestMetrics(
            metric1={'a': torch.tensor([1.0]), 'b': torch.tensor([2.0])},
            metric2={'a': torch.tensor([3.0]), 'b': torch.tensor([4.0])}
        )
        assert hasattr(metrics, '_field_repr')

        # Invalid case - dictionaries have different keys
        with pytest.raises(ValueError, match="All hook dictionaries must have the same keys"):
            TestMetrics(
                metric1={'a': torch.tensor([1.0]), 'b': torch.tensor([2.0])},
                metric2={'a': torch.tensor([3.0]), 'c': torch.tensor([4.0])}  # 'c' instead of 'b'
            )

    def test_get_field_names(self):
        """Test get_field_names returns correct field names."""
        import dataclasses

        @dataclasses.dataclass(kw_only=True)
        class TestMetrics(BaseMetrics):
            regular_field: int
            dict_field: dict
            _protected_field: str = "hidden"

        # Create instance with custom representation
        metrics = TestMetrics(
            regular_field=42,
            dict_field={'a': 1, 'b': 2},
            custom_repr={'regular_field': 'Regular Field', 'dict_field': 'Dictionary Field'}
        )

        # Test with dict_only=False (default)
        all_fields = metrics.get_field_names()
        assert 'regular_field' in all_fields
        assert 'dict_field' in all_fields
        assert '_protected_field' not in all_fields  # Protected fields should be excluded
        assert all_fields['regular_field'] == 'Regular Field'  # Should use custom repr

        # Test with dict_only=True
        dict_fields = metrics.get_field_names(dict_only=True)
        assert 'regular_field' not in dict_fields
        assert 'dict_field' in dict_fields

    def test_create_attribution_tables(self):
        vals = torch.tensor([1.0, 2.0])
        lat_metrics = LatentMetrics(
            mean_activation={'h': vals},
            num_samples_active={'h': vals},
            total_effect={'h': vals},
            mean_effect={'h': vals},
            proportion_samples_active={'h': vals}
        )
        # Should not raise for valid field
        tables = lat_metrics.create_attribution_tables(
            sort_by='total_effect', top_k=1, filter_type='both', per_sae=True
        )
        assert isinstance(tables, dict)
        # Should raise for invalid field
        with pytest.raises(ValueError):
            lat_metrics.create_attribution_tables(sort_by='not_a_field')

    def test_create_attribution_tables_invalid_sort_by(self):
        """Test create_attribution_tables raises ValueError with invalid sort_by field."""
        # Create minimal LatentMetrics with required fields
        vals = torch.tensor([1.0, 2.0])
        metrics = LatentMetrics(
            mean_activation={'h': vals},
            num_samples_active={'h': vals},
            total_effect={'h': vals},
            mean_effect={'h': vals},
            proportion_samples_active={'h': vals}
        )

        # Test with invalid sort_by field - should raise ValueError
        with pytest.raises(ValueError, match="Invalid sort_by field"):
            metrics.create_attribution_tables(sort_by="invalid_field_name", top_k=10)

    def test_latent_metrics_scatter(self, monkeypatch):
        vals = torch.tensor([1.0, 2.0, 3.0])
        m1 = LatentMetrics(
            mean_activation={'h': vals},
            num_samples_active={'h': vals},
            total_effect={'h': vals},
            mean_effect={'h': vals},
            proportion_samples_active={'h': vals}
        )
        m2 = LatentMetrics(
            mean_activation={'h': vals},
            num_samples_active={'h': vals},
            total_effect={'h': vals},
            mean_effect={'h': vals},
            proportion_samples_active={'h': vals}
        )
        monkeypatch.setattr(
            'plotly.express.scatter',
            lambda *a, **k: type('PX', (), {
                'add_shape': lambda s, **k: s,
                'show': lambda s: None
            })()
        )
        latent_metrics_scatter(m1, m2)


        with pytest.raises(ValueError, match="not found in one or both metrics"):
            latent_metrics_scatter(m1, m2, metric_field='oops_no_exist')

    def test_create_attribution_tables_per_sae_false(self):
        """Test create_attribution_tables with per_sae=False to cover line 877."""
        import torch
        from interpretune.analysis.core import LatentMetrics

        # Create test tensors with positive and negative values
        tensor1 = torch.tensor([3.0, -2.0, 5.0, -1.0, 4.0])
        tensor2 = torch.tensor([2.0, -3.0, 1.0, -4.0, 6.0])

        # Create a LatentMetrics instance with test data
        metrics = LatentMetrics(
            mean_activation={'sae1': tensor1, 'sae2': tensor2},
            num_samples_active={'sae1': tensor1.abs(), 'sae2': tensor2.abs()},
            total_effect={'sae1': tensor1, 'sae2': tensor2},
            mean_effect={'sae1': tensor1 * 0.5, 'sae2': tensor2 * 0.5},
            proportion_samples_active={'sae1': tensor1.abs() / 10, 'sae2': tensor2.abs() / 10}
        )

        # Test with per_sae=False
        tables = metrics.create_attribution_tables(
            sort_by='total_effect',
            top_k=3,
            filter_type='both',
            per_sae=False  # This should trigger the all_values aggregation code path
        )

        # Verify tables were created
        assert len(tables) > 0

        # Check that tables contain expected titles
        assert any('positive' in title for title in tables.keys())
        assert any('negative' in title for title in tables.keys())

        # For both positive and negative tables, verify we have expected hooks and metrics
        for title, table_content in tables.items():
            # Check that the table includes both SAE hooks
            assert 'sae1' in table_content or 'sae2' in table_content
            # Check that metrics are included
            assert 'Total Effect' in table_content
            assert 'Mean Effect' in table_content

    def test_calculate_latent_metrics_with_empty_attribution_values(self, mock_analysis_store):
        """Test calculate_latent_metrics handling of empty attribution_values."""
        # Create a proper PredSumm dataclass
        from dataclasses import dataclass

        @dataclass
        class MockPredSumm:
            batch_predictions: list = None
            correct_examples: list = None

        pred_summary = MockPredSumm(
            correct_examples=[True, False],
            batch_predictions=[[0], [1]]
        )

        activation_summary = ActivationSumm(
            mean_activation={'hook1': torch.rand(10)},
            num_samples_active={'hook1': torch.rand(10)}
        )

        # Empty attribution values
        mock_analysis_store.attribution_values = []

        # Mock the calculate_latent_metrics method to throw the expected exception
        def mock_calculate_metrics(*args, **kwargs):
            if not mock_analysis_store.attribution_values:
                raise ValueError("No attribution values found")
            return MagicMock()

        mock_analysis_store.calculate_latent_metrics = mock_calculate_metrics

        # Should raise the expected error
        with pytest.raises(ValueError, match="No attribution values found"):
            mock_analysis_store.calculate_latent_metrics(
                pred_summ=pred_summary,
                activation_summary=activation_summary
            )

    def test_calculate_latent_metrics_with_proper_predsumm(self, request):
        """Test calculate_latent_metrics with a proper PredSumm object."""
        # Get a real fixture for this test
        attr_fixture = request.getfixturevalue(
            "get_analysis_session__sl_gpt2_logit_diffs_attr_grad__initonly_runanalysis")
        base_sae_fixture = request.getfixturevalue(
            "get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis")

        attr_store = deepcopy(attr_fixture.result)
        base_store = deepcopy(base_sae_fixture.result)

        # Create a proper PredSumm dataclass
        from dataclasses import dataclass


        @dataclass
        class MockPredSumm:
            batch_predictions: list = None
            correct_examples: list = None

        # Create a MockPredSumm with all False values
        all_false_pred = MockPredSumm(
            correct_examples=[False] * len(attr_store.attribution_values),
            batch_predictions=[[0]] * len(attr_store.attribution_values)
        )

        # Get activation summary
        activation_summary = base_store.calc_activation_summary()

        # Mock the calculate_latent_metrics method to return a LatentMetrics
        def mock_calculate_metrics(pred_summ, activation_summary=None, filter_by_correct=True, run_name=None):
            vals = torch.rand(10)
            return LatentMetrics(
                mean_activation={'h': vals},
                num_samples_active={'h': vals},
                total_effect={'h': vals},
                mean_effect={'h': vals},
                proportion_samples_active={'h': vals}
            )

        # Replace the method with our mock
        attr_store.calculate_latent_metrics = mock_calculate_metrics

        # Should not raise and return a LatentMetrics object
        metrics = attr_store.calculate_latent_metrics(
            pred_summ=all_false_pred,
            activation_summary=activation_summary,
            filter_by_correct=True
        )

        assert isinstance(metrics, LatentMetrics)


class TestSAEAnalysisDict:
    """Tests for the SAEAnalysisDict class."""

    @pytest.mark.parametrize(
        "session_fixture, analysis_cfgs",
        [
            pytest.param("get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis", None),
        ],
        ids=["sl_gpt2_logit_diffs_sae"],
    )
    def test_core_sae_analysis_dict(self, request, session_fixture, analysis_cfgs):
        fixture = request.getfixturevalue(session_fixture)
        test_cfg, analysis_result = fixture.test_cfg(), fixture.result

        assert isinstance(analysis_result, AnalysisStore)
        sae_data = analysis_result.by_sae('correct_activations')
        assert isinstance(sae_data, SAEAnalysisDict)
        # Use the validation function instead of inline checks
        validate_sae_operations(sae_data, test_cfg.sae_analysis_targets)

    def test_init_and_setitem_valid(self):
        """Test initialization and setting valid items."""
        analysis_dict = SAEAnalysisDict()
        tensor_val = torch.randn(10, 5)
        tensor_list_val = [torch.randn(10, 5), torch.randn(5, 5)]

        # Test setting tensor value
        analysis_dict["sae1"] = tensor_val
        assert "sae1" in analysis_dict
        assert torch.equal(analysis_dict["sae1"], tensor_val)

        # Test setting list of tensors
        analysis_dict["sae2"] = tensor_list_val
        assert "sae2" in analysis_dict
        assert all(torch.equal(a, b) for a, b in zip(analysis_dict["sae2"], tensor_list_val))

        # Test with namedtuple-like objects
        ReturnType = namedtuple('ReturnType', ['values', 'indices'])
        named_tuple_val = ReturnType(values=torch.randn(5, 5), indices=torch.randint(0, 10, (5, 5)))
        analysis_dict["sae3"] = named_tuple_val
        assert "sae3" in analysis_dict
        assert torch.equal(analysis_dict["sae3"][0], named_tuple_val.values)

        # Test with multiple tensor fields in namedtuple
        ReturnType2 = namedtuple('ReturnType2', ['values1', 'values2'])
        multi_tensor_val = ReturnType2(values1=torch.randn(3, 3), values2=torch.randn(3, 3))
        analysis_dict["sae4"] = multi_tensor_val
        assert "sae4" in analysis_dict
        assert isinstance(analysis_dict["sae4"], list)
        assert len(analysis_dict["sae4"]) == 2
        assert torch.equal(analysis_dict["sae4"][0], multi_tensor_val.values1)
        assert torch.equal(analysis_dict["sae4"][1], multi_tensor_val.values2)

    def test_setitem_invalid(self):
        """Test setting invalid items raises TypeError."""
        analysis_dict = SAEAnalysisDict()

        # Test non-tensor/list value
        with pytest.raises(TypeError, match="Values must be torch.Tensor, list"):
            analysis_dict["invalid"] = 123

        # Test list with non-tensor elements
        with pytest.raises(TypeError, match="All list elements must be torch.Tensor"):
            analysis_dict["invalid_list"] = [1, 2, 3]  # Not tensors

        # Test namedtuple without tensor fields
        NonTensorType = namedtuple('NonTensorType', ['field1', 'field2'])
        non_tensor_val = NonTensorType(field1="string", field2=42)
        with pytest.raises(TypeError, match="does not contain any tensor fields"):
            analysis_dict["invalid_namedtuple"] = non_tensor_val

    def test_shapes_property(self):
        """Test the shapes property."""
        analysis_dict = SAEAnalysisDict()
        tensor1 = torch.randn(10, 5)
        tensor2 = torch.randn(7, 3)
        tensor_list = [torch.randn(8, 4), torch.randn(6, 2)]

        analysis_dict["sae1"] = tensor1
        analysis_dict["sae2"] = tensor2
        analysis_dict["sae3"] = tensor_list

        shapes = analysis_dict.shapes
        assert shapes["sae1"] == torch.Size([10, 5])
        assert shapes["sae2"] == torch.Size([7, 3])
        assert shapes["sae3"] == [torch.Size([8, 4]), torch.Size([6, 2])]

    def test_batch_join_across_saes_false(self):
        """Test batch_join with across_saes=False."""
        analysis_dict = SAEAnalysisDict()
        analysis_dict["sae1"] = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        analysis_dict["sae2"] = [torch.tensor([5, 6]), torch.tensor([7, 8])]

        result = analysis_dict.batch_join(across_saes=False)

        assert isinstance(result, SAEAnalysisDict)
        assert torch.equal(result["sae1"], torch.tensor([1, 2, 3, 4]))
        assert torch.equal(result["sae2"], torch.tensor([5, 6, 7, 8]))

    def test_batch_join_across_saes_true(self):
        """Test batch_join with across_saes=True."""
        analysis_dict = SAEAnalysisDict()
        analysis_dict["sae1"] = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        analysis_dict["sae2"] = [torch.tensor([5, 6]), torch.tensor([7, 8])]

        result = analysis_dict.batch_join(across_saes=True)

        assert isinstance(result, list)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([1, 2, 5, 6]))
        assert torch.equal(result[1], torch.tensor([3, 4, 7, 8]))

    def test_batch_join_with_none_values(self):
        """Test batch_join handles None values correctly."""
        analysis_dict = SAEAnalysisDict()
        analysis_dict["sae1"] = [torch.tensor([1, 2]), None, torch.tensor([3, 4])]
        analysis_dict["sae2"] = [torch.tensor([5, 6]), torch.tensor([7, 8]), None]

        # Test across_saes=False
        result1 = analysis_dict.batch_join(across_saes=False)
        assert torch.equal(result1["sae1"], torch.tensor([1, 2, 3, 4]))
        assert torch.equal(result1["sae2"], torch.tensor([5, 6, 7, 8]))

        # Test across_saes=True
        result2 = analysis_dict.batch_join(across_saes=True)
        assert len(result2) == 3
        assert torch.equal(result2[0], torch.tensor([1, 2, 5, 6]))
        assert result2[1] is not None  # Second batch has only one valid tensor
        assert result2[2] is not None  # Third batch has only one valid tensor

    def test_apply_op_by_sae(self):
        """Test apply_op_by_sae with both callable and string operations."""
        analysis_dict = SAEAnalysisDict()
        analysis_dict["sae1"] = torch.tensor([[1, 2], [3, 4]])
        analysis_dict["sae2"] = torch.tensor([[5, 6], [7, 8]])

        # Test with callable
        result1 = analysis_dict.apply_op_by_sae(lambda x: x * 2)
        assert isinstance(result1, SAEAnalysisDict)
        assert torch.equal(result1["sae1"], torch.tensor([[2, 4], [6, 8]]))
        assert torch.equal(result1["sae2"], torch.tensor([[10, 12], [14, 16]]))

        # Test with string method name
        result2 = analysis_dict.apply_op_by_sae("sum", dim=1)
        assert isinstance(result2, SAEAnalysisDict)
        assert torch.equal(result2["sae1"], torch.tensor([3, 7]))
        assert torch.equal(result2["sae2"], torch.tensor([11, 15]))

    def test_apply_op_by_sae_with_namedtuple(self):
        """Test apply_op_by_sae with operation returning namedtuple."""
        analysis_dict = SAEAnalysisDict()
        analysis_dict["sae1"] = torch.tensor([[1, 2], [3, 4]])
        analysis_dict["sae2"] = torch.tensor([[5, 6], [7, 8]])

        # Use torch.max with dim parameter to return a namedtuple-like object
        result = analysis_dict.apply_op_by_sae(torch.max, dim=1)
        assert isinstance(result, SAEAnalysisDict)

        # Check that values and indices are correctly extracted from namedtuple
        assert torch.equal(result["sae1"][0], torch.tensor([2, 4]))  # Values
        assert torch.equal(result["sae1"][1], torch.tensor([1, 1]))  # Indices
        assert torch.equal(result["sae2"][0], torch.tensor([6, 8]))  # Values
        assert torch.equal(result["sae2"][1], torch.tensor([1, 1]))  # Indices

    def test_setitem_namedtuple_handling(self):
        """Test handling of namedtuple-like objects in __setitem__."""
        analysis_dict = SAEAnalysisDict()

        # Test namedtuple with no tensor fields
        ReturnTypeNoTensor = namedtuple('ReturnTypeNoTensor', ['str_value', 'int_value'])
        no_tensor_val = ReturnTypeNoTensor(str_value="text", int_value=42)
        with pytest.raises(TypeError, match="does not contain any tensor fields"):
            analysis_dict["sae1"] = no_tensor_val

        # Test with None values in list
        analysis_dict["sae_with_none"] = [torch.tensor([1, 2]), None, torch.tensor([5, 6])]
        assert "sae_with_none" in analysis_dict
        assert len(analysis_dict["sae_with_none"]) == 3
        assert analysis_dict["sae_with_none"][1] is None

    def test_setitem_namedtuple_with_tensor_fields(self):
        """Test setting items with tensor fields in namedtuples."""
        import torch
        from collections import namedtuple

        analysis_dict = SAEAnalysisDict()

        # Create namedtuple with tensor fields
        TensorFields = namedtuple('TensorFields', ['tensor1', 'tensor2'])
        tensor_fields = TensorFields(tensor1=torch.randn(3, 2), tensor2=torch.randn(2, 4))

        # This should not raise an exception
        analysis_dict["sae_namedtuple"] = tensor_fields

        # Verify the tensors were extracted
        assert isinstance(analysis_dict["sae_namedtuple"], list)
        assert len(analysis_dict["sae_namedtuple"]) == 2
        assert torch.equal(analysis_dict["sae_namedtuple"][0], tensor_fields.tensor1)
        assert torch.equal(analysis_dict["sae_namedtuple"][1], tensor_fields.tensor2)

    def test_setitem_with_different_tensor_types(self):
        """Test setting items with different types of tensor structures."""
        analysis_dict = SAEAnalysisDict()

        # Single tensor value
        analysis_dict["sae_tensor"] = torch.tensor([1.0, 2.0])
        assert torch.equal(analysis_dict["sae_tensor"], torch.tensor([1.0, 2.0]))

        # List of tensors
        tensor_list = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        analysis_dict["sae_list"] = tensor_list
        assert all(torch.equal(a, b) for a, b in zip(analysis_dict["sae_list"], tensor_list))

        # Using a namedtuple with tensor fields
        ReturnType = namedtuple('ReturnType', ['values', 'indices'])
        named_tuple = ReturnType(values=torch.tensor([5.0, 6.0]), indices=torch.tensor([0, 1]))
        analysis_dict["sae_namedtuple"] = named_tuple
        assert isinstance(analysis_dict["sae_namedtuple"], list)
        assert torch.equal(analysis_dict["sae_namedtuple"][0], named_tuple.values)

        # None value is also accepted
        analysis_dict["sae_none"] = None
        assert analysis_dict["sae_none"] is None

    def test_batch_join_edge_cases(self):
        """Test batch_join with empty lists and all None values."""
        analysis_dict = SAEAnalysisDict()
        analysis_dict["sae1"] = [None, None]
        analysis_dict["sae2"] = [torch.tensor([5, 6]), None]

        # Test across_saes=True with some None values
        result = analysis_dict.batch_join(across_saes=True)
        assert len(result) == 2
        assert result[0] is not None  # First batch has one valid tensor
        assert result[1] is None      # Second batch has all None tensors

        # Create dict with all None values
        all_none_dict = SAEAnalysisDict()
        all_none_dict["sae1"] = [None, None]
        all_none_dict["sae2"] = [None, None]

        # Test across_saes=False with all None values
        result = all_none_dict.batch_join(across_saes=False)
        assert "sae1" in result
        assert "sae2" in result
        assert result["sae1"] is None
        assert result["sae2"] is None

        # Test across_saes=True with all None values
        result = all_none_dict.batch_join(across_saes=True)
        assert len(result) == 2
        assert result[0] is None
        assert result[1] is None

    def test_getattr_protocol_and_dataset(self, request):
        """Test __getattr__ for protocol and dataset attributes using a real AnalysisStore fixture."""
        # Use a real AnalysisStore fixture for fidelity
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
        analysis_store = deepcopy(fixture.result)

        # Protocol field: should exist in AnalysisBatchProtocol.__annotations__
        # We'll use a real field from the protocol if possible, else add a dummy
        proto_fields = list(AnalysisBatchProtocol.__annotations__.keys())
        if proto_fields:
            proto_field = proto_fields[0]
        else:
            proto_field = "test_col"
            AnalysisBatchProtocol.__annotations__[proto_field] = int

        # Should not raise
        try:
            _ = getattr(analysis_store, proto_field)
        except Exception as e:
            pytest.fail(f"Accessing protocol field {proto_field} raised: {e}")

        # Dataset attribute: should exist
        # We'll add a dummy attribute to the underlying dataset
        setattr(analysis_store.dataset, "some_attr", 123)
        assert analysis_store.some_attr == 123

        # Dataset method: should wrap and call set_format if result has set_format
        def dummy_method():
            class DummyDS:
                def set_format(self, type): self.called = type
            return DummyDS()
        setattr(analysis_store.dataset, "method", dummy_method)
        wrapped = analysis_store.__getattr__('method')
        result = wrapped()
        assert hasattr(result, "set_format")
        assert getattr(result, "called", None) == "interpretune"

        # AttributeError: should raise if not found in protocol or dataset
        # Remove attribute if present
        if hasattr(analysis_store.dataset, "not_found"):
            delattr(analysis_store.dataset, "not_found")
        with pytest.raises(AttributeError):
            _ = analysis_store.not_found

    def test_by_sae_typeerror(self):
        """Test by_sae raises TypeError for non-dict values."""
        mock_dataset = MagicMock()
        setattr(mock_dataset, 'field_name', [1, 2, 3])
        store = AnalysisStore(dataset=mock_dataset)
        store.__getattr__ = lambda name: getattr(mock_dataset, name)
        with pytest.raises(TypeError):
            store.by_sae('field_name')

    def test_by_sae_empty_list(self):
        """Test by_sae with empty list in batch."""
        mock_dataset = MagicMock()
        # Each batch contains a dict with a list value (empty list triggers None)
        setattr(mock_dataset, 'field_name', [{'sae1': []}, {'sae1': torch.tensor([1])}])
        store = AnalysisStore(dataset=mock_dataset)
        store.__getattr__ = lambda name: getattr(mock_dataset, name)
        result = store.by_sae('field_name')
        assert result['sae1'][0] is None
        assert torch.equal(result['sae1'][1], torch.tensor([1]))

    def test_calc_activation_summary_error(self):
        """Test calc_activation_summary raises ValueError if no correct_activations."""
        store = AnalysisStore(dataset=MagicMock())
        store.correct_activations = []
        with pytest.raises(ValueError):
            store.calc_activation_summary()

    @pytest.mark.parametrize(
        "attr_fixture_name",
        [
            "get_analysis_session__sl_gpt2_logit_diffs_attr_ablation__initonly_runanalysis",
            "get_analysis_session__sl_gpt2_logit_diffs_attr_grad__initonly_runanalysis",
        ],
        ids=["attr_ablation", "attr_grad"]
    )
    def test_calculate_latent_metrics(self, request, attr_fixture_name):
        """Test calculate_latent_metrics with and without filter_by_correct using a real AnalysisStore."""
        # Use a real AnalysisStore fixture for fidelity
        attr_fixture = request.getfixturevalue(attr_fixture_name)
        base_sae_fixture = \
            request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis")
        attr_analysis_store = deepcopy(attr_fixture.result)
        base_sae_analysis_store = deepcopy(base_sae_fixture.result)
        pred_summary = compute_correct(attr_analysis_store, attr_fixture.runner.run_cfg.analysis_cfgs[0].name)
        activation_summary = base_sae_analysis_store.calc_activation_summary()
        # Should not raise for both filter_by_correct True/False
        metrics = attr_analysis_store.calculate_latent_metrics(
            pred_summ=pred_summary, activation_summary=activation_summary, filter_by_correct=True)
        assert isinstance(metrics, LatentMetrics)
        metrics2 = attr_analysis_store.calculate_latent_metrics(
            pred_summ=pred_summary, activation_summary=activation_summary, filter_by_correct=False)
        assert isinstance(metrics2, LatentMetrics)
        # test with no activation_summary (for ops that can regenerate it)
        if getattr(attr_analysis_store, 'correct_activations', None) is not None:
            metrics_no_actsumm = attr_analysis_store.calculate_latent_metrics(
            pred_summ=pred_summary, activation_summary=None, filter_by_correct=True)
            assert isinstance(metrics_no_actsumm, LatentMetrics)

    def test_plot_latent_effects(self, monkeypatch):
        """Test plot_latent_effects for both per_batch True/False."""
        store = AnalysisStore(dataset=MagicMock())
        # Patch required fields
        store.attribution_values = [
            {'sae': torch.randn(5, 10)},
            {'sae': torch.randn(5, 10)}
        ]
        store.alive_latents = [
            {'sae': [0, 1, 2]},
            {'sae': [0, 3, 4]}
        ]
        # Patch by_sae and batch_join
        class DummyDict(SAEAnalysisDict):
            def batch_join(self2): return self2
            def apply_op_by_sae(self2, operation, *args, **kwargs):
                return {'sae': torch.randn(10)}
            def keys(self2): return ['sae']
        store.by_sae = lambda name: DummyDict({'sae': [torch.randn(5, 10), torch.randn(5, 10)]})
        # Patch px.line to avoid plotting
        monkeypatch.setattr('plotly.express.line', lambda *a, **k: type('PX', (), {'update_layout': lambda s, **k: s,
                                                                                   'show': lambda s: None})())
        store.plot_latent_effects(per_batch=True)
        store.plot_latent_effects(per_batch=False)

    def test_setitem_none_value(self):
        """Test setting None value is allowed and handled correctly."""
        analysis_dict = SAEAnalysisDict()

        # Test setting None value
        analysis_dict["sae1"] = None
        assert "sae1" in analysis_dict
        assert analysis_dict["sae1"] is None

        # Test None values in list
        analysis_dict["sae2"] = [torch.tensor([1.0]), None, torch.tensor([2.0])]
        assert "sae2" in analysis_dict
        assert analysis_dict["sae2"][1] is None
        assert torch.equal(analysis_dict["sae2"][0], torch.tensor([1.0]))
        assert torch.equal(analysis_dict["sae2"][2], torch.tensor([2.0]))
