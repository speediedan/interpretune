from __future__ import annotations
from collections import namedtuple
from unittest.mock import MagicMock, patch
import pandas as pd
from copy import deepcopy


import pytest
import torch
from torch.testing import assert_close

from interpretune.analysis.core import (SAEAnalysisDict, AnalysisStore, schema_to_features,
                                       default_sae_id_factory_fn, default_sae_hook_match_fn,
                                       SAEAnalysisTargets, BaseMetrics, ActivationSumm, LatentMetrics,
                                       base_vs_sae_logit_diffs,
                                       compute_correct)
from interpretune.analysis.ops.base import ColCfg



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
            analysis_dict["sae1"] = "not a tensor"

        # Test list with non-tensor elements
        with pytest.raises(TypeError, match="All list elements must be torch.Tensor"):
            analysis_dict["sae2"] = [torch.randn(5, 5), "not a tensor"]

        # Test namedtuple without tensor fields
        NonTensorType = namedtuple('NonTensorType', ['field1', 'field2'])
        non_tensor_val = NonTensorType(field1="string", field2=42)
        with pytest.raises(TypeError, match="does not contain any tensor fields"):
            analysis_dict["sae3"] = non_tensor_val

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


class TestSchemaToFeatures:
    """Tests for schema_to_features function."""

    def test_schema_to_features_basic(self):
        """Test schema_to_features with basic schema."""
        # Create a mock module with necessary attributes
        mock_module = MagicMock()
        mock_module.datamodule.itdm_cfg.eval_batch_size = 8
        mock_module.it_cfg.generative_step_cfg.lm_generation_cfg.max_new_tokens = 20
        mock_module.it_cfg.num_labels = 3
        mock_module.it_cfg.entailment_mapping = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        # Create a simple schema
        test_schema = {
            'simple_col': ColCfg(datasets_dtype='float32', array_dtype='float32'),
            'sequence_col': ColCfg(datasets_dtype='string', sequence_type=True),
            'array_col': ColCfg(datasets_dtype='float32', array_shape=(10, 'max_answer_tokens'), array_dtype='float32'),
            'intermediate_col': ColCfg(datasets_dtype='float32', intermediate_only=True, array_dtype='float32')
        }

        # Test with direct schema
        features = schema_to_features(mock_module, schema=test_schema)

        # Verify features were created correctly
        assert 'simple_col' in features
        assert 'sequence_col' in features
        assert 'array_col' in features
        assert 'intermediate_col' not in features  # Should be excluded as it's intermediate_only

        # Check dimensions
        arr_col = features['array_col']
        if hasattr(arr_col, 'shape'):
            assert arr_col.shape == (10, 20)
        else:
            # If it's a DatasetsSequence, check feature type
            assert isinstance(arr_col, type(features['sequence_col']))

    def test_schema_to_features_per_sae(self):
        """Test schema_to_features with per_sae_hook fields."""
        # Create a mock module with necessary attributes
        mock_module = MagicMock()
        mock_module.datamodule.itdm_cfg.eval_batch_size = 8
        mock_module.it_cfg.generative_step_cfg.lm_generation_cfg.max_new_tokens = 20
        mock_module.it_cfg.num_labels = 3
        mock_module.it_cfg.entailment_mapping = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        # Create mock SAE handles
        mock_handle1 = MagicMock()
        mock_handle1.cfg.hook_name = 'blocks.0.attn'
        mock_handle1.cfg.d_sae = 64
        mock_handle1.cfg.dtype = 'float32'
        mock_handle1.hook_dict = {'hook_z': None, 'hook_q': None}

        mock_handle2 = MagicMock()
        mock_handle2.cfg.hook_name = 'blocks.1.attn'
        mock_handle2.cfg.d_sae = 64
        mock_handle2.cfg.dtype = 'float32'
        mock_handle2.hook_dict = {'hook_z': None, 'hook_q': None}

        mock_module.sae_handles = [mock_handle1, mock_handle2]
        mock_module.analysis_cfg.names_filter = lambda x: True

        # Create a schema with per_sae_hook and per_latent fields
        test_schema = {
            'sae_field': ColCfg(datasets_dtype='float32', per_sae_hook=True),
            'latent_field': ColCfg(datasets_dtype='float32', per_latent=True, array_shape=(5, 5),
                                   array_dtype='float32'),
            'seq_sae_field': ColCfg(per_sae_hook=True, non_tensor=True, sequence_type=True, datasets_dtype='string')
        }

        # Test with direct schema
        features = schema_to_features(mock_module, schema=test_schema)

        # Verify features were created correctly
        assert 'sae_field' in features
        assert 'latent_field' in features
        assert 'seq_sae_field' in features

        # Check that SAE hooks were properly added
        assert 'blocks.0.attn.hook_z' in features['sae_field']
        assert 'blocks.0.attn.hook_q' in features['sae_field']
        assert 'blocks.1.attn.hook_z' in features['sae_field']
        assert 'blocks.1.attn.hook_q' in features['sae_field']

        # Check latent field structure
        assert 'blocks.0.attn.hook_z' in features['latent_field']
        assert 'latents' in features['latent_field']['blocks.0.attn.hook_z']
        assert 'per_latent' in features['latent_field']['blocks.0.attn.hook_z']

    def test_schema_to_features_no_schema(self):
        """Test schema_to_features returns empty dict when no schema is provided."""
        mock_module = MagicMock()
        # No schema and no op, and module.analysis_cfg.output_schema is not present
        mock_module.analysis_cfg = MagicMock()
        del mock_module.analysis_cfg.output_schema
        features = schema_to_features(mock_module, op=None, schema=None)
        assert features == {}

        # Test with schema=None and op=None and module=None
        features = schema_to_features(None, op=None, schema=None)
        assert features == {}


class TestAnalysisStore:
    """Tests for the AnalysisStore class."""

    def test_initialization(self):
        """Test AnalysisStore initialization with different parameters."""
        # Test with None dataset
        store = AnalysisStore(dataset=None, op_output_dataset_path="/tmp/test", stack_batches=True)
        assert store.dataset is None
        assert store.stack_batches is True

        # Test with mocked dataset
        mock_dataset = MagicMock()
        store = AnalysisStore(
            dataset=mock_dataset,
            op_output_dataset_path="/tmp/test",
            cache_dir="/tmp/cache",
            dataset_trust_remote_code=True,
            streaming=True,
            split="test",
            stack_batches=False
        )
        assert store.dataset is mock_dataset
        assert store.cache_dir == "/tmp/cache"
        assert store.streaming is True
        assert store.dataset_trust_remote_code is True
        assert store.split == "test"
        assert store.stack_batches is False

        # Verify dataset.set_format was called
        mock_dataset.set_format.assert_called_with(type='interpretune')

    @patch('os.path.abspath')
    @patch('interpretune.analysis.core.load_dataset')
    def test_load_dataset_path(self, mock_load_dataset, mock_abspath):
        """Test _load_dataset with path input."""
        mock_abspath.return_value = "/abs/path/to/dataset"
        mock_load_dataset.return_value = MagicMock()

        store = AnalysisStore(op_output_dataset_path="/tmp/output")
        store._load_dataset("/path/to/dataset")

        mock_load_dataset.assert_called_with(
            "/abs/path/to/dataset",
            split="validation",
            streaming=False,
            trust_remote_code=False
        )

    @patch('os.path.abspath')
    def test_load_dataset_path_conflict(self, mock_abspath):
        """Test _load_dataset with conflicting paths."""
        mock_abspath.side_effect = lambda x: x  # Return input unchanged

        store = AnalysisStore(op_output_dataset_path="/tmp/same_path")

        with pytest.raises(ValueError, match="must not overlap"):
            store._load_dataset("/tmp/same_path")

    def test_reset(self):
        """Test reset method."""
        # Create a mock store with a dataset
        store = AnalysisStore()
        mock_dataset = MagicMock()
        store.dataset = mock_dataset
        # Don't assign to save_dir property (remove this line)
        # store.save_dir = MagicMock()
        # Mock _load_dataset to track calls
        store._load_dataset = MagicMock()
        # Test reset
        store.reset()
        # Verify _load_dataset was NOT called (since MagicMock triggers except block)
        store._load_dataset.assert_not_called()

    def test_getitem(self):
        """Test __getitem__ method for different key types."""
        mock_dataset = MagicMock()
        # Mock dataset.__getitem__ to return different types
        mock_dataset.__getitem__.side_effect = lambda key: {
            'col1': torch.tensor([[1, 2], [3, 4]]),  # 2D tensor
            'col2': torch.tensor([5, 6]),            # 1D tensor
            'col3': 'string_value',                  # Non-tensor
            0: {'item': 'first'},                    # Integer index
            'multiple_cols': {'col1': torch.tensor([1]), 'col2': torch.tensor([2])}  # Multiple columns
        }[key]

        store = AnalysisStore(dataset=mock_dataset, stack_batches=False)

        # Test single column string key with 2D tensor
        result = store['col1']
        assert isinstance(result, list)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([1, 2]))

        # Test single column string key with 1D tensor
        result = store['col2']
        assert isinstance(result, list)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor(5))

        # Test non-tensor column
        result = store['col3']
        assert result == 'string_value'

        # Test integer index
        result = store[0]
        assert result == {'item': 'first'}

        # Test list of column names
        store.stack_batches = True  # Test with stack_batches=True
        result = store[['col1', 'col2']]
        assert isinstance(result, dict)
        assert 'col1' in result
        assert 'col2' in result

    def test_select_columns(self, request):
        """Test select_columns method using a real AnalysisStore from fixture."""
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis")
        analysis_store = deepcopy(fixture.result)
        # Pick a couple of columns that are known to exist in the dataset
        # For demonstration, use the first two columns if available
        if hasattr(analysis_store.dataset, 'column_names'):
            columns = list(analysis_store.dataset.column_names)[:2]
        else:
            # Fallback: use a default list if column_names is not available
            columns = ['correct_activations', 'labels']
        result = analysis_store.select_columns(columns)
        # Verify result is an AnalysisStore with the selected dataset
        assert isinstance(result, AnalysisStore)
        # The dataset should only have the selected columns
        if hasattr(result.dataset, 'column_names'):
            assert set(result.dataset.column_names) == set(columns)

    def test_getattr(self, request):
        """Test __getattr__ method using a real AnalysisStore from fixture."""
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis")
        analysis_store = fixture.result
        with pytest.raises(AttributeError):
            getattr(analysis_store, 'nonexistent_attr')


class TestBySAE:
    """Tests for by_sae method in AnalysisStore."""

    def test_by_sae_basic(self):
        """Test by_sae with basic dictionary values."""
        # Create mock data
        mock_data = [
            {'sae1': torch.tensor([1, 2]), 'sae2': torch.tensor([5, 6])},
            {'sae1': torch.tensor([3, 4]), 'sae2': torch.tensor([7, 8])}
        ]

        mock_dataset = MagicMock()
        # Instead of setting __getattr__.return_value, set the attribute directly
        setattr(mock_dataset, 'field_name', mock_data)

        store = AnalysisStore(dataset=mock_dataset)
        # Patch __getattr__ to return the attribute
        store.__getattr__ = lambda name: getattr(mock_dataset, name)

        # Test by_sae
        result = store.by_sae('field_name')

        # Verify result
        assert isinstance(result, SAEAnalysisDict)
        assert 'sae1' in result
        assert 'sae2' in result
        assert len(result['sae1']) == 2
        assert torch.equal(result['sae1'][0], torch.tensor([1, 2]))

    def test_by_sae_nested(self):
        """Test by_sae with nested dictionary values."""
        # Use 2D tensors so the shape matches (2, 2)
        mock_data = [
            {'sae1': torch.tensor([[1, 2], [3, 4]]), 'sae2': torch.tensor([[5, 6], [7, 8]])},
            {'sae1': torch.tensor([[9, 10], [11, 12]]), 'sae2': torch.tensor([[13, 14], [15, 16]])}
        ]
        mock_dataset = MagicMock()
        setattr(mock_dataset, 'field_name', mock_data)
        store = AnalysisStore(dataset=mock_dataset)
        store.__getattr__ = lambda name: getattr(mock_dataset, name)
        # Test by_sae with stack_latents=True
        result = store.by_sae('field_name', stack_latents=True)
        assert isinstance(result, SAEAnalysisDict)
        assert 'sae1' in result
        assert 'sae2' in result
        assert len(result['sae1']) == 2
        assert result['sae1'][0].shape == (2, 2)  # stacked latent tensors
        # Test by_sae with stack_latents=False
        result = store.by_sae('field_name', stack_latents=False)
        assert isinstance(result, SAEAnalysisDict)
        assert torch.equal(result['sae1'][0], mock_data[0]['sae1'])


class TestActivationSummary:
    """Tests for calc_activation_summary method."""

    def test_calc_activation_summary(self, request):
        """Test calc_activation_summary using a real AnalysisStore from fixture."""
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis")
        analysis_store = fixture.result
        from interpretune.analysis.core import ActivationSumm
        activation_summ = analysis_store.calc_activation_summary()
        assert isinstance(activation_summ, ActivationSumm)
        assert hasattr(activation_summ, 'mean_activation')
        assert hasattr(activation_summ, 'num_samples_active')
        for sae_name in analysis_store.by_sae('correct_activations').keys():
            assert sae_name in activation_summ.mean_activation
            assert sae_name in activation_summ.num_samples_active

    def test_calc_activation_summary_error(self):
        """Test calc_activation_summary with missing data."""
        mock_dataset = MagicMock()
        store = AnalysisStore(dataset=mock_dataset)

        # Instead of side_effect, set the attribute directly
        setattr(mock_dataset, 'correct_activations', [])
        store.__getattr__ = lambda name: getattr(mock_dataset, name)

        # Test that it raises ValueError
        with pytest.raises(AssertionError, match="No values found for field correct_activations"):
            store.calc_activation_summary()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_compute_correct(self):
        """Test compute_correct function."""
        # Use real tensors for orig_labels and preds
        class DummyStore:
            orig_labels = [torch.tensor([0, 1, 2]), torch.tensor([1, 0])]
            preds = [torch.tensor([0, 1, 0]), torch.tensor([1, 1])]
        mock_analysis_op = MagicMock()
        mock_analysis_op.alias = "standard_op"
        result = compute_correct(DummyStore(), mock_analysis_op)
        assert isinstance(result, tuple) or hasattr(result, 'total_correct')


class TestVisualizationFunctions:
    """Tests for visualization functions."""

    def test_latent_metrics_scatter(self, monkeypatch):
        """Test latent_metrics_scatter function with manually created metrics."""

        # Create dummy stores and tokenizer for base_vs_sae_logit_diffs
        labels = [torch.tensor([0, 1]), torch.tensor([1, 0])]
        clean_diffs = [0.5, 1.0]
        sae_diffs = [-0.5, 0.2]
        prompts = ["prompt1", "prompt2"]

        base_ref = MagicMock()
        base_ref.labels = labels
        base_ref.logit_diffs = clean_diffs

        sae_store = MagicMock()
        sae_store.prompts = prompts
        sae_store.logit_diffs = sae_diffs

        mock_tokenizer = MagicMock()
         # Mock numpy to avoid actual conversion
        mock_numpy = MagicMock()
        mock_numpy.return_value = [1, 2]
        monkeypatch.setattr(torch.Tensor, "numpy", mock_numpy)

        # Mock DataFrame to avoid creation issues
        mock_df = MagicMock()
        filtered_df = MagicMock()  # Create a separate mock for the filtered DataFrame

        # Configure the mocks for each method in the chain
        mock_df.explode.return_value = mock_df
        mock_df.sort_values.return_value = mock_df
        mock_df.head.return_value = mock_df

        # Setup a custom __getitem__ method that handles both column selection and boolean filtering
        def custom_getitem(key):
            if isinstance(key, list) and all(isinstance(k, str) for k in key):
                # For column selection like df[["sample_id", ...]]
                return mock_df
            else:
                # For boolean filtering like df[df.clean_logit_diff > 0]
                return filtered_df

        mock_df.__getitem__ = MagicMock(side_effect=custom_getitem)

        # Add clean_logit_diff as an attribute (not using __getattr__)
        clean_logit_diff = MagicMock()
        clean_logit_diff.__gt__ = MagicMock(return_value="filter_condition")
        mock_df.clean_logit_diff = clean_logit_diff

        # Configure the filtered DataFrame
        filtered_df.sort_values.return_value = filtered_df
        filtered_df.head.return_value = filtered_df

        # Mock DataFrame constructor to return our mock
        mock_df_constructor = MagicMock(return_value=mock_df)
        monkeypatch.setattr(pd, "DataFrame", mock_df_constructor)

        # Mock tabulate
        mock_tabulate = MagicMock()
        monkeypatch.setattr("interpretune.analysis.core.tabulate", mock_tabulate)

        # Now call the original function
        base_vs_sae_logit_diffs(
            sae=sae_store,
            base_ref=base_ref,
            tokenizer=mock_tokenizer,
            top_k=5
        )

        # Verify tokenizer.batch_decode was called twice (once per label tensor)
        assert mock_tokenizer.batch_decode.call_count == 2


class TestSAETargets:
    """Tests for SAEAnalysisTargets class."""

    def test_default_sae_id_factory_fn(self):
        """Test default_sae_id_factory_fn with various parameters."""
        # Default parameters
        sae_id = default_sae_id_factory_fn(layer=5)
        assert sae_id == "blocks.5.hook_z"

        # Custom parameters
        sae_id = default_sae_id_factory_fn(layer=3, prefix_pat="transformer", suffix_pat="attn_output")
        assert sae_id == "transformer.3.attn_output"

    def test_default_sae_hook_match_fn(self):
        """Test default_sae_hook_match_fn with various parameters."""
        # Default parameters, matching name
        assert default_sae_hook_match_fn("blocks.5.hook_sae_acts_post")

        # Non-matching name
        assert not default_sae_hook_match_fn("blocks.5.other_hook")

        # With layers parameter
        assert default_sae_hook_match_fn("blocks.5.hook_sae_acts_post", layers=[5, 6])
        assert not default_sae_hook_match_fn("blocks.5.hook_sae_acts_post", layers=[3, 4])

        # With single layer parameter
        assert default_sae_hook_match_fn("blocks.7.hook_sae_acts_post", layers=7)
        assert not default_sae_hook_match_fn("blocks.7.hook_sae_acts_post", layers=8)

        # With custom prefix and suffix
        assert default_sae_hook_match_fn(
            "transformer.5.post_attention",
            hook_point_suffix="post_attention",
            hook_point_prefix="transformer"
        )

    def test_sae_analysis_targets_init(self):
        """Test SAEAnalysisTargets initialization."""
        # With explicit SAE FQNs
        targets = SAEAnalysisTargets(
            sae_fqns=[("release1", "sae1"), ("release2", "sae2")]
        )
        assert len(targets.sae_fqns) == 2
        assert targets.sae_fqns[0].release == "release1"
        assert targets.sae_fqns[0].sae_id == "sae1"

        # With target_sae_ids
        targets = SAEAnalysisTargets(
            sae_release="custom-release",
            target_sae_ids=["sae1", "sae2"]
        )
        assert len(targets.sae_fqns) == 2
        assert targets.sae_fqns[0].release == "custom-release"
        assert targets.sae_fqns[0].sae_id == "sae1"

        # With target_layers
        targets = SAEAnalysisTargets(
            sae_release="custom-release",
            target_layers=[0, 1, 2]
        )
        assert len(targets.sae_fqns) == 3
        assert targets.sae_fqns[0].sae_id == "blocks.0.hook_z"
        assert targets.sae_fqns[1].sae_id == "blocks.1.hook_z"

        # With custom sae_id_factory_fn
        custom_factory = lambda layer: f"custom.{layer}.hook"
        targets = SAEAnalysisTargets(
            sae_release="custom-release",
            target_layers=[0, 1],
            sae_id_factory_fn=custom_factory
        )
        assert targets.sae_fqns[0].sae_id == "custom.0.hook"
        assert targets.sae_fqns[1].sae_id == "custom.1.hook"

        # With invalid sae_fqns
        with pytest.raises(TypeError):
            SAEAnalysisTargets(sae_fqns=["invalid"])

        # With no configuration (should warn)
        with patch("interpretune.analysis.core.rank_zero_warn") as mock_warn:
            targets = SAEAnalysisTargets()
            assert len(targets.sae_fqns) == 0
            mock_warn.assert_called_once()

    def test_validate_sae_fqns(self):
        """Test validate_sae_fqns method."""
        # Create target with existing sae_fqns
        targets = SAEAnalysisTargets(
            sae_fqns=[("release1", "sae1")]
        )

        # Modify sae_fqns and validate
        targets.sae_fqns = []
        targets.target_sae_ids = ["new_sae1", "new_sae2"]

        results = targets.validate_sae_fqns()
        assert len(results) == 2
        assert results[0].sae_id == "new_sae1"
        assert results[1].sae_id == "new_sae2"


class TestMetricsClasses:
    """Tests for BaseMetrics, ActivationSumm, and LatentMetrics classes."""

    def test_base_metrics(self):
        """Test BaseMetrics initialization and methods."""
        # Basic initialization
        metrics = BaseMetrics(
            custom_repr={'field1': 'Field One', 'field2': 'Field Two'}
        )

        assert metrics.get_field_name('field1') == 'Field One'
        assert metrics.get_field_name('field2') == 'Field Two'
        assert metrics.get_field_name('unknown') == 'unknown'

        field_names = metrics.get_field_names()
        assert 'field1' in field_names
        assert field_names['field1'] == 'Field One'

        # Test validation - simulate mismatched metric dicts
        with pytest.raises(ValueError, match="must have the same keys"):
            # Provide two dictionaries with different keys to trigger validation
            BaseMetrics(custom_repr={})  # init normally
            # Manually inject mismatched metric dicts for validation
            bad = BaseMetrics(custom_repr={})
            bad.some_metric = {'a': 1}
            bad.other_metric = {'b': 2}
            bad.__post_init__()

    def test_activation_summ(self):
        """Test ActivationSumm initialization."""
        mean_activation = {
            'sae1': torch.tensor([0.1, 0.2, 0.3]),
            'sae2': torch.tensor([0.4, 0.5, 0.6])
        }
        num_samples_active = {
            'sae1': torch.tensor([10, 15, 20]),
            'sae2': torch.tensor([25, 30, 35])
        }

        summ = ActivationSumm(
            mean_activation=mean_activation,
            num_samples_active=num_samples_active,
            run_name="test_run"
        )

        assert summ.mean_activation == mean_activation
        assert summ.num_samples_active == num_samples_active
        assert summ.run_name == "test_run"

        # Check field representations
        assert summ.get_field_name('mean_activation') == 'Mean Activation'
        assert summ.get_field_name('num_samples_active') == 'Number Active'

    def test_latent_metrics(self):
        """Test LatentMetrics initialization."""
        mean_activation = {'sae1': torch.tensor([0.1, 0.2])}
        num_samples_active = {'sae1': torch.tensor([10, 15])}
        total_effect = {'sae1': torch.tensor([5.0, 10.0])}
        mean_effect = {'sae1': torch.tensor([0.5, 1.0])}
        proportion_samples_active = {'sae1': torch.tensor([0.2, 0.3])}

        metrics = LatentMetrics(
            mean_activation=mean_activation,
            num_samples_active=num_samples_active,
            total_effect=total_effect,
            mean_effect=mean_effect,
            proportion_samples_active=proportion_samples_active,
            run_name="test_run"
        )

        assert metrics.mean_activation == mean_activation
        assert metrics.total_effect == total_effect
        assert metrics.run_name == "test_run"

        # Check field names
        assert metrics.get_field_name('total_effect') == 'Total Effect'
        assert metrics.get_field_name('mean_effect') == 'Mean Effect'
