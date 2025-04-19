from __future__ import annotations
from collections import namedtuple

import pytest
import torch
from torch.testing import assert_close

from interpretune.analysis.core import SAEAnalysisDict, AnalysisStore


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
            pytest.param("get_analysis_session__sl_gpt2_logit_diffs_sae__setup_runanalysis", None),
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


class TestCoreFunctions:
    """Tests for the core utility functions in the analysis module."""

    def test_get_module_dims(self, monkeypatch):
        """Test the get_module_dims function to extract dimensions from a module."""
        from interpretune.analysis.core import get_module_dims

        # Create a mock module with the necessary attributes
        class MockModule:
            def __init__(self):
                self.datamodule = type('obj', (object,), {
                    'itdm_cfg': type('obj', (object,), {'eval_batch_size': 8})
                })
                self.it_cfg = type('obj', (object,), {
                    'generative_step_cfg': type('obj', (object,), {
                        'lm_generation_cfg': type('obj', (object,), {'max_new_tokens': 20})
                    }),
                    'num_labels': 3,
                    'entailment_mapping': {'entailment': 0, 'neutral': 1, 'contradiction': 2}
                })

        mock_module = MockModule()

        # Test with num_labels present
        batch_size, max_answer_tokens, num_classes = get_module_dims(mock_module)
        assert batch_size == 8
        assert max_answer_tokens == 20
        assert num_classes == 3

        # Test with num_labels = None, should use len(entailment_mapping)
        mock_module.it_cfg.num_labels = None
        batch_size, max_answer_tokens, num_classes = get_module_dims(mock_module)
        assert batch_size == 8
        assert max_answer_tokens == 20
        assert num_classes == 3

    def test_get_filtered_sae_hook_keys(self):
        """Test the get_filtered_sae_hook_keys function."""
        from interpretune.analysis.core import get_filtered_sae_hook_keys

        # Create a mock handle with the necessary attributes
        class MockHandle:
            def __init__(self):
                self.cfg = type('obj', (object,), {'hook_name': 'blocks.0.attn'})
                self.hook_dict = {
                    'hook_z': None,
                    'hook_q': None,
                    'hook_k': None,
                    'hook_v': None,
                    'hook_pattern': None
                }

        mock_handle = MockHandle()

        # Test with a filter that accepts all hooks
        all_keys = get_filtered_sae_hook_keys(mock_handle, lambda x: True)
        assert len(all_keys) == 5
        assert 'blocks.0.attn.hook_z' in all_keys
        assert 'blocks.0.attn.hook_q' in all_keys
        assert 'blocks.0.attn.hook_k' in all_keys
        assert 'blocks.0.attn.hook_v' in all_keys
        assert 'blocks.0.attn.hook_pattern' in all_keys

        # Test with a filter that only accepts hooks containing 'z' or 'q'
        filtered_keys = get_filtered_sae_hook_keys(
            mock_handle,
            lambda x: 'hook_z' in x or 'hook_q' in x
        )
        assert len(filtered_keys) == 2
        assert 'blocks.0.attn.hook_z' in filtered_keys
        assert 'blocks.0.attn.hook_q' in filtered_keys
        assert 'blocks.0.attn.hook_k' not in filtered_keys
        assert 'blocks.0.attn.hook_v' not in filtered_keys
        assert 'blocks.0.attn.hook_pattern' not in filtered_keys
