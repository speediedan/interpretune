from __future__ import annotations
import dataclasses
from collections import namedtuple
from unittest.mock import MagicMock
from copy import deepcopy

import pytest
import torch
from torch.testing import assert_close

from interpretune.analysis.core import (SAEAnalysisDict, AnalysisStore, resolve_names_filter,
                                        schema_to_features, SAEFqn, _make_simple_cache_hook,
                                       SAEAnalysisTargets, BaseMetrics, ActivationSumm, LatentMetrics,
                                       latent_metrics_scatter,
                                       compute_correct)
from interpretune.analysis.core import AnalysisBatchProtocol
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

    def test_getattr_protocol_and_dataset(self, request):
        """Test __getattr__ for protocol and dataset attributes using a real AnalysisStore fixture."""
        # Use a real AnalysisStore fixture for fidelity
        fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis")
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

    def test_calculate_latent_metrics(self, request):
        """Test calculate_latent_metrics with and without filter_by_correct using a real AnalysisStore."""
        # Use a real AnalysisStore fixture for fidelity
        ablation_fixture = \
            request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_attr_ablation__initonly_runanalysis")
        base_sae_fixture = \
            request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis")
        ablation_analysis_store = deepcopy(ablation_fixture.result)
        base_sae_analysis_store = deepcopy(base_sae_fixture.result)
        pred_summary = compute_correct(ablation_analysis_store, ablation_fixture.runner.run_cfg.analysis_cfgs[0].name)
        activation_summary = base_sae_analysis_store.calc_activation_summary()
        # Should not raise for both filter_by_correct True/False
        metrics = ablation_analysis_store.calculate_latent_metrics(
            pred_summ=pred_summary, activation_summary=activation_summary, filter_by_correct=True)
        assert isinstance(metrics, LatentMetrics)
        metrics2 = ablation_analysis_store.calculate_latent_metrics(
            pred_summ=pred_summary, activation_summary=activation_summary, filter_by_correct=False)
        assert isinstance(metrics2, LatentMetrics)

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

    def test_deepcopy_analysisstore(self):
        """Test __deepcopy__ for AnalysisStore with op_output_dataset_path."""
        store = AnalysisStore(op_output_dataset_path='/tmp/foo')
        store2 = deepcopy(store)
        assert store2 is not store
        assert store2.op_output_dataset_path != store.op_output_dataset_path

class TestSAEAnalysisTargets:
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

class TestBaseMetrics:
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

    def test_latent_metrics_scatter(self, monkeypatch):
        vals = torch.tensor([1.0, 2.0, 3.0])
        m1 = LatentMetrics(mean_activation={'h': vals}, num_samples_active={'h': vals},
                           total_effect={'h': vals}, mean_effect={'h': vals}, proportion_samples_active={'h': vals})
        m2 = LatentMetrics(mean_activation={'h': vals}, num_samples_active={'h': vals},
                           total_effect={'h': vals}, mean_effect={'h': vals}, proportion_samples_active={'h': vals})
        monkeypatch.setattr('plotly.express.scatter', lambda *a, **k: type('PX', (), {'add_shape': lambda s, **k: s,
                                                                                      'show': lambda s: None})())
        latent_metrics_scatter(m1, m2)

class TestSchemaToFeatures:
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

class TestHelperFunctions:
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
