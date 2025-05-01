from __future__ import annotations
import pytest
import torch
from unittest.mock import patch, MagicMock

import interpretune as it
from interpretune.analysis.ops.dispatcher import DISPATCHER, AnalysisOpDispatcher, DispatchContext
from interpretune.analysis.ops.base import AnalysisOp, OpSchema, ChainedAnalysisOp, AnalysisBatch


class TestAnalysisOpDispatcher:
    """Tests for the AnalysisOpDispatcher class."""

    def test_get_by_alias(self):
        """Test retrieving operations by alias."""
        # Test with valid alias
        op = DISPATCHER.get_by_alias("logit_diffs_sae")
        assert op is not None
        assert isinstance(op, AnalysisOp)
        assert op.alias == "logit_diffs_sae"

        # Test with invalid alias
        op = DISPATCHER.get_by_alias("non_existent_alias")
        assert op is None

    def test_import_callable(self):
        """Test importing a callable from a path."""
        # Test importing a valid callable
        callable_obj = DISPATCHER._import_callable("interpretune.analysis.ops.definitions.labels_to_ids_impl")
        assert callable(callable_obj)
        assert callable_obj.__name__ == "labels_to_ids_impl"

        # Test importing a non-existent callable
        with pytest.raises(ImportError):
            DISPATCHER._import_callable("non_existent_module.non_existent_function")

        # Test importing from existing module but non-existent function
        with pytest.raises(AttributeError):
            DISPATCHER._import_callable("interpretune.analysis.ops.definitions.non_existent_function")

    def test_convert_to_op_schema(self):
        """Test converting dictionary to OpSchema."""
        # Test basic schema conversion
        schema_def = {
            "field1": {"datasets_dtype": "float32", "required": True},
            "field2": {"datasets_dtype": "int64", "required": False}
        }

        op_schema = DISPATCHER._convert_to_op_schema(schema_def)
        assert isinstance(op_schema, OpSchema)
        assert "field1" in op_schema
        assert "field2" in op_schema
        assert op_schema["field1"].required is True
        assert op_schema["field2"].required is False

        # Test with more complex schema
        complex_schema = {
            "tensor_field": {
                "datasets_dtype": "float32",
                "array_shape": (None, "batch_size"),
                "dyn_dim": 1,
                "sequence_type": False
            }
        }

        op_schema = DISPATCHER._convert_to_op_schema(complex_schema)
        assert op_schema["tensor_field"].array_shape == (None, "batch_size")
        assert op_schema["tensor_field"].dyn_dim == 1


    # NOTE: Our fixtures that specify AnalysisCfg may trigger the instantiation of ops (e.g. tests/unit/cfg_aliases.py)
    #       via op resolution attribute access they do when configuring the test session. This could
    #       be refactored in the future if it leads to dispatcher testing confusion.

    def test_instantiate_op(self):
        """Test instantiating operations from definitions."""
        # Test instantiating a basic operation
        op = DISPATCHER._instantiate_op("model_forward")
        assert isinstance(op, AnalysisOp)
        assert op.name == "model_forward"

        # Test instantiating a chained operation
        op = DISPATCHER._instantiate_op("logit_diffs_sae")
        assert isinstance(op, ChainedAnalysisOp)
        assert op.name == "labels_to_ids.model_cache_forward.logit_diffs_cache.sae_correct_acts"
        assert op.alias == "logit_diffs_sae"
        assert len(op.chain) == 4

        # Test with unknown operation
        with pytest.raises(ValueError, match="Unknown operation:"):
            DISPATCHER._instantiate_op("non_existent_op")

    def test_get_op(self):
        """Test the get_op method."""
        # Test with normal op
        op = DISPATCHER.get_op("model_forward")
        assert isinstance(op, AnalysisOp)
        assert op.name == "model_forward"

        # Test with chained op
        op = DISPATCHER.get_op("logit_diffs_sae")
        assert isinstance(op, ChainedAnalysisOp)
        assert op.alias == "logit_diffs_sae"

        # Test with unknown op
        with pytest.raises(ValueError, match="Unknown operation:"):
            DISPATCHER.get_op("non_existent_op")

        # Test with an already instantiated op (should return cached version)
        del DISPATCHER._dispatch_table['labels_to_ids']
        test_context = DispatchContext()
        op = DISPATCHER.get_op("labels_to_ids", context=test_context, lazy=True)
        assert callable(op) and not isinstance(op, AnalysisOp)
        op = DISPATCHER._maybe_instantiate_op('labels_to_ids', test_context)
        assert op is DISPATCHER._dispatch_table["labels_to_ids"][test_context]

    def test_get_op_aliases(self):
        """Test getting all operation aliases."""
        aliases = list(DISPATCHER.get_op_aliases())
        assert len(aliases) > 0

        # Check if known aliases are present
        alias_dict = dict(aliases)
        assert "logit_diffs_sae" in alias_dict
        assert alias_dict["logit_diffs_sae"] == "logit_diffs_sae"

    def test_instantiate_all_ops(self):
        """Test instantiating all operations."""
        all_ops = DISPATCHER.instantiate_all_ops()
        assert isinstance(all_ops, dict)
        assert len(all_ops) > 0

        # Check if all returned values are AnalysisOp instances
        for name, op in all_ops.items():
            assert isinstance(op, AnalysisOp)
            if chain := getattr(op, 'chain', []):
                assert op.name == ".".join(o.name for o in chain)
            else:
                assert op.name == name

    def test_create_chain(self):
        """Test creating a chain of operations from names."""
        # Create a chain from operation names
        chain = DISPATCHER.create_chain(["labels_to_ids", "model_forward", "logit_diffs"])
        assert isinstance(chain, ChainedAnalysisOp)
        assert len(chain.chain) == 3
        assert chain.chain[0].name == "labels_to_ids"
        assert chain.chain[1].name == "model_forward"
        assert chain.chain[2].name == "logit_diffs"

        # Test with custom alias
        chain = DISPATCHER.create_chain(["labels_to_ids", "model_forward"], alias="custom_chain")
        assert chain.alias == "custom_chain"

        # Test with invalid operation name
        with pytest.raises(ValueError):
            DISPATCHER.create_chain(["non_existent_op"])

    def test_create_chain_from_ops(self):
        """Test creating a chain from operation instances."""
        # Get individual operations
        op1 = DISPATCHER.get_op("labels_to_ids")
        op2 = DISPATCHER.get_op("model_forward")

        # Create chain from operations
        chain = DISPATCHER.create_chain_from_ops([op1, op2], alias="test_chain")
        assert isinstance(chain, ChainedAnalysisOp)
        assert chain.alias == "test_chain"
        assert len(chain.chain) == 2
        assert chain.chain[0] is op1
        assert chain.chain[1] is op2

        # Test without specifying alias
        chain = DISPATCHER.create_chain_from_ops([op1, op2])
        assert chain.alias == "labels_to_ids.model_forward"

    def test_create_chain_from_string(self):
        """Test creating a chain from a dot-separated string."""
        # Create chain from string
        chain = DISPATCHER.create_chain_from_string("labels_to_ids.model_forward")

        assert isinstance(chain, ChainedAnalysisOp)
        assert len(chain.chain) == 2
        assert chain.chain[0].name == "labels_to_ids"
        assert chain.chain[1].name == "model_forward"

        # Test with alias
        chain = DISPATCHER.create_chain_from_string("labels_to_ids.model_forward", alias="custom_chain")
        assert chain.alias == "custom_chain"

    def test_op_lazy_instantiation(self):
        """Test lazy operation instantiation mechanism."""
        # Create a fresh dispatcher to avoid affecting global state
        test_dispatcher = AnalysisOpDispatcher()
        test_dispatcher.yaml_path = DISPATCHER.yaml_path
        test_dispatcher.load_definitions()

        # Get a reference to the operation without instantiating it
        context = DispatchContext()
        op_name = "labels_to_ids"
        lazy_op = test_dispatcher.get_op(op_name, context=context, lazy=True)

        # Verify it's a factory function (callable but not an AnalysisOp)
        assert callable(lazy_op)
        assert not isinstance(lazy_op, AnalysisOp)

        # Check if it's in the dispatch table as a factory function
        dispatch_entry = test_dispatcher._dispatch_table[op_name][context]
        assert callable(dispatch_entry) and not isinstance(dispatch_entry, AnalysisOp)

        # Now instantiate the operation
        instantiated_op = test_dispatcher._maybe_instantiate_op(op_name, context)

        # Verify it's now an AnalysisOp instance
        assert isinstance(instantiated_op, AnalysisOp)
        assert instantiated_op.name == op_name

        # Verify the dispatch table was updated with the instantiated op
        assert test_dispatcher._dispatch_table[op_name][context] is instantiated_op

        # Get it again - should return the instantiated version
        cached_op = test_dispatcher.get_op(op_name, context=context, lazy=True)
        assert cached_op is instantiated_op

    def test_lazy_op_execution(self):
        """Test that operations are instantiated on first execution."""
        # Create a fresh dispatcher for clean testing
        test_dispatcher = AnalysisOpDispatcher()
        test_dispatcher.yaml_path = DISPATCHER.yaml_path
        test_dispatcher.load_definitions()

        op_name = "labels_to_ids"
        context = DispatchContext()

        # Create necessary mocks for execution
        module_mock = MagicMock()

        # Use a real dict instead of a mock to properly handle the 'in' operator and key access
        batch_mock = {"labels": ["label1", "label2"]}
        module_mock.labels_to_ids.return_value = (torch.tensor([0, 1]), torch.tensor([0, 1]))

        # Get lazy reference
        lazy_op = test_dispatcher.get_op(op_name, context=context, lazy=True)
        assert callable(lazy_op) and not isinstance(lazy_op, AnalysisOp)

        # Execute via dispatcher
        result = test_dispatcher(op_name, module=module_mock, analysis_batch=None, batch=batch_mock, batch_idx=0)

        # Verify the op got instantiated during execution
        instantiated_op = test_dispatcher._dispatch_table[op_name][context]
        assert isinstance(instantiated_op, AnalysisOp)

        # Verify the result came through properly
        assert hasattr(result, 'labels')

        # A second call should use the same instantiated op
        # Create a fresh batch dictionary since the first call pops 'labels'
        batch_mock2 = {"labels": ["label3", "label4"]}
        test_dispatcher(op_name, module=module_mock, analysis_batch=None, batch=batch_mock2, batch_idx=0)
        assert test_dispatcher._dispatch_table[op_name][context] is instantiated_op

    def test_call_with_dot_notation(self):
        """Test calling operations with dot notation creates and executes a chain."""
        # Create necessary mocks
        module_mock = MagicMock()
        batch_mock = MagicMock(spec=dict)
        analysis_batch_mock = MagicMock(spec=AnalysisBatch)

        # Use patch to verify the chain creation and execution flow
        with patch.object(DISPATCHER, 'create_chain_from_string') as mock_create_chain:
            # Set up the chain mock without spec constraint to avoid signature issues
            chain_mock = MagicMock()
            mock_create_chain.return_value = chain_mock

            # Call with dot notation
            DISPATCHER("op1.op2", module=module_mock, analysis_batch=analysis_batch_mock,
                      batch=batch_mock, batch_idx=0)

            # Verify chain was created with the right string
            mock_create_chain.assert_called_once_with("op1.op2")

            # Verify the chain was called with the expected arguments
            chain_mock.assert_called_once_with(
                module=module_mock,
                analysis_batch=analysis_batch_mock,
                batch=batch_mock,
                batch_idx=0
            )

    def test_top_level_import_hooks(self):
        """Test that operations are properly imported and exposed at the top level."""
        # Test base operations
        assert hasattr(it, "model_forward")

        # With lazy loading, operations are OpWrapper instances until accessed
        # Access an attribute to trigger instantiation
        assert it.model_forward.name == "model_forward"

        # Test chained operations
        assert hasattr(it, "logit_diffs_sae")

        # Access an attribute to trigger instantiation
        assert it.logit_diffs_sae.alias == "logit_diffs_sae"

        # Test chain creation utilities
        assert hasattr(it, "create_op_chain")
        assert hasattr(it, "create_op_chain_from_ops")

    @pytest.mark.parametrize("op_name", [
        "model_forward", "model_cache_forward", "logit_diffs", "sae_correct_acts",
        "logit_diffs_sae", "logit_diffs_attr_grad"
    ])
    def test_op_attributes(self, op_name):
        """Test that each operation has the expected attributes."""
        op = getattr(it, op_name)

        # Access an attribute to trigger instantiation if it's lazily loaded
        # This will ensure the OpWrapper instantiates the actual operation
        _ = op.name

        # Now after accessing an attribute, we should have either the original op wrapper
        # that proxies attribute access or the actual instantiated operation
        assert hasattr(op, "name")
        assert hasattr(op, "description")
        assert hasattr(op, "input_schema")
        assert hasattr(op, "output_schema")
        if (op_alias := getattr(op, 'alias', None)) is not None:
            assert op.alias == op_alias

    def test_integration_with_session(self, request):
        """Test dispatcher integration with an analysis session."""
        try:
            # Use existing analysis session fixture if available
            fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis")

            # Verify the fixture contains expected data
            assert fixture.result is not None
            assert fixture.it_session is not None

            # Verify that the session used a dispatched operation
            test_cfg = fixture.test_cfg()
            assert hasattr(test_cfg, "analysis_cfgs")
            assert hasattr(test_cfg.analysis_cfgs[0], "op")
            assert test_cfg.analysis_cfgs[0].op == it.logit_diffs_sae

        except (LookupError, AttributeError):
            pytest.skip("Required fixture not available, skipping integration test")
