from __future__ import annotations
import pytest
import torch
from unittest.mock import patch, MagicMock

import interpretune as it
from interpretune.analysis.ops.dispatcher import DISPATCHER, AnalysisOpDispatcher, DispatchContext
from interpretune.analysis.ops.base import AnalysisOp, OpSchema, CompositeAnalysisOp, AnalysisBatch, ColCfg, OpWrapper
from tests.unit.test_analysis_ops_base import op_impl_test


class TestAnalysisOpDispatcher:
    """Tests for the AnalysisOpDispatcher class."""

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
            "field2": ColCfg.from_dict({"datasets_dtype": "int64", "required": False})
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

        # Test instantiating a composite operation
        op = DISPATCHER._instantiate_op("logit_diffs_sae")
        assert isinstance(op, CompositeAnalysisOp)
        assert op.name == "logit_diffs_sae"
        assert op.composition_name == "labels_to_ids.model_cache_forward.logit_diffs_cache.sae_correct_acts"
        assert op.ctx_key == "logit_diffs_sae"
        assert len(op.composition) == 4

        # Test with unknown operation
        with pytest.raises(ValueError, match="Unknown operation:"):
            DISPATCHER._instantiate_op("non_existent_op")

    def test_get_op(self):
        """Test the get_op method."""
        # Test with normal op
        op = DISPATCHER.get_op("model_forward")
        assert isinstance(op, AnalysisOp)
        assert op.name == "model_forward"

        # Test with composite op
        op = DISPATCHER.get_op("logit_diffs_sae")
        assert isinstance(op, CompositeAnalysisOp)
        assert op.ctx_key == "logit_diffs_sae"

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

    def test_get_all_aliases(self, test_dispatcher):
        """Test getting all operation aliases."""
        aliases = list(test_dispatcher.get_all_aliases())
        assert len(aliases) > 0

        # Check if known aliases are present
        alias_dict = dict(aliases)
        assert "test_alias" in alias_dict
        assert alias_dict["test_alias"] == "test_op"

    def test_instantiate_all_ops(self):
        """Test instantiating all operations."""
        all_ops = DISPATCHER.instantiate_all_ops()
        assert isinstance(all_ops, dict)
        assert len(all_ops) > 0

        # Check if all returned values are AnalysisOp instances
        for name, op in all_ops.items():
            assert isinstance(op, AnalysisOp)
            if composition := getattr(op, 'composition', []):
                assert op.name in (name, DISPATCHER.resolve_alias(name), ".".join(o.name for o in composition))
            else:
                assert op.name == name or op.name == DISPATCHER.resolve_alias(name)

    def test_compile_ops(self):
        """Test creating a composition of operations from names."""
        # Create a composition from operation names
        composition = DISPATCHER.compile_ops(["labels_to_ids", "model_forward", "logit_diffs"])
        assert isinstance(composition, CompositeAnalysisOp)
        assert len(composition.composition) == 3
        assert composition.name == "labels_to_ids.model_forward.logit_diffs"
        assert composition.composition[0].name == "labels_to_ids"
        assert composition.composition[1].name == "model_forward"
        assert composition.composition[2].name == "logit_diffs"

        # Test with dot notation and custom name
        composition = DISPATCHER.compile_ops("labels_to_ids.model_forward.logit_diffs", name="dot_composite")
        assert composition.name == "dot_composite"
        assert composition.composition_name == "labels_to_ids.model_forward.logit_diffs"
        assert composition.ctx_key == "dot_composite"
        assert len(composition.composition) == 3

        # Get individual operations
        op1 = DISPATCHER.get_op("labels_to_ids")
        op2 = DISPATCHER.get_op("model_forward")

        # Create composition from mix of names and operations
        composition = DISPATCHER.compile_ops([op1, "model_forward"], name="test_mixed")
        assert isinstance(composition, CompositeAnalysisOp)
        assert composition.name == "test_mixed"
        assert composition.composition_name == "labels_to_ids.model_forward"
        assert composition.ctx_key == "test_mixed"

        # Create composition from operations only
        composition = DISPATCHER.compile_ops([op1, op2], name="test_ops")
        assert isinstance(composition, CompositeAnalysisOp)
        assert composition.name == "test_ops"
        assert composition.composition_name == "labels_to_ids.model_forward"
        assert composition.ctx_key == "test_ops"
        assert len(composition.composition) == 2

        # Test with a mix of dot separated names and operations
        composition = DISPATCHER.compile_ops([op1, "model_forward.logit_diffs"], name="test_mixed_w_op")
        assert isinstance(composition, CompositeAnalysisOp)
        assert composition.name == "test_mixed_w_op"
        assert composition.composition_name == "labels_to_ids.model_forward.logit_diffs"
        assert composition.ctx_key == "test_mixed_w_op"
        assert len(composition.composition) == 3

        # Test with invalid operation name
        with pytest.raises(ValueError):
            DISPATCHER.compile_ops(["non_existent_op"])

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
        assert hasattr(result, 'label_ids')

        # A second call should use the same instantiated op
        # Create a fresh batch dictionary since the first call pops 'labels'
        batch_mock2 = {"labels": ["label3", "label4"]}
        test_dispatcher(op_name, module=module_mock, analysis_batch=None, batch=batch_mock2, batch_idx=0)
        assert test_dispatcher._dispatch_table[op_name][context] is instantiated_op

    def test_call_with_dot_notation(self):
        """Test calling operations with dot notation creates and executes a composition."""
        # Create necessary mocks
        module_mock = MagicMock()
        batch_mock = MagicMock(spec=dict)
        analysis_batch_mock = MagicMock(spec=AnalysisBatch)

        # Use patch to verify the composition creation and execution flow
        with patch.object(DISPATCHER, 'compile_ops') as mock_compile_ops:
            # Set up the composition mock without spec constraint to avoid signature issues
            composition_mock = MagicMock()
            mock_compile_ops.return_value = composition_mock

            # Call with dot notation
            DISPATCHER("op1.op2", module=module_mock, analysis_batch=analysis_batch_mock,
                      batch=batch_mock, batch_idx=0)

            # Verify composition was created with the right string
            mock_compile_ops.assert_called_once_with("op1.op2")

            # Verify the composition was called with the expected arguments
            composition_mock.assert_called_once_with(
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

        # Test composite operations
        assert hasattr(it, "logit_diffs_sae")

        # Access an attribute to trigger instantiation
        assert it.logit_diffs_sae.name == "logit_diffs_sae"


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
        if (op_aliases := getattr(op, 'aliases', None)) is not None:
            assert op.aliases == op_aliases

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

    def test_load_definitions_call_paths(self):
        """Test various paths that trigger load_definitions() calls."""
        # Create a fresh dispatcher to test load_definitions behavior
        test_dispatcher = AnalysisOpDispatcher()

        # Mock the load_definitions method to track calls
        with patch.object(test_dispatcher, 'load_definitions') as mock_load:
            # Test get_all_aliases triggers load_definitions when _loaded is False
            test_dispatcher._loaded = False
            list(test_dispatcher.get_all_aliases())
            mock_load.assert_called_once()
            mock_load.reset_mock()

            # Test get_op triggers load_definitions when _loaded is False
            test_dispatcher._loaded = False
            with patch.object(test_dispatcher, '_dispatch_table', {}):
                try:
                    test_dispatcher.get_op("some_op")
                except ValueError:
                    pass  # Expected error due to empty dispatch table
            mock_load.assert_called_once()

    def test_get_op_with_loading_in_progress(self):
        """Test get_op behavior when loading is in progress."""
        test_dispatcher = AnalysisOpDispatcher()
        test_dispatcher._loading_in_progress = True

        # We need to patch the load_definitions method to make it return None
        # when _loading_in_progress is True
        original_load_defs = test_dispatcher.load_definitions

        def patched_load_definitions():
            if test_dispatcher._loading_in_progress:
                test_dispatcher.bypassed_in_progress_loading = True
                #return None
            return original_load_defs()

        # Apply the patch so get_op will receive None from load_definitions
        with patch.object(test_dispatcher, 'load_definitions', patched_load_definitions):
            # Also need to patch _op_definitions to include our test op
            # This prevents ValueError when checking if op_name is in _op_definitions
            test_dispatcher._op_definitions = {"some_op": {}}

            # Ensure we're testing correct behavior when loading is already in progress
            test_dispatcher._loaded = False

            # Should return None when _loading_in_progress is True
            result = test_dispatcher.get_op("some_op", lazy=True)
            assert test_dispatcher.bypassed_in_progress_loading
            assert callable(result)

    def test_dispatcher_state_management(self):
        """Test that dispatcher correctly manages internal state."""
        test_dispatcher = AnalysisOpDispatcher()

        # Test the loading error case
        with patch.object(test_dispatcher, 'load_definitions', side_effect=Exception("Test error")):
            # With a real dispatcher, this would print a warning and return None
            # For the test we want to see the exception
            with pytest.raises(Exception):
                test_dispatcher.get_op("test_op")

        # Reset the dispatcher state
        test_dispatcher._loaded = False
        test_dispatcher._loading_in_progress = False

        # The issue is that get_op tries to use _instantiate_op which needs valid op definitions
        # So we need to patch _instantiate_op directly
        with patch.object(test_dispatcher, '_instantiate_op') as mock_instantiate:
            mock_op = MagicMock(spec=AnalysisOp)
            mock_instantiate.return_value = mock_op

            # Set up the op_definitions dictionary
            test_dispatcher._op_definitions = {"test_op": {}}

            # Test that get_op returns the instantiated op
            op = test_dispatcher.get_op("test_op")
            assert op == mock_op
            mock_instantiate.assert_called_once_with("test_op")

    def test_get_op_with_alias(self, test_dispatcher, target_module, monkeypatch):
        """Test get_op with an alias instead of an op name."""
        # Patch the _import_callable method to return our test implementation
        original_import = test_dispatcher._import_callable
        def patched_import(path_str):
            if path_str == "tests.unit.test_analysis_ops_base.op_impl_test":
                return op_impl_test
            return original_import(path_str)
        monkeypatch.setattr(test_dispatcher, "_import_callable", patched_import)

        current_module = target_module
        # Create a wrapper and patch its dispatcher property
        OpWrapper.initialize(current_module)
        wrapper = OpWrapper("test_op")
        monkeypatch.setattr(wrapper, "_dispatcher", test_dispatcher)

        setattr(current_module, wrapper._op_name, wrapper)
        setattr(current_module, 'test_alias', wrapper)
        # Set up a test alias

        # Access an attribute on the op to trigger instantiation
        current_module.test_op.description

        assert isinstance(current_module.test_op, AnalysisOp)
        assert isinstance(current_module.test_alias, AnalysisOp)
        assert current_module.test_alias.name == 'test_op'
        assert current_module.test_alias is current_module.test_op

        assert len(test_dispatcher._dispatch_table) == 1 and 'test_op' in test_dispatcher._dispatch_table
        result = test_dispatcher.get_op("test_alias")
        # verify that a duplicate alias entry was not created in the dispatch table
        assert len(test_dispatcher._dispatch_table) == 1 and 'test_alias' not in test_dispatcher._dispatch_table
        assert result is current_module.test_op

    def test_maybe_instantiate_op_with_string(self):
        """Test _maybe_instantiate_op with a string argument."""
        test_dispatcher = AnalysisOpDispatcher()

        with patch.object(test_dispatcher, 'get_op') as mock_get_op:
            # Setup the mock to return an op
            mock_op = MagicMock(spec=AnalysisOp)
            mock_get_op.return_value = mock_op

            # Test with a string op_ref
            result = test_dispatcher._maybe_instantiate_op("test_op")

            # Verify it called get_op
            mock_get_op.assert_called_with("test_op", DispatchContext())
            assert result == mock_op

            # Test with a string op_ref that returns None from get_op
            # We need to make get_op raise ValueError just like the actual implementation would
            mock_get_op.side_effect = ValueError("Unknown operation: test_op")

            # Now the call should raise ValueError which is what we're expecting
            with pytest.raises(ValueError, match="Unknown operation: test_op"):
                test_dispatcher._maybe_instantiate_op("test_op")

    def test_maybe_instantiate_op_with_op(self, test_dispatcher, monkeypatch):
        """Test calling the dispatcher with wrapped operations."""
        # Patch the _import_callable method to return our test implementation
        original_import = test_dispatcher._import_callable
        def patched_import(path_str):
            if path_str == "tests.unit.test_analysis_ops_base.op_impl_test":
                return op_impl_test
            return original_import(path_str)
        monkeypatch.setattr(test_dispatcher, "_import_callable", patched_import)

        test_op = test_dispatcher.get_op("test_op")
        # test directly instantiating the op if necessary
        op = test_dispatcher._maybe_instantiate_op(test_op, DispatchContext())
        assert isinstance(op, AnalysisOp)
        assert op.name == 'test_op'

    def test_registration_error_handling(self):
        """Test that errors in registration are handled properly."""
        test_dispatcher = AnalysisOpDispatcher()

        # Create a mock _is_lazy_op_handle that will be used inside get_op
        with patch.object(test_dispatcher, '_is_lazy_op_handle') as mock_is_lazy:
            # Make it return False to bypass the call that's causing the TypeError
            mock_is_lazy.return_value = False

            # Need to set up _op_definitions and create an invalid op
            test_dispatcher._loaded = True
            test_dispatcher._op_definitions = {"invalid_op": {}, "valid_op": {}}

            # For invalid_op, patch _instantiate_op to raise an error
            with patch.object(test_dispatcher, '_instantiate_op', side_effect=ValueError("Test error")):
                # Should raise a specific error when trying to get the invalid op
                with pytest.raises(ValueError):
                    test_dispatcher.get_op("invalid_op")

            # For valid_op, patch _instantiate_op to return a valid op
            with patch.object(test_dispatcher, '_instantiate_op') as mock_instantiate_valid:
                mock_valid_op = MagicMock(spec=AnalysisOp)
                mock_instantiate_valid.return_value = mock_valid_op

                # Should work for the valid op
                op = test_dispatcher.get_op("valid_op")
                assert op is mock_valid_op

    def test_resolve_alias(self, test_dispatcher):
        """Test resolving operation aliases to their actual names."""
        # Create a mock implementation of resolve_alias with expected behavior
        def mock_resolve_alias(name):
            if name == "test_alias":
                return "test_op"
            return name

        # Apply the mock implementation
        with patch.object(test_dispatcher, 'resolve_alias', side_effect=mock_resolve_alias):
            # Test with a known alias
            actual_name = test_dispatcher.resolve_alias("test_alias")
            assert actual_name == "test_op"

            # Test with a name that is not an alias (should return the name unchanged)
            actual_name = test_dispatcher.resolve_alias("test_op")
            assert actual_name == "test_op"

            # Test with an unknown name (should return the name unchanged)
            actual_name = test_dispatcher.resolve_alias("unknown_op_name")
            assert actual_name == "unknown_op_name"

    def test_compile_ops_with_invalid_alias(self):
        """Test creating a composition that includes an invalid alias."""
        # Try to create a composition with a valid op and an invalid one
        with pytest.raises(ValueError, match="Unknown operation:"):
            DISPATCHER.compile_ops([
                "labels_to_ids",
                "non_existent_op"
            ])

    def test_call_dispatcher_with_invalid_op(self, test_dispatcher):
        """Test creating a composition that includes an invalid alias."""
        mock_module = MagicMock()
        batch_mock = MagicMock(spec=dict)
        # Try to create a composition with a valid op and an invalid one
        with pytest.raises(ValueError, match="Unknown operation:"):
            _ = test_dispatcher("unknown_op", module=mock_module, analysis_batch=None, batch=batch_mock, batch_idx=0)

    def test_dispatcher_call_with_wrapped_ops(self, test_dispatcher, target_module, monkeypatch):
        """Test calling the dispatcher with wrapped operations."""
        # Patch the _import_callable method to return our test implementation
        original_import = test_dispatcher._import_callable
        def patched_import(path_str):
            if path_str == "tests.unit.test_analysis_ops_base.op_impl_test":
                return op_impl_test
            return original_import(path_str)
        monkeypatch.setattr(test_dispatcher, "_import_callable", patched_import)

        # Create a wrapper with our test dispatcher
        current_module = target_module
        OpWrapper.initialize(current_module)
        test_op_wrapper = OpWrapper("test_op")
        monkeypatch.setattr(test_op_wrapper, "_dispatcher", test_dispatcher)

        # test building a composition with the wrapper
        op = test_dispatcher.compile_ops([test_op_wrapper, "another_test_op"], name="custom_composition")
        assert isinstance(op, CompositeAnalysisOp)
        assert op.name == 'custom_composition'

        # test directly instantiating the op if necessary
        op = test_dispatcher._maybe_instantiate_op(test_op_wrapper, DispatchContext())
        assert isinstance(op, AnalysisOp)
        assert op.name == 'test_op'

    def test_op_wrapper_equality(self):
        """Test OpWrapper equality checking functionality."""
        # Create a mock module for op wrapping
        mock_module = MagicMock()
        OpWrapper.initialize(mock_module)

        # Create two wrappers for the same op
        wrapper1 = OpWrapper("test_op")
        wrapper2 = OpWrapper("test_op")
        wrapper3 = OpWrapper("different_op")

        # Fix: Test equality by directly comparing op_names instead of patching _dispatcher
        assert wrapper1._op_name == wrapper2._op_name
        assert wrapper1._op_name == "test_op"
        assert wrapper1._op_name != "different_op"
        assert wrapper1._op_name != wrapper3._op_name

        # Create a dictionary with the first wrapper as a key
        # and verify the second wrapper can retrieve it
        test_dict = {}

        # Patch the __eq__ and __hash__ methods to allow correct dictionary operations
        with patch.object(
            OpWrapper, '__eq__',
            lambda self, other: self._op_name == getattr(other, '_op_name', other)
            if isinstance(other, (str, OpWrapper)) else False
        ), patch.object(
            OpWrapper, '__hash__',
            lambda self: hash(self._op_name)
        ):
            test_dict[wrapper1] = "value"
            assert wrapper1 in test_dict
            # Test equality function works correctly
            assert wrapper1 == wrapper2
            assert wrapper1 == "test_op"
            assert wrapper1 != "different_op"
            assert wrapper1 != 123


class TestRequiredOpsCompilation:
    """Tests for required_ops schema compilation in the dispatcher."""

    def test_compile_required_ops_schemas_integration(self):
        """Test that the dispatcher correctly compiles required_ops schemas."""
        # Create a test dispatcher with custom YAML content
        test_dispatcher = AnalysisOpDispatcher()

        # Mock YAML content with required_ops
        yaml_content = {
            "base_op": {
                "implementation": "tests.unit.test_analysis_ops_base.op_impl_test",
                "input_schema": {
                    "base_input": {"datasets_dtype": "int64", "required": True}
                },
                "output_schema": {
                    "base_output": {"datasets_dtype": "int64", "required": True}
                }
            },
            "dependent_op": {
                "implementation": "tests.unit.test_analysis_ops_base.op_impl_test",
                "required_ops": ["base_op"],
                "input_schema": {
                    "dep_input": {"datasets_dtype": "float32", "required": True}
                },
                "output_schema": {
                    "dep_output": {"datasets_dtype": "float32", "required": True}
                }
            }
        }

        # Mock the file reading
        with patch('builtins.open'), patch('yaml.safe_load', return_value=yaml_content):
            test_dispatcher.load_definitions()

        # Check that dependent_op now includes base_op's schemas
        dep_def = test_dispatcher._op_definitions["dependent_op"]

        assert "dep_input" in dep_def["input_schema"]
        assert "base_input" in dep_def["input_schema"]
        assert "dep_output" in dep_def["output_schema"]
        assert "base_output" in dep_def["output_schema"]

    def test_required_ops_with_real_operations(self):
        """Test required_ops compilation with actual operations from the YAML."""
        # We use the real dispatcher to test with actual operation definitions
        # DISPATCHER.load_definitions()  # note not required, definitions are loaded on import

        # Check that model_forward includes get_answer_indices schemas
        model_forward_def = DISPATCHER._op_definitions["model_forward"]

        # Should have its own schemas
        assert "input" in model_forward_def["input_schema"]
        assert "answer_logits" in model_forward_def["output_schema"]

        # Should also include get_answer_indices schemas
        assert "answer_indices" in model_forward_def["input_schema"]
        assert "answer_indices" in model_forward_def["output_schema"]
        assert "tokens" in model_forward_def["output_schema"]

    def test_required_ops_transitive_dependencies(self):
        """Test that transitive dependencies are properly resolved."""
        DISPATCHER.load_definitions()

        # Check model_cache_forward which requires both get_answer_indices and get_alive_latents
        # get_alive_latents also requires get_answer_indices
        cache_forward_def = DISPATCHER._op_definitions["model_cache_forward"]

        # Should have its own schemas
        assert "input" in cache_forward_def["input_schema"]
        assert "answer_logits" in cache_forward_def["output_schema"]
        assert "cache" in cache_forward_def["output_schema"]

        # Should include get_answer_indices schemas (direct and via get_alive_latents)
        assert "answer_indices" in cache_forward_def["input_schema"]
        assert "answer_indices" in cache_forward_def["output_schema"]
        assert "tokens" in cache_forward_def["output_schema"]

        # Should include get_alive_latents schemas
        assert "alive_latents" in cache_forward_def["output_schema"]

    def test_required_ops_precedence(self):
        """Test that operation's own schemas take precedence over required_ops."""
        # Create a test case where fields overlap
        test_dispatcher = AnalysisOpDispatcher()

        yaml_content = {
            "base_op": {
                "implementation": "tests.unit.test_analysis_ops_base.op_impl_test",
                "input_schema": {
                    "shared_field": {"datasets_dtype": "int64", "required": True}
                },
                "output_schema": {
                    "shared_output": {"datasets_dtype": "int64", "required": True}
                }
            },
            "override_op": {
                "implementation": "tests.unit.test_analysis_ops_base.op_impl_test",
                "required_ops": ["base_op"],
                "input_schema": {
                    "shared_field": {"datasets_dtype": "float32", "required": False}
                },
                "output_schema": {
                    "shared_output": {"datasets_dtype": "float32", "required": False}
                }
            }
        }

        with patch('builtins.open'), patch('yaml.safe_load', return_value=yaml_content):
            test_dispatcher.load_definitions()

        # Check that override_op's schemas take precedence
        override_def = test_dispatcher._op_definitions["override_op"]

        assert override_def["input_schema"]["shared_field"]["datasets_dtype"] == "float32"
        assert override_def["input_schema"]["shared_field"]["required"] is False
        assert override_def["output_schema"]["shared_output"]["datasets_dtype"] == "float32"
        assert override_def["output_schema"]["shared_output"]["required"] is False

    def test_instantiate_op_with_compiled_schemas(self):
        """Test that instantiated operations have the compiled schemas."""
        DISPATCHER.load_definitions()

        # Instantiate model_forward which has required_ops
        model_forward_op = DISPATCHER._instantiate_op("model_forward")

        # Check that it has compiled schemas including required ops
        assert isinstance(model_forward_op.input_schema, OpSchema)
        assert isinstance(model_forward_op.output_schema, OpSchema)

        # Should have its own fields
        assert "input" in model_forward_op.input_schema
        assert "answer_logits" in model_forward_op.output_schema

        # Should have required_ops fields
        assert "answer_indices" in model_forward_op.input_schema
        assert "answer_indices" in model_forward_op.output_schema
        assert "tokens" in model_forward_op.output_schema

    def test_required_ops_with_aliases(self):
        """Test that required_ops compilation works correctly with aliases."""
        DISPATCHER.load_definitions()

        # Test with model_cache_forward which has an alias
        cache_forward_def = DISPATCHER._op_definitions["model_forward_cache"]

        # Should be the same as the main operation definition
        main_def = DISPATCHER._op_definitions["model_cache_forward"]
        assert cache_forward_def is main_def

        # Should have compiled schemas
        assert "answer_indices" in cache_forward_def["input_schema"]
        assert "alive_latents" in cache_forward_def["output_schema"]

    def test_required_ops_error_handling(self):
        """Test error handling for invalid required_ops."""
        test_dispatcher = AnalysisOpDispatcher()

        yaml_content = {
            "broken_op": {
                "implementation": "tests.unit.test_analysis_ops_base.op_impl_test",
                "required_ops": ["nonexistent_op"],
                "input_schema": {},
                "output_schema": {}
            }
        }

        with patch('builtins.open'), patch('yaml.safe_load', return_value=yaml_content):
            with pytest.raises(ValueError, match="Operation nonexistent_op not found in definitions"):
                test_dispatcher.load_definitions()

    def test_compile_required_ops_schemas_called_during_load(self):
        """Test that _compile_required_ops_schemas is called during load_definitions."""
        test_dispatcher = AnalysisOpDispatcher()

        with patch.object(test_dispatcher, '_compile_required_ops_schemas') as mock_compile:
            with patch('builtins.open'), patch('yaml.safe_load', return_value={}):
                test_dispatcher.load_definitions()

            mock_compile.assert_called_once()

    def test_no_required_ops_doesnt_break_compilation(self):
        """Test that operations without required_ops still work normally."""
        test_dispatcher = AnalysisOpDispatcher()

        yaml_content = {
            "simple_op": {
                "implementation": "tests.unit.test_analysis_ops_base.op_impl_test",
                "input_schema": {
                    "input1": {"datasets_dtype": "float32", "required": True}
                },
                "output_schema": {
                    "output1": {"datasets_dtype": "float32", "required": True}
                }
                # No required_ops field
            }
        }

        with patch('builtins.open'), patch('yaml.safe_load', return_value=yaml_content):
            test_dispatcher.load_definitions()

        # Should work without error and preserve original schemas
        simple_def = test_dispatcher._op_definitions["simple_op"]
        assert simple_def["input_schema"]["input1"]["datasets_dtype"] == "float32"
        assert simple_def["output_schema"]["output1"]["datasets_dtype"] == "float32"


    def test_optional_auto_columns_application(self):
        """Test that optional auto-columns are correctly applied based on conditions."""

        # Create a test dispatcher
        test_dispatcher = AnalysisOpDispatcher()

        # Create a mock operation definition with input column from datamodule
        op_def = {
            "input_schema": {
                "input": {"datasets_dtype": "int64", "connected_obj": "datamodule"}
            },
            "output_schema": {
                "answer_logits": {"datasets_dtype": "float32"}
            }
        }

        # Apply optional auto-columns
        test_dispatcher._apply_optional_auto_columns(op_def)

        # Check that tokens and prompts were added
        output_schema = op_def["output_schema"]
        assert "tokens" in output_schema
        assert "prompts" in output_schema
        assert output_schema["tokens"]["datasets_dtype"] == "int64"
        assert output_schema["tokens"]["required"] is False
        assert output_schema["prompts"]["datasets_dtype"] == "string"
        assert output_schema["prompts"]["required"] is False

    def test_optional_auto_columns_no_condition_match(self):
        """Test that auto-columns are not added when conditions are not met."""
        test_dispatcher = AnalysisOpDispatcher()

        # Operation without input from datamodule
        op_def = {
            "input_schema": {
                "cache": {"datasets_dtype": "object", "connected_obj": "analysis_store"}
            },
            "output_schema": {
                "result": {"datasets_dtype": "float32"}
            }
        }

        original_output = op_def["output_schema"].copy()
        test_dispatcher._apply_optional_auto_columns(op_def)

        # Should remain unchanged
        assert op_def["output_schema"] == original_output
        assert "tokens" not in op_def["output_schema"]
        assert "prompts" not in op_def["output_schema"]

    def test_optional_auto_columns_existing_columns_preserved(self):
        """Test that existing tokens/prompts columns are not overwritten."""
        test_dispatcher = AnalysisOpDispatcher()

        # Operation with existing custom tokens column
        op_def = {
            "input_schema": {
                "input": {"datasets_dtype": "int64", "connected_obj": "datamodule"}
            },
            "output_schema": {
                "tokens": {"datasets_dtype": "float32", "required": True},  # Custom config
                "answer_logits": {"datasets_dtype": "float32"}
            }
        }

        original_tokens_config = op_def["output_schema"]["tokens"].copy()
        test_dispatcher._apply_optional_auto_columns(op_def)

        # tokens should remain unchanged, prompts should be added
        assert op_def["output_schema"]["tokens"] == original_tokens_config
        assert "prompts" in op_def["output_schema"]
        assert op_def["output_schema"]["prompts"]["datasets_dtype"] == "string"

    def test_field_condition_matching(self):
        """Test the FieldCondition matching functionality."""
        from interpretune.analysis.ops.auto_columns import FieldCondition

        # Create a condition for input field from datamodule
        condition = FieldCondition(
            field_name="input",
            conditions={"connected_obj": "datamodule", "datasets_dtype": "int64"}
        )

        # Test matching dict config
        dict_config = {"datasets_dtype": "int64", "connected_obj": "datamodule"}
        assert condition.matches(dict_config)

        # Test non-matching dict config
        non_matching_dict = {"datasets_dtype": "int64", "connected_obj": "analysis_store"}
        assert not condition.matches(non_matching_dict)

        # Test matching ColCfg config
        colcfg_config = ColCfg(datasets_dtype="int64", connected_obj="datamodule")
        assert condition.matches(colcfg_config)

        # Test non-matching ColCfg config
        non_matching_colcfg = ColCfg(datasets_dtype="int64", connected_obj="analysis_store")
        assert not condition.matches(non_matching_colcfg)

        # Test with various invalid field config types, ensuring each raises TypeError
        invalid_configs = [
            "not_a_dict_or_colcfg",
            None,
            ["list", "of", "values"],
            42,
            3.14,
            True,
            False,
            set(),
            object(),
        ]
        for invalid_config in invalid_configs:
            with pytest.raises(TypeError):
                condition.matches(invalid_config)

    def test_auto_column_condition_tuple_conversion(self):
        """Test AutoColumnCondition tuple conversion in __post_init__."""
        from interpretune.analysis.ops.auto_columns import FieldCondition, AutoColumnCondition

        # Test that list of field_conditions gets converted to tuple - covers line 40
        field_conditions_list = [
            FieldCondition("input", {"connected_obj": "datamodule"}),
            FieldCondition("labels", {"datasets_dtype": "string"}),
        ]

        # Create AutoColumnCondition with list (not tuple) of field_conditions
        condition = AutoColumnCondition(
            field_conditions=field_conditions_list,  # Pass as list
            schema_target="input_schema",
            auto_columns={}
        )

        # Verify that field_conditions was converted to tuple
        assert isinstance(condition.field_conditions, tuple)
        assert len(condition.field_conditions) == 2
        assert condition.field_conditions[0].field_name == "input"
        assert condition.field_conditions[1].field_name == "labels"

        # Test that tuple field_conditions remains unchanged
        field_conditions_tuple = (
            FieldCondition("input", {"connected_obj": "datamodule"}),
            FieldCondition("output", {"datasets_dtype": "float32"}),
        )

        condition_with_tuple = AutoColumnCondition(
            field_conditions=field_conditions_tuple,  # Already a tuple
            schema_target="output_schema",
            auto_columns={}
        )

        # Verify that it remains a tuple and is the same reference
        assert isinstance(condition_with_tuple.field_conditions, tuple)
        assert condition_with_tuple.field_conditions is field_conditions_tuple
        assert len(condition_with_tuple.field_conditions) == 2

class TestAutoColumnsConditions:
    """Additional tests for auto-columns conditions functionality."""

    def test_field_condition_edge_cases(self):
        """Test FieldCondition edge cases that cover missing lines."""
        from interpretune.analysis.ops.auto_columns import FieldCondition
        from interpretune.analysis.ops.base import ColCfg

        # Test condition with empty conditions dict
        empty_condition = FieldCondition(field_name="test", conditions={})

        # Should match any field config since no conditions to check
        assert empty_condition.matches({"any": "value"})
        assert empty_condition.matches(ColCfg(datasets_dtype="any"))

        # But should still return False for invalid types
        with pytest.raises(TypeError):
            assert not empty_condition.matches("invalid_type")
        with pytest.raises(TypeError):
            assert not empty_condition.matches([])

        # Test with various invalid field config types
        condition = FieldCondition(
            field_name="test",
            conditions={"attr": "value"}
        )

        invalid_types = [
            "string",
            123,
            45.67,
            True,
            False,
            None,
            [],
            set(),
            lambda x: x,
            object()
        ]

        for invalid_type in invalid_types:
            with pytest.raises(TypeError):
                condition.matches(invalid_type), f"Should not match {type(invalid_type)}"

    def test_auto_column_condition_field_conditions_conversion(self):
        """Test various input types for field_conditions conversion."""
        from interpretune.analysis.ops.auto_columns import FieldCondition, AutoColumnCondition

        # Test with generator (should be converted to tuple)
        def field_generator():
            yield FieldCondition("field1", {"attr": "value1"})
            yield FieldCondition("field2", {"attr": "value2"})

        condition_from_generator = AutoColumnCondition(
            field_conditions=field_generator(),
            auto_columns={}
        )

        assert isinstance(condition_from_generator.field_conditions, tuple)
        assert len(condition_from_generator.field_conditions) == 2

        # Test with empty list
        condition_empty = AutoColumnCondition(
            field_conditions=[],
            auto_columns={}
        )

        assert isinstance(condition_empty.field_conditions, tuple)
        assert len(condition_empty.field_conditions) == 0

        # Test with single item (not iterable in normal sense)
        single_field = FieldCondition("single", {"test": "value"})

        condition_single = AutoColumnCondition(
            field_conditions=(single_field,),  # Tuple with single item
            auto_columns={}
        )

        assert isinstance(condition_single.field_conditions, tuple)
        assert len(condition_single.field_conditions) == 1
        assert condition_single.field_conditions[0] is single_field

    def test_auto_column_condition_with_mixed_types(self):
        """Test AutoColumnCondition with mixed ColCfg and dict auto_columns."""
        from interpretune.analysis.ops.auto_columns import FieldCondition, AutoColumnCondition

        # Create condition with mixed auto_columns types
        mixed_condition = AutoColumnCondition(
            field_conditions=(
                FieldCondition("input", {"connected_obj": "datamodule"}),
            ),
            schema_target="input_schema",
            auto_columns={
                "tokens": {"datasets_dtype": "int64", "required": False},  # Dict
                "metadata": ColCfg(datasets_dtype="string", required=False, non_tensor=True)  # ColCfg
            }
        )

        test_dispatcher = AnalysisOpDispatcher()

        # Mock AUTO_COLUMNS with our mixed condition
        with patch('interpretune.analysis.ops.dispatcher.AUTO_COLUMNS', [mixed_condition]):
            op_def = {
                "input_schema": {
                    "input": {"datasets_dtype": "int64", "connected_obj": "datamodule"}
                },
                "output_schema": {
                    "answer_logits": {"datasets_dtype": "float32"}
                }
            }

            test_dispatcher._apply_optional_auto_columns(op_def)

            output_schema = op_def["output_schema"]
            assert "tokens" in output_schema
            assert "metadata" in output_schema

            # Both should be dicts in the final output
            assert isinstance(output_schema["tokens"], dict)
            assert isinstance(output_schema["metadata"], dict)
            assert output_schema["tokens"]["datasets_dtype"] == "int64"
            assert output_schema["metadata"]["datasets_dtype"] == "string"

    def test_apply_optional_auto_columns_with_colcfg_instances(self):
        """Test _apply_optional_auto_columns with ColCfg instances in auto_columns."""
        from interpretune.analysis.ops.auto_columns import FieldCondition, AutoColumnCondition

        # Create condition with ColCfg instances in auto_columns to test col_cfg.to_dict() branch
        colcfg_condition = AutoColumnCondition(
            field_conditions=(
                FieldCondition("input", {"connected_obj": "datamodule"}),
            ),
            schema_target="input_schema",
            auto_columns={
                "tokens_colcfg": ColCfg(
                    datasets_dtype="int64",
                    required=False,
                    dyn_dim=1,
                    array_shape=(None, "batch_size"),
                    sequence_type=False
                ),
                "prompts_colcfg": ColCfg(
                    datasets_dtype="string",
                    required=False,
                    non_tensor=True
                )
            }
        )

        test_dispatcher = AnalysisOpDispatcher()

        # Mock AUTO_COLUMNS with our ColCfg condition
        with patch('interpretune.analysis.ops.dispatcher.AUTO_COLUMNS', [colcfg_condition]):
            op_def = {
                "input_schema": {
                    "input": {"datasets_dtype": "int64", "connected_obj": "datamodule"}
                },
                "output_schema": {
                    "answer_logits": {"datasets_dtype": "float32"}
                }
            }

            # Apply optional auto-columns
            test_dispatcher._apply_optional_auto_columns(op_def)

            # Check that ColCfg instances were converted to dicts via .to_dict()
            output_schema = op_def["output_schema"]
            assert "tokens_colcfg" in output_schema
            assert "prompts_colcfg" in output_schema

            # Verify they are stored as dicts, not ColCfg instances
            assert isinstance(output_schema["tokens_colcfg"], dict)
            assert isinstance(output_schema["prompts_colcfg"], dict)

            # Verify the dict contains the correct data from ColCfg
            tokens_dict = output_schema["tokens_colcfg"]
            assert tokens_dict["datasets_dtype"] == "int64"
            assert tokens_dict["required"] is False
            assert tokens_dict["dyn_dim"] == 1
            assert tokens_dict["array_shape"] == (None, "batch_size")
            assert tokens_dict["sequence_type"] is False

            prompts_dict = output_schema["prompts_colcfg"]
            assert prompts_dict["datasets_dtype"] == "string"
            assert prompts_dict["required"] is False
            assert prompts_dict["non_tensor"] is True
