import pytest
from unittest.mock import patch

from tests.warns import unmatched_warns
from interpretune.analysis.ops.compiler.schema_compiler import resolve_required_ops, compile_op_schema



class TestRequiredOpsResolution:
    """Test the required_ops resolution logic."""

    def test_exact_match_resolution(self):
        """Test that exact matches are found correctly."""
        op_definitions = {
            'gradient_attribution': {
                'required_ops': ['get_alive_latents']
            },
            'get_alive_latents': {
                'description': 'Extract alive latents'
            }
        }

        op_def = op_definitions['gradient_attribution']
        resolved = resolve_required_ops('gradient_attribution', op_def, op_definitions)

        assert resolved == ['get_alive_latents']

    def test_basename_resolution_single_match(self):
        """Test basename resolution when single namespaced match exists."""
        op_definitions = {
            'testuser.test.another_op': {
                'required_ops': ['some_op_req']
            },
            'testuser.test.some_op_req': {
                'description': 'A test operation from hub'
            }
        }

        op_def = op_definitions['testuser.test.another_op']
        resolved = resolve_required_ops('testuser.test.another_op', op_def, op_definitions)

        assert resolved == ['testuser.test.some_op_req']

    def test_basename_resolution_multiple_matches_with_warning(self):
        """Test basename resolution when multiple matches exist - should warn and use first."""
        op_definitions = {
            'testuser.test.another_op': {
                'required_ops': ['some_op_req']
            },
            'anotheruser.test.some_op_req': {
                'description': 'A test operation from hub'
            },
            'testuser.test.some_op_req': {
                'description': 'Another test operation from hub'
            }
        }

        op_def = op_definitions['testuser.test.another_op']

        with patch('interpretune.analysis.ops.compiler.schema_compiler.rank_zero_warn') as mock_warn:
            resolved = resolve_required_ops('testuser.test.another_op', op_def, op_definitions)

            # Should use first match alphabetically
            assert resolved == ['anotheruser.test.some_op_req']

            # Should have issued warning
            mock_warn.assert_called_once()
            warning_msg = mock_warn.call_args[0][0]
            assert 'multiple matching operations found' in warning_msg
            assert 'Consider using a fully-qualified name' in warning_msg

    def test_prioritize_exact_match_over_basename(self):
        """Test that exact matches are prioritized over basename matches."""
        op_definitions = {
            'testuser.test.another_op': {
                'required_ops': ['some_op_req']
            },
            'some_op_req': {
                'description': 'Exact match operation'
            },
            'anotheruser.test.some_op_req': {
                'description': 'Basename match operation'
            }
        }

        op_def = op_definitions['testuser.test.another_op']
        resolved = resolve_required_ops('testuser.test.another_op', op_def, op_definitions)

        # Should use exact match, not basename match
        assert resolved == ['some_op_req']

    def test_no_match_raises_error(self):
        """Test that missing required_ops raise ValueError."""
        op_definitions = {
            'testuser.test.another_op': {
                'required_ops': ['nonexistent_op']
            }
        }

        op_def = op_definitions['testuser.test.another_op']

        with pytest.raises(ValueError, match="Required operation 'nonexistent_op' not found"):
            resolve_required_ops('testuser.test.another_op', op_def, op_definitions)


    def test_empty_required_ops(self):
        """Test that operations with no required_ops work correctly."""
        op_definitions = {
            'simple_op': {
                'description': 'Simple operation with no dependencies'
            }
        }

        op_def = op_definitions['simple_op']
        resolved = resolve_required_ops('simple_op', op_def, op_definitions)

        assert resolved == []

    def test_multiple_required_ops_mixed_resolution(self):
        """Test resolution of multiple required_ops with different match types."""
        op_definitions = {
            'complex_op': {
                'required_ops': ['exact_match', 'basename_match']
            },
            'exact_match': {
                'description': 'Exact match operation'
            },
            'namespace.basename_match': {
                'description': 'Basename match operation'
            }
        }

        op_def = op_definitions['complex_op']
        resolved = resolve_required_ops('complex_op', op_def, op_definitions)

        assert resolved == ['exact_match', 'namespace.basename_match']


class TestCompileOpSchema:
    """Tests for the compile_op_schema function."""

    def test_compile_op_schema_basic(self):
        """Test basic schema compilation without dependencies."""
        op_definitions = {
            "simple_op": {
                "input_schema": {"input1": {"datasets_dtype": "float32"}},
                "output_schema": {"output1": {"datasets_dtype": "float32"}},
            }
        }

        # Compile the operation
        compiled_def = compile_op_schema("simple_op", op_definitions)

        # Should remain unchanged since no required_ops
        assert compiled_def["input_schema"] == {"input1": {"datasets_dtype": "float32"}}
        assert compiled_def["output_schema"] == {"output1": {"datasets_dtype": "float32"}}

    def test_compile_op_schema_with_required_ops(self):
        """Test that compile_op_schema properly merges schemas from required operations."""
        from interpretune.analysis.ops.compiler.schema_compiler import compile_op_schema

        # Create base operation definition
        base_op_def = {
            "input_schema": {
                "base_input": {"datasets_dtype": "int64", "required": True}
            },
            "output_schema": {
                "base_output": {"datasets_dtype": "int64", "required": True}
            }
        }

        # Create dependent operation definition with required_ops
        dependent_op_def = {
            "required_ops": ["base_op"],
            "input_schema": {
                "dep_input": {"datasets_dtype": "float32", "required": True}
            },
            "output_schema": {
                "dep_output": {"datasets_dtype": "float32", "required": True}
            }
        }

        # Create operation definitions dict
        op_definitions = {
            "base_op": base_op_def,
            "dependent_op": dependent_op_def
        }

        # Compile the dependent operation schema
        compiled_schema = compile_op_schema("dependent_op", op_definitions)

        # Verify that both input and output schemas are properly merged
        dep_input = compiled_schema["input_schema"]
        dep_output = compiled_schema["output_schema"]

        # Should have both dependent and base operation fields
        assert "dep_input" in dep_input
        assert "base_input" in dep_input
        assert "dep_output" in dep_output
        assert "base_output" in dep_output

    def test_compile_op_schema_precedence(self):
        """Test that existing fields take precedence over required op fields."""
        op_definitions = {
            "base_op": {
                "input_schema": {"shared_field": {"datasets_dtype": "int64"}},
                "output_schema": {"shared_output": {"datasets_dtype": "int64"}},
            },
            "dependent_op": {
                "required_ops": ["base_op"],
                "input_schema": {"shared_field": {"datasets_dtype": "float32"}},
                "output_schema": {"shared_output": {"datasets_dtype": "float32"}},
            }
        }

        # Compile the dependent operation
        compiled_schema = compile_op_schema("dependent_op", op_definitions)

        # Check that dependent_op's original fields take precedence
        dep_input = compiled_schema["input_schema"]
        dep_output = compiled_schema["output_schema"]

        assert dep_input["shared_field"]["datasets_dtype"] == "float32"
        assert dep_output["shared_output"]["datasets_dtype"] == "float32"

    def test_compile_op_schema_multiple_dependencies(self):
        """Test schema compilation with multiple required operations."""
        op_definitions = {
            "op1": {
                "input_schema": {"input1": {"datasets_dtype": "int64"}},
                "output_schema": {"output1": {"datasets_dtype": "int64"}},
            },
            "op2": {
                "input_schema": {"input2": {"datasets_dtype": "float32"}},
                "output_schema": {"output2": {"datasets_dtype": "float32"}},
            },
            "dependent_op": {
                "required_ops": ["op1", "op2"],
                "input_schema": {"dep_input": {"datasets_dtype": "string"}},
                "output_schema": {"dep_output": {"datasets_dtype": "string"}},
            }
        }

        # Compile the dependent operation
        compiled_schema = compile_op_schema("dependent_op", op_definitions)

        # Check that all schemas were merged
        dep_input = compiled_schema["input_schema"]
        dep_output = compiled_schema["output_schema"]

        expected_input_fields = {"dep_input", "input1", "input2"}
        expected_output_fields = {"dep_output", "output1", "output2", "input1", "input2"}
        expected_required_false_fields = {"input1", "input2", "output1", "output2"}

        assert set(dep_input.keys()) == expected_input_fields
        assert set(dep_output.keys()) == expected_output_fields
        # Check that required fields are set correctly
        for field in expected_required_false_fields:
            assert dep_output[field]["required"] is False

    def test_compile_op_schema_transitive_dependencies(self):
        """Test schema compilation with transitive dependencies."""
        op_definitions = {
            "base_op": {
                "input_schema": {"base_input": {"datasets_dtype": "int64"}},
                "output_schema": {"base_output": {"datasets_dtype": "int64"}},
            },
            "middle_op": {
                "required_ops": ["base_op"],
                "input_schema": {"middle_input": {"datasets_dtype": "float32"}},
                "output_schema": {"middle_output": {"datasets_dtype": "float32"}},
            },
            "top_op": {
                "required_ops": ["middle_op"],
                "input_schema": {"top_input": {"datasets_dtype": "string"}},
                "output_schema": {"top_output": {"datasets_dtype": "string"}},
            }
        }

        # Compile the top operation
        compiled_schema = compile_op_schema("top_op", op_definitions)

        # Check that all transitive dependencies were included
        top_input = compiled_schema["input_schema"]
        top_output = compiled_schema["output_schema"]

        expected_input_fields = {"top_input", "middle_input", "base_input"}
        expected_output_fields = {"top_output", "middle_output", "base_output", "base_input", "middle_input"}
        expected_required_false_fields = {"base_input", "middle_input", "base_output", "middle_output"}

        assert set(top_input.keys()) == expected_input_fields
        assert set(top_output.keys()) == expected_output_fields
        for field in expected_required_false_fields:
            assert top_output[field]["required"] is False

    def test_compile_op_schema_missing_main_operation(self):
        """Test error handling for missing main operation."""
        op_definitions = {}

        with pytest.raises(ValueError, match="Operation missing_op not found in definitions"):
            compile_op_schema("missing_op", op_definitions)

    def test_compile_op_schema_no_required_ops(self):
        """Test compilation of operation without required_ops field."""
        op_definitions = {
            "simple_op": {
                "input_schema": {"input1": {"datasets_dtype": "float32"}},
                "output_schema": {"output1": {"datasets_dtype": "float32"}},
                # No required_ops field
            }
        }

        # Should work without error
        compiled_schema = compile_op_schema("simple_op", op_definitions)

        # Schema should remain unchanged
        assert compiled_schema["input_schema"] == {"input1": {"datasets_dtype": "float32"}}
        assert compiled_schema["output_schema"] == {"output1": {"datasets_dtype": "float32"}}

    def test_compile_op_schema_empty_schemas(self):
        """Test compilation with empty input/output schemas."""
        op_definitions = {
            "base_op": {
                "input_schema": {"base_input": {"datasets_dtype": "int64"}},
                "output_schema": {"base_output": {"datasets_dtype": "int64"}},
            },
            "empty_schema_op": {
                "required_ops": ["base_op"],
                # Empty or missing schemas
            }
        }

        # Compile the operation with empty schemas
        compiled_schema = compile_op_schema("empty_schema_op", op_definitions)

        # Should have inherited schemas from base_op
        assert "base_input" in compiled_schema["input_schema"]
        assert "base_output" in compiled_schema["output_schema"]

    def test_compile_op_schema_with_compiled_set(self):
        """Test that already compiled operations are not reprocessed."""
        op_definitions = {
            "base_op": {
                "input_schema": {"base_input": {"datasets_dtype": "int64"}},
                "output_schema": {"base_output": {"datasets_dtype": "int64"}},
            },
            "dependent_op": {
                "required_ops": ["base_op"],
                "input_schema": {"dep_input": {"datasets_dtype": "float32"}},
                "output_schema": {"dep_output": {"datasets_dtype": "float32"}},
            }
        }

        # Compile base_op first
        compiled_base = compile_op_schema("base_op", op_definitions)
        assert "base_input" in compiled_base["input_schema"]
        assert "base_output" in compiled_base["output_schema"]

        # Now compile dependent_op - base_op should be processed as part of dependencies
        compiled_dependent = compile_op_schema("dependent_op", op_definitions)

        # Should have both dependent and base operation fields
        assert "dep_input" in compiled_dependent["input_schema"]
        assert "base_input" in compiled_dependent["input_schema"]
        assert "dep_output" in compiled_dependent["output_schema"]
        assert "base_output" in compiled_dependent["output_schema"]

class TestCompileOpSchemaIntegration:
    """Test that compile_op_schema properly integrates required_ops resolution."""

    def test_compile_with_resolved_required_ops(self):
        """Test that compile_op_schema uses resolved required_ops."""
        op_definitions = {
            'parent_op': {
                'description': 'Parent operation',
                'required_ops': ['child_op'],
                'input_schema': {},
                'output_schema': {'parent_output': {'datasets_dtype': 'float32'}}
            },
            'namespace.child_op': {
                'description': 'Child operation',
                'input_schema': {'child_input': {'datasets_dtype': 'int64'}},
                'output_schema': {'child_output': {'datasets_dtype': 'string'}}
            }
        }

        compile_op_schema('parent_op', op_definitions)

        # Check that required_ops was resolved
        assert op_definitions['parent_op']['required_ops'] == ['namespace.child_op']

        # Check that schemas were merged correctly
        parent_input = op_definitions['parent_op']['input_schema']
        parent_output = op_definitions['parent_op']['output_schema']
        # Our contract currently requires that parent ops have required input fields of their required_ops child ops
        # this may be made configurable in the future (since we could change our required_ops semantics to make ops
        # responsible for internally generating the input fields for the required_ops)
        # if actual usage patterns dictate we change the semantics of required_ops we can do so, but
        # this seems like a sensible place to start semantically until we see what patterns and preferences emerge
        assert 'child_input' in parent_input
        assert 'parent_output' in parent_output
        # non-overridden child input and output keys should not be required in the parent output schema since
        # our contract currently allows them to be pruned
        assert parent_output['child_output']['required'] is False
        assert parent_output['child_input']['required'] is False

    def test_compile_op_not_found_raises_error(self):
        """Test that compiling a non-existent operation raises ValueError."""
        op_definitions = {}

        with pytest.raises(ValueError, match="Operation nonexistent_op not found in definitions"):
            compile_op_schema('nonexistent_op', op_definitions)

    def test_compile_op_circular_dependency_raises_error(self):
        """Test that circular dependencies raise ValueError."""
        op_definitions = {
            'op_a': {
                'required_ops': ['op_b'],
                'input_schema': {},
                'output_schema': {}
            },
            'op_b': {
                'required_ops': ['op_a'],
                'input_schema': {},
                'output_schema': {}
            }
        }

        with pytest.raises(ValueError, match="Circular dependency detected"):
            compile_op_schema('op_a', op_definitions)

    def test_compile_op_unresolvable_required_ops_raises_error(self):
        """Test that unresolvable required_ops raise ValueError."""
        op_definitions = {
            'parent_op': {
                'required_ops': ['nonexistent_child'],
                'input_schema': {},
                'output_schema': {}
            }
        }

        with pytest.raises(ValueError, match="cannot be compiled due to unresolved required operations"):
            compile_op_schema('parent_op', op_definitions)

    def test_compile_op_required_op_not_found_raises_error(self):
        """Test that missing required operation in definitions raises ValueError."""
        op_definitions = {
            'parent_op': {
                'required_ops': ['missing_child'],
                'input_schema': {},
                'output_schema': {}
            }
        }

        # First resolve_required_ops will fail and raise ValueError
        with pytest.raises(ValueError, match="cannot be compiled due to unresolved required operations"):
            compile_op_schema('parent_op', op_definitions)

    def test_compile_ops_schemas_error_handling(self, recwarn):
        """Test that _compile_required_ops_schemas handles errors and continues processing."""
        from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

        # Create a dispatcher instance to test the private method
        dispatcher = AnalysisOpDispatcher()

        op_definitions = {
            'good_op': {
                'description': 'A working operation',
                'input_schema': {'input1': {'datasets_dtype': 'float32'}},
                'output_schema': {'output1': {'datasets_dtype': 'string'}}
            },
            'bad_op_unresolvable': {
                'description': 'Operation with unresolvable dependencies',
                'required_ops': ['nonexistent_op'],
                'input_schema': {},
                'output_schema': {}
            },
            'bad_op_circular_a': {
                'description': 'Operation in circular dependency',
                'required_ops': ['bad_op_circular_b'],
                'input_schema': {},
                'output_schema': {}
            },
            'bad_op_circular_b': {
                'description': 'Operation in circular dependency',
                'required_ops': ['bad_op_circular_a'],
                'input_schema': {},
                'output_schema': {}
            },
            'another_good_op': {
                'description': 'Another working operation',
                'input_schema': {'input2': {'datasets_dtype': 'int64'}},
                'output_schema': {'output2': {'datasets_dtype': 'float32'}}
            }
        }

        original_count = len(op_definitions)

        # This should not raise an error, just issue warnings and remove bad ops
        dispatcher._compile_required_ops_schemas(op_definitions)

        w_expected = [
            ".*Operation 'bad_op_unresolvable' cannot be compiled due to unresolved required operations.*",
            ".*Circular dependency detected.* bad_op_circular_a.*",
            ".*Operation 'bad_op_circular_b' cannot be compiled due to unresolved required operations.*"
        ]
        unmatched = unmatched_warns(rec_warns=recwarn.list, expected_warns=w_expected)
        assert not unmatched

        assert len(op_definitions) == original_count - len(w_expected)  # one warning per skipped operation

        # Bad operations should be removed from definitions
        assert 'bad_op_unresolvable' not in op_definitions
        assert 'bad_op_circular_a' not in op_definitions
        assert 'bad_op_circular_b' not in op_definitions

        # Good operations should remain
        assert 'good_op' in op_definitions
        assert 'another_good_op' in op_definitions


    def test_compile_deep_circular_dependency_detection(self):
        """Test detection of circular dependencies in deep chains."""
        op_definitions = {
            'op_a': {
                'required_ops': ['op_b'],
                'input_schema': {},
                'output_schema': {}
            },
            'op_b': {
                'required_ops': ['op_c'],
                'input_schema': {},
                'output_schema': {}
            },
            'op_c': {
                'required_ops': ['op_a'],  # Creates cycle: a -> b -> c -> a
                'input_schema': {},
                'output_schema': {}
            }
        }

        with pytest.raises(ValueError, match="Circular dependency detected"):
            compile_op_schema('op_a', op_definitions)

    def test_compile_with_multiple_resolution_warnings(self):
        """Test compilation with multiple required_ops resolution warnings."""
        op_definitions = {
            'parent_op': {
                'required_ops': ['ambiguous_op'],
                'input_schema': {},
                'output_schema': {}
            },
            'namespace1.ambiguous_op': {
                'description': 'First ambiguous operation',
                'input_schema': {'input1': {'datasets_dtype': 'float32'}},
                'output_schema': {'output1': {'datasets_dtype': 'string'}}
            },
            'namespace2.ambiguous_op': {
                'description': 'Second ambiguous operation',
                'input_schema': {'input2': {'datasets_dtype': 'int64'}},
                'output_schema': {'output2': {'datasets_dtype': 'float32'}}
            }
        }

        with patch('interpretune.analysis.ops.compiler.schema_compiler.rank_zero_warn') as mock_warn:
            compile_op_schema('parent_op', op_definitions)

            # Should resolve to first alphabetical match
            assert op_definitions['parent_op']['required_ops'] == ['namespace1.ambiguous_op']

            # Should have issued warning about multiple matches
            assert mock_warn.call_count >= 1
            warning_msg = mock_warn.call_args_list[0][0][0]
            assert 'multiple matching operations found' in warning_msg

    def test_complex_dependency_resolution_scenario(self):
        """Test the complex scenario described in the requirements."""
        op_definitions = {
            'testuser.test.another_op': {
                'description': 'Another test operation',
                'required_ops': ['some_op_req'],
                'input_schema': {},
                'output_schema': {}
            },
            'some_op_req': {
                'description': 'A test operation from hub',
                'input_schema': {},
                'output_schema': {'some_output': {'datasets_dtype': 'float32'}}
            },
            'anotheruser.test.some_op_req': {
                'description': 'A test operation from hub',
                'input_schema': {},
                'output_schema': {}
            },
            'testuser.test.some_op_req': {
                'aliases': ['testuser.test.hub_test'],
                'description': 'A test operation from hub',
                'input_schema': {},
                'output_schema': {}
            }
        }

        # Should resolve to exact match 'some_op_req'
        compile_op_schema('testuser.test.another_op', op_definitions)

        assert op_definitions['testuser.test.another_op']['required_ops'] == ['some_op_req']

        # Now test scenario 2: remove exact match
        del op_definitions['some_op_req']

        # Reset the required_ops to test resolution again
        op_definitions['testuser.test.another_op']['required_ops'] = ['some_op_req']

        with patch('interpretune.analysis.ops.compiler.schema_compiler.rank_zero_warn') as mock_warn:
            compile_op_schema('testuser.test.another_op', op_definitions)

            # Should resolve to first alphabetical match and issue warning
            resolved_ops = op_definitions['testuser.test.another_op']['required_ops']
            assert resolved_ops == ['anotheruser.test.some_op_req']

            # Should have warned about multiple matches
            mock_warn.assert_called_once()
            warning_msg = mock_warn.call_args[0][0]
            assert 'multiple matching operations found' in warning_msg
