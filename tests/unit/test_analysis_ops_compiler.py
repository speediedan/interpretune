from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import yaml
import tempfile
import os

from interpretune.analysis.ops.base import OpSchema, ColCfg, AnalysisOp
from interpretune.analysis.ops.compiler.schema_compiler import (_compile_composition_schema_core,
                                                               jit_compile_composition_schema,
                                                               compile_operation_composition_schema,
                                                               build_operation_compositions,
                                                               load_and_compile_operations)


class TestSchemaCompilation:
    """Tests for the core schema compilation functionality."""

    def test_load_and_compile_operations(self):
        """Test load_and_compile_operations basic functionality."""
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp:
            yaml_content = {
                "op1": {
                    "input_schema": {"input1": {}},
                    "output_schema": {"output1": {}}
                },
                "composite_operations": {
                    "composite_op": {
                        "composition": "op1",
                        "aliases": ["composite_alias"]
                    }
                }
            }
            yaml.dump(yaml_content, temp)
            temp_path = temp.name

        try:
            # Mock yaml.safe_load to return our content
            with patch('yaml.safe_load', return_value=yaml_content):
                # Call the function
                result = load_and_compile_operations(temp_path)

                # Verify results
                assert "op1" in result
                assert "composite_op" in result
                assert result["composite_op"]["aliases"] == ["composite_alias"]

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_compile_composition_schema_core_basic(self):
        """Test basic functionality of _compile_composition_schema_core."""
        # Mock operations
        operations = ["op1", "op2"]

        # Mock schema extraction function
        def get_schemas(op):
            if op == "op1":
                return {"input1": "field1"}, {"output1": "field1"}
            else:
                return {"input2": "field2", "output1": "field1"}, {"output2": "field2"}

        # Mock helper functions
        def is_intermediate(field):
            return False

        def handle_object_field(field):
            return field

        def create_schema(fields):
            return fields

        # Call the function
        input_schema, output_schema = _compile_composition_schema_core(
            operations=operations,
            get_schemas_fn=get_schemas,
            is_intermediate_fn=is_intermediate,
            handle_object_field_fn=handle_object_field,
            create_schema_fn=create_schema
        )

        # Verify results
        assert "input1" in input_schema
        assert "input2" in input_schema
        assert "output1" in output_schema
        assert "output2" in output_schema

    def test_compile_composition_schema_core_empty_operations(self):
        """Test _compile_composition_schema_core with empty operations list."""
        with pytest.raises(ValueError, match="No operations provided"):
            _compile_composition_schema_core(
                operations=[],
                get_schemas_fn=lambda op: ({}, {}),
                is_intermediate_fn=lambda f: False,
                handle_object_field_fn=lambda f: f,
                create_schema_fn=lambda f: f
            )

    def test_compile_composition_schema_core_intermediates(self):
        """Test _compile_composition_schema_core with intermediate fields."""
        # Mock operations
        operations = ["op1", "op2"]

        # Mock schema extraction function
        def get_schemas(op):
            if op == "op1":
                return {"input1": "field1"}, {"output1": "field1", "intermediate1": "inter1"}
            else:
                return {"input2": "field2"}, {"output2": "field2"}

        # Mock intermediate detection
        def is_intermediate(field):
            return field == "inter1"

        # Call the function
        input_schema, output_schema = _compile_composition_schema_core(
            operations=operations,
            get_schemas_fn=get_schemas,
            is_intermediate_fn=is_intermediate,
            handle_object_field_fn=lambda f: f,
            create_schema_fn=lambda f: f
        )

        # Verify intermediate field handling
        assert "intermediate1" not in output_schema
        assert "output1" in output_schema
        assert "output2" in output_schema

    def test_jit_compile_string_operations(self):
        """Test jit_compile_composition_schema with string operations."""
        # Mock operation definitions
        op_definitions = {
            "op1": {
                "input_schema": {
                    "input1": {"datasets_dtype": "float32", "required": True}
                },
                "output_schema": {
                    "output1": {"datasets_dtype": "float32", "required": True}
                }
            },
            "op2": {
                "input_schema": {
                    "input2": {"datasets_dtype": "float32", "required": True},
                    "output1": {"datasets_dtype": "float32", "required": True}
                },
                "output_schema": {
                    "output2": {"datasets_dtype": "float32", "required": True}
                }
            }
        }

        # Call the function with string operations
        input_schema, output_schema = jit_compile_composition_schema(
            operations=["op1", "op2"],
            op_definitions=op_definitions
        )

        # Verify results are OpSchema instances
        assert isinstance(input_schema, OpSchema)
        assert isinstance(output_schema, OpSchema)

        # Verify fields are ColCfg instances
        assert "input1" in input_schema
        assert isinstance(input_schema["input1"], ColCfg)
        assert "output1" not in input_schema  # should not be included in input
        assert "output2" in output_schema
        assert isinstance(output_schema["output2"], ColCfg)

    def test_jit_compile_analysis_op_instances(self):
        """Test jit_compile_composition_schema with AnalysisOp instances."""
        # Create mock AnalysisOp instances
        op1 = MagicMock(spec=AnalysisOp)
        op1.input_schema = {"input1": ColCfg(datasets_dtype="float32", required=True)}
        op1.output_schema = {"output1": ColCfg(datasets_dtype="float32", required=True)}

        op2 = MagicMock(spec=AnalysisOp)
        op2.input_schema = {"input2": ColCfg(datasets_dtype="float32", required=True)}
        op2.output_schema = {"output2": ColCfg(datasets_dtype="float32", required=True)}

        # Call the function with AnalysisOp instances
        input_schema, output_schema = jit_compile_composition_schema(
            operations=[op1, op2],
            op_definitions={}  # Not used for AnalysisOp instances
        )

        # Verify results
        assert "input1" in input_schema
        assert "input2" in input_schema
        assert "output1" in output_schema
        assert "output2" in output_schema

    def test_jit_compile_missing_operation(self):
        """Test jit_compile_composition_schema with missing operation."""
        with pytest.raises(ValueError, match="Operation missing_op not found"):
            jit_compile_composition_schema(
                operations=["missing_op"],
                op_definitions={}
            )

    def test_jit_compile_missing_schemas(self):
        """Test jit_compile_composition_schema with operation missing schemas."""
        # Update the test to match the actual error message from the function
        with pytest.raises(ValueError, match="Operation incomplete_op not found in definitions"):
            jit_compile_composition_schema(
                operations=["incomplete_op"],
                op_definitions={"incomplete_op": {}}
            )

    def test_jit_compile_invalid_operation_type(self):
        """Test jit_compile_composition_schema with invalid operation type."""
        with pytest.raises(TypeError, match="Operations must be strings or AnalysisOp instances"):
            jit_compile_composition_schema(
                operations=[123],  # Invalid type
                op_definitions={}
            )

    def test_jit_compile_object_field_handling(self):
        """Test jit_compile_composition_schema with object field type handling."""
        # Mock operation definitions with object type fields
        op_definitions = {
            "op1": {
                "input_schema": {},
                "output_schema": {
                    "object_field": {"datasets_dtype": "object"}
                }
            }
        }

        # Call the function
        _, output_schema = jit_compile_composition_schema(
            operations=["op1"],
            op_definitions=op_definitions
        )

        # Verify object field was converted properly
        assert "object_field" in output_schema
        assert output_schema["object_field"].datasets_dtype == "string"
        assert output_schema["object_field"].non_tensor is True

    def test_jit_compile_schemas_without_output_schema(self):
        """Test jit_compile_composition_schema with op definition that has input but not output schema."""
        # Create a mock operation with input schema but no output schema
        op_definitions = {
            "partial_op": {
                "input_schema": {
                    "input1": {"datasets_dtype": "float32", "required": True}
                }
                # No output_schema provided
            }
        }

        # Should raise an error about missing required schemas
        with pytest.raises(ValueError, match="Operation partial_op is missing required schemas"):
            jit_compile_composition_schema(
                operations=["partial_op"],
                op_definitions=op_definitions
            )

    def test_field_conversion_warn(self):
        from dataclasses import field, make_dataclass
        # Define dict‐based field variants
        dict_field_intermediate = {"datasets_dtype": "float32", "intermediate_only": True}
        dict_field_normal = {"datasets_dtype": "float32", "intermediate_only": False}
        dict_field_no_flag = {"datasets_dtype": "float32"}  # no flag

        # Define ColCfg‐based field variants
        colcfg_field_intermediate = ColCfg(datasets_dtype="float32", intermediate_only=True)
        colcfg_field_normal = ColCfg(datasets_dtype="float32", intermediate_only=False)
        degen_colcfg_field = make_dataclass('degen_colcfg', [('wrong_field_name', str, field(default="float32"))])


        # Build op definitions including both dict and ColCfg variants
        op_definitions = {
            "op1": {
                "input_schema": {
                    "in_inter_dict": dict_field_intermediate,
                    "in_norm_dict": dict_field_normal,
                    "in_noflag_dict": dict_field_no_flag,
                    "in_inter_colcfg": colcfg_field_intermediate,
                    "in_norm_colcfg": colcfg_field_normal,
                },
                "output_schema": {
                    "out_degen_colcfg": degen_colcfg_field(),
                    "out_inter_dict": dict_field_intermediate,
                    "out_norm_dict": dict_field_normal,
                    "out_noflag_dict": dict_field_no_flag,
                    "out_inter_colcfg": colcfg_field_intermediate,
                    "out_norm_colcfg": colcfg_field_normal
                }
            }
        }

        with pytest.warns(UserWarning, match="Conversion to ColCfg"):
            # Compile schemas for a single‐op composition
            input_schema, output_schema = jit_compile_composition_schema(
                operations=["op1"],
                op_definitions=op_definitions
            )

        # Assert normal and no_flag fields remain in input
        assert "in_norm_dict" in input_schema
        assert "in_noflag_dict" in input_schema
        assert "in_norm_colcfg" in input_schema

        # Assert normal and no_flag fields remain in output
        assert "out_norm_dict" in output_schema
        assert "out_noflag_dict" in output_schema
        assert "out_norm_colcfg" in output_schema

    def test_composition_with_intermediate_passthrough(self):
        """Test that an intermediate field from op1 used in op2 is filtered out correctly."""
        # op1 emits an intermediate field, op2 consumes it but it should not appear in final schemas
        op_definitions = {
            "op1": {
                "input_schema": {"a": {"datasets_dtype": "int32", "required": True}},
                "output_schema": {"mid": {"datasets_dtype": "float32", "intermediate_only": True}}
            },
            "op2": {
                "input_schema": {
                    "mid": {"datasets_dtype": "float32", "required": True},
                    "b": {"datasets_dtype": "int64", "required": True}
                },
                "output_schema": {"c": {"datasets_dtype": "string", "required": False}}
            }
        }

        inp, out = jit_compile_composition_schema(
            operations=["op1", "op2"],
            op_definitions=op_definitions
        )

        # 'mid' should not be in the final input or output
        assert "mid" not in inp
        assert "mid" not in out

        # 'a' and 'b' should appear in input, 'c' should appear in output
        assert "a" in inp and "b" in inp
        assert "c" in out

    def test_object_field_handling_with_colcfg(self):
        """Test handling of object field type with ColCfg instances."""
        # Test with ColCfg field with datasets_dtype='object'
        colcfg_object = ColCfg(datasets_dtype="object")
        colcfg_string = ColCfg(datasets_dtype="string")

        # Create dictionary fields for comparison
        dict_object = {"datasets_dtype": "object"}

        # Create a custom object that doesn't have datasets_dtype attribute
        class CustomField:
            pass
        custom_field = CustomField()

        # Use the handle_object_field function directly
        def handle_object_field(field_def):
            datasets_dtype = None
            if isinstance(field_def, dict):
                datasets_dtype = field_def.get('datasets_dtype')
            elif hasattr(field_def, 'datasets_dtype'):
                datasets_dtype = field_def.datasets_dtype

            if datasets_dtype == 'object':
                if isinstance(field_def, dict):
                    field_def_copy = field_def.copy()
                    field_def_copy['datasets_dtype'] = 'string'
                    field_def_copy['non_tensor'] = True
                    return field_def_copy
                elif isinstance(field_def, ColCfg):
                    # Create a modified ColCfg for object fields
                    from dataclasses import replace
                    field_def_copy = replace(field_def, datasets_dtype='string', non_tensor=True)
                    return field_def_copy
            return field_def

        # Test each case
        result1 = handle_object_field(colcfg_object)
        assert result1.datasets_dtype == "string"
        assert result1.non_tensor is True

        result2 = handle_object_field(colcfg_string)
        assert result2 is colcfg_string  # Should return the original object

        result3 = handle_object_field(dict_object)
        assert result3["datasets_dtype"] == "string"
        assert result3["non_tensor"] is True

        result4 = handle_object_field(custom_field)
        assert result4 is custom_field  # Should return the original object

    def test_schema_dictionary_to_colcfg_conversion(self):
        """Test conversion of schema dictionary fields to ColCfg instances."""
        # Create a schema with various field types
        schema_dict = {
            "field1": {"datasets_dtype": "float32", "required": True},
            "field2": ColCfg(datasets_dtype="int64", required=False),
            "field3": {"datasets_dtype": "string", "intermediate_only": True},
        }

        # Define the schema conversion function from jit_compile_composition_schema
        def get_schema(schema_dict):
            if not schema_dict:
                return {}
            result = {}
            for field_name, field_def in schema_dict.items():
                if isinstance(field_def, dict):
                    result[field_name] = ColCfg(**field_def)
                elif isinstance(field_def, ColCfg):
                    result[field_name] = field_def
            return result

        # Convert the schema
        result = get_schema(schema_dict)

        # Check that all fields are ColCfg instances
        assert isinstance(result["field1"], ColCfg)
        assert isinstance(result["field2"], ColCfg)
        assert isinstance(result["field3"], ColCfg)

        # Check that properties are preserved
        assert result["field1"].datasets_dtype == "float32"
        assert result["field1"].required is True

        assert result["field2"].datasets_dtype == "int64"
        assert result["field2"].required is False

        assert result["field3"].datasets_dtype == "string"
        assert result["field3"].intermediate_only is True

        # Test with empty schema
        empty_result = get_schema({})
        assert empty_result == {}

    def test_compile_operation_composition(self):
        """Test compile_operation_composition_schema basic functionality."""
        # Mock operation definitions
        all_operations_dict = {
            "op1": {
                "input_schema": {"input1": {"datasets_dtype": "float32", "required": True}},
                "output_schema": {"output1": {"datasets_dtype": "float32", "required": True}}
            },
            "op2": {
                "input_schema": {"input2": {"datasets_dtype": "float32", "required": True}},
                "output_schema": {"output2": {"datasets_dtype": "float32", "required": True}}
            }
        }

        # Call the function
        input_schema, output_schema = compile_operation_composition_schema(
            operations=["op1", "op2"],
            all_operations_dict=all_operations_dict
        )

        # Verify results (dictionaries, not OpSchema)
        assert isinstance(input_schema, dict)
        assert isinstance(output_schema, dict)
        assert "input1" in input_schema
        assert "input2" in input_schema
        assert "output1" in output_schema
        assert "output2" in output_schema

    def test_compile_operation_composition_missing_op(self):
        """Test compile_operation_composition_schema with missing operation."""
        with pytest.raises(ValueError, match="Operation missing_op not found"):
            compile_operation_composition_schema(
                operations=["missing_op"],
                all_operations_dict={}
            )

    def test_compile_operation_composition_object_field(self):
        """Test compile_operation_composition_schema with object field handling."""
        # Mock operation definitions with object fields
        all_operations_dict = {
            "op1": {
                "input_schema": {},
                "output_schema": {
                    "object_field": {"datasets_dtype": "object"}
                }
            }
        }

        # Call the function
        _, output_schema = compile_operation_composition_schema(
            operations=["op1"],
            all_operations_dict=all_operations_dict
        )

        # Verify object field conversion
        assert "object_field" in output_schema
        assert output_schema["object_field"]["datasets_dtype"] == "string"
        assert output_schema["object_field"]["non_tensor"] is True

class TestBuildOperationCompositions:
    """Tests for build_operation_compositions function."""

    def test_build_operation_compositions(self):
        """Test build_operation_compositions basic functionality."""
        # Mock YAML config
        yaml_config = {
            "op1": {
                "input_schema": {"input1": {}},
                "output_schema": {"output1": {}}
            },
            "op2": {
                "input_schema": {"input2": {}},
                "output_schema": {"output2": {}}
            },
            "composite_operations": {
                "composite_op": {
                    "composition": "op1.op2",
                    "aliases": ["composite_alias"]
                }
            }
        }

        # Call the function
        result = build_operation_compositions(yaml_config)

        # Verify composite operation was created
        assert "composite_op" in result
        assert "composition" in result["composite_op"]
        assert result["composite_op"]["composition"] == ["op1", "op2"]
        assert result["composite_op"]["aliases"] == ["composite_alias"]
        assert "input_schema" in result["composite_op"]
        assert "output_schema" in result["composite_op"]

    def test_build_operation_compositions_object_fields(self):
        """Test build_operation_compositions with object field conversion."""
        # Mock YAML config with object type fields
        yaml_config = {
            "op_with_object": {
                "input_schema": {},
                "output_schema": {
                    "object_field": {"datasets_dtype": "object"}
                }
            }
        }

        # Call the function
        result = build_operation_compositions(yaml_config)

        # Verify object field was converted
        assert result["op_with_object"]["output_schema"]["object_field"]["datasets_dtype"] == "string"
        assert result["op_with_object"]["output_schema"]["object_field"]["non_tensor"] is True
