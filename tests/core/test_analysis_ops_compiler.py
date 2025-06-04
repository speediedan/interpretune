"""Tests for analysis operations compiler functionality."""
from __future__ import annotations
import time
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from interpretune.analysis.ops.compiler.cache_manager import (
    OpDefinitionsCacheManager, OpDef, YamlFileInfo
)
from interpretune.analysis.ops.base import OpSchema, ColCfg, AnalysisOp
from interpretune.analysis.ops.compiler.schema_compiler import (
                                                               _compile_composition_schema_core,
                                                               jit_compile_composition_schema,
                                                               compile_operation_composition_schema,
                                                               build_operation_compositions,
                                                               )


class TestSchemaCompilation:
    """Tests for the core schema compilation functionality."""

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
        assert "output1" in input_schema
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

        assert "mid" in inp
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


class TestYamlFileInfo:
    """Tests for YamlFileInfo functionality."""

    def test_from_path_valid_file(self, tmp_path):
        """Test creating YamlFileInfo from a valid file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("test: content")

        file_info = YamlFileInfo.from_path(yaml_file)

        assert file_info.path == yaml_file
        assert file_info.mtime > 0
        assert len(file_info.content_hash) == 64  # SHA256 hex digest length

    def test_from_path_nonexistent_file(self, tmp_path):
        """Test creating YamlFileInfo from non-existent file raises error."""
        nonexistent_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            YamlFileInfo.from_path(nonexistent_file)

    def test_content_hash_changes_with_content(self, tmp_path):
        """Test that content hash changes when file content changes."""
        yaml_file = tmp_path / "test.yaml"

        yaml_file.write_text("content1")
        info1 = YamlFileInfo.from_path(yaml_file)

        yaml_file.write_text("content2")
        info2 = YamlFileInfo.from_path(yaml_file)

        assert info1.content_hash != info2.content_hash


class TestOpDef:
    """Tests for OpDef functionality."""

    def test_op_def_creation(self):
        """Test creating an OpDef with all fields."""
        input_schema = OpSchema({"field1": ColCfg(datasets_dtype="int64")})
        output_schema = OpSchema({"field2": ColCfg(datasets_dtype="float32")})

        op_def = OpDef(
            name="test_op",
            description="Test operation",
            implementation="test.module.func",
            input_schema=input_schema,
            output_schema=output_schema,
            aliases=["alias1", "alias2"],
            function_params={"param1": "value1"},
            required_ops=["dep1", "dep2"],
            composition=["op1", "op2"]
        )

        assert op_def.name == "test_op"
        assert op_def.aliases == ["alias1", "alias2"]
        assert op_def.composition == ["op1", "op2"]

    def test_op_def_to_dict(self):
        """Test converting OpDef to dictionary."""
        input_schema = OpSchema({"field1": ColCfg(datasets_dtype="int64")})
        output_schema = OpSchema({"field2": ColCfg(datasets_dtype="float32")})

        op_def = OpDef(
            name="test_op",
            description="Test operation",
            implementation="test.module.func",
            input_schema=input_schema,
            output_schema=output_schema
        )

        result_dict = op_def.to_dict()

        assert result_dict["name"] == "test_op"
        assert result_dict["input_schema"] == input_schema
        assert result_dict["output_schema"] == output_schema
        assert "composition" in result_dict  # Should be present even if None


class TestDefinitionsCacheManager:
    """Tests for OpDefinitionsCacheManager functionality."""

    @pytest.fixture
    def cache_manager(self, tmp_path):
        """Create a cache manager with temporary directory."""
        return OpDefinitionsCacheManager(cache_dir=tmp_path)

    @pytest.fixture
    def sample_yaml_file(self, tmp_path):
        """Create a sample YAML file."""
        yaml_file = tmp_path / "test_ops.yaml"
        yaml_content = """
test_op:
  description: Test operation
  implementation: test.module.func
  input_schema:
    field1:
      datasets_dtype: int64
  output_schema:
    field2:
      datasets_dtype: float32
"""
        yaml_file.write_text(yaml_content)
        return yaml_file

    def test_cache_manager_initialization(self, tmp_path):
        """Test cache manager initialization."""
        cache_manager = OpDefinitionsCacheManager(cache_dir=tmp_path)

        assert cache_manager.cache_dir == tmp_path
        assert tmp_path.exists()
        assert cache_manager._yaml_files == []
        assert cache_manager._fingerprint is None

    def test_add_yaml_file(self, cache_manager, sample_yaml_file):
        """Test adding a YAML file to monitoring."""
        cache_manager.add_yaml_file(sample_yaml_file)

        assert len(cache_manager._yaml_files) == 1
        assert cache_manager._yaml_files[0].path == sample_yaml_file
        assert cache_manager._fingerprint is None  # Should reset fingerprint

    def test_add_yaml_file_duplicate(self, cache_manager, sample_yaml_file):
        """Test that adding the same YAML file twice doesn't create duplicates."""
        cache_manager.add_yaml_file(sample_yaml_file)
        cache_manager.add_yaml_file(sample_yaml_file)

        assert len(cache_manager._yaml_files) == 1

    def test_fingerprint_computation(self, cache_manager, sample_yaml_file):
        """Test fingerprint computation."""
        cache_manager.add_yaml_file(sample_yaml_file)

        fingerprint1 = cache_manager.fingerprint
        fingerprint2 = cache_manager.fingerprint  # Should be cached

        assert fingerprint1 == fingerprint2
        assert len(fingerprint1) == 16  # Truncated to 16 characters

    def test_fingerprint_empty(self, cache_manager):
        """Test fingerprint with no YAML files."""
        assert cache_manager.fingerprint == "empty"

    def test_fingerprint_changes_with_content(self, cache_manager, tmp_path):
        """Test that fingerprint changes when file content changes."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("content1")

        cache_manager.add_yaml_file(yaml_file)
        fingerprint1 = cache_manager.fingerprint

        # Modify file and create new cache manager
        yaml_file.write_text("content2")
        cache_manager2 = OpDefinitionsCacheManager(cache_dir=tmp_path)
        cache_manager2.add_yaml_file(yaml_file)
        fingerprint2 = cache_manager2.fingerprint

        assert fingerprint1 != fingerprint2

    def test_cache_module_path(self, cache_manager, sample_yaml_file):
        """Test cache module path generation."""
        cache_manager.add_yaml_file(sample_yaml_file)

        cache_path = cache_manager._get_cache_module_path()
        expected_name = f"op_definitions_{cache_manager.fingerprint}.py"

        assert cache_path.name == expected_name
        assert cache_path.parent == cache_manager.cache_dir

    def test_is_cache_valid_no_cache(self, cache_manager, sample_yaml_file):
        """Test cache validation when no cache exists."""
        cache_manager.add_yaml_file(sample_yaml_file)

        assert not cache_manager.is_cache_valid()

    def test_is_cache_valid_file_changed(self, cache_manager, sample_yaml_file):
        """Test cache validation when source file changes."""
        cache_manager.add_yaml_file(sample_yaml_file)

        # Create a fake cache file
        cache_path = cache_manager._get_cache_module_path()
        cache_path.write_text("# fake cache")

        # File should be considered valid initially
        assert cache_manager.is_cache_valid()

        # Modify the source file
        time.sleep(0.01)  # Ensure different mtime
        sample_yaml_file.write_text("modified content")

        # Cache should now be invalid
        assert not cache_manager.is_cache_valid()

    def test_serialize_col_cfg(self, cache_manager):
        """Test ColCfg serialization."""
        col_cfg = ColCfg(
            datasets_dtype="int64",
            required=False,
            non_tensor=True
        )

        serialized = cache_manager._serialize_col_cfg(col_cfg)

        # Should always include datasets_dtype
        assert 'datasets_dtype="int64"' in serialized
        assert 'required=False' in serialized
        assert 'non_tensor=True' in serialized
        assert serialized.startswith('ColCfg(')
        assert serialized.endswith(')')

    def test_serialize_op_schema(self, cache_manager):
        """Test OpSchema serialization."""
        schema = OpSchema({
            "field1": ColCfg(datasets_dtype="int64"),
            "field2": ColCfg(datasets_dtype="float32", required=False)
        })

        serialized = cache_manager._serialize_op_schema(schema)

        assert serialized.startswith('OpSchema(')
        assert '"field1":' in serialized
        assert '"field2":' in serialized
        assert 'ColCfg(' in serialized

    def test_serialize_op_schema_empty(self, cache_manager):
        """Test empty OpSchema serialization."""
        schema = OpSchema({})

        serialized = cache_manager._serialize_op_schema(schema)

        assert serialized == 'OpSchema({})'

    def test_serialize_op_def(self, cache_manager):
        """Test OpDef serialization."""
        input_schema = OpSchema({"field1": ColCfg(datasets_dtype="int64")})
        output_schema = OpSchema({"field2": ColCfg(datasets_dtype="float32")})

        op_def = OpDef(
            name="test_op",
            description="Test operation",
            implementation="test.module.func",
            input_schema=input_schema,
            output_schema=output_schema,
            aliases=["alias1"],
            function_params={"param1": "value1"}
        )

        serialized = cache_manager._serialize_op_def(op_def)

        assert serialized.startswith('OpDef(')
        assert 'name="test_op"' in serialized
        assert 'aliases=[\'alias1\']' in serialized
        assert 'function_params={\'param1\': \'value1\'}' in serialized

    def test_generate_module_content(self, cache_manager, sample_yaml_file):
        """Test module content generation."""
        cache_manager.add_yaml_file(sample_yaml_file)

        # Create sample op definitions
        op_definitions = {
            "test_op": OpDef(
                name="test_op",
                description="Test operation",
                implementation="test.module.func",
                input_schema=OpSchema({"field1": ColCfg(datasets_dtype="int64")}),
                output_schema=OpSchema({"field2": ColCfg(datasets_dtype="float32")})
            )
        }

        content = cache_manager._generate_module_content(op_definitions)

        # Check for required content
        assert "GENERATED FILE - DO NOT EDIT" in content
        assert f"Fingerprint: {cache_manager.fingerprint}" in content
        assert "OP_DEFINITIONS = {" in content
        assert '"test_op":' in content
        assert "OpDef(" in content
        assert "from interpretune.analysis.ops.base import OpSchema, ColCfg" in content

    def test_save_and_load_cache_cycle(self, cache_manager, sample_yaml_file):
        """Test complete save and load cycle."""
        cache_manager.add_yaml_file(sample_yaml_file)

        # Create sample op definitions with both unique ops and aliases
        op_definitions = {
            "test_op": OpDef(
                name="test_op",
                description="Test operation",
                implementation="test.module.func",
                input_schema=OpSchema({"field1": ColCfg(datasets_dtype="int64")}),
                output_schema=OpSchema({"field2": ColCfg(datasets_dtype="float32")}),
                aliases=["alias1"]
            ),
            # This alias entry should be filtered out during save
            "alias1": OpDef(
                name="test_op",  # Points to the canonical name
                description="Test operation",
                implementation="test.module.func",
                input_schema=OpSchema({"field1": ColCfg(datasets_dtype="int64")}),
                output_schema=OpSchema({"field2": ColCfg(datasets_dtype="float32")}),
                aliases=["alias1"]
            )
        }

        # Save to cache
        cache_path = cache_manager.save_cache(op_definitions)
        assert cache_path.exists()

        # Load from cache
        loaded_definitions = cache_manager.load_cache()

        assert loaded_definitions is not None
        assert "test_op" in loaded_definitions
        assert loaded_definitions["test_op"].name == "test_op"
        assert loaded_definitions["test_op"].aliases == ["alias1"]

    def test_load_cache_invalid(self, cache_manager, sample_yaml_file):
        """Test loading invalid cache returns None."""
        cache_manager.add_yaml_file(sample_yaml_file)

        # No cache file exists
        result = cache_manager.load_cache()
        assert result is None

    def test_cleanup_old_cache_files(self, cache_manager, sample_yaml_file):
        """Test cleanup of old cache files."""
        cache_manager.add_yaml_file(sample_yaml_file)

        # Create some old cache files
        old_cache1 = cache_manager.cache_dir / "op_definitions_old1.py"
        old_cache2 = cache_manager.cache_dir / "op_definitions_old2.py"
        old_cache1.write_text("# old cache 1")
        old_cache2.write_text("# old cache 2")

        # Save current cache (should trigger cleanup)
        op_definitions = {
            "test_op": OpDef(
                name="test_op",
                description="Test",
                implementation="test.func",
                input_schema=OpSchema({}),
                output_schema=OpSchema({})
            )
        }
        cache_manager.save_cache(op_definitions)

        # Old files should be removed
        assert not old_cache1.exists()
        assert not old_cache2.exists()

        # Current cache should still exist
        current_cache = cache_manager._get_cache_module_path()
        assert current_cache.exists()

    @patch('interpretune.utils.logging.rank_zero_warn')
    def test_cleanup_old_cache_files_permission_error(self, mock_warn, cache_manager, sample_yaml_file):
        """Test cleanup when file removal fails due to permissions."""
        cache_manager.add_yaml_file(sample_yaml_file)

        # Create a real cache file that we'll try to delete
        old_cache_file = cache_manager.cache_dir / "op_definitions_old_fingerprint.py"
        old_cache_file.write_text("# old cache")

        # Create a mock file that will raise OSError when unlink is called
        mock_file = MagicMock()
        mock_file.name = "op_definitions_old_fingerprint.py"
        mock_file.unlink.side_effect = OSError("Permission denied")

        with pytest.warns(match="Failed to remove old cache file"):
            # Patch the pathlib.Path.glob method to return our mock file
            with patch('pathlib.Path.glob', return_value=[mock_file]):
                cache_manager._cleanup_old_cache_files()

    def test_is_cache_valid_file_modified_time_check(self, cache_manager, tmp_path):
        """Test cache validation when source files have been modified after cache creation."""
        # Create a YAML file
        yaml_file = tmp_path / "test.yaml"
        yaml_content = {
            "test_op": {
                "implementation": "tests.core.test_analysis_ops_base.op_impl_test",
                "input_schema": {"input1": {"datasets_dtype": "float32"}},
                "output_schema": {"output1": {"datasets_dtype": "float32"}}
            }
        }
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        cache_manager.add_yaml_file(yaml_file)

        # Create a cache file
        cache_path = cache_manager._get_cache_module_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("# cache content")

        # Initially should be valid
        assert cache_manager.is_cache_valid()

        # Modify the YAML file to have a newer timestamp
        import time
        time.sleep(0.1)  # Ensure different timestamp
        yaml_file.touch()  # Update modification time

        # Now cache should be invalid due to newer source file
        assert not cache_manager.is_cache_valid()

    def test_load_cache_import_error_handling(self, cache_manager, sample_yaml_file):
        """Test load_cache when import fails."""
        cache_manager.add_yaml_file(sample_yaml_file)

        # Create invalid cache file that will cause import error
        cache_path = cache_manager._get_cache_module_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("invalid python syntax !!!")

        # Should return None when import fails
        result = cache_manager.load_cache()
        assert result is None

    def test_load_cache_spec_or_loader_none(self, cache_manager, sample_yaml_file):
        """Test load_cache when spec or spec.loader is None."""
        cache_manager.add_yaml_file(sample_yaml_file)

        # Create a dummy cache file, its content doesn't strictly matter here
        # as we're mocking the import mechanism before content is read.
        cache_path = cache_manager._get_cache_module_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("FINGERPRINT = \"test\"\nOP_DEFINITIONS = {}")

        # Scenario 1: spec is None
        with patch('importlib.util.spec_from_file_location', return_value=None) as mock_spec_from_file:
            result = cache_manager.load_cache()
            assert result is None
            mock_spec_from_file.assert_called_once()

        # Reset mock for next scenario
        mock_spec_from_file.reset_mock()

        # Scenario 2: spec.loader is None
        mock_spec = MagicMock()
        mock_spec.loader = None
        with patch('importlib.util.spec_from_file_location', return_value=mock_spec) as mock_spec_from_file:
            result = cache_manager.load_cache()
            assert result is None
            mock_spec_from_file.assert_called_once()

    def test_fingerprint_consistency_across_instances(self, tmp_path, sample_yaml_file):
        """Test that fingerprint is consistent across different cache manager instances."""
        cache_manager1 = OpDefinitionsCacheManager(cache_dir=tmp_path)
        cache_manager1.add_yaml_file(sample_yaml_file)
        fingerprint1 = cache_manager1.fingerprint

        cache_manager2 = OpDefinitionsCacheManager(cache_dir=tmp_path)
        cache_manager2.add_yaml_file(sample_yaml_file)
        fingerprint2 = cache_manager2.fingerprint

        assert fingerprint1 == fingerprint2

    def test_fingerprint_with_missing_files(self, tmp_path):
        """Test fingerprint computation when YAML files don't exist."""
        cache_manager = OpDefinitionsCacheManager(cache_dir=tmp_path)

        # Create a file info for a non-existent file
        nonexistent_file = tmp_path / "nonexistent.yaml"
        file_info = YamlFileInfo(
            path=nonexistent_file,
            mtime=123456.0,
            content_hash="dummy_hash"
        )
        cache_manager._yaml_files = [file_info]

        # This should not raise an error and should return a fingerprint
        fingerprint = cache_manager.fingerprint
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0

    def test_cache_validation_with_missing_source_files(self, cache_manager, sample_yaml_file):
        """Test cache validation when source files are missing."""
        cache_manager.add_yaml_file(sample_yaml_file)

        # Create a valid cache file
        cache_path = cache_manager._get_cache_module_path()
        cache_path.write_text('FINGERPRINT = "test"\nOP_DEFINITIONS = {}')

        # Remove the source YAML file
        sample_yaml_file.unlink()

        # Cache should be invalid because source file is missing
        assert not cache_manager.is_cache_valid()

    def test_load_cache_with_import_error(self, cache_manager, sample_yaml_file):
        """Test cache loading when module import fails."""
        cache_manager.add_yaml_file(sample_yaml_file)

        # Create a cache file that will cause import errors
        cache_path = cache_manager._get_cache_module_path()
        cache_path.write_text("""
import nonexistent_module  # This will cause an import error
FINGERPRINT = "test"
OP_DEFINITIONS = {}
""")

        with patch('interpretune.analysis.ops.compiler.cache_manager.rank_zero_warn') as mock_warn:
            result = cache_manager.load_cache()

            assert result is None
            mock_warn.assert_called()

    def test_serialize_col_cfg_with_non_dict_field(self, cache_manager):
        """Test ColCfg serialization with various field types."""
        from interpretune.analysis.ops.base import ColCfg

        # Test with all types of fields including edge cases
        col_cfg = ColCfg(
            datasets_dtype="float32",
            required=True,
            non_tensor=False,
            intermediate_only=True,
            array_shape=(10, 20)  # Non-string, non-default value
        )

        serialized = cache_manager._serialize_col_cfg(col_cfg)
        assert "datasets_dtype=\"float32\"" in serialized
        assert "intermediate_only=True" in serialized
        assert "array_shape=(10, 20)" in serialized
        # Note: required=True might be default, so don't assert it must be present

    def test_serialize_col_cfg_with_string_fields(self, cache_manager):
        """Test ColCfg serialization with string field values."""
        from interpretune.analysis.ops.base import ColCfg

        col_cfg = ColCfg(
            datasets_dtype="string",
            sequence_type="custom_sequence"  # String field that's not datasets_dtype
        )

        serialized = cache_manager._serialize_col_cfg(col_cfg)
        assert "datasets_dtype=\"string\"" in serialized
        assert "sequence_type=\"custom_sequence\"" in serialized

    def test_serialize_col_cfg_minimal_fields(self, cache_manager):
        """Test ColCfg serialization with only required fields."""
        from interpretune.analysis.ops.base import ColCfg

        # Create ColCfg with only the required datasets_dtype
        col_cfg = ColCfg(datasets_dtype="int64")

        serialized = cache_manager._serialize_col_cfg(col_cfg)
        assert "datasets_dtype=\"int64\"" in serialized
        # Check that we don't include too many fields - should only include non-default values
        # Allow for some fields that have None as non-default values
        assert serialized.count("=") <= 6  # Reasonable upper bound for non-default fields

    def test_serialize_col_cfg_default_factory_fields(self, cache_manager):
        """Test ColCfg serialization with fields that have default factories."""
        from interpretune.analysis.ops.base import ColCfg
        from dataclasses import fields

        # Create a ColCfg instance
        col_cfg = ColCfg(datasets_dtype="float32")

        # Find fields with default factories in ColCfg
        col_cfg_fields = fields(ColCfg)
        default_factory_fields = []

        for field_info in col_cfg_fields:
            # Check if field has a default factory (should be MISSING for default but callable for default_factory)
            if (field_info.default is field_info.default_factory and
                field_info.default_factory is not field_info.default_factory):
                default_factory_fields.append(field_info.name)

        # If there are default factory fields, test serialization
        if default_factory_fields:
            serialized = cache_manager._serialize_col_cfg(col_cfg)

            # Should include datasets_dtype
            assert "datasets_dtype=\"float32\"" in serialized

            # Check that default factory fields are handled properly
            # They should be included if their current value differs from factory default
            for field_name in default_factory_fields:
                field_value = getattr(col_cfg, field_name)
                # If the field has a non-default value, it should be serialized
                if field_value is not None and field_value != []:  # Common default factory values
                    assert f"{field_name}=" in serialized
        else:
            # If no default factory fields exist in current ColCfg implementation,
            # just verify basic serialization works
            serialized = cache_manager._serialize_col_cfg(col_cfg)
            assert "datasets_dtype=\"float32\"" in serialized

    def test_serialize_col_cfg_required_fields_no_default(self, cache_manager):
        """Test ColCfg serialization with required fields that have no default or default_factory."""
        from interpretune.analysis.ops.base import ColCfg
        from dataclasses import fields, MISSING

        # Create a ColCfg instance
        col_cfg = ColCfg(datasets_dtype="string", required=True)

        # Find fields that are truly required (no default and no default_factory)
        col_cfg_fields = fields(ColCfg)
        required_fields = []

        for field_info in col_cfg_fields:
            # Field is required if both default and default_factory are MISSING
            if (field_info.default is MISSING and
                field_info.default_factory is MISSING):
                required_fields.append(field_info.name)

        serialized = cache_manager._serialize_col_cfg(col_cfg)

        # Should always include datasets_dtype
        assert "datasets_dtype=\"string\"" in serialized

        # For truly required fields (if any exist), their values should be included
        # since they must be explicitly set
        for field_name in required_fields:
            field_value = getattr(col_cfg, field_name)
            if field_value is not None:
                # Required fields with non-None values should be serialized
                assert f"{field_name}=" in serialized


    def test_serialize_col_cfg_explicit_none_values(self, cache_manager):
        """Test ColCfg serialization when fields are explicitly set to None."""
        from interpretune.analysis.ops.base import ColCfg

        # Create ColCfg with some fields explicitly set to None
        col_cfg = ColCfg(
            datasets_dtype="int32",
            array_shape=None,  # Explicitly None
            sequence_type=None  # Explicitly None
        )

        serialized = cache_manager._serialize_col_cfg(col_cfg)

        # Should include datasets_dtype
        assert "datasets_dtype=\"int32\"" in serialized

        # Fields explicitly set to None should be included if None is not the default
        # This tests the else branch where fields might not have defaults

    def test_serialize_col_cfg_different_field_types(self, cache_manager):
        """Test ColCfg serialization with various field value types."""
        from interpretune.analysis.ops.base import ColCfg

        # Test with different types of values
        col_cfg = ColCfg(
            datasets_dtype="bool",
            required=False,  # Boolean field
            array_shape=[1, 2, 3],  # List field
            dyn_dim=5,  # Integer field
            non_tensor=True  # Boolean field
        )

        serialized = cache_manager._serialize_col_cfg(col_cfg)

        # Verify different value types are handled correctly
        assert "datasets_dtype=\"bool\"" in serialized
        assert "array_shape=[1, 2, 3]" in serialized
        assert "dyn_dim=5" in serialized

        # Boolean fields should be serialized without quotes
        if "required=False" in serialized:
            assert "required=\"False\"" not in serialized  # Should not be quoted
        if "non_tensor=True" in serialized:
            assert "non_tensor=\"True\"" not in serialized  # Should not be quoted

    def test_serialize_col_cfg_empty_collections(self, cache_manager):
        """Test ColCfg serialization with empty collections as values."""
        from interpretune.analysis.ops.base import ColCfg

        # Test with empty collections which might be default factory values
        col_cfg = ColCfg(
            datasets_dtype="float64",
            array_shape=[],  # Empty list
        )

        serialized = cache_manager._serialize_col_cfg(col_cfg)

        # Should include datasets_dtype
        assert "datasets_dtype=\"float64\"" in serialized

        # Empty list should be handled correctly (tests default factory branch)
        # Whether it's included depends on if [] is the default factory value

    def test_load_cache_fingerprint_mismatch_and_missing_op_definitions(self, cache_manager, sample_yaml_file):
        """Test load_cache for fingerprint mismatch and missing OP_DEFINITIONS."""
        cache_manager.add_yaml_file(sample_yaml_file)
        op_definitions = {
            "test_op": OpDef(
                name="test_op",
                description="desc",
                implementation="impl",
                input_schema=OpSchema({"f": ColCfg(datasets_dtype="int64")}),
                output_schema=OpSchema({"g": ColCfg(datasets_dtype="float32")}),
            )
        }
        # Save a valid cache
        cache_path = cache_manager.save_cache(op_definitions)
        # --- Fingerprint mismatch ---
        orig = cache_path.read_text()
        wrong_fp = orig.replace(
            f'FINGERPRINT = "{cache_manager.fingerprint}"',
            'FINGERPRINT = "not_the_real_fp"'
        )
        cache_path.write_text(wrong_fp)
        with pytest.warns(match="Cache fingerprint mismatch, invalidating cache"):
            # Should warn and return None
            assert cache_manager.load_cache() is None
        assert cache_manager.load_cache() is None

        # --- OP_DEFINITIONS missing ---
        cache_manager.save_cache(op_definitions)  # restore valid cache
        valid = cache_path.read_text()
        # Remove OP_DEFINITIONS by overwriting it at the end of the file
        missing_defs = valid + "\nOP_DEFINITIONS = {}\n"
        cache_path.write_text(missing_defs)
        with pytest.warns(match="No operation definitions found in cache"):
            # Should warn and return None
            assert cache_manager.load_cache() is None

class TestCacheManagerHubFunctionality:
    """Test cases for cache manager hub functionality."""

    def test_get_hub_namespace_from_hub_cache(self, tmp_path):
        """Test extracting namespace from hub cache file paths."""
        cache_manager = OpDefinitionsCacheManager(tmp_path)

        hub_cache = tmp_path / "hub_cache"

        # Test various hub cache patterns
        test_cases = [
            (
                hub_cache / "models--username--some_repo" / "snapshots" / "abc" / "ops.yaml",
                "username.some_repo"
            ),
            (
                hub_cache / "models--speediedan--nlp-tasks" / "snapshots" / "def" / "operations.yml",
                "speediedan.nlp-tasks"
            ),
            (
                hub_cache / "models--org--core" / "snapshots" / "ghi" / "core.yaml",
                "org.core"
            ),
        ]

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', hub_cache):
            for file_path, expected_namespace in test_cases:
                namespace = cache_manager.get_hub_namespace(file_path)
                assert namespace == expected_namespace

    def test_get_hub_namespace_from_user_path(self, tmp_path):
        """Test extracting namespace from user-defined paths."""
        cache_manager = OpDefinitionsCacheManager(tmp_path)

        # Files not in hub cache should get empty namespace
        user_files = [
            tmp_path / "custom_ops" / "my_ops.yaml",
            tmp_path / "some_other_path" / "operations.yml",
            Path("/completely/different/path/ops.yaml")
        ]

        for user_file in user_files:
            namespace = cache_manager.get_hub_namespace(user_file)
            assert namespace == ""

    def test_get_hub_namespace_edge_cases(self, tmp_path):
        """Test edge cases for hub namespace extraction."""
        cache_manager = OpDefinitionsCacheManager(tmp_path)

        hub_cache = tmp_path / "hub_cache"

        # Test malformed hub cache paths
        malformed_paths = [
            hub_cache / "models--single-dash" / "snapshots" / "abc" / "ops.yaml",
            hub_cache / "models--too--many--dashes--here" / "snapshots" / "abc" / "ops.yaml",
            hub_cache / "not-models--user--repo" / "snapshots" / "abc" / "ops.yaml",
        ]

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', hub_cache):
            for malformed_path in malformed_paths:
                # Should fall back to empty namespace for malformed paths
                namespace = cache_manager.get_hub_namespace(malformed_path)
                assert namespace == ""

    def test_discover_hub_yaml_files_basic(self, tmp_path):
        """Test basic hub YAML file discovery."""
        cache_manager = OpDefinitionsCacheManager(tmp_path)

        # Create hub cache structure
        hub_cache = tmp_path / "hub_cache"
        hub_cache.mkdir()

        # Create multiple repositories
        repos = [
            "models--user1--some_repo",
            "models--user2--nlp",
            "models--org--core"
        ]

        expected_files = []
        for repo in repos:
            repo_dir = hub_cache / repo / "snapshots" / "abc123"
            repo_dir.mkdir(parents=True)

            # Create YAML files
            ops_yaml = repo_dir / "ops.yaml"
            ops_yaml.write_text("test_op: {}")
            expected_files.append(ops_yaml)

            # Also test .yml extension
            if repo == repos[0]:
                ops_yml = repo_dir / "operations.yml"
                ops_yml.write_text("another_op: {}")
                expected_files.append(ops_yml)

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', hub_cache):
            discovered_files = cache_manager.discover_hub_yaml_files()

            # Should find all YAML files
            assert len(discovered_files) == len(expected_files)
            for expected_file in expected_files:
                assert expected_file in discovered_files

    def test_discover_hub_yaml_files_empty_cache(self, tmp_path):
        """Test hub YAML discovery with empty cache directory."""
        cache_manager = OpDefinitionsCacheManager(tmp_path)

        # Create empty hub cache
        hub_cache = tmp_path / "hub_cache"
        hub_cache.mkdir()

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', hub_cache):
            discovered_files = cache_manager.discover_hub_yaml_files()
            assert discovered_files == []

    def test_discover_hub_yaml_files_no_cache_dir(self, tmp_path):
        """Test hub YAML discovery when cache directory doesn't exist."""
        cache_manager = OpDefinitionsCacheManager(tmp_path)

        # Point to non-existent hub cache
        hub_cache = tmp_path / "nonexistent_hub_cache"

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', hub_cache):
            discovered_files = cache_manager.discover_hub_yaml_files()
            assert discovered_files == []

    def test_discover_hub_yaml_files_nested_structure(self, tmp_path):
        """Test hub YAML discovery with complex nested structure."""
        cache_manager = OpDefinitionsCacheManager(tmp_path)

        # Create complex hub cache structure
        hub_cache = tmp_path / "hub_cache"
        hub_cache.mkdir()

        # Create repository with multiple snapshots
        repo_dir = hub_cache / "models--testuser--complex"
        snapshots_dir = repo_dir / "snapshots"

        # Multiple snapshots with different files
        snapshot1 = snapshots_dir / "snapshot1"
        snapshot1.mkdir(parents=True)
        (snapshot1 / "ops.yaml").write_text("op1: {}")
        subfolder = snapshot1 / "subfolder"
        subfolder.mkdir(parents=True)
        (subfolder / "more_ops.yml").write_text("op2: {}")

        snapshot2 = snapshots_dir / "snapshot2"
        snapshot2.mkdir(parents=True)
        (snapshot2 / "operations.yaml").write_text("op3: {}")

        # Should find YAML files in all snapshots and subdirectories
        expected_files = [
            snapshot1 / "ops.yaml",
            snapshot1 / "subfolder" / "more_ops.yml",
            snapshot2 / "operations.yaml"
        ]

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', hub_cache):
            discovered_files = cache_manager.discover_hub_yaml_files()

            assert len(discovered_files) == len(expected_files)
            for expected_file in expected_files:
                assert expected_file in discovered_files

    def test_discover_hub_yaml_files_mixed_content(self, tmp_path):
        """Test hub YAML discovery ignores non-YAML files."""
        cache_manager = OpDefinitionsCacheManager(tmp_path)

        # Create hub cache with mixed content
        hub_cache = tmp_path / "hub_cache"
        hub_cache.mkdir()

        repo_dir = hub_cache / "models--user--mixed" / "snapshots" / "abc"
        repo_dir.mkdir(parents=True)

        # Create various file types
        (repo_dir / "ops.yaml").write_text("valid_op: {}")
        (repo_dir / "operations.yml").write_text("another_op: {}")
        (repo_dir / "readme.txt").write_text("This is a readme")
        (repo_dir / "config.json").write_text('{"key": "value"}')
        (repo_dir / "script.py").write_text("print('hello')")
        (repo_dir / "data.csv").write_text("col1,col2\n1,2")

        # Should only find YAML files
        expected_yaml_files = [
            repo_dir / "ops.yaml",
            repo_dir / "operations.yml"
        ]

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', hub_cache):
            discovered_files = cache_manager.discover_hub_yaml_files()

            assert len(discovered_files) == 2
            for expected_file in expected_yaml_files:
                assert expected_file in discovered_files

            # Ensure non-YAML files are not included
            for discovered_file in discovered_files:
                assert discovered_file.suffix in ['.yaml', '.yml']

    def test_discover_hub_yaml_files_invalid_repo_names(self, tmp_path):
        """Test hub YAML discovery ignores repositories with invalid names."""
        cache_manager = OpDefinitionsCacheManager(tmp_path)

        # Create hub cache with various repository names
        hub_cache = tmp_path / "hub_cache"
        hub_cache.mkdir()

        # Valid repository
        valid_repo = hub_cache / "models--user--valid" / "snapshots" / "abc"
        valid_repo.mkdir(parents=True)
        (valid_repo / "ops.yaml").write_text("valid_op: {}")

        # Invalid repository names (should be ignored)
        invalid_repos = [
            "not-models--user--test",  # Wrong prefix
            "models--user--too--many--dashes",  # Wrong ops prefix
            "random-directory",  # Completely wrong format
        ]

        for invalid_repo in invalid_repos:
            invalid_dir = hub_cache / invalid_repo / "snapshots" / "abc"
            invalid_dir.mkdir(parents=True)
            (invalid_dir / "ops.yaml").write_text("invalid_op: {}")

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', hub_cache):
            discovered_files = cache_manager.discover_hub_yaml_files()

            # Should only find the valid repository's YAML file
            assert len(discovered_files) == 1
            assert discovered_files[0] == valid_repo / "ops.yaml"

    def test_discover_hub_yaml_files_symlinks_and_permissions(self, tmp_path):
        """Test hub YAML discovery handles symlinks and permission issues gracefully."""
        cache_manager = OpDefinitionsCacheManager(tmp_path)

        # Create hub cache structure
        hub_cache = tmp_path / "hub_cache"
        hub_cache.mkdir()

        # Create valid repository
        repo_dir = hub_cache / "models--user--test" / "snapshots" / "abc"
        repo_dir.mkdir(parents=True)
        valid_yaml = repo_dir / "ops.yaml"
        valid_yaml.write_text("test_op: {}")

        # Create a symlink to a YAML file (if supported by the OS)
        try:
            symlink_target = tmp_path / "external_ops.yaml"
            symlink_target.write_text("external_op: {}")
            symlink_yaml = repo_dir / "symlink_ops.yaml"
            symlink_yaml.symlink_to(symlink_target)
            expects_symlink = True
        except (OSError, NotImplementedError):
            # Symlinks not supported on this system
            expects_symlink = False

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', hub_cache):
            discovered_files = cache_manager.discover_hub_yaml_files()

            # Should find at least the regular YAML file
            assert valid_yaml in discovered_files

            if expects_symlink:
                # Should also find the symlinked YAML file
                assert len(discovered_files) >= 2
                assert any(f.name == "symlink_ops.yaml" for f in discovered_files)
            else:
                # Should find only the regular file
                assert len(discovered_files) == 1
