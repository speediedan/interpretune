from __future__ import annotations
import pytest
import torch
from transformers import BatchEncoding
# Import classes from correct modules
from interpretune.analysis.ops.base import AnalysisOp, ChainedAnalysisOp, AnalysisBatch
from interpretune.analysis.ops.base import ColCfg, OpSchema
from interpretune.analysis.ops.base import AttrDict, CallableAnalysisOp, wrap_summary
# from interpretune.protocol import DIM_VAR


class TestAttrDict:
    """Tests for the AttrDict class."""

    def test_init_and_getattr(self):
        """Test initialization and attribute access."""
        attr_dict = AttrDict({"key1": "value1", "key2": 42})

        assert attr_dict["key1"] == "value1"
        assert attr_dict.key1 == "value1"
        assert attr_dict["key2"] == 42
        assert attr_dict.key2 == 42

    def test_setattr(self):
        """Test setting attributes."""
        attr_dict = AttrDict()

        attr_dict.key1 = "value1"
        assert attr_dict["key1"] == "value1"
        assert attr_dict.key1 == "value1"

        attr_dict["key2"] = 42
        assert attr_dict["key2"] == 42
        assert attr_dict.key2 == 42

    def test_getattr_error(self):
        """Test that accessing a non-existent attribute raises AttributeError."""
        attr_dict = AttrDict()

        with pytest.raises(AttributeError):
            _ = attr_dict.nonexistent

    def test_delattr(self):
        """Test deleting attributes."""
        attr_dict = AttrDict({"key1": "value1", "key2": 42})

        del attr_dict.key1
        assert "key1" not in attr_dict

        with pytest.raises(AttributeError):
            _ = attr_dict.key1

        del attr_dict["key2"]
        assert "key2" not in attr_dict

        with pytest.raises(AttributeError):
            del attr_dict.nonexistent


class TestColCfg:
    """Tests for the ColCfg class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        cfg = ColCfg(datasets_dtype="float32")

        assert cfg.datasets_dtype == "float32"
        assert cfg.required is True
        assert cfg.dyn_dim is None
        assert cfg.non_tensor is False
        assert cfg.per_latent is False
        assert cfg.per_sae_hook is False
        assert cfg.intermediate_only is False
        assert cfg.connected_obj == 'analysis_store'
        assert cfg.array_shape is None
        assert cfg.sequence_type is True
        assert cfg.array_dtype is None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        cfg = ColCfg(
            datasets_dtype="int64",
            required=False,
            dyn_dim=1,
            non_tensor=True,
            per_latent=True,
            per_sae_hook=True,
            intermediate_only=True,
            connected_obj='datamodule',
            array_shape=(None, 'batch_size', 10),
            sequence_type=False,
            array_dtype="float32"
        )

        assert cfg.datasets_dtype == "int64"
        assert cfg.required is False
        assert cfg.dyn_dim == 1
        assert cfg.non_tensor is True
        assert cfg.per_latent is True
        assert cfg.per_sae_hook is True
        assert cfg.intermediate_only is True
        assert cfg.connected_obj == 'datamodule'
        assert cfg.array_shape == (None, 'batch_size', 10)
        assert cfg.sequence_type is False
        assert cfg.array_dtype == "float32"

    def test_to_dict_method(self):
        """Test the to_dict method."""
        cfg = ColCfg(
            datasets_dtype="float32",
            dyn_dim=2,
            array_shape=(None, 'batch_size'),
        )

        result = cfg.to_dict()
        assert isinstance(result, dict)
        assert result["datasets_dtype"] == "float32"
        assert result["dyn_dim"] == 2
        assert result["array_shape"] == (None, 'batch_size')

    def test_from_dict_method(self):
        """Test the from_dict method."""
        data = {
            "datasets_dtype": "int64",
            "required": False,
            "dyn_dim": 3,
            "non_tensor": True,
        }

        cfg = ColCfg.from_dict(data)
        assert cfg.datasets_dtype == "int64"
        assert cfg.required is False
        assert cfg.dyn_dim == 3
        assert cfg.non_tensor is True

    def test_hash_method(self):
        """Test the __hash__ method."""
        # Test with simple values
        cfg1 = ColCfg(datasets_dtype="float32")
        cfg2 = ColCfg(datasets_dtype="float32")
        assert hash(cfg1) == hash(cfg2)

        # Test with array_shape containing dimensions
        cfg3 = ColCfg(datasets_dtype="float32", array_shape=(None, 'batch_size', 10))
        cfg4 = ColCfg(datasets_dtype="float32", array_shape=(None, 'batch_size', 10))
        assert hash(cfg3) == hash(cfg4)

        # Different configurations should have different hashes
        cfg5 = ColCfg(datasets_dtype="int64")
        assert hash(cfg1) != hash(cfg5)


class TestOpSchema:
    """Tests for the OpSchema class."""

    def test_init_and_getitem(self):
        """Test initialization and getting items."""
        schema = OpSchema({
            "field1": ColCfg(datasets_dtype="float32"),
            "field2": ColCfg(datasets_dtype="int64", required=False)
        })

        assert "field1" in schema
        assert "field2" in schema
        assert schema["field1"].datasets_dtype == "float32"
        assert schema["field2"].datasets_dtype == "int64"
        assert schema["field2"].required is False

    def test_items_keys_values(self):
        """Test dictionary-like methods."""
        schema = OpSchema({
            "field1": ColCfg(datasets_dtype="float32"),
            "field2": ColCfg(datasets_dtype="int64")
        })

        assert set(schema.keys()) == {"field1", "field2"}
        assert len(schema.values()) == 2
        assert len(list(schema.items())) == 2

        for key, value in schema.items():
            assert key in ["field1", "field2"]
            assert isinstance(value, ColCfg)

    def test_validate_wrong_values(self):
        """Test validation of values."""
        # Test with non-ColCfg values
        with pytest.raises(TypeError, match="Values must be ColCfg instances"):
            OpSchema({"field1": "not a ColCfg"})

    def test_hash_and_equality(self):
        """Test the __hash__ and __eq__ methods."""
        schema1 = OpSchema({
            "field1": ColCfg(datasets_dtype="float32"),
            "field2": ColCfg(datasets_dtype="int64")
        })

        schema2 = OpSchema({
            "field1": ColCfg(datasets_dtype="float32"),
            "field2": ColCfg(datasets_dtype="int64")
        })

        # Same schemas should have same hash
        assert hash(schema1) == hash(schema2)
        assert schema1 == schema2

        # Different schemas should have different hashes
        schema3 = OpSchema({
            "field1": ColCfg(datasets_dtype="float32"),
            "field3": ColCfg(datasets_dtype="int64")
        })

        assert hash(schema1) != hash(schema3)
        assert schema1 != schema3

        # Non-OpSchema comparison
        assert schema1 != "not an OpSchema"


class TestAnalysisBatch:
    """Tests for the AnalysisBatch class."""

    def test_init_and_access(self):
        """Test initialization and protocol attribute access."""
        # Initialize with dictionaries using valid AnalysisBatchProtocol attributes
        batch = AnalysisBatch()
        batch.logit_diffs = torch.tensor([0.5, 1.0])
        batch.answer_logits = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
        batch.loss = torch.tensor([0.1, 0.2])
        batch.labels = torch.tensor([1, 0])
        batch.prompts = ["prompt 1", "prompt 2"]

        # Check values using protocol-defined attributes
        assert torch.equal(batch.logit_diffs, torch.tensor([0.5, 1.0]))
        assert torch.equal(batch.answer_logits, torch.tensor([[0.7, 0.3], [0.2, 0.8]]))
        assert torch.equal(batch.loss, torch.tensor([0.1, 0.2]))
        assert torch.equal(batch.labels, torch.tensor([1, 0]))
        assert batch.prompts == ["prompt 1", "prompt 2"]

    def test_default_init(self):
        """Test initialization with default values."""
        batch = AnalysisBatch()

        # Verify protocol attributes are None by default
        assert not hasattr(batch, "logit_diffs") or batch.logit_diffs is None
        assert not hasattr(batch, "answer_logits") or batch.answer_logits is None
        assert not hasattr(batch, "loss") or batch.loss is None
        assert not hasattr(batch, "labels") or batch.labels is None
        assert not hasattr(batch, "prompts") or batch.prompts is None
        assert not hasattr(batch, "tokens") or batch.tokens is None
        assert not hasattr(batch, "cache") or batch.cache is None
        assert not hasattr(batch, "grad_cache") or batch.grad_cache is None

    def test_update(self):
        """Test updating attributes."""
        batch = AnalysisBatch()

        # Update batch with valid protocol attributes
        batch.update(
            logit_diffs=torch.tensor([0.1, 0.2]),
            loss=torch.tensor([0.3, 0.4]),
            prompts=["test prompt 1", "test prompt 2"]
        )

        # Check that attributes were updated
        assert torch.equal(batch.logit_diffs, torch.tensor([0.1, 0.2]))
        assert torch.equal(batch.loss, torch.tensor([0.3, 0.4]))
        assert batch.prompts == ["test prompt 1", "test prompt 2"]

        # Update existing attribute
        batch.update(logit_diffs=torch.tensor([0.5, 0.6]))
        assert torch.equal(batch.logit_diffs, torch.tensor([0.5, 0.6]))

    def test_getattr_setattr(self):
        """Test attribute access."""
        batch = AnalysisBatch()

        # Test valid attribute access
        batch.logit_diffs = torch.tensor([0.1, 0.2])
        assert torch.equal(batch.logit_diffs, torch.tensor([0.1, 0.2]))

        # Test non-existent attribute still raises AttributeError
        with pytest.raises(AttributeError):
            _ = batch.nonexistent_attribute

        # Test setting arbitrary attributes (no longer constrained by protocol)
        batch.custom_attribute = "custom value"
        assert batch.custom_attribute == "custom value"

        # Test accessing these arbitrary attributes
        assert batch.custom_attribute == "custom value"

        # Dictionary-style access still works
        assert batch["custom_attribute"] == "custom value"

    def test_to_cpu(self):
        """Test the to_cpu method."""
        # Create a batch with GPU tensors if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = AnalysisBatch()
        batch.logit_diffs = torch.tensor([1.0, 2.0], device=device)
        batch.answer_logits = torch.tensor([[3.0, 4.0]], device=device)

        # Move to CPU
        batch.to_cpu()

        # Check that tensors are on CPU
        assert batch.logit_diffs.device.type == "cpu"
        assert batch.answer_logits.device.type == "cpu"

    def test_nested_tensor_handling(self):
        """Test handling of nested tensor structures."""
        # Create a batch with nested dictionary tensors (common for attribution values)
        batch = AnalysisBatch()

        # Create nested attribute structure common in SAE analysis
        hook_dict = {
            "hook1": {"latent1": torch.tensor([1.0, 2.0]), "latent2": torch.tensor([3.0, 4.0])},
            "hook2": {"latent1": torch.tensor([5.0, 6.0]), "latent2": torch.tensor([7.0, 8.0])}
        }
        batch.attribution_values = hook_dict

        # Check that the nested structure was properly handled
        assert isinstance(batch.attribution_values, dict)
        assert "hook1" in batch.attribution_values
        assert "latent1" in batch.attribution_values["hook1"]
        assert torch.equal(batch.attribution_values["hook1"]["latent1"], torch.tensor([1.0, 2.0]))

        # Test moving nested structures to CPU
        batch.to_cpu()
        assert batch.attribution_values["hook1"]["latent1"].device.type == "cpu"

    def test_cycle_detection_in_to_cpu(self):
        """Test that to_cpu correctly handles cyclic references."""
        batch = AnalysisBatch()
        # Create a cyclic reference
        d1 = {"tensor": torch.tensor([1.0, 2.0])}
        d2 = {"ref": d1}
        d1["cycle"] = d2  # Create cycle

        batch.cyclic_data = d1

        # This should not cause infinite recursion
        batch.to_cpu()

        # Check tensor was moved to CPU
        assert batch.cyclic_data["tensor"].device.type == "cpu"

    def test_dictionary_access_fallback(self):
        """Test that dictionary access still works for backward compatibility."""
        batch = AnalysisBatch()

        # Set via protocol attributes
        batch.logit_diffs = torch.tensor([1.0, 2.0])
        batch.prompts = ["test1", "test2"]

        # Check via dictionary access
        assert torch.equal(batch["logit_diffs"], torch.tensor([1.0, 2.0]))
        assert batch["prompts"] == ["test1", "test2"]

        # Set via dictionary access
        batch["answer_logits"] = torch.tensor([[0.1, 0.9], [0.8, 0.2]])

        # Check via protocol attributes
        assert torch.equal(batch.answer_logits, torch.tensor([[0.1, 0.9], [0.8, 0.2]]))

    def test_attribute_protocol_validation(self):
        """Test that attributes are properly validated against the protocol."""
        batch = AnalysisBatch()

        # Valid attributes should work
        batch.logit_diffs = torch.tensor([1.0])
        batch.loss = torch.tensor([0.5])
        batch.labels = torch.tensor([1])
        batch.prompts = ["test"]
        batch.alive_latents = {"hook1": [1, 2, 3], "hook2": [4, 5, 6]}

        # All valid attributes from protocol should be accessible
        for attr in [
            "logit_diffs", "answer_logits", "loss", "preds", "labels",
            "orig_labels", "cache", "grad_cache", "answer_indices",
            "alive_latents", "correct_activations", "attribution_values",
            "tokens", "prompts"
        ]:
            # Should not raise AttributeError when getting attribute (might be None)
            _ = getattr(batch, attr, None)


class TestWrapSummary:
    """Tests for the wrap_summary function."""

    def test_wrap_summary_with_tokenizer(self):
        """Test wrap_summary with tokenizer."""
        # Create mock tokenizer and batch
        mock_tokenizer = type('MockTokenizer', (), {
            'batch_decode': lambda self, input_ids, **kwargs: [f"Text {i}" for i in range(len(input_ids))]
        })()

        batch = BatchEncoding({"input": torch.tensor([[1, 2], [3, 4]])})
        analysis_batch = AnalysisBatch()

        # Test with save_prompts=True
        result = wrap_summary(
            analysis_batch,
            batch,
            tokenizer=mock_tokenizer,
            save_prompts=True,
            save_tokens=False
        )

        assert result.prompts == ["Text 0", "Text 1"]
        assert not hasattr(result, "tokens") or result.tokens is None

        # Test with save_tokens=True
        result = wrap_summary(
            analysis_batch,
            batch,
            tokenizer=mock_tokenizer,
            save_prompts=False,
            save_tokens=True
        )

        assert not hasattr(result, "prompts") or result.prompts is None
        assert torch.equal(result.tokens, batch["input"].cpu())

    def test_wrap_summary_cache_cleaning(self):
        """Test wrap_summary removes cache attributes."""
        batch = BatchEncoding({"input": torch.tensor([[1, 2], [3, 4]])})
        analysis_batch = AnalysisBatch()

        # Add cache attributes that should be cleared
        analysis_batch.cache = "some cache"
        analysis_batch.grad_cache = "some grad cache"

        result = wrap_summary(analysis_batch, batch)

        assert result.cache is None
        assert result.grad_cache is None

    def test_wrap_summary_assert_tokenizer(self):
        """Test wrap_summary raises assertion when tokenizer is missing but prompts requested."""
        batch = BatchEncoding({"input": torch.tensor([[1, 2], [3, 4]])})
        analysis_batch = AnalysisBatch()

        with pytest.raises(AssertionError):
            wrap_summary(analysis_batch, batch, tokenizer=None, save_prompts=True)


class TestAnalysisOp:
    """Tests for the AnalysisOp class."""

    def test_initialize_basic_op(self):
        """Test basic initialization of an AnalysisOp."""
        # Initialize the operation
        op = AnalysisOp(
            name="test_op",
            description="Test operation",
            output_schema=OpSchema({"field": ColCfg(datasets_dtype="float32")}),
            input_schema=OpSchema({"input_field": ColCfg(datasets_dtype="int64")})
        )

        # Check properties
        assert op.name == "test_op"
        assert op.description == "Test operation"
        assert "field" in op.output_schema
        assert "input_field" in op.input_schema

    def test_initialize_with_schema(self):
        """Test initialization with schema objects."""
        # Define schemas
        input_schema = OpSchema({"input_field": ColCfg(datasets_dtype="int64")})
        output_schema = OpSchema({"output_field": ColCfg(datasets_dtype="float32")})

        # Initialize the operation
        op = AnalysisOp(
            name="test_op",
            description="Test operation",
            input_schema=input_schema,
            output_schema=output_schema
        )

        # Check schemas were properly set
        assert op.input_schema == input_schema
        assert op.output_schema == output_schema

    def test_equality(self):
        """Test equality comparison between operations."""
        op1 = AnalysisOp(name="op1", description="Operation 1", output_schema=OpSchema({}))
        op2 = AnalysisOp(name="op1", description="Different description", output_schema=OpSchema({}))
        op3 = AnalysisOp(name="op3", description="Operation 3", output_schema=OpSchema({}))

        # Operations with same name should be equal
        assert op1 == op2
        assert op1 != op3

        # Test comparison with a string
        assert op1 == "op1"
        assert op1 != "op3"

    def test_hash(self):
        """Test that operations with same name have same hash."""
        output_schema = OpSchema({"field": ColCfg(datasets_dtype="float32")})
        input_schema = OpSchema({"input": ColCfg(datasets_dtype="int64")})

        op1 = AnalysisOp(name="op1", description="Operation 1",
                         output_schema=output_schema, input_schema=input_schema)
        op2 = AnalysisOp(name="op1", description="Different description",
                         output_schema=output_schema, input_schema=input_schema)

        # Hash should be based on name, output_schema, and input_schema
        assert hash(op1) == hash(op2)

    def test_string_representation(self):
        """Test string representations of operations."""
        op = AnalysisOp(name="test_op", description="Test operation", output_schema=OpSchema({}))

        # Test __str__ and __repr__
        assert str(op) == "test_op: Test operation"
        assert "test_op" in repr(op)
        assert "Test operation" in repr(op)

    def test_alias_property(self):
        """Test alias property."""
        op = AnalysisOp(name="test_op", description="Test", output_schema=OpSchema({}))
        assert op.alias == "test_op"  # Default to name

        op = AnalysisOp(name="test_op", description="Test", output_schema=OpSchema({}), active_alias="custom")
        assert op.alias == "custom"  # Use active_alias if set

    def test_validate_input_schema(self):
        """Test input schema validation."""
        input_schema = OpSchema({
            "logit_diffs": ColCfg(datasets_dtype="float32", required=True)
        })
        op = AnalysisOp(
            name="test_op",
            description="Test op with schema validation",
            output_schema=OpSchema({}),
            input_schema=input_schema
        )

        # Valid inputs using protocol attributes
        batch = AnalysisBatch()
        batch.logit_diffs = torch.tensor([1.0, 2.0])
        op._validate_input_schema(batch, {})

        # Missing required field
        batch = AnalysisBatch()
        with pytest.raises(ValueError, match="Missing required.*logit_diffs"):
            op._validate_input_schema(batch, {})

    def test_validate_input_schema_from_batch(self):
        """Test validation of inputs from batch."""
        input_schema = OpSchema({
            "batch_field": ColCfg(datasets_dtype="float32", required=True, connected_obj="datamodule")
        })
        op = AnalysisOp(
            name="test_op",
            description="Test op with batch validation",
            output_schema=OpSchema({}),
            input_schema=input_schema
        )

        # Valid batch
        batch = {"batch_field": torch.tensor([1.0, 2.0])}
        op._validate_input_schema(AnalysisBatch(), batch)

        # Missing required field in batch
        with pytest.raises(ValueError, match="Missing required.*batch_field"):
            op._validate_input_schema(AnalysisBatch(), {})

    def test_process_batch(self):
        """Test the process_batch static method."""
        # Create a batch with valid protocol attributes
        batch = AnalysisBatch()
        batch.logit_diffs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        batch.alive_latents = {'hook1': [0, 1, 2]}

        # Create an output schema with dynamic dimension
        output_schema = OpSchema({
            "logit_diffs": ColCfg(datasets_dtype="float32", dyn_dim=1),
            "alive_latents": ColCfg(datasets_dtype="int32", per_latent=True)
        })

        # Process the batch
        result = AnalysisOp.process_batch(
            analysis_batch=batch,
            batch={},
            output_schema=output_schema,
            tokenizer=None,
            save_prompts=False,
            save_tokens=False
        )

        # Check that dimensions were swapped
        assert torch.equal(result.logit_diffs, torch.tensor([[1.0, 3.0], [2.0, 4.0]]))

    def test_process_batch_per_latent_serialization(self):
        """Test the process_batch method with per_latent serialization."""
        # Create a batch with per_latent data structure
        batch = AnalysisBatch()
        latent_dict = {
            'hook1': {
                0: torch.tensor([0.1, 0.2]),
                1: torch.tensor([0.3, 0.4]),
                2: torch.tensor([0.5, 0.6])
            }
        }
        batch.attribution_values = latent_dict

        # Create output schema with per_latent flag
        output_schema = OpSchema({
            "attribution_values": ColCfg(datasets_dtype="float32", per_latent=True)
        })

        # Process the batch
        result = AnalysisOp.process_batch(
            analysis_batch=batch,
            batch={},
            output_schema=output_schema
        )

        # Check the serialized structure
        assert 'hook1' in result.attribution_values
        assert 'latents' in result.attribution_values['hook1']
        assert 'per_latent' in result.attribution_values['hook1']
        assert result.attribution_values['hook1']['latents'] == [0, 1, 2]
        assert len(result.attribution_values['hook1']['per_latent']) == 3
        assert torch.equal(result.attribution_values['hook1']['per_latent'][0], torch.tensor([0.1, 0.2]))

    def test_save_batch(self):
        """Test the save_batch method."""
        op = AnalysisOp(
            name="test_op",
            description="Test op",
            output_schema=OpSchema({"logit_diffs": ColCfg(datasets_dtype="float32", dyn_dim=1)})
        )

        # Create a batch with valid protocol attributes
        batch = AnalysisBatch()
        batch.logit_diffs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Save the batch
        result = op.save_batch(batch, {})

        # Check that dimensions were swapped
        assert torch.equal(result.logit_diffs, torch.tensor([[1.0, 3.0], [2.0, 4.0]]))

    def test_callable_implementation(self):
        """Test calling an op with an implementation function."""
        def implementation(module, analysis_batch, batch, batch_idx, extra_arg=None, **kwargs):
            result = analysis_batch or AnalysisBatch()
            result.loss = torch.tensor(10.0)
            if extra_arg:
                result.logit_diffs = torch.tensor(extra_arg)
            return result

        op = AnalysisOp(
            name="test_op",
            description="Test op with implementation",
            output_schema=OpSchema({"loss": ColCfg(datasets_dtype="float32")}),
            callables={"implementation": implementation, "extra_arg": 42}
        )

        # Call the operation
        result = op(None, None, {}, 0)

        # Check that implementation was called using protocol attribute
        assert torch.equal(result.loss, torch.tensor(10.0))
        assert torch.equal(result.logit_diffs, torch.tensor(42))

    def test_no_implementation(self):
        """Test calling an op with no implementation."""
        op = AnalysisOp(
            name="test_op",
            description="Test op without implementation",
            output_schema=OpSchema({})
        )

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            op(None, None, {}, 0)

    def test_callable_property(self):
        """Test the callable property."""
        op = AnalysisOp(
            name="test_op",
            description="Test op",
            output_schema=OpSchema({})
        )

        callable_op = op.callable
        assert isinstance(callable_op, CallableAnalysisOp)
        assert callable_op.name == "test_op"
        assert callable_op.description == "Test op"

        # Check that calling .callable twice returns the same instance
        callable_op2 = op.callable
        assert callable_op is callable_op2

    def test_protocol_attribute_in_analysis(self):
        """Test that ops properly use protocol attributes."""
        def implementation(module, analysis_batch, batch, batch_idx, **kwargs):
            result = analysis_batch or AnalysisBatch()
            # Set multiple protocol-defined attributes
            result.logit_diffs = torch.tensor([0.5, 1.0])
            result.loss = torch.tensor([0.1, 0.2])
            result.labels = torch.tensor([1, 0])
            result.answer_logits = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
            return result

        op = AnalysisOp(
            name="test_op",
            description="Test with protocol attributes",
            output_schema=OpSchema({
                "logit_diffs": ColCfg(datasets_dtype="float32"),
                "loss": ColCfg(datasets_dtype="float32"),
                "labels": ColCfg(datasets_dtype="int64"),
                "answer_logits": ColCfg(datasets_dtype="float32")
            }),
            callables={"implementation": implementation}
        )

        # Call the operation
        result = op(None, None, {}, 0)

        # Check results using protocol attributes
        assert torch.equal(result.logit_diffs, torch.tensor([0.5, 1.0]))
        assert torch.equal(result.loss, torch.tensor([0.1, 0.2]))
        assert torch.equal(result.labels, torch.tensor([1, 0]))
        assert torch.equal(result.answer_logits, torch.tensor([[0.3, 0.7], [0.6, 0.4]]))

    def test_to_cpu_protocol_attributes(self):
        """Test that to_cpu properly handles protocol attributes."""
        # Create an implementation that sets protocol attributes
        def implementation(module, analysis_batch, batch, batch_idx, **kwargs):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            result = analysis_batch or AnalysisBatch()
            result.logit_diffs = torch.tensor([1.0, 2.0], device=device)
            result.loss = torch.tensor([0.1, 0.2], device=device)
            # Add a nested structure
            result.attribution_values = {
                "hook1": {"latent1": torch.tensor([0.5, 0.6], device=device)}
            }
            return result

        op = AnalysisOp(
            name="test_op",
            description="Test with protocol attributes",
            output_schema=OpSchema({
                "logit_diffs": ColCfg(datasets_dtype="float32"),
                "loss": ColCfg(datasets_dtype="float32"),
                "attribution_values": ColCfg(datasets_dtype="float32", per_latent=True)
            }),
            callables={"implementation": implementation}
        )

        # Call the operation
        result = op(None, None, {}, 0)

        # Call to_cpu
        result.to_cpu()

        # Check that tensors were moved to CPU
        assert result.logit_diffs.device.type == "cpu"
        assert result.loss.device.type == "cpu"
        assert result.attribution_values["hook1"]["latent1"].device.type == "cpu"


class TestCallableAnalysisOp:
    """Tests for the CallableAnalysisOp class."""

    def test_callable_analysis_op_wrapping(self):
        """Test that CallableAnalysisOp correctly wraps an AnalysisOp."""
        # Create a base op
        base_op = AnalysisOp(
            name="test_op",
            description="Test operation",
            output_schema=OpSchema({"field": ColCfg(datasets_dtype="float32")})
        )

        # Create callable wrapper manually
        callable_op = CallableAnalysisOp(base_op)

        # Check properties were correctly copied
        assert callable_op.name == "test_op"
        assert callable_op.description == "Test operation"
        assert callable_op.output_schema == base_op.output_schema
        assert callable_op.input_schema == base_op.input_schema

        # Check special attributes
        assert callable_op.__name__ == "test_op"
        assert callable_op.__module__ == "interpretune"
        assert callable_op.__qualname__ == "test_op"

    def test_callable_analysis_op_str_repr(self):
        """Test string representation of CallableAnalysisOp."""
        base_op = AnalysisOp(
            name="test_op",
            description="Test operation",
            output_schema=OpSchema({})
        )
        callable_op = CallableAnalysisOp(base_op)

        # Test string representations
        assert str(callable_op) == "test_op"
        assert "CallableAnalysisOp" in repr(callable_op)
        assert "test_op" in repr(callable_op)

    def test_callable_property_returns_self(self):
        """Test that callable property returns self for CallableAnalysisOp."""
        base_op = AnalysisOp(
            name="test_op",
            description="Test operation",
            output_schema=OpSchema({})
        )
        callable_op = CallableAnalysisOp(base_op)

        # callable property should return self
        assert callable_op.callable is callable_op


class TestChainedAnalysisOp:
    """Tests for the ChainedAnalysisOp class."""

    def test_chained_analysis_op(self):
        """Test ChainedAnalysisOp basic functionality."""
        # Create simple ops
        op1 = AnalysisOp(
            name="op1",
            description="Simple operation",
            output_schema=OpSchema({"result": ColCfg(datasets_dtype="float32")})
        )

        op2 = AnalysisOp(
            name="op2",
            description="Another operation",
            output_schema=OpSchema({"final": ColCfg(datasets_dtype="float32")}),
            input_schema=OpSchema({"result": ColCfg(datasets_dtype="float32")})
        )

        # Create the chained operation
        chained_op = ChainedAnalysisOp([op1, op2])

        # Test the chain's properties
        assert chained_op.name == "op1.op2"
        assert "Simple operation" in chained_op.description
        assert "Another operation" in chained_op.description
        assert chained_op.output_schema == op2.output_schema
        assert chained_op.input_schema == op1.input_schema

    def test_chained_op_with_alias(self):
        """Test ChainedAnalysisOp with a custom alias."""
        op1 = AnalysisOp(name="op1", description="First op", output_schema=OpSchema({}))
        op2 = AnalysisOp(name="op2", description="Second op", output_schema=OpSchema({}))

        # Set a custom alias
        chained_op = ChainedAnalysisOp([op1, op2], alias="my_chain")

        # Check the alias was set correctly
        assert chained_op.alias == "my_chain"
        assert chained_op.active_alias == "my_chain"

        # Check that each op in the chain also has the alias set
        for op in chained_op.chain:
            assert op.active_alias == "my_chain"

    def test_chained_op_call(self):
        """Test calling a ChainedAnalysisOp."""
        # Create mock ops
        def impl1(module, analysis_batch, batch, batch_idx, **kwargs):
            result = analysis_batch or AnalysisBatch()
            # Use proper protocol attributes
            result.loss = torch.tensor([1.0, 2.0])
            return result

        def impl2(module, analysis_batch, batch, batch_idx, **kwargs):
            # Use the output from the first op with proper protocol attribute access
            intermediate = analysis_batch.loss
            result = analysis_batch
            result.logit_diffs = intermediate * 2
            return result

        # Create ops with required parameters
        op1 = AnalysisOp(
            name="first",
            description="First operation",
            output_schema=OpSchema({"loss": ColCfg(datasets_dtype="float32")}),
            callables={"implementation": impl1}
        )
        op2 = AnalysisOp(
            name="second",
            description="Second operation",
            output_schema=OpSchema({"logit_diffs": ColCfg(datasets_dtype="float32")}),
            input_schema=OpSchema({"loss": ColCfg(datasets_dtype="float32")}),
            callables={"implementation": impl2}
        )

        # Create the chain
        chain = ChainedAnalysisOp([op1, op2])

        # Call the chain with mock data
        module = None  # No need for a real module in this test
        batch = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}  # Mock batch
        result = chain(module, None, batch, 0)

        # Check that both operations were executed in sequence using proper protocol attributes
        assert result.loss is not None
        assert result.logit_diffs is not None
        assert torch.equal(result.logit_diffs, torch.tensor([2.0, 4.0]))

    def test_empty_chain_error(self):
        """Test that an empty chain raises an error."""
        with pytest.raises(IndexError):
            ChainedAnalysisOp([])
