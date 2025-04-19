from __future__ import annotations



# # Define simple test operations for chaining
# class TestOpA(AnalysisOp):
#     """Test operation that adds a constant value."""

#     def forward(self, input_tensor: torch.Tensor, add_value: float = 1.0) -> Dict[str, Any]:
#         return {"result": input_tensor + add_value}


# class TestOpB(AnalysisOp):
#     """Test operation that multiplies the result from previous operation."""

#     def forward(self, result: torch.Tensor, multiply_value: float = 2.0) -> Dict[str, Any]:
#         return {"final_result": result * multiply_value}


# class TestOpC(AnalysisOp):
#     """Test operation that requires multiple inputs."""

#     def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> Dict[str, Any]:
#         return {"combined": tensor1 + tensor2}


# class TestCompilerInit:
#     """Tests for the compiler initialization module."""

#     def test_compile_operation_chain_schema(self):
#         """Test that compile_operation_chain_schema correctly creates a schema for operation chains."""
#         # Create a chain of operations
#         ops = [TestOpA(), TestOpB()]

#         # Compile the chain schema
#         schema = compile_operation_chain_schema(ops)

#         # Verify the schema
#         assert isinstance(schema, OpSchema)

#         # Input schema should have the inputs from TestOpA
#         assert "input_tensor" in schema.input_schema
#         assert "add_value" in schema.input_schema

#         # Output schema should have the outputs from TestOpB (the last op)
#         assert "final_result" in schema.output_schema

#     def test_compile_operation_chain_schema_with_incompatible_ops(self):
#         """Test that compile_operation_chain_schema raises an error for incompatible operations."""
#         # Create incompatible operations (TestOpC expects tensor1 and tensor2, but TestOpB produces final_result)
#         ops = [TestOpB(), TestOpC()]

#         # Should raise an error because TestOpC needs tensor1 and tensor2, but TestOpB provides final_result
#         with pytest.raises(ValueError, match="Missing required input"):
#             compile_operation_chain_schema(ops)

#     def test_build_operation_chains(self):
#         """Test that build_operation_chains correctly builds operation chains from specifications."""
#         # Create a simple chain specification
#         chain_specs = {
#             "test_chain": [
#                 {"op": "test_op_a", "params": {"add_value": 5.0}},
#                 {"op": "test_op_b", "params": {"multiply_value": 3.0}}
#             ]
#         }

#         # Create a dictionary of available operations
#         available_ops = {
#             "test_op_a": TestOpA,
#             "test_op_b": TestOpB
#         }

#         # Build the chains
#         chains = build_operation_chains(chain_specs, available_ops)

#         # Verify the result
#         assert "test_chain" in chains
#         assert len(chains["test_chain"]) == 2
#         assert isinstance(chains["test_chain"][0], TestOpA)
#         assert isinstance(chains["test_chain"][1], TestOpB)

#         # Test the chain works as expected
#         test_input = torch.tensor([1.0, 2.0])

#         # Process through first op
#         result1 = chains["test_chain"][0].forward(test_input)
#         assert torch.equal(result1["result"], torch.tensor([6.0, 7.0]))  # 1+5, 2+5

#         # Process through second op
#         result2 = chains["test_chain"][1].forward(result1["result"])
#         assert torch.equal(result2["final_result"], torch.tensor([18.0, 21.0]))  # (1+5)*3, (2+5)*3

#     def test_build_operation_chains_with_unknown_op(self):
#         """Test that build_operation_chains handles unknown operations appropriately."""
#         # Create a chain specification with an unknown operation
#         chain_specs = {
#             "test_chain": [
#                 {"op": "unknown_op"},
#                 {"op": "test_op_b"}
#             ]
#         }

#         available_ops = {
#             "test_op_a": TestOpA,
#             "test_op_b": TestOpB
#         }

#         # Should raise a KeyError for unknown operation
#         with pytest.raises(KeyError, match="unknown_op"):
#             build_operation_chains(chain_specs, available_ops)
