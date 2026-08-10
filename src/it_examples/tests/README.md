# Example Notebook Tests

This directory contains parameterized tests for Jupyter notebooks in the `src/it_examples/notebooks/` directory.

## Test Structure

- **test_notebooks.py**: Main test module for notebook execution
- **conftest.py**: Pytest configuration for notebook tests

## Testing Approach

Tests use `papermill` to execute notebooks with different parameter configurations:

1. **Parameterization**: Notebooks with a "parameters" cell can be tested with different input values
2. **Isolation**: Each test run uses a temporary output directory
3. **Cleanup**: Artifacts created during notebook execution are cleaned up automatically

## Running Notebook Tests

Notebook tests are marked as `standalone` and most require `bf16_cuda` capabilities.

### Via special_tests.sh (Recommended for GPU tests)

```bash
# Run all standalone tests (includes notebook tests)
./tests/special_tests.sh --mark_type=standalone
```

### Via pytest directly (for development)

```bash
# Run all notebook tests
pytest tests/examples/ -v

# Run a specific notebook test
pytest tests/examples/test_notebooks.py::test_gradient_flow_analysis_notebook -v

# Run with specific parameterization
pytest tests/examples/test_notebooks.py::test_gradient_flow_analysis_notebook[baseline_salient_logits=True_analysis=True_transcoder=SLT] -v
```

## Adding New Notebook Tests

1. **Add parameter cell**: If your notebook needs parameterization, add a cell tagged with `parameters` at the top
2. **Define parameters**: Add test parameterizations to `test_notebooks.py`
3. **Create test function**: Follow the pattern in existing test functions
4. **Mark appropriately**: Use `@RunIf(**{"standalone": True, "bf16_cuda": True})` for GPU notebooks

## Excluded Notebooks

The following notebooks are excluded from testing (old versions):
- `saelens_adapter_example.ipynb`
- `circuit_tracer_adapter_example_basic.ipynb`

## Artifact Cleanup

Tests automatically clean up temporary artifacts in `/tmp`:
- `it_analysis_*` directories
- `attribution_flow_analysis_*.log` files
- Other test-related temporary files

## CI/CD Integration

- **GitHub Actions**: Notebook tests are skipped (CPU-only CI)
- **Azure Pipelines**: Notebook tests run as part of standalone GPU tests
