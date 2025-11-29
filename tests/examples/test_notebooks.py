"""Test notebook execution using papermill.

This module contains parameterized tests for Jupyter notebooks in the publish directory. Notebooks are executed with
different parameter configurations to ensure they work correctly.
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import pytest

from tests.runif import RunIf

# Directory containing published notebooks (processed versions without dev cells)
NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "src" / "it_examples" / "notebooks" / "publish"


def execute_notebook_with_params(
    notebook_path: Path,
    parameters: Dict[str, Any],
    output_dir: Path,
    timeout: int = 1800,  # 30 minutes
) -> Path:
    """Execute a notebook with parameters using papermill."""
    import papermill as pm

    output_notebook = output_dir / f"{notebook_path.stem}_output.ipynb"

    # Execute the notebook from its directory to ensure relative imports work
    pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_notebook),
        parameters=parameters,
        timeout=timeout,
        log_output=True,
        cwd=str(notebook_path.parent),  # Execute from notebook's directory
    )

    return output_notebook


def _cleanup_notebook_artifacts():
    """Clean up common notebook execution artifacts."""
    cleanup_patterns = [
        # "/tmp/it_analysis_*",
        "/tmp/attribution_flow_analysis_*.log",
        # "/tmp/gen_it_coverage_*",
        # "/tmp/special_tests_*",
    ]

    for pattern in cleanup_patterns:
        for path in glob.glob(pattern):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except (OSError, PermissionError):
                # Ignore cleanup failures
                pass


def validate_notebook_outputs(
    output_notebook: Path,
    params: Dict[str, Any],
    check_prompt_errors: bool = True,
    check_analysis_points: bool = True,
    check_prompt_success: bool = True,
) -> None:
    """Validate notebook execution outputs.

    Args:
        output_notebook: Path to the executed notebook
        params: Parameters used for notebook execution
        check_prompt_errors: Whether to check for prompt processing errors
        check_analysis_points: Whether to check for missing analysis point data
        check_prompt_success: Whether to check that at least one prompt succeeded

    Raises:
        pytest.fail: If validation checks fail
        AssertionError: If prompt success count validation fails
    """
    import nbformat
    import re

    # Read the executed notebook
    with open(output_notebook) as f:
        nb = nbformat.read(f, as_version=4)

    # Check for errors in prompt processing
    prompt_errors = []
    missing_analysis_points = []
    prompt_success_count = None

    for cell in nb.cells:
        if cell.cell_type == "code" and cell.get("outputs"):
            for output in cell.outputs:
                if output.output_type in ("stream", "execute_result", "display_data"):
                    text = output.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)

                    # Check for prompt processing errors
                    if check_prompt_errors and "Error processing prompt:" in text:
                        prompt_errors.append(text)

                    # Check for missing analysis points (if analysis injection enabled)
                    if check_analysis_points and params.get("enable_analysis_injection", False):
                        if "No analysis data for analysis point" in text:
                            missing_analysis_points.append(text)

                    # Extract prompt success count
                    if check_prompt_success and "Processed" in text and "prompts successfully" in text:
                        match = re.search(r"Processed (\d+) prompts successfully", text)
                        if match:
                            prompt_success_count = int(match.group(1))

    # Fail test if any prompts had errors
    if check_prompt_errors and prompt_errors:
        error_msg = "\n".join(prompt_errors)
        pytest.fail(f"Prompt processing errors detected:\n{error_msg}")

    # Verify at least one prompt was processed successfully
    if check_prompt_success and prompt_success_count is not None:
        assert prompt_success_count > 0, (
            f"Expected at least 1 prompt to be processed successfully, got {prompt_success_count}"
        )

    # Fail test if analysis injection was enabled but analysis points didn't produce data
    if check_analysis_points and params.get("enable_analysis_injection", False) and missing_analysis_points:
        # For now, we expect 0 missing analysis points when analysis injection is enabled
        error_msg = "\n".join(missing_analysis_points)
        pytest.fail(
            f"Analysis injection enabled but {len(missing_analysis_points)} analysis points "
            f"did not produce data:\n{error_msg}"
        )


# Test parameters for attribution analysis notebook
ATTRIBUTION_ANALYSIS_PARAMS = [
    pytest.param(
        {
            "use_baseline_salient_logits": True,
            "enable_analysis_injection": True,
            "use_baseline_transcoder_arch": True,  # SLT
        },
        id="analysis_inj_salient_logits_SLT",
    ),
    pytest.param(
        {
            "use_baseline_salient_logits": True,
            "enable_analysis_injection": True,
            "use_baseline_transcoder_arch": False,  # CLT
        },
        id="analysis_inj_salient_logits_CLT",
    ),
]


@RunIf(standalone=True, bf16_cuda=True)
@pytest.mark.parametrize("params", ATTRIBUTION_ANALYSIS_PARAMS)
def test_attribution_analysis_notebook(params: Dict[str, Any], tmp_path: Path):
    """Test attribution analysis notebook with different parameterizations."""
    notebook_path = NOTEBOOKS_DIR / "attribution_analysis" / "attribution_analysis.ipynb"

    # Create output directory
    output_dir = tmp_path / "notebook_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute notebook with parameters
    output_notebook = execute_notebook_with_params(
        notebook_path=notebook_path,
        parameters=params,
        output_dir=output_dir,
    )

    # Verify output
    assert output_notebook.exists(), f"Output notebook not created at {output_notebook}"

    # Validate notebook outputs
    validate_notebook_outputs(output_notebook, params)

    # Clean up
    _cleanup_notebook_artifacts()


@RunIf(standalone=True)
@pytest.mark.parametrize("notebook_file", ["op_collection_example.ipynb"])
def test_op_collection_notebooks(notebook_file: str, tmp_path: Path):
    """Test operation collection notebooks."""
    notebook_path = NOTEBOOKS_DIR / "example_op_collections" / notebook_file

    # Create output directory
    output_dir = tmp_path / "notebook_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute notebook (no parameters needed)
    output_notebook = execute_notebook_with_params(
        notebook_path=notebook_path,
        parameters={},
        output_dir=output_dir,
    )

    # Verify output
    assert output_notebook.exists(), f"Output notebook not created at {output_notebook}"

    # Clean up
    _cleanup_notebook_artifacts()


# Test parameters for circuit tracer notebooks
CIRCUIT_TRACER_PARAMS = [
    pytest.param(
        {
            "use_baseline_salient_logits": True,
            "enable_analysis_injection": False,
            "use_baseline_transcoder_arch": False,  # CLT
        },
        id="ct_salient_logits_CLT",
    ),
    pytest.param(
        {
            "use_baseline_salient_logits": True,
            "enable_analysis_injection": False,
            "use_baseline_transcoder_arch": True,  # SLT
        },
        id="ct_w_neuronpedia_SLT",
    ),
]


@RunIf(standalone=True, bf16_cuda=True)
@pytest.mark.parametrize("params", CIRCUIT_TRACER_PARAMS)
def test_circuit_tracer_notebooks(params: Dict[str, Any], tmp_path: Path):
    """Test circuit tracer notebooks with different parameterizations."""
    # Use the CLT notebook for these tests
    notebook_path = NOTEBOOKS_DIR / "circuit_tracer_examples" / "circuit_tracer_adapter_example_basic_clt.ipynb"

    # Create output directory
    output_dir = tmp_path / "notebook_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute notebook with parameters
    output_notebook = execute_notebook_with_params(
        notebook_path=notebook_path,
        parameters=params,
        output_dir=output_dir,
    )

    # Verify output
    assert output_notebook.exists(), f"Output notebook not created at {output_notebook}"

    # Clean up
    _cleanup_notebook_artifacts()


@RunIf(standalone=True, bf16_cuda=True)
@pytest.mark.parametrize("notebook_file", ["saelens_adapter_example_registry.ipynb"])
def test_sae_lens_notebooks(notebook_file: str, tmp_path: Path):
    """Test SAE Lens adapter notebooks."""
    notebook_path = NOTEBOOKS_DIR / "saelens_adapter_example" / notebook_file

    # Create output directory
    output_dir = tmp_path / "notebook_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute notebook (no parameters needed for registry test)
    output_notebook = execute_notebook_with_params(
        notebook_path=notebook_path,
        parameters={},
        output_dir=output_dir,
    )

    # Verify output
    assert output_notebook.exists(), f"Output notebook not created at {output_notebook}"

    # Clean up
    _cleanup_notebook_artifacts()


@RunIf(standalone=True)
def test_notebook_discovery():
    """Test that notebooks can be discovered in the publish directory."""
    assert NOTEBOOKS_DIR.exists(), f"Notebooks directory not found: {NOTEBOOKS_DIR}"

    # Find all .ipynb files
    notebook_files = list(NOTEBOOKS_DIR.rglob("*.ipynb"))
    assert len(notebook_files) > 0, f"No notebooks found in {NOTEBOOKS_DIR}"

    # Verify expected notebooks exist
    expected_notebooks = [
        "attribution_analysis/attribution_analysis.ipynb",
        "circuit_tracer_examples/circuit_tracer_adapter_example_basic_clt.ipynb",
        "circuit_tracer_examples/circuit_tracer_adapter_example_basic.ipynb",
        "example_op_collections/op_collection_example.ipynb",
        "neuronpedia_example/circuit_tracer_w_neuronpedia_example.ipynb",
        "saelens_adapter_example/saelens_adapter_example_registry.ipynb",
        "saelens_adapter_example/saelens_adapter_example.ipynb",
    ]

    for expected in expected_notebooks:
        expected_path = NOTEBOOKS_DIR / expected
        assert expected_path.exists(), f"Expected notebook not found: {expected_path}"
