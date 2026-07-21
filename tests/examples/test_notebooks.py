"""Test notebook execution using papermill.

This module contains parameterized tests for Jupyter notebooks in the publish directory. Notebooks are executed with
different parameter configurations to ensure they work correctly.
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from interpretune.utils.resource_mgmt import cleanup_python_cuda
from tests.analysis_resource_utils import clear_nnsight_test_state
from tests.runif import RunIf

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

# Directory containing published notebooks (processed versions without dev cells)
NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "src" / "it_examples" / "notebooks" / "publish"

# Local Neuronpedia dev-webapp base URL used by local-dashboard-mode notebook tests
LOCAL_NP_WEBAPP_URL = os.environ.get("IT_LOCAL_NP_WEBAPP_URL", "http://localhost:3000")


def _local_neuronpedia_available(url: str = LOCAL_NP_WEBAPP_URL, timeout: float = 3.0) -> bool:
    """Lightweight availability probe for the local Neuronpedia dev webapp.

    Any HTTP response (including error statuses) counts as available — the probe only verifies a listening webapp so
    local-dashboard-mode tests can short-circuit to a skip when the local services are down.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError):
        return False


def execute_notebook_with_params(
    notebook_path: Path,
    parameters: dict[str, Any],
    output_dir: Path,
    timeout: int = 1800,  # 30 minutes
) -> Path:
    """Execute a notebook with parameters using papermill."""
    import papermill as pm

    output_notebook = output_dir / f"{notebook_path.stem}_output.ipynb"

    clear_nnsight_test_state(None)
    cleanup_python_cuda()

    try:
        # Execute the notebook from its directory to ensure relative imports work
        pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_notebook),
            parameters=parameters,
            timeout=timeout,
            log_output=True,
            cwd=str(notebook_path.parent),  # Execute from notebook's directory
        )
    finally:
        clear_nnsight_test_state(None)
        cleanup_python_cuda()

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
    params: dict[str, Any],
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


@RunIf(bf16_cuda=True)
@pytest.mark.parametrize("params", ATTRIBUTION_ANALYSIS_PARAMS)
def test_attribution_analysis_notebook(params: dict[str, Any], tmp_path: Path):
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


_hf_token_available = bool(os.environ.get("HF_TRIVIAL_OP_REPO_EXAMPLE_AUTH_KEY") or os.environ.get("HF_TOKEN"))


@pytest.mark.skipif(not _hf_token_available, reason="HF_TRIVIAL_OP_REPO_EXAMPLE_AUTH_KEY or HF_TOKEN required")
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


@RunIf(bf16_cuda=True)
@pytest.mark.parametrize("params", CIRCUIT_TRACER_PARAMS)
def test_circuit_tracer_notebooks(params: dict[str, Any], tmp_path: Path):
    """Test circuit tracer notebooks with different parameterizations."""
    notebook_path = NOTEBOOKS_DIR / "circuit_tracer_examples" / "circuit_tracer_adapter_example_basic.ipynb"

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


# Test parameters for CT Analysis Backend Demo notebook (public dashboard mode). Both
# circuit-tracer backends are validated; the notebook's in-cell sanity gates (unit direction
# norm, post-gap > pre-gap) catch wild cross-backend divergences without heavy assertions.
CT_ANALYSIS_BACKEND_PARAMS = [
    pytest.param(
        {"backend": "nnsight", "dashboard_mode": "public"},
        id="ct_analysis_backend_nnsight_public",
    ),
    pytest.param(
        {"backend": "transformerlens", "dashboard_mode": "public"},
        id="ct_analysis_backend_tl_public",
    ),
]

# Local-dashboard-mode params: identical analysis flow, feature-dashboard links point at the
# local Neuronpedia dev webapp instead of neuronpedia.org
CT_ANALYSIS_BACKEND_LOCAL_PARAMS = [
    pytest.param(
        {"backend": "nnsight", "dashboard_mode": "local", "local_webapp_url": LOCAL_NP_WEBAPP_URL},
        id="ct_analysis_backend_nnsight_local",
    ),
]


def _run_ct_analysis_backend_notebook(params: dict[str, Any], tmp_path: Path) -> None:
    notebook_path = NOTEBOOKS_DIR / "circuit_tracer_examples" / "ct_analysis_backend_demo.ipynb"

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


@RunIf(bf16_cuda=True)
@pytest.mark.parametrize("params", CT_ANALYSIS_BACKEND_PARAMS)
def test_ct_analysis_backend_notebook(params: dict[str, Any], tmp_path: Path):
    """Test CT analysis backend demo notebook (public dashboard mode)."""
    _run_ct_analysis_backend_notebook(params, tmp_path)


@RunIf(bf16_cuda=True, optional=True)
@pytest.mark.parametrize("params", CT_ANALYSIS_BACKEND_LOCAL_PARAMS)
def test_ct_analysis_backend_notebook_local(params: dict[str, Any], tmp_path: Path):
    """Test CT analysis backend demo notebook against a local Neuronpedia dev webapp."""
    if not _local_neuronpedia_available():
        pytest.skip(f"local Neuronpedia webapp not reachable at {LOCAL_NP_WEBAPP_URL}")
    _run_ct_analysis_backend_notebook(params, tmp_path)


# Test parameters for the concept-direction steering demo notebook (replaced the archived
# RTE-focused cross-backend demo, 2026-07-11; rebuilt on the gemma-2-2b public-dashboard default
# with a public/local dashboard-mode parameterization, 2026-07-16 — see EXPERIMENT_STATUS.md
# "7c Amendments" items 4-5 and their 2026-07-16 revision notes)
CT_CONCEPT_STEERING_PARAMS = [
    pytest.param(
        {"DASHBOARD_MODE": "public"},
        id="ct_concept_steering_public",
    ),
    # TransformerLens circuit-tracer backend; the notebook's gap assertions (both phases) are the
    # cross-backend sanity gate
    pytest.param(
        {"BACKEND": "transformerlens", "DASHBOARD_MODE": "public"},
        id="ct_concept_steering_tl_public",
    ),
]

# Local-dashboard-mode params (explanation generation disabled under test to avoid an
# explanation-CLI dependency): the gemma-2-2b default substrate plus the instruction-tuned
# gemma-3-1b-it + local-262k variant that exercises the locally-generated-explanations flow
CT_CONCEPT_STEERING_LOCAL_PARAMS = [
    pytest.param(
        {
            "DASHBOARD_MODE": "local",
            "LOCAL_WEBAPP_URL": LOCAL_NP_WEBAPP_URL,
            "GENERATE_MISSING_LOCAL_EXPLANATIONS": False,
        },
        id="ct_concept_steering_local_gemma2",
    ),
    pytest.param(
        {
            "REGISTRY_KEY": "gemma3.rte_demo.circuit_tracer_w_neuronpedia",
            "MODEL_NAME": "gemma-3-1b-it",
            "TRANSCODER_SET": None,
            "NEURONPEDIA_MODEL_ID": "gemma-3-1b-it",
            # must match the registry transcoder WIDTH (16k) — feature indices are only meaningful
            # within one feature space (2026-07-17 audit: earlier runs mislabeled links as 262k)
            "NEURONPEDIA_SOURCE_SET": "gemmascope-2-transcoder-16k",
            "CHAT_FORMAT_PROMPT": True,
            "DASHBOARD_MODE": "local",
            "LOCAL_WEBAPP_URL": LOCAL_NP_WEBAPP_URL,
            "GENERATE_MISSING_LOCAL_EXPLANATIONS": False,
        },
        id="ct_concept_steering_local_gemma3_16k",
    ),
]


def _run_ct_concept_steering_notebook(params: dict[str, Any], tmp_path: Path) -> None:
    notebook_path = NOTEBOOKS_DIR / "circuit_tracer_examples" / "ct_concept_steering_demo.ipynb"

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


@RunIf(bf16_cuda=True)
@pytest.mark.parametrize("params", CT_CONCEPT_STEERING_PARAMS)
def test_ct_concept_steering_notebook(params: dict[str, Any], tmp_path: Path):
    """Test the concept-direction steering demo notebook (store + embed paths, public dashboards)."""
    _run_ct_concept_steering_notebook(params, tmp_path)


@RunIf(bf16_cuda=True, optional=True)
@pytest.mark.parametrize("params", CT_CONCEPT_STEERING_LOCAL_PARAMS)
def test_ct_concept_steering_notebook_local(params: dict[str, Any], tmp_path: Path):
    """Test the steering demo against a local Neuronpedia webapp + DB (explanation coverage path).

    Skips when the local services are unavailable; the gemma-3 variant additionally requires the local 262k Monology
    dashboards for meaningful explanation coverage (the coverage check itself degrades gracefully to zero counts).
    """
    if not _local_neuronpedia_available():
        pytest.skip(f"local Neuronpedia webapp not reachable at {LOCAL_NP_WEBAPP_URL}")
    _run_ct_concept_steering_notebook(params, tmp_path)


# Test parameters for SAE Lens notebooks (parameterized by backend)
SAE_LENS_PARAMS = [
    pytest.param(
        {"backend": "transformerlens"},
        id="sae_lens_tl",
    ),
    pytest.param(
        {"backend": "nnsight"},
        id="sae_lens_nnsight",
    ),
]


@RunIf(bf16_cuda=True, standalone=True)
@pytest.mark.parametrize("params", SAE_LENS_PARAMS)
def test_sae_lens_notebooks(params: dict[str, Any], tmp_path: Path):
    """Test SAE Lens adapter notebooks with different backend parameterizations."""
    notebook_path = NOTEBOOKS_DIR / "saelens_adapter_example" / "saelens_adapter_example.ipynb"

    # Create output directory
    output_dir = tmp_path / "notebook_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute notebook with backend parameter
    output_notebook = execute_notebook_with_params(
        notebook_path=notebook_path,
        parameters=params,
        output_dir=output_dir,
    )

    # Verify output
    assert output_notebook.exists(), f"Output notebook not created at {output_notebook}"

    # Clean up
    _cleanup_notebook_artifacts()


def test_notebook_discovery():
    """Test that notebooks can be discovered in the publish directory."""
    assert NOTEBOOKS_DIR.exists(), f"Notebooks directory not found: {NOTEBOOKS_DIR}"

    # Find all .ipynb files
    notebook_files = list(NOTEBOOKS_DIR.rglob("*.ipynb"))
    assert len(notebook_files) > 0, f"No notebooks found in {NOTEBOOKS_DIR}"

    # Verify expected notebooks exist
    expected_notebooks = [
        "attribution_analysis/attribution_analysis.ipynb",
        "circuit_tracer_examples/circuit_tracer_adapter_example_basic.ipynb",
        "circuit_tracer_examples/ct_analysis_backend_demo.ipynb",
        "circuit_tracer_examples/ct_concept_steering_demo.ipynb",
        "example_op_collections/op_collection_example.ipynb",
        "neuronpedia_example/circuit_tracer_w_neuronpedia_example.ipynb",
        "saelens_adapter_example/saelens_adapter_example.ipynb",
    ]

    for expected in expected_notebooks:
        expected_path = NOTEBOOKS_DIR / expected
        assert expected_path.exists(), f"Expected notebook not found: {expected_path}"
