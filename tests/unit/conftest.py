from __future__ import annotations
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any

import interpretune as it
from tests.configuration import get_deepcopied_session
from tests.orchestration import save_reload_results_dataset
from interpretune.config import AnalysisCfg
from interpretune.runners.analysis import maybe_init_analysis_cfg
from interpretune.analysis.core import get_module_dims
from interpretune.analysis.ops.base import OpWrapper
from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher


@pytest.fixture
def op_serialization_fixt():
    """Create a test utility for serializing and loading analysis results."""
    def _op_serialization_fixt(
        it_session,
        result_batches,
        batches,
        request=None,
    ):
        """Test serialization and loading of analysis results.

        Args:
            it_session: The interpretune session
            result_batch: The analysis batch result(s) to serialize (single or list)
            batch: The original input batch(es) (single or list)
            request: Optional pytest request object for test identification

        Returns:
            loaded_dataset: The loaded dataset with the serialized batch(es)
        """
        module = it_session.module

        # Generate a timestamp for the dataset directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Use test name if request is provided, otherwise use a generic name
        if request:
            test_name = request.node.name.replace("[", "_").replace("]", "_").replace(" ", "_")
            dataset_name = f"{test_name}_{timestamp}"
        else:
            dataset_name = f"test_dataset_{timestamp}"

        # Determine the save directory
        if hasattr(module.analysis_cfg.output_store, 'save_dir'):
            base_dir = module.analysis_cfg.output_store.save_dir
            save_dir = base_dir / "test_datasets" / dataset_name
        else:
            # Fallback to temporary directory if no save_dir configured
            tmp_dir = tempfile.mkdtemp()
            base_dir = Path(tmp_dir)
            save_dir = base_dir / dataset_name

        # Store the original save_dir and restore it after the test
        original_save_dir = None
        if hasattr(module.analysis_cfg.output_store, 'save_dir'):
            original_save_dir = module.analysis_cfg.output_store.save_dir

        try:
            # Create the directory structure
            save_dir.parent.mkdir(exist_ok=True, parents=True)

            # Set the temporary save_dir
            module.analysis_cfg.output_store.save_dir = save_dir

            # Check if we're dealing with multiple batches
            is_multi_batch = isinstance(result_batches, list)

            # Ensure batch is also a list if result_batch is a list
            if is_multi_batch and not isinstance(batches, list):
                raise ValueError("If result_batch is a list, batch must also be a list")

            # If single batch, convert to list for uniform processing
            if not is_multi_batch:
                result_batches = [result_batches]
                batches = [batches]

            return save_reload_results_dataset(it_session, result_batches, batches)

        finally:
            # Restore original save_dir if needed
            if original_save_dir is not None:
                module.analysis_cfg.output_store.save_dir = original_save_dir

    return _op_serialization_fixt


@pytest.fixture
def initialized_analysis_cfg():
    def _initialized_analysis_cfg(fixture, target_op: Any = it.logit_diffs_attr_ablation):
        it_session = get_deepcopied_session(fixture.it_session)
        # Configure the analysis
        analysis_cfg = AnalysisCfg(target_op=target_op, ignore_manual=True, save_tokens=False,
                                   sae_analysis_targets=fixture.test_cfg().sae_analysis_targets)
        # Initialize analysis config on the module
        maybe_init_analysis_cfg(it_session.module, analysis_cfg)

        batch_size, max_answer_tokens, num_classes, vocab_size, max_seq_len = get_module_dims(it_session.module)
        dim_vars = {
            'batch_size': batch_size,
            'max_answer_tokens': max_answer_tokens,
            'num_classes': num_classes,
            'vocab_size': vocab_size,
            'max_seq_len': max_seq_len,
        }
        return it_session, dim_vars

    return _initialized_analysis_cfg

@pytest.fixture
def test_ops_yaml(tmp_path):
    """Create a temporary YAML file with test operation definitions."""
    # Create main YAML file with primary test operations
    main_yaml_content = """
test_op:
  implementation: tests.unit.test_analysis_ops_base.op_impl_test
  description: A test operation for unit tests
  input_schema:
    input1:
      datasets_dtype: float32
      required: true
  output_schema:
    output1:
      datasets_dtype: float32
      required: true
  aliases:
    - test_alias

another_test_op:
  implementation: tests.unit.test_analysis_ops_base.op_impl_test
  description: Another test operation
  input_schema:
    input2:
      datasets_dtype: int64
      required: true
  output_schema:
    output2:
      datasets_dtype: int64
      required: true
"""

    # Create a subdirectory for additional YAML files
    sub_dir = tmp_path / "sub_ops"
    sub_dir.mkdir()

    # Create additional YAML files in subdirectory
    extra_yaml1_content = """
extra_op1:
  implementation: tests.unit.test_analysis_ops_base.op_impl_test
  description: Extra operation 1
  input_schema:
    extra_input1:
      datasets_dtype: string
      required: false
  output_schema:
    extra_output1:
      datasets_dtype: string
      required: false
"""

    extra_yaml2_content = """
extra_op2:
  implementation: tests.unit.test_analysis_ops_base.op_impl_test
  description: Extra operation 2
  input_schema:
    extra_input2:
      datasets_dtype: bool
      required: true
  output_schema:
    extra_output2:
      datasets_dtype: bool
      required: true
"""

    # Write the main YAML file
    main_yaml_file = tmp_path / "test_ops.yaml"
    with open(main_yaml_file, "w") as f:
        f.write(main_yaml_content)

    # Write additional YAML files in subdirectory
    extra_yaml1_file = sub_dir / "extra_ops1.yaml"
    with open(extra_yaml1_file, "w") as f:
        f.write(extra_yaml1_content)

    extra_yaml2_file = sub_dir / "extra_ops2.yaml"
    with open(extra_yaml2_file, "w") as f:
        f.write(extra_yaml2_content)

    # Return a structure containing paths for different test scenarios
    return {
        'main_file': main_yaml_file,
        'sub_dir': sub_dir,
        'all_files': [main_yaml_file, extra_yaml1_file, extra_yaml2_file],
        'main_dir': tmp_path
    }

@pytest.fixture
def test_dispatcher(test_ops_yaml):
    """Create a test dispatcher with test operation definitions."""
    # Create a test dispatcher that loads from our test YAML main file
    dispatcher = AnalysisOpDispatcher(yaml_paths=test_ops_yaml['main_file'])
    dispatcher.load_definitions()
    return dispatcher

@pytest.fixture
def multi_file_test_dispatcher(test_ops_yaml):
    """Create a test dispatcher that discovers YAML files from a directory."""
    # Create a dispatcher that discovers all YAML files in the directory
    dispatcher = AnalysisOpDispatcher(yaml_paths=test_ops_yaml['main_dir'])
    dispatcher.load_definitions()
    return dispatcher

@pytest.fixture
def target_module():
    """Create a mock module to use as a target for OpWrapper."""
    return type('MockModule', (), {})()

def test_initialize():
    """Test initializing the OpWrapper class."""
    # Create a mock module
    mock_module = type('MockModule', (), {})()

    # Initialize OpWrapper with the module
    OpWrapper.initialize(mock_module)

    # Check that the target module was set
    assert OpWrapper._target_module is mock_module
