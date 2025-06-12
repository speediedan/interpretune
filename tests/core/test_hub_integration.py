"""Integration tests for HuggingFace Hub analysis operations."""
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import os

from tests.warns import unmatched_warns
from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher
from interpretune.analysis.ops.compiler.cache_manager import OpDefinitionsCacheManager


class TestHubIntegration:
    """Integration tests for the complete hub operations workflow."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.hub_cache = self.temp_dir / "hub_cache"
        self.hub_cache.mkdir()

        self.cache_dir = Path(tempfile.mkdtemp())

        # Create mock hub structure
        self.repo_cache = self.hub_cache / "models--testuser--test"
        self.snapshots_dir = self.repo_cache / "snapshots"
        self.snapshot_dir = self.snapshots_dir / "abc123def456"
        self.snapshot_dir.mkdir(parents=True)

        # Create test operation definition
        self.test_ops_yaml = self.snapshot_dir / "ops.yaml"
        self.test_ops_yaml.write_text("""
test_hub_op:
  description: A test operation from hub
  implementation: test_module.test_function
  aliases: ['hub_test']
  input_schema:
    text_data:
      datasets_dtype: string
  output_schema:
    processed_data:
      datasets_dtype: string

another_op:
  description: Another test operation
  implementation: test_module.another_function
  required_ops: ['test_hub_op']
  input_schema:
    input_text:
      datasets_dtype: string
  output_schema:
    result:
      datasets_dtype: string
""")

        # Create additional ops path
        self.custom_ops_root_dir = Path(tempfile.mkdtemp())
        self.custom_ops_dir = self.custom_ops_root_dir / "custom_ops"
        self.custom_ops_dir.mkdir()
        self.custom_ops_yaml = self.custom_ops_dir / "my_ops.yaml"
        self.custom_ops_yaml.write_text("""
custom_op:
  description: A custom operation
  implementation: custom.module.custom_function
  input_schema:
    data:
      datasets_dtype: string
  output_schema:
    result:
      datasets_dtype: string
""")

    def teardown_method(self):
        """Clean up test environment after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)


    def test_namespace_collision_handling(self, recwarn):
        """Test handling of namespace collisions between operations."""

        # Create another operation with same name but different namespace
        another_repo_cache = self.hub_cache / "models--otheruser--test"
        another_snapshots = another_repo_cache / "snapshots"
        another_snapshot = another_snapshots / "def789abc123"
        another_snapshot.mkdir(parents=True)

        another_ops_yaml = another_snapshot / "ops.yaml"
        another_ops_yaml.write_text("""
test_hub_op:
  description: Same-named operation from different user
  implementation: other_module.test_function
  input_schema:
    data:
      datasets_dtype: string
  output_schema:
    result:
      datasets_dtype: string
""")

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', self.hub_cache), \
             patch('interpretune.analysis.ops.dispatcher.IT_ANALYSIS_CACHE', self.cache_dir), \
             patch('interpretune.analysis.IT_ANALYSIS_OP_PATHS', []):

            dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)

            dispatcher.load_definitions()

            all_ops = dispatcher.list_operations()

            # Both operations should exist with different namespaces
            assert "testuser.test.test_hub_op" in all_ops
            assert "otheruser.test.test_hub_op" in all_ops
            assert "test_hub_op" in all_ops  # Unnamespaced version
            assert 'testuser.test.hub_test' in dispatcher._aliases
            # assert 'otheruser.test.hub_test' not in dispatcher._aliases
            assert dispatcher._op_definitions['testuser.test.hub_test'].name == 'testuser.test.test_hub_op'
            assert 'testuser.test.hub_test' in dispatcher._op_to_aliases['test_hub_op']
            assert dispatcher._op_definitions['model_forward_cache'].name == 'model_cache_forward'
            assert dispatcher._op_definitions['otheruser.test.test_hub_op'].name == 'otheruser.test.test_hub_op'
            assert dispatcher._op_definitions['test_hub_op'].name == 'testuser.test.test_hub_op'
            assert 'hub_test' in all_ops

            assert 'hub_test' in dispatcher._aliases
            w_expected = ['The fully-qualified name will need.*', ".*multiple matching operations found.*"]
            unmatched = unmatched_warns(rec_warns=recwarn.list, expected_warns=w_expected)
            assert not unmatched

    def test_operation_with_dependencies(self):
        """Test loading operations with dependencies."""

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', self.hub_cache), \
             patch('interpretune.analysis.IT_ANALYSIS_OP_PATHS', []):

            dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)
            dispatcher.load_definitions()

            # Get the operation with dependencies
            another_op_def = dispatcher._op_definitions["testuser.test.another_op"]
            assert another_op_def is not None

            # Dependencies should be properly namespaced
            assert another_op_def.required_ops == ["testuser.test.test_hub_op"]


    def test_error_handling_invalid_yaml(self):
        """Test graceful handling of invalid YAML files."""

        # Create invalid YAML file
        invalid_yaml = self.snapshot_dir / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [")

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', self.hub_cache), \
             patch('interpretune.analysis.IT_ANALYSIS_OP_PATHS', []):

            dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)

            # Should not raise exception, but handle gracefully
            dispatcher.load_definitions()

            # Should still load valid operations
            all_ops = dispatcher.list_operations()
            assert "testuser.test.test_hub_op" in all_ops

    def test_error_handling_missing_implementation(self):
        """Test handling of operations with missing implementation."""

        # Create operation with missing implementation
        incomplete_yaml = self.snapshot_dir / "incomplete.yaml"
        incomplete_yaml.write_text("""
incomplete_op:
  description: Operation missing implementation
  input_schema:
    data:
      datasets_dtype: string
  output_schema:
    result:
      datasets_dtype: string
""")

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', self.hub_cache), \
             patch('interpretune.analysis.IT_ANALYSIS_OP_PATHS', []):

            dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)
            dispatcher.load_definitions()

            # Should handle missing implementation gracefully
            all_ops = dispatcher.list_operations()

            # Valid operations should still be loaded
            assert "testuser.test.test_hub_op" in all_ops

    def test_environment_variable_configuration(self):
        """Test configuration via environment variables."""

        with patch.dict(os.environ, {'IT_ANALYSIS_HUB_CACHE': str(self.hub_cache)}), \
        patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', self.hub_cache):

            cache_manager = OpDefinitionsCacheManager(self.cache_dir)

            # Should use custom paths
            yaml_files = cache_manager.discover_hub_yaml_files()

            # Should find files from custom ops dir
            yaml_names = [f.name for f in yaml_files]
            assert "ops.yaml" in yaml_names

    def test_performance_with_large_cache(self):
        """Test performance characteristics with many cached repositories."""

        # Create multiple fake repositories
        for i in range(10):
            repo_cache = self.hub_cache / f"models--user{i}--repo{i}"
            snapshot_dir = repo_cache / "snapshots" / f"snapshot{i}"
            snapshot_dir.mkdir(parents=True)

            ops_file = snapshot_dir / "ops.yaml"
            ops_file.write_text(f"""
op_{i}:
  description: Operation {i}
  implementation: module{i}.function{i}
  input_schema:
    data:
      datasets_dtype: string
  output_schema:
    result:
      datasets_dtype: string
""")

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', self.hub_cache), \
             patch('interpretune.analysis.IT_ANALYSIS_OP_PATHS', []):

            dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)

            # Loading should complete in reasonable time
            import time
            start_time = time.time()
            dispatcher.load_definitions()
            load_time = time.time() - start_time

            # Should complete within reasonable time (generous for testing)
            assert load_time < 5.0

            # Should have loaded all operations
            all_ops = dispatcher.list_operations()
            assert len(all_ops) >= 10  # At least our 10 test ops

            # Should have proper namespacing
            for i in range(10):
                expected_op = f"user{i}.repo{i}.op_{i}"
                assert expected_op in all_ops


class TestDynamicModuleIntegration:
    """Integration tests for dynamic module loading with dispatcher."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "cache"
        self.cache_dir.mkdir()

    def teardown_method(self):
        """Clean up test environment after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dispatcher_dynamic_loading_integration(self):
        """Test complete integration of dispatcher with dynamic module loading."""
        # Set up mock hub repository structure
        repo_cache = self.cache_dir / "models--testuser--test_repo"
        snapshot_dir = repo_cache / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)

        # Create test function file
        test_module = snapshot_dir / "__init__.py"
        test_module.write_text('''
def test_dynamic_function(module, analysis_batch, batch, batch_idx):
    """A test function for dynamic loading."""
    from interpretune.analysis.ops.base import AnalysisBatch

    if analysis_batch is None:
        analysis_batch = AnalysisBatch()

    # Add some test data
    analysis_batch["test_result"] = "dynamic_success"
    return analysis_batch

def another_dynamic_function(module, analysis_batch, batch, batch_idx):
    """Another test function for dynamic loading."""
    from interpretune.analysis.ops.base import AnalysisBatch

    if analysis_batch is None:
        analysis_batch = AnalysisBatch()

    analysis_batch["another_result"] = "another_dynamic_success"
    return analysis_batch
''')

        # Create a YAML file to define the operation
        ops_yaml = snapshot_dir / "ops.yaml"
        ops_yaml.write_text('''
test_dynamic_function:
  description: A dynamically loaded test function
  implementation: __init__.test_dynamic_function
  input_schema: {}
  output_schema: {}
''')

        # Mock all hub and dynamic module interactions to avoid HTTP calls
        with patch('interpretune.analysis.ops.hub_manager.snapshot_download', return_value=str(snapshot_dir)), \
             patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', self.cache_dir), \
             patch('interpretune.analysis.ops.dynamic_module_utils.get_cached_module_file_it',
                   return_value=str(test_module)), \
             patch('interpretune.analysis.ops.compiler.cache_manager.scan_cache_dir') as mock_scan:

            # Mock scan_cache_dir to return our test repository
            from huggingface_hub.utils import CachedRepoInfo, CachedRevisionInfo

            mock_file_info = Mock()
            mock_file_info.file_name = "ops.yaml"
            mock_file_info.file_path = ops_yaml

            mock_revision = Mock(spec=CachedRevisionInfo)
            mock_revision.files = [mock_file_info]

            mock_repo = Mock(spec=CachedRepoInfo)
            mock_repo.repo_type = "model"
            mock_repo.revisions = [mock_revision]
            mock_repo.refs = {"main": mock_revision}

            mock_cache_info = Mock()
            mock_cache_info.repos = [mock_repo]
            mock_scan.return_value = mock_cache_info

            # Create dispatcher with hub ops enabled
            dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)
            dispatcher.load_definitions()

            # Test dynamic loading of namespaced operation
            op_name = "testuser.test_repo.test_dynamic_function"

            # The operation should be available in the dispatcher
            assert op_name in dispatcher._op_definitions

            op = dispatcher.get_op(op_name)

            # Verify the operation was dynamically loaded
            assert op.name == op_name
            assert callable(op)

            # Test execution of dynamically loaded operation
            from interpretune.analysis.ops.base import AnalysisBatch
            from transformers import BatchEncoding

            module_mock = Mock()
            batch = BatchEncoding({"input_ids": [[1, 2, 3]]})

            result = op(module_mock, None, batch, 0)
            assert isinstance(result, AnalysisBatch)
            assert result["test_result"] == "dynamic_success"

    def test_dynamic_loading_with_caching(self):
        """Test that dynamic loading works with dispatcher caching."""
        # Set up mock hub repository
        repo_cache = self.cache_dir / "models--testuser--test_repo"
        snapshot_dir = repo_cache / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)

        test_module = snapshot_dir / "__init__.py"
        test_module.write_text('''
def cached_dynamic_function(module, analysis_batch, batch, batch_idx):
    from interpretune.analysis.ops.base import AnalysisBatch
    if analysis_batch is None:
        analysis_batch = AnalysisBatch()
    analysis_batch["cached_result"] = "cached_success"
    return analysis_batch
''')

        # Create a YAML file to define the operation
        ops_yaml = snapshot_dir / "ops.yaml"
        ops_yaml.write_text('''
cached_dynamic_function:
  description: A cached dynamic function
  implementation: __init__.cached_dynamic_function
  input_schema: {}
  output_schema: {}
''')

        # Mock all hub and dynamic module interactions to avoid HTTP calls
        with patch('interpretune.analysis.ops.hub_manager.snapshot_download', return_value=str(snapshot_dir)), \
             patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', self.cache_dir), \
             patch('interpretune.analysis.ops.dynamic_module_utils.get_cached_module_file_it',
                   return_value=str(test_module)), \
             patch('interpretune.analysis.ops.compiler.cache_manager.scan_cache_dir') as mock_scan:

            # Mock scan_cache_dir to return our test repository
            from huggingface_hub.utils import CachedRepoInfo, CachedRevisionInfo

            mock_file_info = Mock()
            mock_file_info.file_name = "ops.yaml"
            mock_file_info.file_path = ops_yaml

            mock_revision = Mock(spec=CachedRevisionInfo)
            mock_revision.files = [mock_file_info]

            mock_repo = Mock(spec=CachedRepoInfo)
            mock_repo.repo_type = "model"
            mock_repo.revisions = [mock_revision]
            mock_repo.refs = {"main": mock_revision}

            mock_cache_info = Mock()
            mock_cache_info.repos = [mock_repo]
            mock_scan.return_value = mock_cache_info

            dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)
            dispatcher.load_definitions()

            # Get operation multiple times to test caching
            op_name = "testuser.test_repo.cached_dynamic_function"
            op_first = dispatcher.get_op(op_name)
            op_second = dispatcher.get_op(op_name)

            # Assert that the operation is callable and the same object (cached)
            assert callable(op_first)
            assert op_first is op_second
