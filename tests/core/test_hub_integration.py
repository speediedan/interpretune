"""Integration tests for HuggingFace Hub analysis operations."""
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import os

from tests.warns import unmatched_warns
from interpretune.analysis.ops.hub_manager import HubAnalysisOpManager
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

    def test_end_to_end_hub_download_and_discovery(self):
        """Test complete workflow: download from hub, discover, and load operations."""

        # Mock HfApi for download
        mock_hf_api = Mock()
        mock_snapshot_download = Mock(return_value=str(self.snapshot_dir))

        with patch('interpretune.analysis.ops.hub_manager.HfApi', return_value=mock_hf_api), \
             patch('interpretune.analysis.ops.hub_manager.snapshot_download', mock_snapshot_download), \
             patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', self.hub_cache), \
             patch('interpretune.analysis.ops.dispatcher.IT_ANALYSIS_CACHE', self.cache_dir), \
             patch('interpretune.analysis.ops.dispatcher.IT_ANALYSIS_OP_PATHS', [str(self.custom_ops_dir)]):

            # Step 1: Download operation from hub
            hub_manager = HubAnalysisOpManager()
            result = hub_manager.download_operation("testuser/test")

            assert result is not None
            assert result.exists()
            mock_snapshot_download.assert_called_once()

            # Step 2: Create dispatcher and load all operations
            dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)
            dispatcher.load_definitions()

            # Should have discovered both hub and custom ops
            all_ops = dispatcher.list_operations()

            # Hub operations should be namespaced
            assert "testuser.test.test_hub_op" in all_ops
            assert "testuser.test.another_op" in all_ops

            # Custom operations should not be namespaced
            assert "custom_op" in all_ops

            # # Test operation info
            # info = dispatcher.get_operation_info()
            # hub_op_info = info["it.testuser.test.test_hub_op"]
            # assert hub_op_info["is_hub_operation"] is True
            # assert hub_op_info["namespace"] == "it.testuser.test"
            # assert hub_op_info["description"] == "A test operation from hub"

            # custom_op_info = info["it.user.custom_op"]
            # assert custom_op_info["is_hub_operation"] is True  # From custom path, treated as hub
            # assert custom_op_info["namespace"] == "it.user"

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

            #with pytest.warns(UserWarning, match="The fully-qualified name will need"):
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

    def test_upload_and_download_roundtrip(self):
        """Test uploading an operation and then downloading it."""

        # Create a local operation definition file
        local_ops_file = self.temp_dir / "upload_test.yaml"
        local_ops_file.write_text("""
upload_test_op:
  description: Operation for upload testing
  implementation: upload_test.test_function
  input_schema:
    input_data:
      datasets_dtype: string
  output_schema:
    output_data:
      datasets_dtype: string
""")

        mock_hf_api = Mock()
        mock_upload_file = Mock()
        mock_snapshot_download = Mock(return_value=str(self.snapshot_dir))

        with patch('interpretune.analysis.ops.hub_manager.HfApi', return_value=mock_hf_api), \
             patch('interpretune.analysis.ops.hub_manager.upload_file', mock_upload_file), \
             patch('interpretune.analysis.ops.hub_manager.snapshot_download', mock_snapshot_download):

            hub_manager = HubAnalysisOpManager()

            # Upload operation
            hub_manager.upload_operation(
                local_ops_file,
                "testuser/upload-test",
                commit_message="Test upload"
            )

            # Verify upload was called correctly
            mock_upload_file.assert_called_once()
            upload_args = mock_upload_file.call_args
            assert upload_args[1]['repo_id'] == "testuser/upload-test"
            assert upload_args[1]['repo_type'] == "model"
            assert upload_args[1]['commit_message'] == "Test upload"

            # Download operation
            downloaded_path = hub_manager.download_operation("testuser/upload-test")

            # Verify download was called correctly
            assert downloaded_path is not None
            mock_snapshot_download.assert_called_once()
            download_args = mock_snapshot_download.call_args
            assert download_args[1]['repo_id'] == "testuser/upload-test"
            assert download_args[1]['repo_type'] == "model"

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
