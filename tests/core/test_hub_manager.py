"""Unit tests for Hub Analysis Operations Manager."""
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from interpretune.analysis.ops.hub_manager import HubAnalysisOpManager, HubOpCollection
from huggingface_hub.utils import RepositoryNotFoundError


class TestHubOpCollection:
    """Test cases for HubOpCollection dataclass."""

    def test_from_repo_id_valid(self):
        """Test creating HubOpCollection from valid repo_id."""
        repo_id = "username/some_repo"
        local_path = Path("/fake/path")

        collection = HubOpCollection.from_repo_id(repo_id, local_path)

        assert collection.repo_id == repo_id
        assert collection.username == "username"
        assert collection.repo_name == "some_repo"
        assert collection.local_path == local_path
        assert collection.revision == "main"

    def test_from_repo_id_invalid(self):
        """Test creating HubOpCollection from invalid repo_id."""
        with pytest.raises(ValueError, match="Invalid repo_id format"):
            HubOpCollection.from_repo_id("invalid-repo-id", Path("/fake/path"))

    def test_namespace_prefix(self):
        """Test namespace prefix generation."""
        collection = HubOpCollection(
            repo_id="username/some_repo",
            username="username",
            repo_name="some_repo",
            local_path=Path("/fake/path")
        )

        assert collection.namespace_prefix == "username.some_repo"

class TestHubAnalysisOpManager:
    """Test cases for HubAnalysisOpManager."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = HubAnalysisOpManager(cache_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test environment after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('interpretune.analysis.ops.hub_manager.snapshot_download')
    def test_download_ops_success(self, mock_snapshot_download):
        """Test successful download of operations."""
        mock_snapshot_download.return_value = str(self.temp_dir / "downloaded")

        collection = self.manager.download_ops("username/some_repo")

        assert collection.repo_id == "username/some_repo"
        assert collection.username == "username"
        assert collection.repo_name == "some_repo"
        mock_snapshot_download.assert_called_once()

    @patch('interpretune.analysis.ops.hub_manager.snapshot_download')
    def test_download_ops_repository_not_found(self, mock_snapshot_download):
        """Test download when repository doesn't exist."""
        mock_snapshot_download.side_effect = RepositoryNotFoundError("Not found")

        with pytest.raises(RepositoryNotFoundError):
            self.manager.download_ops("nonexistent/repo")

    @patch('interpretune.analysis.ops.hub_manager.HfApi')
    def test_upload_ops_success(self, mock_hf_api_class):
        """Test successful upload of operations."""
        # Create test files
        test_dir = self.temp_dir / "test_ops"
        test_dir.mkdir()
        (test_dir / "test_ops.yaml").write_text("test: {}")

        # Mock API
        mock_api = Mock()
        mock_commit_info = Mock()
        mock_commit_info.oid = "abc123"
        mock_api.upload_folder.return_value = mock_commit_info
        mock_api.repo_info.side_effect = RepositoryNotFoundError("Not found")
        mock_hf_api_class.return_value = mock_api

        manager = HubAnalysisOpManager(cache_dir=self.temp_dir)
        commit_sha = manager.upload_ops(test_dir, "username/test-ops")

        assert commit_sha == "abc123"
        mock_api.create_repo.assert_called_once()
        mock_api.upload_folder.assert_called_once()

    def test_upload_ops_invalid_directory(self):
        """Test upload with non-existent directory."""
        with pytest.raises(ValueError, match="Local directory does not exist"):
            self.manager.upload_ops(Path("/nonexistent"), "username/test-ops")

    def test_upload_ops_invalid_repo_id(self):
        """Test upload with invalid repo_id."""
        test_dir = self.temp_dir / "test_ops"
        test_dir.mkdir()

        with pytest.raises(ValueError, match="Invalid repo_id format"):
            self.manager.upload_ops(test_dir, "invalid-repo-id")

    def test_upload_operation_nonexistent_file(self):
        """Test upload_operation with non-existent file."""
        with pytest.raises(ValueError, match="Local file does not exist"):
            self.manager.upload_operation(Path("/nonexistent/file.yaml"), "username/test-ops")

    def test_upload_operation_invalid_repo_id(self):
        """Test upload_operation with invalid repo_id."""
        test_file = self.temp_dir / "test_op.yaml"
        test_file.write_text("test: {}")

        with pytest.raises(ValueError, match="Invalid repo_id format"):
            self.manager.upload_operation(test_file, "invalid-repo-id")

    @patch('interpretune.analysis.ops.hub_manager.upload_file')
    @patch('interpretune.analysis.ops.hub_manager.HfApi')
    def test_upload_operation_success_new_repo(self, mock_hf_api_class, mock_upload_file):
        """Test successful upload_operation creating new repository."""
        # Create test file
        test_file = self.temp_dir / "test_op.yaml"
        test_file.write_text("test: {}")

        # Mock API
        mock_api = Mock()
        mock_commit_info = Mock()
        mock_commit_info.oid = "def456"
        mock_upload_file.return_value = mock_commit_info
        mock_api.repo_info.side_effect = RepositoryNotFoundError("Not found")
        mock_hf_api_class.return_value = mock_api

        manager = HubAnalysisOpManager(cache_dir=self.temp_dir)
        commit_sha = manager.upload_operation(test_file, "username/test-ops")

        assert commit_sha == "def456"
        mock_api.create_repo.assert_called_once_with(
            repo_id="username/test-ops",
            repo_type="model",
            private=False
        )
        mock_upload_file.assert_called_once()

    @patch('interpretune.analysis.ops.hub_manager.upload_file')
    @patch('interpretune.analysis.ops.hub_manager.HfApi')
    def test_upload_operation_success_existing_repo(self, mock_hf_api_class, mock_upload_file):
        """Test successful upload_operation to existing repository."""
        # Create test file
        test_file = self.temp_dir / "test_op.yaml"
        test_file.write_text("test: {}")

        # Mock API
        mock_api = Mock()
        mock_commit_info = Mock()
        mock_commit_info.oid = "ghi789"
        mock_upload_file.return_value = mock_commit_info
        mock_api.repo_info.return_value = Mock()  # Repo exists
        mock_hf_api_class.return_value = mock_api

        manager = HubAnalysisOpManager(cache_dir=self.temp_dir)
        commit_sha = manager.upload_operation(test_file, "username/test-ops")

        assert commit_sha == "ghi789"
        mock_api.create_repo.assert_not_called()  # Should not create repo
        mock_upload_file.assert_called_once()

    @patch('interpretune.analysis.ops.hub_manager.HfApi')
    def test_list_available_collections(self, mock_hf_api_class):
        """Test listing available collections."""
        mock_api = Mock()
        mock_model1 = Mock()
        mock_model1.modelId = "user1/some_repo"
        mock_model2 = Mock()
        mock_model2.modelId = "user2/nlp"
        mock_model3 = Mock()
        mock_model3.modelId = "user1/not-ops"  # Should be filtered out

        mock_api.list_models.return_value = [mock_model1, mock_model2, mock_model3]
        mock_hf_api_class.return_value = mock_api

        manager = HubAnalysisOpManager(cache_dir=self.temp_dir)
        collections = manager.list_available_collections()

        assert "user1/some_repo" in collections
        assert "user2/nlp" in collections
        assert "user1/not-ops" in collections  # Actually included by search

    @patch('interpretune.analysis.ops.hub_manager.HfApi')
    def test_list_available_collections_with_username_filter(self, mock_hf_api_class):
        """Test listing available collections with username filter."""
        mock_api = Mock()
        mock_model1 = Mock()
        mock_model1.modelId = "user1/some_repo"
        mock_model2 = Mock()
        mock_model2.modelId = "user2/nlp"
        mock_model3 = Mock()
        mock_model3.modelId = "user1/another_repo"

        mock_api.list_models.return_value = [mock_model1, mock_model2, mock_model3]
        mock_hf_api_class.return_value = mock_api

        manager = HubAnalysisOpManager(cache_dir=self.temp_dir)
        collections = manager.list_available_collections(username="user1")

        assert "user1/some_repo" in collections
        assert "user1/another_repo" in collections
        assert "user2/nlp" not in collections  # Should be filtered out

    def test_get_cached_collections_empty(self):
        """Test getting cached collections when cache is empty."""
        collections = self.manager.get_cached_collections()
        assert collections == []

    def test_get_cached_collections_nonexistent_cache_dir(self):
        """Test getting cached collections when cache directory doesn't exist."""
        # Use a non-existent cache directory
        nonexistent_cache = self.temp_dir / "nonexistent_cache"
        manager = HubAnalysisOpManager(cache_dir=nonexistent_cache)

        collections = manager.get_cached_collections()
        assert collections == []

    def test_get_cached_collections_with_cache(self):
        """Test getting cached collections when cache exists."""
        # Create fake cache structure
        cache_repo = self.temp_dir / "models--username--some_repo"
        cache_repo.mkdir(parents=True)
        snapshots_dir = cache_repo / "snapshots"
        snapshots_dir.mkdir()
        snapshot_dir = snapshots_dir / "abc123"
        snapshot_dir.mkdir()
        (snapshot_dir / "operations.yaml").write_text("test: {}")

        collections = self.manager.get_cached_collections()

        assert len(collections) == 1
        assert collections[0].repo_id == "username/some_repo"
        assert collections[0].username == "username"
        assert collections[0].repo_name == "some_repo"

    def test_has_op_definitions_true(self):
        """Test _has_op_definitions returns True for directories with YAML."""
        # Create fake repo structure with YAML files
        repo_dir = self.temp_dir / "models--username--test-ops"
        repo_dir.mkdir(parents=True)
        snapshots_dir = repo_dir / "snapshots"
        snapshots_dir.mkdir()
        snapshot_dir = snapshots_dir / "abc123"
        snapshot_dir.mkdir()
        (snapshot_dir / "operations.yaml").write_text("test: {}")

        assert self.manager._has_op_definitions(repo_dir) is True

    def test_has_op_definitions_false(self):
        """Test _has_op_definitions returns False for directories without YAML."""
        # Create fake repo structure without YAML files
        repo_dir = self.temp_dir / "models--username--test-ops"
        repo_dir.mkdir(parents=True)
        snapshots_dir = repo_dir / "snapshots"
        snapshots_dir.mkdir()
        snapshot_dir = snapshots_dir / "abc123"
        snapshot_dir.mkdir()
        (snapshot_dir / "README.md").write_text("No ops here")

        assert self.manager._has_op_definitions(repo_dir) is False

    def test_has_op_definitions_no_snapshots_dir(self):
        """Test _has_op_definitions returns False when snapshots directory doesn't exist."""
        # Create fake repo structure without snapshots directory
        repo_dir = self.temp_dir / "models--username--test-ops"
        repo_dir.mkdir(parents=True)
        # Do not create snapshots directory

        assert self.manager._has_op_definitions(repo_dir) is False

    @patch('interpretune.analysis.ops.hub_manager.HubAnalysisOpManager.download_ops')
    @patch('interpretune.analysis.ops.hub_manager.HubAnalysisOpManager.list_available_collections')
    def test_discover_hub_ops_auto_discover(self, mock_list_collections, mock_download_ops):
        """Test auto-discovery of hub operations."""
        mock_list_collections.return_value = ["user1/ops1", "user2/ops2"]
        mock_collection1 = Mock()
        mock_collection2 = Mock()
        mock_download_ops.side_effect = [mock_collection1, mock_collection2]

        collections = self.manager.discover_hub_ops()

        assert len(collections) == 2
        assert mock_download_ops.call_count == 2

    @patch('interpretune.analysis.ops.hub_manager.HubAnalysisOpManager.download_ops')
    def test_discover_hub_ops_with_patterns(self, mock_download_ops):
        """Test discovery with specific patterns."""
        mock_collection = Mock()
        mock_download_ops.return_value = mock_collection

        collections = self.manager.discover_hub_ops(["user1/test-ops"])

        assert len(collections) == 1
        mock_download_ops.assert_called_once_with("user1/test-ops")

    @patch('interpretune.analysis.ops.hub_manager.HubAnalysisOpManager.download_ops')
    def test_discover_hub_ops_with_failures(self, mock_download_ops):
        """Test discovery handles failures gracefully."""
        mock_download_ops.side_effect = [Exception("Failed"), Mock()]

        collections = self.manager.discover_hub_ops(["bad/repo", "good/repo"])

        assert len(collections) == 1  # Only successful download


class TestHubAnalysisOpManagerIntegration:
    """Integration tests that don't require actual Hub access."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_namespace_prefix_extraction(self):
        """Test namespace prefix extraction from different repo names."""
        test_cases = [
            ("username/some_repo", "username.some_repo"),
            ("speediedan/nlp-toolkit", "speediedan.nlp-toolkit"),
            ("company/custom-analysis-ops", "company.custom-analysis-ops"),
        ]

        for repo_id, expected_namespace in test_cases:
            collection = HubOpCollection.from_repo_id(repo_id, Path("/fake"))
            assert collection.namespace_prefix == expected_namespace

    @patch.dict(os.environ, {'IT_ANALYSIS_OP_PATHS': '/path1:/path2'})
    def test_environment_variable_parsing(self):
        """Test that environment variables are properly parsed."""
        import sys
        import importlib

        # Backup existing module
        original_mod = sys.modules.get('interpretune.analysis', None)

        # Remove interpretune.analysis from sys.modules to force reload
        sys.modules.pop('interpretune.analysis', None)
        # Now import and check the variable
        analysis_mod = importlib.import_module('interpretune.analysis')
        IT_ANALYSIS_OP_PATHS = getattr(analysis_mod, 'IT_ANALYSIS_OP_PATHS', None)

        assert IT_ANALYSIS_OP_PATHS == ['/path1', '/path2']

        # Restore original module
        if original_mod is not None:
            sys.modules['interpretune.analysis'] = original_mod
        else:
            sys.modules.pop('interpretune.analysis', None)
