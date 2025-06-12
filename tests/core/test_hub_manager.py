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


class TestDynamicModuleUtils:
    """Test cases for dynamic module utilities."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('interpretune.analysis.ops.dynamic_module_utils.get_cached_module_file_it')
    @patch('interpretune.analysis.ops.dynamic_module_utils.get_function_in_module')
    def test_get_function_from_dynamic_module_success(self, mock_get_function, mock_get_file):
        """Test successful dynamic function loading."""
        from interpretune.analysis.ops.dynamic_module_utils import get_function_from_dynamic_module

        # Mock the dependencies
        mock_get_file.return_value = "fake_module_file.py"
        mock_function = Mock()
        mock_get_function.return_value = mock_function

        # Test the function
        result = get_function_from_dynamic_module(
            function_reference="__init__.test_function",
            op_repo_name_or_path="test_user.test_repo"
        )

        # Verify calls
        mock_get_file.assert_called_once_with(
            "test_user.test_repo",
            "__init__.py",
            cache_dir=None,
            force_download=False,
            resume_download=None,
            proxies=None,
            token=None,
            revision=None,
            local_files_only=False,
            repo_type=None,
        )
        mock_get_function.assert_called_once_with("test_function", "fake_module_file.py", force_reload=False)

        assert result == mock_function

    @patch('interpretune.analysis.ops.dynamic_module_utils.get_cached_module_file_it')
    def test_get_function_from_dynamic_module_file_not_found(self, mock_get_file):
        """Test dynamic function loading when module file is not found."""
        from interpretune.analysis.ops.dynamic_module_utils import get_function_from_dynamic_module

        # Mock file not found
        mock_get_file.return_value = None

        # Test should raise a OSError when None is passed to get_function_in_module
        with pytest.raises(OSError):
            get_function_from_dynamic_module(
                function_reference="__init__.test_function",
                op_repo_name_or_path="test_user.test_repo"
            )

    def test_init_it_modules(self):
        """Test initialization of interpretune modules cache."""
        from interpretune.analysis.ops.dynamic_module_utils import init_it_modules

        with patch('interpretune.analysis.ops.dynamic_module_utils.IT_MODULES_CACHE', str(self.temp_dir)), \
             patch('interpretune.analysis.ops.dynamic_module_utils.IT_DYNAMIC_MODULE_NAME', "interpretune_modules"):
            init_it_modules()
            # The function creates the cache dir but not the dynamic modules subdir
            assert self.temp_dir.exists()
            init_file = self.temp_dir / "__init__.py"
            assert init_file.exists()

    def test_create_dynamic_module_it(self):
        """Test creation of dynamic module directories."""
        from interpretune.analysis.ops.dynamic_module_utils import create_dynamic_module_it

        with patch('interpretune.analysis.ops.dynamic_module_utils.IT_MODULES_CACHE', str(self.temp_dir)), \
             patch('interpretune.analysis.ops.dynamic_module_utils.IT_DYNAMIC_MODULE_NAME', "interpretune_modules"):
            # Create test module file
            test_file = self.temp_dir / "test_module.py"
            test_file.write_text("def test_func(): pass")

            # The function takes only name parameter (no module_files)
            create_dynamic_module_it(name="test_module")

            # Check that module directory was created in the cache directory
            module_dir = self.temp_dir / "test_module"
            assert module_dir.exists()

    def test_get_function_in_module(self):
        """Test getting function from module."""
        from interpretune.analysis.ops.dynamic_module_utils import get_function_in_module

        with patch('interpretune.analysis.ops.dynamic_module_utils.IT_MODULES_CACHE', str(self.temp_dir)):
            # Create a test module
            module_dir = self.temp_dir / "test_module"
            module_dir.mkdir(parents=True)
            module_file = module_dir / "test_func.py"
            module_file.write_text("def my_function():\n    return 'test_result'")

            # Get the function
            func = get_function_in_module("my_function", "test_module/test_func.py")
            assert callable(func)
            assert func() == "test_result"

    def test_get_cached_module_file_it_with_hub_manager(self):
        """Test getting cached module file with hub manager."""
        from interpretune.analysis.ops.dynamic_module_utils import get_cached_module_file_it

        # Create a test file in temp directory
        test_file = self.temp_dir / "__init__.py"
        test_file.write_text("# test init file")

        # Mock all the HuggingFace Hub API functions to prevent external calls
        with patch('interpretune.analysis.ops.dynamic_module_utils.cached_file') as mock_cached_file, \
             patch('interpretune.analysis.ops.dynamic_module_utils.try_to_load_from_cache') as mock_try_load, \
             patch('interpretune.analysis.ops.dynamic_module_utils.check_imports') as mock_check_imports, \
             patch('interpretune.analysis.ops.dynamic_module_utils.extract_commit_hash') as mock_extract_hash, \
             patch('interpretune.analysis.ops.dynamic_module_utils.HubAnalysisOpManager') as mock_manager_class:

            # Mock the cached_file to return our test file path
            mock_cached_file.return_value = str(test_file)

            # Mock other hub functions
            mock_try_load.return_value = None
            mock_check_imports.return_value = []
            mock_extract_hash.return_value = "test_commit_hash"

            # Mock the HubAnalysisOpManager
            mock_manager = Mock()
            mock_collection = Mock()
            mock_collection.local_path = self.temp_dir
            mock_manager.download_ops.return_value = mock_collection
            mock_manager_class.return_value = mock_manager
            mock_manager.cache_dir = self.temp_dir

            result = get_cached_module_file_it(
                op_repo_name_or_path="test_user.test_repo",
                module_file="__init__.py"
            )

            assert result is not None
            # Verify that cached_file was called with the correct repo_id (with slash)
            mock_cached_file.assert_called_once()
            call_args = mock_cached_file.call_args
            assert call_args[0][0] == "test_user/test_repo"  # repo_id with slash
            assert call_args[0][1] == "__init__.py"  # module_file

    def test_get_cached_module_file_it_file_not_exists(self):
        """Test getting cached module file when file doesn't exist."""
        from interpretune.analysis.ops.dynamic_module_utils import get_cached_module_file_it

        # Mock all the HuggingFace Hub API functions to raise OSError (simulating file not found)
        with patch('interpretune.analysis.ops.dynamic_module_utils.cached_file') as mock_cached_file, \
             patch('interpretune.analysis.ops.dynamic_module_utils.try_to_load_from_cache') as mock_try_load, \
             patch('interpretune.analysis.ops.dynamic_module_utils.HubAnalysisOpManager') as mock_manager_class:

            # Mock cached_file to raise OSError
            mock_cached_file.side_effect = OSError("File not found")
            mock_try_load.return_value = None

            # Mock the HubAnalysisOpManager
            mock_manager = Mock()
            mock_collection = Mock()
            mock_collection.local_path = self.temp_dir  # No nonexistent.py file created
            mock_manager.download_ops.return_value = mock_collection
            mock_manager_class.return_value = mock_manager
            # Mock the cache_dir attribute to return a string path
            mock_manager.cache_dir = self.temp_dir

            with pytest.raises(OSError):
                get_cached_module_file_it(
                    op_repo_name_or_path="test_user.test_repo",
                    module_file="nonexistent.py"
                )
