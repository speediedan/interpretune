"""Tests for dynamic module utilities."""
import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import shutil

from interpretune.analysis.ops.dynamic_module_utils import (
    get_cached_module_file_it,
    get_function_from_dynamic_module,
    ensure_op_paths_in_syspath,
    remove_op_paths_from_syspath,
    cleanup_op_paths,
    get_added_op_paths,
)


class TestGetCachedModuleFileIt:
    """Test suite for get_cached_module_file_it function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_module_content = "def test_function():\n    return 'test'"

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('interpretune.analysis.ops.dynamic_module_utils.filecmp.cmp')
    def test_deprecated_use_auth_token_warning(self, mock_filecmp):
        """Test that use_auth_token parameter raises deprecation warning."""
        mock_filecmp.return_value = True

        with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.isdir', return_value=True):
            with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.exists', return_value=True):
                with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.basename', return_value='test'):
                    with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.join',
                               return_value='/test/path'):
                        with patch('interpretune.analysis.ops.dynamic_module_utils.check_imports', return_value=[]):
                            with patch('interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it'):
                                with patch('interpretune.analysis.ops.dynamic_module_utils.Path') as mock_path:
                                    mock_submodule_path = Mock()
                                    mock_file_path = Mock()
                                    mock_file_path.exists.return_value = True
                                    mock_submodule_path.__truediv__ = Mock(return_value=mock_file_path)
                                    mock_path.return_value.__truediv__.return_value.__truediv__.return_value = \
                                        mock_submodule_path

                                    with pytest.warns(FutureWarning, match="use_auth_token.*deprecated"):
                                        get_cached_module_file_it(
                                            "/test/path",
                                            "test_module.py",
                                            use_auth_token="test_token"
                                        )

    def test_use_auth_token_and_token_conflict(self):
        """Test that specifying both use_auth_token and token raises ValueError."""
        with pytest.raises(ValueError, match="token.*and.*use_auth_token.*both specified"):
            get_cached_module_file_it(
                "/test/path",
                "test_module.py",
                token="new_token",
                use_auth_token="old_token"
            )

    @patch('interpretune.analysis.ops.dynamic_module_utils.filecmp.cmp')
    def test_use_auth_token_sets_token(self, mock_filecmp):
        """Test that use_auth_token value is transferred to token parameter."""
        mock_filecmp.return_value = True

        with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.isdir', return_value=True):
            with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.exists', return_value=True):
                with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.basename', return_value='test'):
                    with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.join',
                               return_value='/test/path'):
                        with patch('interpretune.analysis.ops.dynamic_module_utils.check_imports', return_value=[]):
                            with patch('interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it'):
                                with patch('interpretune.analysis.ops.dynamic_module_utils.Path') as mock_path:
                                    mock_submodule_path = Mock()
                                    mock_file_path = Mock()
                                    mock_file_path.exists.return_value = True
                                    mock_submodule_path.__truediv__ = Mock(return_value=mock_file_path)
                                    mock_path.return_value.__truediv__.return_value.__truediv__.return_value = \
                                        mock_submodule_path

                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        result = get_cached_module_file_it(
                                            "/test/path",
                                            "test_module.py",
                                            use_auth_token="test_token"
                                        )
                                        # Should complete without error, indicating token was set
                                        assert result is not None

    @patch('interpretune.analysis.ops.dynamic_module_utils.filecmp.cmp')
    @patch('interpretune.analysis.ops.dynamic_module_utils.is_offline_mode')
    @patch('interpretune.analysis.ops.dynamic_module_utils.rank_zero_debug')
    def test_offline_mode_forces_local_files_only(self, mock_debug, mock_offline, mock_filecmp):
        """Test that offline mode forces local_files_only=True."""
        mock_offline.return_value = True
        mock_filecmp.return_value = True

        with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.isdir', return_value=True):
            with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.exists', return_value=True):
                with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.basename', return_value='test'):
                    with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.join',
                               return_value='/test/path'):
                        with patch('interpretune.analysis.ops.dynamic_module_utils.check_imports', return_value=[]):
                            with patch('interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it'):
                                with patch('interpretune.analysis.ops.dynamic_module_utils.Path') as mock_path:
                                    mock_submodule_path = Mock()
                                    mock_file_path = Mock()
                                    mock_file_path.exists.return_value = True
                                    mock_submodule_path.__truediv__ = Mock(return_value=mock_file_path)
                                    mock_path.return_value.__truediv__.return_value.__truediv__.return_value = \
                                        mock_submodule_path

                                    get_cached_module_file_it(
                                        "/test/path",
                                        "test_module.py",
                                        local_files_only=False
                                    )

                                    mock_debug.assert_called_with("Offline mode: forcing local_files_only=True")


    @patch('interpretune.analysis.ops.dynamic_module_utils.filecmp.cmp')
    def test_local_directory_processing(self, mock_filecmp):
        """Test local directory processing with existing file."""
        mock_filecmp.return_value = True

        with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.isdir', return_value=True):
            with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.exists', return_value=True):
                with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.basename', return_value='test_repo'):
                    with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.join',
                               return_value='/test/path/module.py'):
                        with patch('interpretune.analysis.ops.dynamic_module_utils.check_imports', return_value=[]):
                            with patch('interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it'):
                                with patch('interpretune.analysis.ops.dynamic_module_utils.Path') as mock_path:
                                    mock_submodule_path = Mock()
                                    mock_file_path = Mock()
                                    mock_file_path.exists.return_value = True
                                    mock_submodule_path.__truediv__ = Mock(return_value=mock_file_path)
                                    mock_path.return_value.__truediv__.return_value.__truediv__.return_value = \
                                        mock_submodule_path

                                    result = get_cached_module_file_it(
                                        "/test/path",
                                        "module.py"
                                    )

                                    assert result is not None

    @patch('interpretune.analysis.ops.dynamic_module_utils.filecmp.cmp')
    @patch('interpretune.analysis.ops.dynamic_module_utils.shutil.copy')
    @patch('interpretune.analysis.ops.dynamic_module_utils.importlib.invalidate_caches')
    def test_local_file_copying_when_different(self, mock_invalidate, mock_copy, mock_filecmp):
        """Test file copying logic when local files are different."""
        mock_filecmp.return_value = False  # Files are different

        with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.isdir', return_value=True):
            with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.exists', return_value=True):
                with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.basename', return_value='test_repo'):
                    with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.join',
                               return_value='/test/path/module.py'):
                        with patch('interpretune.analysis.ops.dynamic_module_utils.check_imports',
                                   return_value=['helper']):
                            with patch('interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it'):
                                with patch('interpretune.analysis.ops.dynamic_module_utils.Path') as mock_path:
                                    mock_submodule_path = Mock()
                                    mock_file_path = Mock()
                                    mock_file_path.exists.return_value = False
                                    mock_submodule_path.__truediv__ = Mock(return_value=mock_file_path)
                                    mock_path.return_value.__truediv__.return_value.__truediv__.return_value = \
                                        mock_submodule_path

                                    get_cached_module_file_it(
                                        "/test/path",
                                        "module.py"
                                    )

                                    # Should copy main file and helper module
                                    assert mock_copy.call_count >= 2
                                    assert mock_invalidate.call_count >= 2

    @patch('interpretune.analysis.ops.dynamic_module_utils.filecmp.cmp')
    @patch('interpretune.analysis.ops.dynamic_module_utils.extract_commit_hash')
    def test_commit_hash_local_fallback(self, mock_extract_commit, mock_filecmp):
        """Test that local repos get 'local' as commit hash."""
        mock_extract_commit.return_value = "abc123"
        mock_filecmp.return_value = True

        with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.isdir', return_value=True):
            with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.exists', return_value=True):
                with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.basename',
                           return_value='different_name'):
                    with patch('interpretune.analysis.ops.dynamic_module_utils.os.path.join',
                               return_value='/test/path/module.py'):
                        with patch('interpretune.analysis.ops.dynamic_module_utils.check_imports', return_value=[]):
                            with patch('interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it'):
                                with patch('interpretune.analysis.ops.dynamic_module_utils.Path') as mock_path:
                                    # Create a proper mock hierarchy for Path operations
                                    mock_submodule_path = Mock()
                                    mock_commit_path = Mock()
                                    mock_file_path = Mock()
                                    mock_file_path.exists.return_value = True
                                    mock_commit_path.__truediv__ = Mock(return_value=mock_file_path)
                                    mock_submodule_path.__truediv__ = Mock(return_value=mock_commit_path)
                                    mock_path.return_value.__truediv__.return_value.__truediv__.return_value = \
                                        mock_submodule_path

                                    get_cached_module_file_it(
                                        "/test/path",
                                        "module.py"
                                    )

                                    # For local repos, should not call extract_commit_hash since basename matches
                                    # But this test uses different_name to trigger the versioning path
                                    mock_extract_commit.assert_not_called()

    def test_ensure_op_paths_in_syspath(self):
        """Test adding operation paths to sys.path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_paths = [temp_dir]

            # Clean up any existing paths first
            cleanup_op_paths()

            ensure_op_paths_in_syspath(test_paths)

            added_paths = get_added_op_paths()
            assert str(Path(temp_dir).resolve()) in added_paths

            # Cleanup
            remove_op_paths_from_syspath(test_paths)

    def test_remove_op_paths_from_syspath(self):
        """Test removing operation paths from sys.path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_paths = [temp_dir]

            # Add then remove
            ensure_op_paths_in_syspath(test_paths)
            remove_op_paths_from_syspath(test_paths)

            added_paths = get_added_op_paths()
            assert str(Path(temp_dir).resolve()) not in added_paths

    def test_cleanup_op_paths(self):
        """Test cleaning up all added operation paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_paths = [temp_dir]

            ensure_op_paths_in_syspath(test_paths)
            cleanup_op_paths()

            added_paths = get_added_op_paths()
            assert len(added_paths) == 0

    @patch('interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it')
    @patch('interpretune.analysis.ops.dynamic_module_utils.get_cached_module_file_it')
    def test_get_function_from_dynamic_module_repo_parsing(self, mock_get_cached, mock_create):
        """Test function reference parsing with repo specification."""
        mock_get_cached.return_value = "/test/module.py"

        with patch('interpretune.analysis.ops.dynamic_module_utils.get_function_in_module') as mock_get_func:
            mock_get_func.return_value = lambda: "test"

            # Test with repo specification in function reference
            _ = get_function_from_dynamic_module(
                "test/repo--module.function_name",
                "default/repo"
            )

            # Should parse repo from function reference
            mock_get_cached.assert_called_once()
            args, kwargs = mock_get_cached.call_args
            assert args[0] == "test/repo"  # Should use parsed repo
            assert args[1] == "module.py"

    def test_get_function_from_dynamic_module_missing_module_error(self):
        """Test error handling when module file is not found."""
        with patch('interpretune.analysis.ops.dynamic_module_utils.get_cached_module_file_it', return_value=None):
            with pytest.raises(OSError, match="Could not locate the module file"):
                get_function_from_dynamic_module(
                    "module.function_name",
                    "/test/repo"
                )
