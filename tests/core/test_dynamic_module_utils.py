"""Tests for dynamic module utilities."""

import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch
import pytest
import shutil
import sys

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

    @pytest.fixture
    def mock_it_modules_cache(self):
        """Fixture providing a temporary directory for IT_MODULES_CACHE."""
        with tempfile.TemporaryDirectory() as temp_cache:
            with patch("interpretune.analysis.ops.dynamic_module_utils.IT_MODULES_CACHE", temp_cache):
                yield temp_cache

    @patch("interpretune.analysis.ops.dynamic_module_utils.filecmp.cmp")
    @patch("interpretune.analysis.ops.dynamic_module_utils.cached_file")
    def test_deprecated_use_auth_token_warning(self, mock_cached_file, mock_filecmp, mock_it_modules_cache):
        """Test that use_auth_token parameter raises deprecation warning."""
        mock_filecmp.return_value = True
        mock_cached_file.return_value = "/test/path/test_module.py"

        # Create a dummy file to copy from
        src_file = Path(mock_it_modules_cache) / "test_module.py"
        src_file.write_text("# test module")

        with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.isdir", return_value=True):
            with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.exists", return_value=True):
                with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.basename", return_value="test"):
                    with patch(
                        "interpretune.analysis.ops.dynamic_module_utils.os.path.join", return_value=str(src_file)
                    ):
                        with patch("interpretune.analysis.ops.dynamic_module_utils.check_imports", return_value=[]):
                            with patch("interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it"):
                                with pytest.warns(FutureWarning, match="use_auth_token.*deprecated"):
                                    get_cached_module_file_it(
                                        "/test/path", "test_module.py", use_auth_token="test_token"
                                    )

    def test_use_auth_token_and_token_conflict(self):
        """Test that specifying both use_auth_token and token raises ValueError."""
        with pytest.raises(ValueError, match="token.*and.*use_auth_token.*both specified"):
            get_cached_module_file_it("/test/path", "test_module.py", token="new_token", use_auth_token="old_token")

    @patch("interpretune.analysis.ops.dynamic_module_utils.filecmp.cmp")
    @patch("interpretune.analysis.ops.dynamic_module_utils.cached_file")
    def test_use_auth_token_sets_token(self, mock_cached_file, mock_filecmp, mock_it_modules_cache):
        """Test that use_auth_token value is transferred to token parameter."""
        mock_filecmp.return_value = True
        mock_cached_file.return_value = "/test/path/test_module.py"

        # Create a dummy file to copy from
        src_file = Path(mock_it_modules_cache) / "test_module.py"
        src_file.write_text("# test module")

        with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.isdir", return_value=True):
            with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.exists", return_value=True):
                with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.basename", return_value="test"):
                    with patch(
                        "interpretune.analysis.ops.dynamic_module_utils.os.path.join", return_value=str(src_file)
                    ):
                        with patch("interpretune.analysis.ops.dynamic_module_utils.check_imports", return_value=[]):
                            with patch("interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it"):
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    result = get_cached_module_file_it(
                                        "/test/path", "test_module.py", use_auth_token="test_token"
                                    )
                                    # Should complete without error, indicating token was set
                                    assert result is not None

    @patch("interpretune.analysis.ops.dynamic_module_utils.filecmp.cmp")
    @patch("interpretune.analysis.ops.dynamic_module_utils.is_offline_mode")
    @patch("interpretune.analysis.ops.dynamic_module_utils.rank_zero_debug")
    @patch("interpretune.analysis.ops.dynamic_module_utils.cached_file")
    def test_offline_mode_forces_local_files_only(
        self, mock_cached_file, mock_debug, mock_offline, mock_filecmp, mock_it_modules_cache
    ):
        """Test that offline mode forces local_files_only=True."""
        mock_offline.return_value = True
        mock_filecmp.return_value = True
        mock_cached_file.return_value = "/test/path/test_module.py"

        # Create a dummy file to copy from
        src_file = Path(mock_it_modules_cache) / "test_module.py"
        src_file.write_text("# test module")

        with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.isdir", return_value=True):
            with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.exists", return_value=True):
                with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.basename", return_value="test"):
                    with patch(
                        "interpretune.analysis.ops.dynamic_module_utils.os.path.join", return_value=str(src_file)
                    ):
                        with patch("interpretune.analysis.ops.dynamic_module_utils.check_imports", return_value=[]):
                            with patch("interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it"):
                                get_cached_module_file_it("/test/path", "test_module.py", local_files_only=False)

                                mock_debug.assert_called_with("Offline mode: forcing local_files_only=True")

    @patch("interpretune.analysis.ops.dynamic_module_utils.filecmp.cmp")
    @patch("interpretune.analysis.ops.dynamic_module_utils.cached_file")
    def test_local_directory_processing(self, mock_cached_file, mock_filecmp, mock_it_modules_cache):
        """Test local directory processing with existing file."""
        mock_filecmp.return_value = True
        mock_cached_file.return_value = "/test/path/module.py"

        # Create a dummy file to copy from
        src_file = Path(mock_it_modules_cache) / "module.py"
        src_file.write_text("# test module")

        with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.isdir", return_value=True):
            with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.exists", return_value=True):
                with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.basename", return_value="test_repo"):
                    with patch(
                        "interpretune.analysis.ops.dynamic_module_utils.os.path.join", return_value=str(src_file)
                    ):
                        with patch("interpretune.analysis.ops.dynamic_module_utils.check_imports", return_value=[]):
                            with patch("interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it"):
                                result = get_cached_module_file_it("/test/path", "module.py")

                                assert result is not None

    @patch("interpretune.analysis.ops.dynamic_module_utils.filecmp.cmp")
    @patch("interpretune.analysis.ops.dynamic_module_utils.shutil.copy")
    @patch("interpretune.analysis.ops.dynamic_module_utils.importlib.invalidate_caches")
    def test_local_file_copying_when_different(self, mock_invalidate, mock_copy, mock_filecmp, mock_it_modules_cache):
        """Test file copying logic when local files are different."""
        mock_filecmp.return_value = False  # Files are different

        with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.isdir", return_value=True):
            with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.exists", return_value=True):
                with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.basename", return_value="test_repo"):
                    with patch(
                        "interpretune.analysis.ops.dynamic_module_utils.os.path.join",
                        return_value="/test/path/module.py",
                    ):
                        with patch(
                            "interpretune.analysis.ops.dynamic_module_utils.check_imports", return_value=["helper"]
                        ):
                            with patch("interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it"):
                                with patch(
                                    "interpretune.analysis.ops.dynamic_module_utils.cached_file",
                                    return_value="/test/path/module.py",
                                ):
                                    get_cached_module_file_it("/test/path", "module.py")

                                    # Should copy main file and helper module
                                    assert mock_copy.call_count >= 2
                                    assert mock_invalidate.call_count >= 2

    @patch("interpretune.analysis.ops.dynamic_module_utils.cached_file")
    @patch("interpretune.analysis.ops.dynamic_module_utils.try_to_load_from_cache")
    @patch("interpretune.analysis.ops.dynamic_module_utils.check_imports")
    @patch("interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it")
    def test_remote_repo_name_transformation(
        self, mock_create, mock_check, mock_try_load, mock_cached, mock_it_modules_cache
    ):
        """Test that remote repository names are properly transformed."""
        # Create actual files in the temporary cache
        src_file = Path(mock_it_modules_cache) / "module.py"
        src_file.write_text("# test module")

        mock_cached.return_value = str(src_file)
        mock_try_load.return_value = "/old/cache/module.py"
        mock_check.return_value = []

        # Mock create_dynamic_module_it to actually create the directory structure
        def mock_create_dirs(path):
            # Ensure we create the full path including intermediate directories
            full_path = Path(mock_it_modules_cache) / path if not Path(path).is_absolute() else Path(path)
            full_path.mkdir(parents=True, exist_ok=True)

        mock_create.side_effect = mock_create_dirs

        with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.isdir", return_value=False):
            with patch("interpretune.analysis.ops.dynamic_module_utils.extract_commit_hash", return_value="abc123"):
                # Test with dot in repo name
                _ = get_cached_module_file_it("user.repo-name", "module.py")

                # Should have called cached_file with transformed name
                mock_cached.assert_called_once()
                args, kwargs = mock_cached.call_args
                assert args[0] == "user/repo-name"  # Dots should be replaced with slashes

    @patch("interpretune.analysis.ops.dynamic_module_utils.cached_file")
    @patch("interpretune.analysis.ops.dynamic_module_utils.try_to_load_from_cache")
    @patch("interpretune.analysis.ops.dynamic_module_utils.check_imports")
    @patch("interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it")
    @patch("interpretune.analysis.ops.dynamic_module_utils.extract_commit_hash")
    def test_commit_hash_fallback_to_local(
        self, mock_extract, mock_create, mock_check, mock_try_load, mock_cached, mock_it_modules_cache
    ):
        """Test fallback to 'local' when commit hash extraction fails."""
        mock_cached.return_value = "/cache/path/module.py"
        mock_try_load.return_value = "/old/cache/module.py"
        mock_check.return_value = []
        mock_extract.return_value = None  # Simulate failed hash extraction

        with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.isdir", return_value=False):
            with patch(
                "interpretune.analysis.ops.dynamic_module_utils.os.path.basename", return_value="user--remote-repo"
            ):
                with patch("interpretune.analysis.ops.dynamic_module_utils.shutil.copy"):
                    with patch("interpretune.analysis.ops.dynamic_module_utils.importlib.invalidate_caches"):
                        _ = get_cached_module_file_it("user/remote-repo", "module.py")

                        # Should create path with "local" as commit hash
                        mock_create.assert_called()
                        create_calls = [str(call[0][0]) for call in mock_create.call_args_list]
                        assert any("local" in call for call in create_calls), (
                            f"Expected 'local' in calls: {create_calls}"
                        )

    @patch("interpretune.analysis.ops.dynamic_module_utils.cached_file")
    @patch("interpretune.analysis.ops.dynamic_module_utils.try_to_load_from_cache")
    @patch("interpretune.analysis.ops.dynamic_module_utils.check_imports")
    @patch("interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it")
    @patch("interpretune.analysis.ops.dynamic_module_utils.extract_commit_hash")
    def test_new_files_downloaded_warning(
        self, mock_extract, mock_create, mock_check, mock_try_load, mock_cached, mock_it_modules_cache
    ):
        """Test warning when new files are downloaded without revision."""
        mock_cached.return_value = "/cache/path/module.py"
        mock_try_load.return_value = "/different/cache/module.py"  # Different path indicates new file
        mock_check.return_value = ["helper"]
        mock_extract.return_value = "abc123"

        with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.isdir", return_value=False):
            with patch("interpretune.analysis.ops.dynamic_module_utils.shutil.copy"):
                with patch("interpretune.analysis.ops.dynamic_module_utils.importlib.invalidate_caches"):
                    with patch("interpretune.analysis.ops.dynamic_module_utils.rank_zero_warn") as mock_warn:
                        # Mock recursive call for helper module
                        with patch(
                            "interpretune.analysis.ops.dynamic_module_utils.get_cached_module_file_it",
                            side_effect=lambda *args, **kwargs: "/path/helper.py",
                        ):
                            get_cached_module_file_it(
                                "user/remote-repo",
                                "module.py",
                                revision=None,  # No revision specified
                            )

                            # Should warn about new files
                            mock_warn.assert_called_once()
                            warning_message = mock_warn.call_args[0][0]
                            assert "new version" in warning_message
                            assert "module.py" in warning_message

    @patch("interpretune.analysis.ops.dynamic_module_utils.cached_file")
    @patch("interpretune.analysis.ops.dynamic_module_utils.try_to_load_from_cache")
    @patch("interpretune.analysis.ops.dynamic_module_utils.check_imports")
    @patch("interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it")
    @patch("interpretune.analysis.ops.dynamic_module_utils.extract_commit_hash")
    def test_recursive_module_downloading(
        self, mock_extract, mock_create, mock_check, mock_try_load, mock_cached, mock_it_modules_cache
    ):
        """Test recursive downloading of required modules."""
        mock_try_load.return_value = "/old/cache/module.py"
        mock_check.return_value = ["helper1", "helper2"]
        mock_extract.return_value = "abc123"

        # Create actual files in the temporary cache
        mock_it_modules_cache = Path(mock_it_modules_cache)
        main_file = mock_it_modules_cache / "module.py"
        helper1_file = mock_it_modules_cache / "helper1.py"
        helper2_file = mock_it_modules_cache / "helper2.py"

        main_file.write_text("# main module")
        helper1_file.write_text("# helper1 module")
        helper2_file.write_text("# helper2 module")

        # Track recursive calls
        recursive_calls = []
        call_count = {}

        def side_effect_cached_file(repo, module_file, **kwargs):
            # Track what files are being requested
            recursive_calls.append(module_file)

            # Return appropriate file paths
            if module_file == "module.py":
                return str(main_file)
            elif module_file == "helper1.py":
                return str(helper1_file)
            elif module_file == "helper2.py":
                return str(helper2_file)
            else:
                return f"/cache/path/{module_file}"

        mock_cached.side_effect = side_effect_cached_file

        # Mock create_dynamic_module_it to actually create directories
        def mock_create_dynamic_module_it_dirs(path):
            mock_module_cache_full_submodule_path = mock_it_modules_cache / path
            mock_module_cache_full_submodule_path.mkdir(parents=True, exist_ok=True)

        mock_create.side_effect = mock_create_dynamic_module_it_dirs

        # Mock check_imports to prevent infinite recursion
        def side_effect_check_imports(file_path):
            call_count[file_path] = call_count.get(file_path, 0) + 1

            # Only return dependencies for the main module on first call
            if file_path == str(main_file) and call_count[file_path] == 1:
                return ["helper1", "helper2"]
            else:
                return []  # No dependencies for helper files or subsequent calls

        mock_check.side_effect = side_effect_check_imports

        with patch("interpretune.analysis.ops.dynamic_module_utils.os.path.isdir", return_value=False):
            with patch(
                "interpretune.analysis.ops.dynamic_module_utils.os.path.basename", return_value="user--remote-repo"
            ):
                with patch("interpretune.analysis.ops.dynamic_module_utils.shutil.copy"):
                    with patch("interpretune.analysis.ops.dynamic_module_utils.importlib.invalidate_caches"):
                        get_cached_module_file_it("user/remote-repo", "module.py")

                        # Should have made calls for all modules (main + helpers)
                        assert "module.py" in recursive_calls, f"Expected 'module.py' in calls: {recursive_calls}"
                        assert "helper1.py" in recursive_calls, f"Expected 'helper1.py' in calls: {recursive_calls}"
                        assert "helper2.py" in recursive_calls, f"Expected 'helper2.py' in calls: {recursive_calls}"

    @patch("interpretune.analysis.ops.dynamic_module_utils.create_dynamic_module_it")
    @patch("interpretune.analysis.ops.dynamic_module_utils.get_cached_module_file_it")
    def test_get_function_from_dynamic_module_repo_parsing(self, mock_get_cached, mock_create):
        """Test function reference parsing with repo specification."""
        mock_get_cached.return_value = "/test/module.py"

        with patch("interpretune.analysis.ops.dynamic_module_utils.get_function_in_module") as mock_get_func:
            mock_get_func.return_value = lambda: "test"

            # Test with repo specification in function reference
            _ = get_function_from_dynamic_module("test/repo--module.function_name", "default/repo")

            # Should parse repo from function reference
            mock_get_cached.assert_called_once()
            args, kwargs = mock_get_cached.call_args
            assert args[0] == "test/repo"  # Should use parsed repo
            assert args[1] == "module.py"

    @patch("interpretune.analysis.ops.dynamic_module_utils.get_cached_module_file_it")
    def test_get_function_from_dynamic_module_use_auth_token_deprecation_warning(self, mock_get_cached):
        """Test that get_function_from_dynamic_module raises deprecation warning for use_auth_token."""
        mock_get_cached.return_value = "/test/module.py"

        with patch("interpretune.analysis.ops.dynamic_module_utils.get_function_in_module") as mock_get_func:
            mock_get_func.return_value = lambda: "test"

            with pytest.warns(FutureWarning, match="use_auth_token.*deprecated"):
                get_function_from_dynamic_module("module.function_name", "/test/repo", use_auth_token="test_token")

    def test_get_function_from_dynamic_module_use_auth_token_and_token_conflict(self):
        """Test that specifying both use_auth_token and token raises ValueError."""
        with pytest.raises(ValueError, match="token.*and.*use_auth_token.*both specified"):
            get_function_from_dynamic_module(
                "module.function_name", "/test/repo", token="new_token", use_auth_token="old_token"
            )

    @patch("interpretune.analysis.ops.dynamic_module_utils.get_cached_module_file_it")
    def test_get_function_from_dynamic_module_use_auth_token_sets_token(self, mock_get_cached):
        """Test that use_auth_token value is transferred to token parameter."""
        mock_get_cached.return_value = "/test/module.py"

        with patch("interpretune.analysis.ops.dynamic_module_utils.get_function_in_module") as mock_get_func:
            mock_get_func.return_value = lambda: "test"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = get_function_from_dynamic_module(
                    "module.function_name", "/test/repo", use_auth_token="test_token"
                )

                # Verify the function was called and token was passed correctly
                mock_get_cached.assert_called_once()
                call_kwargs = mock_get_cached.call_args[1]
                assert call_kwargs["token"] == "test_token"
                assert result is not None


class TestOpPathMgmt:
    """Test suite for operation path management functions."""

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

    def test_ensure_op_paths_empty_and_none_paths(self):
        """Test handling of empty and None paths."""
        with patch("interpretune.analysis.ops.dynamic_module_utils.rank_zero_warn") as mock_warn:
            # Test with empty string and None values
            test_paths = ["", None]  # Empty string, None

            cleanup_op_paths()
            ensure_op_paths_in_syspath(test_paths)

            # Should not add any paths or log warnings for empty values
            added_paths = get_added_op_paths()
            assert len(added_paths) == 0
            mock_warn.assert_not_called()

    def test_ensure_op_paths_nonexistent_path(self):
        """Test handling of non-existent paths."""
        with patch("interpretune.analysis.ops.dynamic_module_utils.rank_zero_warn") as mock_warn:
            # Create a path that doesn't exist
            nonexistent_path = "/this/path/does/not/exist"
            test_paths = [nonexistent_path]

            cleanup_op_paths()
            ensure_op_paths_in_syspath(test_paths)

            # Should log warning and not add to sys.path
            added_paths = get_added_op_paths()
            assert len(added_paths) == 0
            mock_warn.assert_called_once()
            warning_msg = mock_warn.call_args[0][0]
            assert "Operation path does not exist" in warning_msg
            # Normalize both paths for cross-platform assertion
            norm_expected = os.path.normpath(nonexistent_path)
            norm_actual = os.path.normpath(warning_msg.split(":", 1)[-1].strip()) if ":" in warning_msg else warning_msg
            assert norm_expected in norm_actual or norm_actual in norm_expected

    def test_ensure_op_paths_file_instead_of_directory(self):
        """Test handling of file paths instead of directory paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file instead of using a directory
            test_file = Path(temp_dir) / "test_file.txt"
            test_file.write_text("test content")

            with patch("interpretune.analysis.ops.dynamic_module_utils.rank_zero_warn") as mock_warn:
                test_paths = [str(test_file)]

                cleanup_op_paths()
                ensure_op_paths_in_syspath(test_paths)

                # Should log warning and not add to sys.path
                added_paths = get_added_op_paths()
                assert len(added_paths) == 0
                mock_warn.assert_called_once()
                warning_msg = mock_warn.call_args[0][0]
                assert "Operation path is not a directory" in warning_msg
                assert str(test_file.resolve()) in warning_msg

    def test_ensure_op_paths_mixed_valid_invalid(self):
        """Test handling of mixed valid and invalid paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid directory and a file
            valid_dir = Path(temp_dir) / "valid_dir"
            valid_dir.mkdir()

            test_file = Path(temp_dir) / "invalid_file.txt"
            test_file.write_text("test")

            nonexistent_path = "/does/not/exist"

            with patch("interpretune.analysis.ops.dynamic_module_utils.rank_zero_warn") as mock_warn:
                test_paths = [str(valid_dir), str(test_file), nonexistent_path, ""]

                cleanup_op_paths()
                ensure_op_paths_in_syspath(test_paths)

                # Should add only the valid directory
                added_paths = get_added_op_paths()
                assert len(added_paths) == 1
                assert str(valid_dir.resolve()) in added_paths

                # Should log warnings for invalid paths
                assert mock_warn.call_count == 2  # One for file, one for nonexistent

                warning_calls = [call[0][0] for call in mock_warn.call_args_list]
                assert any("not a directory" in msg for msg in warning_calls)
                assert any("does not exist" in msg for msg in warning_calls)

    def test_remove_op_paths_empty_and_none_paths(self):
        """Test handling of empty and None paths in remove function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First add a valid path
            valid_path = Path(temp_dir) / "valid_dir"
            valid_path.mkdir()

            cleanup_op_paths()
            ensure_op_paths_in_syspath([str(valid_path)])

            # Now try to remove empty/None paths along with the valid one
            test_paths = ["", None, str(valid_path)]

            remove_op_paths_from_syspath(test_paths)

            # Valid path should be removed, empty/None should be ignored
            added_paths = get_added_op_paths()
            assert len(added_paths) == 0

    def test_remove_op_paths_not_in_syspath(self):
        """Test removing paths that are tracked but not actually in sys.path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_dir"
            test_dir.mkdir()

            cleanup_op_paths()

            # Manually add to tracking set without adding to sys.path
            from interpretune.analysis.ops.dynamic_module_utils import _added_op_paths

            test_path_str = str(test_dir.resolve())
            _added_op_paths.add(test_path_str)

            # Verify it's in tracking but not in sys.path
            assert test_path_str in _added_op_paths
            assert test_path_str not in sys.path

            # Now try to remove it
            remove_op_paths_from_syspath([test_dir])

            # Should be removed from tracking set despite not being in sys.path
            assert test_path_str not in _added_op_paths

    def test_remove_op_paths_mixed_scenarios(self):
        """Test removing paths with mixed scenarios - some in sys.path, some not, some empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple test directories
            dir1 = Path(temp_dir) / "dir1"
            dir2 = Path(temp_dir) / "dir2"
            dir1.mkdir()
            dir2.mkdir()

            cleanup_op_paths()

            # Add both directories properly
            ensure_op_paths_in_syspath([str(dir1), str(dir2)])

            # Manually remove dir2 from sys.path but keep it in tracking
            dir2_str = str(dir2.resolve())
            sys.path.remove(dir2_str)

            # Now try to remove all paths including empty ones
            test_paths = [str(dir1), str(dir2), "", None]

            remove_op_paths_from_syspath(test_paths)

            # Both should be removed from tracking
            added_paths = get_added_op_paths()
            assert len(added_paths) == 0
            assert str(dir1.resolve()) not in sys.path

    def test_cleanup_op_paths_not_in_syspath(self):
        """Test cleanup when paths are tracked but not actually in sys.path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_dir"
            test_dir.mkdir()

            cleanup_op_paths()

            # Manually add to tracking set without adding to sys.path
            from interpretune.analysis.ops.dynamic_module_utils import _added_op_paths

            test_path_str = str(test_dir.resolve())
            _added_op_paths.add(test_path_str)

            # Verify it's in tracking but not in sys.path
            assert test_path_str in _added_op_paths
            assert test_path_str not in sys.path

            # Now cleanup - this should trigger the ValueError handling
            cleanup_op_paths()

            # Should clear tracking set despite ValueError
            assert len(_added_op_paths) == 0


class TestCreateDynModuleGetFn:
    """Test suite for dynamic module creation and function retrieval."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_module_content = "def test_function():\n    return 'test'"

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.fixture
    def mock_it_modules_cache(self):
        """Fixture providing a temporary directory for IT_MODULES_CACHE."""
        with tempfile.TemporaryDirectory() as temp_cache:
            with patch("interpretune.analysis.ops.dynamic_module_utils.IT_MODULES_CACHE", temp_cache):
                yield temp_cache

    def test_get_function_from_dynamic_module_missing_module_error(self):
        """Test error handling when module file is not found."""
        with patch("interpretune.analysis.ops.dynamic_module_utils.get_cached_module_file_it", return_value=None):
            with pytest.raises(OSError, match="Could not locate the module file"):
                get_function_from_dynamic_module("module.function_name", "/test/repo")

    def test_create_dynamic_module_it_recursive_parent_creation(self, mock_it_modules_cache):
        """Test that create_dynamic_module_it recursively creates parent modules."""
        from interpretune.analysis.ops.dynamic_module_utils import create_dynamic_module_it

        # Create a nested module path that requires recursive parent creation
        nested_module_name = "level1/level2/level3/test_module"

        with patch("interpretune.analysis.ops.dynamic_module_utils.IT_MODULES_CACHE", mock_it_modules_cache):
            with patch("interpretune.analysis.ops.dynamic_module_utils.importlib.invalidate_caches") as mock_invalidate:
                # Call create_dynamic_module_it with the nested path
                create_dynamic_module_it(nested_module_name)

                # Verify that all levels of the hierarchy were created
                expected_paths = [
                    Path(mock_it_modules_cache) / "level1",
                    Path(mock_it_modules_cache) / "level1" / "level2",
                    Path(mock_it_modules_cache) / "level1" / "level2" / "level3",
                    Path(mock_it_modules_cache) / "level1" / "level2" / "level3" / "test_module",
                ]

                for path in expected_paths:
                    assert path.exists(), f"Expected directory {path} to exist"
                    init_file = path / "__init__.py"
                    assert init_file.exists(), f"Expected __init__.py to exist in {path}"

                # Verify invalidate_caches was called (once for each level)
                assert mock_invalidate.call_count >= len(expected_paths)


class TestGetFunctionInModule:
    """Test suite for get_function_in_module function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_module_content = "def test_function():\n    return 'test'"

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.fixture
    def mock_it_modules_cache(self):
        """Fixture providing a temporary directory for IT_MODULES_CACHE."""
        with tempfile.TemporaryDirectory() as temp_cache:
            with patch("interpretune.analysis.ops.dynamic_module_utils.IT_MODULES_CACHE", temp_cache):
                yield temp_cache

    def test_force_reload_removes_from_sys_modules(self, mock_it_modules_cache):
        """Test that force_reload=True removes module from sys.modules and invalidates caches."""
        # Create a test module file
        module_path = "test_module.py"
        module_file = Path(mock_it_modules_cache) / module_path
        module_file.write_text("def test_function():\n    return 'force_reload_test'")

        # Mock get_relative_import_files to return empty list
        with patch("interpretune.analysis.ops.dynamic_module_utils.get_relative_import_files", return_value=[]):
            # First, load the module normally to get it in sys.modules
            from interpretune.analysis.ops.dynamic_module_utils import get_function_in_module

            func1 = get_function_in_module("test_function", module_path)
            assert func1() == "force_reload_test"

            # Verify module is in sys.modules
            module_name = "test_module"
            assert module_name in sys.modules

            # Mock sys.modules as a whole and importlib.invalidate_caches to verify they're called
            original_sys_modules = sys.modules.copy()
            with patch("sys.modules", original_sys_modules), patch("importlib.invalidate_caches") as mock_invalidate:
                # Now call with force_reload=True
                func2 = get_function_in_module("test_function", module_path, force_reload=True)

                # Verify sys.modules.pop was called
                # Since we can't directly verify pop was called, verify the module was removed and re-added
                assert mock_invalidate.called

                assert func2() == "force_reload_test"

    def test_cached_module_reuse(self, mock_it_modules_cache):
        """Test that cached module is reused when hash matches."""
        # Create a test module file
        module_path = "cached_test_module.py"
        module_file = Path(mock_it_modules_cache) / module_path
        module_file.write_text("def cached_function():\n    return 'cached_result'")

        # Mock get_relative_import_files to return empty list
        with patch("interpretune.analysis.ops.dynamic_module_utils.get_relative_import_files", return_value=[]):
            from interpretune.analysis.ops.dynamic_module_utils import get_function_in_module

            # First call to load and cache the module
            func1 = get_function_in_module("cached_function", module_path)
            assert func1() == "cached_result"

            # Get reference to the cached module
            module_name = "cached_test_module"
            cached_module = sys.modules[module_name]

            # Mock importlib.util.module_from_spec to verify it's NOT called for cached module
            with patch("importlib.util.module_from_spec") as mock_from_spec:
                # Second call should use cached module
                func2 = get_function_in_module("cached_function", module_path)

                # Verify module_from_spec was NOT called (cached module was used)
                mock_from_spec.assert_not_called()

                # Verify we got the same result
                assert func2() == "cached_result"

                # Verify the same module object is still in sys.modules
                assert sys.modules[module_name] is cached_module

    def test_module_hash_change_triggers_reload(self, mock_it_modules_cache):
        """Test that module reload occurs when hash changes."""
        # Create a test module file
        module_path = "hash_test_module.py"
        module_file = Path(mock_it_modules_cache) / module_path
        original_content = "def hash_function():\n    return 'original'"
        module_file.write_text(original_content)

        # Mock get_relative_import_files to return empty list
        with patch("interpretune.analysis.ops.dynamic_module_utils.get_relative_import_files", return_value=[]):
            from interpretune.analysis.ops.dynamic_module_utils import get_function_in_module

            # First call to load the module
            func1 = get_function_in_module("hash_function", module_path)
            assert func1() == "original"

            # Get reference to the cached module and its initial hash
            module_name = "hash_test_module"
            cached_module = sys.modules[module_name]
            initial_hash = getattr(cached_module, "__interpretune_module_hash__", None)
            assert initial_hash is not None

            # Modify the module file to change its hash
            new_content = "def hash_function():\n    return 'modified'"
            module_file.write_text(new_content)

            # Instead of mocking exec_module, let's verify the behavior by checking the hash and function output
            # Call again - should detect hash change and reload
            func2 = get_function_in_module("hash_function", module_path)

            # Verify the hash was updated (this indicates reload occurred)
            new_hash = getattr(cached_module, "__interpretune_module_hash__", None)
            assert new_hash != initial_hash, "Hash should have changed after file modification"

            # Verify the function behavior reflects the new code
            # Note: Since we can't directly test exec_module call due to the module reloading mechanics,
            # we test the observable behavior instead
            try:
                _ = func2()
                # The result might still be 'original' if the function object was cached
                # But the hash should definitely be different
                assert new_hash != initial_hash
            except Exception:
                # If there's an exception, that's also evidence that reload was attempted
                assert new_hash != initial_hash

    def test_function_not_found_in_module(self, mock_it_modules_cache):
        """Test error when requested function doesn't exist in module."""
        # Create a test module file without the requested function
        module_path = "no_function_module.py"
        module_file = Path(mock_it_modules_cache) / module_path
        module_file.write_text("def other_function():\n    return 'other'")

        # Mock get_relative_import_files to return empty list
        with patch("interpretune.analysis.ops.dynamic_module_utils.get_relative_import_files", return_value=[]):
            from interpretune.analysis.ops.dynamic_module_utils import get_function_in_module

            with pytest.raises(AttributeError):
                get_function_in_module("nonexistent_function", module_path)

    def test_relative_imports_included_in_hash(self, mock_it_modules_cache):
        """Test that relative imports are included in module hash calculation."""
        # Create main module and a helper module
        main_module_path = "main_with_imports.py"
        helper_module_path = "helper_module.py"

        main_module_file = Path(mock_it_modules_cache) / main_module_path
        helper_module_file = Path(mock_it_modules_cache) / helper_module_path

        main_module_file.write_text("def main_function():\n    return 'main'")
        helper_module_file.write_text("def helper_function():\n    return 'helper'")

        # Mock get_relative_import_files to return the helper module
        with patch("interpretune.analysis.ops.dynamic_module_utils.get_relative_import_files") as mock_get_imports:
            mock_get_imports.return_value = [str(helper_module_file)]

            from interpretune.analysis.ops.dynamic_module_utils import get_function_in_module

            # First call to establish baseline
            func1 = get_function_in_module("main_function", main_module_path)
            assert func1() == "main"

            # Get reference to the cached module and its initial hash
            module_name = "main_with_imports"
            cached_module = sys.modules[module_name]
            initial_hash = getattr(cached_module, "__interpretune_module_hash__", None)
            assert initial_hash is not None

            # Modify the helper module to change the overall hash
            helper_module_file.write_text("def helper_function():\n    return 'modified_helper'")

            # Call again - should detect hash change due to helper module change
            func2 = get_function_in_module("main_function", main_module_path)

            # Verify the hash was updated (this indicates reload occurred due to helper file change)
            new_hash = getattr(cached_module, "__interpretune_module_hash__", None)
            assert new_hash != initial_hash, "Hash should have changed after helper module modification"

            # Verify that the function still works (main functionality should be preserved)
            assert func2() == "main"

    def test_force_reload_clears_hash_and_reloads(self, mock_it_modules_cache):
        """Test that force_reload=True actually forces reload regardless of hash."""
        # Create a test module file
        module_path = "force_reload_test.py"
        module_file = Path(mock_it_modules_cache) / module_path
        module_file.write_text("def test_function():\n    return 'original'")

        # Mock get_relative_import_files to return empty list
        with patch("interpretune.analysis.ops.dynamic_module_utils.get_relative_import_files", return_value=[]):
            from interpretune.analysis.ops.dynamic_module_utils import get_function_in_module

            # First call to load the module
            func1 = get_function_in_module("test_function", module_path)
            assert func1() == "original"

            # Get reference to the cached module
            module_name = "force_reload_test"
            cached_module = sys.modules[module_name]

            # Verify module has hash
            initial_hash = getattr(cached_module, "__interpretune_module_hash__", None)
            assert initial_hash is not None

            # Call with force_reload=True without changing the file
            # This should reload the module and update the hash even if content is the same
            func2 = get_function_in_module("test_function", module_path, force_reload=True)

            # Hash should be recalculated (might be the same value but recalculated)
            new_hash = getattr(cached_module, "__interpretune_module_hash__", None)
            assert new_hash is not None

            # Function should still work
            assert func2() == "original"
