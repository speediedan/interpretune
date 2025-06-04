"""Unit tests for enhanced AnalysisOpDispatcher with hub operations."""
import tempfile
from pathlib import Path
from unittest.mock import patch

from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher
from interpretune.analysis.ops.compiler.cache_manager import OpDefinitionsCacheManager


class TestAnalysisOpDispatcherHub:
    """Test cases for AnalysisOpDispatcher hub functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create a test YAML file
        self.test_yaml = self.temp_dir / "test_ops.yaml"
        self.test_yaml.write_text("""
test_op:
  description: A test operation
  implementation: test.module.test_function
  aliases: ['test_alias']
  input_schema:
    input_data:
      datasets_dtype: string
  output_schema:
    output_data:
      datasets_dtype: string
""")

    def teardown_method(self):
        """Clean up test environment after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dispatcher_with_hub_ops_disabled(self):
        """Test dispatcher with hub operations disabled."""
        dispatcher = AnalysisOpDispatcher(
            yaml_paths=[self.test_yaml],
            enable_hub_ops=False
        )

        assert dispatcher.enable_hub_ops is False

        with patch.object(dispatcher._cache_manager, 'add_hub_yaml_files') as mock_add_hub:
            dispatcher.load_definitions()
            mock_add_hub.assert_not_called()

    def test_dispatcher_with_hub_ops_enabled(self):
        """Test dispatcher with hub operations enabled."""
        dispatcher = AnalysisOpDispatcher(
            yaml_paths=[self.test_yaml],
            enable_hub_ops=True
        )

        assert dispatcher.enable_hub_ops is True

        with patch.object(dispatcher._cache_manager, 'add_hub_yaml_files') as mock_add_hub, \
             patch.object(dispatcher._cache_manager, 'discover_hub_yaml_files', return_value=[]):
            dispatcher.load_definitions()
            mock_add_hub.assert_called_once()

    def test_apply_hub_namespacing_native_file(self):
        """Test namespacing is not applied to native files."""
        dispatcher = AnalysisOpDispatcher(yaml_paths=[self.test_yaml])

        raw_definitions = {
            "test_op": {
                "description": "Test operation",
                "aliases": ["test_alias"]
            }
        }

        # Mock native file path
        native_file = Path(__file__).parent.parent / "src" / "interpretune" / "analysis" / "ops" / "native.yaml"

        with patch.object(dispatcher._cache_manager, 'get_hub_namespace', return_value=""):
            result = dispatcher._apply_hub_namespacing(raw_definitions, native_file)

        assert "test_op" in result

    def test_apply_hub_namespacing_hub_file(self):
        """Test namespacing is applied to hub files."""
        dispatcher = AnalysisOpDispatcher(yaml_paths=[self.test_yaml])

        raw_definitions = {
            "test_op": {
                "description": "Test operation",
                "aliases": ["test_alias"],
                "required_ops": ["other_op"]
            },
            "other_op": {
                "description": "Other operation"
            }
        }

        hub_file = Path("/fake/hub/cache/models--user--repo/snapshots/abc/ops.yaml")

        with patch.object(dispatcher._cache_manager, 'get_hub_namespace', return_value="user.repo"):
            result = dispatcher._apply_hub_namespacing(raw_definitions, hub_file)

        assert "user.repo.test_op" in result
        assert "user.repo.other_op" in result
        assert result["user.repo.test_op"]["aliases"] == ["user.repo.test_alias"]
        # TODO: verify that relying on the provided required_ops name resolving works as intended
        #       if collisions are too frequent, we may need to adjust this logic
        assert result["user.repo.test_op"]["required_ops"] == ["other_op"]

    @patch('interpretune.analysis.ops.dispatcher.yaml.safe_load')
    def test_load_yaml_with_errors(self, mock_yaml_load):
        """Test loading YAML files with errors is handled gracefully."""
        mock_yaml_load.side_effect = Exception("YAML parse error")

        dispatcher = AnalysisOpDispatcher(yaml_paths=[self.test_yaml])

        # Should not raise exception, just log and continue
        dispatcher.load_definitions()

        # Should have empty definitions due to error
        assert len(dispatcher._op_definitions) == 0

    def test_load_empty_yaml_file(self):
        """Test loading empty YAML file."""
        empty_yaml = self.temp_dir / "empty.yaml"
        empty_yaml.write_text("")

        dispatcher = AnalysisOpDispatcher(yaml_paths=[empty_yaml])
        dispatcher.load_definitions()

        # Should handle empty file gracefully
        assert len(dispatcher._op_definitions) == 0


class TestOpDefinitionsCacheManagerHub:
    """Test cases for OpDefinitionsCacheManager hub functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = Path(tempfile.mkdtemp())
        self.hub_cache = self.temp_dir / "hub_cache"

    def teardown_method(self):
        """Clean up test environment after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discover_hub_yaml_files_hub_cache(self):
        """Test discovering YAML files from hub cache."""
        # Create fake hub cache structure
        hub_cache = self.temp_dir / "hub"
        hub_cache.mkdir()
        repo_dir = hub_cache / "models--username--repo"
        repo_dir.mkdir(parents=True)
        snapshots_dir = repo_dir / "snapshots"
        snapshots_dir.mkdir()
        snapshot_dir = snapshots_dir / "abc123"
        snapshot_dir.mkdir()
        (snapshot_dir / "ops.yaml").write_text("test: {}")

        cache_manager = OpDefinitionsCacheManager(self.temp_dir)

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', hub_cache), \
             patch('interpretune.analysis.IT_ANALYSIS_OP_PATHS', []):
            yaml_files = cache_manager.discover_hub_yaml_files()

        assert len(yaml_files) == 1
        assert yaml_files[0].name == "ops.yaml"

    def test_get_hub_namespace_hub_file(self):
        """Test getting namespace for hub file."""
        cache_manager = OpDefinitionsCacheManager(self.temp_dir)

        hub_file = Path("/cache/hub/interpretune-ops/models--username--some_repo/snapshots/abc/ops.yaml")

        with patch('interpretune.analysis.IT_ANALYSIS_HUB_CACHE', Path("/cache/hub/interpretune-ops/")):
            namespace = cache_manager.get_hub_namespace(hub_file)

        assert namespace == "username.some_repo"

    def test_get_hub_namespace_user_file(self):
        """Test getting namespace for user file."""
        cache_manager = OpDefinitionsCacheManager(self.temp_dir)

        user_file = Path("/some/other/path/ops.yaml")

        namespace = cache_manager.get_hub_namespace(user_file)

        assert namespace == ""

    def test_add_hub_yaml_files(self):
        """Test adding hub YAML files to cache manager."""
        cache_manager = OpDefinitionsCacheManager(self.temp_dir)

        # Create test file
        test_file = self.temp_dir / "test.yaml"
        test_file.write_text("test: {}")

        with patch.object(cache_manager, 'discover_hub_yaml_files', return_value=[test_file]):
            cache_manager.add_hub_yaml_files()

        # Should have added the file
        assert any(info.path == test_file for info in cache_manager._yaml_files)

    def test_add_hub_yaml_files_with_error(self):
        """Test adding hub YAML files handles errors gracefully."""
        cache_manager = OpDefinitionsCacheManager(self.temp_dir)

        # Create non-existent file
        nonexistent_file = self.temp_dir / "nonexistent.yaml"

        with patch.object(cache_manager, 'discover_hub_yaml_files', return_value=[nonexistent_file]):
            # Should not raise exception
            cache_manager.add_hub_yaml_files()

        # Should not have added the file
        assert not any(info.path == nonexistent_file for info in cache_manager._yaml_files)
