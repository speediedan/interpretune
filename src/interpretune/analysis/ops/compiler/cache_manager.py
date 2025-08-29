"""Cache manager for pre-compiled operation definitions."""

from __future__ import annotations
import re
import hashlib
import importlib.util
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field


from huggingface_hub import scan_cache_dir
from huggingface_hub.utils._cache_manager import CachedRepoInfo, CachedRevisionInfo
from transformers.dynamic_module_utils import resolve_trust_remote_code

from interpretune.utils.logging import rank_zero_debug, rank_zero_warn
from interpretune.analysis.ops.base import OpSchema, ColCfg
from interpretune.analysis import IT_TRUST_REMOTE_CODE


@dataclass(frozen=True)
class OpDef:
    """Frozen dataclass representing a pre-compiled operation definition."""

    name: str
    description: str
    implementation: str
    input_schema: OpSchema
    output_schema: OpSchema
    aliases: List[str] = field(default_factory=list)
    importable_params: Dict[str, str] = field(default_factory=dict)
    normal_params: Dict[str, Any] = field(default_factory=dict)
    required_ops: List[str] = field(default_factory=list)
    composition: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility with existing code."""
        result = {
            "name": self.name,
            "description": self.description,
            "implementation": self.implementation,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "aliases": self.aliases,
            "importable_params": self.importable_params,
            "normal_params": self.normal_params,
            "required_ops": self.required_ops,
            "composition": self.composition,
        }
        return result


class YamlFileInfo:
    """Information about a YAML file for caching purposes."""

    def __init__(self, path: Path, mtime: float, content_hash: str):
        self.path = path
        self.mtime = mtime
        self.content_hash = content_hash

    @classmethod
    def from_path(cls, path: Path) -> "YamlFileInfo":
        """Create YamlFileInfo from a file path."""
        stat = path.stat()
        content = path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        return cls(path, stat.st_mtime, content_hash)


def _get_latest_revision(repo: CachedRepoInfo) -> Optional[CachedRevisionInfo]:
    """Get the latest revision for a repository, preferring 'main' ref.

    Args:
        repo: CachedRepoInfo object from the cache scanner

    Returns:
        CachedRevisionInfo for the latest revision, or None if no revisions found
    """
    if not repo.revisions:
        return None

    # First, try to find revision referenced by 'main'
    main_revision = repo.refs.get("main")
    if main_revision is not None:
        return main_revision

    # Fallback: get the most recently modified revision
    return max(repo.revisions, key=lambda rev: rev.last_modified)


class OpDefinitionsCacheManager:
    """Manages caching of compiled operation definitions."""

    _it_trust_remote_code_warning = (
        "The environmental variable IT_TRUST_REMOTE_CODE is not currently set. In order "
        "to load analysis operations from previously downloaded op collection modules without being re-prompted "
        "repository by repository, you can set the environmental variable IT_TRUST_REMOTE_CODE to "
        "('1', 'yes' or 'true')"
    )
    _it_trust_false_skipping = "Skipping loading ops from hub repositories due to IT_TRUST_REMOTE_CODE being `False`."

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._yaml_files: List[YamlFileInfo] = []
        self._fingerprint: Optional[str] = None

    def add_yaml_file(self, yaml_file: Path) -> None:
        """Add a YAML file to be monitored for changes."""
        # Avoid duplicates
        if any(info.path == yaml_file for info in self._yaml_files):
            return

        try:
            file_info = YamlFileInfo.from_path(yaml_file)
            self._yaml_files.append(file_info)
            self._fingerprint = None  # Reset fingerprint when files change
        except FileNotFoundError:
            # Skip files that don't exist anymore
            pass

    def add_hub_yaml_files(self) -> List[Path]:
        """Add hub YAML files to monitoring."""
        hub_yaml_files = []
        try:
            # we can short-circuit if IT_TRUST_REMOTE_CODE is explicitly set to False
            if IT_TRUST_REMOTE_CODE is False:
                rank_zero_warn(OpDefinitionsCacheManager._it_trust_false_skipping)
                rank_zero_debug("[ANALYSIS_HUB_CACHE] Returning early due to IT_TRUST_REMOTE_CODE=False")
                return []

            hub_yaml_files = self.discover_hub_yaml_files()
            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Discovered {len(hub_yaml_files)} YAML files")

            for yaml_file in hub_yaml_files:
                rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Adding YAML file: {yaml_file}")
                self.add_yaml_file(yaml_file)

        except Exception as e:
            rank_zero_warn(f"Error discovering hub YAML files: {e}")
            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Exception details: {type(e).__name__}: {str(e)}")
            import traceback

            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Traceback: {traceback.format_exc()}")

        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Returning {len(hub_yaml_files)} YAML files")
        return hub_yaml_files  # type: ignore[return-value]

    def discover_hub_yaml_files(self) -> List[Path]:
        """Discover YAML files from the most recent revision of each hub repository.

        Uses HuggingFace's cache manager to efficiently find YAML files only from
        the latest revision (preferring 'main' ref) of each cached model repository.

        Returns:
            List of Path objects pointing to YAML files from latest revisions only.
        """
        from interpretune.analysis import IT_ANALYSIS_HUB_CACHE

        yaml_files = []

        # in case this method is called directly, short-circuit if IT_TRUST_REMOTE_CODE is explicitly set to False
        if IT_TRUST_REMOTE_CODE is False:
            # If IT_TRUST_REMOTE_CODE is explicitly set to False, we skip loading ops from hub repositories
            rank_zero_warn(OpDefinitionsCacheManager._it_trust_false_skipping)
            return yaml_files

        # Convert to Path object for consistent handling
        hub_cache_path = Path(IT_ANALYSIS_HUB_CACHE)
        if not hub_cache_path.exists():
            return yaml_files

        try:
            # Use HuggingFace cache manager
            cache_info = scan_cache_dir(hub_cache_path)

            if len(cache_info.repos) > 0 and IT_TRUST_REMOTE_CODE is None:
                rank_zero_warn(OpDefinitionsCacheManager._it_trust_remote_code_warning)

            # Sort repos by repo_id for deterministic ordering
            sorted_repos = sorted(cache_info.repos, key=lambda repo: repo.repo_id)

            for repo in sorted_repos:
                trust_remote_code = IT_TRUST_REMOTE_CODE or resolve_trust_remote_code(
                    IT_TRUST_REMOTE_CODE, repo.repo_id, False, True
                )

                if not trust_remote_code:
                    rank_zero_warn(
                        f"Skipping loading ops from repository {repo.repo_id} due to trust_remote_code being `False`."
                    )
                    continue
                # Only consider model repositories
                if repo.repo_type != "model":
                    continue

                # Find the latest revision for this repo (preferring 'main' ref)
                latest_revision = _get_latest_revision(repo)
                if latest_revision is None:
                    continue

                # Check for YAML files in this revision
                for file_info in latest_revision.files:
                    if file_info.file_name.endswith((".yaml", ".yml")):
                        # use file_path if available, otherwise construct path
                        if hasattr(file_info, "file_path") and file_info.file_path is not None:
                            yaml_path = Path(file_info.file_path)
                        else:
                            yaml_path = latest_revision.snapshot_path / file_info.file_name

                        if yaml_path.exists():
                            yaml_files.append(yaml_path)

        except Exception as e:
            rank_zero_warn(f"Failed to discover hub YAML files: {e}")

        return sorted(yaml_files)  # Sort for deterministic results

    def _parse_hub_file_path(self, yaml_file: Path) -> tuple[bool, str]:
        """Parse a file path to determine if it's a hub ops file and extract namespace.

        Args:
            yaml_file: Path to the YAML file to analyze

        Returns:
            Tuple of (is_hub_file, namespace) where:
            - is_hub_file: True if this is a hub operations file
            - namespace: The extracted namespace (empty string if not a hub file)
        """
        from interpretune.analysis import IT_ANALYSIS_HUB_CACHE

        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Input yaml_file: {yaml_file}")
        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] IT_ANALYSIS_HUB_CACHE: {IT_ANALYSIS_HUB_CACHE}")

        # Check if file is in hub cache
        try:
            # Resolve both paths to handle symlinks (especially on macOS where /tmp -> /private/tmp)
            resolved_yaml_file = yaml_file.resolve()
            resolved_hub_cache = Path(IT_ANALYSIS_HUB_CACHE).resolve()

            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Resolved yaml_file: {resolved_yaml_file}")
            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Resolved hub_cache: {resolved_hub_cache}")

            relative_path = resolved_yaml_file.relative_to(resolved_hub_cache)
            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Relative path: {relative_path}")
            parts = relative_path.parts

            # Look for models-- pattern and snapshots or blobs directory
            for i, part in enumerate(parts):
                rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Checking part {i}: '{part}'")
                if part.startswith("models--"):
                    # Check if this is a valid hub cache structure (has snapshots or blobs)
                    remaining_parts = parts[i:]
                    if "snapshots" in remaining_parts or "blobs" in remaining_parts:
                        # Use regex to properly extract user and repo parts
                        # Only match exactly two sets of '--' (models--user--repo)
                        regex_pattern = r"models--([^-]+(?:-[^-]+)*)--([^-]+(?:-[^-]+)*)$"
                        match = re.match(regex_pattern, part)

                        if match:
                            user, repo = match.groups()
                            namespace = f"{user}.{repo}"
                            rank_zero_debug(
                                f"[ANALYSIS_HUB_CACHE] REGEX MATCH SUCCESS: user='{user}', "
                                f"repo='{repo}', namespace='{namespace}'"
                            )
                            # Return namespace without top-level package name
                            return True, namespace
                    break
        except ValueError as e:
            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] ValueError in relative_to: {e}")
            pass

        # Not a hub file
        rank_zero_debug("[ANALYSIS_HUB_CACHE] NOT A HUB FILE - returning (False, '')")
        return False, ""

    def get_hub_namespace(self, yaml_file: Path) -> str:
        """Extract namespace from hub file path."""
        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] get_hub_namespace input: {yaml_file}")

        is_hub_file, namespace = self._parse_hub_file_path(yaml_file)

        rank_zero_debug(
            f"[ANALYSIS_HUB_CACHE] get_hub_namespace result: is_hub_file={is_hub_file}, namespace='{namespace}'"
        )
        return namespace

    @property
    def fingerprint(self) -> str:
        """Get a fingerprint representing the current state of all YAML files."""
        if self._fingerprint is None:
            if not self._yaml_files:
                self._fingerprint = "empty"
            else:
                # Create a combined hash of all file information
                combined_info = []
                for file_info in self._yaml_files:
                    # Include path, mtime, and content hash
                    combined_info.append(f"{file_info.path}:{file_info.mtime}:{file_info.content_hash}")

                combined_str = "|".join(sorted(combined_info))
                full_hash = hashlib.sha256(combined_str.encode()).hexdigest()
                self._fingerprint = full_hash[:16]  # Truncate for readability

        return self._fingerprint

    def _get_cache_module_path(self) -> Path:
        """Get the path for the cache module file."""
        return self.cache_dir / f"op_definitions_{self.fingerprint}.py"

    def _cleanup_old_cache_files(self) -> None:
        """Remove old cache files."""
        pattern = "op_definitions_*.py"
        current_file = f"op_definitions_{self.fingerprint}.py"

        for old_file in self.cache_dir.glob(pattern):
            if old_file.name != current_file:
                try:
                    old_file.unlink()
                    rank_zero_debug(f"Removed old cache file: {old_file}")
                except OSError as e:
                    rank_zero_warn(f"Failed to remove old cache file {old_file}: {e}")

    def is_cache_valid(self) -> bool:
        """Check if the current cache is valid."""

        cache_path = self._get_cache_module_path()
        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Cache path: {cache_path}")

        if not cache_path.exists():
            rank_zero_debug("[ANALYSIS_HUB_CACHE] Cache invalid: file does not exist")
            return False

        cache_mtime = cache_path.stat().st_mtime
        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Cache mtime: {cache_mtime}")
        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Checking {len(self._yaml_files)} source files")

        # Check if any source files are newer than cache
        for file_info in self._yaml_files:
            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Checking source file: {file_info.path}")
            if not file_info.path.exists():
                rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Cache invalid: source file missing {file_info.path}")
                return False

            source_mtime = file_info.path.stat().st_mtime
            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Source mtime: {source_mtime} vs cache: {cache_mtime}")
            if source_mtime > cache_mtime:
                rank_zero_debug("[ANALYSIS_HUB_CACHE] Cache invalid: source newer than cache")
                return False

        rank_zero_debug("[ANALYSIS_HUB_CACHE] Cache is valid")
        return True

    def _generate_module_content(self, op_definitions: Dict[str, OpDef]) -> str:
        """Generate Python module content for the cache."""
        lines = [
            "# GENERATED FILE - DO NOT EDIT",
            "# This file contains cached operation definitions",
            f"# Fingerprint: {self.fingerprint}",
            "",
            "from interpretune.analysis.ops.base import OpSchema, ColCfg",
            "from interpretune.analysis.ops.compiler.cache_manager import OpDef",
            "",
            f'FINGERPRINT = "{self.fingerprint}"',
            "",
            "OP_DEFINITIONS = {",
        ]

        # Filter out alias entries - only serialize canonical operation definitions
        # Aliases will be reconstructed from the dispatcher's _aliases mapping
        canonical_ops = {}
        for name, op_def in op_definitions.items():
            # Only include operations where the name matches the canonical name
            if op_def.name == name:
                canonical_ops[name] = op_def

        for name, op_def in canonical_ops.items():
            op_def_str = self._serialize_op_def(op_def)
            lines.append(f'    "{name}": {op_def_str},')

        lines.append("}")

        return "\n".join(lines)

    def _serialize_op_def(self, op_def: OpDef) -> str:
        """Serialize an OpDef to Python code."""
        fields = []

        # Always include required fields
        fields.append(f'name="{op_def.name}"')
        fields.append(f'description="{op_def.description}"')
        fields.append(f'implementation="{op_def.implementation}"')
        fields.append(f"input_schema={self._serialize_op_schema(op_def.input_schema)}")
        fields.append(f"output_schema={self._serialize_op_schema(op_def.output_schema)}")

        # Include optional fields that have values
        if op_def.aliases:
            fields.append(f"aliases={op_def.aliases!r}")
        if op_def.importable_params:
            fields.append(f"importable_params={op_def.importable_params!r}")
        if op_def.normal_params:
            fields.append(f"normal_params={op_def.normal_params!r}")
        if op_def.required_ops:
            fields.append(f"required_ops={op_def.required_ops!r}")
        if op_def.composition:
            fields.append(f"composition={op_def.composition!r}")

        return f"OpDef({', '.join(fields)})"

    def _serialize_op_schema(self, schema: OpSchema) -> str:
        """Serialize an OpSchema to Python code."""
        if not schema:
            return "OpSchema({})"

        fields = []
        for field_name, col_cfg in schema.items():
            fields.append(f'"{field_name}": {self._serialize_col_cfg(col_cfg)}')

        return f"OpSchema({{{', '.join(fields)}}})"

    def _serialize_col_cfg(self, col_cfg: ColCfg) -> str:
        """Serialize a ColCfg to Python code."""
        from dataclasses import fields, MISSING

        # Get all fields of ColCfg
        cfg_fields = fields(ColCfg)
        args = []

        # Always include datasets_dtype as it's required
        args.append(f'datasets_dtype="{col_cfg.datasets_dtype}"')

        # Include other fields only if they differ from defaults
        for field_info in cfg_fields:
            if field_info.name == "datasets_dtype":
                continue  # Already handled

            value = getattr(col_cfg, field_info.name)

            # Check if this field has a default value
            has_default = field_info.default is not MISSING

            if has_default:
                default_value = field_info.default
                if value != default_value:
                    if isinstance(value, str):
                        args.append(f'{field_info.name}="{value}"')
                    else:
                        args.append(f"{field_info.name}={value!r}")

        return f"ColCfg({', '.join(args)})"

    def save_cache(self, op_definitions: Dict[str, OpDef]) -> Path:
        """Save operation definitions to cache."""
        cache_path = self._get_cache_module_path()

        # Clean up old cache files first
        self._cleanup_old_cache_files()

        # Generate module content
        content = self._generate_module_content(op_definitions)

        # Write to cache file
        cache_path.write_text(content)

        return cache_path

    def load_cache(self) -> Optional[Dict[str, OpDef]]:
        """Load operation definitions from cache."""
        if not self.is_cache_valid():
            return None

        cache_path = self._get_cache_module_path()

        try:
            # Import the cache module dynamically
            spec = importlib.util.spec_from_file_location("op_definitions_cache", cache_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Verify fingerprint matches
            if hasattr(module, "FINGERPRINT") and module.FINGERPRINT != self.fingerprint:
                rank_zero_warn("Cache fingerprint mismatch, invalidating cache")
                return None

            # Return the operations
            if hasattr(module, "OP_DEFINITIONS"):
                op_definitions = module.OP_DEFINITIONS
                if not op_definitions:
                    rank_zero_warn("No operation definitions found in cache")
                    return None
                return op_definitions

        except Exception as e:
            rank_zero_warn(f"Failed to load cache: {e}")

        return None
