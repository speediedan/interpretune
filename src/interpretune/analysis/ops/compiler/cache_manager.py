"""Cache manager for pre-compiled operation definitions."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, fields
import datetime

from interpretune.analysis import IT_ANALYSIS_CACHE
from interpretune.utils.logging import rank_zero_debug, rank_zero_warn
from interpretune.analysis.ops.base import OpSchema, ColCfg


@dataclass(frozen=True)
class OpDef:
    """Frozen dataclass representing a pre-compiled operation definition."""
    name: str
    description: str
    implementation: str
    input_schema: OpSchema
    output_schema: OpSchema
    aliases: List[str] = field(default_factory=list)
    function_params: Dict[str, str] = field(default_factory=dict)
    required_ops: List[str] = field(default_factory=list)
    composition: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility with existing code."""
        result = {
            'name': self.name,
            'description': self.description,
            'implementation': self.implementation,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'aliases': self.aliases,
            'function_params': self.function_params,
            'required_ops': self.required_ops,
            'composition': self.composition
        }
        return result


@dataclass
class YamlFileInfo:
    """Information about a YAML file for cache management."""
    path: Path
    mtime: float
    content_hash: str

    @classmethod
    def from_path(cls, path: Path) -> 'YamlFileInfo':
        """Create YamlFileInfo from a file path."""
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        stat = path.stat()
        with open(path, 'rb') as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()

        return cls(
            path=path,
            mtime=stat.st_mtime,
            content_hash=content_hash
        )


class OpDefinitionsCacheManager:
    """Manages pre-compiled operation definitions cache."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or IT_ANALYSIS_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._yaml_files: List[YamlFileInfo] = []
        self._fingerprint: Optional[str] = None

    def add_yaml_file(self, yaml_path: Path) -> None:
        """Add a YAML file to be monitored for changes."""
        file_info = YamlFileInfo.from_path(yaml_path)
        # Avoid duplicates
        if not any(info.path == yaml_path for info in self._yaml_files):
            self._yaml_files.append(file_info)
            self._fingerprint = None  # Reset fingerprint

    def _compute_fingerprint(self) -> str:
        """Compute fingerprint based on all monitored YAML files."""
        if not self._yaml_files:
            return "empty"

        # Sort by path to ensure consistent ordering
        sorted_files = sorted(self._yaml_files, key=lambda x: str(x.path))

        # Combine all file hashes and modification times
        combined = ""
        for file_info in sorted_files:
            combined += f"{file_info.path}:{file_info.mtime}:{file_info.content_hash}:"

        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    @property
    def fingerprint(self) -> str:
        """Get the current fingerprint for all monitored files."""
        if self._fingerprint is None:
            self._fingerprint = self._compute_fingerprint()
        return self._fingerprint

    def _get_cache_module_path(self) -> Path:
        """Get the path for the cached module."""
        return self.cache_dir / f"op_definitions_{self.fingerprint}.py"

    def _cleanup_old_cache_files(self) -> None:
        """Remove old cache files with different fingerprints."""
        pattern = "op_definitions_*.py"
        current_file = self._get_cache_module_path().name

        for cache_file in self.cache_dir.glob(pattern):
            if cache_file.name != current_file:
                try:
                    cache_file.unlink()
                    rank_zero_debug(f"Removed old cache file: {cache_file}")
                except OSError as e:
                    rank_zero_warn(f"Failed to remove old cache file {cache_file}: {e}")

    def is_cache_valid(self) -> bool:
        """Check if the current cache is valid."""
        cache_path = self._get_cache_module_path()
        if not cache_path.exists():
            return False

        # Check if any source files have been modified
        for file_info in self._yaml_files:
            if not file_info.path.exists():
                return False

            current_info = YamlFileInfo.from_path(file_info.path)
            if (current_info.mtime != file_info.mtime or
                current_info.content_hash != file_info.content_hash):
                return False

        return True

    def _generate_module_content(self, op_definitions: Dict[str, OpDef]) -> str:
        """Generate the content for the cache module."""
        timestamp = datetime.datetime.now().isoformat()
        source_files = [str(f.path) for f in self._yaml_files]
        source_names = [f.path.name for f in self._yaml_files]

        content = f'''"""
GENERATED FILE - DO NOT EDIT
Edit the canonical YAML files instead: {', '.join(source_names)}
Generated at: {timestamp}
Fingerprint: {self.fingerprint}
Source files: {source_files}
"""
from interpretune.analysis.ops.base import OpSchema, ColCfg
from interpretune.analysis.ops.compiler.cache_manager import OpDef

# Metadata
FINGERPRINT = "{self.fingerprint}"
GENERATED_AT = "{timestamp}"
SOURCE_FILES = {source_files!r}

# Pre-compiled operation definitions
OP_DEFINITIONS = {{
'''

        for op_name, op_def in op_definitions.items():
            content += f'    "{op_name}": {self._serialize_op_def(op_def)},\n'

        content += '}\n'
        return content

    def _serialize_op_def(self, op_def: OpDef) -> str:
        """Serialize an OpDef to Python code."""
        # Serialize input_schema
        input_schema_code = self._serialize_op_schema(op_def.input_schema)
        output_schema_code = self._serialize_op_schema(op_def.output_schema)

        # Build the OpDef constructor call
        args = [
            f'name="{op_def.name}"',
            f'description={op_def.description!r}',
            f'implementation="{op_def.implementation}"',
            f'input_schema={input_schema_code}',
            f'output_schema={output_schema_code}',
        ]

        if op_def.aliases:
            args.append(f'aliases={op_def.aliases!r}')

        if op_def.function_params:
            args.append(f'function_params={op_def.function_params!r}')

        if op_def.required_ops:
            args.append(f'required_ops={op_def.required_ops!r}')

        if op_def.composition is not None:
            args.append(f'composition={op_def.composition!r}')

        return f'OpDef({", ".join(args)})'

    def _serialize_op_schema(self, schema: OpSchema) -> str:
        """Serialize an OpSchema to Python code."""
        if not schema:
            return 'OpSchema({})'

        schema_dict_parts = []
        for field_name, col_cfg in schema.items():
            col_cfg_code = self._serialize_col_cfg(col_cfg)
            schema_dict_parts.append(f'"{field_name}": {col_cfg_code}')

        schema_dict_code = '{' + ', '.join(schema_dict_parts) + '}'
        return f'OpSchema({schema_dict_code})'

    def _serialize_col_cfg(self, col_cfg: ColCfg) -> str:
        """Serialize a ColCfg to Python code."""
        # Get field information from dataclass
        col_cfg_fields = fields(ColCfg)

        args = []

        # Always include datasets_dtype since it's required
        args.append(f'datasets_dtype="{col_cfg.datasets_dtype}"')

        # Get all field values from the ColCfg instance and include non-default values
        for field_info in col_cfg_fields:
            field_name = field_info.name

            # Skip datasets_dtype since we already handled it
            if field_name == 'datasets_dtype':
                continue

            value = getattr(col_cfg, field_name)

            # Determine the default value for this field
            if field_info.default is not field_info.default_factory:
                # Field has an explicit default value
                default_value = field_info.default
                has_default = True

            # Include field if it has no default or if the value differs from default
            if not has_default or value != default_value:
                if isinstance(value, str):
                    args.append(f'{field_name}="{value}"')
                else:
                    args.append(f'{field_name}={value!r}')

        return f'ColCfg({", ".join(args)})'

    def save_cache(self, op_definitions: Dict[str, OpDef]) -> Path:
        """Save operation definitions to cache module."""
        cache_path = self._get_cache_module_path()
        content = self._generate_module_content(op_definitions)

        # Write the module
        with open(cache_path, 'w') as f:
            f.write(content)

        # Cleanup old cache files
        self._cleanup_old_cache_files()

        rank_zero_debug(f"Saved operation definitions cache to: {cache_path}")
        return cache_path

    def load_cache(self) -> Optional[Dict[str, OpDef]]:
        """Load operation definitions from cache if valid."""
        if not self.is_cache_valid():
            return None

        cache_path = self._get_cache_module_path()
        module_name = f"op_definitions_{self.fingerprint}"

        try:
            # Import the cached module
            spec = importlib.util.spec_from_file_location(module_name, cache_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Verify fingerprint matches
            if getattr(module, 'FINGERPRINT', None) != self.fingerprint:
                rank_zero_warn("Cache fingerprint mismatch, invalidating cache")
                return None

            op_definitions = getattr(module, 'OP_DEFINITIONS', None)
            if op_definitions is None or (isinstance(op_definitions, dict) and len(op_definitions) == 0):
                rank_zero_warn(f"No operation definitions found in cache at {cache_path} associated with source "
                               f"files: {self._yaml_files}")
                return None

            rank_zero_debug(f"Loaded {len(op_definitions)} operation definitions from cache")
            return op_definitions

        except Exception as e:
            rank_zero_warn(f"Failed to load cache: {e}")
            return None
