"""Configuration parser for analysis injection framework.

This module loads and validates YAML configuration for analysis hooks. Hooks are specified via regex patterns, not line
numbers, for robustness.
"""

from __future__ import annotations

import re
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

import yaml


def _normalize_regex_pattern(pattern: str) -> str:
    """Normalize regex pattern escapes loaded from YAML.

    YAML literal strings often preserve backslashes verbatim, which can result in
    doubled escape sequences (e.g. ``"\\\\s"``) once loaded. Decoding with
    ``unicode_escape`` collapses those sequences back to their intended form.
    If decoding fails, the original pattern is returned unchanged.
    """

    try:
        return pattern.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return pattern


@dataclass
class FileHook:
    """Represents a hook insertion point in a file.

    Attributes:
        point_id: Unique identifier for this analysis point
        file_path: Path to file to patch
        regex_pattern: Regex pattern to match the line where hook should be inserted
        description: Human-readable description of the hook
        insert_after: If True, insert hook after matched line; if False, before
        enabled: Whether the hook should be registered and patched
    """

    point_id: str
    file_path: Path
    regex_pattern: str
    description: str = ""
    insert_after: bool = True
    enabled: bool = True


@dataclass
class AnalysisInjectionConfig:
    """Complete analysis injection configuration.

    Attributes:
        enabled: Master switch for all hooks
        target_package_version: Optional required version of the target package (e.g., '0.1.0')
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Directory for log files
        analysis_log_prefix: Prefix for analysis log filenames
        enabled_points: List of enabled analysis point IDs (derived from hooks)
        file_hooks: Dict mapping point_id to FileHook objects
        shared_context: Optional dict of context variables to make available to hooks
    """

    enabled: bool
    log_to_console: bool
    log_to_file: bool
    log_dir: str
    analysis_log_prefix: str
    enabled_points: List[str]
    file_hooks: Dict[str, FileHook]
    target_package_version: Optional[str] = None
    analysis_points_module_path: Optional[Path] = None
    shared_context: Optional[Dict[str, Any]] = None


def merge_config_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries.

    The merge strategy is:
    - Recursively merge nested dictionaries.
    - Replace non-dict values (including lists) with the override value.
    - For file_hooks, recursively merge hook configurations by point_id.
    """

    def _merge(base_value: Any, override_value: Any, *, key: str | None = None) -> Any:
        if key == "file_hooks" and isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            return _merge_file_hooks(base_value, override_value)

        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged: Dict[str, Any] = {k: deepcopy(v) for k, v in base_value.items()}
            for child_key, child_value in override_value.items():
                existing = merged.get(child_key)
                if existing is not None:
                    merged[child_key] = _merge(existing, child_value, key=child_key)
                else:
                    merged[child_key] = deepcopy(child_value)
            return merged

        # Fallback: override completely (lists, scalars, etc.)
        return deepcopy(override_value)

    def _merge_file_hooks(
        base_hooks: Mapping[str, Any], override_hooks: Mapping[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Merge file_hooks with recursive merging by point_id."""
        merged: Dict[str, Dict[str, Any]] = {k: deepcopy(v) for k, v in base_hooks.items()}

        for point_id, hook_config in override_hooks.items():
            if isinstance(hook_config, Mapping):
                if point_id in merged:
                    # Recursively merge existing hook config
                    merged[point_id] = _merge(merged[point_id], hook_config)
                else:
                    # New hook
                    merged[point_id] = cast(Dict[str, Any], deepcopy(hook_config))
            else:
                # Override completely
                merged[point_id] = deepcopy(hook_config)

        return merged

    merged_root: Dict[str, Any] = {k: deepcopy(v) for k, v in base.items()}
    for top_key, override_value in override.items():
        existing_value = merged_root.get(top_key)
        if existing_value is not None:
            merged_root[top_key] = _merge(existing_value, override_value, key=top_key)
        else:
            merged_root[top_key] = deepcopy(override_value)
    return merged_root


def parse_config_dict(raw_config: Mapping[str, Any], *, source_path: Path | None = None) -> AnalysisInjectionConfig:
    """Parse a raw configuration dictionary into an :class:`AnalysisInjectionConfig`."""

    raw_config = deepcopy(raw_config)

    def _resolve_path(maybe_path: Optional[str]) -> Optional[Path]:
        if maybe_path is None:
            return None
        candidate = Path(maybe_path)
        if not candidate.is_absolute() and source_path is not None:
            candidate = (source_path.parent / candidate).resolve()
        return candidate

    # Parse settings
    settings = raw_config.get("settings", {})
    enabled = settings.get("enabled", True)
    target_package_version = settings.get("target_package_version")
    log_to_console = settings.get("log_to_console", True)
    log_to_file = settings.get("log_to_file", True)
    # Use tempfile.gettempdir() for better cross-platform compatibility
    log_dir = settings.get("log_dir", tempfile.gettempdir())
    analysis_log_prefix = settings.get("analysis_log_prefix", "attribution_flow_analysis")
    analysis_points_module_path = _resolve_path(settings.get("analysis_points_module_path"))

    # Parse shared context
    shared_context = raw_config.get("shared_context")

    # Parse file hooks
    file_hooks = {}
    file_hooks_config = raw_config.get("file_hooks", {})

    enabled_point_ids: set[str] = set()

    for point_id, hook_config in file_hooks_config.items():
        # Validate required fields
        if "file_path" not in hook_config:
            raise ValueError(f"Hook '{point_id}' missing 'file_path'")
        if "regex_pattern" not in hook_config:
            raise ValueError(f"Hook '{point_id}' missing 'regex_pattern'")

        # Validate regex pattern
        raw_pattern = hook_config["regex_pattern"]
        regex_pattern = _normalize_regex_pattern(raw_pattern)

        try:
            re.compile(regex_pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern for {point_id}: {e}") from e

        enabled = hook_config.get("enable", hook_config.get("enabled", True))

        file_hook = FileHook(
            point_id=point_id,
            file_path=Path(hook_config["file_path"]),
            regex_pattern=regex_pattern,
            description=hook_config.get("description", ""),
            insert_after=hook_config.get("insert_after", True),
            enabled=enabled,
        )

        file_hooks[point_id] = file_hook
        if enabled:
            enabled_point_ids.add(point_id)

    enabled_points = sorted(enabled_point_ids)

    return AnalysisInjectionConfig(
        enabled=enabled,
        target_package_version=target_package_version,
        log_to_console=log_to_console,
        log_to_file=log_to_file,
        log_dir=log_dir,
        analysis_log_prefix=analysis_log_prefix,
        enabled_points=enabled_points,
        file_hooks=file_hooks,
        analysis_points_module_path=analysis_points_module_path,
        shared_context=shared_context,
    )


def load_config(config_path: Path | str, overrides: Optional[Mapping[str, Any]] = None) -> AnalysisInjectionConfig:
    """Load and parse analysis injection configuration from YAML.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        AnalysisInjectionConfig object with parsed configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    if overrides:
        raw_config = merge_config_dict(raw_config, overrides)

    return parse_config_dict(raw_config, source_path=config_path)


def get_enabled_points(config: AnalysisInjectionConfig) -> List[str]:
    """Get list of enabled analysis points.

    Args:
        config: AnalysisInjectionConfig object

    Returns:
        List of enabled point IDs
    """
    return config.enabled_points
