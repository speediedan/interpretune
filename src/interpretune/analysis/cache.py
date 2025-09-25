"""AnalysisStore cache control and temporary-cache helpers.

This module provides a small feature-flag API for an Interpretune-specific
AnalysisStore caching layer. The feature is disabled by default. If enabled,
we currently emit a warning that the feature is planned but not yet
implemented for the MVP. The module also exposes a helper to obtain an
appropriate cache directory for an analysis run: either a permanent location
under `IT_ANALYSIS_CACHE` (when both dataset-level caching and the
interpretune analysisstore caching are enabled) or a temporary directory that
will be registered for deletion at process exit.

We prefer reusing huggingface/datasets helpers when available and fall back
to the stdlib when needed.

Much of the logic below is modeled to mirror HF datasets' internal caching/fingerprinting just applied to another layer
of abstraction.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from interpretune.utils import rank_zero_warn

try:
    # Preferred helpers from HF datasets if available
    from datasets.fingerprint import (
        is_caching_enabled as _hf_is_caching_enabled,
        get_temporary_cache_files_directory as _hf_get_temp_cache_dir,
        maybe_register_dataset_for_temp_dir_deletion as _hf_maybe_register_tempdir,
    )
except Exception:
    _hf_is_caching_enabled = None
    _hf_get_temp_cache_dir = None
    _hf_maybe_register_tempdir = None

from interpretune.analysis import IT_ANALYSIS_CACHE

# Feature flag (disabled by default)
_ANALYSISSTORE_CACHING_ENABLED = False


def enable_analysisstore_caching(enable: bool) -> None:
    """Enable or disable the Interpretune AnalysisStore caching layer.

    Notes:
    - Default is False. Setting True will currently emit a warning indicating
      the feature is planned but not implemented for the MVP.
    - We also check whether the HF datasets-level caching is enabled; if not,
      we emit an additional warning explaining that dataset-level caching is
      required for a persistent AnalysisStore cache.
    """
    global _ANALYSISSTORE_CACHING_ENABLED
    if enable:
        rank_zero_warn(
            "enable_analysisstore_caching() is a planned feature and not yet implemented; "
            "enabling currently has no effect beyond this warning."
        )
        # Warn if HF dataset caching isn't enabled
        if _hf_is_caching_enabled is not None and not _hf_is_caching_enabled():
            rank_zero_warn(
                "HuggingFace datasets caching is disabled. "
                "AnalysisStore persistent caching requires datasets.enable_caching()."
            )
    _ANALYSISSTORE_CACHING_ENABLED = bool(enable)


def is_analysisstore_caching_enabled() -> bool:
    """Return whether the Interpretune AnalysisStore caching flag is enabled."""
    return _ANALYSISSTORE_CACHING_ENABLED


def _create_tempdir() -> Path:
    """Create a temporary cache directory and register it for deletion.

    Prefer HF's helper which keeps a shared temp-cache root; otherwise use tempfile.mkdtemp and register an atexit
    cleanup.
    """
    if _hf_get_temp_cache_dir is not None:
        tmpdir = Path(_hf_get_temp_cache_dir())
        # Register for cleanup if HF exposes the register helper
        if _hf_maybe_register_tempdir is not None:
            try:
                _hf_maybe_register_tempdir(str(tmpdir))
            except Exception:
                # Fall back to local atexit registration if HF helper fails
                atexit.register(lambda p=tmpdir: shutil.rmtree(p, ignore_errors=True))
        else:
            atexit.register(lambda p=tmpdir: shutil.rmtree(p, ignore_errors=True))
        return tmpdir

    # Fallback path
    tmpdir = Path(tempfile.mkdtemp(prefix="interpretune_analysis_"))
    atexit.register(lambda p=tmpdir: shutil.rmtree(p, ignore_errors=True))
    return tmpdir


def get_analysis_cache_dir(module, explicit_cache_dir: Optional[str | Path] = None) -> Path:
    """Return the cache directory to use for analysis for the given module.

    Behavior:
    - If `explicit_cache_dir` is provided, return that (creating it if needed).
    - If analysisstore caching is enabled and HF dataset caching is enabled,
      return a path under `IT_ANALYSIS_CACHE/<dataset_config>/<dataset_fingerprint>/<module>`.
    - Otherwise return a temporary directory registered for deletion at exit.
    """
    # Respect an explicit override
    if explicit_cache_dir is not None:
        out = Path(explicit_cache_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out

    # If both toggles are enabled, choose permanent cache beneath IT_ANALYSIS_CACHE
    if is_analysisstore_caching_enabled() and (_hf_is_caching_enabled is None or _hf_is_caching_enabled()):
        try:
            dataset = module.datamodule.dataset["validation"]  # type: ignore[attr-defined]
            cfg_name = dataset.config_name
            fingerprint = dataset._fingerprint
        except Exception:
            # Best-effort fallback - put node under IT_ANALYSIS_CACHE/module-name
            cfg_name = "unknown"
            fingerprint = "unknown"
        cache_dir = Path(IT_ANALYSIS_CACHE) / cfg_name / fingerprint / module.__class__._orig_module_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    # Default: temporary cache dir
    return _create_tempdir()
