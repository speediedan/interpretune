"""Analysis injection framework.

This package provides tools for instrumenting code with analysis hooks to inspect intermediate states.

The framework is configuration-driven via YAML files that specify:
- Which analysis points to enable
- Regex patterns for hook insertion
"""

from __future__ import annotations

from .orchestrator import AnalysisInjectionOrchestrator, setup_analysis_injection
from .analysis_hook_patcher import get_module_debug_info, verify_patching, HOOK_REGISTRY
from .version_manager import PackageVersionManager

__all__ = [
    "AnalysisInjectionOrchestrator",
    "setup_analysis_injection",
    "get_module_debug_info",
    "verify_patching",
    "HOOK_REGISTRY",
    "PackageVersionManager",
]
