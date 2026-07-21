"""Typed structural fallbacks for the optional ``neuronpedia_utils`` package.

``neuronpedia_utils`` is a local-checkout package (neuronpedia repo, ``utils/neuronpedia-utils``,
installed editable with ``--no-deps``); it is not on PyPI and is absent in plain CI and fresh
environments. ``interpretune.utils.neuronpedia_db_utils`` imports these fallbacks when the real
package is unavailable so that module (and everything re-exporting from it) stays importable and
statically analyzable; every functional entry point raises with an actionable install message at
call time instead.
"""

from __future__ import annotations

from typing import Any

_MISSING_NEURONPEDIA_UTILS_MSG = (
    "This functionality requires the 'neuronpedia_utils' package (neuronpedia repo, "
    "utils/neuronpedia-utils). Install it into the active environment with: "
    "uv pip install -e <neuronpedia-repo>/utils/neuronpedia-utils --no-deps"
)

DEFAULT_COLUMNAR_COPY_IMPORT_TABLES: frozenset[str] = frozenset()
DEFAULT_COLUMNAR_IMPORT_TABLES: frozenset[str] = frozenset()
DEFAULT_IMPORT_MODE_CONFIGS: dict[str, Any] = {}


class _MissingNeuronpediaUtils:
    """Placeholder for neuronpedia_utils types; instantiation raises with install guidance.

    ``__getattr__`` is declared so static analysis of first-party code that accesses real
    attributes of the genuine classes (e.g. ``NeuronpediaLocalImportSummary.imported_row_counts``)
    type-checks against these stand-ins; at runtime instances never exist (``__init__`` raises).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ModuleNotFoundError(_MISSING_NEURONPEDIA_UTILS_MSG)

    def __getattr__(self, name: str) -> Any: ...


class NeuronpediaBundleImportParity(_MissingNeuronpediaUtils):
    """Stand-in for ``neuronpedia_utils.local_db_import.NeuronpediaBundleImportParity``."""


class NeuronpediaBundleSummaryParity(_MissingNeuronpediaUtils):
    """Stand-in for ``neuronpedia_utils.local_db_import.NeuronpediaBundleSummaryParity``."""


class NeuronpediaExportBundleSummary(_MissingNeuronpediaUtils):
    """Stand-in for ``neuronpedia_utils.local_db_import.NeuronpediaExportBundleSummary``."""


class NeuronpediaLocalImportSummary(_MissingNeuronpediaUtils):
    """Stand-in for ``neuronpedia_utils.local_db_import.NeuronpediaLocalImportSummary``."""


class NeuronpediaLocalDBImportError(RuntimeError):
    """Stand-in for the neuronpedia_utils error type when the package is unavailable."""


def _raise_missing_neuronpedia_utils(*args: Any, **kwargs: Any) -> Any:
    raise ModuleNotFoundError(_MISSING_NEURONPEDIA_UTILS_MSG)


def benchmark_neuronpedia_export_bundle_local_db_modes(*args: Any, **kwargs: Any) -> Any:
    return _raise_missing_neuronpedia_utils(*args, **kwargs)


def compare_neuronpedia_export_bundle_summaries(*args: Any, **kwargs: Any) -> Any:
    return _raise_missing_neuronpedia_utils(*args, **kwargs)


def compare_neuronpedia_export_bundle_to_import_summary(*args: Any, **kwargs: Any) -> Any:
    return _raise_missing_neuronpedia_utils(*args, **kwargs)


def summarize_neuronpedia_export_bundle(*args: Any, **kwargs: Any) -> Any:
    return _raise_missing_neuronpedia_utils(*args, **kwargs)


def summarize_neuronpedia_export_bundle_arrow(*args: Any, **kwargs: Any) -> Any:
    return _raise_missing_neuronpedia_utils(*args, **kwargs)


def summarize_neuronpedia_export_bundle_parquet(*args: Any, **kwargs: Any) -> Any:
    return _raise_missing_neuronpedia_utils(*args, **kwargs)


def import_saedashboard_columnar_bundle_local_db(*args: Any, **kwargs: Any) -> Any:
    return _raise_missing_neuronpedia_utils(*args, **kwargs)


def import_neuronpedia_export_bundle_local_db(*args: Any, **kwargs: Any) -> Any:
    return _raise_missing_neuronpedia_utils(*args, **kwargs)
