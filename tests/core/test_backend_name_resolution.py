"""Cold-registry resolution of every bundled analysis backend NAME.

``resolve_analysis_backend`` swallows the ``ImportError`` from its lazy import, so a module path that
goes stale degrades into "no backend registered" rather than failing where the cause is visible. The
failure mode is therefore an ABSENCE, which passes silently unless something checks it: these are the
positive controls for the ``_BUNDLED_BACKEND_MODULES`` table.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from interpretune.analysis.backends import capabilities as caps


@pytest.fixture
def cold_registry():
    """Simulate a genuinely cold start: empty registry AND the table's modules evicted from sys.modules.

    Clearing the registry alone is not enough. A backend registers at module import, so a module already
    in ``sys.modules`` is returned from cache without re-executing, and resolution would fail for a
    reason that never occurs in practice. Evicting makes the import actually run, which is the thing
    under test. Both the registry and ``sys.modules`` are restored afterwards, so the transient second
    module object does not outlive the test.
    """
    saved_registry = dict(caps.ANALYSIS_BACKEND_REGISTRY)
    saved_modules = {m: sys.modules[m] for m in caps._BUNDLED_BACKEND_MODULES.values() if m in sys.modules}
    caps.ANALYSIS_BACKEND_REGISTRY.clear()
    for m in saved_modules:
        del sys.modules[m]
    try:
        yield caps.ANALYSIS_BACKEND_REGISTRY
    finally:
        for m in list(caps._BUNDLED_BACKEND_MODULES.values()):
            sys.modules.pop(m, None)
        sys.modules.update(saved_modules)
        caps.ANALYSIS_BACKEND_REGISTRY.clear()
        caps.ANALYSIS_BACKEND_REGISTRY.update(saved_registry)


@pytest.mark.parametrize("name", sorted(caps._BUNDLED_BACKEND_MODULES))
def test_every_bundled_name_resolves_from_a_cold_registry(name, cold_registry):
    assert name not in cold_registry  # the control: the registry really is cold
    backend = caps.resolve_analysis_backend(name)
    assert backend is not None
    assert name in cold_registry


@pytest.mark.parametrize("name,module_path", sorted(caps._BUNDLED_BACKEND_MODULES.items()))
def test_table_paths_are_importable_and_register_their_name(name, module_path, cold_registry):
    """The table's VALUE must be the module that registers the KEY, not merely an importable module."""
    importlib.import_module(module_path)
    assert name in cold_registry, f"{module_path} did not register {name!r}"


def test_an_unknown_name_raises_rather_than_resolving(cold_registry):
    with pytest.raises(KeyError, match="No analysis backend registered as 'not_a_backend'"):
        caps.resolve_analysis_backend("not_a_backend")


def test_a_stale_table_path_would_be_caught(cold_registry, monkeypatch):
    """Negative control: with a wrong path the resolve fails, so the tests above are not vacuous."""
    monkeypatch.setitem(caps._BUNDLED_BACKEND_MODULES, "circuit_tracer", "interpretune.adapters.circuit_tracer.moved")
    with pytest.raises(KeyError):
        caps.resolve_analysis_backend("circuit_tracer")
