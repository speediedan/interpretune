"""Run the doctests that used to be collected from ``src/interpretune`` (Phase 4 two-root migration).

The suite invocation dropped ``src/interpretune`` as a collection root (`pytest tests src/it_examples/tests`),
which would have silently retired the five docstring examples ``--doctest-modules`` collected there. They stay
in the docstrings (documentation stays with the code); this module keeps them executing from the main test
root.
"""

from __future__ import annotations

import doctest

import pytest

import interpretune.adapters.nnsight.config
import interpretune.utils.import_utils

_DOCTEST_MODULES = (interpretune.utils.import_utils, interpretune.adapters.nnsight.config)


@pytest.mark.parametrize("module", _DOCTEST_MODULES, ids=lambda m: m.__name__)
def test_module_doctests(module):
    results = doctest.testmod(module, verbose=False)
    assert results.attempted > 0, f"no doctest examples found in {module.__name__} — surface silently shrank"
    assert results.failed == 0, f"{results.failed} doctest failure(s) in {module.__name__}"


def test_expected_doctest_surface():
    """Fail if a known docstring example gets deleted rather than migrated."""
    finder = doctest.DocTestFinder()
    found = {t.name for mod in _DOCTEST_MODULES for t in finder.find(mod) if t.examples}
    expected = {
        "interpretune.utils.import_utils.compare_version",
        "interpretune.utils.import_utils.module_available",
        "interpretune.utils.import_utils.package_available",
        "interpretune.adapters.nnsight.config.ITNNsightConfig",
        "interpretune.adapters.nnsight.config.NNsightConfig",
    }
    assert expected <= found, f"doctest surface shrank: missing {expected - found}"
