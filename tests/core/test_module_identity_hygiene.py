"""A module reimported by a test must leave both of its references consistent.

Importing a submodule binds it in TWO places: ``sys.modules["pkg.sub"]`` and ``pkg.sub`` as an attribute of
the parent package object. A test that reimports a module to observe import-time behavior must restore both,
because they are read by different mechanisms:

- ``monkeypatch.setattr("pkg.sub.NAME", value)`` and ``unittest.mock.patch("pkg.sub.NAME")`` walk the PARENT
  ATTRIBUTE;
- a call-time ``from pkg.sub import NAME`` reads ``SYS.MODULES``.

Leave those disagreeing and a patch lands on a module nothing reads: the patch "succeeds", the code under
test sees the original value, and the failure surfaces as a confusing symptom far from the cause. Measured
2026-08-18: ``test_hub_manager.py::test_environment_variable_parsing`` restored only ``sys.modules``, which
took 41 op-collection tests red in the full suite while every one of them passed in isolation, since nothing
before them in a targeted run had done the reimport.

**Attribution is the hard part, not detection.** The check below reports that the invariant is broken; it
cannot say who broke it, and a bare "module identity is split" failure is only marginally better than the
symptoms it replaces. The ``interpretune_module_identity`` hook in ``tests/conftest.py`` is the instrument
that matters: it runs after every test and fails the test that ACTUALLY caused the split. This module keeps
the invariant documented and gives a fast standalone check.
"""

from __future__ import annotations

import sys

import pytest

from tests.module_identity import WATCHED_MODULES, module_identity_split


@pytest.mark.parametrize("dotted", WATCHED_MODULES)
def test_parent_attribute_and_sys_modules_agree(dotted: str):
    """The parent-package attribute and ``sys.modules`` must be the SAME object."""
    __import__(dotted)
    assert sys.modules.get(dotted) is not None, f"{dotted} is not in sys.modules after import"
    split = module_identity_split(dotted)
    assert split is None, split
