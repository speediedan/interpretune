"""A module reimported by a test must leave both of its references consistent.

Importing a submodule binds it in TWO places: ``sys.modules["pkg.sub"]`` and ``pkg.sub`` as an attribute of
the parent package object. A test that reimports a module to observe import-time behavior must restore both,
because they are read by different mechanisms:

- ``monkeypatch.setattr("pkg.sub.NAME", value)`` and ``unittest.mock.patch("pkg.sub.NAME")`` walk the PARENT
  ATTRIBUTE;
- a call-time ``from pkg.sub import NAME`` reads ``SYS.MODULES``.

Leave those disagreeing and a patch lands on a module nothing reads: the patch "succeeds", the code under
test sees the original value, and the test fails with a confusing symptom far from the cause. Measured
2026-08-18: ``test_hub_manager.py::test_environment_variable_parsing`` restored only ``sys.modules``, which
took 41 op-collection tests red in the full suite while every one of them passed in isolation, since nothing
before them in a targeted run had done the reimport.

This is a session-wide invariant rather than a property of one test, so it is asserted directly. It is cheap
and it fails loudly at the point of breakage instead of somewhere downstream.
"""

from __future__ import annotations

import sys

import pytest

# Packages whose reimport is known to be attempted by tests, plus the ones op/hub tests patch through.
WATCHED_MODULES = (
    "interpretune",
    "interpretune.analysis",
    "interpretune.analysis.ops",
    "interpretune.analysis.ops.dispatcher",
    "interpretune.analysis.ops.compiler.cache_manager",
    "interpretune.hub",
    "interpretune.hub.cache",
    "interpretune.hub.manager",
)


@pytest.mark.parametrize("dotted", WATCHED_MODULES)
def test_parent_attribute_and_sys_modules_agree(dotted: str):
    """The parent-package attribute and ``sys.modules`` must be the SAME object."""
    __import__(dotted)
    in_sys = sys.modules.get(dotted)
    assert in_sys is not None, f"{dotted} is not in sys.modules after import"

    parent_name, _, leaf = dotted.rpartition(".")
    if not parent_name:
        return  # a top-level package has no parent attribute to disagree with

    parent = sys.modules.get(parent_name)
    assert parent is not None, f"{parent_name} is not in sys.modules"
    via_attribute = getattr(parent, leaf, None)
    assert via_attribute is in_sys, (
        f"{dotted} is bound to two different module objects: {parent_name}.{leaf} is not "
        f"sys.modules[{dotted!r}]. Some test reimported it and restored only one of the two references. "
        "Patches that walk the parent attribute and reads that go through sys.modules now disagree, so a "
        "patched value can be invisible to the code under test."
    )
