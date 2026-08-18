"""Detection for a module bound to two different objects, and the modules worth watching.

Importing a submodule binds it in TWO places -- ``sys.modules["pkg.sub"]`` and ``pkg.sub`` as an attribute of
the parent package object -- and those two are read by DIFFERENT mechanisms:

- patch targets (``monkeypatch.setattr("pkg.sub.NAME", v)``, ``mock.patch("pkg.sub.NAME")``) walk the
  **parent attribute**;
- a call-time ``from pkg.sub import NAME`` reads **sys.modules**.

So a test that reimports a module and restores only one of the two leaves them disagreeing for the rest of
the session, and every later patch through that dotted path lands on a module nothing reads: the patch
reports success, the code under test keeps the original value, and the failure appears somewhere unrelated.

Shared by the standalone invariant test (``tests/core/test_module_identity_hygiene.py``) and the
after-every-test hook in ``tests/conftest.py`` that attributes a split to the test which caused it.
"""

from __future__ import annotations

import sys

# The interpretune packages tests reimport, plus the ones op/hub tests patch values through. `interpretune`
# itself is omitted: it has no parent to disagree with.
WATCHED_MODULES = (
    "interpretune.analysis",
    "interpretune.analysis.ops",
    "interpretune.analysis.ops.dispatcher",
    "interpretune.analysis.ops.compiler.cache_manager",
    "interpretune.hub",
    "interpretune.hub.cache",
    "interpretune.hub.manager",
)


def module_identity_split(dotted: str) -> str | None:
    """Return an explanation if ``dotted`` is bound to two different module objects, else ``None``.

    Only reports on modules already imported: a module absent from ``sys.modules`` is not split, and importing
    it here to find out would perturb the state being checked.
    """
    in_sys = sys.modules.get(dotted)
    if in_sys is None:
        return None
    parent_name, _, leaf = dotted.rpartition(".")
    parent = sys.modules.get(parent_name) if parent_name else None
    if parent is None:
        return None
    via_attribute = getattr(parent, leaf, None)
    if via_attribute is None or via_attribute is in_sys:
        return None
    return (
        f"{dotted} is bound to two different module objects: {parent_name}.{leaf} is not "
        f"sys.modules[{dotted!r}]. A test reimported it and restored only one of the two references. "
        "Patches that walk the parent attribute (monkeypatch/mock with a dotted string) and reads that go "
        "through sys.modules (a call-time `from ... import`) now disagree, so a patched value can be "
        "invisible to the code under test. Restore BOTH references -- via monkeypatch, so the restore also "
        "survives an assertion failure."
    )


def first_module_identity_split() -> str | None:
    """The first watched module found split, or ``None`` when all are consistent."""
    for dotted in WATCHED_MODULES:
        if (split := module_identity_split(dotted)) is not None:
            return split
    return None
