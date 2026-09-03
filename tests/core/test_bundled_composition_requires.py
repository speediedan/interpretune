"""A bundled adapter's compositions must not vanish silently when its optional dependency is absent.

Measured on `main` before this change: with circuit-tracer absent, `register_all_adapters` survived, **18 of
48 compositions disappeared, and nothing was printed** (#431). The skip itself was never the defect --
registering a subset is correct when a dependency is genuinely absent. The defect was that "this
composition is unavailable here" and "this composition does not exist" became indistinguishable at exactly
the moment a user needs to tell them apart.

`_light_register`'s own docstring has promised a warning here since it was written and never emitted one.
"""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from interpretune.adapters._light_register import (
    _implementation_module,
    discover_adapter_entrypoints,
    register_all_adapters,
)
from interpretune.adapters.registration import CompositionRegistry

ADAPTER_PACKAGES = ("circuit_tracer", "transformer_lens", "sae_lens", "nnsight")
SRC = Path(__file__).parent.parent.parent / "src" / "interpretune" / "adapters"


class TestDeclarationIsReadableWithoutTheDependency:
    """The declaration must be EAGER. Routing it through the lazy map would look like tidying and break it.

    A lazily-resolved ``__it_requires__`` would import the implementation submodule to answer the question,
    which needs the very dependency being tested for -- silently restoring the bootstrap problem while
    reading as consistency.
    """

    @pytest.mark.parametrize("pkg", ADAPTER_PACKAGES)
    def test_declaration_is_a_module_level_assignment(self, pkg):
        """Static guard: the constant is assigned at module level, not routed through ``__getattr__``."""
        init = SRC / pkg / "__init__.py"
        tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
        assigned = [
            t.id
            for node in tree.body  # module level ONLY
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name)
        ]
        assert "__it_requires__" in assigned, (
            f"{pkg}/__init__.py must assign __it_requires__ at module level; a lazily-resolved declaration "
            "imports the submodule to answer 'should I import the submodule'."
        )

    def test_declaration_is_readable_with_the_dependency_blocked(self):
        """Runtime proof, in a subprocess so the import blocker cannot leak into the rest of the suite."""
        probe = textwrap.dedent("""
            import sys, importlib.abc, warnings
            warnings.filterwarnings("ignore")
            import interpretune  # noqa: F401

            class Blocker(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] == "circuit_tracer":
                        raise ModuleNotFoundError(f"No module named {name!r}")
                    return None

            sys.meta_path.insert(0, Blocker())
            for m in [k for k in sys.modules if k.split(".")[0] == "circuit_tracer"]:
                del sys.modules[m]
            for m in [k for k in sys.modules if "adapters.circuit_tracer" in k]:
                del sys.modules[m]

            # POSITIVE CONTROL: a blocker that fails to block reports "readable" for something it never
            # guarded, which is indistinguishable from the result being sought.
            try:
                import circuit_tracer  # noqa: F401
                print("CONTROL_FAILED")
                raise SystemExit(0)
            except ModuleNotFoundError:
                print("CONTROL_OK")

            from interpretune.adapters._light_register import _declared_requires
            declared = _declared_requires("interpretune.adapters.circuit_tracer")
            print("DECLARED_OK" if declared else "DECLARED_MISSING", declared)
        """)
        proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=300)
        out = proc.stdout
        assert "CONTROL_OK" in out, f"blocker did not block; probe proves nothing.\n{out}\n{proc.stderr}"
        assert "DECLARED_OK" in out, (
            f"the declaration was unreadable with circuit-tracer absent, which is the one case it exists "
            f"for.\n{out}\n{proc.stderr}"
        )


class TestUnavailableAdaptersAreReported:
    def test_an_unmet_declared_requirement_is_reported_with_its_reason(self, monkeypatch, caplog):
        """The reason must name the DEPENDENCY, not a symptom.

        A caught ``ImportError`` says only that something failed; an evaluated requirement says what is
        missing, which is the difference between a user knowing to `pip install` and a user filing a bug.
        """
        import interpretune.adapters.circuit_tracer as ctpkg

        if not discover_adapter_entrypoints():
            pytest.skip("installed interpretune metadata predates the entry-point group; reinstall to exercise")
        monkeypatch.setattr(ctpkg, "__it_requires__", {"pip": ["a-package-nobody-has"]}, raising=False)
        with caplog.at_level(logging.INFO):
            register_all_adapters(CompositionRegistry())
        text = caplog.text
        assert "a-package-nobody-has" in text, f"the skip did not name the missing dependency:\n{text}"
        assert "not installed" in text

    def test_nothing_is_reported_when_everything_is_available(self, caplog):
        """Negative control: the report fires on absence, not on every import.

        Without this, a report that fired unconditionally would pass the test above while telling a
        fully-provisioned user their adapters are unavailable.
        """
        with caplog.at_level(logging.INFO):
            register_all_adapters(CompositionRegistry())
        assert "unavailable in this environment" not in caplog.text


class TestImplementationModuleResolution:
    """The entry point names the IMPORT-SAFE module; the registrable classes may live one level down.

    Resolving this AFTER the requirement check is what keeps the heavy import behind the predicate. Getting it backwards
    would import the framework in order to decide whether the framework should be imported.
    """

    def test_a_packaged_adapter_resolves_to_its_adapter_submodule(self):
        assert (
            _implementation_module("interpretune.adapters.circuit_tracer")
            == "interpretune.adapters.circuit_tracer.adapter"
        )

    def test_a_flat_adapter_module_resolves_to_itself(self):
        """`core` and `lightning` carry no heavy module-level import, so they are both halves at once."""
        assert _implementation_module("interpretune.adapters.core") == "interpretune.adapters.core"
        assert _implementation_module("interpretune.adapters.lightning") == "interpretune.adapters.lightning"

    def test_a_nonexistent_module_resolves_to_itself_rather_than_raising(self):
        """Resolution must not raise on a name a third party got wrong; the import that follows reports it."""
        assert _implementation_module("not.a.real.module") == "not.a.real.module"


class TestDiscoveryIsEntrypointDriven:
    def test_an_empty_group_warns_rather_than_registering_nothing_quietly(self, monkeypatch):
        """Registering nothing must never be indistinguishable from an environment with no adapters.

        This is the #431 lesson at the discovery layer: the failure is silent by default, and its most likely cause --
        installed metadata predating the group -- looks exactly like a correct empty result.
        """
        import importlib.metadata as md

        # Patch the LOOKUP, not `discover_adapter_entrypoints` -- patching the function would replace the
        # code under test, and the assertion would then be about the stub rather than about the rails.
        monkeypatch.setattr(md, "entry_points", lambda **kw: [])
        with pytest.warns(UserWarning, match="entry-point group is empty"):
            discover_adapter_entrypoints()

    def test_the_bundled_adapters_are_discoverable_when_metadata_is_current(self):
        """Skips rather than fails when the installed metadata predates the group, and SAYS which it is."""
        found = discover_adapter_entrypoints()
        if not found:
            pytest.skip("installed interpretune metadata predates the entry-point group; reinstall to exercise")
        assert set(found) >= {"core", "lightning", "circuit_tracer", "nnsight"}
        for name, value in found.items():
            assert not value.endswith(".adapter"), (
                f"{name} names an implementation module; entry points must name the IMPORT-SAFE module, "
                "or resolving one imports the framework it is deciding about."
            )


class TestManifestCompositionsArePartitioned:
    """The hub half of the same predicate: a declared composition may be skipped, never silently."""

    @staticmethod
    def _manifest(extra_requires):
        return {
            "adapters": {
                "compositions": [
                    {"component": "module", "adapters": ["core", "interp_engine"]},
                    {
                        "component": "module",
                        "adapters": ["core", "interp_engine", "circuit_tracer"],
                        "requires": extra_requires,
                    },
                ]
            }
        }

    def test_an_unsatisfiable_entry_is_partitioned_out_with_its_reason(self):
        from interpretune.hub.adapters import _partition_declared_compositions

        satisfiable, unsupported = _partition_declared_compositions(
            self._manifest({"pip": ["a-package-nobody-has"]}), source="demo"
        )
        assert satisfiable == (("module", "core", "interp_engine"),)
        assert len(unsupported) == 1
        assert "a-package-nobody-has" in unsupported[0][1], "the skip must carry the REASON, not just the fact"

    def test_a_satisfiable_entry_is_kept(self):
        """Positive control: the partition is caused by the requirement, not by having a `requires` key."""
        from interpretune.hub.adapters import _partition_declared_compositions

        satisfiable, unsupported = _partition_declared_compositions(self._manifest({"pip": ["pytest"]}), source="demo")
        assert len(satisfiable) == 2 and not unsupported

    def test_composition_identity_is_order_insensitive(self):
        """Manifest author order must not matter: the registry side value-sorts, so this side must too."""
        from interpretune.hub.adapters import _composition_key

        a = _composition_key({"component": "module", "adapters": ["lightning", "interp_engine"]})
        b = _composition_key({"component": "module", "adapters": ["interp_engine", "lightning"]})
        assert a == b


class TestDirectImportPathIsNamed:
    def test_supported_compositions_is_None_outside_a_load(self):
        """`None` means "no manifest governs this invocation" -- the pip / direct-import path.

        Not empty (which would silently register nothing) and not the full declared set (which would import an absent
        dependency). A component reached by `pip install` has no rails to consult.
        """
        from interpretune.hub.adapters import supported_compositions

        assert supported_compositions() is None


class TestSupportedCompositionsIsThreadScoped:
    """The set is a ContextVar, so a spawned thread reads `None` -- which MEANS something different there.

    `None` is documented as "no manifest governs this invocation". In a thread spawned during a load that
    reading is false: a manifest does govern, the thread just cannot see it. Unreachable today because
    entrypoints execute synchronously, and pinned so the docstring cannot quietly stop being true.
    """

    def test_a_spawned_thread_does_not_inherit_the_set(self):
        import threading

        from interpretune.hub.adapters import _SUPPORTED_COMPOSITIONS, supported_compositions

        token = _SUPPORTED_COMPOSITIONS.set((("module", "core", "demo"),))
        try:
            seen = {}

            def worker():
                seen["value"] = supported_compositions()

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            assert supported_compositions() == (("module", "core", "demo"),), "the setting thread must see it"
            assert seen["value"] is None, (
                "if a spawned thread ever DOES inherit the set, supported_compositions()'s docstring is "
                "wrong in the other direction and must be updated."
            )
        finally:
            _SUPPORTED_COMPOSITIONS.reset(token)
