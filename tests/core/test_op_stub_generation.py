"""Bundled-op stub generation (``scripts/generate_op_stubs.py``).

The committed stub (``src/interpretune/__init__.pyi``) is derived from the bundled op YAMLs only, so the
stale-stubs CI check stays hermetic. That check is a diff against the committed file, which makes it a good
gate and poor feedback: when the generator broke on the ``collection:`` header it failed with "Duplicate
bundled op definition 'collection'", naming neither the header nor the real cause. These tests give that
class of failure a direct signal.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
BUNDLED_DIR = PROJECT_ROOT / "src" / "interpretune" / "analysis" / "ops" / "bundled"


@pytest.fixture(scope="module")
def stub_script():
    """Import ``scripts/generate_op_stubs.py`` as a module (it guards its own CLI entrypoint)."""
    spec = importlib.util.spec_from_file_location(
        "_it_generate_op_stubs", PROJECT_ROOT / "scripts" / "generate_op_stubs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestLoadBundledDefinitions:
    def test_collection_headers_are_not_treated_as_ops(self, stub_script, tmp_path):
        """Every bundled family declares one, so treating it as an op fails on the SECOND family loaded."""
        for name in ("a.yaml", "b.yaml"):
            (tmp_path / name).write_text(
                f"collection:\n  name: {name[0]}\n  version: 0.1.0\n"
                f"op_{name[0]}:\n  description: fixture\n  implementation: x.y\n"
            )
        merged = stub_script.load_bundled_definitions(sorted(tmp_path.glob("*.yaml")))
        assert set(merged) == {"op_a", "op_b"}

    def test_genuine_duplicate_op_names_still_raise(self, stub_script, tmp_path):
        """The duplicate check is load-bearing: two families must not silently claim one op name."""
        for name in ("a.yaml", "b.yaml"):
            (tmp_path / name).write_text("same_op:\n  description: fixture\n  implementation: x.y\n")
        with pytest.raises(ValueError, match="Duplicate bundled op definition 'same_op'"):
            stub_script.load_bundled_definitions(sorted(tmp_path.glob("*.yaml")))

    def test_composite_operations_merge_rather_than_collide(self, stub_script, tmp_path):
        (tmp_path / "a.yaml").write_text("composite_operations:\n  comp_a:\n    composition: [x]\n")
        (tmp_path / "b.yaml").write_text("composite_operations:\n  comp_b:\n    composition: [y]\n")
        merged = stub_script.load_bundled_definitions(sorted(tmp_path.glob("*.yaml")))
        assert set(merged["composite_operations"]) == {"comp_a", "comp_b"}

    def test_the_real_bundled_set_loads(self, stub_script):
        """End-to-end over the shipped families: the case the CI stale-stubs check exercises."""
        merged = stub_script.load_bundled_definitions(sorted(BUNDLED_DIR.glob("**/*.yaml")))
        assert "model_fwd" in merged and "collection" not in merged


OP_DEF = {
    "description": "fixture op",
    "implementation": "no_such_module.no_such_fn",
    "input_schema": {"some_input": {"datasets_dtype": "float32", "required": True}},
    "output_schema": {"some_output": {"datasets_dtype": "int64"}},
    "aliases": ["user.repo.an_alias"],
}


class TestUnimportableImplementations:
    """The silent-degradation failure mode, and the hermeticity rule that scopes the fallback (§3.10)."""

    def test_a_bundled_op_with_an_unimportable_impl_raises(self, stub_script):
        """A bundled implementation ships in the wheel, so this is a defect, not a degraded environment.

        The previous behavior printed a message and emitted an untyped stub, which the stale-stubs CI check
        would then accept: the op was still present, so nothing downstream could tell the difference.
        """
        with pytest.raises(RuntimeError, match="Cannot generate a stub for bundled op"):
            stub_script.generate_operation_stub("fixture_op", OP_DEF, {}, require_importable=True)

    def test_a_collection_op_falls_back_to_a_yaml_derived_stub(self, stub_script):
        stub = stub_script.generate_operation_stub("fixture_op", OP_DEF, {}, require_importable=False)
        assert "YAML-derived stub" in stub, "the degradation has to be visible in the artifact itself"
        assert "no_such_module.no_such_fn not importable" in stub

    def test_the_fallback_keeps_the_schema_documentation(self, stub_script):
        """Losing the schema exactly when introspection is unavailable is the worst time to lose it."""
        stub = stub_script.generate_operation_stub("fixture_op", OP_DEF, {}, require_importable=False)
        assert "some_input (float32) (required)" in stub
        assert "some_output (int64)" in stub

    def test_the_fallback_does_not_invent_parameters_from_the_schema(self, stub_script):
        """A schema names an op's DATA contract, not its Python parameters.

        Synthesizing parameters from it would type-check calls the runtime rejects, which is worse than a conservative
        signature.
        """
        stub = stub_script.generate_operation_stub("fixture_op", OP_DEF, {}, require_importable=False)
        signature = stub.split('"""')[0]  # everything up to the docstring; the schema lives inside it
        assert "some_input" not in signature
        assert "batch_idx: int" in signature and "**kwargs" in signature


class TestCollectionStubs:
    def test_namespaced_collections_map_to_importable_module_names(self, stub_script):
        assert stub_script.stub_module_name("speediedan.concept_direction_ops") == "speediedan__concept_direction_ops"
        assert stub_script.stub_module_name("user/my-ops") == "user__my_ops"

    def test_bundled_ops_are_excluded_from_collection_stubs(self, stub_script):
        """The committed stub stays the only home for bundled ops, so the CI check stays offline-derivable."""
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.load_definitions()
        grouped = stub_script.group_definitions_by_collection(DISPATCHER.registered_ops)
        assert "bundled" not in grouped
        for ops in grouped.values():
            assert all(not op_def.source.startswith("bundled") for op_def in ops.values())

    def test_aliases_are_emitted_under_bare_names(self, stub_script):
        """``user.repo.alias = op`` is not valid Python -- it reads as an attribute assignment on ``user``."""
        stub = stub_script.generate_operation_stub("fixture_op", OP_DEF, {}, require_importable=False)
        assert "an_alias = fixture_op" in stub
        assert "user.repo.an_alias" not in stub

    def test_alias_entries_are_not_generated_as_separate_ops(self, stub_script):
        """Bare-name aliasing registers extra ``_op_definitions`` keys for the SAME OpDef.

        Emitting one stub per key would produce duplicate function definitions in the collection stub.
        """
        from interpretune.analysis.ops.compiler.cache_manager import OpDef
        from interpretune.analysis.ops.base import OpSchema

        canonical = OpDef(
            name="u.r.op",
            description="",
            implementation="m.f",
            input_schema=OpSchema(),
            output_schema=OpSchema(),
            source="hub:u.r",
        )
        registry = {"u.r.op": canonical, "op": canonical, "u.r.some_alias": canonical}
        grouped = stub_script.group_definitions_by_collection(registry)
        assert grouped == {"u.r": {"u.r.op": canonical}}


class TestCompositeProtocolResolution:
    """#60: a composite's stub names the protocol it can actually justify.

    Composites had no implementation to introspect, so the generator hardcoded the BASE protocol while
    simple ops were introspected into the richer `Default`. Two better answers exist now: a declared
    `protocol_cls` (#56), or the protocol every constituent agrees on.
    """

    @staticmethod
    def _defs(**protocol_by_op):
        """Constituent definitions; a value of None means the op declares no protocol."""
        return {
            name: ({"protocol_cls": declared} if declared else {"description": "x"})
            for name, declared in protocol_by_op.items()
        }

    def test_declared_protocol_wins_over_inference(self, stub_script):
        """An explicit answer beats an inferred one, even when constituents would agree on another."""
        op_def = {"composition": ["a", "b"], "protocol_cls": "pkg.mod.MyProtocol"}
        defs = self._defs(a=None, b=None)  # both would infer Default
        assert stub_script.composite_protocol_name(op_def, defs) == "MyProtocol"

    def test_constituents_that_all_declare_nothing_infer_default(self, stub_script):
        op_def = {"composition": ["a", "b", "c"]}
        assert stub_script.composite_protocol_name(op_def, self._defs(a=None, b=None, c=None)) == (
            "DefaultAnalysisBatchProtocol"
        )

    def test_constituents_agreeing_on_a_declared_protocol(self, stub_script):
        op_def = {"composition": ["a", "b"]}
        defs = self._defs(a="pkg.mod.Shared", b="other.mod.Shared")
        assert stub_script.composite_protocol_name(op_def, defs) == "Shared"

    def test_disagreeing_constituents_fall_back_to_base(self, stub_script):
        """Naming a protocol the batch may not satisfy is worse than naming the weaker one it does."""
        op_def = {"composition": ["a", "b"]}
        defs = self._defs(a="pkg.mod.One", b=None)  # One vs Default
        assert stub_script.composite_protocol_name(op_def, defs) == "BaseAnalysisBatchProtocol"

    def test_unresolvable_constituent_falls_back_rather_than_guessing_for_the_rest(self, stub_script):
        op_def = {"composition": ["a", "missing"]}
        assert stub_script.composite_protocol_name(op_def, self._defs(a=None)) == "BaseAnalysisBatchProtocol"

    @pytest.mark.parametrize("definitions", [None, {}])
    def test_no_definitions_available_falls_back(self, stub_script, definitions):
        op_def = {"composition": ["a"]}
        expected = "BaseAnalysisBatchProtocol"
        assert stub_script.composite_protocol_name(op_def, definitions) == expected

    def test_dotted_composition_string_is_accepted(self, stub_script):
        """`composition` may arrive as a dotted string rather than a list."""
        op_def = {"composition": "a.b"}
        assert stub_script.composite_protocol_name(op_def, self._defs(a=None, b=None)) == (
            "DefaultAnalysisBatchProtocol"
        )
