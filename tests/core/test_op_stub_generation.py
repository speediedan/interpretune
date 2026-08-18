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
