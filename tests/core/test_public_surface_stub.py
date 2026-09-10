# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`src/interpretune/__init__.pyi` must cover the whole public surface, not merely agree with its generator.

CI regenerates the stub and fails if the result differs from the committed one. That verifies the stub agrees with its
GENERATOR; it cannot notice a generator that emits two thirds of the API, which is the state this file was added for (23
of 75 names). These check the property the stale-stubs job does not.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

import interpretune

PROJECT_ROOT = Path(__file__).parent.parent.parent
STUB_PATH = PROJECT_ROOT / "src" / "interpretune" / "__init__.pyi"
INIT_PATH = PROJECT_ROOT / "src" / "interpretune" / "__init__.py"


def names_reexported_by(stub_source: str) -> set[str]:
    """Every name the stub RE-EXPORTS, by the rule a type checker actually applies.

    Not the same as "bound", and the difference is the whole point. In a stub, an imported name is
    re-exported only under the redundant ``X as X`` form; imported plainly it is visible inside the stub
    and invisible to anyone importing it from the module. An earlier version of this helper counted any
    import and so reported full coverage while pyright still failed on ten names -- a check answering a
    narrower question than the one being asked of it.
    """
    exported: set[str] = set()
    for node in ast.parse(stub_source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exported.add(node.name)
        elif isinstance(node, ast.Assign):
            exported.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            exported.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            exported.update(a.asname for a in node.names if a.asname and a.asname == a.name)
    return exported


def _generator():
    spec = importlib.util.spec_from_file_location("_gen_op_stubs", PROJECT_ROOT / "scripts" / "generate_op_stubs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheStubCoversThePublicSurface:
    def test_every_exported_name_is_bound_by_the_stub(self):
        """The property itself.

        A `.pyi` SHADOWS the module it names rather than adding to it, so a name missing here is invisible to a
        type checker even though it exists and is exported. The consumer that reaches is our own adapter
        repository, where the natural response to an error on real public API is a permanent `# type: ignore`.
        """
        missing = sorted(set(interpretune.__all__) - names_reexported_by(STUB_PATH.read_text()))
        assert not missing, (
            f"{len(missing)} of {len(interpretune.__all__)} public names are absent from __init__.pyi and so are "
            f"invisible to a type checker: {missing}. Regenerate with `python scripts/generate_op_stubs.py`."
        )

    def test_the_check_notices_a_planted_omission(self):
        """The positive control, and the reason this file is worth having.

        The assertion above is a check for an ABSENCE, which passes when it never really ran. Deleting one re-export
        from the stub source has to make it report that name, or it is measuring nothing.
        """
        source = STUB_PATH.read_text()
        victim = "ITSession"
        assert victim in names_reexported_by(source), f"{victim} should be re-exported before planting the defect"
        planted = "\n".join(line for line in source.splitlines() if f"{victim} as {victim}" not in line)
        missing = sorted(set(interpretune.__all__) - names_reexported_by(planted))
        assert victim in missing, (
            "removing a re-export did not make the coverage check report it, so the check cannot fail and is "
            "not evidence of anything"
        )

    def test_the_generator_refuses_a_name_it_cannot_place(self, tmp_path):
        """An unplaceable name is refused rather than skipped.

        Skipping would reproduce the original defect exactly: a name silently absent from the generated file,
        invisible in the output and invisible in review.
        """
        gen = _generator()
        doctored = tmp_path / "__init__.py"
        doctored.write_text(INIT_PATH.read_text().replace("__all__ = [", '__all__ = [\n    "a_name_from_nowhere",', 1))
        with pytest.raises(RuntimeError, match="a_name_from_nowhere"):
            gen.public_surface_imports(doctored)

    def test_the_surface_is_derived_without_importing_interpretune(self):
        """Derivable from source alone, which is what keeps the stale-stubs CI job hermetic.

        Resolving the lazy names would import every optional framework as a side effect of generating stubs,
        so the generator reads `__all__` and `_LAZY_MODULE_ATTRS` statically instead.
        """
        gen = _generator()
        lines = gen.public_surface_imports(INIT_PATH)
        emitted = names_reexported_by("\n".join(line for line in lines if not line.startswith("#")))
        assert set(interpretune.__all__) <= emitted
