"""The out-of-tree notebook experiment rails.

Every case runs from a directory that is NOT this repository, because "works from inside the tree" is
the property that was already true and is not what these rails are for. The issue this closes is
explicit that a fix working for one out-of-tree location has not fixed anything, so the fixtures use
`tmp_path` rather than a second checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from interpretune.utils.notebook_experiments import (
    ExperimentHooks,
    ExperimentsConfig,
    default_config_dir,
    default_output_dir,
    resolve_extends_path,
)

TABLE = """\
[project]
name = "someone-elses-experiments"
version = "0.0.1"

[tool.interpretune.experiments]
config_dir = "my_configs"
output_root = "artifacts"
harness_paths = ["shared"]
"""


@pytest.fixture
def out_of_tree(tmp_path: Path) -> Path:
    """A repository that is not this one, at an arbitrary depth."""
    root = tmp_path / "elsewhere" / "project"
    (root / "my_configs").mkdir(parents=True)
    (root / "shared").mkdir()
    (root / "deeply" / "nested" / "notebooks").mkdir(parents=True)
    (root / "pyproject.toml").write_text(TABLE, encoding="utf-8")
    return root


class TestRootDiscovery:
    def test_the_root_is_found_by_walking_up_not_by_assuming_depth(self, out_of_tree):
        """The old rule took `cwd.parents[1]`, which is only right at one depth."""
        deep = out_of_tree / "deeply" / "nested" / "notebooks"
        assert ExperimentsConfig.from_pyproject(_find_root_from(deep)).root == out_of_tree

    def test_a_directory_with_no_pyproject_anywhere_above_is_its_own_root(self, tmp_path):
        """Degenerate but reachable, and it must not raise or climb to the filesystem root."""
        orphan = tmp_path / "no_project"
        orphan.mkdir()
        assert ExperimentsConfig.from_pyproject(_find_root_from(orphan)).root == orphan


def _find_root_from(start: Path) -> Path:
    from interpretune.utils.notebook_publishing import find_root

    return find_root(start)


class TestTheTable:
    def test_declared_keys_are_read_and_resolved_against_the_root(self, out_of_tree):
        config = ExperimentsConfig.from_pyproject(out_of_tree)
        assert config.config_dir == out_of_tree / "my_configs"
        assert config.output_root == out_of_tree / "artifacts"
        assert config.harness_paths == (out_of_tree / "shared",)

    def test_a_repository_with_no_table_gets_working_defaults(self, tmp_path):
        """Writing no table must not be a failure mode; it is the common case for a first experiment."""
        root = tmp_path / "bare"
        root.mkdir()
        (root / "pyproject.toml").write_text('[project]\nname = "bare"\nversion = "0"\n', encoding="utf-8")
        config = ExperimentsConfig.from_pyproject(root)
        assert config.config_dir is None and config.output_root is None and config.harness_paths == ()

    def test_unset_keys_fall_back_beside_the_notebook_rather_than_to_a_fixed_location(self, tmp_path):
        """The previous launcher sent one experiment's runs to a hard-coded /tmp root."""
        root = tmp_path / "bare"
        root.mkdir()
        (root / "pyproject.toml").write_text('[project]\nname = "bare"\nversion = "0"\n', encoding="utf-8")
        config = ExperimentsConfig.from_pyproject(root)
        notebook = tmp_path / "somewhere" / "run.ipynb"
        assert default_config_dir(notebook, config) == notebook.parent / "configs"
        assert default_output_dir(notebook, config) == notebook.parent / "generated_experiments"

    def test_declared_keys_win_over_the_notebook_relative_defaults(self, out_of_tree):
        config = ExperimentsConfig.from_pyproject(out_of_tree)
        notebook = out_of_tree / "deeply" / "nested" / "notebooks" / "run.ipynb"
        assert default_config_dir(notebook, config) == out_of_tree / "my_configs"
        assert default_output_dir(notebook, config) == out_of_tree / "artifacts"


class TestExtendsResolution:
    """The blocker: an out-of-tree config naming base configs that ship inside an installed package."""

    def test_a_package_resource_resolves_from_outside_the_tree(self, tmp_path):
        config = tmp_path / "mine.yaml"
        resolved = resolve_extends_path(config, "it_examples.experiments.notebook:configs/base.yaml")
        assert resolved.is_file() and resolved.name == "base.yaml"

    def test_a_package_qualified_subpackage_form_also_resolves(self, tmp_path):
        resolved = resolve_extends_path(tmp_path / "mine.yaml", "it_examples.experiments.notebook.configs:base.yaml")
        assert resolved.is_file()

    def test_relative_paths_still_resolve_against_the_config_that_named_them(self, tmp_path):
        """Unchanged behaviour, asserted because every in-tree config depends on it."""
        (tmp_path / "configs").mkdir()
        base = tmp_path / "configs" / "base.yaml"
        base.write_text("A: 1\n", encoding="utf-8")
        assert resolve_extends_path(tmp_path / "child.yaml", "configs/base.yaml") == base.resolve()

    def test_an_absolute_path_is_used_as_given(self, tmp_path):
        base = tmp_path / "abs.yaml"
        base.write_text("A: 1\n", encoding="utf-8")
        assert resolve_extends_path(tmp_path / "child.yaml", str(base)) == base

    def test_a_missing_package_says_so_and_says_what_to_use_instead(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="which is not installed"):
            resolve_extends_path(tmp_path / "mine.yaml", "no_such_package_anywhere:base.yaml")

    def test_a_missing_resource_inside_a_real_package_is_distinguished_from_a_missing_package(self, tmp_path):
        """Two different mistakes with two different fixes, so they must not share one message."""
        with pytest.raises(FileNotFoundError, match="which does not exist"):
            resolve_extends_path(tmp_path / "mine.yaml", "it_examples.experiments.notebook:configs/nope.yaml")

    def test_a_resource_inside_a_ZIPPED_package_materializes_to_a_real_readable_file(self, tmp_path):
        """The case that fails silently without `as_file`.

        For an unpacked install the traversable is already a `Path`, so `str()` of it happens to work and
        every ordinary test passes. Inside a zipped wheel it is not a filesystem object at all: the string
        is path-shaped, nothing can open it, and the failure surfaces later as an unrelated read error in
        the YAML loader rather than here. This pins that a real, readable file comes back either way.
        """
        import sys
        import zipfile

        archive = tmp_path / "zipped_pkg.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("zipped_experiment_pkg/__init__.py", "")
            zf.writestr("zipped_experiment_pkg/configs/base.yaml", "ZIPPED: true\n")
        sys.path.insert(0, str(archive))
        try:
            resolved = resolve_extends_path(tmp_path / "mine.yaml", "zipped_experiment_pkg:configs/base.yaml")
            assert resolved.is_file(), "a zipped resource must be materialized, not string-cast"
            assert resolved.read_text(encoding="utf-8") == "ZIPPED: true\n"
            # Say WHICH branch ran. Everything installed here is unpacked, so a test that only checked a
            # file came back would pass through the fast path and assert nothing about materialization.
            assert archive not in resolved.parents, "resolved inside the archive: materialization was skipped"
        finally:
            sys.path.remove(str(archive))
            sys.modules.pop("zipped_experiment_pkg", None)

    def test_two_same_named_resources_in_one_package_do_not_collide(self, tmp_path):
        """Fails against a basename-keyed cache, which is the shape a config tree actually has.

        `configs/base.yaml` and `configs/gemma/base.yaml` is not a contrived pair; it is how these trees are organized.
        Keyed by basename, the second materialization overwrites what the first call's cached path points at, so the
        first EXTENDS silently reads the second resource's content. Nothing raises and both reads succeed, which is why
        it needs a test rather than a guard.
        """
        import sys
        import zipfile

        archive = tmp_path / "collide.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("collide_pkg/__init__.py", "")
            zf.writestr("collide_pkg/configs/base.yaml", "WHICH: outer\n")
            zf.writestr("collide_pkg/configs/gemma/base.yaml", "WHICH: inner\n")
        sys.path.insert(0, str(archive))
        try:
            outer = resolve_extends_path(tmp_path / "a.yaml", "collide_pkg:configs/base.yaml")
            inner = resolve_extends_path(tmp_path / "b.yaml", "collide_pkg:configs/gemma/base.yaml")
            assert outer != inner, "same-named resources must not share a destination"
            assert outer.read_text(encoding="utf-8") == "WHICH: outer\n"
            assert inner.read_text(encoding="utf-8") == "WHICH: inner\n"
        finally:
            sys.path.remove(str(archive))
            for name in [m for m in sys.modules if m.startswith("collide_pkg")]:
                del sys.modules[name]

    def test_a_placeholder_prefix_is_refused_with_the_spelling_to_use(self):
        """`<shared>` was considered and rejected; falling through to "missing file" would hide that.

        Naming one package as special is the assumption the component rails removed, so the refusal says so rather than
        leaving someone to conclude the feature is broken.
        """
        with pytest.raises(ValueError, match="looks like a placeholder"):
            resolve_extends_path(Path("/tmp/x.yaml"), "<shared>/base.yaml")

    def test_a_path_containing_a_colon_is_still_a_path_when_it_exists(self, tmp_path):
        """A colon is legal in a filename, so existence decides before the separator does."""
        odd = tmp_path / "weird:name.yaml"
        odd.write_text("A: 1\n", encoding="utf-8")
        assert resolve_extends_path(tmp_path / "child.yaml", str(odd)) == odd


class TestExperimentHooks:
    def test_an_unsupplied_hook_names_itself_and_what_to_do(self):
        with pytest.raises(ValueError, match="needs the 'tensor_fingerprint' hook"):
            ExperimentHooks().require("tensor_fingerprint")

    def test_a_supplied_hook_is_returned(self):
        hooks = ExperimentHooks(tensor_fingerprint=lambda v: "fp")
        assert hooks.require("tensor_fingerprint")(object()) == "fp"

    def test_the_shared_harness_imports_without_any_experiment(self):
        """The defect this inverts: importing the SHARED rails required one specific experiment."""
        import sys

        for name in [m for m in sys.modules if "concept_direction" in m]:
            del sys.modules[name]
        from it_examples.experiments.notebook import nb_harness_utils  # noqa: F401

        assert not any("concept_direction" in m for m in sys.modules)

    def test_the_experiment_registers_its_hooks_on_import(self):
        import it_examples.experiments.notebook.concept_direction  # noqa: F401
        from it_examples.experiments.notebook import nb_harness_utils

        for name in (
            "build_classification_prompt_text",
            "resolve_artifact_output_dir",
            "save_preserved_intervention_artifacts",
            "tensor_fingerprint",
        ):
            assert nb_harness_utils._EXPERIMENT_HOOKS.require(name) is not None
