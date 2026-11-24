"""The ``helpers`` kind: a component's declared helper modules, fetched as payloads and imported through the
gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

import interpretune.hub as hub
from interpretune.hub import helpers as helpers_mod
from interpretune.hub.components import ComponentFileNotCachedError, declared_component_payloads, local_publish
from interpretune.hub.manifest import ComponentManifestError, validate_component_manifest

REPO = "org/plot-helpers"
MODULES = {"plots": "plots.py", "shared": "lib/shared.py"}


def _manifest(**overrides) -> dict:
    return {"it_schema_version": 1, "kinds": ["helpers"], "helpers": {"modules": MODULES}, **overrides}


def _component(root: Path, *, value: int = 1) -> Path:
    (root / "lib").mkdir(parents=True)
    (root / "it_component.yaml").write_text(yaml.safe_dump(_manifest()), encoding="utf-8")
    (root / "plots.py").write_text("from .shared import VALUE\n\n\ndef value():\n    return VALUE\n", encoding="utf-8")
    (root / "lib" / "shared.py").write_text(f"VALUE = {value}\n", encoding="utf-8")
    (root / "undeclared.py").write_text("RAN = True\n", encoding="utf-8")
    return root


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A private components cache, with every helper module and package registration undone afterwards."""
    before = dict(helpers_mod._PACKAGES)
    yield tmp_path / "cache"
    helpers_mod._PACKAGES.clear()
    helpers_mod._PACKAGES.update(before)
    for name in [m for m in sys.modules if m.partition(".")[0] == helpers_mod.HUB_HELPERS_PACKAGE]:
        del sys.modules[name]


@pytest.fixture
def trusted(monkeypatch):
    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "1")


@pytest.fixture
def published(tmp_path, cache) -> str:
    return local_publish(_component(tmp_path / "src"), REPO, cache_dir=cache)


class TestSchema:
    def test_the_kind_is_accepted_alone_and_beside_others(self):
        validate_component_manifest(_manifest(), source="t")
        both = _manifest(kinds=["ops", "helpers"], ops={"files": ["ops.yaml"]})
        validate_component_manifest(both, source="t")

    @pytest.mark.parametrize("block", [None, {}, {"modules": {}}, {"modules": ["plots.py"]}])
    def test_a_missing_or_malformed_block_is_refused(self, block):
        with pytest.raises(ComponentManifestError, match="non-empty `helpers.modules` mapping"):
            validate_component_manifest(_manifest(helpers=block), source="t")

    @pytest.mark.parametrize("name", ["not-a-name", "a.b", "class", "1st"])
    def test_a_name_that_cannot_be_imported_is_refused(self, name):
        with pytest.raises(ComponentManifestError, match="must be a Python identifier"):
            validate_component_manifest(_manifest(helpers={"modules": {name: "x.py"}}), source="t")

    @pytest.mark.parametrize("rel", ["../x.py", "/x.py", "..\\x.py", "C:\\x.py", "x.ipynb", "lib", 3])
    def test_a_path_outside_the_component_or_not_python_is_refused(self, rel):
        with pytest.raises(ComponentManifestError, match="repo-relative .py path inside the component"):
            validate_component_manifest(_manifest(helpers={"modules": {"x": rel}}), source="t")

    def test_an_orphan_block_is_refused(self):
        orphan = _manifest(kinds=["ops"], ops={"files": ["ops.yaml"]})
        with pytest.raises(ComponentManifestError, match="without kind `helpers` is an orphan"):
            validate_component_manifest(orphan, source="t")


class TestPublishAndPull:
    def test_the_declared_files_are_payloads(self):
        assert declared_component_payloads(_manifest()) == ["plots.py", "lib/shared.py"]

    def test_publish_stages_the_declared_files_only(self, cache, published):
        snapshot = cache / "models--org--plot-helpers" / "snapshots" / published
        built = sorted(p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file())
        assert built == ["README.md", "it_component.yaml", "lib/shared.py", "plots.py"]
        card = (snapshot / "README.md").read_text(encoding="utf-8")
        assert "## Helper modules" in card and "- `shared`: `lib/shared.py`" in card

    def test_publish_refuses_a_declared_file_that_is_absent(self, tmp_path, cache):
        root = _component(tmp_path / "src")
        (root / "plots.py").unlink()
        with pytest.raises(FileNotFoundError, match="helper module 'plots' at 'plots.py'"):
            local_publish(root, REPO, cache_dir=cache)


class TestLoad:
    def test_a_declared_module_loads_and_imports_its_sibling_relatively(self, cache, published, trusted):
        path_before = list(sys.path)
        plots = hub.load_helpers(REPO, "plots", cache_dir=cache)
        assert plots.value() == 1
        assert published in plots.__name__
        assert hub.load_helpers(REPO, "plots", cache_dir=cache) is plots
        assert sys.path == path_before

    def test_two_revisions_load_side_by_side(self, tmp_path, cache, published, trusted):
        newer = local_publish(_component(tmp_path / "src2", value=2), REPO, cache_dir=cache)
        assert newer != published
        old = hub.load_helpers(REPO, "plots", revision=published, cache_dir=cache)
        new = hub.load_helpers(REPO, "plots", revision=newer, cache_dir=cache)
        assert (old.value(), new.value()) == (1, 2)

    def test_untrusted_is_refused_before_anything_executes(self, cache, published, monkeypatch):
        monkeypatch.delenv("IT_TRUST_REMOTE_CODE", raising=False)
        with pytest.raises(hub.RemoteCodeNotTrustedError, match="helper module 'plots.py'"):
            hub.load_helpers(REPO, "plots", cache_dir=cache)
        assert not any(m.startswith(helpers_mod.HUB_HELPERS_PACKAGE) for m in sys.modules)

    def test_a_relative_import_is_gated_too(self, cache, published, monkeypatch, trusted):
        shared = hub.load_helpers(REPO, "shared", cache_dir=cache)
        monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "0")
        package = shared.__name__.rpartition(".")[0]
        with pytest.raises(hub.RemoteCodeNotTrustedError):
            __import__(f"{package}.plots")
        with pytest.raises(hub.RemoteCodeNotTrustedError):  # an already imported module, after an opt-out
            hub.load_helpers(REPO, "shared", cache_dir=cache)

    def test_an_undeclared_name_is_refused_listing_the_declared_ones(self, cache, published, trusted):
        with pytest.raises(hub.HelperModuleError, match="no helper module 'undeclared'.*plots, shared"):
            hub.load_helpers(REPO, "undeclared", cache_dir=cache)

    def test_an_undeclared_file_is_not_importable_from_the_package(self, cache, published, trusted):
        package = hub.load_helpers(REPO, "plots", cache_dir=cache).__name__.rpartition(".")[0]
        snapshot = cache / "models--org--plot-helpers" / "snapshots" / published
        (snapshot / "undeclared.py").write_text("RAN = True\n", encoding="utf-8")
        with pytest.raises(ModuleNotFoundError):
            __import__(f"{package}.undeclared")

    def test_a_declared_file_missing_from_the_cache_names_the_pull(self, cache, published, trusted):
        (cache / "models--org--plot-helpers" / "snapshots" / published / "plots.py").unlink()
        with pytest.raises(ComponentFileNotCachedError, match=r"interpretune.hub.pull\('org/plot-helpers', revision="):
            hub.load_helpers(REPO, "plots", cache_dir=cache)

    def test_a_component_without_the_kind_is_refused_by_name(self, tmp_path, cache, trusted):
        root = tmp_path / "ops"
        root.mkdir()
        (root / "it_component.yaml").write_text(
            yaml.safe_dump({"it_schema_version": 1, "kinds": ["ops"], "ops": {"files": ["ops.yaml"]}}), encoding="utf-8"
        )
        (root / "ops.yaml").write_text("{}\n", encoding="utf-8")
        local_publish(root, "org/ops-only", cache_dir=cache)
        with pytest.raises(hub.HelperModuleError, match="no kind `helpers`"):
            hub.load_helpers("org/ops-only", "plots", cache_dir=cache)

    def test_repositories_whose_names_differ_only_in_punctuation_are_refused(self, tmp_path, cache, published, trusted):
        hub.load_helpers(REPO, "plots", cache_dir=cache)
        twin = local_publish(_component(tmp_path / "twin"), "org/plot_helpers", cache_dir=cache)
        helpers_mod._PACKAGES[f"it_hub_helpers.org__plot_helpers.{twin}"] = helpers_mod._PACKAGES[
            f"it_hub_helpers.org__plot_helpers.{published}"
        ]
        with pytest.raises(hub.HelperModuleError, match="differ only in punctuation"):
            hub.load_helpers("org/plot_helpers", "plots", cache_dir=cache)

    def test_a_module_that_fails_to_import_is_not_left_half_loaded(self, tmp_path, cache, trusted):
        root = _component(tmp_path / "src")
        (root / "plots.py").write_text("raise RuntimeError('broken')\n", encoding="utf-8")
        local_publish(root, REPO, cache_dir=cache)
        for _ in range(2):
            with pytest.raises(RuntimeError, match="broken"):
                hub.load_helpers(REPO, "plots", cache_dir=cache)
