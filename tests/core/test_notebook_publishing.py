"""Interpretune's notebook reader (``interpretune.utils.notebook_config``), its agreement with nbpress, and the
static notebook-form cases.

nbpress (a git-deps pin) publishes and renders the notebooks and is tested in its own repository. The published library
keeps a small reader of nbpress's configuration and hashes so it does not depend on nbpress; these tests hold the two to
the same answers, keep the three places nbpress is pinned at one commit, and run the notebook-form cases against
interpretune's own published notebooks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from nbpress import publish as nbpress_publish

from interpretune.testing.conformance import NotebookFormConformance
from interpretune.utils.notebook_config import (
    NotebookConfigError,
    PublisherConfig,
    changed_files,
    compute_file_hash,
    load_notebook_hashes,
    publishable_files,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_TABLE = (
    'dev_dir = "examples/dev"\npublish_dir = "examples/publish"\ncolab_repo = "org/adapter"\n'
    'install_cell = ["# %pip install adapter"]\n'
)


def _notebook(*cells):
    return {"cells": list(cells), "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True), "outputs": []}


def _repo(tmp_path: Path, table: str = "", *, header: str = "[tool.nbpress]") -> Path:
    root = tmp_path / "adapter"
    (root / "examples" / "dev").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "0"\n\n{header}\n' + BASE_TABLE + table, encoding="utf-8"
    )
    (root / "examples" / "dev" / "demo.ipynb").write_text(
        json.dumps(_notebook(_code("import json\n"))), encoding="utf-8"
    )
    return root


class TestReaderAgreesWithNbpress:
    """A reader that hashed or keyed files differently from nbpress would report every published notebook stale."""

    def test_hashes_agree_on_text_crlf_and_binary_files(self, tmp_path):
        samples = {
            "lf.py": b"a = 1\nb = 2\n",
            "crlf.py": b"a = 1\r\nb = 2\r\n",
            "img.png": b"\x89PNG\r\n\x1a\n\x00binary",
            "latin1.txt": "caf\xe9\r\n".encode("latin-1"),
        }
        for name, data in samples.items():
            (tmp_path / name).write_bytes(data)
            assert compute_file_hash(tmp_path / name) == nbpress_publish.compute_file_hash(tmp_path / name), name
        assert compute_file_hash(tmp_path / "lf.py") == compute_file_hash(tmp_path / "crlf.py")

    def test_both_read_interpretunes_table_to_the_same_inputs_and_keys(self):
        ours = PublisherConfig.from_pyproject(REPO_ROOT)
        theirs = nbpress_publish.PublisherConfig.from_pyproject(REPO_ROOT)
        assert (ours.dev_dir, ours.publish_dir, ours.colab_repo, ours.colab_branch) == (
            theirs.dev_dir,
            theirs.publish_dir,
            theirs.colab_repo,
            theirs.colab_branch,
        )
        files = publishable_files(ours.dev_dir)
        assert files == nbpress_publish.publishable_files(theirs.dev_dir)
        assert [ours.hash_key(p) for p in files] == [theirs.hash_key(p) for p in files]
        assert ours.hash_key(files[0]).startswith("notebooks/dev/"), "the stored hash keys must keep resolving"

    def test_a_file_nbpress_just_published_reads_as_current_and_an_edit_as_stale(self, tmp_path):
        root = _repo(tmp_path)
        assert nbpress_publish.main(["--root", str(root)]) == 0
        cfg = PublisherConfig.from_pyproject(root)
        assert changed_files(cfg, load_notebook_hashes(cfg.publish_dir)) == []
        (cfg.dev_dir / "demo.ipynb").write_text(json.dumps(_notebook(_code("import os\n"))), encoding="utf-8")
        assert changed_files(cfg, load_notebook_hashes(cfg.publish_dir)) == [cfg.dev_dir / "demo.ipynb"]


class TestReaderTable:
    @pytest.mark.parametrize("legacy", ["[tool.notebook-publishing]", "[tool.interpretune.notebooks]"])
    def test_each_earlier_table_name_is_read_identically(self, tmp_path, legacy):
        new = PublisherConfig.from_pyproject(_repo(tmp_path / "new"))
        old = PublisherConfig.from_pyproject(_repo(tmp_path / "old", header=legacy))
        assert (new.dev_dir.name, new.colab_repo) == (old.dev_dir.name, old.colab_repo)

    def test_two_tables_are_refused_rather_than_merged(self, tmp_path):
        root = _repo(tmp_path, table="\n[tool.notebook-publishing]\n" + BASE_TABLE)
        with pytest.raises(NotebookConfigError, match="keep one"):
            PublisherConfig.from_pyproject(root)

    def test_no_table_and_a_missing_key_are_refused_by_name(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
        with pytest.raises(NotebookConfigError, match=r"no \[tool.nbpress\] table"):
            PublisherConfig.from_pyproject(tmp_path)
        (tmp_path / "pyproject.toml").write_text('[tool.nbpress]\ndev_dir = "d"\npublish_dir = "p"\n', encoding="utf-8")
        with pytest.raises(NotebookConfigError, match="colab_repo"):
            PublisherConfig.from_pyproject(tmp_path)


class TestNbpressPin:
    """Nbpress is pinned in three places that install it separately; a bump that misses one runs two versions."""

    @staticmethod
    def _commits() -> dict[str, list[str]]:
        url = r"nbpress @ git\+https://github\.com/speediedan/nbpress@([0-9a-f]{40})"
        precommit = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        return {
            "pyproject.toml git-deps": re.findall(url, (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")),
            "requirements/docs.txt": re.findall(
                url, (REPO_ROOT / "requirements" / "docs.txt").read_text(encoding="utf-8")
            ),
            ".pre-commit-config.yaml rev": re.findall(
                r"repo: https://github\.com/speediedan/nbpress\n\s+rev: ([0-9a-f]{40})\b", precommit
            ),
        }

    def test_each_place_pins_nbpress_once_and_all_to_one_commit(self):
        commits = self._commits()
        assert all(len(found) == 1 for found in commits.values()), commits
        assert len({found[0] for found in commits.values()}) == 1, commits


class TestForbiddenImportsPositiveControl:
    def test_a_phantom_forbidden_module_is_refused_rather_than_passing(self, tmp_path):
        """A misspelled pip-form name would otherwise forbid nothing and report success forever."""
        root = _repo(tmp_path)
        assert nbpress_publish.main(["--root", str(root)]) == 0

        class Probe(NotebookFormConformance):
            forbidden_imports = ("no_such_module_anywhere",)

        published = sorted((root / "examples" / "publish").rglob("*.ipynb"))
        with pytest.raises(AssertionError, match="vacuous.*no_such_module_anywhere"):
            Probe().test_published_notebooks_avoid_the_forbidden_imports(published)

        class Real(NotebookFormConformance):
            forbidden_imports = ("json",)  # the demo notebook imports json, so a real name is caught

        with pytest.raises(AssertionError, match="import the pip form directly"):
            Real().test_published_notebooks_avoid_the_forbidden_imports(published)


class TestInterpretuneNotebookForm(NotebookFormConformance):
    """The notebook-form cases, run against interpretune's own published notebooks."""

    root = REPO_ROOT
    # `exp` is the concept-direction component's package: the latent-dynamics notebook imports it from the snapshot it
    # fetches, after the reader opts into that component's code, so no environment has it installed
    ignored_imports = ("exp",)
