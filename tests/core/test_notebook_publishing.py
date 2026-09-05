"""The notebook publishing engine (``interpretune-publish-notebooks``) and the static notebook-form cases.

The engine is parameterized from ``[tool.interpretune.notebooks]`` so an adapter repository publishes its example
notebooks with it rather than with a copy of interpretune's script; these tests drive it against a throwaway
repository and then run the notebook-form cases against interpretune's own published notebooks.
"""

from __future__ import annotations

import json

import pytest
from pathlib import Path


from interpretune.testing.conformance import NotebookFormConformance
from interpretune.utils.notebook_publishing import (
    HASH_FILE,
    PublisherConfig,
    load_notebook,
    main,
    publishable_files,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _notebook(*cells):
    return {"cells": list(cells), "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _code(text, tags=()):
    return {
        "cell_type": "code",
        "metadata": {"tags": list(tags)},
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def _repo(tmp_path: Path, table: str = "") -> Path:
    root = tmp_path / "adapter"
    (root / "examples" / "dev" / "nested").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\n\n[tool.interpretune.notebooks]\n'
        'dev_dir = "examples/dev"\npublish_dir = "examples/publish"\ncolab_repo = "org/adapter"\n' + table
    )
    nb = _notebook(
        _code("from examples.dev.helpers import h\n"),
        _code("secret = 1\n", tags=["remove-cell"]),
        _code("%pip install nothing\nimport json\n"),
    )
    (root / "examples" / "dev" / "nested" / "demo.ipynb").write_text(json.dumps(nb))
    (root / "examples" / "dev" / "helpers.py").write_text("h = 1\n")
    (root / "examples" / "dev" / "__pycache__").mkdir()
    (root / "examples" / "dev" / "__pycache__" / "helpers.cpython-313.pyc").write_bytes(b"\x00")
    return root


class TestEngine:
    def test_publishes_transforms_and_tracks_hashes(self, tmp_path, capsys):
        root = _repo(
            tmp_path, table='[tool.interpretune.notebooks.import_rewrites]\n"examples.dev." = "examples.publish."\n'
        )
        assert main(["--root", str(root)]) == 0
        published = root / "examples" / "publish" / "nested" / "demo.ipynb"
        cells = load_notebook(published)["cells"]
        assert [c["metadata"].get("id") for c in cells[:2]] == ["colab-badge", "install-deps"]
        assert "github/org/adapter/blob/main/examples/publish/nested/demo.ipynb" in "".join(cells[0]["source"])
        # the rewritten cell keeps the list-of-lines form; its final line carries no newline, as the engine has
        # always written it
        assert "".join(cells[2]["source"]) == "from examples.publish.helpers import h"
        assert len(cells) == 4, "the remove-cell tagged cell must be gone"
        assert (root / "examples" / "publish" / "helpers.py").read_text() == "h = 1\n"
        assert not list((root / "examples" / "publish").rglob("*.pyc")), "bytecode caches are never publication inputs"
        hashes = json.loads((root / "examples" / "publish" / HASH_FILE).read_text())
        assert set(hashes) == {"examples/dev/nested/demo.ipynb", "examples/dev/helpers.py"}
        # a second pass is a no-op, and --check-only agrees
        assert main(["--root", str(root), "--check-only"]) == 0
        capsys.readouterr()
        assert main(["--root", str(root)]) == 0
        assert "All files are up to date" in capsys.readouterr().out

    def test_check_only_names_the_stale_file_and_force_republishes_everything(self, tmp_path, capsys):
        root = _repo(tmp_path)
        assert main(["--root", str(root)]) == 0
        (root / "examples" / "dev" / "helpers.py").write_text("h = 2\n")
        assert main(["--root", str(root), "--check-only"]) == 1
        assert "examples/dev/helpers.py" in capsys.readouterr().out
        assert main(["--root", str(root), "--force", "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert out.count("WOULD ") == 2 and "pyc" not in out
        assert (root / "examples" / "publish" / "helpers.py").read_text() == "h = 1\n", "dry run must not write"

    def test_install_cell_and_branch_come_from_the_table(self, tmp_path):
        root = _repo(tmp_path, table='colab_branch = "release"\ninstall_cell = ["%pip install adapter[examples]"]\n')
        cfg = PublisherConfig.from_pyproject(root)
        assert cfg.colab_branch == "release" and cfg.install_cell == ("%pip install adapter[examples]\n",)
        assert main(["--root", str(root)]) == 0
        cells = load_notebook(root / "examples" / "publish" / "nested" / "demo.ipynb")["cells"]
        assert cells[1]["source"] == ["%pip install adapter[examples]\n"]
        assert "/blob/release/" in "".join(cells[0]["source"])

    def test_interpretunes_own_table_reproduces_the_legacy_hash_keys(self):
        cfg = PublisherConfig.from_pyproject(REPO_ROOT)
        assert cfg.dev_dir == REPO_ROOT / "src" / "it_examples" / "notebooks" / "dev"
        some = publishable_files(cfg.dev_dir)[0]
        assert cfg.hash_key(some).startswith("notebooks/dev/"), (
            "the stored .notebook_hashes.json keys must keep resolving"
        )

    def test_a_missing_dev_dir_is_an_error(self, tmp_path, capsys):
        (tmp_path / "pyproject.toml").write_text('[tool.interpretune.notebooks]\ndev_dir = "nope"\n')
        assert main(["--root", str(tmp_path)]) == 1
        assert "dev directory not found" in capsys.readouterr().out


class TestForbiddenImportsPositiveControl:
    def test_a_phantom_forbidden_module_is_refused_rather_than_passing(self, tmp_path):
        """A misspelled pip-form name would otherwise forbid nothing and report success forever."""
        from interpretune.utils.notebook_publishing import main

        root = _repo(tmp_path)
        assert main(["--root", str(root)]) == 0

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
