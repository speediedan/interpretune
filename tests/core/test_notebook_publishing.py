"""The notebook publishing engine (``interpretune-publish-notebooks``) and the static notebook-form cases.

The engine is parameterized from ``[tool.notebook-publishing]`` so another repository publishes its notebooks with it
rather than with a copy of interpretune's script; these tests drive it against a throwaway repository and then run the
notebook-form cases against interpretune's own published notebooks.
"""

from __future__ import annotations

import json

import pytest
from pathlib import Path


from interpretune.testing.conformance import NotebookFormConformance
from interpretune.utils.notebook_publishing import (
    HASH_FILE,
    NotebookConfigError,
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


BASE_TABLE = (
    'dev_dir = "examples/dev"\npublish_dir = "examples/publish"\ncolab_repo = "org/adapter"\n'
    'install_cell = ["# %pip install adapter"]\n'
)


def _repo(
    tmp_path: Path, table: str = "", *, header: str = "[tool.notebook-publishing]", base: str = BASE_TABLE
) -> Path:
    root = tmp_path / "adapter"
    (root / "examples" / "dev" / "nested").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "0"\n\n{header}\n' + base + table)
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
            tmp_path, table='[tool.notebook-publishing.import_rewrites]\n"examples.dev." = "examples.publish."\n'
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
        base = BASE_TABLE.replace('"# %pip install adapter"', '"%pip install adapter[examples]"')
        root = _repo(tmp_path, table='colab_branch = "release"\n', base=base)
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

    def test_hashes_ignore_line_ending_rewrites(self, tmp_path):
        """A CRLF checkout of the same file must hash as the LF original, or Windows reports everything stale."""
        from interpretune.utils.notebook_publishing import compute_file_hash

        lf = tmp_path / "lf.py"
        crlf = tmp_path / "crlf.py"
        lf.write_bytes(b"a = 1\nb = 2\n")
        crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
        assert compute_file_hash(lf) == compute_file_hash(crlf)
        other = tmp_path / "other.py"
        other.write_bytes(b"a = 1\nb = 3\n")
        assert compute_file_hash(lf) != compute_file_hash(other)
        # a binary file with an incidental CRLF byte pair hashes raw: a checkout never rewrites it
        png = tmp_path / "img.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n\x00binary")
        assert compute_file_hash(png) == __import__("hashlib").sha256(png.read_bytes()).hexdigest()

    def test_a_missing_dev_dir_is_an_error(self, tmp_path, capsys):
        root = _repo(tmp_path, base=BASE_TABLE.replace('"examples/dev"', '"nope"'))
        assert main(["--root", str(root)]) == 1
        assert "dev directory not found" in capsys.readouterr().out


class TestTableContract:
    """The engine holds no repository defaults, so the table is refused by name rather than half-read."""

    def test_the_legacy_table_name_is_read_identically(self, tmp_path):
        new = PublisherConfig.from_pyproject(_repo(tmp_path / "new"))
        old = PublisherConfig.from_pyproject(_repo(tmp_path / "old", header="[tool.interpretune.notebooks]"))
        assert (new.dev_dir.name, new.colab_repo, new.install_cell) == (
            old.dev_dir.name,
            old.colab_repo,
            old.install_cell,
        )

    def test_both_tables_are_refused_rather_than_merged(self, tmp_path, capsys):
        root = _repo(tmp_path, table="\n[tool.interpretune.notebooks]\n" + BASE_TABLE)
        with pytest.raises(NotebookConfigError, match="both"):
            PublisherConfig.from_pyproject(root)
        assert main(["--root", str(root)]) == 1
        assert "legacy [tool.interpretune.notebooks]" in capsys.readouterr().out

    @pytest.mark.parametrize("dropped", ["dev_dir", "publish_dir", "colab_repo", "install_cell"])
    def test_a_missing_required_key_is_refused_by_name(self, tmp_path, dropped):
        base = "".join(line + "\n" for line in BASE_TABLE.splitlines() if not line.startswith(dropped))
        with pytest.raises(NotebookConfigError, match=dropped):
            PublisherConfig.from_pyproject(_repo(tmp_path, base=base))

    def test_an_unknown_key_is_refused_by_name(self, tmp_path):
        """A misspelled optional key would otherwise fall back to its default without a word."""
        with pytest.raises(NotebookConfigError, match="colab_brnach"):
            PublisherConfig.from_pyproject(_repo(tmp_path, table='colab_brnach = "release"\n'))

    def test_no_table_is_refused(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
        with pytest.raises(NotebookConfigError, match="no \\[tool.notebook-publishing\\] table"):
            PublisherConfig.from_pyproject(tmp_path)

    def test_hash_keys_are_relative_to_the_configured_base(self, tmp_path):
        cfg = PublisherConfig.from_pyproject(_repo(tmp_path, table='hash_key_base = "examples"\n'))
        assert cfg.hash_key(cfg.dev_dir / "nested" / "demo.ipynb") == "dev/nested/demo.ipynb"


class TestParametersCell:
    """A papermill ``parameters`` cell must survive publishing and output stripping, so a test or smoke run can
    shrink a notebook by injecting smaller values."""

    def test_publishing_and_stripping_keep_the_parameters_cell(self, tmp_path):
        import importlib.util

        root = _repo(tmp_path)
        params = _code("model = 'tiny'\nepochs = 1\n", tags=["parameters"])
        params["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["x\n"]}]
        (root / "examples" / "dev" / "nested" / "demo.ipynb").write_text(json.dumps(_notebook(params)))
        assert main(["--root", str(root)]) == 0
        published = root / "examples" / "publish" / "nested" / "demo.ipynb"
        cell = load_notebook(published)["cells"][2]
        assert cell["metadata"]["tags"] == ["parameters"]
        assert "".join(cell["source"]).rstrip("\n") == "model = 'tiny'\nepochs = 1"

        spec = importlib.util.spec_from_file_location(
            "_nbstripout_hook", REPO_ROOT / "scripts" / "run_nbstripout_hook.py"
        )
        assert spec is not None and spec.loader is not None
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)
        hook.main([str(published)])
        stripped = load_notebook(published)["cells"][2]
        assert stripped["metadata"]["tags"] == ["parameters"] and stripped["source"] == cell["source"]


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
