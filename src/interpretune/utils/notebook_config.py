"""Read a repository's notebook layout and publication hashes, as nbpress writes them.

Publishing and rendering live in nbpress (https://github.com/speediedan/nbpress), a development dependency. Two
parts of the published library read what it writes, so the reader stays here and the library does not depend on
it: the notebook-form conformance cases (``interpretune.testing.conformance.notebooks``) and the experiment harness
(``interpretune.harness.experiments``, which needs only :func:`find_root` and :func:`load_toml`).

The layout comes from the ``[tool.nbpress]`` table of the repository's ``pyproject.toml``, or one of its earlier names,
``[tool.notebook-publishing]`` and ``[tool.interpretune.notebooks]``; more than one is refused, as nbpress refuses it.
Only the keys read here are checked. nbpress refuses a key no nbpress tool reads, and a repository's own pre-commit
and docs checks run nbpress, so a second copy of that list here would only drift from it.

The hashing must agree with nbpress byte for byte, or every published notebook reads as stale;
``tests/core/test_notebook_publishing.py`` compares the two.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HASH_FILE = ".notebook_hashes.json"
TABLE = ("tool", "nbpress")
LEGACY_TABLES = (("tool", "notebook-publishing"), ("tool", "interpretune", "notebooks"))
REQUIRED_KEYS = ("dev_dir", "publish_dir", "colab_repo")
EXCLUDED_PARTS = ("__pycache__",)
EXCLUDED_SUFFIXES = (".pyc",)


class NotebookConfigError(ValueError):
    """The repository's notebook table is missing, ambiguous, or lacks a key this reader needs."""


def _table_name(path: tuple[str, ...]) -> str:
    return "[" + ".".join(path) + "]"


def _lookup(data: Any, path: tuple[str, ...]) -> dict[str, Any] | None:
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return dict(data) if isinstance(data, dict) else None


def load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file."""
    import tomllib

    with open(path, "rb") as fh:
        return tomllib.load(fh)


def find_root(start: Path | None = None) -> Path:
    """The nearest ancestor of ``start`` (default: cwd) holding a ``pyproject.toml``, or ``start`` itself."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def load_table(root: Path) -> dict[str, Any]:
    """The notebook table from ``root/pyproject.toml``, refusing a missing pyproject or table, or more than one."""
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        raise NotebookConfigError(f"no pyproject.toml at {root}; the notebook layout is read from {_table_name(TABLE)}")
    data = load_toml(pyproject)
    found = [(path, table) for path in (TABLE, *LEGACY_TABLES) if (table := _lookup(data, path)) is not None]
    if len(found) > 1:
        names = " and ".join(_table_name(path) for path, _ in found)
        raise NotebookConfigError(f"{pyproject} carries {names}; keep one (the current name is {_table_name(TABLE)})")
    if not found:
        raise NotebookConfigError(f"{pyproject} has no {_table_name(TABLE)} table")
    return found[0][1]


@dataclass(frozen=True)
class PublisherConfig:
    """Where a repository's dev notebooks live and where nbpress publishes them."""

    root: Path
    dev_dir: Path
    publish_dir: Path
    colab_repo: str
    colab_branch: str = "main"
    hash_key_base: Path | None = None

    @classmethod
    def from_pyproject(cls, root: Path) -> PublisherConfig:
        """Read the notebook table from ``root/pyproject.toml``, refusing a missing required key by name."""
        table = load_table(root)
        missing = [k for k in REQUIRED_KEYS if k not in table]
        if missing:
            raise NotebookConfigError(f"{_table_name(TABLE)} in {root / 'pyproject.toml'} is missing {missing}")
        return cls(
            root=root,
            dev_dir=root / table["dev_dir"],
            publish_dir=root / table["publish_dir"],
            colab_repo=str(table["colab_repo"]),
            colab_branch=str(table.get("colab_branch", "main")),
            hash_key_base=root / str(table.get("hash_key_base", "")),
        )

    def hash_key(self, dev_path: Path) -> str:
        """The key a file is tracked under: its path relative to ``hash_key_base`` (default: the root)."""
        return dev_path.relative_to(self.hash_key_base or self.root).as_posix()


def compute_file_hash(file_path: Path) -> str:
    """SHA256 of a file's bytes, with CRLF folded to LF in text files (UTF-8 without a NUL byte), as nbpress
    hashes.

    A checkout with ``core.autocrlf`` rewrites every text file's line endings, and a raw hash would then report every
    notebook as out of date on that platform alone.
    """
    data = file_path.read_bytes()
    if b"\x00" not in data:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_notebook_hashes(publish_dir: Path) -> dict[str, str]:
    """The stored hashes, or an empty mapping when nothing has been published yet."""
    hash_file = publish_dir / HASH_FILE
    if hash_file.exists():
        with open(hash_file, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def publishable_files(dev_dir: Path) -> list[Path]:
    """Every file under ``dev_dir`` that nbpress publishes: bytecode caches are never one."""
    return sorted(
        p
        for p in dev_dir.rglob("*")
        if p.is_file() and not any(part in EXCLUDED_PARTS for part in p.parts) and p.suffix not in EXCLUDED_SUFFIXES
    )


def changed_files(cfg: PublisherConfig, stored: dict[str, str]) -> list[Path]:
    """The publication inputs whose content hash differs from the stored one."""
    return [p for p in publishable_files(cfg.dev_dir) if stored.get(cfg.hash_key(p)) != compute_file_hash(p)]


def load_notebook(path: Path) -> dict[str, Any]:
    """Parse a notebook file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
