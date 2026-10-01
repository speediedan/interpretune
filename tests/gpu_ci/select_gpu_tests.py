"""Compose the GPU test selection for a change set from ``areas.yaml``.

Stdlib plus PyYAML only, so it runs on a bare hosted agent before any project dependency is installed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

AREAS_FILE = Path(__file__).with_name("areas.yaml")
TEST_ROOTS = ("tests/", "src/it_examples/tests/")


@dataclass
class Selection:
    """The composed selection: ``full``, or the union of areas plus test modules."""

    full: bool = False
    areas: set[str] = field(default_factory=set)
    test_modules: set[str] = field(default_factory=set)
    # path -> what it selected, so every decision is attributable
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not (self.full or self.areas or self.test_modules)

    def to_json(self) -> dict:
        return {
            "mode": "full" if self.full else ("none" if self.empty else "selected"),
            "areas": sorted(self.areas),
            "test_modules": sorted(self.test_modules),
            "reasons": self.reasons,
        }


@lru_cache(maxsize=None)
def _glob_re(pattern: str) -> re.Pattern:
    # `**` crosses directories, `*` and `?` stay within one path segment.
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _matches(path: str, patterns: list[str]) -> bool:
    return any(_glob_re(p).match(path) for p in patterns)


def load_areas(path: Path = AREAS_FILE) -> dict:
    spec = yaml.safe_load(path.read_text())
    if spec.get("version") != 1:
        raise ValueError(f"{path}: unsupported areas version {spec.get('version')!r}; this selector reads version 1")
    return spec


def _is_test_module(path: str) -> bool:
    return path.startswith(TEST_ROOTS) and Path(path).name.startswith("test_") and path.endswith(".py")


def _module_name(path: str) -> str | None:
    if path.startswith("tests/") and path.endswith(".py"):
        return path[:-3].replace("/", ".").removesuffix(".__init__")
    return None


def _test_importers(root: Path) -> dict[str, set[str]]:
    """Map each ``tests.*`` support module to the test modules that import it, transitively."""
    direct: dict[str, set[str]] = {}
    for f in [*root.glob("tests/**/*.py"), *root.glob("src/it_examples/tests/**/*.py")]:
        rel = f.relative_to(root).as_posix()
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "tests":
                mods.add(node.module)
                mods.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                mods.update(a.name for a in node.names if a.name.split(".")[0] == "tests")
        direct[rel] = mods
    importers: dict[str, set[str]] = {}

    def reach(rel: str, seen: set[str]) -> set[str]:
        out = set()
        for mod in direct.get(rel, ()):
            out.add(mod)
            dep = mod.replace(".", "/") + ".py"
            if dep in direct and dep not in seen:
                seen.add(dep)
                out |= reach(dep, seen)
        return out

    for rel in direct:
        if _is_test_module(rel):
            for mod in reach(rel, {rel}):
                importers.setdefault(mod, set()).add(rel)
    return importers


def select(changed: list[str], spec: dict, root: Path | None = None) -> Selection:
    sel = Selection()
    importers = _test_importers(root) if root is not None else None
    for path in changed:
        if _matches(path, spec["run_all"]):
            sel.full = True
            sel.reasons[path] = "run_all"
            continue
        if _matches(path, spec["ignore"]):
            sel.reasons[path] = "ignore"
            continue
        hit = sorted(name for name, area in spec["areas"].items() if _matches(path, area["paths"]))
        sel.areas.update(hit)
        if _is_test_module(path):
            sel.test_modules.add(path)
            sel.reasons[path] = "self" + ("+" + ",".join(hit) if hit else "")
            continue
        mod = _module_name(path)
        if mod is not None and importers is not None:
            users = sorted(importers.get(mod, ()))
            sel.test_modules.update(users)
            sel.reasons[path] = (
                "imported by "
                + (", ".join(users) if users else "no test module")
                + ("; " + ",".join(hit) if hit else "")
            )
            continue
        if hit:
            sel.reasons[path] = ",".join(hit)
            continue
        # Refuse to guess: a path no rule classifies runs everything, and is named in the output.
        sel.full = True
        sel.reasons[path] = "UNCLASSIFIED (full run)"
    return sel


def selected_prefixes(sel: Selection, spec: dict) -> list[str] | None:
    """The node-id prefixes a selection runs, or None for the full set."""
    if sel.full:
        return None
    return sorted({t for a in sel.areas for t in spec["areas"][a]["tests"]} | sel.test_modules)


def write_selection_file(sel: Selection, spec: dict, path: Path) -> None:
    """Write what ``tests/conftest.py`` reads from ``IT_GPU_SELECTION_FILE``: ``full``, or one prefix per line."""
    prefixes = selected_prefixes(sel, spec)
    path.write_text("full\n" if prefixes is None else "".join(f"{p}\n" for p in prefixes))


def changed_files(base: str, head: str = "HEAD") -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"], check=True, capture_output=True, text=True
    ).stdout
    return [line for line in out.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--base", help="diff base ref (merge-base diff against HEAD)")
    src.add_argument("--files", nargs="*", help="explicit changed paths")
    ap.add_argument("--root", default=".", help="repository root (for the test-support import closure)")
    ap.add_argument("--emit", type=Path, default=None, help="also write the selection file conftest.py reads")
    args = ap.parse_args(argv)
    changed = args.files if args.files is not None else changed_files(args.base)
    spec = load_areas()
    sel = select(changed, spec, Path(args.root))
    if args.emit is not None:
        write_selection_file(sel, spec, args.emit)
    json.dump(sel.to_json(), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
