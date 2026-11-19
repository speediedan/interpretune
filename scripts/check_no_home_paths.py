#!/usr/bin/env python
"""Fail when a tracked Markdown file or notebook carries a contributor's home or cache directory.

Captured output (profiling tables, stack traces, notebook cells) arrives by paste and records the absolute paths of
whoever produced it. In a public repository that publishes a personal filesystem layout, and a reader cannot tell
a captured path from an instruction: a link or a ``cd`` written with someone's home directory works for nobody else.

The check enumerates tracked ``*.md`` and ``*.ipynb`` files with ``git ls-files`` rather than taking a file list,
because the repository-level pre-commit ``exclude:`` removes both skills trees from every hook's list, and a skill
file is where this was first found. Placeholder names (``runner``, ``username``, ``user``, ``nobody``) are allowed,
as are paths written with a variable (``/mnt/cache/$USER``, ``${HOME}``).

Usage: ``python scripts/check_no_home_paths.py``. Exits 1 naming each offending line, and 2 when no file was found
to check.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PLACEHOLDERS = {"runner", "runneradmin", "username", "user", "nobody"}
# a home or cache root followed by a literal account name (a variable such as $USER or ${HOME} does not match)
HOME_PATH = re.compile(r"(?:/home|/Users|/mnt/cache(?:_extended)?)/([A-Za-z_][A-Za-z0-9_.-]*)/")


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "*.md", "*.ipynb"], capture_output=True, check=True)
    return [Path(p) for p in out.stdout.decode().split("\0") if p]


def main() -> int:
    files = tracked_files()
    if not files:
        print("no tracked *.md or *.ipynb files found; nothing was checked", file=sys.stderr)
        return 2  # an empty check must not read as a passing one

    found: list[str] = []
    for path in files:
        if not path.is_file():  # a tracked file deleted in the working tree
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for match in HOME_PATH.finditer(line):
                if match.group(1) not in PLACEHOLDERS:
                    found.append(f"{path}:{lineno}: {match.group(0)}")
    if found:
        print("contributor-specific paths in published files (use a placeholder or an environment variable):")
        for entry in found:
            print(f"  {entry}")
        return 1
    print(f"checked {len(files)} files: no contributor-specific paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
