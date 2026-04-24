#!/usr/bin/env python
"""Pre-commit hook: ensure benchmark_registry.yaml is committed in isolation.

When benchmark_registry.yaml is staged, the only other files allowed in the
same commit are tests/benchmarks/benchmark_update.allow, documentation files
(markdown, rst), and files under scripts/, docs/, and dockers/.  This keeps
every registry update traceable to the exact codebase state that produced it
while allowing incidental documentation and tooling updates.
"""

from __future__ import annotations

import subprocess
import sys


_ALWAYS_ALLOWED = {
    "tests/benchmarks/benchmark_registry.yaml",
    "tests/benchmarks/benchmark_update.allow",
}

_ALLOWED_EXTENSIONS = {".md", ".rst"}
_ALLOWED_PREFIXES = ("scripts/", "docs/", "dockers/")


def main() -> int:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
    )
    staged = {f.strip() for f in result.stdout.splitlines() if f.strip()}

    extra = set()
    for f in staged:
        if f in _ALWAYS_ALLOWED:
            continue
        if any(f.endswith(ext) for ext in _ALLOWED_EXTENSIONS):
            continue
        if any(f.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
            continue
        extra.add(f)

    if extra:
        print(
            "ERROR: benchmark_registry.yaml must be committed in isolation.\n"
            "The following unrelated files are also staged:\n"
            + "\n".join(f"  {f}" for f in sorted(extra))
            + "\n\nCommit other changes first, then commit the registry update separately.\n"
            "Note: markdown/rst files and files under scripts/, docs/, dockers/ are allowed."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
