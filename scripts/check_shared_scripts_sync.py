#!/usr/bin/env python3
"""Check that the shell scripts shared with finetuning-scheduler are byte-identical to its copies.

Each script in :data:`SHARED` is kept byte-identical between the two repositories, and the copies once drifted with
nothing saying so: a fix landed in one and stayed missing from the other. This compares every one of this
repository's copies with finetuning-scheduler's at a ref (``main`` by default), read from GitHub, or with the copies
in a local directory (``--against-dir``).

It is online by construction, so it runs as its own CI job rather than a pre-commit hook: a hook that fails on a clean
offline checkout teaches contributors to skip hooks. A drift is fixed by making the two copies identical again, in
whichever repository is behind. What differs per project stays outside these files (``manage_standalone_regex.cfg``
beside ``manage_standalone_processes.sh``).

Exit status: 0 every copy identical, 1 any drifted (a unified diff is printed for each), 2 any other copy could not
be read.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
#: The shared scripts, by their path relative to the repository root (the same in both repositories).
SHARED = ("scripts/gpu_lease_wrap.sh", "scripts/manage_standalone_processes.sh")
UPSTREAM = "speediedan/finetuning-scheduler"
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def read_other(path: str, against_dir: str | None, repo: str, ref: str) -> tuple[bytes, str]:
    """The other copy of ``path`` and a label for it, from ``against_dir`` (a directory) or GitHub."""
    if against_dir:
        return (Path(against_dir) / Path(path).name).read_bytes(), against_dir
    url = RAW.format(repo=repo, ref=ref, path=path)
    req = urllib.request.Request(url, headers={"User-Agent": "interpretune-ci"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read(), f"{repo}@{ref}"


def compare(path: str, local: bytes, other: bytes, other_label: str) -> list[str]:
    """A unified diff from this copy of ``path`` to the other one; empty when they are byte-identical."""
    if local == other:
        return []
    return list(
        difflib.unified_diff(
            local.decode(errors="replace").splitlines(keepends=True),
            other.decode(errors="replace").splitlines(keepends=True),
            fromfile=f"interpretune:{path}",
            tofile=f"{other_label}:{path}",
        )
    ) or ["(the copies differ only in bytes a text diff does not show, such as line endings)\n"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--against-dir", help="compare with the copies in this directory instead of GitHub's")
    ap.add_argument("--repo", default=UPSTREAM)
    ap.add_argument("--ref", default="main")
    args = ap.parse_args(argv)
    status = 0
    for path in SHARED:
        name = Path(path).name
        try:
            other, label = read_other(path, args.against_dir, args.repo, args.ref)
        except (OSError, urllib.error.URLError) as exc:
            print(f"REFUSED: could not read the other copy of {name}: {exc}", file=sys.stderr)
            status = 2
            continue
        diff = compare(path, (REPO / path).read_bytes(), other, label)
        if not diff:
            print(f"{name} is byte-identical to {label}")
            continue
        print(f"{name} has drifted from {label}; make the copies identical again:")
        sys.stdout.writelines(diff)
        status = max(status, 1)
    return status


if __name__ == "__main__":
    sys.exit(main())
