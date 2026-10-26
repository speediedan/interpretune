#!/usr/bin/env python3
"""Check that ``scripts/gpu_lease_wrap.sh`` is byte-identical to finetuning-scheduler's copy.

Both repositories source the same opt-in GPU lease shim, and the file says it is kept byte-identical between them.
Nothing enforced that, and the two copies once drifted by about forty lines. This compares this repository's copy with
finetuning-scheduler's at a ref (``main`` by default), read from GitHub, or with a local file (``--against``).

It is online by construction, so it runs as its own CI job rather than a pre-commit hook: a hook that fails on a clean
offline checkout teaches contributors to skip hooks. A drift is fixed by making the two copies identical again, in
whichever repository is behind.

Exit status: 0 identical, 1 drifted (a unified diff is printed), 2 the other copy could not be read.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL = REPO / "scripts" / "gpu_lease_wrap.sh"
UPSTREAM = "speediedan/finetuning-scheduler"
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/scripts/gpu_lease_wrap.sh"


def read_other(against: str | None, repo: str, ref: str) -> tuple[bytes, str]:
    """The other copy's bytes and a label for it, from ``against`` (a path) or GitHub."""
    if against:
        return Path(against).read_bytes(), against
    url = RAW.format(repo=repo, ref=ref)
    req = urllib.request.Request(url, headers={"User-Agent": "interpretune-ci"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read(), f"{repo}@{ref}"


def compare(local: bytes, other: bytes, other_label: str) -> list[str]:
    """A unified diff from this copy to the other one; empty when they are byte-identical."""
    if local == other:
        return []
    return list(
        difflib.unified_diff(
            local.decode(errors="replace").splitlines(keepends=True),
            other.decode(errors="replace").splitlines(keepends=True),
            fromfile="interpretune:scripts/gpu_lease_wrap.sh",
            tofile=f"{other_label}:scripts/gpu_lease_wrap.sh",
        )
    ) or ["(the copies differ only in bytes a text diff does not show, such as line endings)\n"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--against", help="compare with this file instead of finetuning-scheduler's copy on GitHub")
    ap.add_argument("--repo", default=UPSTREAM)
    ap.add_argument("--ref", default="main")
    args = ap.parse_args(argv)
    try:
        other, label = read_other(args.against, args.repo, args.ref)
    except (OSError, urllib.error.URLError) as exc:
        print(f"REFUSED: could not read the other copy of gpu_lease_wrap.sh: {exc}", file=sys.stderr)
        return 2
    diff = compare(LOCAL.read_bytes(), other, label)
    if not diff:
        print(f"gpu_lease_wrap.sh is byte-identical to {label}")
        return 0
    print(f"gpu_lease_wrap.sh has drifted from {label}; make the copies identical again:")
    sys.stdout.writelines(diff)
    return 1


if __name__ == "__main__":
    sys.exit(main())
