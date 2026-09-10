"""Revisions that key a measured artifact to the source it measured.

Two sides compute a revision and a card compares them: the conformance suite, when it writes its report, and the
publisher, when it stages a component. If the two computed different things (a repository head on one side and the
last commit touching the component's directory on the other) they would essentially never be equal, the card would
fail closed on every publish, and nobody would notice, because absence is a legitimate state with a legitimate
message. So both sides call the same function on the same kind of path, and the comparison means one thing: this
report measured this component's source as it now stands.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    out = result.stdout.strip()
    return out if result.returncode == 0 and out else None


def repo_head(path: Path) -> str | None:
    """The full sha of HEAD of the repository ``path`` sits in, or ``None`` when it is not a checkout."""
    head = _git(["rev-parse", "HEAD"], path if path.is_dir() else path.parent)
    return head if head is not None and len(head) == 40 else None


def directory_revision(path: Path) -> str | None:
    """The full sha of the last commit that touched ``path``, or ``None`` when it is not tracked in a checkout.

    Stable across unrelated commits elsewhere in the repository, which is what makes it the right key: a report that
    measured a component stays valid until the component's own source changes.
    """
    target = path.resolve()
    cwd = target if target.is_dir() else target.parent
    sha = _git(["log", "-1", "--format=%H", "--", str(target)], cwd)
    return sha if sha is not None and len(sha) == 40 else None
