#!/usr/bin/env python
"""Fail by name when a documentation page cites, as open, an issue that has closed.

A page that records gaps as rows naming an open issue is true only while those issues are open; nothing on the page
changes when one closes. This check reads every ``#N`` cited in the given pages, asks GitHub for each issue's state,
and exits non-zero naming every closed one and the line that cites it. It runs on a schedule beside the docs link
check, online by construction, so it cannot agree with itself the way an offline fixture would.

Usage: ``python scripts/check_doc_issue_rows.py --repo owner/name docs/page.md [docs/other.md ...]``. Reads
``GITHUB_TOKEN`` (or ``GH_TOKEN``) from the environment for the API; without one it still runs at the anonymous rate
limit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ISSUE_REF = re.compile(r"(?<![\w/])#(\d{2,6})\b")


def issue_state(repo: str, number: int, token: str | None) -> str:
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/issues/{number}")
    request.add_header("Accept", "application/vnd.github+json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:  # - fixed https host
        return json.load(response)["state"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True, help="owner/name of the repository the issue numbers refer to")
    parser.add_argument("pages", nargs="+", type=Path)
    args = parser.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    citations: dict[int, list[str]] = {}
    for page in args.pages:
        for lineno, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            for match in ISSUE_REF.finditer(line):
                citations.setdefault(int(match.group(1)), []).append(f"{page}:{lineno}")
    if not citations:
        print(f"no issue citations found in {', '.join(str(p) for p in args.pages)}; nothing to check", file=sys.stderr)
        return 2  # an empty check must not read as a passing one

    closed: list[str] = []
    for number in sorted(citations):
        state = issue_state(args.repo, number, token)
        print(f"#{number}: {state}")
        if state != "open":
            closed.append(f"#{number} is {state}, cited at {', '.join(citations[number])}")
    if closed:
        print("\nrows citing a closed issue must be updated to state what landed:", file=sys.stderr)
        for entry in closed:
            print(f"  {entry}", file=sys.stderr)
        return 1
    print(f"all {len(citations)} cited issues are open")
    return 0


if __name__ == "__main__":
    sys.exit(main())
