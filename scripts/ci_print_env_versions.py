#!/usr/bin/env python3
"""Print environment diagnostics and versions of key runtime packages for CI."""
from __future__ import annotations
import importlib.metadata as md
import sys


def version_of(pkg: str) -> str:
    try:
        return md.version(pkg)
    except Exception as e:
        return f"not found ({e})"


def main(repo_home: str) -> None:

    packages = [
        'torch',
        'lightning',
        'transformer_lens',
        'sae_lens',
        'finetuning_scheduler',
        'interpretune',
        'datasets',
    ]

    for p in packages:
        print(f"{p} version: {version_of(p)}")


if __name__ == '__main__':
    # Allow caller to pass repo_home as first arg; default to current working dir
    repo_home = sys.argv[1] if len(sys.argv) > 1 else "."
    main(repo_home)
