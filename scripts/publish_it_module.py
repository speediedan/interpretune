#!/usr/bin/env python
"""Publish an in-repo interpretune component tree to the Hugging Face Hub.

Thin wrapper over :mod:`interpretune.hub.publish` (library code carries the logic so the parity tests cover
it). The in-repo tree mirrors the Hub tree; publishing copies it and adds the generated card + entrypoint.

Examples:
    # publish the rte task component (private until deliberately flipped public)
    python scripts/publish_it_module.py --component src/it_examples/examples/rte \\
        --repo-id speediedan/rte --entrypoint src/it_examples/experiments/rte_boolq.py --private

    # regenerate + push ONLY the card/manifest for an op-collection repo
    python scripts/publish_it_module.py --component <dir-with-it_component.yaml> --repo-id <org>/<repo>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--component", type=Path, required=True, help="in-repo component dir (has it_component.yaml)")
    parser.add_argument("--repo-id", required=True, help="<org>/<repo> target on the Hub")
    parser.add_argument("--entrypoint", type=Path, default=None, help="module entrypoint source file, if declared")
    parser.add_argument("--private", action="store_true", help="create the repo private (default public)")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--build-dir", type=Path, default=None, help="keep the built tree here instead of a tempdir")
    args = parser.parse_args()

    from interpretune.hub.publish import publish_component

    token = os.getenv("IT_HF_TOKEN") or os.getenv("HF_TOKEN")
    sha = publish_component(
        args.component,
        args.repo_id,
        entrypoint_src=args.entrypoint,
        build_dir=args.build_dir,
        private=args.private,
        token=token,
        commit_message=args.commit_message,
    )
    print(f"Published {args.component} -> {args.repo_id} @ {sha}")


if __name__ == "__main__":
    main()
