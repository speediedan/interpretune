#!/usr/bin/env python
"""Publish an in-tree bundled op family to the Hub as an op collection.

The first published seed collection (``speediedan/concept_direction_ops``) is generated from
``src/interpretune/analysis/ops/bundled/concept/`` rather than maintained separately, so the bundled copy
remains the single source and the two cannot drift (design §3.9, D9). The concept family stays bundled: it is
load-bearing in hermetic CI (the GPU parity gates, cross-backend compatibility, capability discovery), so
extracting it would make the hosted matrix network-dependent and trust-consenting.

Publishing it anyway is what makes the publishability claim constructive rather than asserted -- the same
lint that says a bundled family carries no privileged dependency on interpretune internals is proved by
lifting one onto the Hub and pulling it back.

    python scripts/publish_op_collection.py --family concept --repo-id speediedan/concept_direction_ops
    python scripts/publish_op_collection.py --family concept --repo-id me/my_ops --dry-run --build-dir /tmp/x
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

BUNDLED_ROOT = project_root / "src" / "interpretune" / "analysis" / "ops" / "bundled"

# A hub collection CAN be out of step with the installed interpretune, so unlike a bundled family it declares
# a window. The `.dev0` suffix is load-bearing rather than decorative: a bare `>=0.1` floor does NOT match the
# `0.1.0.devN+g<sha>` versions setuptools_scm produces between tags (a dev release sorts before its release),
# so it would skip this collection in every source checkout, CI included.
DEFAULT_REQUIRES = {"interpretune": ">=0.1.0.dev0"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, help="bundled family directory name (e.g. concept)")
    parser.add_argument("--repo-id", required=True, help="target Hub repo, e.g. speediedan/concept_direction_ops")
    parser.add_argument("--collection-name", help="collection handle (defaults to the repo name)")
    parser.add_argument(
        "--collection-version",
        help="override the collection version (defaults to tracking the bundled family's declared version)",
    )
    parser.add_argument("--private", action="store_true", help="create/keep the repo private")
    parser.add_argument("--no-requires", action="store_true", help="publish without a compatibility window")
    parser.add_argument("--dry-run", action="store_true", help="build the tree and print it; do not upload")
    parser.add_argument("--build-dir", type=Path, help="where to build (required with --dry-run)")
    parser.add_argument(
        "--token-env",
        default="IT_HF_TOKEN",
        help="environment variable holding the write token (default: IT_HF_TOKEN)",
    )
    args = parser.parse_args()

    family_dir = BUNDLED_ROOT / args.family
    if not family_dir.is_dir():
        parser.error(f"no bundled family at {family_dir}")
    if args.dry_run and args.build_dir is None:
        parser.error("--dry-run requires --build-dir")

    requires = None if args.no_requires else DEFAULT_REQUIRES

    if args.dry_run:
        from interpretune.hub.cards import generate_component_card
        from interpretune.hub.publish import build_op_collection_tree

        manifest = build_op_collection_tree(
            family_dir,
            args.build_dir,
            args.repo_id,
            collection_name=args.collection_name,
            collection_version=args.collection_version,
            requires=requires,
        )
        generate_component_card(manifest, args.repo_id).save(args.build_dir / "README.md")
        print(f"Built {args.repo_id} tree at {args.build_dir} (not uploaded):")
        for path in sorted(p for p in args.build_dir.rglob("*") if p.is_file()):
            print(f"  {path.relative_to(args.build_dir)} ({path.stat().st_size} bytes)")
        return 0

    token = os.environ.get(args.token_env)
    if not token:
        parser.error(f"{args.token_env} is not set; a write token is required to publish")

    from interpretune.hub.publish import publish_op_collection

    commit = publish_op_collection(
        family_dir,
        args.repo_id,
        collection_name=args.collection_name,
        collection_version=args.collection_version,
        requires=requires,
        private=args.private,
        token=token,
    )
    print(f"Published {args.repo_id} at {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
