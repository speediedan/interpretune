#!/usr/bin/env python
"""Warm the local Hugging Face cache from ``tests/hf_warm_manifest.yaml`` so the suite can run offline.

The hosted CI matrix runs three jobs at once, and each job re-validates every cached Hub file and lists
repository trees on every tokenizer load, so a warm cache alone barely reduces the Hub API traffic. The
only shape that removes the rate-limit failures is to make the tests themselves issue no Hub requests:
warm the cache here (one sequential pass per job, retried with backoff), then run pytest with
``HF_HUB_OFFLINE=1``. A test that needs an artifact this manifest does not list fails offline with a
message naming the manifest, so drift is caught on the first run rather than masked by a lucky download.

Usage::

    python scripts/warm_hf_cache.py                # warm everything in the manifest
    python scripts/warm_hf_cache.py --dry-run      # print what would be fetched
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "tests" / "hf_warm_manifest.yaml"
ATTEMPTS = 4
FIRST_DELAY_SECONDS = 20.0


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise SystemExit(f"{path}: manifest must be a mapping with `models` and `datasets` lists")
    return manifest


def _retry(label: str, fn, attempts: int = ATTEMPTS) -> None:
    """Run ``fn`` with exponential backoff.

    The Hub client already backs off on 429 for file downloads; this outer loop covers the metadata calls that do not,
    and transient network failures on the hosted runners.
    """
    delay = FIRST_DELAY_SECONDS
    for attempt in range(1, attempts + 1):
        try:
            fn()
            return
        except Exception as exc:  # - any failure here is retried, the last one is re-raised
            if attempt == attempts:
                raise
            print(f"  {label}: attempt {attempt} failed ({type(exc).__name__}: {exc}); retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2


def warm_models(entries: list[dict[str, Any]], dry_run: bool) -> None:
    for entry in entries:
        repo_id = entry["repo_id"]
        kwargs = {
            "repo_id": repo_id,
            "repo_type": entry.get("repo_type", "model"),
            "revision": entry.get("revision"),
            "allow_patterns": entry.get("allow_patterns"),
        }
        print(
            f"model {repo_id} (revision={kwargs['revision'] or 'main'}, patterns={kwargs['allow_patterns'] or 'all'})"
        )
        if not dry_run:
            from huggingface_hub import snapshot_download

            _retry(repo_id, lambda: snapshot_download(**kwargs))


def warm_datasets(entries: list[dict[str, Any]], dry_run: bool) -> None:
    for entry in entries:
        path = entry["path"]
        config_name = entry.get("config_name")
        revision = entry.get("revision")
        print(f"dataset {path} (config={config_name or 'default'}, revision={revision or 'main'})")
        if not dry_run:
            from datasets import load_dataset

            # The same call shape the tests use, so the prepared arrow cache the tests look up is the one written.
            _retry(path, lambda: load_dataset(path, config_name, revision=revision))


def _cache_size() -> str:
    from huggingface_hub.constants import HF_HOME

    total = 0
    for root, _dirs, files in os.walk(HF_HOME):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return f"{total / 2**30:.2f} GiB under {HF_HOME}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true", help="print the plan without fetching anything")
    args = parser.parse_args(argv)
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        raise SystemExit("HF_HUB_OFFLINE is set; the warm step must run online (unset it for this step only)")
    manifest = _load_manifest(args.manifest)
    print(f"warming from {args.manifest} (cache_version={manifest.get('cache_version')})")
    warm_models(manifest.get("models") or [], args.dry_run)
    warm_datasets(manifest.get("datasets") or [], args.dry_run)
    if not args.dry_run:
        print(f"Hub cache: {_cache_size()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
