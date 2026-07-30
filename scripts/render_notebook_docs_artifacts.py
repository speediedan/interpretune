#!/usr/bin/env python3
"""Build the executed notebook artifacts the docs site renders.

Three notebook lanes, each with a different job:

1. ``src/it_examples/notebooks/dev/`` — authoring copies.
2. ``src/it_examples/notebooks/publish/`` — the notebooks users clone and run. Kept **stripped** of
   outputs by the ``nbstripout`` pre-commit hook and shipped in the sdist, so a fresh clone gives a
   clean notebook and the package stays small.
3. ``docs/notebook_artifacts/`` — what THIS script produces: the same notebooks, executed, **with**
   outputs, for the docs site. Excluded from the sdist (``prune docs`` in ``MANIFEST.in``) and
   exempted from the ``nbstripout`` hook, which would otherwise strip exactly what makes them useful.

``docs/source/conf.py`` stages lane 3 in preference to lane 2, so a rendered page shows real output
once the artifact exists and falls back to code-only before that.

Docs artifacts also drop the ``install-deps`` cell: it is a commented-out ``%pip install`` block that
exists so a Colab/fresh-environment reader can uncomment it. On a static docs page it is noise.

Usage:
    # refresh artifacts without running anything (strip-only; keeps existing outputs)
    python scripts/render_notebook_docs_artifacts.py --no-execute

    # execute and capture outputs (needs the full env; some notebooks need a GPU,
    # gated gemma weights, and a running local Neuronpedia stack)
    python scripts/render_notebook_docs_artifacts.py --notebook saelens_adapter_example

    python scripts/render_notebook_docs_artifacts.py --list
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISH_DIR = REPO_ROOT / "src" / "it_examples" / "notebooks" / "publish"
ARTIFACT_DIR = REPO_ROOT / "docs" / "notebook_artifacts"

# Cells removed from the docs artifact, matched on `metadata.id` set by publish_notebooks.py.
DOCS_EXCLUDED_CELL_IDS = {"install-deps"}


def load_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(notebook: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def strip_docs_excluded_cells(notebook: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Drop cells that only make sense in a runnable copy (the commented-out installer)."""
    cells = notebook.get("cells", [])
    kept = [c for c in cells if c.get("metadata", {}).get("id") not in DOCS_EXCLUDED_CELL_IDS]
    return notebook | {"cells": kept}, len(cells) - len(kept)


def has_outputs(notebook: dict[str, Any]) -> bool:
    return any(c.get("outputs") for c in notebook.get("cells", []) if c.get("cell_type") == "code")


def discover(selector: str | None) -> list[Path]:
    paths = sorted(PUBLISH_DIR.rglob("*.ipynb"))
    if selector:
        paths = [p for p in paths if selector in str(p.relative_to(PUBLISH_DIR))]
    return paths


def execute(source: Path, target: Path, timeout: int, parameters: dict[str, Any]) -> None:
    """Execute via papermill — the same mechanism tests/examples/test_notebooks.py uses."""
    import papermill as pm

    target.parent.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        str(source),
        str(target),
        parameters=parameters or {},
        kernel_name="python3",
        cwd=str(source.parent),  # notebooks resolve sidecar assets relative to themselves
        progress_bar=False,
        execution_timeout=timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", help="Substring selecting which notebooks to process (default: all).")
    parser.add_argument(
        "--no-execute",
        dest="execute",
        action="store_false",
        help="Do not run anything. Re-strips excluded cells and preserves any outputs already present.",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="Per-cell execution timeout (default: 1800).")
    parser.add_argument("--parameters", default="{}", help="JSON dict of papermill parameters.")
    parser.add_argument("--list", action="store_true", help="List candidate notebooks and their artifact status.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen; write nothing.")
    args = parser.parse_args()

    notebooks = discover(args.notebook)
    if not notebooks:
        print(f"No notebooks matched under {PUBLISH_DIR}", file=sys.stderr)
        return 1

    if args.list:
        for source in notebooks:
            rel = source.relative_to(PUBLISH_DIR)
            artifact = ARTIFACT_DIR / rel
            if not artifact.exists():
                state = "MISSING  (docs render code-only)"
            elif has_outputs(load_notebook(artifact)):
                state = "executed (docs render real output)"
            else:
                state = "no outputs (docs render code-only)"
            print(f"  {state:<34} {rel}")
        return 0

    failures: list[str] = []
    for source in notebooks:
        rel = source.relative_to(PUBLISH_DIR)
        artifact = ARTIFACT_DIR / rel
        if args.dry_run:
            print(f"[dry-run] {'execute' if args.execute else 'copy'} {rel}")
            continue

        try:
            if args.execute:
                print(f"executing {rel} ...", flush=True)
                execute(source, artifact, args.timeout, json.loads(args.parameters))
            else:
                # Preserve outputs already captured; only fall back to the stripped copy.
                if not artifact.exists():
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, artifact)

            notebook, removed = strip_docs_excluded_cells(load_notebook(artifact))
            save_notebook(notebook, artifact)
            outputs = "with outputs" if has_outputs(notebook) else "NO outputs"
            print(f"  wrote {rel} ({outputs}, {removed} cell(s) removed)")
        except Exception as exc:  # - report and continue to the next notebook
            # Discard whatever papermill managed to write. A half-executed notebook still lands on
            # disk, carrying the traceback as an error output and (because the strip step never ran)
            # the install-deps cell -- and conf.py prefers artifacts over the publish lane, so the
            # docs site would render that traceback as though it were the example's real output.
            # Removing it makes the page fall back to code-only, which is the honest result.
            if artifact.exists():
                artifact.unlink()
                print(f"  discarded partial artifact for {rel}", file=sys.stderr)
            failures.append(f"{rel}: {type(exc).__name__}: {exc}")
            print(f"  FAILED {rel}: {type(exc).__name__}: {exc}", file=sys.stderr)

    # Sidecar assets (images, helper modules) the notebooks reference relatively.
    if not args.dry_run:
        for asset in PUBLISH_DIR.rglob("*"):
            if asset.is_file() and asset.suffix != ".ipynb" and not asset.name.startswith("."):
                target = ARTIFACT_DIR / asset.relative_to(PUBLISH_DIR)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(asset, target)

    if failures:
        print(f"\n{len(failures)} notebook(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
