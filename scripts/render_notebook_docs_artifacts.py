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

BUNDLED_OPS_DIR = REPO_ROOT / "src" / "interpretune" / "analysis" / "ops" / "bundled"

# Where the rendered op surface is stamped, so `--check-stale` can tell whether an artifact's captured
# OUTPUTS have gone stale even when its sources have not.
OP_SURFACE_KEY = "interpretune_op_surface"

# Drift gets its own exit code so a caller can tell it from this script FAILING TO RUN, which also exits
# non-zero. docs-build previously mapped every non-zero exit to "an artifact has drifted", so when a missing
# import made the script crash it reported drift -- a cause that was not merely unproven but false. No
# amount of rewording fixes that; only a distinct code lets the caller say WHICH happened.
#
# 3, NOT 2, and the difference is load-bearing: argparse exits 2 on a usage error, which is not something
# this script chooses. Measured here -- an unrecognized flag exits 2 before main() is ever entered. Using 2
# for drift would therefore report a mistyped or renamed flag as "an artifact has drifted", which is the
# original defect wearing a new hat. The taken codes are 0 success, 1 uncaught exception, 2 argparse usage.
DRIFT_EXIT_CODE = 3

# Cells removed from the docs artifact, matched on `metadata.id` set by publish_notebooks.py.
DOCS_EXCLUDED_CELL_IDS = {"install-deps"}

# Cells removed by papermill TAG. `injected-parameters` is the cell papermill inserts to record the
# parameters a run was executed with. On a docs page it is actively misleading: a parameterized run
# renders e.g. `GENERATE_MISSING_LOCAL_EXPLANATIONS = True` directly beneath the notebook's own
# parameters cell documenting it as False, so the page appears to contradict itself.
DOCS_EXCLUDED_CELL_TAGS = {"injected-parameters"}


def _is_excluded_cell(cell: dict[str, Any]) -> bool:
    meta = cell.get("metadata", {})
    if meta.get("id") in DOCS_EXCLUDED_CELL_IDS:
        return True
    return bool(DOCS_EXCLUDED_CELL_TAGS.intersection(meta.get("tags", []) or []))


def load_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(notebook: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def strip_docs_excluded_cells(notebook: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Drop cells that only make sense in a runnable copy (the commented-out installer)."""
    cells = notebook.get("cells", [])
    kept = [c for c in cells if not _is_excluded_cell(c)]
    return notebook | {"cells": kept}, len(cells) - len(kept)


def has_outputs(notebook: dict[str, Any]) -> bool:
    return any(c.get("outputs") for c in notebook.get("cells", []) if c.get("cell_type") == "code")


def bundled_op_names() -> set[str]:
    """Live bundled op names, read from YAML rather than the dispatcher.

    Deliberately NOT `DISPATCHER._op_definitions`: that reflects whatever the current session has loaded,
    including any hub collection pulled at the time, so a gate built on it would give different answers to
    different people. The bundled YAML is the same for everyone with the same checkout, which is the
    property a CI gate needs.
    """
    try:
        import yaml
    except ImportError:
        # docs-build runs this check ONCE BEFORE installing requirements, deliberately: the comment there
        # calls it "cheap, stdlib-only, and BEFORE the heavy install so it fails fast". Requiring a
        # third-party import would break that contract, so the op-surface half degrades to unavailable and
        # the source-drift half still runs. The same workflow runs the check again after the install,
        # where yaml is present and the full check happens.
        return set()

    names: set[str] = set()
    for path in sorted(BUNDLED_OPS_DIR.rglob("*.yaml")):
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key, value in content.items():
            if key == "composite_operations" and isinstance(value, dict):
                names |= set(value)
            elif key != "collection" and isinstance(value, dict):
                names.add(key)
    return names


def stamp_op_surface(notebook: dict[str, Any], op_names: set[str]) -> None:
    """Record the op surface an artifact was rendered against, in notebook metadata."""
    notebook.setdefault("metadata", {})[OP_SURFACE_KEY] = sorted(op_names)


def captured_output_text(notebook: dict[str, Any]) -> str:
    """All captured output text in one blob, which is the half `artifact_matches_source` cannot see."""
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []) or []:
            for key in ("text",):
                chunks.append("".join(output.get(key, []) or []))
            data = output.get("data") or {}
            for mime, payload in data.items():
                if mime.startswith("text/"):
                    chunks.append("".join(payload if isinstance(payload, list) else [payload]))
    return "".join(chunks)


def stale_output_references(artifact: dict[str, Any], current_ops: set[str]) -> list[str]:
    """Op names an artifact's OUTPUTS mention that the current op surface no longer has.

    This is the gap `artifact_matches_source` structurally cannot close: its signature is built from
    ``(cell_type, source)``, so output drift is not merely unchecked, it is unrepresentable. A renamed op
    appearing only in captured output therefore passes every CPU job and leaves the docs site rendering a
    symbol that no longer exists.

    Comparing outputs directly would fix that at the cost of needing a rendering environment -- and a
    re-render rewrites thousands of lines of timings and object ids for a one-symbol rename, so the diff
    is close to unreviewable. This compares the recorded op SURFACE instead: names the artifact was
    rendered against, minus names that exist now, intersected with what its outputs actually mention.
    No execution, no GPU, and it only fires when a specific dead name is really present.

    Returns an empty list for an artifact with no recorded surface -- absence of a stamp is not evidence
    of freshness, and `unstamped_artifacts` reports those separately rather than letting them read as
    clean.
    """
    if not current_ops:
        # Empty means the surface could not be READ (pyyaml absent pre-install), not that zero ops exist.
        # Without this, "recorded minus current" is the whole recorded surface, so every stamped artifact
        # is reported stale at once -- a failure mode that looks like catastrophic drift and is really a
        # missing import. The caller also gates on this, deliberately: the caller decides what to REPORT,
        # while this keeps the comparison itself from producing a confidently wrong answer.
        return []

    recorded = artifact.get("metadata", {}).get(OP_SURFACE_KEY)
    if not recorded:
        return []
    text = captured_output_text(artifact)
    return sorted(name for name in set(recorded) - current_ops if name in text)


def artifact_matches_source(artifact: dict[str, Any], source: dict[str, Any]) -> bool:
    """Do artifact and publish source agree on everything except outputs?

    The docs site renders the ARTIFACT, and conf.py prefers it over the publish lane. So a notebook edit that changes
    only prose is invisible on the site until the artifact is rebuilt -- the page keeps rendering the old text with no
    error anywhere. This compares cell types and sources (ignoring the install-deps cell the artifact deliberately
    drops) so that drift is detectable.
    """

    def signature(notebook: dict[str, Any]) -> list[tuple[str, str]]:
        return [
            (c["cell_type"], "".join(c.get("source", [])))
            for c in notebook.get("cells", [])
            if not _is_excluded_cell(c)
        ]

    return signature(artifact) == signature(source)


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
    parser.add_argument(
        "--check-stale",
        action="store_true",
        help="Exit non-zero if any artifact has drifted from its publish source, or if its captured outputs "
        "reference an op name the current bundled surface no longer has. The docs render artifacts, so "
        "either kind of drift means the site is showing something stale with no build error.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen; write nothing.")
    args = parser.parse_args()

    notebooks = discover(args.notebook)
    if not notebooks:
        print(f"No notebooks matched under {PUBLISH_DIR}", file=sys.stderr)
        return 1

    if args.check_stale:
        stale: list[str] = []
        output_stale: list[str] = []
        unstamped: list[str] = []
        current_ops = bundled_op_names()
        # An empty surface means yaml was unavailable (see bundled_op_names), not that there are no ops.
        # Reporting the difference matters: silently skipping would read as "checked and clean".
        op_check_available = bool(current_ops)
        for source in notebooks:
            rel = source.relative_to(PUBLISH_DIR)
            artifact = ARTIFACT_DIR / rel
            if not artifact.exists():
                continue
            artifact_nb = load_notebook(artifact)
            if not artifact_matches_source(artifact_nb, load_notebook(source)):
                stale.append(str(rel))
                continue  # a source-drifted artifact needs a rebuild regardless of its outputs
            if not op_check_available:
                continue
            if dead := stale_output_references(artifact_nb, current_ops):
                output_stale.append(f"{rel}: outputs reference {', '.join(dead)}")
            elif not artifact_nb.get("metadata", {}).get(OP_SURFACE_KEY):
                unstamped.append(str(rel))
        for rel_str in stale:
            print(f"STALE artifact (rebuild it): {rel_str}", file=sys.stderr)
        for entry in output_stale:
            print(f"STALE artifact OUTPUT (re-execute it): {entry}", file=sys.stderr)
        # Reported, not failed: an artifact predating the stamp cannot be checked, and failing on that
        # would make adopting the stamp a wall rather than a ramp. Silence would be worse -- it would read
        # as "checked and clean" when it is "not checkable".
        for rel_str in unstamped:
            print(f"note: no recorded op surface, output drift not checkable: {rel_str}", file=sys.stderr)
        if not op_check_available:
            print("note: pyyaml unavailable, output-drift check skipped (source drift still checked)", file=sys.stderr)
        total = len(stale) + len(output_stale)
        print(f"{total} stale artifact(s); {len(unstamped)} unstamped")
        return DRIFT_EXIT_CODE if total else 0

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
            # Stamp on EVERY write, including --no-execute. A strip-only pass does not change outputs, so
            # the surface those outputs were produced against is still the one recorded here; refusing to
            # stamp then would leave artifacts permanently unstampable without a GPU re-render.
            stamp_op_surface(notebook, bundled_op_names())
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
