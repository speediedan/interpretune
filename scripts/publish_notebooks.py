#!/usr/bin/env python3
"""Notebook Publisher Script.

Copies notebooks from dev/ to publish/ directory, strips "remove-cell" tags,
and adds Colab badges and installation cells for published versions.

Usage:
    python scripts/publish_notebooks.py [--dry-run] [--check-only] [--force]

Options:
    --dry-run: Show what would be done without making changes
    --check-only: Check if any notebooks need publishing (returns non-zero if changes needed)
    --force: Publish all files (ignores stored hashes), useful when changing the
             publisher behavior (for example, the installation cell) to re-run
             publication steps on already-published notebooks.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Set


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def load_notebook_hashes(publish_dir: Path) -> Dict[str, str]:
    """Load stored notebook hashes from .notebook_hashes.json."""
    hash_file = publish_dir / ".notebook_hashes.json"
    if hash_file.exists():
        with open(hash_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_notebook_hashes(publish_dir: Path, hashes: Dict[str, str]) -> None:
    """Save notebook hashes to .notebook_hashes.json."""
    hash_file = publish_dir / ".notebook_hashes.json"
    with open(hash_file, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)


def get_changed_files(dev_dir: Path, stored_hashes: Dict[str, str], file_pattern: str = "*") -> Set[Path]:
    """Get set of files that have changed since last publish."""
    changed_files = set()
    exclude_patterns = ["__pycache__", ".pyc"]

    for file_path in dev_dir.rglob(file_pattern):
        if not file_path.is_file():
            continue

        # Skip excluded patterns
        if any(pattern in str(file_path) for pattern in exclude_patterns):
            continue

        # Get relative path from dev directory
        rel_path = file_path.relative_to(dev_dir)
        hash_key = f"notebooks/dev/{rel_path}"

        current_hash = compute_file_hash(file_path)
        stored_hash = stored_hashes.get(hash_key)

        if current_hash != stored_hash:
            changed_files.add(file_path)

    return changed_files


def load_notebook(path: Path) -> Dict[str, Any]:
    """Load a Jupyter notebook from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(notebook: Dict[str, Any], path: Path) -> None:
    """Save a Jupyter notebook to file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)


def strip_remove_cell_tags(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """Remove cells tagged with 'remove-cell'."""
    cells = []
    for cell in notebook.get("cells", []):
        tags = cell.get("metadata", {}).get("tags", [])
        if "remove-cell" not in tags:
            cells.append(cell)
    notebook["cells"] = cells
    return notebook


def add_colab_badge_and_install_cell(notebook: Dict[str, Any], relative_path: str) -> Dict[str, Any]:
    """Add Colab badge and installation cell at the top of the notebook."""
    # Create Colab badge markdown cell
    org = "speediedan"
    repo = "interpretune"
    # Convert dev path to publish path for the URL
    publish_relative_path = relative_path.replace("/dev/", "/publish/")
    colab_url = f"https://colab.research.google.com/github/{org}/{repo}/blob/main/{publish_relative_path}"

    badge_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})"],
    }

    # Create a single, commented-out installation cell that preserves editable
    # installs by default. The cell is left commented to avoid inadvertently
    # overwriting developer checkouts. We'll uncomment this install command in
    # published notebooks in the future once we no longer require preserving the
    # editable install in the published runtime.
    install_runner_cell = {
        "cell_type": "code",
        "metadata": {"language": "python"},
        "source": [
            "# Uncomment to run installation steps if you do not have a development\n",
            "# editable install and want to run this notebook in a fresh environment.\n",
            "# %pip install uv\n",
            "# %uv pip install --upgrade pip setuptools wheel && \\\n",
            "# %uv pip install 'git+https://github.com/speediedan/interpretune.git@main[examples]'\n",
            "# %uv pip install --group git-deps\n",
            "#\n",
            "# NOTE: This cell is intentionally commented out. We will uncomment these\n",
            "# install commands once we no longer need to preserve editable installs\n",
            "# for active developer venvs.\n",
        ],
        "outputs": [],
        "execution_count": None,
    }

    cells = notebook.get("cells", [])
    notebook["cells"] = [badge_cell, install_runner_cell] + cells

    return notebook


def fix_import_paths(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """Fix import paths in notebooks to work from publish directory."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                source_text = "".join(source)
            else:
                source_text = source

            # Fix imports that reference dev paths
            if "it_examples.notebooks.dev." in source_text:
                source_text = source_text.replace("it_examples.notebooks.dev.", "it_examples.notebooks.publish.")

            # Fix directory path references from dev to publish
            if '"notebooks" / "dev"' in source_text:
                source_text = source_text.replace('"notebooks" / "dev"', '"notebooks" / "publish"')

            # Update the cell source while preserving formatting
            if isinstance(cell["source"], list):
                # Split by \n but keep the newlines
                lines = source_text.split("\n")
                if source_text.endswith("\n"):
                    lines = lines[:-1]  # Remove empty last line if it ends with \n
                cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
            else:
                cell["source"] = source_text

    return notebook


def publish_file(dev_path: Path, publish_path: Path, relative_path: str, dry_run: bool = False) -> bool:
    """Publish a file from dev to publish directory."""
    if dry_run:
        action = "PROCESS" if dev_path.suffix == ".ipynb" else "COPY"
        print(f"WOULD {action}: {dev_path} -> {publish_path}")
        return True

    # Ensure publish directory exists
    publish_path.parent.mkdir(parents=True, exist_ok=True)

    if dev_path.suffix == ".ipynb":
        # Process notebook: strip tags and add Colab badge/install cell
        notebook = load_notebook(dev_path)
        notebook = strip_remove_cell_tags(notebook)
        notebook = fix_import_paths(notebook)
        notebook = add_colab_badge_and_install_cell(notebook, relative_path)
        save_notebook(notebook, publish_path)
        print(f"PROCESSED: {dev_path} -> {publish_path}")
    else:
        # Just copy other files
        shutil.copy2(dev_path, publish_path)
        print(f"COPIED: {dev_path} -> {publish_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Publish notebooks from dev to publish directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check if any notebooks need publishing (returns non-zero if changes needed)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Publish all notebooks even if there are no detected changes (overrides hashes)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    dev_dir = repo_root / "src" / "it_examples" / "notebooks" / "dev"
    publish_dir = repo_root / "src" / "it_examples" / "notebooks" / "publish"

    if not dev_dir.exists():
        print(f"ERROR: Dev directory not found: {dev_dir}")
        return 1

    # Load stored hashes
    stored_hashes = load_notebook_hashes(publish_dir)

    # Get all changed files (notebooks and non-notebooks)
    changed_files = get_changed_files(dev_dir, stored_hashes, "*")

    # If --force is set, publish all files under dev_dir regardless of change
    if args.force:
        # Publish all *files* under dev_dir. Avoid including directories which
        # would cause shutil.copy2 to error. This ensures we only process files
        # while supporting --force to re-run publication logic.
        changed_files = {p for p in dev_dir.rglob("*") if p.is_file()}

    files_to_process = list(changed_files)

    if not files_to_process and not args.check_only:
        print("All files are up to date. No files to publish.")
        return 0

    if args.check_only:
        if files_to_process:
            notebooks_changed = [f for f in files_to_process if f.suffix == ".ipynb"]
            print(f"Found {len(files_to_process)} file(s) that need publishing:")
            if notebooks_changed:
                print(f"  Notebooks: {len(notebooks_changed)}")
            for f in sorted(files_to_process):
                rel_path = f.relative_to(dev_dir)
                print(f"  notebooks/dev/{rel_path}")
            return 1  # Non-zero exit code indicates changes needed
        else:
            print("All files are up to date.")
            return 0

    notebooks_to_publish = [f for f in files_to_process if f.suffix == ".ipynb"]
    print(f"Found {len(files_to_process)} files to publish")
    print(f"Source: {dev_dir}")
    print(f"Target: {publish_dir}")
    if notebooks_to_publish:
        print(f"Notebooks: {len(notebooks_to_publish)}")
    print()

    published_count = 0
    updated_hashes = stored_hashes.copy()

    for file_path in files_to_process:
        # Calculate relative path for Colab URL
        relative_path = file_path.relative_to(repo_root)
        publish_path = publish_dir / file_path.relative_to(dev_dir)

        if publish_file(file_path, publish_path, str(relative_path), args.dry_run):
            published_count += 1

            # Update hash for all files
            rel_path = file_path.relative_to(dev_dir)
            hash_key = f"notebooks/dev/{rel_path}"
            updated_hashes[hash_key] = compute_file_hash(file_path)

    # Save updated hashes (only if not dry run)
    if not args.dry_run and published_count > 0:
        save_notebook_hashes(publish_dir, updated_hashes)

    if args.dry_run:
        print(f"\nDry run complete. Would publish {published_count} files.")
    else:
        print(f"\nPublished {published_count} files.")

    return 0


if __name__ == "__main__":
    exit(main())
