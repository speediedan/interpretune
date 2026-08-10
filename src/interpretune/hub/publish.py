"""Assemble and publish component repos from in-repo component trees.

The in-repo tree (``src/it_examples/examples/<task>/``) mirrors the Hub tree by construction, so publishing is
a copy plus two generated additions: the entrypoint module file and the card. A parity test walks the built
tree against the source tree so the mirror is enforced rather than aspirational.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from interpretune.hub.cards import generate_component_card
from interpretune.hub.manifest import IT_COMPONENT_MANIFEST, check_config_key_parity, load_component_manifest


def build_component_tree(component_dir: Path, out_dir: Path, entrypoint_src: Path | None = None) -> dict:
    """Build a publishable Hub tree from an in-repo component dir; returns the validated manifest.

    Copies the manifest and payload files verbatim (parity-checking every indexed configuration), copies the
    module entrypoint from ``entrypoint_src`` when the manifest declares one, and writes the generated card.
    """
    component_dir, out_dir = Path(component_dir), Path(out_dir)
    manifest = load_component_manifest(component_dir / IT_COMPONENT_MANIFEST)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(component_dir / IT_COMPONENT_MANIFEST, out_dir / IT_COMPONENT_MANIFEST)

    module_section = manifest.get("module") or {}
    for key, rel in (module_section.get("configs") or {}).items():
        src = component_dir / rel
        import yaml

        check_config_key_parity(src, yaml.safe_load(src.read_text(encoding="utf-8")), expected_key=key)
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    # non-module payloads (datamodule standalone configs, op files) copy verbatim
    for section in ("datamodules",):
        for entry in (manifest.get(section) or {}).values():
            rel = entry.get("config")
            if rel and (component_dir / rel).exists():
                dest = out_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(component_dir / rel, dest)
    for rel in (manifest.get("ops") or {}).get("files") or []:
        shutil.copy2(component_dir / rel, out_dir / rel)

    entrypoint = module_section.get("entrypoint")
    if entrypoint:
        if entrypoint_src is None or not Path(entrypoint_src).is_file():
            raise FileNotFoundError(
                f"Manifest declares entrypoint {entrypoint!r} but no entrypoint_src was provided/found "
                f"({entrypoint_src})."
            )
        shutil.copy2(entrypoint_src, out_dir / entrypoint)
    return manifest


def publish_component(
    component_dir: Path,
    repo_id: str,
    entrypoint_src: Path | None = None,
    build_dir: Path | None = None,
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Build the Hub tree, generate its card, and upload; returns the commit sha."""
    import tempfile

    from interpretune.hub.manager import COMPONENT_KIND, ITHubResourceManager

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(build_dir) if build_dir is not None else Path(tmp) / "build"
        manifest = build_component_tree(component_dir, out_dir, entrypoint_src=entrypoint_src)
        generate_component_card(manifest, repo_id).save(out_dir / "README.md")
        manager = ITHubResourceManager(kind=COMPONENT_KIND, token=token)
        return manager.upload(
            out_dir,
            repo_id,
            private=private,
            commit_message=commit_message or f"Publish interpretune component {component_dir.name}",
        )
