"""Assemble and publish component repos from in-repo component trees.

The in-repo tree (``src/it_examples/examples/<task>/``) mirrors the Hub tree by construction, so publishing is
a copy plus two generated additions: the entrypoint module file and the card. A parity test walks the built
tree against the source tree so the mirror is enforced rather than aspirational.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from interpretune.analysis.ops.collection import COLLECTION_HEADER_KEY
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
    # promptconfigs entrypoints are SELF-CONTAINED by design (no in-repo package imports), so the
    # source lives inside the component dir itself rather than arriving via entrypoint_src
    pc_entrypoint = (manifest.get("promptconfigs") or {}).get("entrypoint")
    if pc_entrypoint:
        pc_src = component_dir / pc_entrypoint
        if not pc_src.is_file():
            raise FileNotFoundError(
                f"Manifest declares promptconfigs entrypoint {pc_entrypoint!r} but "
                f"{pc_src} does not exist (promptconfigs entrypoints live inside the component dir)."
            )
        shutil.copy2(pc_src, out_dir / pc_entrypoint)
    return manifest


def build_op_collection_tree(
    family_dir: Path,
    out_dir: Path,
    repo_id: str,
    collection_name: str | None = None,
    collection_version: str | None = None,
    requires: dict | None = None,
) -> dict:
    """Build a publishable ops repo from an in-tree bundled op family; returns the generated manifest.

    The flagship seed collection is GENERATED from the bundled family rather than maintained beside it (design
    §3.9, D9), so the bundled copy stays the single source and the two cannot drift. One transformation is
    mandatory, not cosmetic: bundled YAMLs address implementations by installed package path
    (``interpretune.analysis.ops.bundled.<family>.<module>.<fn>``), while the hub loader resolves a
    repo-relative ``<module>.<fn>`` pair through the dynamic-module path. Publishing a family verbatim would
    produce a repo whose every op fails to import.

    The published YAML is dumped rather than copied, so it carries a generated-file banner instead of the
    source's comments: an editable-looking copy of a generated file is how single-sourcing quietly dies.
    """
    import yaml

    family_dir, out_dir = Path(family_dir), Path(out_dir)
    family = family_dir.name
    yaml_candidates = sorted(p for p in family_dir.glob("*.yaml") if p.name != IT_COMPONENT_MANIFEST)
    if len(yaml_candidates) != 1:
        raise ValueError(f"expected exactly one op-definitions YAML in {family_dir}, found {yaml_candidates}")
    source_yaml = yaml_candidates[0]
    content = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}

    package_prefix = f"interpretune.analysis.ops.bundled.{family}."
    modules = _rewrite_implementation_paths(content, package_prefix)
    if not modules:
        raise ValueError(f"{source_yaml} declares no implementation under {package_prefix!r}; nothing to publish")

    collection = dict(content.get(COLLECTION_HEADER_KEY) or {})
    # Default the collection handle to the REPO name: a hub copy sharing the bundled family's handle makes
    # `op_info` print the same collection for both, which is exactly the comparison it exists to support.
    collection["name"] = collection_name or repo_id.split("/", 1)[-1]
    # The version tracks the bundled family's by default -- the collection is generated from it, so an
    # independent version would be a claim the single-sourcing cannot back. Pass one explicitly to ship a
    # contract set that has genuinely moved ahead.
    if collection_version:
        collection["version"] = collection_version
    if requires is not None:
        collection["requires"] = requires
    content[COLLECTION_HEADER_KEY] = collection

    out_dir.mkdir(parents=True, exist_ok=True)
    published_yaml = out_dir / source_yaml.name
    banner = (
        f"# GENERATED by interpretune.hub.publish.build_op_collection_tree -- do not edit.\n"
        f"# Source of truth: src/interpretune/analysis/ops/bundled/{family}/{source_yaml.name}\n"
        f"# `implementation:` paths are rewritten to the repo-relative `<module>.<function>` form the hub\n"
        f"# loader resolves; the bundled copy addresses the same functions by installed package path.\n"
    )
    published_yaml.write_text(banner + yaml.safe_dump(content, sort_keys=False), encoding="utf-8")

    for module in sorted(modules):
        module_src = family_dir / f"{module}.py"
        if not module_src.is_file():
            raise FileNotFoundError(f"{source_yaml} references module {module!r} but {module_src} does not exist")
        shutil.copy2(module_src, out_dir / module_src.name)

    manifest = {
        "it_schema_version": 1,
        "kinds": ["ops"],
        "ops": {"files": [published_yaml.name]},
    }
    if collection.get("requires"):
        manifest["requires"] = collection["requires"]
    (out_dir / IT_COMPONENT_MANIFEST).write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return load_component_manifest(out_dir / IT_COMPONENT_MANIFEST)


def _rewrite_implementation_paths(content: dict, package_prefix: str) -> set[str]:
    """Rewrite in-place to repo-relative implementations; returns the module basenames referenced."""
    modules: set[str] = set()

    def rewrite(value: str) -> str:
        if not isinstance(value, str) or not value.startswith(package_prefix):
            return value
        relative = value[len(package_prefix) :]
        modules.add(relative.rsplit(".", 1)[0])
        return relative

    for op_name, op_def in content.items():
        if op_name == COLLECTION_HEADER_KEY or not isinstance(op_def, dict):
            continue
        if "implementation" in op_def:
            op_def["implementation"] = rewrite(op_def["implementation"])
        params = op_def.get("importable_params")
        if isinstance(params, dict):
            op_def["importable_params"] = {name: rewrite(path) for name, path in params.items()}
    return modules


def publish_op_collection(
    family_dir: Path,
    repo_id: str,
    build_dir: Path | None = None,
    collection_name: str | None = None,
    collection_version: str | None = None,
    requires: dict | None = None,
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Build an ops repo from a bundled family, card it, and upload; returns the commit sha."""
    import tempfile

    from interpretune.hub.manager import OPS_KIND, ITHubResourceManager

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(build_dir) if build_dir is not None else Path(tmp) / "build"
        manifest = build_op_collection_tree(
            family_dir,
            out_dir,
            repo_id,
            collection_name=collection_name,
            collection_version=collection_version,
            requires=requires,
        )
        generate_component_card(manifest, repo_id).save(out_dir / "README.md")
        manager = ITHubResourceManager(kind=OPS_KIND, token=token)
        return manager.upload(
            out_dir,
            repo_id,
            private=private,
            clean_existing=True,
            commit_message=commit_message or f"Publish interpretune op collection from the {family_dir.name} family",
        )


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
