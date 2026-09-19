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
from interpretune.hub.manifest import (
    IT_COMPONENT_MANIFEST,
    ComponentManifestError,
    check_config_key_parity,
    load_component_manifest,
)


#: Never staged, whatever a declared directory contains: bytecode and tool caches are not part of any artifact.
STAGING_IGNORES = ("__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".DS_Store")


def source_revision_of(component_dir: Path) -> str | None:
    """The revision of the component source being published: the last commit touching ``component_dir``.

    Computed by :func:`interpretune.hub.revisions.directory_revision`, the same function the conformance suite uses
    to key its report, so a report measured before an unrelated commit elsewhere in the repository still matches.
    ``None`` when the directory is not tracked in a checkout, which makes the card's comparison fail closed.
    """
    from interpretune.hub.revisions import directory_revision

    return directory_revision(component_dir)


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
    # experiment definitions: the config carries the parity check (filename == key ==
    # EXPERIMENT_NAME); pipeline and owned files copy verbatim under the same allowlist rule
    from interpretune.hub.manifest import check_experiment_key_parity

    import yaml

    for key, entry in (manifest.get("experiments") or {}).items():
        cfg_src = component_dir / entry["config"]
        if not cfg_src.is_file():
            raise FileNotFoundError(
                f"Manifest declares experiment config {entry['config']!r} for {key!r}, which is not "
                f"present in {component_dir}."
            )
        check_experiment_key_parity(cfg_src, yaml.safe_load(cfg_src.read_text(encoding="utf-8")), expected_key=key)
        dest = out_dir / entry["config"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg_src, dest)
        for rel in [entry.get("pipeline"), *(entry.get("files") or [])]:
            if not rel:
                continue
            src = component_dir / rel
            if not src.exists():
                raise FileNotFoundError(
                    f"Manifest declares experiment payload {rel!r} for {key!r}, which is not present "
                    f"in {component_dir}."
                )
            dest = out_dir / rel
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*STAGING_IGNORES))
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
    for rel in (manifest.get("ops") or {}).get("files") or []:
        shutil.copy2(component_dir / rel, out_dir / rel)
    # Named snapshot rewrites run after the verbatim stage: the staged tree is rearranged into its
    # runnable layout (moves, import swaps, generated package files) and the staged manifest is
    # updated to describe the tree it ships with. Unmarked manifests stage verbatim, as before.
    rewrite_name = manifest.get("experiment_snapshot_rewrite")
    if rewrite_name is not None:
        try:
            rewrite = EXPERIMENT_SNAPSHOT_REWRITES[rewrite_name]
        except KeyError:
            raise ComponentManifestError(
                f"{component_dir / IT_COMPONENT_MANIFEST}: unknown `experiment_snapshot_rewrite` "
                f"{rewrite_name!r} (available: {sorted(EXPERIMENT_SNAPSHOT_REWRITES)})."
            ) from None
        rewrite(out_dir, manifest)
    # hookmaps documents are data, but data with a schema: a document that does not parse into a
    # ComponentMap is refused HERE, at publish, rather than by the first consumer to load it.
    for rel in (manifest.get("hookmaps") or {}).get("files") or []:
        src = component_dir / rel
        if not src.is_file():
            raise FileNotFoundError(
                f"Manifest declares hookmaps document {rel!r}, which is not present in {component_dir}."
            )
        from interpretune.analysis.points.component_map import load_component_map_file

        load_component_map_file(src)
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    # adapters entrypoints are SELF-CONTAINED for the same reason promptconfigs' are: the file must be
    # readable and runnable straight out of the snapshot, with no in-repo package to import from.
    adapters_entrypoint = (manifest.get("adapters") or {}).get("entrypoint")
    if adapters_entrypoint:
        src = component_dir / adapters_entrypoint
        if not src.is_file():
            raise FileNotFoundError(
                f"Manifest declares adapters entrypoint {adapters_entrypoint!r}, which is not present in "
                f"{component_dir}."
            )
        dest = out_dir / adapters_entrypoint
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    # Declared supplementary files (`extra_files`): a collection's tests, a README fragment, fixture data. The
    # publish path is a strict allowlist of manifest-declared paths, so before this the only way to publish a
    # tests/ directory was a hand-push, which carries whatever else sits in the working directory (a
    # .pytest_cache/ tree reached the first published collection that way). Declared here, they are staged by
    # the same builder, so local_publish, the card and the Hub tree all agree on what the artifact contains.
    for rel in manifest.get("extra_files") or []:
        src = (component_dir / rel).resolve()
        if component_dir.resolve() not in src.parents and src != component_dir.resolve():
            raise ValueError(f"extra_files entry {rel!r} resolves outside {component_dir}; refusing to publish it")
        if not src.exists():
            raise FileNotFoundError(f"Manifest declares extra file {rel!r}, which is not present in {component_dir}.")
        dest = out_dir / rel
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*STAGING_IGNORES))
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

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

    A published seed collection is GENERATED from the bundled family rather than maintained beside it (design
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


#: Carried (not generated) prompt-config shim, injected into rewritten experiment snapshots so the
#: pipeline resolves its chat spelling from the published prompt-configs component instead of the
#: in-repo examples package. Kept tiny and stable on purpose: it is build machinery versioned with
#: the rewriter below, not per-experiment hand maintenance.
_EXPERIMENT_PROMPT_SHIM_TEXT = '''"""Chat spelling resolved from the published prompt-configs component (cache-only)."""


def GemmaPromptConfig():  # noqa: N802 - matches the published definition's name
    """The Gemma chat-spelling config class, without importing the examples package."""
    from interpretune.hub.promptconfigs import resolve_prompt_config_class

    return resolve_prompt_config_class("speediedan/prompt-configs#GemmaPromptConfig")()
'''


def _module_level_it_examples_refs(path: Path) -> list[str]:
    """`it_examples.*` imports NOT nested in a function: the import-time snapshot blockers."""
    import ast

    found: list[str] = []

    def visit(node: ast.AST, nested: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                mods = (
                    [a.name for a in child.names]
                    if isinstance(child, ast.Import)
                    else ([child.module] if child.module else [])
                )
                for mod in mods:
                    if mod == "it_examples" or (mod or "").startswith("it_examples."):
                        if not nested:
                            found.append(f"{path.name}:{child.lineno}:{mod}")
            else:
                visit(child, nested or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))

    visit(ast.parse(path.read_text(encoding="utf-8")), False)
    return found


def _rewrite_concept_direction_snapshot(out_dir: Path, manifest: dict) -> None:
    """Restructure a staged concept-direction tree into the runnable ``exp/`` package layout.

    The source tree keeps the in-repo layout (configs beside the pipeline sources); the snapshot
    instead carries a single importable package, because the snapshot root is what lands on
    ``sys.path`` and bare module names there risk shadowing. Every transformation is enumerated:
    moves, exact import swaps, generated ``__init__`` files plus the prompt shim above, and a
    staged-manifest path update so the manifest describes the tree it ships with. Anything else
    naming ``it_examples`` at module level is refused rather than guessed at.
    """
    import yaml

    moves = {
        "concept_direction/concept_direction.py": "exp/concept_direction.py",
        "pipeline_patterns.py": "exp/pipeline_patterns.py",
        "concept_direction/analysis/concept_direction_analysis.py": "exp/analysis/concept_direction_analysis.py",
        "concept_direction/analysis/intervention_drift_analysis.py": ("exp/analysis/intervention_drift_analysis.py"),
    }
    swaps = [
        (
            "from it_examples.examples.prompt_configs.prompt_configs import GemmaPromptConfig",
            "from exp._prompt_shim import GemmaPromptConfig",
        ),
        (
            "from it_examples.experiments.notebook.pipeline_patterns import (",
            "from exp.pipeline_patterns import (",
        ),
        (
            "from it_examples.experiments.notebook.concept_direction.analysis.concept_direction_analysis import (",
            "from exp.analysis.concept_direction_analysis import (",
        ),
        (
            "from it_examples.experiments.notebook.concept_direction.analysis.intervention_drift_analysis import (",
            "from exp.analysis.intervention_drift_analysis import (",
        ),
        (
            "from it_examples.experiments.notebook.concept_direction.concept_direction import NotebookHarnessConfig",
            "from exp.concept_direction import NotebookHarnessConfig",
        ),
    ]
    (out_dir / "exp" / "analysis").mkdir(parents=True, exist_ok=True)
    for src_rel, dest_rel in moves.items():
        src, dest = out_dir / src_rel, out_dir / dest_rel
        if not src.is_file():
            raise FileNotFoundError(
                f"concept-direction snapshot rewrite needs staged {src_rel!r}, which the manifest did not stage."
            )
        text = src.read_text(encoding="utf-8")
        for old, new in swaps:
            if old in text:
                text = text.replace(old, new)
        dest.write_text(text, encoding="utf-8")
        src.unlink()
    for generated, body in [
        ("exp/__init__.py", ""),
        ("exp/analysis/__init__.py", ""),
        ("exp/_prompt_shim.py", _EXPERIMENT_PROMPT_SHIM_TEXT),
    ]:
        (out_dir / generated).write_text(body, encoding="utf-8")

    blockers: list[str] = []
    for staged in sorted((out_dir / "exp").rglob("*.py")):
        if staged.name == "__init__.py":
            continue
        blockers.extend(f"{staged.relative_to(out_dir)}:{ref}" for ref in _module_level_it_examples_refs(staged))
    if blockers:
        raise ValueError(
            "concept-direction snapshot rewrite refuses unmapped it_examples imports "
            f"(add an explicit swap or carry the source): {sorted(blockers)}"
        )

    for entry in (manifest.get("experiments") or {}).values():
        if not isinstance(entry, dict):
            continue
        for field in ("pipeline",):
            rel = entry.get(field)
            if rel in moves:
                entry[field] = moves[rel]
        entry["files"] = [moves.get(rel, rel) for rel in entry.get("files") or []]
    (out_dir / IT_COMPONENT_MANIFEST).write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


#: Snapshot rewrites by manifest-declared name. A name with no entry here is refused at build time
#: (naming what exists), so manifests can never silently select a rewrite that is not implemented.
EXPERIMENT_SNAPSHOT_REWRITES = {
    "concept-direction-v1": _rewrite_concept_direction_snapshot,
}


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
    """Build the Hub tree, generate its card, and upload; returns the commit sha.

    The published artifact is exactly the manifest's allowlist (its declared payloads and entrypoints) plus the
    manifest's ``extra_files``, plus the generated card. Files already on the Hub that this publish would not
    produce are reported and removed, so a rename cannot leave its old name published and nothing arrives out
    of band.
    """
    import tempfile

    from interpretune.hub.manager import COMPONENT_KIND, ITHubResourceManager

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(build_dir) if build_dir is not None else Path(tmp) / "build"
        manifest = build_component_tree(component_dir, out_dir, entrypoint_src=entrypoint_src)
        generate_component_card(
            manifest, repo_id, tree=out_dir, source_revision=source_revision_of(component_dir)
        ).save(out_dir / "README.md")
        manager = ITHubResourceManager(kind=COMPONENT_KIND, token=token)
        # The published tree is made to MATCH the staged one, not merely to receive it: an upload that only
        # adds leaves a renamed entrypoint's old name live beside the new one, and a hand-pushed cache
        # directory live forever. Anything the current publish would not produce is reported and removed.
        return manager.upload(
            out_dir,
            repo_id,
            private=private,
            match_staged=True,
            commit_message=commit_message or f"Publish interpretune component {component_dir.name}",
        )
