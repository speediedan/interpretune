"""The ``it_component.yaml`` manifest: schema loading, validation, and configuration-key derivation.

The manifest is the one file every tool reads first (manifest-first resolution) and the download-count anchor
(``countDownloads: path:"it_component.yaml"`` at registration), so nothing here may require constructing any
entry: parsing stays cheap and side-effect free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

IT_COMPONENT_MANIFEST = "it_component.yaml"
SUPPORTED_SCHEMA_VERSIONS = (1,)
KNOWN_KINDS = ("module", "datamodule", "ops", "adapters", "promptconfigs")


class ComponentManifestError(ValueError):
    """A component manifest is missing, malformed, or declares an unsupported schema."""


def derive_config_key(cfg: dict) -> str:
    """Derive the canonical configuration key from a config's structured fields.

    ``<task_variant>.<model>.<composition>[.<descriptor>]`` — composition joined with ``+`` in the composition
    registry's canonical (alphabetical-by-value) order, ``core`` written explicitly, extensions surfaced as the
    derived descriptor when no explicit one is given. Derived descriptors are MATERIALIZED into keys and
    filenames at authoring/publish time — never resolved implicitly at load time.
    """
    from interpretune.protocol import Adapter
    from interpretune.adapter_registry import ADAPTER_REGISTRY

    comp = tuple(Adapter[a] if not isinstance(a, Adapter) else a for a in cfg.get("composition", ()))
    comp_str = "+".join(a.name for a in ADAPTER_REGISTRY.canonicalize_composition(comp)) or "core"
    node4 = cfg.get("descriptor") or (".".join(cfg["extensions"]) if cfg.get("extensions") else None)
    return f"{cfg['task_variant']}.{cfg['model']}.{comp_str}" + (f".{node4}" if node4 else "")


def validate_component_manifest(manifest: Any, source: str = "<manifest>") -> dict:
    """Validate the coarse shape of a parsed component manifest, returning it on success."""
    if not isinstance(manifest, dict):
        raise ComponentManifestError(f"{source}: manifest must be a mapping, got {type(manifest).__name__}")
    version = manifest.get("it_schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ComponentManifestError(
            f"{source}: unsupported it_schema_version {version!r} (supported: {SUPPORTED_SCHEMA_VERSIONS}) — "
            "an old manifest and a malformed one must be distinguishable, so the version header is mandatory."
        )
    kinds = manifest.get("kinds")
    if not kinds or not isinstance(kinds, list) or any(k not in KNOWN_KINDS for k in kinds):
        raise ComponentManifestError(f"{source}: `kinds` must be a non-empty subset of {KNOWN_KINDS}, got {kinds!r}")
    if "module" in kinds and "configs" not in (manifest.get("module") or {}):
        raise ComponentManifestError(f"{source}: kind `module` requires a `module.configs` index")
    if "ops" in kinds:
        ops = manifest.get("ops") or {}
        files = ops.get("files")
        if not files or not isinstance(files, list) or not all(isinstance(f, str) and f for f in files):
            raise ComponentManifestError(
                f"{source}: kind `ops` requires a non-empty `ops.files` list of repo-relative op-definition "
                "YAML paths. Op discovery is manifest-routed: a collection declares which YAMLs are op "
                "definitions rather than having every YAML in the repo treated as one."
            )
        if IT_COMPONENT_MANIFEST in files:
            raise ComponentManifestError(
                f"{source}: `ops.files` must not list {IT_COMPONENT_MANIFEST} itself. The manifest declares "
                "the op definitions; it is not one of them, and parsing it as one fails on its scalar keys."
            )
    if "datamodule" in kinds:
        dms = manifest.get("datamodules")
        if not dms or not isinstance(dms, dict):
            raise ComponentManifestError(
                f"{source}: kind `datamodule` requires a non-empty `datamodules` index "
                "(datamodule name -> entry). The manifest names entries; there are no reserved filenames."
            )
        for name, entry in dms.items():
            if not isinstance(entry, dict) or not entry.get("config") or not isinstance(entry["config"], str):
                raise ComponentManifestError(
                    f"{source}: datamodule entry {name!r} requires a repo-relative `config` path (its "
                    "standalone-consumption payload). Module configurations inline their own datamodule "
                    "configuration and never read this payload -- consumption is strictly two-path with "
                    "no merge semantics (#128)."
                )
    if "adapters" in kinds:
        ad = manifest.get("adapters") or {}
        entrypoint = ad.get("entrypoint")
        declares = ad.get("declares")
        if not entrypoint or not isinstance(entrypoint, str):
            raise ComponentManifestError(
                f"{source}: kind `adapters` requires an `adapters.entrypoint` (repo-relative .py that registers "
                "the adapter's compositions). Adapters are the only kind whose payload IS code."
            )
        if (
            not declares
            or not isinstance(declares, list)
            or not all(isinstance(d, str) and d.isidentifier() for d in declares)
        ):
            raise ComponentManifestError(
                f"{source}: kind `adapters` requires a non-empty `adapters.declares` list of Adapter names this "
                "component adds (valid Python identifiers). The manifest declares the names so the loader can "
                "check that the code registered what the manifest advertised, rather than discovering the "
                "adapter surface by executing it."
            )
        for entry in ad.get("compositions") or []:
            if not isinstance(entry, dict) or not entry.get("component") or not isinstance(entry.get("adapters"), list):
                raise ComponentManifestError(
                    f"{source}: each `adapters.compositions` entry needs a `component` name and an `adapters` "
                    f"list, got {entry!r}"
                )
            # OPTIONAL per-composition `requires`: SOFT, unlike the component-wide block. Unmet means this
            # ONE composition is skipped and the others still register; unmet component-wide means the
            # component does not load at all. Same vocabulary either way, so a component can STATE its
            # conditionality rather than overstate its surface -- a manifest advertising three compositions
            # while the environment yields two, with nothing saying so, is #431 in the declarative layer.
            entry_requires = entry.get("requires")
            if entry_requires is not None and not isinstance(entry_requires, dict):
                raise ComponentManifestError(
                    f"{source}: `adapters.compositions[].requires` must be a mapping in the same vocabulary as "
                    f"the component-wide `requires` (interpretune / adapters / pip), got {entry_requires!r}"
                )
    if "promptconfigs" in kinds:
        pc = manifest.get("promptconfigs") or {}
        if not pc.get("entrypoint") or not isinstance(pc.get("definitions"), dict) or not pc["definitions"]:
            raise ComponentManifestError(
                f"{source}: kind `promptconfigs` requires a `promptconfigs.entrypoint` and a non-empty "
                "`promptconfigs.definitions` index (definition name -> metadata; one repo, many definitions)."
            )
    return manifest


def load_component_manifest(manifest_path: Path) -> dict:
    """Load and validate an ``it_component.yaml`` from disk."""
    with open(manifest_path, encoding="utf-8") as fh:
        return validate_component_manifest(yaml.safe_load(fh), source=str(manifest_path))


def check_config_key_parity(config_path: Path, body: dict, expected_key: str | None = None) -> str:
    """Enforce filename == manifest key == key derived from structured fields; return the canonical key."""
    derived = derive_config_key(body)
    stem = config_path.name[: -len(".yaml")]
    if derived != stem or (expected_key is not None and derived != expected_key):
        raise ValueError(
            f"Configuration key parity violation for {config_path}: filename stem {stem!r}, "
            f"manifest key {expected_key!r}, derived-from-fields {derived!r} must all match."
        )
    return derived


def iter_component_manifests(registry_root: Path | None):
    """Yield ``(component_dir, parsed_manifest)`` for every component tree under ``registry_root``."""
    if registry_root is None or not Path(registry_root).is_dir():
        return
    for manifest_path in sorted(Path(registry_root).glob(f"*/{IT_COMPONENT_MANIFEST}")):
        yield manifest_path.parent, load_component_manifest(manifest_path)


def load_config_file(config_path: Path, expected_key: str | None = None) -> tuple[str, dict]:
    """Load one configuration file, parity-checking filename == manifest key == derived-from-fields."""
    with open(config_path, encoding="utf-8") as fh:
        body = yaml.safe_load(fh)
    return check_config_key_parity(Path(config_path), body, expected_key=expected_key), body
