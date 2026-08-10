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
KNOWN_KINDS = ("module", "datamodule", "ops", "adapters")


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
