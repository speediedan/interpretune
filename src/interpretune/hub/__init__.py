"""interpretune.hub — the unified Hub client layer (hub-integration design, ACCEPTED v2).

Kind-agnostic machinery for every interpretune Hub resource: cache roots and cache-scan discovery
(`.cache`), the ``it_component.yaml`` manifest schema and configuration-key derivation (`.manifest`),
generated discovery-sentinel cards (`.cards`), the generalized resource manager (`.manager`, relocated from
``interpretune.analysis.ops.hub_manager``), manifest-first component fetch (`.components`), and component
publishing (`.publish`).

Attribute access is lazy (PEP 562) so importing ``interpretune.hub`` stays cheap.
"""

from __future__ import annotations

from typing import Any

_LAZY_ATTRS = {
    "IT_CACHE": "interpretune.hub.cache",
    "IT_COMPONENTS_HUB_CACHE": "interpretune.hub.cache",
    "IT_ARTIFACTS_HUB_CACHE": "interpretune.hub.cache",
    "IT_ANALYSIS_HUB_CACHE": "interpretune.hub.cache",
    "parse_hub_cache_path": "interpretune.hub.cache",
    "scan_cached_repos": "interpretune.hub.cache",
    "IT_COMPONENT_MANIFEST": "interpretune.hub.manifest",
    "ComponentManifestError": "interpretune.hub.manifest",
    "derive_config_key": "interpretune.hub.manifest",
    "load_component_manifest": "interpretune.hub.manifest",
    "validate_component_manifest": "interpretune.hub.manifest",
    "check_config_key_parity": "interpretune.hub.manifest",
    "component_card_metadata": "interpretune.hub.cards",
    "generate_component_card": "interpretune.hub.cards",
    "HubResourceKind": "interpretune.hub.manager",
    "ITHubResourceManager": "interpretune.hub.manager",
    "HubAnalysisOpManager": "interpretune.hub.manager",
    "HubOpCollection": "interpretune.hub.manager",
    "OPS_KIND": "interpretune.hub.manager",
    "COMPONENT_KIND": "interpretune.hub.manager",
    "ComponentRequirementError": "interpretune.hub.components",
    "enforce_component_requires": "interpretune.hub.components",
    "pull_component_manifest": "interpretune.hub.components",
    "pull_component_config": "interpretune.hub.components",
    "resolve_component_manifest": "interpretune.hub.components",
    "resolve_component_config": "interpretune.hub.components",
    "register_component_config": "interpretune.hub.components",
    "local_publish": "interpretune.hub.components",
    "RemoteCodeNotTrustedError": "interpretune.hub.trust",
    "remote_code_trust": "interpretune.hub.trust",
    "ensure_remote_code_trusted": "interpretune.hub.trust",
    "import_cached_entrypoint": "interpretune.hub.promptconfigs",
    "resolve_prompt_config_class": "interpretune.hub.promptconfigs",
    "compose_prompt_config_class": "interpretune.hub.promptconfigs",
    "instantiate_prompt_cfg_node": "interpretune.hub.promptconfigs",
    "IT_ARTIFACT_ENVELOPE": "interpretune.hub.artifacts",
    "ArtifactEnvelopeError": "interpretune.hub.artifacts",
    "validate_artifact_envelope": "interpretune.hub.artifacts",
    "build_analysis_store_envelope": "interpretune.hub.artifacts",
    "content_fingerprint": "interpretune.hub.artifacts",
    "push_analysis_store": "interpretune.hub.artifacts",
    "pull_analysis_store": "interpretune.hub.artifacts",
    "describe_analysis_store": "interpretune.hub.artifacts",
    "generate_artifact_card": "interpretune.hub.cards",
    "build_component_tree": "interpretune.hub.publish",
    "publish_component": "interpretune.hub.publish",
    # the ratified verb surface (design 5e): it.hub.pull / it.hub.load / it.hub.push
    "pull": "interpretune.hub.api",
    "load": "interpretune.hub.api",
    "push": "interpretune.hub.api",
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import importlib

        module = importlib.import_module(_LAZY_ATTRS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
