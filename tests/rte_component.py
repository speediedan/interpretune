"""RTEBoolq experiment classes, resolved from the Hub-resident component.

The experiment moved out of the tree: ``speediedan/rte`` is the only source of truth.
Class bindings resolve lazily (PEP 562) so importing this module costs nothing: only the
attribute actually requested triggers the cache-only snapshot import (warmed in CI from the
manifest; one ``it.hub.pull('speediedan/rte')`` ever for a dev checkout). An uncached component
fails there, loudly, with the fetch that fixes it -- never with a stale or second source.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

_KNOWN = sorted(
    [
        "RTE_ENTRYPOINT_MODULE",
        "RTEBoolqConfig",
        "RTEBoolqCTConfig",
        "RTEBoolqDataModule",
        "RTEBoolqEntailmentMapping",
        "RTEBoolqGenerativeClassificationConfig",
        "RTEBoolqModule",
        "RTEBoolqModuleMixin",
        "RTEBoolqNNsightConfig",
        "RTEBoolqPromptConfig",
        "RTEBoolqChatTemplatePromptConfig",
        "RTEBoolqSLConfig",
        "RTEBoolqSteps",
        "RTEBoolqTLConfig",
        "TASK_TEXT_FIELD_MAP",
    ]
)


def rte_entrypoint_src() -> Path:
    """``rte_boolq.py`` as shipped by the warmed ``speediedan/rte`` snapshot.

    Fixtures needing the entrypoint FILE (local publishes, build-parity checks) source it from the warmed default cache
    rather than the deleted in-tree copy.
    """
    from interpretune.hub.components import resolve_component_manifest

    _, snapshot, _ = resolve_component_manifest("speediedan/rte")
    return snapshot / "rte_boolq.py"


def rte_entrypoint_module() -> ModuleType:
    """Import the cached ``speediedan/rte`` entrypoint (or the component owning the stem)."""
    from interpretune.hub.entrypoints import find_entrypoint_owner, import_snapshot_entrypoint

    owner = find_entrypoint_owner("rte_boolq")
    if owner is None:
        raise ImportError(
            "The RTE experiment is Hub-resident now (speediedan/rte) and is not in the local "
            "components cache. Fetch it once with it.hub.pull('speediedan/rte') (CI warms it "
            "from tests/hf_warm_manifest.yaml)."
        )
    repo_id, entrypoint = owner
    return import_snapshot_entrypoint(
        repo_id, entrypoint, what=f"the experiment entrypoint {entrypoint!r} of {repo_id!r}"
    )


def __getattr__(name: str):
    if name == "RTE_ENTRYPOINT_MODULE":
        return rte_entrypoint_module()
    if name in _KNOWN:
        return getattr(rte_entrypoint_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(_KNOWN) | {"rte_entrypoint_src", "rte_entrypoint_module"})
