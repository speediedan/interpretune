"""RTEBoolq experiment classes, resolved from the Hub-resident component.

The experiment moved out of the tree: ``speediedan/rte`` is the only source of truth.
Class bindings resolve lazily (PEP 562) so importing this module costs nothing: only the
attribute actually requested triggers the cache-only snapshot import (warmed in CI from the
manifest; one ``it.hub.pull('speediedan/rte')`` ever for a dev checkout). An uncached component
fails there, loudly, with the fetch that fixes it -- never with a stale or second source.
"""

from __future__ import annotations

import functools
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

    Fixtures needing the entrypoint FILE (local publishes, build-parity checks) source it from the warmed default cache,
    at the pinned revision, rather than an in-tree copy.
    """
    from interpretune.hub.components import resolve_component_manifest

    from interpretune.testing.conformance.inputs import SEED_REPO, SEED_REVISION

    _, snapshot, _ = resolve_component_manifest(SEED_REPO, revision=SEED_REVISION)
    return snapshot / "rte_boolq.py"


@functools.cache
def rte_component_source() -> Path:
    """The publish source of the pinned ``speediedan/rte`` revision, as a directory a test may build or publish
    from.

    The component's source is ``components/rte`` in interpretune/examples and its published copy is the warmed
    snapshot. A source tree lacks what publishing adds (the entrypoint, passed as ``entrypoint_src``, and the generated
    card), so those are left out, with caches. The copy lives for the process and is shared: a test that edits it copies
    it first.
    """
    import atexit
    import shutil
    import tempfile

    from interpretune.hub.components import resolve_component_manifest
    from interpretune.testing.conformance.inputs import SEED_REPO, SEED_REVISION

    manifest, snapshot, _ = resolve_component_manifest(SEED_REPO, revision=SEED_REVISION)
    root = Path(tempfile.mkdtemp(prefix="it_rte_source_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    generated = (manifest["module"]["entrypoint"], "README.md", ".gitattributes", "__pycache__")
    return Path(shutil.copytree(snapshot, root / "rte", ignore=shutil.ignore_patterns(*generated)))


def rte_component_tree(dest: Path) -> Path:
    """A component tree holding ``<dest>/rte``, copied from the warmed ``speediedan/rte`` snapshot; returns
    ``dest``.

    Hydrator-mode registry tests need a tree on disk to index, copy or mutate. It comes from the pinned seed revision
    rather than a source tree here, so those tests do not depend on ``it_examples``.
    """
    import shutil

    from interpretune.hub.components import resolve_component_manifest
    from interpretune.testing.conformance.inputs import SEED_REPO, SEED_REVISION

    _, snapshot, _ = resolve_component_manifest(SEED_REPO, revision=SEED_REVISION)
    shutil.copytree(snapshot, dest / "rte", ignore=shutil.ignore_patterns("__pycache__"))
    return dest


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
    return sorted(
        set(_KNOWN) | {"rte_entrypoint_src", "rte_entrypoint_module", "rte_component_source", "rte_component_tree"}
    )
