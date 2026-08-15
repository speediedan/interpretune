"""Cross-repo prompt-config references: ``compose_ref: <org>/<repo>#<name>`` (hub design §11.5).

The one reference mechanism for composing a published model-prompt definition with a task-level prompt
schema. A ``prompt_cfg`` node stays a valid one-grammar ``class_path`` node — ``compose_ref`` EXTENDS
its instantiation rather than replacing it, so the node's meaning is statically readable without a
fetch. Resolution is CACHE-ONLY (the no-implicit-network invariant): referenced repos must be cached
via an explicit ``it.hub.pull`` or the local-publish bridge.

``compose_ref`` is deliberately named: a future bare ``ref:`` is reserved for REPLACEMENT references
(datamodule refs, interpretune#128), so one mechanism carries two declared intents.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def import_cached_entrypoint(repo_id: str, cache_dir: Path | None = None) -> ModuleType:
    """Import a cached component's ``promptconfigs`` entrypoint under a revision-scoped module name.

    The synthetic module name incorporates org, repo AND revision so definitions from different cached
    revisions of the same repo can never collide in ``sys.modules`` as repos update.
    """
    from interpretune.hub.components import resolve_component_manifest

    manifest, snapshot, revision = resolve_component_manifest(repo_id, cache_dir=cache_dir)
    pc = manifest.get("promptconfigs") or {}
    entrypoint = pc.get("entrypoint")
    if not entrypoint:
        raise KeyError(f"{repo_id} (cached) declares no promptconfigs entrypoint (kinds: {manifest.get('kinds')}).")
    sanitized = repo_id.replace("/", "__").replace("-", "_").replace(".", "_")
    module_name = f"it_hub_components.{sanitized}.{revision}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    # the gate belongs HERE, at the point of execution: this is the one place outside op discovery
    # where interpretune runs Python that came from a hub repo (interpretune#255)
    from interpretune.hub.trust import ensure_remote_code_trusted

    ensure_remote_code_trusted(repo_id, what=f"the prompt-config entrypoint {entrypoint!r}")
    spec = importlib.util.spec_from_file_location(module_name, snapshot / entrypoint)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_prompt_config_class(ref: str, cache_dir: Path | None = None) -> type:
    """Resolve ``<org>/<repo>#<name>`` to the definition class from the cached entrypoint."""
    repo_id, sep, name = ref.partition("#")
    if not sep or not name or "/" not in repo_id:
        raise ValueError(f"Malformed prompt-config reference {ref!r}: expected '<org>/<repo>#<DefinitionName>'.")
    from interpretune.hub.components import resolve_component_manifest

    manifest, _, _ = resolve_component_manifest(repo_id, cache_dir=cache_dir)
    definitions = (manifest.get("promptconfigs") or {}).get("definitions") or {}
    if name not in definitions:
        raise KeyError(f"{repo_id} declares no prompt-config definition {name!r}. Available: {sorted(definitions)}")
    module = import_cached_entrypoint(repo_id, cache_dir=cache_dir)
    return getattr(module, name)


def compose_prompt_config_class(ref_cls: type, task_schema_cls: type | None) -> type:
    """Synthesize the composed prompt-config class, ``(RefClass, TaskSchema)`` MRO order.

    Mirrors the dissolving hand-written compositions exactly (e.g. ``RTEBoolqGemmaPromptConfig =
    (GemmaPromptConfig, RTEBoolqPromptConfig)``) — the base order is pinned by an MRO-equivalence
    test against those classes.
    """
    if task_schema_cls is None:
        return ref_cls
    return dataclasses.make_dataclass(
        f"{ref_cls.__name__}_{task_schema_cls.__name__}", [], bases=(ref_cls, task_schema_cls), kw_only=True
    )


def instantiate_prompt_cfg_node(node: dict[str, Any], cache_dir: Path | None = None) -> Any:
    """Instantiate a ``prompt_cfg`` node carrying ``compose_ref`` (with optional task-schema ``class_path``)."""
    from interpretune.utils import instantiate_class

    node = dict(node)
    ref_cls = resolve_prompt_config_class(node.pop("compose_ref"), cache_dir=cache_dir)
    task_cls = None
    if "class_path" in node:
        task_cls = instantiate_class(init={"class_path": node["class_path"]}, import_only=True)
    composed = compose_prompt_config_class(ref_cls, task_cls)
    return composed(**(node.get("init_args") or {}))
