"""The unified declarative session-configuration loader (hub design v3 §11.4, lane 4a).

ONE DOOR: the input mapping is the published configuration body (design §4.4) — the same schema whether
the body came from a hub-fetched component configuration, an in-repo ``examples/`` tree, or (post-4b) an
``experiments/cli`` config. Structured key fields, when present, are validated by the same parity code the
registry and publish paths use.

ONE MERGE SITE: ``shared_config`` application happens here, through the registry factories
(``itdm_cfg_factory`` / ``it_cfg_factory``) — the same code path that hydrates registry entries, which is
also what keeps ``AutoCompConfig`` ``make_dataclass`` synthesis working (it happens inside the config
constructors the factories invoke). No deep-merge happens here: published bodies are self-contained by
design; defaults-directory merging is argv-shim territory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

from interpretune.protocol import Adapter
from interpretune.utils import instantiate_class

if TYPE_CHECKING:
    from interpretune.config import ITDataModuleConfig


def resolve_adapter_ctx(body: dict[str, Any], adapter_ctx: Sequence[Adapter | str] | None = None) -> tuple:
    """Resolve the session's adapter context from an explicit override or the body's declarations.

    Precedence: explicit ``adapter_ctx`` param → the body's own ``adapter_ctx`` (session-shaped bodies) →
    the FIRST ``reg_info.adapter_combinations`` entry (published bodies declare supported combinations;
    the first is the publisher's primary) → ``(Adapter.core,)``, the session default.
    """
    from interpretune.adapter_registry import ADAPTER_REGISTRY

    ctx = adapter_ctx if adapter_ctx is not None else body.get("adapter_ctx")
    if ctx is None:
        combos = (body.get("reg_info") or {}).get("adapter_combinations") or []
        if combos:
            first = combos[0]
            ctx = (first,) if isinstance(first, str) else tuple(first)
    if ctx is None:
        # session-shaped bodies may omit adapter_ctx entirely — the session default is core, exactly
        # as ITSessionConfig's own field default provided on the retired parser path
        ctx = (Adapter.core,)
    resolved = tuple(Adapter[a] if not isinstance(a, Adapter) else a for a in ctx)
    return ADAPTER_REGISTRY.canonicalize_composition(resolved)


def load_session_cfg(
    body: dict[str, Any],
    adapter_ctx: Sequence[Adapter | str] | None = None,
    datamodule_cls: Any = None,
    module_cls: Any = None,
    expected_key: str | None = None,
):
    """Construct an :class:`~interpretune.session.ITSessionConfig` from a declarative configuration body.

    Args:
        body: The configuration body. Published shape: ``registered_cfg`` (``datamodule_cfg`` /
            ``module_cfg`` / optional ``datamodule_cls`` / ``module_cls``) + optional ``shared_config`` +
            optional ``reg_info`` + optional structured key fields. ``datamodule_cfg`` may either be plain
            ``ITDataModuleConfig`` kwargs or carry ``class_path``/``init_args`` for a config subclass.
        adapter_ctx: Explicit adapter context override (see :func:`resolve_adapter_ctx`).
        datamodule_cls: Default datamodule class when the body declares none.
        module_cls: Default module class when the body declares none.
        expected_key: When given (hub/manifest resolution), the body's structured fields must derive to
            exactly this key (the loader-side half of the filename == manifest key == derived parity).

    Returns:
        A fully-populated ``ITSessionConfig`` (``shared_cfg`` deliberately ``None`` — shared-config
        application already happened in the factory path; the ``__post_init__`` fan-out must see nothing
        to re-apply).
    """
    import copy

    from interpretune.registry import it_cfg_factory, itdm_cfg_factory
    from interpretune.session import ITSessionConfig

    # the factories mutate nested nodes in place (class_path dicts become instances); a loader that
    # mutates its caller's mapping would corrupt cached/hub-fetched bodies on first use
    body = copy.deepcopy(body)

    if expected_key is not None or "task_variant" in body:
        from interpretune.hub.manifest import derive_config_key

        if expected_key is not None and "task_variant" not in body:
            raise ValueError(
                f"Configuration key parity requested (expected_key={expected_key!r}) but the body carries no "
                "structured key fields (task_variant/model/composition/...) to derive from."
            )
        # deriving even without an expected_key is deliberate: it validates that bodies carrying
        # structured fields remain derivable (malformed composition/adapter names fail HERE, at the
        # door, rather than later inside registry lookups)
        derived = derive_config_key(body)
        if expected_key is not None and derived != expected_key:
            raise ValueError(
                f"Configuration key parity violation: expected {expected_key!r}, derived {derived!r} "
                "from the body's structured fields."
            )

    registered = body.get("registered_cfg") or {
        k: body[k] for k in ("datamodule_cfg", "module_cfg", "datamodule_cls", "module_cls") if k in body
    }
    shared = dict(body.get("shared_config") or {})

    dm_cfg_body = dict(registered.get("datamodule_cfg") or {})
    if "class_path" in dm_cfg_body:
        # config-subclass form (the CLI dialect's shape): same class_path/init_args grammar; nested
        # class_path nodes (prompt_cfg, collator cfgs, ...) instantiate recursively exactly as
        # instantiate_nested does for module configs — one grammar, one instantiaton behavior
        from interpretune.registry import instantiate_nested

        init_args = dict(dm_cfg_body.get("init_args") or {})
        dm_cfg = cast(
            "ITDataModuleConfig",
            instantiate_nested({"class_path": dm_cfg_body["class_path"], "init_args": {**shared, **init_args}}),
        )
    else:
        dm_cfg = itdm_cfg_factory(dm_cfg_body, shared)

    m_cfg_body = dict(registered.get("module_cfg") or {})
    m_cfg = it_cfg_factory(m_cfg_body, shared)

    def _resolve_cls(entry: Any, default: Any) -> Any:
        if entry is None:
            return default
        if isinstance(entry, dict) and "class_path" in entry:
            return instantiate_class(init=entry, import_only=True)
        if isinstance(entry, str):
            return instantiate_class(init={"class_path": entry}, import_only=True)
        return entry

    return ITSessionConfig(
        adapter_ctx=resolve_adapter_ctx(body, adapter_ctx),
        datamodule_cfg=dm_cfg,
        module_cfg=m_cfg,
        datamodule_cls=_resolve_cls(registered.get("datamodule_cls"), datamodule_cls),
        module_cls=_resolve_cls(registered.get("module_cls"), module_cls),
    )


def session_body_from_cli_mapping(session_cfg_mapping: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy ``session_cfg:``-shaped CLI mapping into the one-door configuration body.

    The argv shim's translation until 4b rewrites the ``experiments/cli`` configs into the published
    body outright: the ``ITSharedConfig`` fields the CLI propagated datamodule → module via
    ``link_arguments`` are extracted into a ``shared_config`` block, so the factories apply the SAME
    values to both configs (the one merge site replacing the parser-time link DAG).
    """
    from interpretune.config.shared import ITSharedConfig

    dm = dict(session_cfg_mapping.get("datamodule_cfg") or {})
    dm_init = dict(dm.get("init_args") or {})
    shared = {k: dm_init[k] for k in ITSharedConfig.__dataclass_fields__ if k in dm_init}
    module = dict(session_cfg_mapping.get("module_cfg") or {})
    if "init_args" in module:
        # values the link DAG would overwrite must not linger as conflicting init_args
        module = {**module, "init_args": {k: v for k, v in module["init_args"].items() if k not in shared}}
        if "class_path" not in module:
            # jsonargparse's implied-class form ({init_args: ...} under a concretely-typed arg means
            # "the declared type"): normalize to plain kwargs for the factory's default construction
            module = module["init_args"]
    body: dict[str, Any] = {
        "shared_config": shared,
        "registered_cfg": {
            "datamodule_cfg": {**dm, "init_args": {k: v for k, v in dm_init.items() if k not in shared}}
            if "class_path" in dm
            else {k: v for k, v in dm_init.items() if k not in shared},
            "module_cfg": module,
        },
    }
    if session_cfg_mapping.get("adapter_ctx"):
        body["adapter_ctx"] = list(session_cfg_mapping["adapter_ctx"])
    for cls_key in ("datamodule_cls", "module_cls"):
        if session_cfg_mapping.get(cls_key):
            body["registered_cfg"][cls_key] = session_cfg_mapping[cls_key]
    return body
