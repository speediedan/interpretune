"""Lightweight adapter registration utilities.

This module performs the minimal imports necessary to populate the
`ADAPTER_REGISTRY` by calling `register_adapter_ctx` on adapter classes.
It intentionally avoids importing heavy runtime dependencies to enable
adapter implementation modules to be written (optionally) to be safe to import at
module-level when optional heavy dependencies are guarded by TYPE_CHECKING
or local imports.

Keep this module small and dependency-free so it can be imported at
package initialization time to ensure the registry is populated for
runtime consumers that expect registrations to exist.
"""

from importlib import import_module
from types import ModuleType
from typing import Iterable


def _import_adapter_module(module_path: str) -> ModuleType | None:
    """Import an adapter module and return the module object.

    We rely on adapter modules to avoid importing heavy third-party dependencies at module import time (they use
    TYPE_CHECKING/local imports). If a module import raises, we surface a warning but do not fail hard so the rest of
    registration can continue.
    """
    try:
        return import_module(module_path)
    except Exception:
        # Import failures here are non-fatal for the import-time
        # registration pass; leave the registry empty for adapters that
        # could not be imported and allow them to register lazily later.
        return None


def _declared_requires(declared_module: str) -> dict | None:
    """Read an adapter's declared requirements WITHOUT importing its implementation module.

    ``declared_module`` is what the entry point named, which is by contract the IMPORT-SAFE half: a
    package `__init__` for an adapter with a heavy implementation, or the module itself for one that has
    none. So importing it to read the declaration costs nothing and needs no optional dependency — which
    is the property the whole read-before-import design rests on.

    ``None`` means nothing is declared. Those fall back to import-and-report below, which is weaker (a
    caught exception is a symptom, not a reason) but still never silent.
    """
    mod = _import_adapter_module(declared_module)
    return getattr(mod, "__it_requires__", None) if mod is not None else None


#: The entry-point group through which ANY adapter — bundled or third-party — announces itself.
ADAPTER_ENTRYPOINT_GROUP = "interpretune.adapters"


def discover_adapter_entrypoints() -> dict[str, str]:
    """Adapter name -> the import-safe module that declares it, from the entry-point group.

    This replaces a hardcoded tuple of six module paths. That tuple was a RAILS file naming packages, so a
    bundled adapter was privileged by construction and a third-party replacement had no way to occupy a
    slot in it — the concrete form of "the rails may depend only on what a component DECLARES, never on
    which package it is".

    **Each value names what is IMPORT-SAFE, never the heavy implementation module**, because resolving an
    entry point imports what it names. A group entry pointing at `<pkg>.adapter` would import the framework
    in order to decide whether the framework should be imported.

    An EMPTY result is reported rather than returned quietly: registering nothing is indistinguishable from
    an environment with no adapters, and the overwhelmingly likely cause is installed metadata that predates
    this group (an editable install not refreshed after the entry points were declared).
    """
    from importlib.metadata import entry_points

    found = {ep.name: ep.value for ep in entry_points(group=ADAPTER_ENTRYPOINT_GROUP)}
    if not found:
        from interpretune.utils.logging import rank_zero_warn

        rank_zero_warn(
            f"No adapters discovered: the {ADAPTER_ENTRYPOINT_GROUP!r} entry-point group is empty, so NO "
            "compositions will register. This usually means the installed interpretune metadata predates "
            "the group — reinstall (`uv pip install -e .`) to refresh it. It does not mean this environment "
            "has no adapters."
        )
    return found


def _implementation_module(declared: str) -> str:
    """Where an adapter's registrable classes live, given the import-safe module the entry point named.

    A packaged adapter keeps its classes in `<pkg>.adapter` and its declaration in `<pkg>/__init__.py`; a flat adapter
    module is both. Resolving this AFTER the requirement check is what keeps the heavy import behind the predicate.
    """
    from importlib.util import find_spec

    try:
        if find_spec(f"{declared}.adapter") is not None:
            return f"{declared}.adapter"
    except (ImportError, AttributeError, ValueError):
        pass
    return declared


def register_all_adapters(registry) -> None:
    """Call `register_adapter_ctx` on each known adapter class.

    This function imports adapter implementation modules and invokes the registration classmethod on matching classes.
    The set of modules is initially explicit and small but may switch to an entrypoint-based discovery mechanism in the
    future.
    """
    discovered = discover_adapter_entrypoints()
    adapter_modules: Iterable[str] = tuple(discovered.values())

    from interpretune.utils.requirements import requirement_status

    skipped: list[tuple[str, str]] = []  # (adapter, reason)

    for declared_module in adapter_modules:
        # EVALUATE BEFORE IMPORTING. Deciding by catching an ImportError conflates "this optional
        # dependency is absent" with "this adapter is broken", and reports neither -- which is #431: an
        # absent dependency silently removed 18 of 48 compositions with nothing printed. A declared
        # requirement yields a REASON instead of a symptom.
        requires = _declared_requires(declared_module)
        if requires and (unmet := requirement_status(requires, source=declared_module)):
            skipped.append((declared_module, unmet[0].message))
            continue
        # Only NOW resolve and import the implementation, which for a packaged adapter is the heavy half.
        mod_path = _implementation_module(declared_module)
        mod = _import_adapter_module(mod_path)
        if mod is None:
            # Nothing declared, or declared-and-satisfied yet the import still failed. Either way this is
            # NOT an expected absence, so it is reported rather than swallowed -- the module docstring has
            # always promised a warning here and never emitted one.
            skipped.append((mod_path, f"{mod_path} could not be imported (no unmet declared requirement)"))
            continue
        # Each adapter module defines one or more adapter classes that implement
        # `register_adapter_ctx`; find them and call the registration method.
        # Avoid importing heavy symbols here; rely on module-level classes.
        try:
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                # We intentionally check for callability and the presence of
                # the registration method rather than concrete types to keep
                # this pass lightweight.
                if hasattr(attr, "register_adapter_ctx") and callable(getattr(attr, "register_adapter_ctx")):
                    try:
                        attr.register_adapter_ctx(registry)
                    except Exception:
                        # Don't let a single adapter registration failure stop the rest.
                        continue
        except Exception:
            # Defensive: if introspection on the module fails, skip it.
            skipped.append((mod_path, f"{mod_path} imported but could not be introspected for adapter classes"))
            continue

    # REPORT, ONCE, AT RANK ZERO. The skip itself was never the defect -- registering a subset is correct
    # when a dependency is genuinely absent. The defect was that it was SILENT, so "this composition is
    # unavailable here" and "this composition does not exist" became indistinguishable at exactly the
    # moment a user needs to tell them apart (#431).
    if skipped:
        from interpretune.utils.logging import rank_zero_info

        lines = "\n".join(f"  - {path}: {reason}" for path, reason in skipped)
        rank_zero_info(
            f"Registered adapters from {len(adapter_modules) - len(skipped)} of {len(adapter_modules)} "
            f"modules; {len(skipped)} unavailable in this environment:\n{lines}"
        )
