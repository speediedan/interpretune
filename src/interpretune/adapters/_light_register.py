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


def _owning_package(module_path: str) -> str | None:
    """The adapter PACKAGE owning an implementation submodule, or ``None`` for a flat adapter module.

    ``adapter_modules`` names ``<pkg>.adapter`` submodules; the declaration lives one level up, in the
    package ``__init__``, which is the half that stays importable without the heavy dependency.
    """
    pkg, _, leaf = module_path.rpartition(".")
    return pkg if leaf == "adapter" and pkg else None


def _declared_requires(module_path: str) -> dict | None:
    """Read an adapter's declared requirements WITHOUT importing its implementation module.

    This is the whole point of the declaration being an eager constant in the package ``__init__``: the
    answer to "should this be imported" must not need the import it is deciding about.

    ``None`` means nothing is declared -- a flat adapter module with no owning package, or a package that
    declares nothing. Those fall back to import-and-report below, which is weaker (a caught exception is a
    symptom, not a reason) but still never silent.
    """
    pkg_path = _owning_package(module_path)
    if pkg_path is None:
        return None
    pkg = _import_adapter_module(pkg_path)
    return getattr(pkg, "__it_requires__", None) if pkg is not None else None


def register_all_adapters(registry) -> None:
    """Call `register_adapter_ctx` on each known adapter class.

    This function imports adapter implementation modules and invokes the registration classmethod on matching classes.
    The set of modules is initially explicit and small but may switch to an entrypoint-based discovery mechanism in the
    future.
    """
    # TODO: consider making this auto-discoverable via entrypoints
    # NOTE: these name the DEFINING submodule (`<pkg>.adapter`), not the package. Per-adapter packages
    # export lazily via PEP 562 `__getattr__`, and the discovery below walks `dir(mod)`, which does not
    # trigger a lazy resolver -- pointing at a package would find no adapter classes and register
    # NOTHING, silently. The submodules are the import-safe half of each package anyway (their heavy
    # imports are TYPE_CHECKING/local), which is the property this pass depends on.
    adapter_modules: Iterable[str] = (
        "interpretune.adapters.core",
        "interpretune.adapters.lightning",
        "interpretune.adapters.sae_lens.adapter",
        "interpretune.adapters.transformer_lens.adapter",
        "interpretune.adapters.circuit_tracer.adapter",
        "interpretune.adapters.nnsight.adapter",
    )

    from interpretune.utils.requirements import requirement_status

    skipped: list[tuple[str, str]] = []  # (adapter module, reason)

    for mod_path in adapter_modules:
        # EVALUATE BEFORE IMPORTING. Deciding by catching an ImportError conflates "this optional
        # dependency is absent" with "this adapter is broken", and reports neither -- which is #431: an
        # absent dependency silently removed 18 of 48 compositions with nothing printed. A declared
        # requirement yields a REASON instead of a symptom.
        declared = _declared_requires(mod_path)
        if declared and (unmet := requirement_status(declared, source=mod_path)):
            skipped.append((mod_path, unmet[0].message))
            continue
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
