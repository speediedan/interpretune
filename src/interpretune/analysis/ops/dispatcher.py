"""Dispatcher for analysis operations."""

from __future__ import annotations
from typing import Dict, NamedTuple, Iterator, Callable, Any
from dataclasses import dataclass
from pathlib import Path
from functools import wraps
from collections import defaultdict
import importlib
import os
import yaml

import torch
from transformers import BatchEncoding

from interpretune.analysis import IT_ANALYSIS_CACHE, IT_ANALYSIS_OP_PATHS, IT_ANALYSIS_HUB_CACHE
from interpretune.hub.manifest import IT_COMPONENT_MANIFEST
from interpretune.analysis.inputs import OpStateSpec
from interpretune.analysis.ops.base import AnalysisOp, CompositeAnalysisOp, OpSchema, ColCfg, OpWrapper
from interpretune.analysis.ops.collection import COLLECTION_HEADER_KEY, CollectionSpec
from interpretune.analysis.ops.auto_columns import apply_auto_columns
from interpretune.analysis.ops.compiler.cache_manager import OpDefinitionsCacheManager, OpDef
from interpretune.analysis.ops.compiler.load_policy import OpLoadError, op_load_failure, strict_op_load
from interpretune.analysis.ops.dynamic_module_utils import ensure_op_paths_in_syspath, get_function_from_dynamic_module
from interpretune.protocol import BaseAnalysisBatchProtocol
from interpretune.utils.logging import rank_zero_debug, rank_zero_warn


def _ensure_loaded(func):
    """Decorator to ensure operations are loaded before access."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._loaded:
            self.load_definitions()
        return func(self, *args, **kwargs)

    return wrapper


class DispatchContext(NamedTuple):
    """Context for dispatching operations."""

    pass  # We don't use context keys yet but may in the future


# Ordered, comma-separated namespaces whose ops win bare-name resolution. The env parity for
# `it.hub.prefer_ops`, for CLI and scripted runs where there is no place to call it.
IT_OP_PRECEDENCE_ENV_VAR = "IT_OP_PRECEDENCE"


def _cached_op_revision(source: str) -> str | None:
    """Cached commit a hub op collection currently resolves to (``None`` for bundled/local ops).

    Read from the ops hub cache, never the network: this answers "which revision am I running", and a lookup that could
    silently fetch would change the answer while reporting it.
    """
    if not source.startswith("hub:"):
        return None
    namespace = source.split(":", 1)[1]
    user, _, repo = namespace.partition(".")
    ref = Path(IT_ANALYSIS_HUB_CACHE) / f"models--{user}--{repo}" / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


@dataclass(frozen=True)
class OpCandidate:
    """One definition a bare op name could reach, with its provenance and declared collection identity."""

    name: str
    source: str
    collection: str | None
    version: str | None
    revision: str | None

    def __str__(self) -> str:
        detail = [self.source]
        if self.collection:
            detail.append(f"collection {self.collection} {self.version or '(unversioned)'}")
        if self.revision:
            detail.append(f"revision {self.revision[:12]}")
        return f"{self.name} [{', '.join(detail)}]"


@dataclass(frozen=True)
class OpProvenance:
    """What produced a set of analysis-store columns, recorded at WRITE time.

    Distinct from :class:`OpCandidate`, which answers "what would this name reach now". This answers "what
    did reach it, then" -- a recorded fact rather than a resolution, which is the whole point: precedence is
    session-mutable (:attr:`AnalysisOpDispatcher.op_precedence` re-reads ``IT_OP_PRECEDENCE`` on every
    access), so the same name can resolve elsewhere by the time a store is published.

    ``requested_name`` preserves the name AS WRITTEN, bare or fully qualified. That distinction is
    load-bearing rather than cosmetic: ``_preferred_name`` re-ranks bare names only, so it is exactly the
    discriminator whose absence makes reconstructing provenance after the fact unsound.
    """

    requested_name: str
    resolved_name: str
    source: str
    collection: str | None = None
    version: str | None = None
    revision: str | None = None

    def to_dict(self) -> dict[str, str]:
        """JSON-native form for the artifact envelope; keys with nothing to say are omitted.

        Omitted rather than emitted as ``null`` so absence reads as absence: a bundled op HAS no collection
        revision, and recording ``"revision": null`` invites a reader to treat it as an unresolved lookup.
        """
        record = {
            "requested_name": self.requested_name,
            "resolved_name": self.resolved_name,
            "source": self.source,
        }
        for key in ("collection", "version", "revision"):
            value = getattr(self, key)
            if value is not None:
                record[key] = value
        return record


@dataclass(frozen=True)
class OpResolution:
    """What a name resolves to right now, the alternatives, and the precedence that chose between them."""

    requested: str
    resolved: str
    active: OpCandidate
    alternatives: tuple[OpCandidate, ...] = ()
    precedence: tuple[str, ...] = ()

    @property
    def is_shadowing_bundled(self) -> bool:
        """Whether a non-bundled definition is currently winning a name a bundled op also defines."""
        return not self.active.source.startswith("bundled") and any(
            candidate.source == "bundled" for candidate in self.alternatives
        )

    def __str__(self) -> str:
        lines = [f"{self.requested!r} resolves to {self.active}"]
        if self.alternatives:
            lines.append("  also available:")
            lines += [f"    {candidate}" for candidate in self.alternatives]
        lines.append(f"  precedence: {list(self.precedence) or 'none (bundled ops win bare names)'}")
        return "\n".join(lines)


class AnalysisOpDispatcher:
    """Dispatcher for analysis operations with lazy loading and caching.

    This class handles loading operation definitions from YAML and dispatching them based on a given context. Operations
    are dynamically instantiated from their definitions when first accessed.
    """

    # TODO:
    #  - decide whether to make the dispatcher a singleton or not
    #  - decide whether to make the dispatcher thread-safe
    def __init__(self, yaml_paths: Path | list[Path] | None = None, enable_hub_ops: bool = True):
        # Initialize yaml_paths
        self.yaml_paths = [Path(p.strip()) for p in IT_ANALYSIS_OP_PATHS]  # Start with op_paths

        # Always include the bundled op-family definitions shipped with the package. Bundled YAMLs
        # reference fully-qualified interpretune implementation paths, so the bundled tree is
        # deliberately excluded from op_paths/sys.path exposure (see _resolve_op_paths_from_yaml_paths).
        self._bundled_ops_dir = Path(__file__).parent / "bundled"
        self.yaml_paths.append(self._bundled_ops_dir)

        # Handle user-provided yaml_paths
        if yaml_paths:  # otherwise use only use the default
            if isinstance(yaml_paths, (Path, str)):
                self.yaml_paths.append(Path(yaml_paths))
            else:
                # Handle list/iterable of paths (convert strings to Path objects)
                self.yaml_paths.extend(Path(p) for p in yaml_paths)
                assert all(isinstance(p, Path) for p in self.yaml_paths), (
                    "yaml_paths must be a Path, string, or a list of Paths/strings"
                )

        self.enable_hub_ops = enable_hub_ops
        self._op_definitions: dict[str, OpDef] = {}
        # Declared-name -> originating YAML, for diagnostics only. Deliberately NOT a field on OpDef:
        # OpDef is serialized into the generated cache, so adding one would force a
        # CACHE_FORMAT_VERSION bump and invalidate every user's cache for a string used only in error
        # messages. If a source path is ever needed at RUNTIME rather than at load, promote it then.
        self._op_declaration_sites: dict[str, str] = {}
        # Originating YAML -> its declared `collection:` header, for compatibility enforcement at load and
        # for op_info attribution. Load-time only, like the declaration sites above.
        self._op_collections: dict[str, CollectionSpec] = {}
        # Namespaces (`user.repo`) whose ops win BARE-name resolution, highest precedence first. Empty by
        # default: bundled ops win bare names, and opting into a newer hub copy is explicit (D8 item 2).
        self._preferred_op_namespaces: list[str] = []
        self._dispatch_table = {}  # {op_name: {context: instantiated_op}}
        self._aliases = {}  # {alias: op_name}
        self._op_to_aliases = defaultdict(list)  # {op_name: [aliases]}
        self._loaded = False
        self._loading_in_progress = False
        # resolve op_paths from yaml_paths
        self.op_paths = []
        # Resolve op_paths from yaml_paths
        self._resolve_op_paths_from_yaml_paths()
        # Ensure op_paths are in sys.path
        ensure_op_paths_in_syspath(self.op_paths)
        self._cache_manager = OpDefinitionsCacheManager(IT_ANALYSIS_CACHE)

    def _normalize_op_name(self, name: str) -> str:
        # Normalize operation names for consistent lookup (case-insensitive, cross-platform)
        return name.replace("/", ".").replace("-", "_").lower()

    def _discover_yaml_files(self, paths: list[Path]) -> list[Path]:
        """Discover all YAML files from the given paths (files or directories).

        A component manifest is never an op-definitions file. It shares the ``.yaml`` suffix and sits at the root of
        every interpretune component repo, including op collections, so a directory holding both a manifest and its op
        YAMLs (the ordinary shape when authoring a collection locally) would otherwise feed the manifest through the op
        compiler, where its scalar keys (``it_schema_version: 1``) raise on ``op_def.get(...)`` and take down the whole
        load -- every bundled op included.
        """
        yaml_files = []
        for path in paths:
            if path.is_file() and path.suffix.lower() in (".yaml", ".yml"):
                yaml_files.append(path)
            elif path.is_dir():
                # Recursively find all YAML files in the directory
                yaml_files.extend(path.glob("**/*.yaml"))
                yaml_files.extend(path.glob("**/*.yml"))
        return sorted({p for p in yaml_files if p.name != IT_COMPONENT_MANIFEST})

    def load_definitions(self) -> None:
        """Load operation definitions from YAML files."""
        if self._loaded or self._loading_in_progress:
            return

        self._loading_in_progress = True
        try:
            # Discover all YAML files from the configured paths
            yaml_files = self._discover_yaml_files(self.yaml_paths)
            rank_zero_debug(f"[DISPATCHER] Discovered {len(yaml_files)} local YAML files")

            # Add hub operations if enabled
            if self.enable_hub_ops:
                rank_zero_debug("[DISPATCHER] Hub ops enabled, adding hub YAML files")
                hub_yaml_files = self._cache_manager.add_hub_yaml_files()
                yaml_files.extend(hub_yaml_files or [])
                rank_zero_debug(f"[DISPATCHER] Total YAML files after hub: {len(yaml_files)}")
            else:
                rank_zero_debug("[DISPATCHER] Hub ops disabled")

            # Set up cache manager with discovered YAML files # TODO: might be able to remove since we already discover
            for yaml_file in yaml_files:
                if yaml_file not in [info.path for info in self._cache_manager._yaml_files]:
                    self._cache_manager.add_yaml_file(yaml_file)

            rank_zero_debug("[DISPATCHER] Attempting to load from cache")
            # Strict loading vetoes the cache. Every check `op_load_failure` guards -- failed compiles,
            # unresolvable importable_params, invalid op_state, unsanctioned hub params, name collisions --
            # runs while definitions are COMPILED, so reading a precompiled artifact skips all of them and
            # `IT_STRICT_OP_LOAD=1` silently becomes a no-op against a warm cache. Measured before fixing:
            # a non-strict load warned once and cached; the next load with strict enabled reused that cache
            # and reported nothing. Verifying is the whole point of the flag, so it recompiles.
            cached_definitions = None if strict_op_load() else self._cache_manager.load_cache()
            if cached_definitions is not None:
                rank_zero_debug(f"[DISPATCHER] Cache HIT: Loaded {len(cached_definitions)} definitions from cache")
                self._op_definitions = cached_definitions
                self._set_default_hub_op_aliases()
            else:
                rank_zero_debug("[DISPATCHER] Cache MISS: Compiling from source")
                # Cache miss or invalid - load from YAML and compile
                rank_zero_debug("Cache miss or invalid, loading from YAML and compiling")
                self._load_from_yaml_and_compile(yaml_files)

            # Build aliases mapping
            self._populate_aliases_from_definitions()

            self._loaded = True
            rank_zero_debug(f"[DISPATCHER] Loaded {len(self._op_definitions)} operation definitions")

        except Exception as e:
            rank_zero_warn(f"Failed to load operation definitions: {e}")
            raise
        finally:
            self._loading_in_progress = False

    def reload_definitions(self) -> None:
        """Re-discover and reload every op definition, picking up collections cached since the last load.

        Needed because a session loads definitions once. Without this, fetching a collection and then using
        one of its ops in the SAME process raises ``Unknown operation`` -- the ops are on disk, and nothing
        looks at the disk again -- so the fetch appeared to do nothing until the next process. Discovery
        re-runs the trust gate and every collection's compatibility window, exactly as the first load did.
        """
        self._op_definitions = {}
        self._op_declaration_sites = {}
        self._op_collections = {}
        self._aliases = {}
        self._op_to_aliases = defaultdict(list)
        self._dispatch_table = {}
        self._cache_manager._yaml_files = []
        self._cache_manager._fingerprint = None
        self._loaded = False
        self.load_definitions()
        # Re-sync the top-level `it.<op>` wrappers with the reloaded registry. `OpWrapper.register_operations`
        # runs once when `interpretune.analysis` is imported and SNAPSHOTS the op names as module attributes,
        # so without this a collection fetched mid-session is "usable immediately" at the dispatcher layer
        # (as pull_ops documents) while `it.<its_op>` still raises AttributeError -- and the top-level wrapper
        # is precisely the surface the composition guide tells notebook authors to use. `_target_module` is
        # None only when the wrapper surface was never installed, in which case there is nothing to sync.
        if OpWrapper._target_module is not None:
            # `_target_module is not None` proves the wrapper surface was installed in this process;
            # it does NOT prove the stored object is still the module readers see. A test or notebook
            # that purges and reimports `interpretune` (the op-collection example does exactly this)
            # leaves `_target_module` pointing at the ORPHANED module object, and registering onto it
            # satisfies every wrapper-side invariant while `it.<op>` -- which resolves through
            # sys.modules -- still raises. Measured: op_in_dispatcher=True, attr_on_target=True,
            # target_module_is_it=False. So sync onto the CANONICAL module, falling back to the
            # stored one only when interpretune is somehow absent from sys.modules.
            import sys as _sys

            target_module = _sys.modules.get("interpretune", OpWrapper._target_module)
            OpWrapper.register_operations(target_module, self)

    def _load_from_yaml_and_compile(self, yaml_files: list[Path]):
        """Load from YAML files and compile to cache."""
        # Load and merge all YAML files
        raw_definitions = {}
        composite_operations = {}

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    yaml_content = yaml.safe_load(f)

                if not yaml_content:
                    rank_zero_debug(f"Empty YAML file: {yaml_file}")
                    continue

                if not isinstance(yaml_content, dict):
                    rank_zero_debug(f"Skipping non-dictionary YAML file: {yaml_file}")
                    continue

                collection = self._collection_for(yaml_file, yaml_content)
                if collection is not None and (incompatible := collection.incompatibility()) is not None:
                    # Skip the WHOLE collection, not individual ops: the compatibility window is declared
                    # per collection, so a partial load would present half a contract set.
                    op_load_failure(
                        f"Skipping op collection {collection.name!r} (version {collection.version}) from "
                        f"{yaml_file}: {incompatible}"
                    )
                    continue

                # Drop the header BEFORE namespacing. Namespacing rewrites every top-level key, so a
                # hub collection's header arrived as `<user>.<repo>.collection` and no longer matched the
                # header key -- it was then registered as an op, giving every hub collection a junk
                # `collection` op with the header's mapping as its definition. Bundled files are not
                # namespaced, which is why the equality check held there and hid this.
                op_content = {key: value for key, value in yaml_content.items() if key != COLLECTION_HEADER_KEY}
                namespaced_content = self._apply_hub_namespacing(op_content, yaml_file)

                source = self._op_source_for(yaml_file)

                # Stamped onto every definition from this file so the identity survives the cache: the
                # `collection:` header is parsed at compile time only, and `op_info` needs it at runtime.
                provenance = {
                    "source": source,
                    "collection_name": collection.name if collection else None,
                    "collection_version": collection.version if collection else None,
                }

                # Separate composite operations from regular operations
                for key, value in namespaced_content.items():
                    if key == "composite_operations":
                        for comp_name, comp_def in value.items():
                            composite_operations[comp_name] = {**comp_def, **provenance}
                            self._op_declaration_sites[comp_name] = str(yaml_file)
                    else:
                        if not isinstance(value, dict):
                            # Reject at INGEST, not at conversion: `_compile_required_ops_schemas` runs
                            # first and catches only ValueError, so a scalar reaching it raised
                            # AttributeError from `op_def.get(...)` and took down every op in the process,
                            # bundled included. The usual cause is a non-op YAML being read as op
                            # definitions, where scalar top-level values are perfectly normal.
                            op_load_failure(
                                f"Skipping '{key}' in {yaml_file}: an op definition must be a mapping, got "
                                f"{type(value).__name__}. Is this file an op-definitions YAML?"
                            )
                            continue
                        if key in raw_definitions:
                            rank_zero_debug(f"Operation '{key}' redefined in {yaml_file}, using latest definition")
                        raw_definitions[key] = {**value, **provenance}
                        self._op_declaration_sites[key] = str(yaml_file)

            except OpLoadError:
                # Strict loading must not be swallowed by the per-file fail-soft handler: the whole point
                # of IT_STRICT_OP_LOAD is that a definition problem stops the run, and a broad `except
                # Exception` here would quietly downgrade it to a debug line -- the same way a warm cache
                # used to skip these checks entirely.
                raise
            except Exception as e:
                rank_zero_debug(f"Failed to load YAML file {yaml_file}: {e}")
                # Continue processing other files rather than failing completely
                continue

        # Second pass: Compile schemas with required_ops dependencies
        self._compile_required_ops_schemas(raw_definitions)

        # Process composite operations with schema compilation
        if composite_operations:
            from interpretune.analysis.ops.compiler.schema_compiler import build_operation_compositions

            # Create a complete YAML structure for build_operation_compositions
            complete_yaml = raw_definitions.copy()
            complete_yaml["composite_operations"] = composite_operations

            # Apply schema compilation for composite operations
            compiled_ops = build_operation_compositions(complete_yaml)

            # Update definitions with compiled operation schemas
            for op_name, op_def in compiled_ops.items():
                if op_name not in raw_definitions:
                    raw_definitions[op_name] = op_def
                else:
                    # Update existing definition with compiled schemas
                    if "input_schema" in op_def:
                        raw_definitions[op_name]["input_schema"] = op_def["input_schema"]
                    if "output_schema" in op_def:
                        raw_definitions[op_name]["output_schema"] = op_def["output_schema"]

        # Convert raw definitions to OpDef objects
        self._convert_raw_definitions_to_opdefs(raw_definitions)
        self._set_default_hub_op_aliases()
        # Build aliases mapping
        self._populate_aliases_from_definitions()

        # Save to cache for next time
        self._cache_manager.save_cache(self._op_definitions)

        self._loaded = True

    def _compile_required_ops_schemas(self, definitions_to_compile: dict[str, Dict]):
        """Compile schemas by recursively including required_ops dependencies."""
        from interpretune.analysis.ops.compiler.schema_compiler import compile_op_schema

        # TODO: consider moving this compilation to schema_compiler.py, we're keeping this here for now because
        #       applying auto-columns should not be part of schema_compiler.py
        # Compile all operations
        for op_name in list(definitions_to_compile.keys()):
            try:
                compile_op_schema(op_name, definitions_to_compile)
                # Apply optional auto-columns after compilation
                apply_auto_columns(definitions_to_compile[op_name])
            except ValueError as e:
                # Dropping an op whose required_ops do not resolve is exactly the silent failure
                # #266 flags, so strict loading turns it into an error.
                definitions_to_compile.pop(op_name, None)
                op_load_failure(f"Failed to compile operation '{op_name}': {e}")

    def _collection_for(self, yaml_file: Path, yaml_content: dict[str, Any]) -> CollectionSpec | None:
        """Parse a YAML's ``collection:`` header, recording it per file for later attribution.

        Fail-soft on a malformed header, matching the surrounding load paths: the ops still load, just without
        collection identity, which surfaces as a missing version in ``op_info`` rather than a dead session.
        """
        try:
            collection = CollectionSpec.from_raw(yaml_content.get(COLLECTION_HEADER_KEY))
        except (ValueError, TypeError) as bad_header:
            op_load_failure(f"Ignoring invalid `{COLLECTION_HEADER_KEY}` header in {yaml_file}: {bad_header}")
            return None
        if collection is not None:
            self._op_collections[str(yaml_file)] = collection
        return collection

    def _declaration_site(self, declared_name: str) -> str:
        """Where a declared op name came from, for diagnostics ("<unknown source>" if untracked)."""
        return self._op_declaration_sites.get(declared_name, "<unknown source>")

    def _check_normalization_collision(self, declared_name: str, normalized: str, claimed_by: dict[str, str]) -> None:
        """Report when two distinct declared names normalize to the same op name.

        Distinct declared names collapse to one key because :meth:`_normalize_op_name` case-folds and maps ``-``/``/``
        (``my-collide-op`` and ``my_collide_op`` both become ``my_collide_op``). Undetected, the later definition
        silently replaces the earlier one and lookup returns whichever won: a wrong answer rather than an error, and
        invisible when the two come from different collections. The check spans the whole merged definition set, not a
        single file, because that cross-collection case is the realistic one and no per-file check can see it.

        ``OpDef.source`` cannot name the sources here: it is a category (``bundled`` | ``local`` | ``hub:<user.repo>``),
        so two local collections both report ``local``. The declared-name -> YAML side map supplies the actual files.
        """
        prior = claimed_by.get(normalized)
        if prior is None or prior == declared_name:
            return
        op_load_failure(
            f"Operation name collision: '{declared_name}' (declared in {self._declaration_site(declared_name)}) "
            f"normalizes to '{normalized}', which is already declared as '{prior}' in "
            f"{self._declaration_site(prior)}. Op names are matched case-insensitively with '-' and '/' normalized, "
            f"so these are the same operation. Rename one of them; the later definition currently wins."
        )

    def _convert_raw_definitions_to_opdefs(self, raw_definitions: dict[str, Dict]):
        """Convert raw dictionary definitions to OpDef objects."""
        # normalized name -> the declared name that claimed it, so a collision can name both sides.
        claimed_by: dict[str, str] = {}
        for op_name, op_def in raw_definitions.items():
            self._check_normalization_collision(op_name, self._normalize_op_name(op_name), claimed_by)
            claimed_by[self._normalize_op_name(op_name)] = op_name
            op_name = self._normalize_op_name(op_name)
            # Convert schemas to OpSchema objects
            input_schema = self._convert_to_op_schema(op_def.get("input_schema", {}))
            output_schema = self._convert_to_op_schema(op_def.get("output_schema", {}))

            importable_params = op_def.get("importable_params", {})

            # Create OpDef
            op_def_obj = OpDef(
                name=op_name,
                description=op_def.get("description", ""),
                implementation=op_def.get("implementation", ""),
                input_schema=input_schema,
                output_schema=output_schema,
                aliases=op_def.get("aliases", []),
                importable_params=importable_params,
                normal_params=op_def.get("normal_params", {}),
                required_ops=op_def.get("required_ops", []),
                required_capabilities=op_def.get("required_capabilities", []),
                composition=op_def.get("composition", None),
                op_state=self._resolve_op_state_spec(op_name, op_def.get("op_state")),
                source=str(op_def.get("source", "bundled")),
                collection_name=op_def.get("collection_name"),
                collection_version=op_def.get("collection_version"),
                uses_default_hooks=bool(op_def.get("uses_default_hooks", False)),
                requires_grad=bool(op_def.get("requires_grad", False)),
                per_latent_preds=bool(op_def.get("per_latent_preds", False)),
                protocol_cls=op_def.get("protocol_cls"),
            )

            self._op_definitions[op_name] = op_def_obj

    def _op_source_for(self, yaml_file: Path) -> str:
        """Classify where an op definition came from: ``bundled``, ``local``, or ``hub:<namespace>``.

        Replaces dot-counting on the op NAME as the hub test. A dotted name means "namespaced", which
        is how hub ops are addressed, but it is a property of the name rather than of provenance --
        and provenance is what the version/precedence work in #266 Phase 3 needs to key on.
        """
        try:
            resolved = Path(yaml_file).resolve()
            bundled_dir = getattr(self, "_bundled_ops_dir", None)
            if bundled_dir is not None and Path(bundled_dir).resolve() in resolved.parents:
                return "bundled"
            namespace = self._cache_manager.get_hub_namespace(resolved)
            if namespace and "." in namespace:
                return f"hub:{namespace}"
        except Exception as source_error:  # provenance must never break loading
            rank_zero_debug(f"[DISPATCHER] Could not classify source for {yaml_file}: {source_error}")
        return "local"

    @staticmethod
    def _resolve_op_state_spec(op_name: str, raw: Any) -> OpStateSpec | None:
        """Compile an op's ``op_state`` trait, warning and dropping the trait if it is malformed.

        Fail-soft matches the surrounding YAML/compile paths (a malformed hub op must not take down the dispatcher); the
        op still loads, just without declared cross-batch state, which surfaces as a clear error the first time an impl
        tries to use it.
        """
        try:
            return OpStateSpec.from_raw(raw)
        except (ValueError, TypeError) as spec_error:
            op_load_failure(f"Ignoring invalid op_state declaration for operation '{op_name}': {spec_error}")
            return None

    def _apply_hub_namespacing(self, yaml_content: dict[str, Any], yaml_file: Path) -> dict[str, Any]:
        """Apply hub namespacing to operations from hub files."""
        rank_zero_debug(f"[DISPATCHER] Processing yaml_file: {yaml_file}")

        # Get namespace for this file
        namespace = self._cache_manager.get_hub_namespace(yaml_file)
        rank_zero_debug(f"[DISPATCHER] Retrieved namespace: '{namespace}'")

        # If it's a top-level namespace (non-hub), return unchanged
        if "." not in namespace:
            rank_zero_debug(f"[DISPATCHER] No dots in namespace '{namespace}' - returning unchanged")
            return yaml_content

        rank_zero_debug(f"[DISPATCHER] Applying namespace '{namespace}' to operations: {list(yaml_content.keys())}")

        # Apply namespacing to hub operations
        namespaced_content = {}

        for op_name, op_config in yaml_content.items():
            if op_name == "composite_operations":
                # Handle composite operations separately - namespace the compositions
                namespaced_composites = {}
                for comp_name, comp_config in op_config.items():
                    namespaced_comp_name = f"{namespace}.{comp_name}"
                    namespaced_composites[namespaced_comp_name] = comp_config.copy()

                    # Also namespace any aliases
                    if "aliases" in comp_config:
                        namespaced_aliases = []
                        for alias in comp_config["aliases"]:
                            namespaced_aliases.append(f"{namespace}.{alias}")
                        namespaced_composites[namespaced_comp_name]["aliases"] = namespaced_aliases

                namespaced_content["composite_operations"] = namespaced_composites
                continue

            # Add namespace prefix to operation name
            namespaced_name = f"{namespace}.{op_name}"
            rank_zero_debug(f"[DISPATCHER] Namespacing '{op_name}' -> '{namespaced_name}'")
            namespaced_content[namespaced_name] = op_config.copy()

            # Also namespace any aliases
            if "aliases" in op_config:
                namespaced_aliases = []
                for alias in op_config["aliases"]:
                    namespaced_alias = f"{namespace}.{alias}"
                    rank_zero_debug(f"[DISPATCHER] Namespacing alias '{alias}' -> '{namespaced_alias}'")
                    namespaced_aliases.append(namespaced_alias)
                namespaced_content[namespaced_name]["aliases"] = namespaced_aliases

        rank_zero_debug(f"[DISPATCHER] Final namespaced operations: {list(namespaced_content.keys())}")
        return namespaced_content

    def _populate_aliases_from_definitions(self):
        """Build alias mappings from operation definitions."""
        # Clear existing mappings
        self._aliases.clear()
        self._op_to_aliases.clear()

        op_definitions = self._op_definitions.copy()

        for op_name, op_def in op_definitions.items():
            op_name_norm = self._normalize_op_name(op_name)
            # Only process canonical entries (skip alias entries added to _op_definitions)
            canonical_name = self._normalize_op_name(op_def.name)
            if op_name_norm != canonical_name:
                continue
            # Build mapping for each alias
            for alias in op_def.aliases:
                alias_norm = self._normalize_op_name(alias)
                # Prevent self-referencing aliases
                if alias_norm == op_name_norm:
                    continue

                # Add alias reference to definitions if not already present (normally should be already present)
                if alias_norm not in self._op_definitions:
                    self._op_definitions[alias_norm] = op_def
                if self._op_definitions[alias_norm] == op_def:
                    self._aliases[alias_norm] = op_name_norm
                    self._op_to_aliases[op_name_norm].append(alias_norm)
                else:
                    rank_zero_warn(
                        f"The alias '{alias}' is already associated with different operation "
                        f"({self._op_definitions[alias_norm]}) so will not be added."
                    )
                # For namespaced operations, also add non-namespaced convenience alias mapping
                # This allows "test_hub_alias" to resolve to "testuser.test.test_op"
                if "." in op_name_norm:
                    # Extract the original (non-namespaced) alias
                    original_alias = (
                        alias_norm.split(".", 3)[-1] if alias_norm.count(".") >= 3 else alias_norm.split(".")[-1]
                    )
                    if original_alias in self._aliases:
                        # If the original alias already exists, ensure it points to the same op_name
                        if self._aliases[original_alias] != op_name_norm:
                            incumbent = self._op_definitions.get(self._aliases[original_alias])
                            message = (
                                f"The name '{original_alias}' already exists for a different operation. "
                                f"The fully-qualified alias name ({alias_norm}) has been added as an alias "
                                f"for {op_name_norm}."
                            )
                            if incumbent is None:
                                rank_zero_warn(message)
                            else:
                                self._report_bare_name_contest(incumbent, op_def, message)
                    else:
                        if self._aliases and original_alias != op_name_norm:
                            self._aliases[original_alias] = op_name_norm
                            self._op_to_aliases[op_name_norm].append(alias_norm)

    @_ensure_loaded
    def list_operations(self) -> list[str]:
        """Get a list of all available operation names.

        Returns:
            List of operation names including both native and hub operations
        """
        return list(self._op_definitions.keys())

    @property
    @_ensure_loaded
    def registered_ops(self) -> dict[str, OpDef]:
        """Get all registered operation definitions without instantiating them."""
        # TODO: return a generator here instead of a dict? May be better to provide a separate method for that
        return {name: op_def for name, op_def in self._op_definitions.items()}

    @_ensure_loaded
    def resolve_alias(self, op_alias: str) -> str | None:
        return self._aliases.get(op_alias, None)

    @_ensure_loaded
    def get_op_aliases(self, op_name: str) -> list[str]:
        return self._op_to_aliases[op_name]

    @_ensure_loaded
    def get_all_aliases(self) -> Iterator[tuple[str, str]]:
        """Get all registered operation aliases."""
        for alias, op_name in self._aliases.items():
            yield (alias, op_name)

    def _resolve_name_safe(self, op_name: str, visited: set | None = None) -> str:
        """Safely resolve names with cycle detection, normalizing the way storage does.

        Normalizing here is what makes lookup symmetric with registration. Definitions are stored under
        :meth:`_normalize_op_name` of their declared name (case-folded, ``-``->``_``, ``/``->``.``), and this is the
        single choke point every external lookup passes through, so without it an op declared ``my-hyphen-op`` or
        ``MyCasedOp`` registered under a name its own author could not use -- ``get_op`` raised ``Unknown operation``
        with no warning at load. The docstring on ``_normalize_op_name`` always claimed normalization was "for
        consistent lookup"; only the storage half implemented it.
        """
        if visited is None:
            visited = set()
        op_name = self._normalize_op_name(op_name)

        if (preferred := self._preferred_name(op_name)) is not None and preferred not in visited:
            op_name = preferred

        if op_name in visited:
            # Cycle detected, return the original name
            return op_name

        if op_name not in self._aliases:
            return op_name

        visited.add(op_name)
        resolved = self._resolve_name_safe(self._aliases[op_name], visited)
        visited.remove(op_name)

        return resolved

    @property
    def op_precedence(self) -> list[str]:
        """Namespaces whose ops win bare-name resolution, highest precedence first.

        In-process :meth:`prefer_ops` declarations come first, then any from ``IT_OP_PRECEDENCE`` (comma
        separated, ordered) that they do not already name. The env var is read on every access rather than
        cached at construction because the dispatcher is a module-level singleton built at import: a value
        exported after that would otherwise never be seen.
        """
        from_env = [
            namespace
            for entry in os.environ.get(IT_OP_PRECEDENCE_ENV_VAR, "").split(",")
            if (namespace := self._normalize_op_name(entry.strip()))
        ]
        ordered = list(self._preferred_op_namespaces)
        ordered += [namespace for namespace in from_env if namespace not in ordered]
        return ordered

    def prefer_ops(self, *repo_ids: str, replace: bool = False) -> list[str]:
        """Opt into resolving bare op names to a collection's ops; returns the active precedence list.

        Ops are addressed by bare name throughout examples and notebooks, and bundled ops win those names by
        default. This is how a caller opts into a hub collection's copy of an op instead -- per namespace and
        explicitly, never as a side effect of pulling. Fully-qualified names are unaffected in both
        directions: they always address exactly what they name.

        Called with no arguments it clears the in-process opt-in (``IT_OP_PRECEDENCE`` still applies).
        """
        namespaces = [self._normalize_op_name(repo_id) for repo_id in repo_ids]
        if replace or not repo_ids:
            self._preferred_op_namespaces = namespaces
        else:
            for namespace in namespaces:
                if namespace in self._preferred_op_namespaces:
                    self._preferred_op_namespaces.remove(namespace)
                self._preferred_op_namespaces.append(namespace)
        # Bare names may already be bound in the dispatch table; drop those so the flip takes effect for ops
        # that have been called earlier in the session.
        self._dispatch_table.clear()
        return self.op_precedence

    def _preferred_name(self, op_name: str) -> str | None:
        """The namespaced name a BARE op name resolves to under an explicit precedence opt-in, if any.

        Re-ranking happens at lookup rather than by rewriting ``_op_definitions``, for two reasons: a bundled
        op's only name IS its bare name (nothing else addresses it), so rebinding that key would make the
        bundled copy unreachable rather than merely lower-priority; and precedence stays reversible within a
        session, which is what the opt-in demo notebook shows.
        """
        if "." in op_name:
            return None  # explicit beats implicit: a fully-qualified name is never re-ranked
        for namespace in self.op_precedence:
            candidate = f"{namespace}.{op_name}"
            if candidate in self._op_definitions:
                return candidate
        return None

    @_ensure_loaded
    def op_info(self, op_name: str) -> "OpResolution":
        """Report which collection a name currently resolves to, and what the alternatives are.

        The question "am I running the bundled ``concept_direction`` or the one I pulled" has no answer from
        the op object itself, so this reports the resolution: the definition a name reaches now, its
        provenance and declared collection identity, every other definition sharing that bare name, and the
        precedence that decided between them.
        """
        normalized = self._normalize_op_name(op_name)
        resolved = self._resolve_name_safe(normalized)
        reached = self._op_definitions.get(resolved)
        if reached is None:
            # Raising rather than reporting an empty resolution: "what does this name resolve to" has no
            # useful answer for a name that resolves to nothing, and a typo must not read as a finding.
            raise ValueError(f"Unknown operation: {op_name}")
        # Bare-name aliasing registers a second `_op_definitions` key pointing at the SAME OpDef, so report
        # the canonical name; otherwise a namespaced op's own bare entry shows up as an alternative to itself.
        resolved = reached.name if reached.name in self._op_definitions else resolved
        bare = resolved.split(".")[-1]
        candidates = [
            name
            for name, op_def in self._op_definitions.items()
            if name.split(".")[-1] == bare and op_def.name == name  # canonical entries only, not aliases
        ]
        return OpResolution(
            requested=normalized,
            resolved=resolved,
            active=self._describe_candidate(self._op_definitions[resolved], resolved),
            alternatives=tuple(
                self._describe_candidate(self._op_definitions[name], name)
                for name in sorted(candidates)
                if name != resolved
            ),
            precedence=tuple(self.op_precedence),
        )

    @_ensure_loaded
    def op_provenance(self, op) -> tuple["OpProvenance", ...]:
        """Record what an op resolves to NOW, for stamping onto a store at write time.

        Returns one entry per contributing definition: a composite contributes its constituents, because a
        composition can mix collections (a bundled op composed with a pulled one) and a single record would
        have to pick one and lie about the rest.

        Returns an EMPTY tuple when there is nothing to record -- an op object with no registered definition,
        or a store assembled outside the op path. Absence must read as absence: defaulting an unresolvable op
        to ``bundled`` would fabricate exactly the provenance this exists to make trustworthy.
        """
        records: list[OpProvenance] = []
        for member in self._provenance_members(op):
            requested = getattr(member, "name", None) or str(member)
            resolved = self._resolve_name_safe(self._normalize_op_name(requested))
            op_def = self._op_definitions.get(resolved) if resolved else None
            if op_def is None:
                continue  # unregistered: record nothing rather than guess
            canonical = op_def.name if op_def.name in self._op_definitions else resolved
            records.append(
                OpProvenance(
                    requested_name=requested,
                    resolved_name=canonical,
                    source=op_def.source,
                    collection=op_def.collection_name,
                    version=op_def.collection_version,
                    revision=_cached_op_revision(op_def.source),
                )
            )
        return tuple(records)

    @staticmethod
    def _provenance_members(op) -> tuple:
        """The definitions that contribute columns: a composite's constituents, else the op itself."""
        composition = getattr(op, "composition", None)
        if composition:
            # flatten one level per member so a composite of composites still reports leaf definitions
            members: list = []
            for member in composition:
                nested = getattr(member, "composition", None)
                members.extend(nested if nested else [member])
            return tuple(members)
        return (op,) if op is not None else ()

    @staticmethod
    def _describe_candidate(op_def: OpDef, name: str) -> "OpCandidate":
        """Provenance, collection identity and cached revision for one definition."""
        return OpCandidate(
            name=name,
            source=op_def.source,
            collection=op_def.collection_name,
            version=op_def.collection_version,
            revision=_cached_op_revision(op_def.source),
        )

    def _report_bare_name_contest(self, incumbent: OpDef, challenger: OpDef, message: str) -> None:
        """Report a contested bare name at the volume the contest actually warrants.

        A hub op losing a bare name to a BUNDLED op is the documented default rather than an anomaly: bundled
        ops win bare names so a session behaves identically with and without hub access, and
        :func:`interpretune.hub.prefer_ops` is the supported way to change that. Warning on it is actively
        harmful once collections mirror bundled families -- pulling the published concept mirror emitted nine
        warnings about working-as-designed behavior, which is how a warning channel stops being read. A
        contest whose incumbent is NOT bundled is genuinely ambiguous and still warns.
        """
        by_design = incumbent.source == "bundled" and challenger.source.startswith("hub:")
        if by_design:
            rank_zero_debug(f"[ANALYSIS_OPS] {message}")
        else:
            rank_zero_warn(message)

    def _set_default_hub_op_aliases(self) -> dict[str, "OpDef"]:
        """Ensure operations are accessible both with and without namespaces."""
        # Use existing definitions if no raw definitions provided
        target_ops = self._op_definitions
        current_ops = dict(self._op_definitions)

        for op_name, op_def in current_ops.items():
            # If this is a namespaced operation, also add it without namespace
            if "." in op_name:
                # Extract the base name (last part after final dot)
                base_name = op_name.split(".")[-1]

                # Only add if there's no existing operation with that base name
                # and it's not a self-reference
                if base_name in target_ops:
                    # If the base name already exists, ensure it points to the same OpDef
                    if target_ops[base_name] != target_ops[op_name]:
                        self._report_bare_name_contest(
                            target_ops[base_name],
                            target_ops[op_name],
                            f"Base name '{base_name}' already has an assigned op or alias so '{op_name}' "
                            f"cannot be mapped to it. Address it by its fully-qualified name, or opt into "
                            f"resolving the bare name to this collection with "
                            f"it.hub.prefer_ops('{op_name.rsplit('.', 1)[0].replace('.', '/', 1)}').",
                        )
                else:
                    if base_name != op_name:
                        target_ops[base_name] = target_ops[op_name]

            # # Handle aliases - ensure they point to the same OpDef
            for alias in op_def.aliases:
                # Skip self-referencing aliases
                if alias == op_name:
                    continue

                if alias in target_ops:
                    # If alias already exists, ensure it points to the same OpDef
                    if target_ops[alias] != target_ops[op_name]:
                        self._report_bare_name_contest(
                            target_ops[alias],
                            target_ops[op_name],
                            f"Alias '{alias}' already has an assigned op or alias so the "
                            f"alias specified by '{op_name}' cannot be mapped to it",
                        )
                else:
                    target_ops[alias] = target_ops[op_name]

                # Extract base alias name if it's namespaced
                # This allows "test_hub_op" to resolve to "testuser.test.test_hub_op"
                if "." in alias:
                    base_alias = alias.split(".")[-1]
                    if base_alias in target_ops:
                        # If base alias already exists, ensure it points to the same OpDef
                        if target_ops[base_alias] != target_ops[op_name]:
                            self._report_bare_name_contest(
                                target_ops[base_alias],
                                target_ops[op_name],
                                f"Base alias '{base_alias}' already has an assigned op or alias so the alias "
                                f"specified by '{alias}' cannot be mapped to it. Address it by its "
                                f"fully-qualified name, or opt into the bare name with it.hub.prefer_ops().",
                            )
                    else:
                        if base_alias != op_name and base_alias != alias:
                            target_ops[base_alias] = target_ops[op_name]

        return target_ops

    def _import_callable(self, callable_path: str) -> Callable:
        """Import a callable from a path."""
        module_path, func_name = callable_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            imported_fn = getattr(module, func_name)
        except Exception as e:
            raise ValueError(
                f"Import of the specified function {func_name} from {module_path} (specified callable "
                f"path {callable_path}) failed with the following exception: {e}"
            )
        return imported_fn

    def _import_hub_callable(self, op_name: str, op_def: OpDef) -> Callable:
        """Import a callable from a hub path."""
        rank_zero_debug(f"Attempting dynamic loading for namespaced operation: {op_name}")

        # Extract repo name from the operation name and module/function from implementation field
        # Format of op_name: "repo_name.function_name" or "user.repo.function_name"
        parts = op_name.split(".")
        if len(parts) >= 3:
            # Take the first two parts as repo identifier
            repo_name = ".".join(parts[:2])
        else:
            raise ValueError(f"Invalid namespaced operation format: {op_name}. Expected 'user.repo.function_name'")

        # Extract module and function names from implementation field
        if not op_def.implementation:
            raise ValueError(f"No implementation specified for hub operation: {op_name}")

        implementation_parts = op_def.implementation.split(".")
        if len(implementation_parts) < 2:
            raise ValueError(f"Invalid implementation format: {op_def.implementation}. Expected 'module.function'")

        # Last part is function name, everything before is module path
        function_name = implementation_parts[-1]
        module_name = ".".join(implementation_parts[:-1])

        function_reference = f"{module_name}.{function_name}"

        implementation = get_function_from_dynamic_module(
            function_reference=function_reference,
            op_repo_name_or_path=repo_name,
            cache_dir=IT_ANALYSIS_HUB_CACHE,
        )
        rank_zero_debug(f"Successfully loaded dynamic operation: {op_name}")
        return implementation

    @staticmethod
    def _function_param_from_hub_module(param_path: str, implementation: Callable) -> Callable | None:
        # Try to use the dynamically loaded module if module names match
        func_name = param_path.rsplit(".", 1)[-1]
        param_module = param_path.rsplit(".", 1)[0]
        imported_module_name = implementation.__module__.split(".")[-1]
        resolved_fn_param = None

        if param_module == imported_module_name:
            # Get the module object from the implementation function
            import sys

            module_obj = sys.modules.get(implementation.__module__)
            if module_obj is not None:
                resolved_fn_param = getattr(module_obj, func_name, None)
        return resolved_fn_param

    # The only interpretune namespace a hub op may bind an importable_param to. Anything else is an
    # unsanctioned reach into internals -- the `_import_callable` fallback would happily import any
    # installed dotted path, which is the one privilege leak #266 left open on the hub side.
    _SANCTIONED_HUB_PARAM_NAMESPACE = "interpretune.analysis.optools"

    def _resolve_protocol_cls(self, op_name: str, op_def: OpDef):
        """Import an op's declared `BaseAnalysisBatchProtocol` subclass, or None to use the default (#56).

        Trust-gated for hub ops on the SAME footing as `importable_params`, and for the same reason: a
        declared protocol is an arbitrary class import, so a hub collection pointing one at an
        interpretune-internal module would resolve our code on its behalf. Its own repo modules and
        third-party targets are unchanged -- those carry the trust the op itself already carries.

        Fails SOFT. A protocol is descriptive of a batch's shape, so an unresolvable one degrades to the
        default rather than taking down a load: the op still runs, it just does not get its richer
        attribute surface. Reported through the load policy, so it warns by default and raises under
        IT_STRICT_OP_LOAD.
        """
        path = op_def.protocol_cls
        if not path:
            return None
        if op_def.source.startswith("hub") and not self._hub_param_target_is_sanctioned(op_name, "protocol_cls", path):
            return None
        try:
            protocol_cls = self._import_callable(path)
        except Exception as e:
            op_load_failure(f"Could not import protocol_cls '{path}' for operation '{op_name}': {e}")
            return None
        # NOT `issubclass(..., BaseAnalysisBatchProtocol)`: that base is a `Protocol` and is not
        # `@runtime_checkable`, so issubclass against it raises TypeError rather than returning False.
        # Walking the MRO asks the question we actually mean -- "does this declare the base in its
        # ancestry" -- without depending on protocol runtime semantics or forcing a decorator onto a
        # type that has no other reason to carry one.
        if not isinstance(protocol_cls, type) or BaseAnalysisBatchProtocol not in protocol_cls.__mro__:
            op_load_failure(
                f"protocol_cls '{path}' for operation '{op_name}' is not a BaseAnalysisBatchProtocol "
                f"subclass; falling back to the default protocol."
            )
            return None
        return protocol_cls

    def _hub_param_target_is_sanctioned(self, op_name: str, param_name: str, param_path: str) -> bool:
        """Whether a hub op may bind ``param_name`` to ``param_path`` outside its own repo modules.

        This is the opening half of the warn-then-error window: an unsanctioned target is *skipped*
        (resolving it is what the restriction exists to prevent) and reported through the load
        policy, so it warns by default and raises under ``IT_STRICT_OP_LOAD``. Promoting the default
        to an error is a follow-on once the window has elapsed.
        """
        if param_path == self._SANCTIONED_HUB_PARAM_NAMESPACE or param_path.startswith(
            self._SANCTIONED_HUB_PARAM_NAMESPACE + "."
        ):
            return True
        if param_path.split(".")[0] != "interpretune":
            # Non-interpretune targets (its own repo modules, third-party deps) are unchanged: the
            # hub-module resolution above is what normally binds them.
            return True
        op_load_failure(
            f"Importable parameter '{param_name}' in hub operation '{op_name}' targets "
            f"'{param_path}', which is interpretune-internal. Hub op collections may bind "
            f"importable_params only to modules in their own repo or to "
            f"'{self._SANCTIONED_HUB_PARAM_NAMESPACE}'. This parameter is skipped."
        )
        return False

    @_ensure_loaded
    def _instantiate_op(self, op_name: str) -> AnalysisOp:
        """Instantiate an operation from its definition."""
        op_def = self._op_definitions.get(op_name)
        if not op_def:
            raise ValueError(f"Unknown operation: {op_name}")

        # Handle composite operations
        if op_def.composition is not None:
            composition = op_def.composition
            # instantiate each operation in the composition
            raw_ops = [self.get_op(op) for op in composition]
            # Filter to ensure we only have AnalysisOp objects
            ops = [op for op in raw_ops if isinstance(op, AnalysisOp)]
            if len(ops) != len(raw_ops):
                raise ValueError(f"Composition for {op_name} contains non-AnalysisOp objects")
            op = CompositeAnalysisOp(ops, name=op_name, aliases=op_def.aliases)
            op.description = op_def.description
            op.input_schema = op_def.input_schema
            op.output_schema = op_def.output_schema
            # A composite carries its OWN declared traits; its members carry theirs (op_state in
            # particular is bound per-member at call time, not on the composite).
            op.uses_default_hooks = op_def.uses_default_hooks
            op.requires_grad = op_def.requires_grad
            op.per_latent_preds = op_def.per_latent_preds
            return op

        # Hub ops load dynamically from the hub cache. Keyed on declared provenance rather than on
        # counting dots in the op name.
        if _is_hub_op := (op_def.source.startswith("hub") and self.enable_hub_ops):
            implementation = self._import_hub_callable(op_def.name, op_def)
        else:
            # Handle regular operations
            implementation = self._import_callable(op_def.implementation)

        # Build impl_params from importable_params and normal_params
        impl_params = {}

        # Import any additional functions specified in importable_params
        for param_name, param_path in op_def.importable_params.items():
            resolved_fn_param = None
            if _is_hub_op:
                resolved_fn_param = AnalysisOpDispatcher._function_param_from_hub_module(param_path, implementation)
            if not resolved_fn_param:
                if _is_hub_op and not self._hub_param_target_is_sanctioned(op_name, param_name, param_path):
                    continue
                resolved_fn_param = self._import_callable(param_path)
            if resolved_fn_param is None:
                op_load_failure(
                    f"Importable parameter '{param_name}' in operation '{op_name}' could not be resolved: "
                    f"{param_path}. It will not be available in the operation."
                )
                continue
            impl_params[param_name] = resolved_fn_param

        # Add normal parameters
        impl_params.update(op_def.normal_params)

        op = AnalysisOp(
            name=op_name,
            description=op_def.description,
            output_schema=op_def.output_schema,
            input_schema=op_def.input_schema,
            aliases=op_def.aliases,
            impl_params=impl_params,
            required_capabilities=op_def.required_capabilities,
            op_state=op_def.op_state,
            uses_default_hooks=op_def.uses_default_hooks,
            requires_grad=op_def.requires_grad,
            per_latent_preds=op_def.per_latent_preds,
            protocol_cls=self._resolve_protocol_cls(op_name, op_def),
        )

        # Set the implementation
        op._impl = implementation

        return op

    def _is_lazy_op_handle(self, obj) -> bool:
        """Check if an object is a lazy operation handle (factory function)."""
        return callable(obj) and not isinstance(obj, AnalysisOp)

    def _convert_to_op_schema(self, schema_dict: Dict) -> OpSchema:
        """Convert a schema dictionary to an OpSchema object with ColCfg values."""
        result = {}
        for field_name, field_config in schema_dict.items():
            if isinstance(field_config, dict):
                result[field_name] = ColCfg(**field_config)
            elif isinstance(field_config, ColCfg):
                result[field_name] = field_config
        return OpSchema(result)

    @_ensure_loaded
    def _is_resolvable_op_name(self, op_name: str) -> bool:
        """Whether ``op_name`` names a single registered op (directly or via an alias)."""
        try:
            return self._resolve_name_safe(op_name) in self._op_definitions
        except Exception:
            return False

    @_ensure_loaded
    def get_op(self, op_name: str, context: DispatchContext | None = None, lazy: bool = False) -> AnalysisOp | Callable:
        """Get an operation by name, optionally instantiating it if needed.

        Args:
            op_name: Name of the operation to retrieve
            context: Optional context for operation dispatching
            lazy: If True, defer instantiation until the operation is actually used

        Returns:
            The requested operation or None if lazy=True and the op hasn't been instantiated yet
        """
        if context is None:
            context = DispatchContext()

        # Resolve names with cycle detection
        resolved_name = self._resolve_name_safe(op_name)

        # Check if operation exists
        if resolved_name not in self._op_definitions:
            raise ValueError(f"Unknown operation: {op_name}")

        # Get or create dispatch table entry for this operation
        if resolved_name not in self._dispatch_table:
            self._dispatch_table[resolved_name] = {}

        ctx_dict = self._dispatch_table[resolved_name]

        # Check if we already have an entry for this context
        if context in ctx_dict:
            existing = ctx_dict[context]
            if lazy:
                # For lazy requests, return whatever we have (factory or instance)
                return existing
            elif self._is_lazy_op_handle(existing):
                # We have a factory function but need an instance
                ctx_dict[context] = self._instantiate_op(resolved_name)
                return ctx_dict[context]
            else:
                # We already have an instantiated operation
                return existing

        # No entry for this context yet
        if lazy:
            # Store a factory function that will instantiate the op when needed
            ctx_dict[context] = lambda: self._instantiate_op(resolved_name)
        else:
            # Eagerly instantiate the operation
            ctx_dict[context] = self._instantiate_op(resolved_name)
        return ctx_dict[context]

    def _maybe_instantiate_op(self, op_ref, context: DispatchContext = DispatchContext()) -> AnalysisOp:
        """Ensure an operation is instantiated based on various reference types."""
        # If it's an OpWrapper, use its _ensure_instantiated method to get the actual op
        if isinstance(op_ref, OpWrapper):
            result = op_ref._ensure_instantiated()  # This now returns the actual op, not the wrapper
            if not isinstance(result, AnalysisOp):
                raise TypeError(f"Expected AnalysisOp, got {type(result)}")
            return result

        # If it's an AnalysisOp, get the op name
        if isinstance(op_ref, AnalysisOp):
            op_name = op_ref.name
        else:
            assert isinstance(op_ref, str), "op_ref must be an OpWrapper, AnalysisOp or a string"
            op_name = op_ref

        ctx_dict = self._dispatch_table.get(op_name, {})
        op = ctx_dict.get(context)

        # TODO: decide if we want to handle this edge case where the dispatch_table contains a factory function
        #       that was not added by OpWrapper, basically custom op lazy loading
        # Check if the stored value is a factory function or needs to be instantiated
        if callable(op) and not isinstance(op, AnalysisOp):
            # Instantiate the operation and update the dispatch table
            instantiated_op = op()
            if not isinstance(instantiated_op, AnalysisOp):
                raise TypeError(f"Factory function returned {type(instantiated_op)}, expected AnalysisOp")
            ctx_dict[context] = instantiated_op
            return instantiated_op
        elif op is not None:
            if not isinstance(op, AnalysisOp):
                raise TypeError(f"Expected AnalysisOp, got {type(op)}")
            return op
        else:
            # Try to get the op if it's not in the dispatch table
            result = self.get_op(op_name, context)
            if not isinstance(result, AnalysisOp):
                raise TypeError(f"get_op returned {type(result)}, expected AnalysisOp")
            return result

    @_ensure_loaded
    def instantiate_all_ops(self) -> dict[str, AnalysisOp]:
        """Get all operations as instantiated AnalysisOp objects."""
        instantiated_ops = {}

        # Only instantiate operations that are not aliases pointing to other operations
        for op_name in self._op_definitions:
            # Skip if this is an alias that points to a different operation
            if op_name in self._aliases and self._aliases[op_name] != op_name:
                continue

            try:
                op = self.get_op(op_name)
                if isinstance(op, AnalysisOp):
                    instantiated_ops[op_name] = op
            except Exception as e:
                rank_zero_warn(f"Failed to instantiate operation '{op_name}': {e}")
                continue

        return instantiated_ops

    @_ensure_loaded
    def compile_ops(
        self, op_names: str | list[str | AnalysisOp], name: str | None = None, aliases: list[str] | None = None
    ) -> CompositeAnalysisOp:
        """Create a composition of operations from a list of operation names."""
        # See NOTE [Composition and Compilation Limitations]
        # Support for dot-separated string format
        if isinstance(op_names, str):
            op_names = op_names.split(".")  # type: ignore[assignment]  # Converting str to list[str]
        # If op_names is a list, split any string elements containing '.' into multiple op names
        elif isinstance(op_names, list):
            split_names = []
            for op_name in op_names:
                if isinstance(op_name, str) and "." in op_name:
                    split_names.extend(op_name.split("."))
                else:
                    split_names.append(op_name)
            op_names = split_names

        # Convert all op references to AnalysisOp objects
        ops = []
        for op_name in op_names:
            if isinstance(op_name, str):
                op = self.get_op(op_name)
            else:
                # Handle OpWrapper and other non-string references
                op = self._maybe_instantiate_op(op_name)
            ops.append(op)

        return CompositeAnalysisOp(ops, name=name, aliases=aliases)

    def __call__(
        self,
        op_name: str,
        module: torch.nn.Module | None = None,
        analysis_batch: BaseAnalysisBatchProtocol | None = None,
        batch: BatchEncoding | None = None,
        batch_idx: int | None = None,
    ) -> BaseAnalysisBatchProtocol:
        """Call an operation by name.

        A dotted name is a composition ("op_a.op_b") *unless* it resolves to a single registered op,
        which is how hub ops are namespaced ("user.repo.op"). Resolution wins, matching ``get_op``;
        splitting first meant calling a namespaced hub op through the dispatcher tried to compose
        three nonexistent ops.
        """
        if "." in op_name and not self._is_resolvable_op_name(op_name):
            composite_op = self.compile_ops(op_name)
            return composite_op(module=module, analysis_batch=analysis_batch, batch=batch, batch_idx=batch_idx)

        # Get the operation, instantiating it if it's a factory function
        op = self.get_op(op_name)

        return op(module=module, analysis_batch=analysis_batch, batch=batch, batch_idx=batch_idx)

    def _resolve_op_paths_from_yaml_paths(self):
        """Resolve op_paths from yaml_paths.

        For directories in yaml_paths, add the yaml_path to op_paths. For yaml files, add the direct parent directory of
        the yaml file to op_paths. The bundled op tree is skipped: its YAMLs resolve implementations via fully-qualified
        interpretune module paths and must not leak family packages onto sys.path.
        """
        bundled_dir = getattr(self, "_bundled_ops_dir", None)
        bundled_resolved = Path(bundled_dir).resolve() if bundled_dir is not None else None
        for yaml_path in self.yaml_paths:
            yaml_path = Path(yaml_path).resolve()
            if bundled_resolved is not None and yaml_path == bundled_resolved:
                continue

            if yaml_path.is_dir():
                # Add directory to op_paths if not already present
                if yaml_path not in self.op_paths:
                    self.op_paths.append(yaml_path)
            elif yaml_path.is_file():
                # Add parent directory of yaml file to op_paths if not already present
                parent_dir = yaml_path.parent
                if parent_dir not in self.op_paths:
                    self.op_paths.append(parent_dir)


# Global dispatcher instance
DISPATCHER = AnalysisOpDispatcher()
