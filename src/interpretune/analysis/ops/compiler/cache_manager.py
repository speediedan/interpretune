"""Cache manager for pre-compiled operation definitions."""

from __future__ import annotations
import hashlib
import importlib.util
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field


from huggingface_hub import scan_cache_dir

from interpretune.hub.cache import _get_latest_revision, parse_hub_cache_path
from interpretune.hub.manifest import IT_COMPONENT_MANIFEST
from interpretune.hub.pins import read_op_pin
from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR, remote_code_trust, remote_code_trusted
from interpretune.analysis.ops.compiler.load_policy import OpLoadError, op_load_failure

from interpretune.utils.logging import rank_zero_debug, rank_zero_warn
from interpretune.analysis.inputs import OpStateSpec
from interpretune.analysis.ops.base import OpSchema, ColCfg


# 3: adds the `op_state` trait (declared cross-batch state) to OpDef.
# 4: adds declared collection identity (`collection_name`/`collection_version`).
# 5: inputs inherited via `required_ops` compile to `required=False` (see NOTE [Inherited Inputs Are Not
#    Obligations] in schema_compiler.py). The cache key covers YAML content and the interpretune version but
#    NOT the compiler source, so without this bump an existing cache keeps serving the previous requiredness
#    and the change would silently not take effect.
# 6: adds `protocol_cls` (a user-defined BaseAnalysisBatchProtocol subclass, declared as an import path)
#    to OpDef. A serialized-shape change, so an existing cache would otherwise keep serving OpDefs without
#    the field and a declaring op would silently fall back to the default protocol.
# 7: name/description/implementation emit via repr() -- a description containing a double quote
#    previously rendered an unparseable module (silent full recompile on every load).
CACHE_FORMAT_VERSION = "7"


@dataclass(frozen=True)
class OpDef:
    """Frozen dataclass representing a pre-compiled operation definition."""

    name: str
    description: str
    implementation: str
    input_schema: OpSchema
    output_schema: OpSchema
    aliases: list[str] = field(default_factory=list)
    importable_params: dict[str, str] = field(default_factory=dict)
    normal_params: dict[str, Any] = field(default_factory=dict)
    required_ops: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    composition: list[str] | None = None
    op_state: OpStateSpec | None = None
    # Where this definition came from: "bundled" | "local" | "hub:<user.repo>". Provenance, not name
    # shape: a dotted name means namespaced, which is how hub ops are addressed, but only provenance
    # answers "should this load dynamically from the hub cache".
    source: str = "bundled"
    # Declared collection identity (the `collection:` header). Serialized into the cache -- and worth its
    # CACHE_FORMAT_VERSION bump -- because unlike the declaration-site path this is wanted at RUNTIME:
    # `op_info` reports which collection and version a bare name resolves to, and the cache path would
    # otherwise leave it empty on every warm load.
    collection_name: str | None = None
    collection_version: str | None = None
    # Behavioral traits. These replace name-based special cases: framework code asks what an op
    # NEEDS, so hub and local ops can declare the same things bundled ops do.
    # Import path of a BaseAnalysisBatchProtocol subclass this op's batches conform to. A STRING
    # rather than a class so an OpDef stays serializable into the generated cache module and a YAML author
    # can declare one without importing it; resolution is the dispatcher's job, and is trust-gated for hub
    # ops exactly as `importable_params` is.
    protocol_cls: str | None = None
    uses_default_hooks: bool = False  # install the default activation-cache fwd/bwd hooks
    requires_grad: bool = False  # run the analysis loop with grad enabled
    per_latent_preds: bool = False  # preds are per-latent-model and join across SAEs before scoring

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for compatibility with existing code."""
        result = {
            "name": self.name,
            "description": self.description,
            "implementation": self.implementation,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "aliases": self.aliases,
            "importable_params": self.importable_params,
            "normal_params": self.normal_params,
            "required_ops": self.required_ops,
            "required_capabilities": self.required_capabilities,
            "composition": self.composition,
            "op_state": self.op_state.to_dict() if self.op_state is not None else None,
            "source": self.source,
            "collection_name": self.collection_name,
            "collection_version": self.collection_version,
            "protocol_cls": self.protocol_cls,
            "uses_default_hooks": self.uses_default_hooks,
            "requires_grad": self.requires_grad,
            "per_latent_preds": self.per_latent_preds,
        }
        return result


def _interpretune_version() -> str:
    """The installed interpretune version, for the cache key ("unknown" when undeterminable)."""
    from interpretune.hub.components import _installed_version

    return str(_installed_version("interpretune") or "unknown")


class YamlFileInfo:
    """Information about a YAML file for caching purposes."""

    def __init__(self, path: Path, mtime: float, content_hash: str):
        self.path = path
        self.mtime = mtime
        self.content_hash = content_hash

    @classmethod
    def from_path(cls, path: Path) -> "YamlFileInfo":
        """Create YamlFileInfo from a file path."""
        stat = path.stat()
        content = path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        return cls(path, stat.st_mtime, content_hash)


class OpDefinitionsCacheManager:
    """Manages caching of compiled operation definitions."""

    _it_trust_false_skipping = (
        f"Skipping loading ops from hub repositories: {IT_TRUST_REMOTE_CODE_ENV_VAR} is set to a "
        "non-affirmative value, which is a deliberate opt-out from executing hub-resident code."
    )

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._yaml_files: list[YamlFileInfo] = []
        self._fingerprint: str | None = None

    def add_yaml_file(self, yaml_file: Path) -> None:
        """Add a YAML file to be monitored for changes."""
        # Avoid duplicates
        if any(info.path == yaml_file for info in self._yaml_files):
            return

        try:
            file_info = YamlFileInfo.from_path(yaml_file)
            self._yaml_files.append(file_info)
            self._fingerprint = None  # Reset fingerprint when files change
        except FileNotFoundError:
            # Skip files that don't exist anymore
            pass

    def add_hub_yaml_files(self) -> list[Path]:
        """Add hub YAML files to monitoring."""
        hub_yaml_files = []
        try:
            # short-circuit on a deliberate opt-out; an UNSET decision is reported by discovery,
            # where a real cached repo can be named in the advice
            if remote_code_trust() is False:
                rank_zero_warn(OpDefinitionsCacheManager._it_trust_false_skipping)
                rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Returning early: {IT_TRUST_REMOTE_CODE_ENV_VAR} opt-out")
                return []

            hub_yaml_files = self.discover_hub_yaml_files()
            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Discovered {len(hub_yaml_files)} YAML files")

            for yaml_file in hub_yaml_files:
                rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Adding YAML file: {yaml_file}")
                self.add_yaml_file(yaml_file)

        except OpLoadError:
            raise  # strict loading must not be swallowed by the fail-soft discovery wrapper
        except Exception as e:
            rank_zero_warn(f"Error discovering hub YAML files: {e}")
            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Exception details: {type(e).__name__}: {str(e)}")
            import traceback

            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Traceback: {traceback.format_exc()}")

        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Returning {len(hub_yaml_files)} YAML files")
        return hub_yaml_files  # type: ignore[return-value]

    def discover_hub_yaml_files(self) -> list[Path]:
        """Discover op-definition YAMLs from the latest cached revision of each hub op collection.

        MANIFEST-ROUTED (not a glob): each cached repo's ``it_component.yaml`` is read first and its
        ``ops.files`` list says which YAMLs are op definitions. A repo carrying no manifest is reported
        through ``op_load_failure`` and contributes nothing -- that is the well-formed-ops-repo contract, and
        reporting it is the point, since under the old glob such a repo appeared to work until one of its
        non-op YAMLs reached the compiler and dropped every op in the process.

        PIN-FIRST (#334): a repo pinned by a revision-pinned pull loads exactly the pinned revision,
        beating ``refs/main``; an unpinned repo keeps the refs-based selection below.

        Returns:
            List of Path objects pointing to declared op YAMLs, from one revision per collection.
        """
        from interpretune.analysis import IT_ANALYSIS_HUB_CACHE

        yaml_files = []

        # in case this method is called directly, honor a deliberate opt-out before touching the cache
        trust = remote_code_trust()
        if trust is False:
            rank_zero_warn(OpDefinitionsCacheManager._it_trust_false_skipping)
            return yaml_files

        # Convert to Path object for consistent handling
        hub_cache_path = Path(IT_ANALYSIS_HUB_CACHE)
        if not hub_cache_path.exists():
            return yaml_files

        try:
            # Use HuggingFace cache manager
            cache_info = scan_cache_dir(hub_cache_path)

            # Sort repos by repo_id for deterministic ordering
            sorted_repos = sorted(cache_info.repos, key=lambda repo: repo.repo_id)

            # An unset decision denies, once, with advice naming a real cached repo. Ops discovery
            # degrades rather than raising: a session with no hub ops still works, so failing the
            # first op access over an undeclared preference would be disproportionate.
            if sorted_repos and not remote_code_trusted(sorted_repos[0].repo_id, what="analysis ops"):
                return yaml_files

            for repo in sorted_repos:
                # Only consider model repositories
                if repo.repo_type != "model":
                    continue

                # Find the latest revision for this repo (preferring 'main' ref)
                latest_revision = _get_latest_revision(repo)
                if latest_revision is None:
                    continue

                # A durable pin (written by a revision-pinned pull; see interpretune.hub.pins)
                # beats every ref: the trust posture promises that pinning a revision means trusted
                # code cannot change under you, and discovery is where that promise is either kept
                # or broken. Binding is STRICT -- a pinned revision that is no longer cached (or
                # cached without its manifest) is refused with the restore/release gesture, never
                # silently substituted with whatever `main` moved to.
                pin = read_op_pin(repo.repo_id, cache_root=hub_cache_path)
                if pin is not None:
                    pinned = next((rev for rev in repo.revisions if rev.commit_hash == pin["commit"]), None)
                    if pinned is not None and (pinned.snapshot_path / IT_COMPONENT_MANIFEST).is_file():
                        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] {repo.repo_id}: honoring pin {pin['commit'][:12]}")
                        yaml_files.extend(self._declared_op_files(repo.repo_id, pinned.snapshot_path))
                    else:
                        state = "cached without its manifest" if pinned is not None else "no longer cached"
                        op_load_failure(
                            f"op collection {repo.repo_id!r} is pinned to revision "
                            f"{pin['commit'][:12]}, which is {state}; refusing to substitute another "
                            f"revision for a pinned one. Restore it with "
                            f"it.hub.pull_ops({repo.repo_id!r}, revision={pin['commit']!r}) or "
                            f"release the pin with it.hub.unpin_ops({repo.repo_id!r})."
                        )
                    continue

                # The refs/main snapshot can be PARTIAL: huggingface_hub materializes a snapshot dir per
                # resolved revision containing only the files actually fetched at it, so any single-file
                # fetch at `main` (a card read, a trust inspection) after a REVISION-PINNED pull leaves
                # refs/main pointing at a snapshot without the manifest while the complete pinned
                # snapshot sits beside it. Routing discovery through the partial one skips the whole
                # collection with "no it_component.yaml in the cached snapshot" -- observed as a pull
                # that succeeded while its ops never loaded. Prefer main only when its
                # snapshot is manifest-complete; otherwise fall back to the newest revision that is,
                # and say which mismatch happened rather than silently choosing.
                candidates = [latest_revision] + sorted(
                    (rev for rev in repo.revisions if rev is not latest_revision),
                    key=lambda rev: rev.last_modified,
                    reverse=True,
                )
                chosen = next(
                    (rev for rev in candidates if (rev.snapshot_path / IT_COMPONENT_MANIFEST).is_file()),
                    None,
                )
                if chosen is None:
                    # no cached revision has a manifest: report once via the normal per-collection path
                    self._declared_op_files(repo.repo_id, latest_revision.snapshot_path)
                    continue
                if chosen is not latest_revision:
                    rank_zero_warn(
                        f"op collection {repo.repo_id!r}: refs/main snapshot "
                        f"{latest_revision.commit_hash[:12]} has no {IT_COMPONENT_MANIFEST} (partial "
                        f"fetch); using cached revision {chosen.commit_hash[:12]} instead"
                    )

                # The manifest declares which of this snapshot's YAMLs are op definitions. Every path
                # therefore comes from one revision by construction, and the manifest is never itself fed
                # to the op compiler (it raises on its scalar keys, dropping every op including bundled).
                yaml_files.extend(self._declared_op_files(repo.repo_id, chosen.snapshot_path))

        except OpLoadError:
            raise  # strict loading must not be swallowed by the fail-soft discovery wrapper
        except Exception as e:
            rank_zero_warn(f"Failed to discover hub YAML files: {e}")

        return sorted(yaml_files)  # Sort for deterministic results

    def _declared_op_files(self, repo_id: str, snapshot_path: Path) -> list[Path]:
        """Manifest-routed op YAMLs for one cached collection, with load failures reported not raised.

        ``OpLoadError`` propagates so strict loading still fails the session; every other manifest problem
        is scoped to this one collection, because a malformed third-party collection must not be able to
        deny a session the ops it does have.
        """
        from interpretune.hub.opcollections import resolve_cached_op_files

        try:
            return resolve_cached_op_files(snapshot_path, source=f"op collection {repo_id!r}")
        except OpLoadError:
            raise
        except Exception as failure:
            op_load_failure(f"Skipping op collection {repo_id!r}: {failure}")
            return []

    def _parse_hub_file_path(self, yaml_file: Path) -> tuple[bool, str]:
        """Parse a file path to determine if it's a hub ops file and extract namespace.

        Args:
            yaml_file: Path to the YAML file to analyze

        Returns:
            Tuple of (is_hub_file, namespace) where:
            - is_hub_file: True if this is a hub operations file
            - namespace: The extracted namespace (empty string if not a hub file)
        """
        from interpretune.analysis import IT_ANALYSIS_HUB_CACHE

        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Input yaml_file: {yaml_file}")
        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] IT_ANALYSIS_HUB_CACHE: {IT_ANALYSIS_HUB_CACHE}")
        # cache-layout parsing lives in the unified hub layer now (interpretune.hub.cache)
        is_hub_file, namespace = parse_hub_cache_path(yaml_file, Path(IT_ANALYSIS_HUB_CACHE))
        if not is_hub_file:
            rank_zero_debug("[ANALYSIS_HUB_CACHE] NOT A HUB FILE - returning (False, '')")
        return is_hub_file, namespace

    def hub_commit_for_namespace(self, namespace: str) -> str | None:
        """The snapshot commit the monitored YAMLs for one hub namespace were ACTUALLY loaded from.

        The exact provenance answer: discovery may choose a pinned revision or a manifest-complete
        fallback over ``refs/main``, and only the monitored paths record which snapshot won. Returns
        ``None`` when no monitored file belongs to the namespace (e.g. before discovery has run in
        this process), letting the caller fall back to a filesystem read.
        """
        for info in self._yaml_files:
            is_hub_file, ns = self._parse_hub_file_path(info.path)
            if is_hub_file and ns == namespace:
                parts = info.path.parts
                if "snapshots" in parts:
                    commit = parts[parts.index("snapshots") + 1]
                    if commit:
                        return commit
        return None

    def get_hub_namespace(self, yaml_file: Path) -> str:
        """Extract namespace from hub file path."""
        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] get_hub_namespace input: {yaml_file}")

        is_hub_file, namespace = self._parse_hub_file_path(yaml_file)

        rank_zero_debug(
            f"[ANALYSIS_HUB_CACHE] get_hub_namespace result: is_hub_file={is_hub_file}, namespace='{namespace}'"
        )
        return namespace

    @property
    def fingerprint(self) -> str:
        """Get a fingerprint representing the current state of all YAML files."""
        if self._fingerprint is None:
            if not self._yaml_files:
                self._fingerprint = f"empty_v{CACHE_FORMAT_VERSION}"
            else:
                # Create a combined hash of all file information. The installed interpretune version is part
                # of the key because compiled definitions depend on it: a collection's `requires:` window is
                # enforced at COMPILE time, so without this an upgrade would keep serving definitions from a
                # collection that the new version no longer satisfies, never re-running the check.
                combined_info = [f"cache_format:{CACHE_FORMAT_VERSION}", f"interpretune:{_interpretune_version()}"]
                for file_info in self._yaml_files:
                    # Include path, mtime, and content hash
                    combined_info.append(f"{file_info.path}:{file_info.mtime}:{file_info.content_hash}")

                combined_str = "|".join(sorted(combined_info))
                full_hash = hashlib.sha256(combined_str.encode()).hexdigest()
                self._fingerprint = full_hash[:16]  # Truncate for readability

        return self._fingerprint

    def _get_cache_module_path(self) -> Path:
        """Get the path for the cache module file."""
        return self.cache_dir / f"op_definitions_{self.fingerprint}.py"

    def _cleanup_old_cache_files(self) -> None:
        """Remove old cache files."""
        pattern = "op_definitions_*.py"
        current_file = f"op_definitions_{self.fingerprint}.py"

        for old_file in self.cache_dir.glob(pattern):
            if old_file.name != current_file:
                try:
                    old_file.unlink()
                    rank_zero_debug(f"Removed old cache file: {old_file}")
                except OSError as e:
                    rank_zero_warn(f"Failed to remove old cache file {old_file}: {e}")

    def is_cache_valid(self) -> bool:
        """Check if the current cache is valid."""

        cache_path = self._get_cache_module_path()
        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Cache path: {cache_path}")

        if not cache_path.exists():
            rank_zero_debug("[ANALYSIS_HUB_CACHE] Cache invalid: file does not exist")
            return False

        cache_mtime = cache_path.stat().st_mtime
        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Cache mtime: {cache_mtime}")
        rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Checking {len(self._yaml_files)} source files")

        # Check if any source files are newer than cache
        for file_info in self._yaml_files:
            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Checking source file: {file_info.path}")
            if not file_info.path.exists():
                rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Cache invalid: source file missing {file_info.path}")
                return False

            source_mtime = file_info.path.stat().st_mtime
            rank_zero_debug(f"[ANALYSIS_HUB_CACHE] Source mtime: {source_mtime} vs cache: {cache_mtime}")
            if source_mtime > cache_mtime:
                rank_zero_debug("[ANALYSIS_HUB_CACHE] Cache invalid: source newer than cache")
                return False

        rank_zero_debug("[ANALYSIS_HUB_CACHE] Cache is valid")
        return True

    def _generate_module_content(self, op_definitions: dict[str, OpDef]) -> str:
        """Generate Python module content for the cache."""
        lines = [
            "# GENERATED FILE - DO NOT EDIT",
            "# This file contains cached operation definitions",
            f"# Fingerprint: {self.fingerprint}",
            "",
            "from interpretune.analysis.inputs import OpStateSpec",
            "from interpretune.analysis.ops.base import OpSchema, ColCfg",
            "from interpretune.analysis.ops.compiler.cache_manager import OpDef",
            "",
            f'FINGERPRINT = "{self.fingerprint}"',
            "",
            "OP_DEFINITIONS = {",
        ]

        # Filter out alias entries - only serialize canonical operation definitions
        # Aliases will be reconstructed from the dispatcher's _aliases mapping
        canonical_ops = {}
        for name, op_def in op_definitions.items():
            # Only include operations where the name matches the canonical name
            if op_def.name == name:
                canonical_ops[name] = op_def

        for name, op_def in canonical_ops.items():
            op_def_str = self._serialize_op_def(op_def)
            lines.append(f'    "{name}": {op_def_str},')

        lines.append("}")

        return "\n".join(lines)

    def _serialize_op_def(self, op_def: OpDef) -> str:
        """Serialize an OpDef to Python code."""
        fields = []

        # Always include required fields. `!r` rather than hand-quoting: a description containing a
        # double quote (a hub collection wrote `a lens-coordinate "patch" intervention`) rendered an
        # unparseable module, and the failure mode was maximally quiet -- a "Failed to load cache"
        # warning plus a full recompile on EVERY subsequent load, in every session, for as long as the
        # collection stayed cached. repr() escapes everything Python source needs escaped.
        fields.append(f"name={op_def.name!r}")
        fields.append(f"description={op_def.description!r}")
        fields.append(f"implementation={op_def.implementation!r}")
        fields.append(f"input_schema={self._serialize_op_schema(op_def.input_schema)}")
        fields.append(f"output_schema={self._serialize_op_schema(op_def.output_schema)}")

        # Include optional fields that have values
        if op_def.aliases:
            fields.append(f"aliases={op_def.aliases!r}")
        if op_def.importable_params:
            fields.append(f"importable_params={op_def.importable_params!r}")
        if op_def.normal_params:
            fields.append(f"normal_params={op_def.normal_params!r}")
        if op_def.required_ops:
            fields.append(f"required_ops={op_def.required_ops!r}")
        if op_def.required_capabilities:
            fields.append(f"required_capabilities={op_def.required_capabilities!r}")
        if op_def.composition:
            fields.append(f"composition={op_def.composition!r}")
        if op_def.op_state is not None:
            fields.append(f"op_state={self._serialize_op_state(op_def.op_state)}")
        if op_def.source != "bundled":
            fields.append(f"source={op_def.source!r}")
        if op_def.protocol_cls is not None:
            fields.append(f"protocol_cls={op_def.protocol_cls!r}")
        for collection_field in ("collection_name", "collection_version"):
            if (value := getattr(op_def, collection_field)) is not None:
                fields.append(f"{collection_field}={value!r}")
        for trait in ("uses_default_hooks", "requires_grad", "per_latent_preds"):
            if getattr(op_def, trait):
                fields.append(f"{trait}=True")

        return f"OpDef({', '.join(fields)})"

    def _serialize_op_state(self, op_state: OpStateSpec) -> str:
        """Serialize an OpStateSpec to Python code."""
        return (
            f"OpStateSpec(fields={tuple(op_state.fields)!r}, scope={op_state.scope!r}, "
            f"reset_each_epoch={op_state.reset_each_epoch!r})"
        )

    def _serialize_op_schema(self, schema: OpSchema) -> str:
        """Serialize an OpSchema to Python code."""
        if not schema:
            return "OpSchema({})"

        fields = []
        for field_name, col_cfg in schema.items():
            fields.append(f'"{field_name}": {self._serialize_col_cfg(col_cfg)}')

        return f"OpSchema({{{', '.join(fields)}}})"

    def _serialize_col_cfg(self, col_cfg: ColCfg) -> str:
        """Serialize a ColCfg to Python code."""
        from dataclasses import fields, MISSING

        # Get all fields of ColCfg
        cfg_fields = fields(ColCfg)
        args = []

        # Always include datasets_dtype as it's required
        args.append(f'datasets_dtype="{col_cfg.datasets_dtype}"')

        # Include other fields only if they differ from defaults
        for field_info in cfg_fields:
            if field_info.name == "datasets_dtype":
                continue  # Already handled

            value = getattr(col_cfg, field_info.name)

            # Check if this field has a default value
            has_default = field_info.default is not MISSING

            if has_default:
                default_value = field_info.default
                if value != default_value:
                    if isinstance(value, str):
                        args.append(f'{field_info.name}="{value}"')
                    else:
                        args.append(f"{field_info.name}={value!r}")

        return f"ColCfg({', '.join(args)})"

    def save_cache(self, op_definitions: dict[str, OpDef]) -> Path:
        """Save operation definitions to cache."""
        cache_path = self._get_cache_module_path()

        # Clean up old cache files first
        self._cleanup_old_cache_files()

        # Generate module content
        content = self._generate_module_content(op_definitions)

        # Write to cache file
        cache_path.write_text(content)

        return cache_path

    def load_cache(self) -> dict[str, OpDef] | None:
        """Load operation definitions from cache."""
        if not self.is_cache_valid():
            return None

        cache_path = self._get_cache_module_path()

        try:
            # Import the cache module dynamically
            spec = importlib.util.spec_from_file_location("op_definitions_cache", cache_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Verify fingerprint matches
            if hasattr(module, "FINGERPRINT") and module.FINGERPRINT != self.fingerprint:
                rank_zero_warn("Cache fingerprint mismatch, invalidating cache")
                return None

            # Return the operations
            if hasattr(module, "OP_DEFINITIONS"):
                op_definitions = module.OP_DEFINITIONS
                if not op_definitions:
                    rank_zero_warn("No operation definitions found in cache")
                    return None
                return op_definitions

        except Exception as e:
            rank_zero_warn(f"Failed to load cache: {e}")

        return None
