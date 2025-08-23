from __future__ import annotations
from typing import Any, Optional, Dict, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy
import json
import datetime
import os
import logging
import requests
import re
import random

from jsonschema import ValidationError, validate

from interpretune.config import ITSerializableCfg
from interpretune.utils import (
    rank_zero_warn,
    rank_zero_info,
    _NEURONPEDIA_AVAILABLE,
    RemoveAdditionalPropertiesValidator,
)

from interpretune.analysis import IT_ANALYSIS_CACHE


logger = logging.getLogger(__name__)

if _NEURONPEDIA_AVAILABLE:

    @dataclass(kw_only=True)
    class NeuronpediaConfig(ITSerializableCfg):
        """Configuration for Neuronpedia integration."""

        """Enable Neuronpedia integration."""
        enabled: bool = False
        """Whether to automatically transform Circuit Tracer graphs for Neuronpedia compatibility."""
        auto_transform: bool = True
        """Sample size for json graph validation prior to upload.

        Applied to nodes and links fields. Set to 0 to disable, -1 to validate full graph.
        """
        graph_validation_sample_size: int = 10
        """Whether to create backup of original graph files before transformation."""
        backup_original: bool = False
        """Whether to pretty-print JSON (vs minify) when saving transformed graphs."""
        pretty_print_json: bool = True
        """Whether to dynamically fetch the latest Neuronpedia graph schema."""
        dynamic_np_schema_validation: bool = False
        """Default prefix for graph slugs when not specified."""
        default_slug_prefix: str = "it-generated"
        """Default metadata to add to graphs."""
        default_metadata: Dict[str, Any] = field(
            default_factory=lambda: {
                "info": {
                    "creator_name": "interpretune-user",
                    "creator_url": "https://github.com/speediedan/interpretune",
                    "generator": {
                        "name": "Interpretune Circuit Tracer",
                        "version": "latest",
                        "url": "https://github.com/speediedan/interpretune",
                    },
                },
            }
        )

    class NeuronpediaIntegration:
        """Neuronpedia integration extension for Interpretune.

        This extension handles the complete workflow for integrating Circuit Tracer graphs with Neuronpedia, including
        transformation, validation, and upload capabilities.
        """

        IT_NP_CACHE = os.getenv("IT_NP_CACHE", os.path.join(IT_ANALYSIS_CACHE, "neuronpedia"))
        NP_RAW_REPO_URL_BASE = "https://raw.githubusercontent.com/hijohnnylin/neuronpedia/"
        NP_GRAPH_SCHEMA_PATH = "apps/webapp/app/api/graph/graph-schema.json"
        NP_RELEASES_URL = "https://github.com/hijohnnylin/neuronpedia/releases/latest"

        def __init__(self) -> None:
            super().__init__()
            self.phandle = None
            self._neuronpedia_available = False
            self._np_graph_metadata = None
            self._setup_neuronpedia()

        def _setup_neuronpedia(self) -> None:
            """Setup Neuronpedia imports and check availability."""
            try:
                from neuronpedia.np_graph_metadata import NPGraphMetadata

                self._neuronpedia_available = True
                self._np_graph_metadata = NPGraphMetadata
                logger.info("Neuronpedia package available")
            except ImportError as e:
                rank_zero_warn(f"Neuronpedia package not available: {e}. Install with: pip install neuronpedia")
                self._neuronpedia_available = False

        def connect(self, obj_ref: Any) -> None:
            """Connect to the parent module."""
            self.phandle = obj_ref
            if self.neuronpedia_cfg.enabled and not self._neuronpedia_available:
                rank_zero_warn(
                    "Neuronpedia extension is enabled but neuronpedia package is not available. "
                    "Install with: pip install neuronpedia"
                )

        @property
        def neuronpedia_cfg(self) -> NeuronpediaConfig:
            """Get the Neuronpedia configuration."""
            return self.phandle.it_cfg.neuronpedia_cfg

        def _get_latest_graph_schema(self, schema_path: Union[str, Path] = "graph-schema.json") -> Dict[str, Any]:
            """Fetch the latest Neuronpedia graph schema and return it as a dictionary."""
            if not _NEURONPEDIA_AVAILABLE:
                raise RuntimeError("Neuronpedia package is not available.")

            schema_path = Path(schema_path)
            cache_dir = Path(self.IT_NP_CACHE)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached_schema_path = cache_dir / "graph-schema.json"
            cached_tag_path = cache_dir / "graph-schema.tag"

            fallback_schema_path = Path(__file__).parent / "fallback_neuronpedia_graph_schema.json"

            def get_cached_tag():
                if cached_tag_path.exists():
                    return cached_tag_path.read_text().strip()
                return None

            def cache_schema(schema_bytes, tag):
                with open(cached_schema_path, "wb") as f:
                    f.write(schema_bytes)
                with open(cached_tag_path, "w") as f:
                    f.write(tag)

            if self.neuronpedia_cfg.dynamic_np_schema_validation:
                try:
                    # Fetch latest tag from GitHub
                    resp = requests.get(self.NP_RELEASES_URL, timeout=10)
                    resp.raise_for_status()
                    latest_tag_match = re.search(r'releases/tag/([^"/]+)', resp.url)
                    latest_tag = latest_tag_match.group(1) if latest_tag_match else None
                except Exception as e:
                    rank_zero_warn(f"[NeuronpediaIntegration] Could not fetch latest Neuronpedia tag: {e}")
                    latest_tag = None

                cached_tag = get_cached_tag()

                if latest_tag and latest_tag != cached_tag:
                    # Try to download latest schema
                    url = f"{self.NP_RAW_REPO_URL_BASE}/{latest_tag}/{self.NP_GRAPH_SCHEMA_PATH}"
                    try:
                        schema_resp = requests.get(url, timeout=10)
                        schema_resp.raise_for_status()
                        cache_schema(schema_resp.content, latest_tag)
                        rank_zero_info(f"[NeuronpediaIntegration] Cached latest graph-schema.json (tag: {latest_tag})")
                        return json.loads(schema_resp.content)
                    except Exception as e:
                        rank_zero_warn(f"[NeuronpediaIntegration] Failed to download latest graph-schema.json: {e}")

                # Use cached schema if available
                if cached_schema_path.exists():
                    rank_zero_info(
                        (
                            "[NeuronpediaIntegration] Using cached graph-schema.json "
                            f"(tag: {get_cached_tag() or 'unknown'})"
                        )
                    )
                    return json.loads(cached_schema_path.read_text())

            # Fallback to default schema
            if fallback_schema_path.exists():
                if self.neuronpedia_cfg.dynamic_np_schema_validation:
                    rank_zero_warn("[NeuronpediaIntegration] Falling back to default graph-schema.json")
                return json.loads(fallback_schema_path.read_text())

            raise FileNotFoundError("No graph-schema.json available (failed to fetch, no cache, no fallback)")

        def apply_qparam_transforms(
            self, graph_dict: Dict[str, Any], in_place: bool = True, change_log_path: Optional[Union[str, Path]] = None
        ) -> Tuple[bool, Dict[str, Any]]:
            """Apply transformations to qParams fields based on predefined rules.

            Args:
                graph_dict: The graph dictionary to transform
                in_place: Whether to modify the input dictionary directly
                change_log_path: Path to write change log. If None, auto-generates based on slug

            Returns:
                Tuple of (was_valid_before_transforms, transformed_graph_dict)
            """
            was_valid = True
            log = []

            if not in_place:
                graph_dict = deepcopy(graph_dict)

            # Ensure qParams exists
            if "qParams" not in graph_dict or not isinstance(graph_dict["qParams"], dict):
                log.append({"field": "qParams", "before": graph_dict.get("qParams", None), "after": {}})
                graph_dict["qParams"] = {}
                was_valid = False

            qp = graph_dict["qParams"]

            # Define transformation rules
            def csv_str_to_list(value):
                return [x.strip() for x in value.split(",") if x.strip()]

            field_definitions = [
                {"field": "pinnedIds", "type": list, "default": [], "transform": csv_str_to_list},
                {"field": "clerps", "type": list, "default": [], "transform": csv_str_to_list},
                {"field": "supernodes", "type": list, "default": [], "transform": None},
                {"field": "linkType", "type": str, "default": "both", "transform": None},
                {"field": "clickedId", "type": str, "default": "", "transform": None},
                {"field": "sg_pos", "type": str, "default": "", "transform": None},
            ]

            # Apply transformations
            for field_def in field_definitions:
                field = field_def["field"]
                expected_type = field_def["type"]
                default_value = field_def["default"]
                transform = field_def["transform"]

                if field not in qp:
                    log.append({"field": field, "before": None, "after": default_value})
                    qp[field] = default_value
                    was_valid = False
                elif not isinstance(qp[field], expected_type):
                    if transform is not None:
                        transformed_value = transform(qp[field])
                        log.append({"field": field, "before": qp[field], "after": transformed_value})
                        qp[field] = transformed_value
                        was_valid = False

            # Log and write changes
            if log:
                self._log_qparam_changes(log, graph_dict, change_log_path)

            return was_valid, graph_dict

        def _log_qparam_changes(
            self, log: List[Dict], graph_dict: Dict[str, Any], change_log_path: Optional[Union[str, Path]]
        ) -> None:
            """Log qParams changes to console and file."""
            # Console logging
            for entry in log:
                rank_zero_info(
                    f"[NeuronpediaIntegration] Field '{entry['field']}' changed: "
                    f"before={entry['before']!r}, after={entry['after']!r}"
                )

            # File logging
            if change_log_path is None:
                graph_path = graph_dict.get("graph_path")
                if graph_path:
                    log_dir = Path(graph_path).parent
                else:
                    log_dir = Path(self.phandle.core_log_dir)
                slug = graph_dict.get("metadata", {}).get("slug") or graph_dict.get("slug") or "unknown-slug"
                dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                change_log_path = log_dir / f"{slug}_qparam_changes_{dt}.log"

            try:
                with open(change_log_path, "a") as f:
                    f.write(f"=== qParams changes logged at {datetime.datetime.now().isoformat()} ===\n")
                    for entry in log:
                        f.write(
                            f"Field '{entry['field']}' changed: before={entry['before']!r}, after={entry['after']!r}\n"
                        )
                    f.write("\n")
                rank_zero_info(f"[NeuronpediaIntegration] Changes written to {change_log_path}")
            except Exception as e:
                rank_zero_warn(f"[NeuronpediaIntegration] Failed to write change log: {e}")

        def prune_unsupported_metadata(self, graph_dict: Dict[str, Any]) -> None:
            """Prune metadata fields in graph_dict that do not conform to the graph schema."""
            np_graph_schema = self._get_latest_graph_schema()
            metadata_schema = np_graph_schema.get("properties", {}).get("metadata", {})

            assert graph_dict.get("metadata") is not None, "Graph dictionary must contain 'metadata' key."
            try:
                RemoveAdditionalPropertiesValidator(schema=metadata_schema).validate(instance=graph_dict["metadata"])
            except ValidationError as e:
                # Remove invalid keys based on the error path
                invalid_keys = [error.path[0] for error in e.context]
                for key in invalid_keys:
                    graph_dict["metadata"].pop(key, None)

        def prepare_graph_metadata(
            self,
            graph_dict: Dict[str, Any],
            slug: Optional[str] = None,
            custom_metadata: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """Prepare and enrich graph metadata for Neuronpedia.

            Args:
                graph_dict: The graph dictionary to prepare
                slug: Custom slug for the graph. If None, uses existing or generates one
                custom_metadata: Additional metadata to merge

            Returns:
                The graph dictionary with enriched metadata
            """
            # Ensure metadata section exists
            if "metadata" not in graph_dict:
                graph_dict["metadata"] = {}

            metadata = graph_dict["metadata"]

            # Prune unsupported metadata fields
            self.prune_unsupported_metadata(graph_dict)

            # Set slug
            if slug:
                metadata["slug"] = slug
            elif "slug" not in metadata:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                metadata["slug"] = f"{self.neuronpedia_cfg.default_slug_prefix}-{timestamp}"

            # Add create_time_ms if not present
            if "create_time_ms" not in metadata:
                metadata["create_time_ms"] = int(datetime.datetime.now().timestamp() * 1000)

            # Merge default metadata
            default_meta = deepcopy(self.neuronpedia_cfg.default_metadata)
            for key, value in default_meta.items():
                if key not in metadata:
                    metadata[key] = value
                elif isinstance(value, dict) and isinstance(metadata[key], dict):
                    # Deep merge for nested dictionaries
                    for sub_key, sub_value in value.items():
                        if sub_key not in metadata[key]:
                            metadata[key][sub_key] = sub_value

            # Merge custom metadata
            if custom_metadata:
                for key, value in custom_metadata.items():
                    if isinstance(value, dict) and key in metadata and isinstance(metadata[key], dict):
                        metadata[key].update(value)
                    else:
                        metadata[key] = value

            # TODO: inspect supported CT generation config metadata and add it saved NP metadata
            # (e.g. node_threshold, max_n_logits, max_feature_nodes, etc.)
            return graph_dict

        def transform_circuit_tracer_graph(
            self,
            graph_path: Union[str, Path],
            output_path: Optional[Union[str, Path]] = None,
            slug: Optional[str] = None,
            custom_metadata: Optional[Dict[str, Any]] = None,
        ) -> Tuple[Dict[str, Any], Path]:
            """Transform a Circuit Tracer graph for Neuronpedia compatibility.

            Args:
                graph_path: Path to the Circuit Tracer graph JSON file
                output_path: Path to save the transformed graph. If None, overwrites original
                slug: Custom slug for the graph
                custom_metadata: Additional metadata to add

            Returns:
                Tuple of (transformed_graph_dict, output_path)
            """
            graph_path = Path(graph_path)

            if not graph_path.exists():
                raise FileNotFoundError(f"Graph file not found: {graph_path}")

            # Load the original graph
            with open(graph_path, "r") as f:
                graph_dict = json.load(f)

            rank_zero_info(f"[NeuronpediaIntegration] Transforming graph: {graph_path}")

            # Create backup if requested
            if self.neuronpedia_cfg.backup_original:
                backup_path = graph_path.with_suffix(".backup" + graph_path.suffix)
                if not backup_path.exists():  # Don't overwrite existing backups
                    import shutil

                    shutil.copy2(graph_path, backup_path)
                    rank_zero_info(f"[NeuronpediaIntegration] Created backup: {backup_path}")

            # Prepare metadata
            graph_dict = self.prepare_graph_metadata(graph_dict, slug, custom_metadata)

            was_valid, graph_dict = self.apply_qparam_transforms(graph_dict, in_place=True)
            if not was_valid:
                rank_zero_info("[NeuronpediaIntegration] Transformed qParams for Neuronpedia")

            # Determine output path
            output_path = graph_path if output_path is None else Path(output_path)

            # Save the transformed graph
            self._save_graph_json(graph_dict, output_path)

            rank_zero_info(f"[NeuronpediaIntegration] Transformed graph saved to: {output_path}")

            return graph_dict, output_path

        def _save_graph_json(self, graph_dict: Dict[str, Any], output_path: Path) -> None:
            """Save graph dictionary as JSON."""
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if self.neuronpedia_cfg.pretty_print_json:
                json_str = json.dumps(graph_dict, indent=2, sort_keys=False)
            else:
                json_str = json.dumps(graph_dict)

            with open(output_path, "w") as f:
                f.write(json_str)

        def validate_graph(self, graph_dict: Dict[str, Any]) -> bool:
            """Validate the provided graph dictionary against the Neuronpedia schema.

            Args:
                graph_dict: The graph dictionary to validate

            Returns:
                True if the graph is valid, False otherwise
            """
            np_graph_schema = self._get_latest_graph_schema()

            sample_size = self.neuronpedia_cfg.graph_validation_sample_size

            if sample_size == 0:
                # Bypass validation
                return True

            if sample_size > 0:
                # Sample nodes and links randomly
                graph_dict = deepcopy(graph_dict)
                nodes = graph_dict.get("nodes", [])
                links = graph_dict.get("links", [])

                if sample_size < len(nodes):
                    graph_dict["nodes"] = random.sample(nodes, sample_size)
                if sample_size < len(links):
                    graph_dict["links"] = random.sample(links, sample_size)

            try:
                validate(instance=graph_dict, schema=np_graph_schema)
                return True
            except ValidationError as e:
                rank_zero_warn(f"[NeuronpediaIntegration] Graph validation failed: {e.message}")
                return False

        def upload_graph_to_neuronpedia(self, graph_path: Union[str, Path], api_key: Optional[str] = None) -> Any:
            """Upload a graph to Neuronpedia.

            Args:
                graph_path: Path to graph JSON file or graph dictionary
                api_key: Override API key. If None, uses configured key or environment

            Returns:
                NPGraphMetadata object from successful upload
            """
            if not self._neuronpedia_available:
                raise RuntimeError("Neuronpedia package not available. Install with: pip install neuronpedia")
            # TODO: consider adding support for graph_dict input directly in the future
            # Handle graph input
            graph_path = Path(graph_path)
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph file not found: {graph_path}")
            with open(graph_path, "r") as f:
                graph_dict = json.load(f)

            # Validate graph before upload
            if not self.validate_graph(graph_dict):
                rank_zero_warn("[NeuronpediaIntegration] Graph validation failed. Upload aborted.")
                return None

            # Determine API key
            if api_key is None:
                use_localhost = os.environ.get("USE_LOCALHOST", "false").lower() == "true"
                api_key = (
                    os.environ.get("DEV_NEURONPEDIA_API_KEY")
                    if use_localhost
                    else os.environ.get("NEURONPEDIA_API_KEY")
                )

                if not api_key:
                    raise ValueError("API key not found. Set NEURONPEDIA_API_KEY env var.")

            # Upload with API key context
            import neuronpedia

            rank_zero_info(f"[NeuronpediaIntegration] Uploading graph: {graph_path}")

            api_type = "dev" if use_localhost else "production"
            rank_zero_info(f"[NeuronpediaIntegration] Using {api_type} API")
            if api_type == "dev":
                rank_zero_warn(
                    "[NeuronpediaIntegration] Uploading using a development API key. Ensure this is intended."
                )

            with neuronpedia.api_key(api_key):
                try:
                    graph_metadata = self._np_graph_metadata.upload_file(str(graph_path))
                    rank_zero_info("[NeuronpediaIntegration] Upload successful!")
                    if hasattr(graph_metadata, "url"):
                        rank_zero_info(f"[NeuronpediaIntegration] Graph URL: {graph_metadata.url}")
                    return graph_metadata
                except Exception as e:
                    rank_zero_warn(f"[NeuronpediaIntegration] Upload failed: {e}")
                    raise

        def transform_graph_for_np(
            self,
            graph_path: Union[str, Path],
            slug: Optional[str] = None,
            upload_to_np: bool = False,
            custom_metadata: Optional[Dict[str, Any]] = None,
            api_key: Optional[str] = None,
        ) -> Tuple[Dict[str, Any], Any]:
            """Transform and upload a Circuit Tracer graph to Neuronpedia.

            Args:
                graph_path: Path to the Circuit Tracer graph JSON file
                slug: Custom slug for the graph
                custom_metadata: Additional metadata to add
                api_key: Override API key

            Returns:
                Tuple of (transformed_graph_dict, NPGraphMetadata)
            """
            if not self.neuronpedia_cfg.enabled:
                raise RuntimeError("Neuronpedia integration is not enabled")

            # Transform the graph
            transformed_graph, output_path = self.transform_circuit_tracer_graph(
                graph_path=graph_path, slug=slug, custom_metadata=custom_metadata
            )

            # Upload if enabled
            if upload_to_np:
                graph_metadata = self.upload_graph_to_neuronpedia(output_path, api_key)
                return transformed_graph, graph_metadata
            else:
                rank_zero_info(
                    "[NeuronpediaIntegration] Neuronpedia upload not requested."
                    "Set `upload_to_np` to `True` to automatically upload."
                )
                return transformed_graph, None
else:
    NeuronpediaConfig = object  # type: ignore[assignment]
    NeuronpediaIntegration = object  # type: ignore[assignment]
