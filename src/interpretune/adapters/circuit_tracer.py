from __future__ import annotations
from typing import Any, Optional, Dict, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch

from circuit_tracer import ReplacementModel, Graph, attribute
from circuit_tracer.utils import create_graph_files
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.adapters import (
    CompositionRegistry,
    LightningDataModule,
    LightningModule,
    LightningAdapter,
    BaseITLensModule,
    TLensAttributeMixin,
)
from interpretune.base import CoreHelperAttributes, ITDataModule, BaseITModule
from interpretune.config import CircuitTracerConfig, ITConfig
from interpretune.utils import move_data_to_device, rank_zero_warn, rank_zero_info
from interpretune.protocol import Adapter


@dataclass(kw_only=True)
class InstantiatedGraph:
    handle: Graph
    graph_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


################################################################################
# Mixins to support Circuit Tracer in different adapter contexts
################################################################################


class CircuitTracerAttributeMixin(TLensAttributeMixin):
    it_cfg: ITConfig

    @property
    def circuit_tracer_cfg(self) -> CircuitTracerConfig | None:
        """Get circuit tracer configuration."""
        if hasattr(self.it_cfg, "circuit_tracer_cfg"):
            return self.it_cfg.circuit_tracer_cfg
        return None

    @property
    def replacement_model(self) -> ReplacementModel | None:
        """Get the replacement model handle."""
        if hasattr(self, "_replacement_model"):
            return self._replacement_model  # type: ignore[attr-defined]  # dynamic mixin attribute
        return None


class BaseCircuitTracerModule(BaseITLensModule):
    def __init__(self, *args, **kwargs):
        # Initialize attributes that may be required in base init methods
        self.attribution_graphs: List[InstantiatedGraph] = []
        self._replacement_model: Optional[ReplacementModel] = None
        super().__init__(*args, **kwargs)

    def _convert_hf_to_tl(self) -> None:
        """Convert HF model to TransformerLens/ReplacementModel."""
        # if datamodule is not attached yet, attempt to retrieve tokenizer handle directly from provided it_cfg
        tokenizer_handle = self.datamodule.tokenizer if self.datamodule else self.it_cfg.tokenizer
        hf_preconversion_config = deepcopy(self.model.config)  # capture original hf config before conversion

        # TODO: we currently prune model_name, dtype and the below values from tl_cfg as CT currently forces these
        #       values in from_pretrained. We prob want a sync method in the future instead.
        #           fold_ln=False,
        #           center_writing_weights=False,
        #           center_unembed=False,
        pruned_tl_cfg = self._prune_tl_cfg_dict(
            ["hf_model", "tokenizer", "model_name", "dtype", "fold_ln", "center_writing_weights", "center_unembed"]
        )
        loaded_model_kwargs = {"hf_model": self.model, "tokenizer": tokenizer_handle, **pruned_tl_cfg}
        self._load_replacement_model(pretrained_kwargs=loaded_model_kwargs)
        self.model.config = hf_preconversion_config

    def _load_replacement_model(self, pretrained_kwargs: Optional[dict] = None) -> None:
        """Load the ReplacementModel for circuit tracing."""
        pretrained_kwargs = pretrained_kwargs or {}
        cfg = self.circuit_tracer_cfg
        if not cfg:
            rank_zero_warn("No circuit_tracer_cfg found, using defaults")
            return

        # Use ReplacementModel.from_pretrained for simplicity
        self._replacement_model = ReplacementModel.from_pretrained(
            model_name=cfg.model_name or self.it_cfg.model_name_or_path,
            transcoder_set=cfg.transcoder_set,
            dtype=cfg.dtype,
            **pretrained_kwargs,
        )

        # Replace the model with the replacement model for circuit tracing
        self.model = self._replacement_model

    def tl_config_model_init(self) -> None:
        """Initialize model using TransformerLens configuration."""
        super().tl_config_model_init()
        if self.circuit_tracer_cfg:
            self._load_replacement_model()

    def _capture_hyperparameters(self) -> None:
        """Capture hyperparameters for logging."""
        self._it_state._init_hparams.update({"circuit_tracer_cfg": deepcopy(self.circuit_tracer_cfg)})
        super()._capture_hyperparameters()

    def set_input_require_grads(self) -> None:
        """Set input gradients for circuit tracing."""
        # Circuit tracer handles gradient requirements internally
        rank_zero_info("Input gradient requirements handled by circuit tracer internally.")

    def _get_attribution_targets(self) -> Optional[list | torch.Tensor]:
        """Determine the attribution_targets value based on CircuitTracerConfig.

        Returns:
            - None: Auto-select salient logits (default behavior)
            - list[str]: Token strings to analyze (will be converted by AttributionTargets)
            - torch.Tensor: Tensor of token IDs
        """
        cfg = self.circuit_tracer_cfg
        if not cfg:
            return None

        # If analysis_target_tokens is set, return as list of strings
        # AttributionTargets will handle tokenization internally
        if cfg.analysis_target_tokens is not None:
            return cfg.analysis_target_tokens

        # If target_token_ids is set, process it
        if cfg.target_token_ids is not None:
            ids = cfg.target_token_ids
            if isinstance(ids, torch.Tensor):
                return ids
            elif isinstance(ids, list):
                return torch.tensor(ids, dtype=torch.long)
            elif isinstance(ids, str):
                # Try to get attribute from self.it_cfg
                attr = getattr(self.it_cfg, ids, None)
                if isinstance(attr, torch.Tensor):
                    return attr
                else:
                    return None
            else:
                return None

        # If neither is set, return None (use salient logits)
        return None

    def generate_attribution_graph(self, prompt: str, **kwargs) -> Graph:
        """Generate attribution graph for a given prompt."""
        if not self.replacement_model:
            raise ValueError("ReplacementModel not loaded. Call _load_replacement_model() first.")

        cfg = self.circuit_tracer_cfg

        # Determine attribution_targets using the new method
        attribution_targets = self._get_attribution_targets()

        # Set default attribution parameters
        attribution_kwargs = {
            "attribution_targets": attribution_targets,
            "max_n_logits": cfg.max_n_logits if cfg else 10,
            "desired_logit_prob": cfg.desired_logit_prob if cfg else 0.95,
            "batch_size": cfg.batch_size if cfg else 256,
            "max_feature_nodes": cfg.max_feature_nodes if cfg else None,
            "offload": cfg.offload if cfg else None,
            "verbose": cfg.verbose if cfg else True,
        }

        # Override with any provided kwargs
        attribution_kwargs.update(kwargs)

        # Generate the attribution graph
        graph = attribute(prompt=prompt, model=self.replacement_model, **attribution_kwargs)

        # Store the graph
        instantiated_graph = InstantiatedGraph(handle=graph, metadata={"prompt": prompt, **attribution_kwargs})
        self.attribution_graphs.append(instantiated_graph)

        return graph

    def save_graph(self, graph: Graph, output_path: Union[str, Path]) -> Path:
        """Save attribution graph to file."""
        output_path = Path(output_path)
        graph.to_pt(str(output_path))
        return output_path

    def create_graph_visualization_files(
        self,
        graph: Graph,
        slug: str,
        output_dir: Union[str, Path],
        node_threshold: float = 0.8,
        edge_threshold: float = 0.98,
    ) -> None:
        """Create graph visualization files for frontend."""
        create_graph_files(
            graph_or_path=graph,
            slug=slug,
            output_path=str(output_dir),
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )


################################################################################
# Circuit Tracer Module Composition
################################################################################


class CircuitTracerAdapter(CircuitTracerAttributeMixin):
    def initialize_graph_output_dir(self, core_log_dir: Path) -> None:
        """Initialize graph_output_dir based on configuration or default to core_log_dir/graph_data."""
        if not self.it_cfg.circuit_tracer_cfg.graph_output_dir:
            self.it_cfg.circuit_tracer_cfg.graph_output_dir = core_log_dir / "graph_data"
            self.it_cfg.circuit_tracer_cfg.graph_output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        adapter_ctx_registry.register(
            Adapter.circuit_tracer,
            component_key="datamodule",
            adapter_combination=(Adapter.core, Adapter.circuit_tracer),  # type: ignore[arg-type]
            composition_classes=(ITDataModule,),
            description="Circuit Tracer adapter that can be composed with core...",
        )
        adapter_ctx_registry.register(
            Adapter.circuit_tracer,
            component_key="datamodule",
            adapter_combination=(Adapter.lightning, Adapter.circuit_tracer),  # type: ignore[arg-type]
            composition_classes=(ITDataModule, LightningDataModule),
            description="Circuit Tracer adapter that can be composed with lightning...",
        )
        adapter_ctx_registry.register(
            Adapter.circuit_tracer,
            component_key="module",
            adapter_combination=(Adapter.core, Adapter.circuit_tracer),  # type: ignore[arg-type]
            composition_classes=(CircuitTracerModule,),
            description="Circuit Tracer adapter that can be composed with core...",
        )
        adapter_ctx_registry.register(
            Adapter.circuit_tracer,
            component_key="module_cfg",
            adapter_combination=(Adapter.core, Adapter.circuit_tracer),  # type: ignore[arg-type]
            composition_classes=(CircuitTracerConfig,),
            description="Circuit Tracer configuration that can be composed with core...",
        )
        adapter_ctx_registry.register(
            Adapter.circuit_tracer,
            component_key="module",
            adapter_combination=(Adapter.lightning, Adapter.circuit_tracer),  # type: ignore[arg-type]
            composition_classes=(
                CircuitTracerAttributeMixin,
                BaseCircuitTracerModule,
                LightningAdapter,
                BaseITModule,
                LightningModule,
            ),
            description="Circuit Tracer adapter that can be composed with lightning...",
        )
        adapter_ctx_registry.register(
            Adapter.circuit_tracer,
            component_key="module_cfg",
            adapter_combination=(Adapter.lightning, Adapter.circuit_tracer),  # type: ignore[arg-type]
            composition_classes=(CircuitTracerConfig,),
            description="Circuit Tracer configuration that can be composed with lightning...",
        )

    def batch_to_device(self, batch) -> BatchEncoding:
        if self.input_device is not None:
            move_data_to_device(batch, self.input_device)
        return batch

    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)  # type: ignore[misc]  # mixin call to super
        self.initialize_graph_output_dir(self.core_log_dir)  # type: ignore[attr-defined]  # mixin provides core_log_dir


class CircuitTracerAnalysisMixin:
    """Mixin for circuit tracer analysis operations."""

    def save_graph(
        self,
        graph: Graph,
        output_path: Union[str, Path],
        slug: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        use_neuronpedia: Optional[bool] = None,
    ) -> Path:
        """Save and optionally transform graph for Neuronpedia upload."""
        # Convert output_path to directory for processing
        if output_path is None:
            output_path = self.it_cfg.circuit_tracer_cfg.graph_output_dir

        # If output_path is a file path, use its parent directory
        if output_path is not None:
            output_dir = Path(output_path).parent if Path(output_path).suffix else Path(output_path)
        else:
            raise ValueError("output_path is None and no default graph_output_dir configured")

        slug = slug or f"graph-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        pt_path = output_dir / f"{slug}.pt"
        # Save graph tensors in .pt format
        graph.to_pt(str(pt_path))

        # Create graph visualization files
        self.create_graph_visualization_files(  # type: ignore[attr-defined]  # mixin provides method
            graph=graph,
            slug=slug,
            output_dir=output_dir,
            node_threshold=self.circuit_tracer_cfg.default_node_threshold if self.circuit_tracer_cfg else 0.8,  # type: ignore[attr-defined]  # mixin provides circuit_tracer_cfg
            edge_threshold=self.circuit_tracer_cfg.default_edge_threshold if self.circuit_tracer_cfg else 0.98,  # type: ignore[attr-defined]  # mixin provides circuit_tracer_cfg
        )
        output_json_path = output_dir / f"{slug}.json"

        # Determine whether to use Neuronpedia
        if use_neuronpedia is None:
            use_neuronpedia = (
                self.it_cfg.circuit_tracer_cfg.use_neuronpedia if self.it_cfg.circuit_tracer_cfg else False  # type: ignore[attr-defined]  # mixin provides it_cfg
            )

        if use_neuronpedia and hasattr(self, "neuronpedia") and self.neuronpedia:  # type: ignore[attr-defined]  # mixin provides neuronpedia
            try:
                transformed_graph, graph_path = self.neuronpedia.transform_circuit_tracer_graph(  # type: ignore[attr-defined]  # mixin provides neuronpedia
                    graph_path=output_json_path, slug=slug, custom_metadata=custom_metadata
                )
                return graph_path
            except Exception as e:
                rank_zero_warn(f"Failed to transform graph for Neuronpedia: {e}")

        return output_json_path

    def generate_graph(
        self,
        prompt: str,
        slug: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        upload_to_np: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        use_neuronpedia: Optional[bool] = None,
        **generation_kwargs,
    ) -> Tuple[Graph, Path, Any]:
        """Generate attribution graph and optionally upload to Neuronpedia."""
        if use_neuronpedia is None:
            use_neuronpedia = (
                self.it_cfg.circuit_tracer_cfg.use_neuronpedia if self.it_cfg.circuit_tracer_cfg else False  # type: ignore[attr-defined]  # mixin provides it_cfg
            )

        if use_neuronpedia:
            if not hasattr(self, "neuronpedia") or not self.neuronpedia:  # type: ignore[attr-defined]  # mixin provides neuronpedia
                raise RuntimeError("Neuronpedia extension not available. Enable it in your configuration.")

        # Generate the attribution graph
        graph = self.generate_attribution_graph(prompt, **generation_kwargs)  # type: ignore[attr-defined]  # mixin provides method

        # Default output_dir to graph_output_dir if not set
        if output_dir is None:
            output_dir = self.it_cfg.circuit_tracer_cfg.graph_output_dir  # type: ignore[attr-defined]  # mixin provides it_cfg
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("output_dir is None and no default graph_output_dir configured")

        graph_slug = slug or f"attribution-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Save and optionally transform for Neuronpedia
        graph_path = self.save_graph(
            graph=graph,
            output_path=output_dir,
            slug=graph_slug,
            custom_metadata=custom_metadata,
            use_neuronpedia=use_neuronpedia,
        )

        if upload_to_np and use_neuronpedia:
            neuronpedia_metadata = self.neuronpedia.upload_graph_to_neuronpedia(graph_path)  # type: ignore[attr-defined]  # mixin provides neuronpedia
        else:
            rank_zero_info("Neuronpedia upload not requested. Set upload_to_np to `True` to automatically upload.")
            neuronpedia_metadata = None

        return graph, graph_path, neuronpedia_metadata


class CircuitTracerModule(
    CircuitTracerAnalysisMixin, CircuitTracerAdapter, CoreHelperAttributes, BaseCircuitTracerModule
): ...
