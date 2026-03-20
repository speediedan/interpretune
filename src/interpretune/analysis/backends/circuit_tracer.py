"""Circuit-tracer analysis backend helpers.

This module owns circuit-tracer specific graph hydration/decomposition and concept direction helpers so native op
definitions can stay thin and delegate backend-specific logic to a dedicated analysis backend.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import torch
from transformers import BatchEncoding

from interpretune.analysis.backends import AnalysisBackendCapability
from interpretune.analysis.ops.base import AnalysisBatch, get_batch_input
from interpretune.protocol import GraphComponentPayload


class CircuitTracerAnalysisBackend:
    """Analysis backend for circuit-tracer graph generation and hydration."""

    @property
    def capabilities(self) -> frozenset[AnalysisBackendCapability]:
        return frozenset({AnalysisBackendCapability.ATTRIBUTION_GRAPH, AnalysisBackendCapability.FEATURE_INTERVENTION})

    def supports(self, capability: AnalysisBackendCapability) -> bool:
        return capability in self.capabilities

    def get_tokenizer(self, module: Any) -> Any:
        for attr_name in ("replacement_model", "model"):
            model = getattr(module, attr_name, None)
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is not None:
                return tokenizer
        datamodule = getattr(module, "datamodule", None)
        tokenizer = getattr(datamodule, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        raise ValueError("A tokenizer is required for this analysis operation")

    def get_embedding_weight(self, module: Any) -> torch.Tensor:
        for attr_name in ("replacement_model", "model"):
            model = getattr(module, attr_name, None)
            unembed_weight = getattr(model, "unembed_weight", None)
            if isinstance(unembed_weight, torch.Tensor):
                return unembed_weight
            embed_weight = getattr(model, "embed_weight", None)
            if isinstance(embed_weight, torch.Tensor):
                return embed_weight
            get_input_embeddings = getattr(model, "get_input_embeddings", None)
            if callable(get_input_embeddings):
                embedding_layer = get_input_embeddings()
                weight = getattr(embedding_layer, "weight", None)
                if isinstance(weight, torch.Tensor):
                    return weight
        raise ValueError("An embedding weight matrix is required for concept_direction")

    def flatten_token_ids(self, tokenized: Any) -> list[int]:
        if isinstance(tokenized, torch.Tensor):
            if tokenized.dim() == 0:
                return [int(tokenized.item())]
            return [int(value) for value in tokenized.reshape(-1).tolist()]
        if hasattr(tokenized, "tolist"):
            values = tokenized.tolist()
            if isinstance(values, list) and values and isinstance(values[0], list):
                return [int(value) for sublist in values for value in sublist]
            if isinstance(values, list):
                return [int(value) for value in values]
        if isinstance(tokenized, list):
            if tokenized and isinstance(tokenized[0], list):
                return [int(value) for sublist in tokenized for value in sublist]
            return [int(value) for value in tokenized]
        return [int(tokenized)]

    def token_strings_to_ids(self, tokenizer: Any, token_strings: list[str]) -> list[int]:
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        token_ids: list[int] = []
        for token_str in token_strings:
            if token_str in vocab:
                token_ids.append(int(vocab[token_str]))
                continue
            tokenized = tokenizer(token_str, add_special_tokens=False)["input_ids"]
            token_ids.extend(self.flatten_token_ids(tokenized))
        if not token_ids:
            raise ValueError("Unable to resolve any token ids for the provided concept groups")
        return token_ids

    def resolve_prompt(self, module: Any, analysis_batch: AnalysisBatch, batch: BatchEncoding | None) -> str:
        prompts = getattr(analysis_batch, "prompts", None)
        if isinstance(prompts, str):
            return prompts
        if isinstance(prompts, list) and prompts:
            return str(prompts[0])
        if batch is None:
            raise ValueError("compute_attribution_graph requires batch input or prompts in analysis_batch")
        tokenizer = self.get_tokenizer(module)
        input_ids = get_batch_input(batch)
        prompt_tokens = input_ids[0] if input_ids.dim() > 1 else input_ids
        return str(tokenizer.decode(prompt_tokens.detach().cpu().tolist(), skip_special_tokens=True))

    def build_concept_attribution_targets(
        self,
        module: Any,
        prompt: str,
        concept_direction: Any,
        concept_label: Any,
        *,
        concept_group_a_token_ids: Any = None,
        concept_group_b_token_ids: Any = None,
        concept_direction_mode: Any = None,
    ) -> list[Any] | None:
        from circuit_tracer.attribution.targets import CustomTarget

        group_a_token_ids = [int(token_id) for token_id in (concept_group_a_token_ids or [])]
        if group_a_token_ids:
            concept_logits = module.replacement_model.get_activations(prompt)[0]
            concept_probs = torch.softmax(concept_logits.squeeze(0)[-1].float(), dim=-1)
            concept_prob = max(
                sum(concept_probs[token_id].item() for token_id in group_a_token_ids) / len(group_a_token_ids),
                1e-6,
            )
        else:
            concept_prob = 1e-6
        concept_direction_tensor = torch.as_tensor(concept_direction, dtype=torch.float32).to(
            self.get_embedding_weight(module).device
        )
        return [
            CustomTarget(
                token_str=str(concept_label or concept_direction_mode or "concept_direction"),
                prob=float(concept_prob),
                vec=concept_direction_tensor,
            )
        ]

    def resolve_feature_intervention_settings(
        self,
        module: Any,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = getattr(module, "circuit_tracer_cfg", None)
        override_dict = dict(overrides or {})

        def _resolve(name: str, default: Any) -> Any:
            if name in override_dict:
                return override_dict[name]
            return getattr(cfg, name, default)

        value = _resolve("intervention_value", None)
        value_source = _resolve("intervention_value_source", "top_feature_scores")
        if value is not None:
            value_source = "constant"

        settings = {
            "scale_factor": float(_resolve("intervention_scale_factor", 1.0)),
            "value": None if value is None else float(value),
            "value_source": str(value_source),
            "constrained_layers": _resolve("intervention_constrained_layers", None),
            "freeze_attention": _resolve("intervention_freeze_attention", None),
            "apply_activation_function": _resolve("intervention_apply_activation_function", None),
            "sparse": bool(_resolve("intervention_sparse", False)),
            "return_activations": bool(_resolve("intervention_return_activations", False)),
        }

        if settings["value_source"] not in {"top_feature_scores", "top_feature_activation_values", "constant"}:
            raise ValueError(
                "feature_intervention_forward only supports value_source values "
                "'top_feature_scores', 'top_feature_activation_values', and 'constant'"
            )
        if settings["value_source"] == "constant" and settings["value"] is None:
            raise ValueError("feature_intervention_forward requires a constant intervention_value for constant mode")
        return settings

    def build_feature_interventions(
        self,
        analysis_batch: AnalysisBatch | Mapping[str, Any],
        settings: Mapping[str, Any],
    ) -> tuple[list[tuple[int, int, int, float]], dict[str, Any]]:
        def _value(name: str, default: Any) -> Any:
            if isinstance(analysis_batch, Mapping):
                return analysis_batch.get(name, default)
            return getattr(analysis_batch, name, default)

        top_feature_ids = torch.as_tensor(_value("top_feature_ids", []), dtype=torch.long).reshape(-1, 3)
        top_feature_scores = _value("top_feature_scores", None)
        top_feature_activation_values = _value("top_feature_activation_values", None)
        if top_feature_scores is None and settings["value_source"] == "top_feature_scores":
            raise ValueError("feature_intervention_forward requires top_feature_scores when using score-derived values")
        if top_feature_activation_values is None and settings["value_source"] == "top_feature_activation_values":
            raise ValueError(
                "feature_intervention_forward requires top_feature_activation_values "
                "when using activation-derived values"
            )

        score_tensor = (
            torch.as_tensor(top_feature_scores, dtype=torch.float32).reshape(-1)
            if top_feature_scores is not None
            else None
        )
        activation_value_tensor = (
            torch.as_tensor(top_feature_activation_values, dtype=torch.float32).reshape(-1)
            if top_feature_activation_values is not None
            else None
        )
        if score_tensor is not None and score_tensor.shape[0] != top_feature_ids.shape[0]:
            raise ValueError("top_feature_ids and top_feature_scores must have matching lengths")
        if activation_value_tensor is not None and activation_value_tensor.shape[0] != top_feature_ids.shape[0]:
            raise ValueError("top_feature_ids and top_feature_activation_values must have matching lengths")

        intervention_specs: list[dict[str, Any]] = []
        interventions: list[tuple[int, int, int, float]] = []
        for index, feature_row in enumerate(top_feature_ids.tolist()):
            layer, position, feature_id = (int(value) for value in feature_row)
            if settings["value"] is not None:
                base_value = settings["value"]
            elif settings["value_source"] == "top_feature_activation_values":
                if activation_value_tensor is None:
                    raise ValueError("Unable to resolve activation-derived intervention value for feature row")
                base_value = float(activation_value_tensor[index].item())
            elif score_tensor is not None:
                base_value = float(score_tensor[index].item())
            else:
                raise ValueError("Unable to resolve intervention value for feature row")
            value = float(base_value * settings["scale_factor"])
            interventions.append((layer, position, feature_id, value))
            intervention_specs.append({"layer": layer, "position": position, "feature_id": feature_id, "value": value})

        config_payload = {
            "scale_factor": settings["scale_factor"],
            "value": settings["value"],
            "value_source": settings["value_source"],
            "constrained_layers": settings["constrained_layers"],
            "freeze_attention": settings["freeze_attention"],
            "apply_activation_function": settings["apply_activation_function"],
            "sparse": settings["sparse"],
            "return_activations": settings["return_activations"],
        }
        payload = {
            "intervention_config": json.dumps(config_payload, default=str),
            "intervention_specs_json": json.dumps(intervention_specs, default=str),
            "intervention_layers": [spec["layer"] for spec in intervention_specs],
            "intervention_positions": [spec["position"] for spec in intervention_specs],
            "intervention_feature_ids": [spec["feature_id"] for spec in intervention_specs],
            "intervention_values": [spec["value"] for spec in intervention_specs],
        }
        return interventions, payload

    def feature_intervention_call_kwargs(self, settings: Mapping[str, Any]) -> dict[str, Any]:
        kwargs = {
            "sparse": settings["sparse"],
            "return_activations": settings["return_activations"],
        }
        if settings["constrained_layers"] is not None:
            kwargs["constrained_layers"] = settings["constrained_layers"]
        if settings["freeze_attention"] is not None:
            kwargs["freeze_attention"] = settings["freeze_attention"]
        if settings["apply_activation_function"] is not None:
            kwargs["apply_activation_function"] = settings["apply_activation_function"]
        return kwargs

    def _hydrate_intervention_specs(self, specs_json: str | None) -> list[tuple[int, int, int, float]] | None:
        if not specs_json:
            return None
        if isinstance(specs_json, list):
            specs_json = "".join(str(part) for part in specs_json)
        specs = json.loads(specs_json)
        return [
            (int(spec["layer"]), int(spec["position"]), int(spec["feature_id"]), float(spec["value"])) for spec in specs
        ]

    def graph_cfg_dict(self, graph: Any) -> dict[str, Any]:
        return graph.cfg.to_dict() if hasattr(graph.cfg, "to_dict") else vars(graph.cfg)

    def graph_scan_json(self, graph: Any) -> str:
        return json.dumps(graph.scan, default=str)

    def select_feature_rows(self, active_features: torch.Tensor, selected_features: torch.Tensor) -> torch.Tensor:
        if len(selected_features) == 0:
            return torch.empty((0, 3), dtype=torch.long)
        selected = selected_features.long().detach().cpu()
        return active_features.detach().cpu().index_select(0, selected)

    def graph_metadata(self, graph: Any, extra: dict[str, Any] | None = None) -> str:
        metadata = {
            "input_string": graph.input_string,
            "scan": graph.scan,
            "vocab_size": graph.vocab_size,
            "cfg": self.graph_cfg_dict(graph),
        }
        if extra:
            metadata.update(extra)
        return json.dumps(metadata, default=str)

    def decompose_graph(self, graph: Any, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "input_string": graph.input_string,
            "adjacency_matrix": graph.adjacency_matrix.detach().cpu(),
            "active_features": graph.active_features.detach().cpu(),
            "selected_features": graph.selected_features.detach().cpu(),
            "activation_values": graph.activation_values.detach().cpu(),
            "logit_target_ids": graph.logit_token_ids.detach().cpu(),
            "logit_target_tokens": [target.token_str for target in graph.logit_targets],
            "logit_probabilities": graph.logit_probabilities.detach().cpu(),
            "input_tokens": graph.input_tokens.detach().cpu(),
            "graph_cfg_json": json.dumps(self.graph_cfg_dict(graph), default=str),
            "graph_scan_json": self.graph_scan_json(graph),
            "graph_vocab_size": int(graph.vocab_size),
            "graph_metadata": self.graph_metadata(graph, extra_metadata),
        }

    def graph_components_from_batch(self, analysis_batch: AnalysisBatch | Mapping[str, Any]) -> GraphComponentPayload:
        graph_cfg_json = getattr(analysis_batch, "graph_cfg_json", None)
        if graph_cfg_json is None and isinstance(analysis_batch, Mapping):
            graph_cfg_json = analysis_batch.get("graph_cfg_json")
        if graph_cfg_json is None:
            raise ValueError("Graph rehydration requires graph_cfg_json")

        graph_scan_json = getattr(analysis_batch, "graph_scan_json", "null")
        if isinstance(analysis_batch, Mapping):
            graph_scan_json = analysis_batch.get("graph_scan_json", graph_scan_json)
        logit_target_tokens = list(getattr(analysis_batch, "logit_target_tokens", []))
        if isinstance(analysis_batch, Mapping):
            logit_target_tokens = list(analysis_batch.get("logit_target_tokens", logit_target_tokens))

        def _value(name: str, default: Any) -> Any:
            if isinstance(analysis_batch, Mapping):
                return analysis_batch.get(name, default)
            return getattr(analysis_batch, name, default)

        return {
            "input_string": str(_value("input_string", "")),
            "input_tokens": torch.as_tensor(_value("input_tokens", []), dtype=torch.long),
            "active_features": torch.as_tensor(_value("active_features", []), dtype=torch.long).reshape(-1, 3),
            "adjacency_matrix": torch.as_tensor(_value("adjacency_matrix", []), dtype=torch.float32),
            "selected_features": torch.as_tensor(_value("selected_features", []), dtype=torch.long),
            "activation_values": torch.as_tensor(_value("activation_values", []), dtype=torch.float32),
            "logit_target_ids": torch.as_tensor(_value("logit_target_ids", []), dtype=torch.long),
            "logit_target_tokens": logit_target_tokens,
            "logit_probabilities": torch.as_tensor(_value("logit_probabilities", []), dtype=torch.float32),
            "graph_cfg": json.loads(graph_cfg_json),
            "scan": json.loads(graph_scan_json),
            "vocab_size": int(_value("graph_vocab_size", 0)),
        }

    def hydrate_graph(self, components: GraphComponentPayload):
        from circuit_tracer.attribution.targets import LogitTarget
        from circuit_tracer.graph import Graph
        from circuit_tracer.utils.tl_nnsight_mapping import UnifiedConfig

        logit_targets = [
            LogitTarget(token_str=token_str, vocab_idx=int(token_id))
            for token_str, token_id in zip(components["logit_target_tokens"], components["logit_target_ids"].tolist())
        ]
        cfg = UnifiedConfig.from_dict(dict(components["graph_cfg"]))
        return Graph(
            input_string=components["input_string"],
            input_tokens=components["input_tokens"],
            active_features=components["active_features"],
            adjacency_matrix=components["adjacency_matrix"],
            cfg=cfg,
            selected_features=components["selected_features"],
            activation_values=components["activation_values"],
            logit_targets=logit_targets,
            logit_probabilities=components["logit_probabilities"],
            scan=components["scan"],
            vocab_size=components["vocab_size"],
        )

    def hydrate_graph_from_batch(self, analysis_batch: AnalysisBatch | Mapping[str, Any]):
        return self.hydrate_graph(self.graph_components_from_batch(analysis_batch))

    def maybe_hydrate_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        hydrated = dict(row)
        if "graph_cfg_json" in hydrated and "adjacency_matrix" in hydrated and "attribution_graph" not in hydrated:
            hydrated["attribution_graph"] = self.hydrate_graph_from_batch(hydrated)
        if "intervention_specs_json" in hydrated and "intervention_specs" not in hydrated:
            hydrated["intervention_specs"] = self._hydrate_intervention_specs(hydrated.get("intervention_specs_json"))
        return hydrated

    def maybe_hydrate_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        hydrated = dict(batch)
        required_keys = {"graph_cfg_json", "adjacency_matrix"}
        if required_keys.issubset(hydrated.keys()) and "attribution_graph" not in hydrated:
            row_count = len(hydrated["graph_cfg_json"])
            keys = [key for key, values in hydrated.items() if hasattr(values, "__len__") and len(values) == row_count]
            hydrated["attribution_graph"] = [
                self.hydrate_graph_from_batch({key: hydrated[key][index] for key in keys}) for index in range(row_count)
            ]
        if "intervention_specs_json" in hydrated and "intervention_specs" not in hydrated:
            hydrated["intervention_specs"] = [
                self._hydrate_intervention_specs(specs_json) for specs_json in hydrated["intervention_specs_json"]
            ]
        return hydrated

    def build_pruned_graph(self, graph: Any, node_threshold: float, edge_threshold: float):
        from circuit_tracer.graph import Graph, prune_graph

        prune_result = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
        n_features = len(graph.selected_features)
        kept_feature_nodes = prune_result.node_mask[:n_features].nonzero(as_tuple=False).squeeze(-1)
        rest_indices = torch.arange(n_features, graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device)
        kept_indices = torch.cat((kept_feature_nodes.to(graph.adjacency_matrix.device), rest_indices), dim=0)
        pruned_adjacency = graph.adjacency_matrix.index_select(0, kept_indices).index_select(1, kept_indices).clone()
        edge_mask = prune_result.edge_mask.index_select(0, kept_indices).index_select(1, kept_indices)
        pruned_adjacency *= edge_mask.to(pruned_adjacency.dtype)

        kept_feature_count = int(kept_feature_nodes.numel())
        rest_mask = prune_result.node_mask[n_features:].to(pruned_adjacency.device)
        retained_node_mask = torch.cat(
            (torch.ones(kept_feature_count, dtype=torch.bool, device=pruned_adjacency.device), rest_mask),
            dim=0,
        )
        pruned_adjacency[~retained_node_mask] = 0
        pruned_adjacency[:, ~retained_node_mask] = 0

        return Graph(
            input_string=graph.input_string,
            input_tokens=graph.input_tokens,
            active_features=graph.active_features,
            adjacency_matrix=pruned_adjacency,
            cfg=graph.cfg,
            selected_features=graph.selected_features.index_select(0, kept_feature_nodes.cpu()),
            activation_values=graph.activation_values.index_select(0, kept_feature_nodes.cpu()),
            logit_targets=graph.logit_targets,
            logit_probabilities=graph.logit_probabilities,
            scan=graph.scan,
            vocab_size=graph.vocab_size,
        )

    def compute_node_influence_scores(self, graph: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute node influence scores and the corresponding feature rows for a graph."""

        from circuit_tracer.graph import compute_node_influence

        n_logits = len(graph.logit_targets)
        n_features = len(graph.selected_features)
        logit_weights = torch.zeros(graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device)
        logit_weights[-n_logits:] = graph.logit_probabilities
        node_scores = compute_node_influence(graph.adjacency_matrix, logit_weights)[:n_features]
        node_feature_ids = self.select_feature_rows(graph.active_features, graph.selected_features)
        return node_scores.detach().cpu(), node_feature_ids


DEFAULT_CT_ANALYSIS_BACKEND = CircuitTracerAnalysisBackend()


__all__ = ["CircuitTracerAnalysisBackend", "DEFAULT_CT_ANALYSIS_BACKEND"]
