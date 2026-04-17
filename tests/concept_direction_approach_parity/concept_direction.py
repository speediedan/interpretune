from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import torch

import interpretune as it
import interpretune.analysis
from interpretune.analysis.execution import execute_analysis_op
from interpretune.analysis.ops.helpers import (
    AnalysisInputs,
    FeatureSelectionSpec,
    last_token_logits,
)
from interpretune.config import AnalysisCfg, init_analysis_cfgs
from interpretune.utils import (
    DEFAULT_COPILOT_MAX_RETRIES,
    DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS,
    DEFAULT_COPILOT_TIMEOUT_SECONDS,
    DEFAULT_EXPLANATION_TYPE_NAME,
    DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL,
    DEFAULT_NEURONPEDIA_BASE_URL,
    LocalNeuronpediaServiceStatus,
    check_local_neuronpedia_services,
    check_local_explanation_coverage,
    feature_tuples_to_feature_refs,
    default_np_cache_dir,
    parse_feature_url,
)
from interpretune.utils.neuronpedia_explanations import (
    NeuronpediaFeatureRef,
    NeuronpediaLocalExplanationStatus,
    candidate_cached_activation_batch_paths,
    cached_feature_activation_path,
    load_activation_batch_records,
    load_cached_feature_activations,
    write_cached_feature_activations,
)

from it_examples.example_prompt_configs import GemmaPromptConfig
from tests.nb_experiment_harness.config import (
    SharedHarnessSections,
    build_shared_harness_sections,
    get_config_value,
    get_required_config_value,
    load_experiment_config,
)
from tests.nb_experiment_harness.nb_harness_utils import (
    _build_graph_analysis_inputs as _shared_build_graph_analysis_inputs,
    _build_feature_selection_spec as _shared_build_feature_selection_spec,
    _get_config_value_with_preset_default,
    _extract_top_features_with_optional_filter as _shared_extract_top_features_with_optional_filter,
    build_shared_summary_record,
    create_work_root,
    tensor_to_cpu,
)
from tests.nb_experiment_harness.pipeline_patterns import (
    collect_feature_pool as _shared_collect_feature_pool,
    run_debug_intervention_validation as _shared_run_debug_intervention_validation,
    run_pipeline as _shared_run_pipeline,
    run_direct_projection_pipeline as _shared_run_direct_projection_pipeline,
    run_scale_sweep as _shared_run_scale_sweep,
)
from tests.parity_analysis.intervention_drift_analysis import (
    resolve_artifact_output_dir,
    save_preserved_intervention_artifacts,
    tensor_fingerprint,
)
from tests.nb_experiment_harness.session import (
    experiment_session,
    resolve_model_spec,
    resolve_session_surface_preset_config_defaults,
)

if TYPE_CHECKING:
    from interpretune.extensions.debug_generation import DebugGeneration


PromptRenderMode = Literal["plain", "apply_chat_template", "gemma_dataclass"]
StoreLatentExtractionMode = Literal["answer_position_state", "context_enhanced"]
ConstrainedFeatureSelectionRefValue = str | tuple[str, str, int, int]
AnalysisMode = Literal["concept_pair", "explicit_embedding_difference", "debug_intervention_pipelines"]
ConceptDirectionMode = Literal["mean_difference", "paired_rejection", "single_group"]
DebugSessionSurfacePreset = Literal["notebook_default", "parity_surface"]


@dataclass(frozen=True)
class ConstrainedFeatureSelectionRef:
    ref: ConstrainedFeatureSelectionRefValue
    activation_value: float | None = None


DEFAULT_CONCEPT_PAIR_CONFIG_DIR = Path(__file__).with_name("configs")
DEFAULT_NOTEBOOK_PATH = Path(__file__).resolve().with_name("concept_direction_template.ipynb")
DEFAULT_LOCAL_NEURONPEDIA_EXPORT_ROOT = Path(
    os.getenv(
        "LOCAL_NEURONPEDIA_EXPORT_ROOT",
        "/home/speediedan/repos/neuronpedia/utils/neuronpedia-utils/neuronpedia_utils/exports",
    )
)
NULL_BATCH: Any = cast(Any, None)


@dataclass
class ConceptPair:
    """A pair of concept groups for direction computation and intervention."""

    name: str
    description: str
    group_a_tokens: list[str]
    group_b_tokens: list[str]
    group_a_entities: list[tuple[str, str]]
    group_b_entities: list[tuple[str, str]]
    group_a_name: str
    group_b_name: str
    concept_label: str
    classification_question: str


def _normalize_entity_pairs(raw_rows: Any, field_name: str) -> list[tuple[str, str]]:
    if not isinstance(raw_rows, list):
        raise ValueError(f"Concept pair field '{field_name}' must be a list of [entity, label] rows.")

    normalized: list[tuple[str, str]] = []
    for row in raw_rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"Concept pair field '{field_name}' rows must each contain exactly two values.")
        normalized.append((str(row[0]), str(row[1])))
    return normalized


def load_concept_pair_from_dict(payload: dict[str, Any]) -> ConceptPair:
    """Build a ``ConceptPair`` from a YAML-decoded mapping."""

    required_fields = (
        "name",
        "description",
        "group_a_tokens",
        "group_b_tokens",
        "group_a_entities",
        "group_b_entities",
        "group_a_name",
        "group_b_name",
        "concept_label",
        "classification_question",
    )
    missing_fields = [field_name for field_name in required_fields if field_name not in payload]
    if missing_fields:
        raise ValueError(f"Concept pair YAML is missing required fields: {', '.join(missing_fields)}")

    return ConceptPair(
        name=str(payload["name"]),
        description=str(payload["description"]),
        group_a_tokens=[str(token) for token in payload["group_a_tokens"]],
        group_b_tokens=[str(token) for token in payload["group_b_tokens"]],
        group_a_entities=_normalize_entity_pairs(payload["group_a_entities"], "group_a_entities"),
        group_b_entities=_normalize_entity_pairs(payload["group_b_entities"], "group_b_entities"),
        group_a_name=str(payload["group_a_name"]),
        group_b_name=str(payload["group_b_name"]),
        concept_label=str(payload["concept_label"]),
        classification_question=str(payload["classification_question"]),
    )


def load_concept_pair(path: str | Path) -> ConceptPair:
    """Load a concept pair definition from YAML."""

    import yaml  # type: ignore[import-untyped]

    concept_pair_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(concept_pair_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Concept pair config must parse to a mapping: {concept_pair_path}")
    return load_concept_pair_from_dict(payload)


def _concept_pair_family_slug(model_family: str) -> str:
    lowered = model_family.lower()
    if lowered.startswith("gemma"):
        return "gemma"
    return lowered.replace("-", "_")


def _concept_pair_context_slug(prompt_render_mode: PromptRenderMode) -> str:
    return "pt" if prompt_render_mode == "plain" else "it"


def resolve_concept_pair_config_path(
    concept_pair_name: str | None,
    *,
    model_family: str,
    prompt_render_mode: PromptRenderMode,
    concept_pair_config_path: str | Path | None = None,
    config_dir: Path = DEFAULT_CONCEPT_PAIR_CONFIG_DIR,
) -> Path:
    """Resolve the concept-pair YAML path from an explicit path or naming convention."""

    if concept_pair_config_path is not None:
        candidate = Path(concept_pair_config_path).expanduser()
        if not candidate.is_absolute():
            candidate = (config_dir / candidate).resolve()
        return candidate

    if not concept_pair_name:
        raise ValueError("concept_pair_name is required when concept_pair_config_path is not provided")

    derived_name = (
        f"cp_{concept_pair_name}_{_concept_pair_family_slug(model_family)}_"
        f"{_concept_pair_context_slug(prompt_render_mode)}.yaml"
    )
    return (config_dir / derived_name).resolve()


def _normalize_constrained_feature_selection_ref(raw_ref: Any) -> ConstrainedFeatureSelectionRef:
    if isinstance(raw_ref, ConstrainedFeatureSelectionRef):
        return raw_ref

    activation_value = None
    if isinstance(raw_ref, Mapping):
        ref_value = raw_ref.get("ref", raw_ref.get("feature_ref"))
        activation_value_raw = raw_ref.get("activation_value")
        if ref_value is None:
            raise ValueError(
                "Constrained feature selection mappings must include 'ref' (or legacy 'feature_ref')."
            )
        raw_ref = ref_value
        activation_value = None if activation_value_raw is None else float(activation_value_raw)

    if isinstance(raw_ref, str):
        return ConstrainedFeatureSelectionRef(ref=raw_ref, activation_value=activation_value)
    if isinstance(raw_ref, Sequence) and not isinstance(raw_ref, (str, bytes)) and len(raw_ref) == 4:
        model_id, source_set, layer_number, feature_index = raw_ref
        return ConstrainedFeatureSelectionRef(
            ref=(str(model_id), str(source_set), int(layer_number), int(feature_index)),
            activation_value=activation_value,
        )
    raise ValueError(
        "Constrained feature selection entries must be full Neuronpedia refs or "
        "(model_id, source_set, layer, feature_index) tuples, optionally wrapped in a mapping with "
        "an activation_value override."
    )


def _normalize_constrained_feature_selection_refs(
    raw_refs: Iterable[Any] | None,
) -> tuple[ConstrainedFeatureSelectionRef, ...] | None:
    if raw_refs is None:
        return None
    return tuple(_normalize_constrained_feature_selection_ref(raw_ref) for raw_ref in raw_refs)


def _serialize_constrained_feature_selection_ref(
    raw_ref: ConstrainedFeatureSelectionRef,
) -> str | list[Any] | dict[str, Any]:
    if isinstance(raw_ref.ref, str):
        serialized_ref: str | list[Any] = raw_ref.ref
    else:
        model_id, source_set, layer_number, feature_index = raw_ref.ref
        serialized_ref = [model_id, source_set, layer_number, feature_index]

    if raw_ref.activation_value is None:
        return serialized_ref
    return {"ref": serialized_ref, "activation_value": float(raw_ref.activation_value)}


def _build_classification_prompt(entity_name: str, question: str) -> str:
    """Build a classification-style prompt for a single entity."""
    return f"{question} {entity_name} : "


def _chattify_apply_chat_template(prompt: str, tokenizer: Any) -> str:
    """Wrap a prompt in the model's chat template using tokenizer.apply_chat_template."""
    cfg = GemmaPromptConfig()
    return cfg.apply_chat_template_fn(tokenizer, prompt, tokenize=False, add_generation_prompt=True)


def _chattify_gemma_dataclass(prompt: str) -> str:
    """Wrap a prompt using the GemmaPromptConfig dataclass approach."""
    cfg = GemmaPromptConfig()
    return cfg.model_chat_template_fn(prompt, tokenization_pattern="gemma-chat")


def _chattify(prompt: str, tokenizer: Any, method: str = "apply_chat_template") -> str:
    """Apply chat template using the configured method."""
    if method == "gemma_dataclass":
        return _chattify_gemma_dataclass(prompt)
    return _chattify_apply_chat_template(prompt, tokenizer)


@dataclass
class NotebookNeuronpediaRuntimeConfig:
    use_localhost: bool
    base_url: str
    local_db_url: str | None
    local_webapp_url: str
    upload_local_graphs: bool
    local_graph_slug_prefix: str | None
    local_graph_upload_target: str
    local_graph_owner_username: str | None
    check_local_explanation_coverage: bool
    generate_missing_local_explanations: bool
    local_explanation_feature_limit: int
    service_status: LocalNeuronpediaServiceStatus | None = None
    warning_messages: tuple[str, ...] = ()


def resolve_neuronpedia_runtime_config(
    *,
    use_localhost: bool,
    neuronpedia_base_url_override: str | None = None,
    local_db_url: str | None = None,
    local_webapp_url: str | None = None,
    upload_local_graphs: bool = False,
    local_graph_slug_prefix: str | None = None,
    local_graph_upload_target: str = "localhost",
    local_graph_owner_username: str | None = None,
    check_local_explanation_coverage: bool = False,
    generate_missing_local_explanations: bool = False,
    local_explanation_feature_limit: int = 20,
) -> NotebookNeuronpediaRuntimeConfig:
    """Resolve public vs localhost Neuronpedia mode and explanation workflow settings."""

    resolved_local_webapp_url = (local_webapp_url or DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL).rstrip("/")
    effective_use_localhost = use_localhost
    effective_upload_local_graphs = upload_local_graphs
    effective_check_local_explanation_coverage = check_local_explanation_coverage
    effective_generate_missing_local_explanations = generate_missing_local_explanations
    resolved_upload_target = str(local_graph_upload_target).strip() or "localhost"
    resolved_owner_username = None
    if local_graph_owner_username is not None:
        resolved_owner_username = str(local_graph_owner_username).strip() or None
    warning_messages: list[str] = []

    if resolved_upload_target not in {"localhost", "public_then_sync_local"}:
        raise ValueError(
            "local_graph_upload_target must be 'localhost' or 'public_then_sync_local'"
        )

    should_probe_local_services = (
        effective_use_localhost
        or effective_upload_local_graphs
        or effective_check_local_explanation_coverage
        or effective_generate_missing_local_explanations
        or local_db_url is not None
    )
    service_status = (
        check_local_neuronpedia_services(local_db_url=local_db_url, webapp_url=resolved_local_webapp_url)
        if should_probe_local_services
        else None
    )
    local_services_ready = bool(service_status and service_status.webapp_available and service_status.db_available)

    if (
        effective_check_local_explanation_coverage or effective_generate_missing_local_explanations
    ) and not effective_use_localhost:
        if local_services_ready:
            warning_messages.append(
                "Enabling localhost mode because local explanation coverage requires a live local "
                "Neuronpedia webapp and DB."
            )
            effective_use_localhost = True
        else:
            warning_messages.append(
                "Disabling local explanation coverage because USE_LOCALHOST is false and local "
                "Neuronpedia services are not available."
            )
            effective_check_local_explanation_coverage = False
            effective_generate_missing_local_explanations = False

    if effective_upload_local_graphs and not effective_use_localhost:
        if local_services_ready:
            warning_messages.append(
                "Enabling localhost mode because local graph upload requires a live local Neuronpedia webapp and DB."
            )
            effective_use_localhost = True
        else:
            warning_messages.append(
                "Disabling local graph upload because USE_LOCALHOST is false and local Neuronpedia services are "
                "not available."
            )
            effective_upload_local_graphs = False

    if effective_use_localhost and not local_services_ready:
        warning_messages.append(
            "Falling back to public Neuronpedia because the local Neuronpedia webapp or DB is unavailable."
        )
        effective_use_localhost = False
        effective_upload_local_graphs = False
        effective_check_local_explanation_coverage = False
        effective_generate_missing_local_explanations = False

    if (
        effective_upload_local_graphs
        and resolved_upload_target == "public_then_sync_local"
        and resolved_owner_username is None
    ):
        env_username = os.environ.get("LOCAL_NEURONPEDIA_USERNAME") or os.environ.get("USER")
        resolved_owner_username = None if not env_username else str(env_username).strip() or None
        if resolved_owner_username is None:
            raise ValueError(
                    "public_then_sync_local graph upload requires local_graph_owner_username "
                    "or a USER environment value"
            )

    resolved_base_url = (
        neuronpedia_base_url_override.rstrip("/")
        if neuronpedia_base_url_override
        else resolved_local_webapp_url if effective_use_localhost else DEFAULT_NEURONPEDIA_BASE_URL
    )

    for warning_message in warning_messages:
        warnings.warn(warning_message, stacklevel=2)

    return NotebookNeuronpediaRuntimeConfig(
        use_localhost=effective_use_localhost,
        base_url=resolved_base_url,
        local_db_url=local_db_url,
        local_webapp_url=resolved_local_webapp_url,
        upload_local_graphs=effective_upload_local_graphs,
        local_graph_slug_prefix=(
            None if local_graph_slug_prefix is None else str(local_graph_slug_prefix).strip() or None
        ),
        local_graph_upload_target=resolved_upload_target,
        local_graph_owner_username=resolved_owner_username,
        check_local_explanation_coverage=effective_check_local_explanation_coverage,
        generate_missing_local_explanations=effective_generate_missing_local_explanations,
        local_explanation_feature_limit=int(local_explanation_feature_limit),
        service_status=service_status,
        warning_messages=tuple(warning_messages),
    )


@dataclass
class NotebookHarnessConfig:
    experiment_name: str
    experiment_config_name: str
    model_family: str
    model_variant: str
    model_name: str
    transcoder_set: str
    hf_model_head: str | None
    neuronpedia_model: str
    neuronpedia_set: str
    neuronpedia_base_url: str
    concept_pair_name: str | None
    prompt: str
    prompt_render_mode: PromptRenderMode
    target_tokens: tuple[str, str] | None
    target_token_ids: tuple[int, int] | None
    top_n: int
    default_scale_factor: float
    scale_factor_sweep: list[float]
    ablation_n_list: list[int]
    enable_sign_aware: bool
    force_device: str | None
    work_root: Any
    analysis_mode: AnalysisMode = "concept_pair"
    concept_direction_mode: ConceptDirectionMode = "paired_rejection"
    explicit_direction_tokens: tuple[str, str] | None = None
    enable_zero_softcap: bool = False
    batch_size: int | None = None
    max_feature_nodes: int | None = None
    use_localhost: bool = False
    local_neuronpedia_db_url: str | None = None
    local_neuronpedia_webapp_url: str = DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL
    upload_local_graphs: bool = False
    local_graph_slug_prefix: str | None = None
    local_graph_upload_target: str = "localhost"
    local_graph_owner_username: str | None = None
    check_local_explanation_coverage: bool = False
    generate_missing_local_explanations: bool = False
    local_explanation_feature_limit: int = 20
    local_neuronpedia_service_status: LocalNeuronpediaServiceStatus | None = None
    mode_warning_messages: tuple[str, ...] = field(default_factory=tuple)
    local_explanation_type_name: str = DEFAULT_EXPLANATION_TYPE_NAME
    local_explanation_timeout_seconds: int = DEFAULT_COPILOT_TIMEOUT_SECONDS
    local_explanation_max_retries: int = DEFAULT_COPILOT_MAX_RETRIES
    local_explanation_retry_backoff_seconds: float = DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS
    key_tokens_override: tuple[str, ...] | None = None
    concept_pair_config_path: str | None = None
    constrained_feature_selection_refs: tuple[ConstrainedFeatureSelectionRef, ...] | None = None
    store_latent_extraction_mode: StoreLatentExtractionMode = "answer_position_state"
    context_enhanced_scale: float = 1.0
    direct_projection_interventions: dict[str, Any] | None = None
    direct_projection_intervention_hook_pattern: str | None = None
    direct_projection_intervention_mode: str | None = None
    direct_projection_intervention_scale_factor: float | None = None
    direct_projection_intervention_use_intervention_tensor_as_basis: bool | None = None
    store_concept_cache_key: str = "unembed.hook_in"
    store_concept_correct_only: bool = False
    store_weight_by_logit_diff: bool = True
    enable_baseline_path_debug: bool = False
    debug_validation_logit_atol: float = 1e-4
    debug_validation_logit_rtol: float = 1e-3
    debug_validation_act_atol: float = 1e-3
    debug_validation_act_rtol: float = 1e-5
    debug_validation_top_k: int = 10
    debug_validation_raise_on_failure: bool = True
    debug_session_surface_preset: DebugSessionSurfacePreset = "notebook_default"
    concept_pair: ConceptPair = field(init=False)
    shared_sections: SharedHarnessSections = field(init=False)

    def __post_init__(self) -> None:
        self.analysis_mode = cast(AnalysisMode, str(self.analysis_mode).strip())
        self.concept_direction_mode = cast(ConceptDirectionMode, str(self.concept_direction_mode).strip())
        requested_concept_pair_name = None if self.concept_pair_name is None else str(self.concept_pair_name).strip()
        resolved_concept_pair_path = resolve_concept_pair_config_path(
            requested_concept_pair_name,
            model_family=self.model_family,
            prompt_render_mode=self.prompt_render_mode,
            concept_pair_config_path=self.concept_pair_config_path,
        )
        self.concept_pair = load_concept_pair(resolved_concept_pair_path)
        if requested_concept_pair_name and requested_concept_pair_name != self.concept_pair.name:
            warnings.warn(
                "CONCEPT_PAIR_CONFIG_PATH takes precedence over concept_pair_name; using "
                f"{self.concept_pair.name!r} from {resolved_concept_pair_path.name}.",
                stacklevel=2,
            )
        self.concept_pair_name = self.concept_pair.name
        self.concept_pair_config_path = str(resolved_concept_pair_path)
        if self.key_tokens_override is None or not self.key_tokens_override:
            raise ValueError(
                "NotebookHarnessConfig requires KEY_TOKENS in the experiment config; concept-pair YAMLs no longer "
                "provide key token defaults."
            )
        if self.target_tokens is not None:
            if len(self.target_tokens) != 2:
                raise ValueError("target_tokens must contain exactly two tokens")
            self.target_tokens = (str(self.target_tokens[0]), str(self.target_tokens[1]))
        if self.target_token_ids is not None:
            if len(self.target_token_ids) != 2:
                raise ValueError("target_token_ids must contain exactly two token ids")
            self.target_token_ids = (int(self.target_token_ids[0]), int(self.target_token_ids[1]))
        if self.target_tokens is None and self.target_token_ids is None:
            raise ValueError("either target_tokens or target_token_ids must be provided")
        if self.local_graph_slug_prefix is not None:
            stripped_prefix = str(self.local_graph_slug_prefix).strip()
            self.local_graph_slug_prefix = stripped_prefix or None
        self.local_graph_upload_target = str(self.local_graph_upload_target).strip() or "localhost"
        if self.local_graph_upload_target not in {"localhost", "public_then_sync_local"}:
            raise ValueError(
                "local_graph_upload_target must be 'localhost' or 'public_then_sync_local'"
            )
        if self.local_graph_owner_username is not None:
            stripped_owner_username = str(self.local_graph_owner_username).strip()
            self.local_graph_owner_username = stripped_owner_username or None
        if self.upload_local_graphs and self.local_graph_upload_target == "public_then_sync_local":
            if self.local_graph_owner_username is None:
                env_username = os.environ.get("LOCAL_NEURONPEDIA_USERNAME") or os.environ.get("USER")
                self.local_graph_owner_username = None if not env_username else str(env_username).strip() or None
            if self.local_graph_owner_username is None:
                raise ValueError(
                    "public_then_sync_local graph upload requires local_graph_owner_username "
                    "or a USER environment value"
                )
        if self.explicit_direction_tokens is not None:
            if len(self.explicit_direction_tokens) != 2:
                raise ValueError("explicit_direction_tokens must contain exactly two tokens")
            self.explicit_direction_tokens = (
                str(self.explicit_direction_tokens[0]),
                str(self.explicit_direction_tokens[1]),
            )
        self.scale_factor_sweep = [float(value) for value in self.scale_factor_sweep]
        self.ablation_n_list = [int(value) for value in self.ablation_n_list]
        self.debug_validation_top_k = int(self.debug_validation_top_k)
        self.debug_session_surface_preset = cast(
            DebugSessionSurfacePreset,
            str(self.debug_session_surface_preset).strip(),
        )
        if self.concept_direction_mode not in {"mean_difference", "paired_rejection", "single_group"}:
            raise ValueError("concept_direction_mode must be 'mean_difference', 'paired_rejection', or 'single_group'")
        if self.debug_session_surface_preset not in {"notebook_default", "parity_surface"}:
            raise ValueError(
                "debug_session_surface_preset must be 'notebook_default' or 'parity_surface'"
            )
        if self.debug_validation_top_k <= 0:
            raise ValueError("debug_validation_top_k must be a positive integer")
        self.constrained_feature_selection_refs = _normalize_constrained_feature_selection_refs(
            self.constrained_feature_selection_refs
        )
        if self.direct_projection_interventions is not None:
            if not isinstance(self.direct_projection_interventions, Mapping):
                raise ValueError("ANALYSIS.direct_projection.interventions must be a mapping when provided")
            self.direct_projection_interventions = dict(self.direct_projection_interventions)
        if self.direct_projection_intervention_scale_factor is not None:
            self.direct_projection_intervention_scale_factor = float(self.direct_projection_intervention_scale_factor)
        if self.direct_projection_intervention_use_intervention_tensor_as_basis is not None:
            self.direct_projection_intervention_use_intervention_tensor_as_basis = bool(
                self.direct_projection_intervention_use_intervention_tensor_as_basis
            )

        mode_warning_messages = list(self.mode_warning_messages)
        if self.analysis_mode == "explicit_embedding_difference":
            if self.explicit_direction_tokens is None:
                raise ValueError(
                    "explicit_embedding_difference mode requires explicit_direction_tokens to be provided"
                )
            mode_warning_messages.append(
                "Explicit embedding-difference mode uses EXPLICIT_DIRECTION_TOKENS for attribution and skips "
                "concept-pair comparison phases."
            )
        elif self.analysis_mode == "debug_intervention_pipelines":
            if not self.constrained_feature_selection_refs or len(self.constrained_feature_selection_refs) != 1:
                raise ValueError(
                    "debug_intervention_pipelines mode requires exactly one constrained feature selection ref"
                )
            mode_warning_messages.append(
                "Debug intervention mode bypasses concept-direction phases and validates a single constrained "
                "feature against the attribution graph."
            )
            if not self.debug_validation_raise_on_failure:
                mode_warning_messages.append(
                    "Debug intervention validation failures will be recorded without raising so the notebook can "
                    "complete and retain diagnostics."
                )
        elif self.analysis_mode != "concept_pair":
            raise ValueError(
                "analysis_mode must be one of 'concept_pair', 'explicit_embedding_difference', or "
                "'debug_intervention_pipelines'"
            )

        if self.debug_session_surface_preset == "parity_surface":
            mode_warning_messages.append(
                "Notebook sessions will use the parity-aligned session surface "
                "(preserving the configured or auto-selected device, plus eager attention, float32 NNsight/"
                "circuit-tracer dtype, cleared default analysis target tokens, CPU offload, and quieter "
                "circuit-tracer logging)."
            )

        if self.enable_zero_softcap:
            mode_warning_messages.append(
                "zero_softcap is enabled for all supported forward and intervention paths in this notebook run."
            )
        self.mode_warning_messages = tuple(mode_warning_messages)
        self.shared_sections = build_shared_harness_sections(
            model_family=self.model_family,
            model_variant=self.model_variant,
            model_name=self.model_name,
            transcoder_set=self.transcoder_set,
            neuronpedia_model=self.neuronpedia_model,
            neuronpedia_set=self.neuronpedia_set,
            hf_model_head=self.hf_model_head,
            prompt=self.prompt,
            prompt_render_mode=self.prompt_render_mode,
            target_tokens=self.target_tokens,
            target_token_ids=self.target_token_ids,
            key_tokens=self.key_tokens_override,
            explicit_direction_tokens=self.explicit_direction_tokens,
            force_device=self.force_device,
            batch_size=self.batch_size,
            max_feature_nodes=self.max_feature_nodes,
            debug_session_surface_preset=self.debug_session_surface_preset,
            neuronpedia_base_url=self.neuronpedia_base_url,
            use_localhost=self.use_localhost,
            local_neuronpedia_db_url=self.local_neuronpedia_db_url,
            local_neuronpedia_webapp_url=self.local_neuronpedia_webapp_url,
            upload_local_graphs=self.upload_local_graphs,
            local_graph_slug_prefix=self.local_graph_slug_prefix,
            local_graph_upload_target=self.local_graph_upload_target,
            local_graph_owner_username=self.local_graph_owner_username,
            check_local_explanation_coverage=self.check_local_explanation_coverage,
            generate_missing_local_explanations=self.generate_missing_local_explanations,
            local_explanation_feature_limit=self.local_explanation_feature_limit,
            local_explanation_type_name=self.local_explanation_type_name,
            local_explanation_timeout_seconds=self.local_explanation_timeout_seconds,
            local_explanation_max_retries=self.local_explanation_max_retries,
            local_explanation_retry_backoff_seconds=self.local_explanation_retry_backoff_seconds,
            local_neuronpedia_service_status=self.local_neuronpedia_service_status,
            mode_warning_messages=self.mode_warning_messages,
            enable_zero_softcap=self.enable_zero_softcap,
            enable_baseline_path_debug=self.enable_baseline_path_debug,
            debug_validation_logit_atol=self.debug_validation_logit_atol,
            debug_validation_logit_rtol=self.debug_validation_logit_rtol,
            debug_validation_act_atol=self.debug_validation_act_atol,
            debug_validation_act_rtol=self.debug_validation_act_rtol,
            debug_validation_top_k=self.debug_validation_top_k,
            debug_validation_raise_on_failure=self.debug_validation_raise_on_failure,
        )

    @property
    def use_chat_template(self) -> bool:
        return self.prompt_render_mode != "plain"

    @property
    def uses_explicit_embedding_difference(self) -> bool:
        return self.analysis_mode == "explicit_embedding_difference"

    @property
    def is_debug_intervention_mode(self) -> bool:
        return self.analysis_mode == "debug_intervention_pipelines"

    @property
    def supports_store_direction(self) -> bool:
        return self.analysis_mode == "concept_pair"

    @property
    def analysis_concept_label(self) -> str:
        if self.uses_explicit_embedding_difference and self.explicit_direction_tokens is not None:
            return f"{self.explicit_direction_tokens[0]} - {self.explicit_direction_tokens[1]}"
        if self.is_debug_intervention_mode:
            return "debug_key_token_logits"
        return self.concept_pair.concept_label

    @property
    def analysis_direction_mode_name(self) -> str:
        return self.concept_direction_mode

    @property
    def chat_template_method(self) -> str:
        if self.prompt_render_mode == "gemma_dataclass":
            return "gemma_dataclass"
        if self.prompt_render_mode == "apply_chat_template":
            return "apply_chat_template"
        return "plain"

    @property
    def session_kwargs(self) -> dict[str, Any]:
        """Common kwargs passed to every ``experiment_session`` call."""
        return {
            "model_family": self.model_family,
            "model_variant": self.model_variant,
            "model_name": self.model_name,
            "transcoder_set": self.transcoder_set,
            "hf_model_head": self.hf_model_head,
            "force_device": self.force_device,
            "batch_size": self.batch_size,
            "max_feature_nodes": self.max_feature_nodes,
            "debug_session_surface_preset": self.debug_session_surface_preset,
            "enable_neuronpedia_graph_upload": self.upload_local_graphs,
            "neuronpedia_graph_slug_prefix": self.local_graph_slug_prefix,
            "neuronpedia_model": self.neuronpedia_model,
            "neuronpedia_source_set": self.neuronpedia_set,
        }


def _normalize_token_pair(value: Any, *, field_name: str) -> tuple[str, str] | None:
    if value is None:
        return None
    items = tuple(str(item) for item in value)
    if len(items) != 2:
        raise ValueError(f"{field_name} must contain exactly two values.")
    return cast(tuple[str, str], items)


def _normalize_int_pair(value: Any, *, field_name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    items = tuple(int(item) for item in value)
    if len(items) != 2:
        raise ValueError(f"{field_name} must contain exactly two values.")
    return cast(tuple[int, int], items)


def _resolve_prompt_text(payload: Mapping[str, Any]) -> str:
    prompt_override = get_config_value(payload, section="PROMPT", key="text", flat_key="PROMPT_OVERRIDE")
    if prompt_override is None:
        raise ValueError("PROMPT_OVERRIDE must be provided by the experiment config.")
    return str(prompt_override)


def build_notebook_harness_config(
    config_path: str | Path,
) -> tuple[NotebookHarnessConfig, bool, dict[str, Any]]:
    resolved_payload = load_experiment_config(config_path)
    resolved_config_path = Path(resolved_payload["EXPERIMENT_CONFIG_PATH"]).resolve()
    config_name = str(resolved_payload.get("EXPERIMENT_CONFIG_NAME", resolved_config_path.stem))
    experiment_name = str(resolved_payload.get("EXPERIMENT_NAME", config_name))

    work_root_base = get_config_value(
        resolved_payload,
        section="RUNTIME",
        key="experiment_work_dir",
        flat_key="EXPERIMENT_WORK_DIR",
    )
    work_root = create_work_root(work_root_base, experiment_name, prefix="concept_dir")
    should_cleanup_work_root = work_root_base is None

    model_family = str(
        get_required_config_value(resolved_payload, section="MODEL", key="family", flat_key="MODEL_FAMILY")
    )
    model_variant = str(
        get_required_config_value(resolved_payload, section="MODEL", key="variant", flat_key="MODEL_VARIANT")
    )
    prompt_render_mode = cast(
        PromptRenderMode,
        str(
            get_config_value(
                resolved_payload,
                section="PROMPT",
                key="render_mode",
                flat_key="PROMPT_RENDER_MODE",
                default="plain",
            )
        ).strip(),
    )

    model_spec = resolve_model_spec(
        model_family,
        model_variant,
        model_name_override=get_config_value(
            resolved_payload,
            section="MODEL",
            key="model_name",
            flat_key="MODEL_NAME_OVERRIDE",
        ),
        transcoder_set_override=get_config_value(
            resolved_payload,
            section="MODEL",
            key="transcoder_set",
            flat_key="TRANSCODER_SET_OVERRIDE",
        ),
        neuronpedia_model_override=get_config_value(
            resolved_payload,
            section="MODEL",
            key="neuronpedia_model",
            flat_key="NEURONPEDIA_MODEL_OVERRIDE",
        ),
        neuronpedia_set_override=get_config_value(
            resolved_payload,
            section="MODEL",
            key="neuronpedia_set",
            flat_key="NEURONPEDIA_SET_OVERRIDE",
        ),
    )

    runtime = resolve_neuronpedia_runtime_config(
        use_localhost=bool(
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="use_localhost",
                flat_key="USE_LOCALHOST",
                default=False,
            )
        ),
        neuronpedia_base_url_override=get_config_value(
            resolved_payload,
            section="NEURONPEDIA",
            key="base_url_override",
            flat_key="NEURONPEDIA_BASE_URL_OVERRIDE",
        ),
        local_db_url=get_config_value(
            resolved_payload,
            section="NEURONPEDIA",
            key="local_db_url",
            flat_key="LOCAL_NEURONPEDIA_DB_URL",
        ),
        local_webapp_url=get_config_value(
            resolved_payload,
            section="NEURONPEDIA",
            key="local_webapp_url",
            flat_key="LOCAL_NEURONPEDIA_WEBAPP_URL",
        ),
        upload_local_graphs=bool(
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="upload_local_graphs",
                flat_key="UPLOAD_LOCAL_NEURONPEDIA_GRAPHS",
                default=False,
            )
        ),
        local_graph_slug_prefix=cast(
            str | None,
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="local_graph_slug_prefix",
                flat_key="LOCAL_GRAPH_SLUG_PREFIX",
            ),
        ),
        local_graph_upload_target=cast(
            str,
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="local_graph_upload_target",
                flat_key="LOCAL_GRAPH_UPLOAD_TARGET",
                default="localhost",
            ),
        ),
        local_graph_owner_username=cast(
            str | None,
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="local_graph_owner_username",
                flat_key="LOCAL_GRAPH_OWNER_USERNAME",
            ),
        ),
        check_local_explanation_coverage=bool(
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="check_local_explanation_coverage",
                flat_key="CHECK_LOCAL_EXPLANATION_COVERAGE",
                default=False,
            )
        ),
        generate_missing_local_explanations=bool(
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="generate_missing_local_explanations",
                flat_key="GENERATE_MISSING_LOCAL_EXPLANATIONS",
                default=False,
            )
        ),
        local_explanation_feature_limit=int(
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="local_explanation_feature_limit",
                flat_key="LOCAL_EXPLANATION_FEATURE_LIMIT",
                default=20,
            )
        ),
    )

    concept_pair_path = resolve_concept_pair_config_path(
        cast(str | None, resolved_payload.get("CONCEPT_PAIR_NAME")),
        model_family=model_family,
        prompt_render_mode=prompt_render_mode,
        config_dir=Path(config_path).expanduser().resolve().parent,
        concept_pair_config_path=get_config_value(
            resolved_payload,
            section="EXPERIMENT",
            key="concept_pair_config_path",
            flat_key="CONCEPT_PAIR_CONFIG_PATH",
        ),
    )
    prompt = _resolve_prompt_text(resolved_payload)
    analysis_cfg = cast(Mapping[str, Any], resolved_payload.get("ANALYSIS", {}))
    direct_projection_cfg = cast(Mapping[str, Any], analysis_cfg.get("direct_projection", {}))
    debug_session_surface_preset = cast(
        DebugSessionSurfacePreset,
        str(
            get_config_value(
                resolved_payload,
                section="SESSION",
                key="debug_session_surface_preset",
                flat_key="DEBUG_SESSION_SURFACE_PRESET",
                default="notebook_default",
            )
        ).strip(),
    )
    preset_config_defaults = resolve_session_surface_preset_config_defaults(debug_session_surface_preset)

    cfg = NotebookHarnessConfig(
        experiment_name=experiment_name,
        experiment_config_name=config_name,
        model_family=model_family,
        model_variant=model_variant,
        model_name=model_spec.model_name,
        transcoder_set=model_spec.transcoder_set,
        hf_model_head=model_spec.hf_model_head,
        neuronpedia_model=model_spec.neuronpedia_model,
        neuronpedia_set=model_spec.neuronpedia_set,
        neuronpedia_base_url=runtime.base_url,
        concept_pair_name=cast(str | None, resolved_payload.get("CONCEPT_PAIR_NAME")),
        concept_pair_config_path=str(concept_pair_path),
        prompt=prompt,
        prompt_render_mode=prompt_render_mode,
        target_tokens=_normalize_token_pair(
            get_config_value(resolved_payload, section="PROMPT", key="target_tokens", flat_key="TARGET_TOKENS"),
            field_name="TARGET_TOKENS",
        ),
        target_token_ids=_normalize_int_pair(
            get_config_value(
                resolved_payload,
                section="PROMPT",
                key="target_token_ids",
                flat_key="TARGET_TOKEN_IDS",
            ),
            field_name="TARGET_TOKEN_IDS",
        ),
        top_n=int(get_config_value(resolved_payload, section="ANALYSIS", key="top_n", flat_key="TOP_N", default=10)),
        default_scale_factor=float(
            get_config_value(
                resolved_payload,
                section="ANALYSIS",
                key="default_scale_factor",
                flat_key="DEFAULT_SCALE_FACTOR",
                default=10.0,
            )
        ),
        scale_factor_sweep=list(
            get_config_value(
                resolved_payload,
                section="ANALYSIS",
                key="scale_factor_sweep",
                flat_key="SCALE_FACTOR_SWEEP",
                default=[2.0, 5.0, 10.0, 20.0, 50.0],
            )
        ),
        ablation_n_list=list(
            get_config_value(
                resolved_payload,
                section="ANALYSIS",
                key="ablation_n_list",
                flat_key="ABLATION_N_LIST",
                default=[5, 10, 25, 50, 100],
            )
        ),
        enable_sign_aware=bool(
            get_config_value(
                resolved_payload,
                section="ANALYSIS",
                key="enable_sign_aware",
                flat_key="ENABLE_SIGN_AWARE",
                default=False,
            )
        ),
        force_device=cast(
            str | None,
            get_config_value(resolved_payload, section="SESSION", key="force_device", flat_key="FORCE_DEVICE"),
        ),
        work_root=work_root,
        analysis_mode=cast(
            AnalysisMode,
            str(
                get_config_value(
                    resolved_payload,
                    section="ANALYSIS",
                    key="mode",
                    flat_key="ANALYSIS_MODE",
                    default="concept_pair",
                )
            ).strip(),
        ),
        concept_direction_mode=cast(
            ConceptDirectionMode,
            str(
                get_config_value(
                    resolved_payload,
                    section="ANALYSIS",
                    key="concept_direction_mode",
                    flat_key="CONCEPT_DIRECTION_MODE",
                    default="paired_rejection",
                )
            ).strip(),
        ),
        explicit_direction_tokens=_normalize_token_pair(
            get_config_value(
                resolved_payload,
                section="ANALYSIS",
                key="explicit_direction_tokens",
                flat_key="EXPLICIT_DIRECTION_TOKENS",
            ),
            field_name="EXPLICIT_DIRECTION_TOKENS",
        ),
        enable_zero_softcap=bool(
            _get_config_value_with_preset_default(
                resolved_payload,
                preset_config_defaults,
                section="DEBUG_VALIDATION",
                key="enable_zero_softcap",
                flat_key="ENABLE_ZERO_SOFTCAP",
                default=False,
            )
        ),
        batch_size=cast(
            int | None,
            get_config_value(resolved_payload, section="SESSION", key="batch_size", flat_key="BATCH_SIZE"),
        ),
        max_feature_nodes=cast(
            int | None,
            get_config_value(
                resolved_payload,
                section="SESSION",
                key="max_feature_nodes",
                flat_key="MAX_FEATURE_NODES",
            ),
        ),
        use_localhost=runtime.use_localhost,
        local_neuronpedia_db_url=runtime.local_db_url,
        local_neuronpedia_webapp_url=runtime.local_webapp_url,
        upload_local_graphs=runtime.upload_local_graphs,
        local_graph_slug_prefix=runtime.local_graph_slug_prefix,
        local_graph_upload_target=runtime.local_graph_upload_target,
        local_graph_owner_username=runtime.local_graph_owner_username,
        check_local_explanation_coverage=runtime.check_local_explanation_coverage,
        generate_missing_local_explanations=runtime.generate_missing_local_explanations,
        local_explanation_feature_limit=runtime.local_explanation_feature_limit,
        local_neuronpedia_service_status=runtime.service_status,
        mode_warning_messages=runtime.warning_messages,
        local_explanation_type_name=str(
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="local_explanation_type_name",
                flat_key="LOCAL_EXPLANATION_TYPE_NAME",
                default=DEFAULT_EXPLANATION_TYPE_NAME,
            )
        ),
        local_explanation_timeout_seconds=int(
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="local_explanation_timeout_seconds",
                flat_key="LOCAL_EXPLANATION_TIMEOUT_SECONDS",
                default=DEFAULT_COPILOT_TIMEOUT_SECONDS,
            )
        ),
        local_explanation_max_retries=int(
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="local_explanation_max_retries",
                flat_key="LOCAL_EXPLANATION_MAX_RETRIES",
                default=DEFAULT_COPILOT_MAX_RETRIES,
            )
        ),
        local_explanation_retry_backoff_seconds=float(
            get_config_value(
                resolved_payload,
                section="NEURONPEDIA",
                key="local_explanation_retry_backoff_seconds",
                flat_key="LOCAL_EXPLANATION_RETRY_BACKOFF_SECONDS",
                default=DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS,
            )
        ),
        key_tokens_override=cast(
            tuple[str, ...] | None,
            tuple(
                str(token)
                for token in get_config_value(
                    resolved_payload,
                    section="PROMPT",
                    key="key_tokens",
                    flat_key="KEY_TOKENS",
                    default=(),
                )
            )
            or None,
        ),
        constrained_feature_selection_refs=cast(
            tuple[ConstrainedFeatureSelectionRef, ...] | None,
            get_config_value(
                resolved_payload,
                section="ANALYSIS",
                key="constrained_feature_selection",
                flat_key="CONSTRAINED_FEATURE_SELECTION_LIST",
            ),
        ),
        store_latent_extraction_mode=cast(
            StoreLatentExtractionMode,
            str(
                get_config_value(
                    resolved_payload,
                    section="ANALYSIS",
                    key="store_latent_extraction_mode",
                    flat_key="STORE_LATENT_EXTRACTION_MODE",
                    default="answer_position_state",
                )
            ).strip(),
        ),
        context_enhanced_scale=float(
            get_config_value(
                resolved_payload,
                section="ANALYSIS",
                key="context_enhanced_scale",
                flat_key="CONTEXT_ENHANCED_SCALE",
                default=1.0,
            )
        ),
        direct_projection_interventions=cast(
            dict[str, Any] | None,
            direct_projection_cfg.get("interventions"),
        ),
        direct_projection_intervention_hook_pattern=cast(
            str | None,
            direct_projection_cfg.get("intervention_hook_pattern"),
        ),
        direct_projection_intervention_mode=cast(
            str | None,
            direct_projection_cfg.get("intervention_mode"),
        ),
        direct_projection_intervention_scale_factor=cast(
            float | None,
            direct_projection_cfg.get("intervention_scale_factor"),
        ),
        direct_projection_intervention_use_intervention_tensor_as_basis=cast(
            bool | None,
            direct_projection_cfg.get("intervention_use_intervention_tensor_as_basis"),
        ),
        enable_baseline_path_debug=bool(
            get_config_value(
                resolved_payload,
                section="DEBUG_VALIDATION",
                key="enable_baseline_path_debug",
                flat_key="ENABLE_BASELINE_PATH_DEBUG",
                default=False,
            )
        ),
        debug_validation_logit_atol=float(
            _get_config_value_with_preset_default(
                resolved_payload,
                preset_config_defaults,
                section="DEBUG_VALIDATION",
                key="logit_atol",
                flat_key="DEBUG_VALIDATION_LOGIT_ATOL",
                default=1e-4,
            )
        ),
        debug_validation_logit_rtol=float(
            _get_config_value_with_preset_default(
                resolved_payload,
                preset_config_defaults,
                section="DEBUG_VALIDATION",
                key="logit_rtol",
                flat_key="DEBUG_VALIDATION_LOGIT_RTOL",
                default=1e-3,
            )
        ),
        debug_validation_act_atol=float(
            _get_config_value_with_preset_default(
                resolved_payload,
                preset_config_defaults,
                section="DEBUG_VALIDATION",
                key="act_atol",
                flat_key="DEBUG_VALIDATION_ACT_ATOL",
                default=1e-3,
            )
        ),
        debug_validation_act_rtol=float(
            _get_config_value_with_preset_default(
                resolved_payload,
                preset_config_defaults,
                section="DEBUG_VALIDATION",
                key="act_rtol",
                flat_key="DEBUG_VALIDATION_ACT_RTOL",
                default=1e-5,
            )
        ),
        debug_validation_top_k=int(
            _get_config_value_with_preset_default(
                resolved_payload,
                preset_config_defaults,
                section="DEBUG_VALIDATION",
                key="top_k",
                flat_key="DEBUG_VALIDATION_TOP_K",
                default=10,
            )
        ),
        debug_validation_raise_on_failure=bool(
            _get_config_value_with_preset_default(
                resolved_payload,
                preset_config_defaults,
                section="DEBUG_VALIDATION",
                key="raise_on_failure",
                flat_key="DEBUG_VALIDATION_RAISE_ON_FAILURE",
                default=True,
            )
        ),
        debug_session_surface_preset=debug_session_surface_preset,
    )

    return cfg, should_cleanup_work_root, resolved_payload


def resolve_key_tokens(cfg: NotebookHarnessConfig) -> tuple[str, ...]:
    """Return the experiment-owned key tokens used for analysis and reporting."""

    if cfg.key_tokens_override is None or not cfg.key_tokens_override:
        raise ValueError(
            "NotebookHarnessConfig requires KEY_TOKENS in the experiment config; concept-pair YAMLs no longer "
            "provide key token defaults."
        )
    return tuple(cfg.key_tokens_override)


def _build_key_token_candidates(
    cfg: NotebookHarnessConfig,
    tokenizer: Any,
    *,
    include_space_prefixed_variants: bool = True,
    include_bare_variants: bool = True,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def _append_candidate(
        token_id: int,
        *,
        label: str,
        source_token: str,
        variant: str,
        encoded_ids: Sequence[int],
    ) -> None:
        if token_id in seen_ids:
            return
        seen_ids.add(token_id)
        candidates.append(
            {
                "label": label,
                "token_id": int(token_id),
                "source_token": source_token,
                "variant": variant,
                "ids": [int(value) for value in encoded_ids],
                "decoded": tokenizer.decode([int(token_id)]),
            }
        )

    for source_token in resolve_key_tokens(cfg):
        if cfg.use_chat_template:
            normalized = _normalize_target_token_for_prompt_mode(source_token, cfg)

            if include_space_prefixed_variants:
                prefixed_ids = tokenizer.encode(f" {normalized}", add_special_tokens=False)
                if len(prefixed_ids) == 1:
                    _append_candidate(
                        int(prefixed_ids[0]),
                        label=f"▁{normalized}",
                        source_token=source_token,
                        variant="space_prefixed",
                        encoded_ids=prefixed_ids,
                    )

            if include_bare_variants:
                bare_ids = tokenizer.encode(normalized, add_special_tokens=False)
                if bare_ids:
                    bare_token_id = int(bare_ids[-1])
                    bare_label = normalized if len(bare_ids) == 1 else tokenizer.decode([bare_token_id]).lstrip()
                    _append_candidate(
                        bare_token_id,
                        label=bare_label,
                        source_token=source_token,
                        variant="bare",
                        encoded_ids=bare_ids,
                    )
            continue

        literal_ids = tokenizer.encode(source_token, add_special_tokens=False)
        if literal_ids:
            literal_token_id = int(literal_ids[-1])
            literal_label = source_token if len(literal_ids) == 1 else tokenizer.decode([literal_token_id]).lstrip()
            _append_candidate(
                literal_token_id,
                label=literal_label,
                source_token=source_token,
                variant="literal",
                encoded_ids=literal_ids,
            )

    return candidates


@dataclass(frozen=True)
class LocalExplanationPrefetchStatus:
    """Activation-cache readiness for one local Neuronpedia feature explanation."""

    feature_ref: NeuronpediaFeatureRef
    explanation_count: int
    cache_ready: bool
    cache_source: str
    cache_path: str | None = None
    activation_rows: int = 0
    error: str | None = None


@dataclass(frozen=True)
class LocalExplanationPreparationResult:
    """Summary of explanation availability and cache-prefetch readiness."""

    feature_refs: list[NeuronpediaFeatureRef]
    initial_statuses: list[NeuronpediaLocalExplanationStatus]
    prefetch_statuses: list[LocalExplanationPrefetchStatus]
    export_roots: tuple[str, ...]
    cache_dir: str

    @property
    def missing_feature_refs(self) -> list[NeuronpediaFeatureRef]:
        return [status.feature_ref for status in self.initial_statuses if not status.has_local_explanation]


def _resolve_local_export_roots(local_export_roots: Iterable[Path | str] | None = None) -> tuple[Path, ...]:
    candidate_roots = tuple(Path(root) for root in (local_export_roots or (DEFAULT_LOCAL_NEURONPEDIA_EXPORT_ROOT,)))
    return tuple(root for root in candidate_roots if root.exists())


def _feature_rows_to_layer_feature_tuples(feature_groups: Iterable[Any]) -> list[tuple[int, int]]:
    candidate_feature_tuples: list[tuple[int, int]] = []
    for feature_group in feature_groups:
        for feature_row in feature_group:
            normalized_row = tuple(int(value) for value in feature_row)
            if len(normalized_row) < 2:
                raise ValueError(f"Expected at least 2 values in feature row, got {normalized_row!r}")
            candidate_feature_tuples.append((normalized_row[0], normalized_row[-1]))
    return list(dict.fromkeys(candidate_feature_tuples))


def _populate_feature_cache_from_local_exports(
    feature_ref: NeuronpediaFeatureRef,
    *,
    export_roots: tuple[Path, ...],
    cache_dir: Path | None = None,
) -> tuple[int, Path] | None:
    for export_root in export_roots:
        activations_dir = export_root / feature_ref.model_id / feature_ref.layer / "activations"
        if not activations_dir.exists():
            continue
        for batch_path in sorted(activations_dir.glob("batch-*.jsonl.gz")):
            activation_rows = load_activation_batch_records(batch_path)
            matching_rows = [row for row in activation_rows if str(row.get("index")) == feature_ref.index]
            if not matching_rows:
                continue
            cache_path = write_cached_feature_activations(feature_ref, matching_rows, cache_dir=cache_dir)
            return len(matching_rows), cache_path
    return None


def prepare_local_explanation_backfill(
    cfg: NotebookHarnessConfig,
    *feature_groups: Any,
    cache_dir: Path | None = None,
    local_export_roots: Iterable[Path | str] | None = None,
    timeout_seconds: int = 60,
) -> LocalExplanationPreparationResult:
    """Collect top feature refs, inspect local explanation coverage, and prefetch activation cache rows."""

    feature_tuples = _feature_rows_to_layer_feature_tuples(feature_groups)
    feature_refs = feature_tuples_to_feature_refs(
        model_id=cfg.neuronpedia_model,
        source_set=cfg.neuronpedia_set,
        feature_tuples=feature_tuples,
        base_url=cfg.neuronpedia_base_url,
    )
    initial_statuses = check_local_explanation_coverage(
        feature_refs,
        local_db_url=cfg.local_neuronpedia_db_url,
        type_name=cfg.local_explanation_type_name,
    )
    resolved_export_roots = _resolve_local_export_roots(local_export_roots)
    prefetch_statuses: list[LocalExplanationPrefetchStatus] = []

    for explanation_status in initial_statuses:
        feature_ref = explanation_status.feature_ref
        if explanation_status.has_local_explanation:
            prefetch_statuses.append(
                LocalExplanationPrefetchStatus(
                    feature_ref=feature_ref,
                    explanation_count=explanation_status.explanation_count,
                    cache_ready=True,
                    cache_source="existing_explanation",
                )
            )
            continue

        feature_cache_path = cached_feature_activation_path(feature_ref, cache_dir=cache_dir)
        existing_batch_paths = candidate_cached_activation_batch_paths(feature_ref, cache_dir=cache_dir)
        had_cache_before = feature_cache_path.exists() or any(
            batch_path.exists() for batch_path in existing_batch_paths
        )

        if had_cache_before:
            try:
                cached_rows, resolved_cache_path = load_cached_feature_activations(
                    feature_ref,
                    cache_dir=cache_dir,
                    timeout_seconds=timeout_seconds,
                )
                prefetch_statuses.append(
                    LocalExplanationPrefetchStatus(
                        feature_ref=feature_ref,
                        explanation_count=explanation_status.explanation_count,
                        cache_ready=True,
                        cache_source="existing_cache",
                        cache_path=str(resolved_cache_path),
                        activation_rows=len(cached_rows),
                    )
                )
                continue
            except Exception:
                pass

        local_export_result = _populate_feature_cache_from_local_exports(
            feature_ref,
            export_roots=resolved_export_roots,
            cache_dir=cache_dir,
        )
        if local_export_result is not None:
            activation_rows, resolved_cache_path = local_export_result
            prefetch_statuses.append(
                LocalExplanationPrefetchStatus(
                    feature_ref=feature_ref,
                    explanation_count=explanation_status.explanation_count,
                    cache_ready=True,
                    cache_source="local_export_cache",
                    cache_path=str(resolved_cache_path),
                    activation_rows=activation_rows,
                )
            )
            continue

        try:
            cached_rows, resolved_cache_path = load_cached_feature_activations(
                feature_ref,
                cache_dir=cache_dir,
                timeout_seconds=timeout_seconds,
            )
            prefetch_statuses.append(
                LocalExplanationPrefetchStatus(
                    feature_ref=feature_ref,
                    explanation_count=explanation_status.explanation_count,
                    cache_ready=True,
                    cache_source="existing_cache" if had_cache_before else "downloaded_public_cache",
                    cache_path=str(resolved_cache_path),
                    activation_rows=len(cached_rows),
                )
            )
            continue
        except Exception as exc:
            prefetch_statuses.append(
                LocalExplanationPrefetchStatus(
                    feature_ref=feature_ref,
                    explanation_count=explanation_status.explanation_count,
                    cache_ready=False,
                    cache_source="unavailable",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return LocalExplanationPreparationResult(
        feature_refs=feature_refs,
        initial_statuses=initial_statuses,
        prefetch_statuses=prefetch_statuses,
        export_roots=tuple(str(root) for root in resolved_export_roots),
        cache_dir=str(cache_dir or default_np_cache_dir()),
    )


def phase_run_name(cfg: NotebookHarnessConfig, label: str) -> str:
    cleaned = label.lower().replace(" ", "_").replace("/", "_")
    return f"{cfg.experiment_name}_{cleaned}"


def render_prompt(prompt: str, tokenizer: Any, mode: PromptRenderMode) -> str:
    if mode == "plain":
        return prompt
    chat_method = "gemma_dataclass" if mode == "gemma_dataclass" else "apply_chat_template"
    return _chattify(prompt, tokenizer, chat_method)


def render_prompt_variants(prompt: str, tokenizer: Any) -> dict[str, str | None]:
    gemma_cfg = GemmaPromptConfig()
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    return {
        "plain": prompt,
        "apply_chat_template": gemma_cfg.apply_chat_template_fn(
            tokenizer,
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        if has_chat_template
        else None,
        "gemma_dataclass": gemma_cfg.model_chat_template_fn(prompt, tokenization_pattern="gemma-chat"),
    }


def _tokenize_rendered_prompt(tokenizer: Any, rendered_prompt: str, mode: PromptRenderMode | str) -> list[int]:
    add_special_tokens = mode == "plain"
    return cast(list[int], tokenizer(rendered_prompt, add_special_tokens=add_special_tokens)["input_ids"])


def _build_prompt_batch(tokenizer: Any, rendered_prompt: str, mode: PromptRenderMode, device: Any) -> dict[str, Any]:
    add_special_tokens = mode == "plain"
    encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}


def _topk_token_summaries(logits: torch.Tensor, tokenizer: Any, *, k: int = 5) -> list[dict[str, Any]]:
    logits = logits.float().cpu()
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(logits, k)
    summaries: list[dict[str, Any]] = []
    for token_id, logit in zip(topk.indices.tolist(), topk.values.tolist(), strict=False):
        summaries.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "logit": float(logit),
                "prob": float(probs[int(token_id)].item()),
            }
        )
    return summaries


def _get_prompt_debugger(module: Any) -> DebugGeneration:
    from interpretune.extensions.debug_generation import DebugGeneration

    debug_lm = getattr(module, "debug_lm", None)
    if debug_lm is None:
        debug_lm = DebugGeneration()
        debug_lm.connect(module)
    return cast(DebugGeneration, debug_lm)


def _normalize_target_token_for_prompt_mode(token: str, cfg: NotebookHarnessConfig) -> str:
    if not cfg.use_chat_template:
        return token
    normalized = token.lstrip(" ▁Ġ")
    return normalized or token


def resolve_target_tokens(cfg: NotebookHarnessConfig, tokenizer: Any) -> tuple[tuple[int, int], tuple[str, str]]:
    if cfg.target_tokens is not None:
        resolved_tokens = tuple(_normalize_target_token_for_prompt_mode(token, cfg) for token in cfg.target_tokens)
        resolved_ids = tuple(tokenizer.encode(token, add_special_tokens=False)[-1] for token in resolved_tokens)
        return cast(tuple[int, int], resolved_ids), cast(tuple[str, str], resolved_tokens)
    assert cfg.target_token_ids is not None
    decoded_tokens = tuple(tokenizer.decode([token_id]) for token_id in cfg.target_token_ids)
    return cfg.target_token_ids, cast(tuple[str, str], decoded_tokens)


def resolve_explicit_direction_tokens(
    cfg: NotebookHarnessConfig,
    tokenizer: Any,
) -> tuple[tuple[int, int], tuple[str, str]]:
    if cfg.explicit_direction_tokens is None:
        raise ValueError("explicit_direction_tokens must be provided for explicit embedding-difference mode")
    resolved_tokens = tuple(
        _normalize_target_token_for_prompt_mode(token, cfg) for token in cfg.explicit_direction_tokens
    )
    resolved_ids = tuple(tokenizer.encode(token, add_special_tokens=False)[-1] for token in resolved_tokens)
    return cast(tuple[int, int], resolved_ids), cast(tuple[str, str], resolved_tokens)


def resolve_graph_target_tokens(cfg: NotebookHarnessConfig, tokenizer: Any) -> tuple[list[int], list[str]]:
    if not cfg.is_debug_intervention_mode:
        raise ValueError("resolve_graph_target_tokens is only available in debug_intervention_pipelines mode")

    ids: list[int] = []
    labels: list[str] = []
    seen_ids: set[int] = set()
    for token in resolve_key_tokens(cfg):
        resolved = _normalize_target_token_for_prompt_mode(token, cfg)
        encoded = tokenizer.encode(resolved, add_special_tokens=False)
        if not encoded:
            continue
        token_id = int(encoded[-1])
        if token_id in seen_ids:
            continue
        seen_ids.add(token_id)
        ids.append(token_id)
        labels.append(resolved)
    if not ids:
        raise ValueError("Unable to resolve any debug graph target tokens from KEY_TOKENS")
    return ids, labels


def get_key_token_ids_and_labels(
    cfg: NotebookHarnessConfig,
    tokenizer: Any,
    *,
    include_bare_variants: bool = True,
) -> tuple[list[int], list[str]]:
    """Resolve key-token IDs and labels for reporting and logit displays.

    In chat mode this includes the single-token space-prefixed variant (for example
    ``▁Austin``) plus the bare completion token when available, even if the experiment config
    uses bare labels like ``Austin``.
    """
    candidates = _build_key_token_candidates(
        cfg,
        tokenizer,
        include_space_prefixed_variants=True,
        include_bare_variants=include_bare_variants,
    )
    return [entry["token_id"] for entry in candidates], [entry["label"] for entry in candidates]


def summarize_gap(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    target_a_id: int,
    target_b_id: int,
) -> tuple[float, float, float]:
    pre_gap = float((pre_logits[target_a_id] - pre_logits[target_b_id]).item())
    post_gap = float((post_logits[target_a_id] - post_logits[target_b_id]).item())
    return pre_gap, post_gap, post_gap - pre_gap


def configure_analysis(module: Any, graph_op: Any, scale_factor: float) -> None:
    module.circuit_tracer_cfg.intervention_value_source = "top_feature_activation_values"
    module.circuit_tracer_cfg.intervention_scale_factor = scale_factor
    module.circuit_tracer_cfg.intervention_constrained_layers = list(range(_resolve_model_layer_count(module)))
    module.circuit_tracer_cfg.intervention_apply_activation_function = False
    module.circuit_tracer_cfg.intervention_freeze_attention = None
    module.circuit_tracer_cfg.intervention_sparse = False
    module.circuit_tracer_cfg.intervention_return_activations = False
    module.analysis_cfg = AnalysisCfg(target_op=graph_op, ignore_manual=True, save_tokens=False)
    if not module.analysis_cfg.applied_to(module):
        init_analysis_cfgs(module, [module.analysis_cfg])


def _debug_intervention_artifact_name(
    cfg: NotebookHarnessConfig,
    feature_row: Sequence[int],
) -> str:
    feature_suffix = "_".join(str(int(value)) for value in feature_row)
    config_name = str(cfg.experiment_config_name or "manual").strip().replace(" ", "_")
    return f"{cfg.experiment_name}_{config_name}_{cfg.model_family}_{cfg.model_variant}_{feature_suffix}"


def _maybe_preserve_debug_intervention_artifacts(
    cfg: NotebookHarnessConfig,
    *,
    graph: Any,
    feature_row: Sequence[int],
    interventions: Sequence[Sequence[int | float]],
    baseline_activation_cache: torch.Tensor,
    intervention_activation_cache: torch.Tensor,
    baseline_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    graph_target_ids: Sequence[int],
    graph_target_tokens: Sequence[str],
    selected_feature_score: float,
    selected_feature_activation: float,
    report: Any,
    runtime_state: dict[str, Any] | None = None,
) -> Path | None:
    artifact_dir = resolve_artifact_output_dir(
        artifact_name=_debug_intervention_artifact_name(cfg, feature_row),
    )
    if artifact_dir is None:
        return None

    metadata = {
        "artifact_kind": "concept_direction_debug_validation",
        "experiment_name": cfg.experiment_name,
        "experiment_config_name": cfg.experiment_config_name,
        "analysis_mode": cfg.analysis_mode,
        "model_family": cfg.model_family,
        "model_variant": cfg.model_variant,
        "model_name": cfg.model_name,
        "transcoder_set": cfg.transcoder_set,
        "prompt": cfg.prompt,
        "prompt_render_mode": cfg.prompt_render_mode,
        "graph_target_ids": [int(token_id) for token_id in graph_target_ids],
        "graph_target_tokens": [str(token) for token in graph_target_tokens],
        "selected_feature_score": float(selected_feature_score),
        "selected_feature_activation": float(selected_feature_activation),
        "requested_constrained_feature_selection": [
            _serialize_constrained_feature_selection_ref(raw_ref)
            for raw_ref in (cfg.constrained_feature_selection_refs or ())
        ],
        "validation_tolerances": {
            "act_atol": cfg.debug_validation_act_atol,
            "act_rtol": cfg.debug_validation_act_rtol,
            "logit_atol": cfg.debug_validation_logit_atol,
            "logit_rtol": cfg.debug_validation_logit_rtol,
        },
        "runtime_state": runtime_state or {},
    }
    save_preserved_intervention_artifacts(
        artifact_dir,
        graph=graph,
        feature_row=feature_row,
        interventions=interventions,
        baseline_activation_cache=baseline_activation_cache,
        intervention_activation_cache=intervention_activation_cache,
        baseline_logits=baseline_logits,
        intervention_logits=intervention_logits,
        activation_atol=cfg.debug_validation_act_atol,
        activation_rtol=cfg.debug_validation_act_rtol,
        logit_atol=cfg.debug_validation_logit_atol,
        logit_rtol=cfg.debug_validation_logit_rtol,
        report=report,
        metadata=metadata,
    )
    return artifact_dir


@contextmanager
def maybe_zero_softcap(module: Any, cfg: NotebookHarnessConfig):
    if not cfg.enable_zero_softcap:
        yield
        return

    replacement_model = getattr(module, "replacement_model", None)
    zero_softcap = getattr(replacement_model, "zero_softcap", None)
    if callable(zero_softcap):
        zero_softcap_cm = zero_softcap()
        with cast(Any, zero_softcap_cm):
            yield
        return

    warning_flag = "_interpretune_zero_softcap_warning_emitted"
    if replacement_model is not None and not getattr(replacement_model, warning_flag, False):
        warnings.warn(
            "enable_zero_softcap was requested but the current replacement model does not expose zero_softcap(); "
            "continuing without it.",
            stacklevel=2,
        )
        setattr(replacement_model, warning_flag, True)
    yield


def _build_graph_analysis_inputs(
    cfg: NotebookHarnessConfig,
    tokenizer: Any,
    rendered_prompt: str,
    *,
    direction: torch.Tensor | None,
    group_a_ids: list[int] | None,
    group_b_ids: list[int] | None,
    attribution_target_device: torch.device | str | None = None,
    attribution_targets: Any | None = None,
    graph_call_kwargs: Mapping[str, Any] | None = None,
    analysis_batch_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    return _shared_build_graph_analysis_inputs(
        cfg,
        tokenizer,
        rendered_prompt,
        direction=direction,
        group_a_ids=group_a_ids,
        group_b_ids=group_b_ids,
        attribution_target_device=attribution_target_device,
        attribution_targets=attribution_targets,
        graph_call_kwargs=graph_call_kwargs,
        analysis_batch_kwargs=analysis_batch_kwargs,
    )


def _build_concept_graph_input_builder(
    cfg: NotebookHarnessConfig,
    direction: torch.Tensor,
    group_a_ids: list[int],
    group_b_ids: list[int],
) -> Callable[[Any, str], tuple[Any, dict[str, Any]]]:
    def _builder(tokenizer: Any, rendered_prompt: str) -> tuple[Any, dict[str, Any]]:
        return _build_graph_analysis_inputs(
            cfg,
            tokenizer,
            rendered_prompt,
            direction=direction,
            group_a_ids=group_a_ids,
            group_b_ids=group_b_ids,
        )

    return _builder


def _resolve_model_layer_count(module: Any) -> int:
    replacement_model = getattr(module, "replacement_model", None)
    model_cfg = getattr(replacement_model, "cfg", None)
    if model_cfg is not None:
        for attr_name in ("n_layers", "num_hidden_layers"):
            value = getattr(model_cfg, attr_name, None)
            if value is not None:
                return int(value)

    config = getattr(replacement_model, "config", None)
    for candidate in (config, getattr(config, "text_config", None) if config is not None else None):
        if candidate is None:
            continue
        value = getattr(candidate, "num_hidden_layers", None)
        if value is not None:
            return int(value)

    raise ValueError("Unable to resolve the replacement model layer count for debug intervention validation")


def _match_feature_row_index(active_features: torch.Tensor, feature_row: torch.Tensor) -> int:
    matches = (active_features == feature_row.reshape(1, 3)).all(dim=1).nonzero(as_tuple=False).reshape(-1)
    if matches.numel() != 1:
        raise ValueError(
            "Expected exactly one active-feature row to match the selected debug intervention feature; "
            f"found {int(matches.numel())} matches for {feature_row.tolist()}"
        )
    return int(matches.item())


def _rank_top_indices(values: torch.Tensor, top_k: int) -> torch.Tensor:
    flat_values = tensor_to_cpu(torch.as_tensor(values, dtype=torch.float32)).reshape(-1)
    if flat_values.numel() == 0 or top_k <= 0:
        return torch.empty((0,), dtype=torch.long)
    return torch.argsort(flat_values, descending=True)[: min(int(top_k), flat_values.numel())]


def _summarize_feature_row_deltas(
    feature_rows: torch.Tensor,
    baseline_values: torch.Tensor,
    post_values: torch.Tensor,
    expected_delta: torch.Tensor,
    *,
    top_k: int,
    ranking: Literal["abs_error", "expected_delta", "actual_delta"] = "abs_error",
) -> list[dict[str, Any]]:
    feature_rows_t = tensor_to_cpu(torch.as_tensor(feature_rows, dtype=torch.long)).reshape(-1, 3)
    baseline_t = tensor_to_cpu(torch.as_tensor(baseline_values, dtype=torch.float32)).reshape(-1)
    post_t = tensor_to_cpu(torch.as_tensor(post_values, dtype=torch.float32)).reshape(-1)
    expected_t = tensor_to_cpu(torch.as_tensor(expected_delta, dtype=torch.float32)).reshape(-1)

    count = feature_rows_t.shape[0]
    if not (baseline_t.shape[0] == count == post_t.shape[0] == expected_t.shape[0]):
        raise ValueError("Feature-row diagnostic inputs must all have matching lengths")
    if count == 0:
        return []

    actual_delta_t = post_t - baseline_t
    predicted_t = baseline_t + expected_t
    signed_error_t = post_t - predicted_t
    abs_error_t = signed_error_t.abs()
    if ranking == "expected_delta":
        rank_values = expected_t.abs()
    elif ranking == "actual_delta":
        rank_values = actual_delta_t.abs()
    else:
        rank_values = abs_error_t

    rows: list[dict[str, Any]] = []
    for display_rank, graph_index in enumerate(_rank_top_indices(rank_values, top_k).tolist(), start=1):
        layer, position, feature_id = (int(value) for value in feature_rows_t[graph_index].tolist())
        expected_delta_value = float(expected_t[graph_index].item())
        actual_delta_value = float(actual_delta_t[graph_index].item())
        abs_error_value = float(abs_error_t[graph_index].item())
        rows.append(
            {
                "rank": display_rank,
                "graph_index": int(graph_index),
                "layer": layer,
                "position": position,
                "feature_id": feature_id,
                "row": [layer, position, feature_id],
                "baseline_activation": float(baseline_t[graph_index].item()),
                "predicted_activation": float(predicted_t[graph_index].item()),
                "post_activation": float(post_t[graph_index].item()),
                "expected_delta": expected_delta_value,
                "actual_delta": actual_delta_value,
                "abs_error": abs_error_value,
                "signed_error": float(signed_error_t[graph_index].item()),
                "relative_abs_error": abs_error_value / max(abs(expected_delta_value), 1e-12),
                "sign_mismatch": bool(expected_delta_value * actual_delta_value < 0.0),
                "ranking": ranking,
                "rank_metric": float(rank_values[graph_index].item()),
            }
        )
    return rows


def _summarize_layer_error_rows(
    feature_rows: torch.Tensor,
    baseline_values: torch.Tensor,
    post_values: torch.Tensor,
    expected_delta: torch.Tensor,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    feature_rows_t = tensor_to_cpu(torch.as_tensor(feature_rows, dtype=torch.long)).reshape(-1, 3)
    baseline_t = tensor_to_cpu(torch.as_tensor(baseline_values, dtype=torch.float32)).reshape(-1)
    post_t = tensor_to_cpu(torch.as_tensor(post_values, dtype=torch.float32)).reshape(-1)
    expected_t = tensor_to_cpu(torch.as_tensor(expected_delta, dtype=torch.float32)).reshape(-1)

    count = feature_rows_t.shape[0]
    if not (baseline_t.shape[0] == count == post_t.shape[0] == expected_t.shape[0]):
        raise ValueError("Layer diagnostic inputs must all have matching lengths")
    if count == 0:
        return []

    actual_delta_t = post_t - baseline_t
    abs_error_t = (post_t - (baseline_t + expected_t)).abs()
    summaries: list[dict[str, Any]] = []
    for layer_value in sorted({int(layer) for layer in feature_rows_t[:, 0].tolist()}):
        layer_mask = feature_rows_t[:, 0] == layer_value
        layer_errors = abs_error_t[layer_mask]
        layer_expected = expected_t[layer_mask]
        layer_actual = actual_delta_t[layer_mask]
        sign_mismatches = ((layer_expected * layer_actual) < 0.0).sum().item()
        summaries.append(
            {
                "layer": int(layer_value),
                "feature_count": int(layer_mask.sum().item()),
                "max_abs_error": float(layer_errors.max().item()),
                "mean_abs_error": float(layer_errors.mean().item()),
                "max_abs_expected_delta": float(layer_expected.abs().max().item()),
                "mean_abs_expected_delta": float(layer_expected.abs().mean().item()),
                "max_abs_actual_delta": float(layer_actual.abs().max().item()),
                "mean_abs_actual_delta": float(layer_actual.abs().mean().item()),
                "sign_mismatch_count": int(sign_mismatches),
            }
        )
    summaries.sort(key=lambda entry: entry["max_abs_error"], reverse=True)
    return summaries[: min(int(top_k), len(summaries))]


def _summarize_logit_delta_rows(
    token_ids: Sequence[int],
    token_labels: Sequence[str],
    baseline_logits: torch.Tensor,
    post_logits: torch.Tensor,
    baseline_demeaned_logits: torch.Tensor,
    post_demeaned_logits: torch.Tensor,
    expected_delta: torch.Tensor,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    baseline_logits_t = tensor_to_cpu(torch.as_tensor(baseline_logits, dtype=torch.float32)).reshape(-1)
    post_logits_t = tensor_to_cpu(torch.as_tensor(post_logits, dtype=torch.float32)).reshape(-1)
    baseline_demeaned_t = tensor_to_cpu(torch.as_tensor(baseline_demeaned_logits, dtype=torch.float32)).reshape(-1)
    post_demeaned_t = tensor_to_cpu(torch.as_tensor(post_demeaned_logits, dtype=torch.float32)).reshape(-1)
    expected_t = tensor_to_cpu(torch.as_tensor(expected_delta, dtype=torch.float32)).reshape(-1)

    count = len(token_ids)
    if not (
        baseline_logits_t.shape[0]
        == count
        == post_logits_t.shape[0]
        == baseline_demeaned_t.shape[0]
        == post_demeaned_t.shape[0]
        == expected_t.shape[0]
    ):
        raise ValueError("Logit diagnostic inputs must all have matching lengths")
    if count == 0:
        return []

    actual_delta_t = post_demeaned_t - baseline_demeaned_t
    predicted_t = baseline_demeaned_t + expected_t
    signed_error_t = post_demeaned_t - predicted_t
    abs_error_t = signed_error_t.abs()

    rows: list[dict[str, Any]] = []
    for display_rank, token_index in enumerate(_rank_top_indices(abs_error_t, top_k).tolist(), start=1):
        expected_delta_value = float(expected_t[token_index].item())
        actual_delta_value = float(actual_delta_t[token_index].item())
        abs_error_value = float(abs_error_t[token_index].item())
        rows.append(
            {
                "rank": display_rank,
                "token_id": int(token_ids[token_index]),
                "token": str(token_labels[token_index]),
                "baseline_logit": float(baseline_logits_t[token_index].item()),
                "post_logit": float(post_logits_t[token_index].item()),
                "baseline_demeaned_logit": float(baseline_demeaned_t[token_index].item()),
                "post_demeaned_logit": float(post_demeaned_t[token_index].item()),
                "expected_delta": expected_delta_value,
                "actual_delta": actual_delta_value,
                "abs_error": abs_error_value,
                "signed_error": float(signed_error_t[token_index].item()),
                "relative_abs_error": abs_error_value / max(abs(expected_delta_value), 1e-12),
                "sign_mismatch": bool(expected_delta_value * actual_delta_value < 0.0),
            }
        )
    return rows


def _summarize_same_feature_rows(
    feature_rows: torch.Tensor,
    baseline_values: torch.Tensor,
    post_values: torch.Tensor,
    expected_delta: torch.Tensor,
    *,
    layer: int,
    feature_id: int,
) -> list[dict[str, Any]]:
    feature_rows_t = tensor_to_cpu(torch.as_tensor(feature_rows, dtype=torch.long)).reshape(-1, 3)
    baseline_t = tensor_to_cpu(torch.as_tensor(baseline_values, dtype=torch.float32)).reshape(-1)
    post_t = tensor_to_cpu(torch.as_tensor(post_values, dtype=torch.float32)).reshape(-1)
    expected_t = tensor_to_cpu(torch.as_tensor(expected_delta, dtype=torch.float32)).reshape(-1)

    count = feature_rows_t.shape[0]
    if not (baseline_t.shape[0] == count == post_t.shape[0] == expected_t.shape[0]):
        raise ValueError("Same-feature diagnostic inputs must all have matching lengths")

    actual_delta_t = post_t - baseline_t
    rows: list[dict[str, Any]] = []
    for graph_index in ((feature_rows_t[:, 0] == layer) & (feature_rows_t[:, 2] == feature_id)).nonzero(
        as_tuple=False
    ).reshape(-1).tolist():
        rows.append(
            {
                "graph_index": int(graph_index),
                "row": [int(value) for value in feature_rows_t[graph_index].tolist()],
                "position": int(feature_rows_t[graph_index, 1].item()),
                "baseline_activation": float(baseline_t[graph_index].item()),
                "post_activation": float(post_t[graph_index].item()),
                "expected_delta": float(expected_t[graph_index].item()),
                "actual_delta": float(actual_delta_t[graph_index].item()),
                "abs_error": float(
                    (post_t[graph_index] - (baseline_t[graph_index] + expected_t[graph_index])).abs().item()
                ),
            }
        )
    rows.sort(key=lambda entry: (entry["position"], entry["graph_index"]))
    return rows


def _serialize_intervention_call_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            serialized[key] = tensor_fingerprint(value)
        elif isinstance(value, range):
            serialized[key] = {"kind": "range", "start": value.start, "stop": value.stop, "step": value.step}
        else:
            serialized[key] = value
    return serialized


def _summarize_graph_input_tokens(
    tokenizer: Any,
    rendered_prompt: str,
    prompt_render_mode: PromptRenderMode,
    graph_inputs: Any,
) -> dict[str, Any]:
    rendered_prompt_token_ids = _tokenize_rendered_prompt(tokenizer, rendered_prompt, prompt_render_mode)
    if isinstance(graph_inputs, torch.Tensor):
        graph_input_token_ids = tensor_to_cpu(torch.as_tensor(graph_inputs, dtype=torch.long)).reshape(-1).tolist()
        graph_input_source = "graph_result.input_tokens"
    else:
        graph_input_token_ids = _tokenize_rendered_prompt(tokenizer, str(graph_inputs), prompt_render_mode)
        graph_input_source = type(graph_inputs).__name__

    first_difference_index = next(
        (
            index
            for index, (rendered_token_id, graph_token_id) in enumerate(
                zip(rendered_prompt_token_ids, graph_input_token_ids, strict=False)
            )
            if rendered_token_id != graph_token_id
        ),
        None,
    )
    if first_difference_index is None and len(rendered_prompt_token_ids) != len(graph_input_token_ids):
        first_difference_index = min(len(rendered_prompt_token_ids), len(graph_input_token_ids))

    return {
        "graph_input_source": graph_input_source,
        "rendered_prompt_token_count": len(rendered_prompt_token_ids),
        "graph_input_token_count": len(graph_input_token_ids),
        "graph_inputs_match_rendered_prompt": rendered_prompt_token_ids == graph_input_token_ids,
        "first_difference_index": first_difference_index,
        "rendered_prompt_token_ids": rendered_prompt_token_ids,
        "graph_input_token_ids": graph_input_token_ids,
    }


def _parse_constrained_feature_selection_ref(
    raw_ref: ConstrainedFeatureSelectionRef,
    cfg: NotebookHarnessConfig,
) -> tuple[int, int]:
    ref_value = raw_ref.ref
    layer_identifier: str
    feature_index: str
    if not isinstance(ref_value, str):
        model_id = ref_value[0]
        source_set = ref_value[1]
        layer_number = int(ref_value[2])
        feature_index_value = int(ref_value[3])
        if model_id != cfg.neuronpedia_model:
            raise ValueError(
                f"Constrained feature selection tuple {raw_ref!r} targets model {model_id}, "
                f"expected {cfg.neuronpedia_model}."
            )
        if source_set != cfg.neuronpedia_set:
            raise ValueError(
                f"Constrained feature selection tuple {raw_ref!r} targets source set {source_set}, "
                f"expected {cfg.neuronpedia_set}."
            )
        return int(layer_number), int(feature_index_value)

    if "://" in ref_value:
        feature_ref = parse_feature_url(ref_value)
        if feature_ref.model_id != cfg.neuronpedia_model:
            raise ValueError(
                f"Constrained feature selection ref {ref_value!r} targets model {feature_ref.model_id}, "
                f"expected {cfg.neuronpedia_model}."
            )
        layer_identifier = feature_ref.layer
        feature_index = feature_ref.index
    else:
        parts = [part for part in ref_value.split("/") if part]
        if len(parts) == 3:
            model_id, layer_identifier, feature_index = parts
            if model_id != cfg.neuronpedia_model:
                raise ValueError(
                    f"Constrained feature selection ref {ref_value!r} targets model {model_id}, "
                    f"expected {cfg.neuronpedia_model}."
                )
        elif len(parts) == 2:
            layer_identifier, feature_index = parts
        else:
            raise ValueError(
                "Constrained feature selection refs must be full Neuronpedia URLs or 'model/layer/index' or "
                "'layer/index' strings. "
                f"Got: {ref_value!r}"
            )

    layer_parts = str(layer_identifier).split("-", 1)
    if len(layer_parts) == 2 and layer_parts[1] != cfg.neuronpedia_set:
        raise ValueError(
            f"Constrained feature selection ref {raw_ref!r} targets source set {layer_parts[1]}, "
            f"expected {cfg.neuronpedia_set}."
        )
    layer_number = int(str(layer_identifier).split("-", 1)[0])
    return layer_number, int(feature_index)


def _build_feature_selection_spec(
    cfg: NotebookHarnessConfig,
    active_features: Any,
) -> FeatureSelectionSpec | None:
    return _shared_build_feature_selection_spec(cfg, active_features)


def _extract_top_features_with_optional_filter(
    module: Any,
    cfg: NotebookHarnessConfig,
    top_payload: dict[str, Any],
    *,
    top_n: int,
) -> tuple[Any, list[tuple[int, int, int]]]:
    return _shared_extract_top_features_with_optional_filter(module, cfg, top_payload, top_n=top_n)


def _reduce_top_features_result_to_single_feature(result: Any) -> tuple[Any, int]:
    feature_ids = torch.as_tensor(getattr(result, "top_feature_ids", []), dtype=torch.long).reshape(-1, 3)
    feature_scores = torch.as_tensor(getattr(result, "top_feature_scores", []), dtype=torch.float32).reshape(-1)
    activation_values = getattr(result, "top_feature_activation_values", None)
    activation_tensor = (
        None if activation_values is None else torch.as_tensor(activation_values, dtype=torch.float32).reshape(-1)
    )

    candidate_count = int(feature_ids.shape[0])
    if candidate_count == 0:
        raise ValueError(
            "debug_intervention_pipelines mode expected at least one selected feature row after filtering."
        )
    if candidate_count == 1:
        return result, candidate_count

    if feature_scores.shape[0] == feature_ids.shape[0]:
        selected_index = int(torch.argmax(feature_scores.abs()).item())
    else:
        selected_index = 0

    selected_indices = torch.tensor([selected_index], dtype=torch.long)
    setattr(result, "top_feature_ids", feature_ids.index_select(0, selected_indices).detach().cpu())
    if feature_scores.shape[0] == feature_ids.shape[0]:
        setattr(result, "top_feature_scores", feature_scores.index_select(0, selected_indices).detach().cpu())
    if activation_tensor is not None and activation_tensor.shape[0] == feature_ids.shape[0]:
        setattr(
            result,
            "top_feature_activation_values",
            activation_tensor.index_select(0, selected_indices).detach().cpu(),
        )
    return result, candidate_count


def run_initial_sanity_check(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    """Run initial sanity check for the configured intervention prompt and return logit analysis.

    Uses ``cfg.prompt`` (which incorporates PROMPT_OVERRIDE from YAML configs)
    rather than the concept pair's hardcoded prompts.
    """
    raw_prompt = cfg.prompt

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "initial_sanity_check"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        rendered = render_prompt(raw_prompt, tokenizer, cfg.prompt_render_mode)
        enc = _build_prompt_batch(tokenizer, rendered, cfg.prompt_render_mode, module.device)

        with maybe_zero_softcap(module, cfg), torch.inference_mode():
            gen_out = module.model.generate(
                **enc,
                max_new_tokens=1,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
            )

        gen_text = tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True)
        first_logits = gen_out.logits[0][0]
        probs = torch.softmax(first_logits.float(), dim=-1)

        key_analysis: list[dict[str, Any]] = []
        for candidate in _build_key_token_candidates(cfg, tokenizer):
            token_id = int(candidate["token_id"])
            entry: dict[str, Any] = dict(candidate)
            entry["logit"] = float(first_logits[token_id].item())
            entry["prob"] = float(probs[token_id].item())
            key_analysis.append(entry)

        # Sort key tokens by logit magnitude descending
        key_analysis.sort(key=lambda e: abs(e.get("logit", 0.0)), reverse=True)

        top_id = int(first_logits.argmax(dim=-1).item())
        top_token = tokenizer.decode([top_id])
        top_logit = float(first_logits[top_id].item())
        top_prob = float(probs[top_id].item())

        return {
            "prompt_style": "chat" if cfg.use_chat_template else "plain",
            "rendered_prompt": rendered[:400],
            "generated_text": gen_text,
            "key_tokens": key_analysis,
            "top1_token": top_token,
            "top1_id": top_id,
            "top1_logit": top_logit,
            "top1_prob": top_prob,
        }


def collect_baseline_path_debug(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    """Compare the notebook sanity-check path against replacement-model baseline logits."""

    raw_prompt = cfg.prompt
    generation_kwargs = {
        "max_new_tokens": 1,
        "do_sample": False,
        "output_logits": True,
        "return_dict_in_generate": True,
    }

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "baseline_path_debug"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        rendered_prompt = render_prompt(raw_prompt, tokenizer, cfg.prompt_render_mode)
        prompt_debug = _get_prompt_debugger(module).collect_prompt_debug_info(
            raw_prompt,
            rendered_sequences=rendered_prompt,
            add_special_tokens=cfg.prompt_render_mode == "plain",
        )[0]

        prompt_batch = _build_prompt_batch(tokenizer, rendered_prompt, cfg.prompt_render_mode, module.device)
        prompt_input_ids = cast(torch.Tensor, prompt_batch["input_ids"])[0]

        with maybe_zero_softcap(module, cfg), torch.inference_mode():
            forward_out = module.model(**prompt_batch)
            forward_logits = forward_out.logits if hasattr(forward_out, "logits") else forward_out
            forward_last = forward_logits[0, -1].float().cpu()
            gen_out = module.model.generate(**prompt_batch, **generation_kwargs)
            generate_first = gen_out.logits[0][0].float().cpu()

            replacement_string = last_token_logits(
                module.replacement_model.get_activations(rendered_prompt)[0]
            ).float().cpu()
            replacement_tokens = last_token_logits(
                module.replacement_model.get_activations(prompt_input_ids)[0]
            ).float().cpu()

        return {
            "prompt_render_mode": cfg.prompt_render_mode,
            "generation_kwargs": generation_kwargs,
            "prompt_debug": prompt_debug,
            "render_variants": render_prompt_variants(raw_prompt, tokenizer),
            "generated_text": tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True),
            "baseline_sources": {
                "forward_last": _topk_token_summaries(forward_last, tokenizer),
                "generate_first": _topk_token_summaries(generate_first, tokenizer),
                "replacement_from_string": _topk_token_summaries(replacement_string, tokenizer),
                "replacement_from_tokens": _topk_token_summaries(replacement_tokens, tokenizer),
            },
            "max_abs_diffs": {
                "forward_vs_generate": float((forward_last - generate_first).abs().max().item()),
                "forward_vs_replacement_string": float((forward_last - replacement_string).abs().max().item()),
                "forward_vs_replacement_tokens": float((forward_last - replacement_tokens).abs().max().item()),
                "generate_vs_replacement_string": float((generate_first - replacement_string).abs().max().item()),
                "generate_vs_replacement_tokens": float((generate_first - replacement_tokens).abs().max().item()),
            },
        }


def _build_render_variant_equalities(
    render_variants: dict[str, str | None],
    render_variant_token_ids: dict[str, list[int]],
) -> dict[str, bool | None]:
    """Build equality comparisons between render variants, using None when apply_chat_template is unavailable."""
    has_chat = render_variants.get("apply_chat_template") is not None
    return {
        "apply_chat_template_vs_dataclass": (
            render_variants["apply_chat_template"] == render_variants["gemma_dataclass"] if has_chat else None
        ),
        "apply_chat_template_vs_dataclass_token_ids": (
            render_variant_token_ids["apply_chat_template"] == render_variant_token_ids["gemma_dataclass"]
            if has_chat
            else None
        ),
    }


def run_tokenizer_verification(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "tokenizer_verification"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        render_variants = render_prompt_variants(cfg.prompt, tokenizer)
        selected_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        add_special_tokens = cfg.prompt_render_mode == "plain"
        enc = tokenizer(selected_prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
        resolved_target_ids, resolved_target_tokens = resolve_target_tokens(cfg, tokenizer)
        render_variant_token_ids = {
            mode_name: _tokenize_rendered_prompt(tokenizer, rendered_prompt, mode_name)
            for mode_name, rendered_prompt in render_variants.items()
            if rendered_prompt is not None
        }
        report: dict[str, Any] = {
            "groups": {},
            "key_tokens": {},
            "prompt_token_count": int(enc["input_ids"].shape[-1]),
            "module_type": type(module).__name__,
            "prompt_render_mode": cfg.prompt_render_mode,
            "render_variants": render_variants,
            "render_variant_token_ids": render_variant_token_ids,
            "render_variant_tokens": {
                mode_name: tokenizer.convert_ids_to_tokens(token_ids)
                for mode_name, token_ids in render_variant_token_ids.items()
            },
            "selected_prompt_preview": selected_prompt[:400],
            "selected_prompt_token_ids": enc["input_ids"][0].tolist(),
            "selected_prompt_tokens": tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist()),
            "render_variant_equalities": _build_render_variant_equalities(render_variants, render_variant_token_ids),
            "target_tokens": {
                "group_a": {"id": resolved_target_ids[0], "decoded": resolved_target_tokens[0]},
                "group_b": {"id": resolved_target_ids[1], "decoded": resolved_target_tokens[1]},
            },
        }
        for label, tokens in [
            (cfg.concept_pair.group_a_name, cfg.concept_pair.group_a_tokens),
            (cfg.concept_pair.group_b_name, cfg.concept_pair.group_b_tokens),
        ]:
            entries = []
            for token in tokens:
                ids = tokenizer.encode(token, add_special_tokens=False)
                entries.append({"token": token, "ids": ids, "decoded": tokenizer.decode(ids)})
            report["groups"][label] = entries
        for token in resolve_key_tokens(cfg):
            ids = tokenizer.encode(token, add_special_tokens=False)
            report["key_tokens"][token] = {"ids": ids, "decoded": tokenizer.decode(ids)}
        report["resolved_key_token_candidates"] = _build_key_token_candidates(cfg, tokenizer)
        return report


def compute_embed_direction(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    if cfg.is_debug_intervention_mode:
        raise ValueError("compute_embed_direction is not available in debug_intervention_pipelines mode")

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "embed_direction"),
        **cfg.session_kwargs,
    ) as (_, module, _):
        if cfg.uses_explicit_embedding_difference:
            explicit_tokens = cast(tuple[str, str], cfg.explicit_direction_tokens)
            group_a_tokens, group_b_tokens = ([explicit_tokens[0]], [explicit_tokens[1]])
            concept_label = cfg.analysis_concept_label
        else:
            group_a_tokens = cfg.concept_pair.group_a_tokens
            group_b_tokens = (
                []
                if cfg.analysis_direction_mode_name == "single_group"
                else cfg.concept_pair.group_b_tokens
            )
            concept_label = cfg.concept_pair.concept_label

        analysis_kwargs: dict[str, Any] = {
            "concept_group_a": group_a_tokens,
            "concept_label": concept_label,
            "concept_direction_mode": cfg.analysis_direction_mode_name,
        }
        if group_b_tokens:
            analysis_kwargs["concept_group_b"] = group_b_tokens

        embed_result = cast(
            Any,
            it.concept_direction(
                module,
                it.AnalysisBatch(**analysis_kwargs),
                NULL_BATCH,
                0,
            ),
        )
        return {
            "direction": tensor_to_cpu(embed_result.concept_direction),
            "group_a_ids": list(embed_result.concept_group_a_token_ids),
            "group_b_ids": list(embed_result.concept_group_b_token_ids),
        }


def run_pipeline(
    cfg: NotebookHarnessConfig,
    direction: torch.Tensor,
    label: str,
    *,
    scale_factor: float,
    top_n: int,
    group_a_ids: list[int],
    group_b_ids: list[int],
) -> dict[str, Any]:
    return _shared_run_pipeline(
        cfg,
        label,
        scale_factor=scale_factor,
        top_n=top_n,
        build_graph_analysis_inputs=_build_concept_graph_input_builder(cfg, direction, group_a_ids, group_b_ids),
    )


def run_direct_projection_pipeline(
    cfg: NotebookHarnessConfig,
    direction: torch.Tensor,
    label: str,
    *,
    scale_factor: float,
) -> dict[str, Any]:
    def _inject_concept_direction(raw_value: Any) -> Any:
        if isinstance(raw_value, torch.Tensor):
            return raw_value
        if isinstance(raw_value, Mapping):
            payload = dict(raw_value)
            payload.setdefault("intervention_tensor", direction)
            payload.setdefault(
                "scale_factor",
                cfg.direct_projection_intervention_scale_factor
                if cfg.direct_projection_intervention_scale_factor is not None
                else scale_factor,
            )
            if cfg.direct_projection_intervention_mode is not None:
                payload.setdefault("mode", cfg.direct_projection_intervention_mode)
            if cfg.direct_projection_intervention_use_intervention_tensor_as_basis is not None:
                payload.setdefault(
                    "use_intervention_tensor_as_basis",
                    cfg.direct_projection_intervention_use_intervention_tensor_as_basis,
                )
            return payload
        if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
            return [_inject_concept_direction(item) for item in raw_value]
        raise ValueError(
            "ANALYSIS.direct_projection.interventions values must be mappings or sequences of mappings."
        )

    direct_projection_interventions = dict(cfg.direct_projection_interventions or {})
    use_explicit_interventions = bool(direct_projection_interventions)

    def _build_direct_projection_analysis_batch(
        rendered_prompt: str,
        target_a_id: int,
        target_b_id: int,
        resolved_scale_factor: float,
    ) -> Any:
        payload: dict[str, Any] = {
            "prompts": [rendered_prompt],
            "concept_direction": direction,
            "logit_target_ids": torch.tensor([target_a_id], dtype=torch.long),
            "concept_group_a_token_ids": [target_a_id],
            "concept_group_b_token_ids": [target_b_id],
        }
        if use_explicit_interventions:
            payload["interventions"] = {
                str(hook_pattern): _inject_concept_direction(raw_value)
                for hook_pattern, raw_value in direct_projection_interventions.items()
            }
        else:
            payload["concept_cache_key"] = cfg.store_concept_cache_key
            payload["intervention_hook_pattern"] = (
                cfg.direct_projection_intervention_hook_pattern or cfg.store_concept_cache_key
            )
            if cfg.direct_projection_intervention_mode is not None:
                payload["intervention_mode"] = cfg.direct_projection_intervention_mode
            if cfg.direct_projection_intervention_scale_factor is not None:
                payload["intervention_scale_factor"] = cfg.direct_projection_intervention_scale_factor
            else:
                payload["direction_scale_factor"] = resolved_scale_factor
            if cfg.direct_projection_intervention_use_intervention_tensor_as_basis is not None:
                payload["intervention_use_intervention_tensor_as_basis"] = (
                    cfg.direct_projection_intervention_use_intervention_tensor_as_basis
                )
        return it.AnalysisBatch(**payload)

    return _shared_run_direct_projection_pipeline(
        cfg,
        label,
        scale_factor=scale_factor,
        build_analysis_batch=_build_direct_projection_analysis_batch,
    )


def run_direct_projection_scale_sweep(
    cfg: NotebookHarnessConfig,
    direction: torch.Tensor,
) -> list[dict[str, Any]]:
    """Run a scale sweep using the shared direct-projection helper."""
    return [
        run_direct_projection_pipeline(
            cfg,
            direction,
            "direct_proj_sweep",
            scale_factor=scale_factor,
        )
        for scale_factor in cfg.scale_factor_sweep
    ]


def build_all_prompts(cfg: NotebookHarnessConfig, tokenizer: Any) -> list[tuple[str, str, str]]:
    prompts: list[tuple[str, str, str]] = []
    for entity_name, expected_answer in cfg.concept_pair.group_a_entities:
        raw = _build_classification_prompt(entity_name, cfg.concept_pair.classification_question)
        prompts.append(
            (render_prompt(raw, tokenizer, cfg.prompt_render_mode), expected_answer, cfg.concept_pair.group_a_name)
        )
    for entity_name, expected_answer in cfg.concept_pair.group_b_entities:
        raw = _build_classification_prompt(entity_name, cfg.concept_pair.classification_question)
        prompts.append(
            (render_prompt(raw, tokenizer, cfg.prompt_render_mode), expected_answer, cfg.concept_pair.group_b_name)
        )
    return prompts


def _score_expected_answer(
    tokenizer: Any,
    example_logits: torch.Tensor,
    expected_answer: str,
    target_a_id: int,
    target_b_id: int,
) -> tuple[int, float, bool, int, list[int], list[str]]:
    expected_id = tokenizer.encode(expected_answer, add_special_tokens=False)[-1]
    other_id = target_b_id if expected_id == target_a_id else target_a_id
    topk_ids = torch.topk(example_logits, 10).indices.tolist()
    topk_tokens = [tokenizer.decode([token_id]) for token_id in topk_ids]
    correct = expected_id in topk_ids
    rank = topk_ids.index(expected_id) if correct else -1
    logit_diff = float((example_logits[expected_id] - example_logits[other_id]).item())
    return expected_id, logit_diff, correct, rank, topk_ids, topk_tokens


def compute_store_direction_manual(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    if not cfg.supports_store_direction:
        raise ValueError("compute_store_direction_manual is only available in concept_pair mode")

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "store_direction_manual"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        model_backend = getattr(module, "_model_backend", None)
        assert model_backend is not None, "experiment session module must expose _model_backend"
        device = next(module.model.parameters()).device
        all_prompts = build_all_prompts(cfg, tokenizer)
        (target_a_id, target_b_id), _ = resolve_target_tokens(cfg, tokenizer)
        latent_states: list[torch.Tensor] = []
        prediction_info: dict[str, Any] = {"examples": [], "n_correct": 0}
        prediction_examples = cast(list[dict[str, Any]], prediction_info["examples"])
        with maybe_zero_softcap(module, cfg):
            for prompt_text, expected_answer, group in all_prompts:
                add_special_tokens = cfg.prompt_render_mode == "plain"
                enc = tokenizer(prompt_text, return_tensors="pt", padding=False, add_special_tokens=add_special_tokens)
                batch_dev = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in dict(enc).items()
                }
                with torch.no_grad():
                    logits, cache = model_backend.fwd_w_cache(
                        model=module.model,
                        batch=batch_dev,
                        names_filter="unembed.hook_in",
                    )
                last_pos = logits.shape[1] - 1
                example_logits = logits[0, last_pos]
                cache_tensor = torch.as_tensor(cache["unembed.hook_in"])
                latent_states.append(tensor_to_cpu(cache_tensor[0, last_pos]))
                expected_id, _, correct, rank, topk_ids, topk_tokens = _score_expected_answer(
                    tokenizer,
                    example_logits,
                    expected_answer,
                    target_a_id,
                    target_b_id,
                )
                if correct:
                    prediction_info["n_correct"] = int(prediction_info["n_correct"]) + 1
                prediction_examples.append(
                    {
                        "group": group,
                        "expected": expected_answer,
                        "correct": correct,
                        "rank": rank,
                        "top1": topk_tokens[0] if topk_tokens else None,
                        "top5": topk_tokens[:5],
                        "top10_ids": topk_ids,
                        "prompt": prompt_text,
                        "input_ids": enc["input_ids"][0].tolist(),
                        "tokens": tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist()),
                        "attention_mask": enc["attention_mask"][0].tolist() if "attention_mask" in enc else None,
                    }
                )
        stacked = torch.stack(latent_states)
        n_a = len(cfg.concept_pair.group_a_entities)
        n_b = len(cfg.concept_pair.group_b_entities)
        group_ids = torch.cat([torch.zeros(n_a, dtype=torch.long), torch.ones(n_b, dtype=torch.long)])
        group_names = ([cfg.concept_pair.group_a_name] * n_a) + ([cfg.concept_pair.group_b_name] * n_b)
        store_result = cast(
            Any,
            it.concept_direction(
                module,
                it.AnalysisBatch(
                    concept_latent_state=[stacked],
                    concept_group_id=[group_ids],
                    concept_group_name=[group_names],
                    concept_example_weight=[torch.ones(len(stacked), dtype=torch.float32)],
                    concept_label=cfg.concept_pair.concept_label,
                    concept_direction_mode=cfg.analysis_direction_mode_name,
                    concept_group_a_name=cfg.concept_pair.group_a_name,
                    concept_group_b_name=cfg.concept_pair.group_b_name,
                ),
                NULL_BATCH,
                0,
            ),
        )
        return {
            "direction": tensor_to_cpu(store_result.concept_direction),
            "group_a_ids": [
                tokenizer.encode(token, add_special_tokens=False)[-1] for token in cfg.concept_pair.group_a_tokens
            ],
            "group_b_ids": [
                tokenizer.encode(token, add_special_tokens=False)[-1] for token in cfg.concept_pair.group_b_tokens
            ],
            "prediction_info": prediction_info,
            "n_total": len(all_prompts),
        }


def construct_concept_pair_analysis_inputs(
    cfg: NotebookHarnessConfig,
    module: Any,
    tokenizer: Any,
    model_backend: Any,
    device: Any,
    target_a_id: int,
    target_b_id: int,
) -> tuple[
    list[dict[str, torch.Tensor]],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    dict[str, Any],
    list[tuple[str, str, str]],
]:
    """Build per-example forward-pass caches, answer indices, labels, and logit diffs.

    For each entity prompt in the concept pair, runs a forward pass with caching, scores
    the expected answer, and collects the cached activations and metadata needed by
    ``execute_concept_latent_extraction_ops``.

    Returns:
        (cached_batches, answer_indices, orig_labels, logit_diffs, prediction_info, all_prompts)
    """
    all_prompts = build_all_prompts(cfg, tokenizer)

    cached_batches: list[dict[str, torch.Tensor]] = []
    answer_indices: list[torch.Tensor] = []
    orig_labels: list[torch.Tensor] = []
    logit_diffs: list[torch.Tensor] = []
    prediction_info: dict[str, Any] = {"examples": [], "n_correct": 0}

    for prompt_text, expected_answer, group in all_prompts:
        add_special_tokens = cfg.prompt_render_mode == "plain"
        enc = tokenizer(prompt_text, return_tensors="pt", padding=False, add_special_tokens=add_special_tokens)
        batch_dev = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in dict(enc).items()
        }
        with torch.no_grad():
            logits, cache = model_backend.fwd_w_cache(
                model=module.model,
                batch=batch_dev,
                names_filter=cfg.store_concept_cache_key,
            )

        last_pos = int(logits.shape[1] - 1)
        example_logits = logits[0, last_pos]
        expected_id, example_logit_diff, correct, rank, topk_ids, topk_tokens = _score_expected_answer(
            tokenizer,
            example_logits,
            expected_answer,
            target_a_id,
            target_b_id,
        )
        if correct:
            prediction_info["n_correct"] += 1

        group_id = 0 if group == cfg.concept_pair.group_a_name else 1
        cached_batches.append(
            {cfg.store_concept_cache_key: torch.as_tensor(cache[cfg.store_concept_cache_key]).detach().cpu()}
        )
        answer_indices.append(torch.tensor([last_pos], dtype=torch.long))
        orig_labels.append(torch.tensor([group_id], dtype=torch.long))
        logit_diffs.append(torch.tensor([example_logit_diff], dtype=torch.float32))
        prediction_info["examples"].append(
            {
                "group": group,
                "expected": expected_answer,
                "expected_token_id": expected_id,
                "correct": correct,
                "rank": rank,
                "top1": topk_tokens[0] if topk_tokens else None,
                "top5": topk_tokens[:5],
                "top10_ids": topk_ids,
                "logit_diff": example_logit_diff,
                "prompt": prompt_text,
                "input_ids": enc["input_ids"][0].tolist(),
                "tokens": tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist()),
                "attention_mask": enc["attention_mask"][0].tolist() if "attention_mask" in enc else None,
            }
        )

    return cached_batches, answer_indices, orig_labels, logit_diffs, prediction_info, all_prompts


def execute_concept_latent_extraction_ops(
    module: Any,
    cfg: NotebookHarnessConfig,
    cached_batches: list[dict[str, torch.Tensor]],
    answer_indices: list[torch.Tensor],
    orig_labels: list[torch.Tensor],
    logit_diffs: list[torch.Tensor],
    n_prompts: int,
    *,
    extraction_mode: StoreLatentExtractionMode = "answer_position_state",
) -> list[Any]:
    """Execute ``extract_concept_latent_state`` and ``extract_concept_latent_examples`` ops.

    Wraps the AnalysisCfg setup, AnalysisInputs construction, and per-batch op execution
    loop.  The *extraction_mode* parameter selects which variant of latent extraction to
    use:

    - ``"answer_position_state"``: Default — extracts the hidden state at the answer
      token position.
    - ``"context_enhanced"``: Extracts the answer-position state *and* the immediately
      preceding token's context, then projects the scaled answer state into that context.
      The scale factor is read from ``cfg.context_enhanced_scale`` (default 1.0).
    """
    if extraction_mode not in ("answer_position_state", "context_enhanced"):
        raise ValueError(f"Unsupported extraction_mode: {extraction_mode}")

    extraction_cfg = AnalysisCfg(
        name="concept_latent_rows",
        target_op=[it.extract_concept_latent_state, it.extract_concept_latent_examples],
        ignore_manual=True,
    )
    analysis_inputs = AnalysisInputs(
        store=SimpleNamespace(
            cache=cached_batches,
            answer_indices=answer_indices,
            orig_labels=orig_labels,
            logit_diffs=logit_diffs,
        )
    )

    extracted_batches: list[Any] = []
    context_enhanced = extraction_mode == "context_enhanced"
    for batch_idx in range(n_prompts):
        extracted_batches.append(
            execute_analysis_op(
                module,
                batch=None,
                batch_idx=batch_idx,
                analysis_batch=it.AnalysisBatch(
                    concept_group_a_label_ids=[0],
                    concept_group_b_label_ids=[1],
                    concept_group_a_name=cfg.concept_pair.group_a_name,
                    concept_group_b_name=cfg.concept_pair.group_b_name,
                    concept_cache_key=cfg.store_concept_cache_key,
                    concept_correct_only=cfg.store_concept_correct_only,
                    concept_weight_by_logit_diff=cfg.store_weight_by_logit_diff,
                ),
                analysis_cfg=extraction_cfg,
                analysis_inputs=analysis_inputs,
                context_enhanced=context_enhanced,
                context_scale=cfg.context_enhanced_scale,
            )
        )

    return extracted_batches


def prepare_extracted_concept_example_tensors(
    extracted_batches: list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Prepare tensors from extracted AnalysisBatch rows for ``it.concept_direction``.

    This intermediate step is needed because ``it.extract_concept_latent_examples`` emits
    one AnalysisBatch *per example* (each containing a single-row latent state, group id,
    weight, and group name).  The ``it.concept_direction`` op, however, expects a *single*
    stacked tensor of all concept latent states along with parallel group-id, weight, and
    group-name sequences — the format produced when the two ops are chained via the
    AnalysisStore in a standard analysis session (where the store accumulates rows
    automatically across batches).

    In experiment-harness code we run the ops manually in a per-example loop, so the
    store-level accumulation does not happen.  This function bridges that gap by
    concatenating the per-example rows into the stacked format ``it.concept_direction``
    requires.

    .. note::

       *IG-7 follow-up:* Investigate whether ``extract_concept_latent_examples`` and
       ``concept_direction`` can be adapted (or an additional intermediate op added) so
       their input/output contracts allow direct pipelining via a composite op alias,
       eliminating the need for this manual stacking step.

    Returns:
        (stacked_latent_states, group_ids, example_weights, group_name_rows)
    """
    latent_rows = [batch.concept_latent_state for batch in extracted_batches if batch.concept_latent_state is not None]
    group_id_rows = [batch.concept_group_id for batch in extracted_batches if batch.concept_group_id is not None]
    weight_rows = [
        batch.concept_example_weight for batch in extracted_batches if batch.concept_example_weight is not None
    ]
    group_name_rows: list[str] = []
    for batch in extracted_batches:
        if batch.concept_group_name is not None:
            group_name_rows.extend(list(batch.concept_group_name))

    stacked = torch.cat([tensor_to_cpu(row) for row in latent_rows], dim=0)
    group_ids = torch.cat([tensor_to_cpu(row) for row in group_id_rows], dim=0)
    example_weights = torch.cat([tensor_to_cpu(row) for row in weight_rows], dim=0)
    return stacked, group_ids, example_weights, group_name_rows


def compute_store_direction(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    if not cfg.supports_store_direction:
        raise ValueError("compute_store_direction is only available in concept_pair mode")

    if cfg.store_latent_extraction_mode not in ("answer_position_state", "context_enhanced"):
        raise ValueError(f"Unsupported store_latent_extraction_mode: {cfg.store_latent_extraction_mode}")

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "store_direction"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        model_backend = getattr(module, "_model_backend", None)
        assert model_backend is not None, "experiment session module must expose _model_backend"
        device = next(module.model.parameters()).device
        (target_a_id, target_b_id), _ = resolve_target_tokens(cfg, tokenizer)

        with maybe_zero_softcap(module, cfg):
            cached_batches, answer_indices, orig_labels, logit_diffs, prediction_info, all_prompts = (
                construct_concept_pair_analysis_inputs(
                    cfg,
                    module,
                    tokenizer,
                    model_backend,
                    device,
                    target_a_id,
                    target_b_id,
                )
            )

        extracted_batches = execute_concept_latent_extraction_ops(
            module,
            cfg,
            cached_batches,
            answer_indices,
            orig_labels,
            logit_diffs,
            len(all_prompts),
            extraction_mode=cfg.store_latent_extraction_mode,
        )

        stacked, group_ids, example_weights, group_name_rows = prepare_extracted_concept_example_tensors(
            extracted_batches,
        )

        store_result = cast(
            Any,
            it.concept_direction(
                module,
                it.AnalysisBatch(
                    concept_latent_state=[stacked],
                    concept_group_id=[group_ids],
                    concept_group_name=[group_name_rows],
                    concept_example_weight=[example_weights],
                    concept_label=cfg.concept_pair.concept_label,
                    concept_direction_mode=cfg.analysis_direction_mode_name,
                    concept_group_a_name=cfg.concept_pair.group_a_name,
                    concept_group_b_name=cfg.concept_pair.group_b_name,
                ),
                NULL_BATCH,
                0,
            ),
        )
        return {
            "direction": tensor_to_cpu(store_result.concept_direction),
            "group_a_ids": [
                tokenizer.encode(token, add_special_tokens=False)[-1] for token in cfg.concept_pair.group_a_tokens
            ],
            "group_b_ids": [
                tokenizer.encode(token, add_special_tokens=False)[-1] for token in cfg.concept_pair.group_b_tokens
            ],
            "prediction_info": prediction_info,
            "n_total": len(all_prompts),
            "n_latent_rows": int(stacked.shape[0]),
            "manual_reference_fn": "compute_store_direction_manual",
            "store_latent_extraction_mode": cfg.store_latent_extraction_mode,
        }


def run_scale_sweep(
    cfg: NotebookHarnessConfig,
    direction: torch.Tensor,
    group_a_ids: list[int],
    group_b_ids: list[int],
) -> list[dict[str, Any]]:
    return _shared_run_scale_sweep(
        cfg,
        build_graph_analysis_inputs=_build_concept_graph_input_builder(cfg, direction, group_a_ids, group_b_ids),
    )


def collect_feature_pool(
    cfg: NotebookHarnessConfig,
    direction: torch.Tensor,
    group_a_ids: list[int],
    group_b_ids: list[int],
    *,
    top_n: int,
) -> dict[str, Any]:
    return _shared_collect_feature_pool(
        cfg,
        top_n=top_n,
        build_graph_analysis_inputs=_build_concept_graph_input_builder(cfg, direction, group_a_ids, group_b_ids),
    )


def run_ablations(
    cfg: NotebookHarnessConfig,
    feature_pool: dict[str, Any],
    pre_logits_ref: torch.Tensor,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[int, torch.Tensor]]:
    if cfg.is_debug_intervention_mode:
        raise ValueError("run_ablations is not available in debug_intervention_pipelines mode")

    abl_groups = {
        "baseline": {
            label: float(torch.softmax(pre_logits_ref.float(), dim=-1)[token_id].item())
            for label, token_id in zip(feature_pool["key_labels"][:3], feature_pool["key_ids"][:3])
        }
    }
    abl_logit_diffs = {
        "baseline": float(pre_logits_ref[feature_pool["target_a_id"]] - pre_logits_ref[feature_pool["target_b_id"]])
    }
    results: dict[int, torch.Tensor] = {}
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "progressive_ablation"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        with maybe_zero_softcap(module, cfg):
            for n_value in cfg.ablation_n_list:
                if n_value > int(feature_pool["feature_ids"].shape[0]):
                    continue
                intervention_result = cast(
                    Any,
                    it.feature_intervention_forward(
                        module,
                        it.AnalysisBatch(
                            prompts=[rendered_prompt],
                            top_feature_ids=feature_pool["feature_ids"][:n_value],
                            top_feature_scores=feature_pool["feature_scores"][:n_value],
                            top_feature_activation_values=feature_pool["feature_activations"][:n_value] * 0.0,
                            logit_target_ids=torch.tensor([feature_pool["target_a_id"]], dtype=torch.long),
                        ),
                        NULL_BATCH,
                        0,
                    ),
                )
                post_logits = tensor_to_cpu(intervention_result.post_intervention_logits)
                probs = torch.softmax(post_logits, dim=-1)
                label = f"top-{n_value}"
                abl_groups[label] = {
                    name: float(probs[token_id].item())
                    for name, token_id in zip(feature_pool["key_labels"][:3], feature_pool["key_ids"][:3])
                }
                abl_logit_diffs[label] = float(
                    post_logits[feature_pool["target_a_id"]] - post_logits[feature_pool["target_b_id"]]
                )
                results[n_value] = post_logits
    return abl_groups, abl_logit_diffs, results


def run_sign_aware(
    cfg: NotebookHarnessConfig,
    feature_pool: dict[str, Any],
    pre_logits_ref: torch.Tensor,
) -> dict[str, Any]:
    if cfg.is_debug_intervention_mode:
        raise ValueError("run_sign_aware is not available in debug_intervention_pipelines mode")

    feature_ids = feature_pool["feature_ids"]
    feature_scores = feature_pool["feature_scores"]
    feature_activations = feature_pool["feature_activations"]
    positive_mask = feature_activations > 0
    negative_mask = feature_activations < 0
    result: dict[str, Any] = {
        "positive_features": feature_ids[positive_mask],
        "negative_features": feature_ids[negative_mask],
        "positive_scores": feature_scores[positive_mask],
        "negative_scores": feature_scores[negative_mask],
        "positive_activations": feature_activations[positive_mask],
        "negative_activations": feature_activations[negative_mask],
        "messages": [],
    }
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "sign_aware"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        with maybe_zero_softcap(module, cfg):
            if len(result["positive_features"]) > 0:
                n_pos = min(cfg.top_n, len(result["positive_features"]))
                pos_intervention = cast(
                    Any,
                    it.feature_intervention_forward(
                        module,
                        it.AnalysisBatch(
                            prompts=[rendered_prompt],
                            top_feature_ids=result["positive_features"][:n_pos],
                            top_feature_scores=result["positive_scores"][:n_pos],
                            top_feature_activation_values=result["positive_activations"][:n_pos]
                            * cfg.default_scale_factor,
                            logit_target_ids=torch.tensor([feature_pool["target_a_id"]], dtype=torch.long),
                        ),
                        NULL_BATCH,
                        0,
                    ),
                )
                result["positive_post_logits"] = tensor_to_cpu(pos_intervention.post_intervention_logits)
            else:
                result["messages"].append(
                    "No positive-activation features were available for the current feature pool."
                )
            if len(result["negative_features"]) > 0:
                n_neg = min(cfg.top_n, len(result["negative_features"]))
                neg_intervention = cast(
                    Any,
                    it.feature_intervention_forward(
                        module,
                        it.AnalysisBatch(
                            prompts=[rendered_prompt],
                            top_feature_ids=result["negative_features"][:n_neg],
                            top_feature_scores=result["negative_scores"][:n_neg],
                            top_feature_activation_values=result["negative_activations"][:n_neg] * 0.0,
                            logit_target_ids=torch.tensor([feature_pool["target_a_id"]], dtype=torch.long),
                        ),
                        NULL_BATCH,
                        0,
                    ),
                )
                result["negative_post_logits"] = tensor_to_cpu(neg_intervention.post_intervention_logits)
            else:
                result["messages"].append(
                    "No negative-activation features were available for the current feature pool."
                )
    result["pre_logits_ref"] = pre_logits_ref
    return result


def run_debug_intervention_validation(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    return _shared_run_debug_intervention_validation(cfg)


def run_direction_probes(
    cfg: NotebookHarnessConfig,
    embed_direction: torch.Tensor,
    store_direction: torch.Tensor,
) -> dict[str, Any]:
    if not cfg.supports_store_direction:
        raise ValueError("run_direction_probes is only available in concept_pair mode")

    from interpretune.analysis.backends.circuit_tracer import CircuitTracerAnalysisBackend

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "direction_probes"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        backend = CircuitTracerAnalysisBackend()
        embedding_weight = backend.get_embedding_weight(module).float().detach()
        unembed = embedding_weight.T if embedding_weight.shape[0] > embedding_weight.shape[1] else embedding_weight
        probe_results: dict[str, Any] = {}
        for label, direction in [("Embed", embed_direction), ("Store", store_direction)]:
            direction_dev = direction.to(unembed.device)
            rows = []
            group_a_projections = []
            group_b_projections = []
            for token in cfg.concept_pair.group_a_tokens:
                token_id = tokenizer.encode(token, add_special_tokens=False)[-1]
                projection = float(torch.dot(unembed[:, token_id].float(), direction_dev).item())
                group_a_projections.append(projection)
                rows.append({"token": token, "projection": projection, "group": "A"})
            for token in cfg.concept_pair.group_b_tokens:
                token_id = tokenizer.encode(token, add_special_tokens=False)[-1]
                projection = float(torch.dot(unembed[:, token_id].float(), direction_dev).item())
                group_b_projections.append(projection)
                rows.append({"token": token, "projection": projection, "group": "B"})
            mean_b = None if not group_b_projections else sum(group_b_projections) / len(group_b_projections)
            probe_results[label] = {
                "rows": rows,
                "mean_a": sum(group_a_projections) / len(group_a_projections),
                "mean_b": mean_b,
            }
        return probe_results


def display_tokenizer_target_summary(report: Mapping[str, Any]) -> None:
    target_tokens = cast(Mapping[str, Any], report.get("target_tokens", {}))
    group_a = cast(Mapping[str, Any], target_tokens.get("group_a", {}))
    group_b = cast(Mapping[str, Any], target_tokens.get("group_b", {}))
    if group_a or group_b:
        print(
            "Target tokens: "
            f"A={group_a.get('id')} ({group_a.get('decoded')!r}), "
            f"B={group_b.get('id')} ({group_b.get('decoded')!r})"
        )

    for group_name, entries in cast(Mapping[str, Any], report.get("groups", {})).items():
        print(f"\n{group_name}:")
        for entry in entries:
            print(f"  {entry['token']}: ids={entry['ids']}, decoded={entry['decoded']!r}")


def display_direction_probe_results(probe_results: Mapping[str, Any]) -> None:
    for label, payload in probe_results.items():
        print(f"\n{label} direction probes:")
        print(f"{'Token':<15} {'Projection':>12} {'Group':>8}")
        print(f"{'-' * 15} {'-' * 12} {'-' * 8}")
        for row in payload["rows"]:
            print(f"{row['token']:<15} {row['projection']:>12.4f} {row['group']:>8}")
        mean_b = payload.get("mean_b")
        if mean_b is None:
            print(f"Mean A: {payload['mean_a']:.4f}, Mean B: n/a, Separation: n/a")
            continue
        separation = payload["mean_a"] - mean_b
        print(f"Mean A: {payload['mean_a']:.4f}, Mean B: {mean_b:.4f}, Separation: {separation:.4f}")


def collect_summary(
    cfg: NotebookHarnessConfig,
    results: Mapping[str, Any],
    *,
    config_path: str | Path,
    work_root_removed: bool,
) -> dict[str, Any]:
    summary = build_shared_summary_record(
        cfg,
        config_path=config_path,
        work_root_removed=work_root_removed,
    )
    summary["concept_pair"] = cfg.concept_pair.name
    summary["analysis_concept_label"] = cfg.analysis_concept_label

    if "embed_pipeline" in results:
        summary["embed_gap_delta"] = results["embed_pipeline"]["gap_delta"]
    if "store_pipeline" in results:
        summary["store_gap_delta"] = results["store_pipeline"]["gap_delta"]
    if "comparison" in results:
        summary["cosine_similarity"] = results["comparison"]["cosine_similarity"]
        summary["feature_jaccard"] = results["comparison"]["feature_jaccard"]
    if "store_direction_result" in results:
        summary["prediction_correct"] = results["store_direction_result"]["prediction_info"]["n_correct"]
        summary["prediction_total"] = results["store_direction_result"]["n_total"]
    if "debug_validation" in results:
        summary["debug_validation_passed"] = results["debug_validation"]["all_passed"]
        summary["debug_activation_max_abs_error"] = results["debug_validation"]["activation_max_abs_error"]
        summary["debug_logit_max_abs_error"] = results["debug_validation"]["logit_max_abs_error"]
        summary["debug_selected_feature"] = list(results["debug_validation"]["selected_feature"])
    return summary
