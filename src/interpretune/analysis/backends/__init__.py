"""Model backends for analysis operations.

Provides the ``ModelBackend`` protocol, shared intervention helpers, and backend implementations
for different model execution frameworks (TransformerLens, nnsight, etc.).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Callable, NamedTuple, Protocol, TypeAlias, runtime_checkable

import torch

from interpretune.analysis.backends.hook_mapping import SUBHOOK_SUFFIXES
from interpretune.protocol import NamesFilter


class InterventionSpec(NamedTuple):
    """Specification for a single hook-point intervention.

    Attributes:
        intervention_tensor: Intervention tensor with any shape broadcast-compatible with the
            targeted hook-point slice at the intervention position. This is not restricted to
            ``(d_model,)`` and may instead match higher-rank activations such as
            ``(n_heads, d_head)`` or latent feature activations.
        mode: ``"replace"`` overwrites the activation at the target position with the tensor;
              ``"add"`` adds ``intervention_tensor * scale_factor`` to the activation;
              ``"project"`` replaces the activation with a projection result. By default, the
              current hook input is projected onto the span of ``intervention_tensor``, so the
              intervention tensor acts as the projection basis. When
              ``use_intervention_tensor_as_basis`` is ``False``, the direction is reversed and
              ``intervention_tensor`` is projected onto the span of the current hook input.
        scale_factor: Scalar multiplier applied to *intervention_tensor* before the intervention
            (not used in ``"replace"`` mode and applied to the projected activation in
            ``"project"`` mode).
        use_intervention_tensor_as_basis: Controls which vector defines the projection basis in
            ``"project"`` mode. ``True`` means project the current hook input onto the span of
            ``intervention_tensor``. ``False`` means project ``intervention_tensor`` onto the
            span of the current hook input instead.
    """

    intervention_tensor: torch.Tensor
    mode: str = "replace"
    scale_factor: float = 1.0
    use_intervention_tensor_as_basis: bool = True


InterventionValue: TypeAlias = Any


HOOK_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("hook_in", "hook_resid_pre"),
    ("hook_out", "hook_resid_post"),
    ("attn.hook_in", "hook_attn_in"),
    ("attn.hook_out", "hook_attn_out", "hook_resid_mid"),
    ("attn.o.hook_in", "attn.hook_z"),
    ("attn.q.hook_in", "hook_q_input"),
    ("attn.k.hook_in", "hook_k_input"),
    ("attn.v.hook_in", "hook_v_input"),
    ("attn.q.hook_out", "hook_q"),
    ("attn.k.hook_out", "hook_k"),
    ("attn.v.hook_out", "hook_v"),
    ("mlp.hook_in", "hook_mlp_in"),
    ("mlp.hook_out", "hook_mlp_out"),
    ("embed.hook_out", "hook_embed"),
    ("pos_embed.hook_out", "hook_pos_embed"),
    ("attn.hook_pattern", "attn.hook_attention_weights"),
    ("attn.hook_hidden_states", "attn.hook_result"),
    ("ln1.hook_out", "ln1.hook_normalized", "ln1.hook_scale"),
    ("ln2.hook_out", "ln2.hook_normalized", "ln2.hook_scale"),
)


@dataclass(frozen=True)
class InterventionDict(Mapping[str, tuple[InterventionSpec, ...]]):
    """Canonical mapping from resolved hook names to intervention specs.

    Keys are concrete hook-point names with wildcards already expanded. Values are ordered tuples of intervention specs
    to apply sequentially at that hook.
    """

    hook_map: dict[str, tuple[InterventionSpec, ...]]

    def __getitem__(self, key: str) -> tuple[InterventionSpec, ...]:
        return self.hook_map[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.hook_map)

    def __len__(self) -> int:
        return len(self.hook_map)

    def items(self):
        return self.hook_map.items()

    def keys(self):
        return self.hook_map.keys()

    def values(self):
        return self.hook_map.values()

    @classmethod
    def from_mapping(
        cls,
        hook_map: Mapping[str, InterventionSpec | Sequence[InterventionSpec]],
    ) -> InterventionDict:
        return cls(
            {
                hook_name: (
                    tuple(specs)
                    if isinstance(specs, Sequence) and not isinstance(specs, InterventionSpec)
                    else (specs,)
                )
                for hook_name, specs in hook_map.items()
            }
        )


def _coerce_single_intervention_spec(
    value: InterventionSpec | torch.Tensor | Mapping[str, Any],
    *,
    default_mode: str = "replace",
    default_scale_factor: float = 1.0,
) -> InterventionSpec:
    if isinstance(value, InterventionSpec):
        return value
    if isinstance(value, torch.Tensor):
        return InterventionSpec(
            intervention_tensor=value,
            mode=default_mode,
            scale_factor=default_scale_factor,
        )
    if isinstance(value, Mapping):
        if "intervention_tensor" not in value:
            raise ValueError("Intervention mapping entries must include an 'intervention_tensor' field")
        return InterventionSpec(
            intervention_tensor=torch.as_tensor(value["intervention_tensor"]),
            mode=str(value.get("mode", default_mode)),
            scale_factor=float(value.get("scale_factor", default_scale_factor)),
            use_intervention_tensor_as_basis=bool(value.get("use_intervention_tensor_as_basis", True)),
        )
    raise TypeError(f"Unsupported intervention value type: {type(value)!r}")


def _coerce_shared_intervention_specs(
    value: InterventionValue,
    *,
    default_mode: str = "replace",
    default_scale_factor: float = 1.0,
) -> tuple[InterventionSpec, ...]:
    if isinstance(value, (InterventionSpec, torch.Tensor, Mapping)):
        return (
            _coerce_single_intervention_spec(
                value,
                default_mode=default_mode,
                default_scale_factor=default_scale_factor,
            ),
        )

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(
            _coerce_single_intervention_spec(item, default_mode=default_mode, default_scale_factor=default_scale_factor)
            for item in value
        )

    raise TypeError(f"Unsupported intervention value type: {type(value)!r}")


def get_intervention_target_shape(activation: torch.Tensor) -> tuple[int, ...]:
    """Return the per-example shape targeted by last-token interventions."""

    if activation.ndim < 2:
        raise ValueError(
            "Intervention target activations must include a sequence dimension so the last-token slice can be addressed"
        )
    return tuple(activation.shape[2:]) if activation.ndim > 2 else tuple()


def _ensure_shape_compatible(
    intervention_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    hook_name: str,
) -> None:
    try:
        broadcast_shape = torch.broadcast_shapes(target_shape, intervention_shape)
    except RuntimeError as exc:
        raise ValueError(
            "Intervention tensor shape "
            f"{intervention_shape} is not compatible with hook '{hook_name}' "
            f"target shape {target_shape}"
        ) from exc

    if broadcast_shape != target_shape:
        raise ValueError(
            "Intervention tensor shape "
            f"{intervention_shape} is not compatible with hook '{hook_name}' "
            f"target shape {target_shape}"
        )


def _validate_intervention_spec(
    spec: InterventionSpec,
    target_shape: tuple[int, ...],
    hook_name: str,
) -> InterventionSpec:
    tensor = torch.as_tensor(spec.intervention_tensor)
    _ensure_shape_compatible(tuple(tensor.shape), target_shape, hook_name)
    if spec.mode not in {"replace", "add", "project"}:
        raise ValueError(f"Unknown intervention mode: {spec.mode!r}")
    return InterventionSpec(
        intervention_tensor=tensor,
        mode=spec.mode,
        scale_factor=spec.scale_factor,
        use_intervention_tensor_as_basis=spec.use_intervention_tensor_as_basis,
    )


def expand_intervention_patterns(
    patterns: Sequence[str],
    available_hook_map: Mapping[str, str],
) -> dict[str, list[str]]:
    """Expand raw hook-name patterns to ordered lists of concrete hook names."""
    alias_lookup = {alias_name: alias_group for alias_group in HOOK_ALIAS_GROUPS for alias_name in alias_group}

    def _split_subhook_suffix(pattern: str) -> tuple[str, str]:
        parts = pattern.split(".")
        for index, part in enumerate(parts):
            if part in SUBHOOK_SUFFIXES:
                return ".".join(parts[:index]), "." + ".".join(parts[index:])
        return pattern, ""

    def _pattern_variants(pattern: str) -> tuple[str, ...]:
        base_pattern, subhook_suffix = _split_subhook_suffix(pattern)
        base_name = base_pattern
        prefix = ""
        if base_pattern.startswith("blocks."):
            parts = base_pattern.split(".", 2)
            if len(parts) == 3:
                prefix = f"{parts[0]}.{parts[1]}."
                base_name = parts[2]

        variants = [pattern]
        for alias_name in alias_lookup.get(base_name, (base_name,)):
            variants.append(f"{prefix}{alias_name}{subhook_suffix}")
        return tuple(dict.fromkeys(variants))

    expanded: dict[str, list[str]] = {}
    for pattern in patterns:
        if "*" not in pattern:
            matched_actual = None
            for candidate_pattern in _pattern_variants(pattern):
                matched_actual = available_hook_map.get(candidate_pattern)
                if matched_actual is not None:
                    break
            if matched_actual is None:
                raise ValueError(f"Intervention pattern '{pattern}' did not match any available hook names")
            expanded[pattern] = [matched_actual]
            continue

        matched: list[str] = []
        seen: set[str] = set()
        for candidate_pattern in _pattern_variants(pattern):
            regex = re.compile("^" + re.escape(candidate_pattern).replace(r"\*", ".*") + "$")
            for candidate_name, actual_name in available_hook_map.items():
                if regex.fullmatch(candidate_name) and actual_name not in seen:
                    matched.append(actual_name)
                    seen.add(actual_name)
        if not matched:
            raise ValueError(f"Intervention pattern '{pattern}' did not match any available hook names")
        expanded[pattern] = matched
    return expanded


def _split_tensor_across_matches(
    tensor: torch.Tensor,
    matched_hooks: Sequence[str],
    hook_shapes: Mapping[str, tuple[int, ...]],
    *,
    default_mode: str,
    default_scale_factor: float,
) -> list[tuple[InterventionSpec, ...]] | None:
    if len(matched_hooks) <= 1:
        return None

    target_shapes = [hook_shapes[hook_name] for hook_name in matched_hooks]
    unique_target_shapes = {shape for shape in target_shapes}
    if len(unique_target_shapes) != 1:
        return None

    target_shape = target_shapes[0]
    if tensor.shape[:1] != (len(matched_hooks),) or tensor.ndim != len(target_shape) + 1:
        return None

    per_hook_specs: list[tuple[InterventionSpec, ...]] = []
    for hook_name, hook_tensor in zip(matched_hooks, tensor, strict=True):
        spec = InterventionSpec(
            intervention_tensor=hook_tensor,
            mode=default_mode,
            scale_factor=default_scale_factor,
        )
        per_hook_specs.append((_validate_intervention_spec(spec, hook_shapes[hook_name], hook_name),))
    return per_hook_specs


def _expand_intervention_value_for_matches(
    raw_value: InterventionValue,
    matched_hooks: Sequence[str],
    hook_shapes: Mapping[str, tuple[int, ...]],
    *,
    default_mode: str = "replace",
    default_scale_factor: float = 1.0,
) -> list[tuple[InterventionSpec, ...]]:
    if isinstance(raw_value, Mapping) and "intervention_tensors" in raw_value:
        per_hook_tensors = raw_value["intervention_tensors"]
        if not isinstance(per_hook_tensors, Sequence) or isinstance(per_hook_tensors, (str, bytes)):
            raise TypeError("intervention_tensors must be a sequence")
        if len(per_hook_tensors) != len(matched_hooks):
            raise ValueError(
                "intervention_tensors length must match the number of resolved hook points for wildcard interventions"
            )
        shared_mode = str(raw_value.get("mode", default_mode))
        shared_scale = float(raw_value.get("scale_factor", default_scale_factor))
        shared_basis = bool(raw_value.get("use_intervention_tensor_as_basis", True))
        return [
            (
                _validate_intervention_spec(
                    InterventionSpec(
                        torch.as_tensor(tensor),
                        mode=shared_mode,
                        scale_factor=shared_scale,
                        use_intervention_tensor_as_basis=shared_basis,
                    ),
                    hook_shapes[hook_name],
                    hook_name,
                ),
            )
            for hook_name, tensor in zip(matched_hooks, per_hook_tensors, strict=True)
        ]

    if isinstance(raw_value, torch.Tensor):
        split_specs = _split_tensor_across_matches(
            raw_value,
            matched_hooks,
            hook_shapes,
            default_mode=default_mode,
            default_scale_factor=default_scale_factor,
        )
        if split_specs is not None:
            return split_specs

    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes, torch.Tensor, Mapping)):
        if len(matched_hooks) > 1 and len(raw_value) == len(matched_hooks):
            return [
                (
                    _validate_intervention_spec(
                        _coerce_single_intervention_spec(
                            item,
                            default_mode=default_mode,
                            default_scale_factor=default_scale_factor,
                        ),
                        hook_shapes[hook_name],
                        hook_name,
                    ),
                )
                for hook_name, item in zip(matched_hooks, raw_value, strict=True)
            ]

    shared_specs = _coerce_shared_intervention_specs(
        raw_value,
        default_mode=default_mode,
        default_scale_factor=default_scale_factor,
    )
    return [
        tuple(_validate_intervention_spec(spec, hook_shapes[hook_name], hook_name) for spec in shared_specs)
        for hook_name in matched_hooks
    ]


def build_intervention_dict(
    interventions: InterventionDict | Mapping[str, InterventionValue],
    expanded_matches: Mapping[str, Sequence[str]],
    hook_shapes: Mapping[str, tuple[int, ...]],
    *,
    default_mode: str = "replace",
    default_scale_factor: float = 1.0,
) -> InterventionDict:
    """Canonicalize raw intervention inputs into a resolved :class:`InterventionDict`."""

    if isinstance(interventions, InterventionDict):
        return interventions

    resolved: dict[str, list[InterventionSpec]] = {}
    for pattern, raw_value in interventions.items():
        matched_hooks = list(expanded_matches.get(pattern, ()))
        if not matched_hooks:
            raise ValueError(f"Intervention pattern '{pattern}' did not match any available hook names")

        per_hook_specs = _expand_intervention_value_for_matches(
            raw_value,
            matched_hooks,
            hook_shapes,
            default_mode=default_mode,
            default_scale_factor=default_scale_factor,
        )
        for hook_name, specs in zip(matched_hooks, per_hook_specs, strict=True):
            resolved.setdefault(hook_name, []).extend(specs)

    return InterventionDict({hook_name: tuple(specs) for hook_name, specs in resolved.items()})


def resolve_interventions(
    *,
    analysis_batch: Any,
    resolve_field: Callable[[str], Any],
    load_json_field: Callable[[str], Any],
    kwargs: Mapping[str, Any] | None = None,
    default_hook_qualifier: str = "unembed.hook_in",
) -> InterventionDict | dict[str, Any]:
    """Resolve explicit or shorthand intervention inputs into a standardized payload mapping.

    Explicit ``interventions`` or ``interventions_json`` mappings take precedence. Otherwise,
    shorthand op inputs are assembled into a raw intervention payload keyed by the resolved hook
    qualifier. Shape canonicalization into :class:`InterventionDict` still happens in the backend
    after concrete hook shapes are known.
    """

    def _first_defined(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    kwargs = kwargs or {}
    batch_get = getattr(analysis_batch, "get", lambda *_args, **_kwargs: None)

    raw_interventions = load_json_field("interventions_json")
    if raw_interventions is None:
        raw_interventions = resolve_field("interventions")

    if raw_interventions is not None:
        if isinstance(raw_interventions, InterventionDict):
            return raw_interventions
        if not isinstance(raw_interventions, dict):
            raise TypeError("interventions_json/interventions must resolve to a mapping or InterventionDict")
        return raw_interventions

    hook_qualifier = str(
        _first_defined(
            resolve_field("intervention_hook_pattern"),
            batch_get("concept_cache_key"),
            default_hook_qualifier,
        )
    )
    intervention_mode = _first_defined(resolve_field("intervention_mode"), kwargs.get("mode"))
    scale_factor = _first_defined(
        resolve_field("intervention_scale_factor"),
        batch_get("direction_scale_factor"),
        kwargs.get("scale_factor"),
        1.0,
    )
    use_intervention_tensor_as_basis = _first_defined(
        resolve_field("intervention_use_intervention_tensor_as_basis"),
        kwargs.get("use_intervention_tensor_as_basis"),
        True,
    )

    intervention_tensor = resolve_field("intervention_tensor")
    intervention_tensors = load_json_field("intervention_tensors_json")
    if intervention_tensors is None:
        intervention_tensors = resolve_field("intervention_tensors")

    if intervention_tensor is None and intervention_tensors is None:
        concept_direction = batch_get("concept_direction")
        if concept_direction is None:
            raise ValueError(
                "model_fwd_intervention requires either explicit interventions or shorthand intervention tensor inputs"
            )
        intervention_tensor = concept_direction
        intervention_mode = intervention_mode or "add"

    payload: dict[str, Any] = {
        "mode": str(intervention_mode or "replace"),
        "scale_factor": float(scale_factor),
        "use_intervention_tensor_as_basis": bool(use_intervention_tensor_as_basis),
    }
    if intervention_tensors is not None:
        payload["intervention_tensors"] = intervention_tensors
    else:
        payload["intervention_tensor"] = intervention_tensor

    return {hook_qualifier: payload}


def apply_intervention_to_last_token(
    value: torch.Tensor,
    spec: InterventionSpec,
    *,
    last_pos: int,
) -> torch.Tensor:
    """Apply one intervention spec to the last-token slice of an activation tensor.

    The existing hook value is treated as the projection input and
    ``spec.intervention_tensor`` is treated as the projection target. In ``"project"`` mode,
    the target defines the default projection basis: the input is projected onto the span of
    the intervention tensor. When ``spec.use_intervention_tensor_as_basis`` is ``False``, the
    direction is reversed and the intervention tensor is projected onto the span of the input.
    """

    input_value = value[:, last_pos, ...]
    target = torch.as_tensor(spec.intervention_tensor, device=input_value.device, dtype=input_value.dtype)

    if spec.mode == "replace":
        value[:, last_pos, ...] = target
        return value

    if spec.mode == "add":
        value[:, last_pos, ...] = input_value + target * spec.scale_factor
        return value

    if spec.mode != "project":
        raise ValueError(f"Unknown intervention mode: {spec.mode!r}")

    input_float = input_value.to(dtype=torch.float32)
    target_float = torch.broadcast_to(target, input_value.shape[1:]).to(dtype=torch.float32)
    keepdim_axes = tuple(range(1, input_float.ndim))

    if spec.use_intervention_tensor_as_basis:
        basis = target_float
        denom = basis.pow(2).sum().clamp_min(1e-12)
        coeff = (input_float * basis).sum(dim=keepdim_axes, keepdim=True) / denom
        projected = coeff * basis.reshape((1,) + tuple(basis.shape))
    else:
        basis = input_float
        source = target_float.reshape((1,) + tuple(target_float.shape))
        denom = basis.pow(2).sum(dim=keepdim_axes, keepdim=True).clamp_min(1e-12)
        coeff = (source * basis).sum(dim=keepdim_axes, keepdim=True) / denom
        projected = coeff * basis

    value[:, last_pos, ...] = projected.to(dtype=input_value.dtype) * spec.scale_factor
    return value


@dataclass
class FeatureSelectionSpec:
    """Pre-filter specification for :func:`extract_top_features_impl`.

    All criteria use **OR** semantics: a feature row ``(layer, position, feature_id)``
    passes the filter if it matches *any* of the non-empty criteria.

    Numeric slice notation is supported for ``layers`` and ``positions`` — pass a
    Python ``slice`` object alongside (or instead of) explicit ``int`` lists. The
    slice is applied as a numeric range over the observed values in
    *active_features*, so ``slice(10, None)`` means "layer >= 10" and
    ``slice(0, 10)`` means "position >= 0 and < 10".

    Attributes:
        layers: Explicit layer indices to include.
        positions: Explicit token-position indices to include.
        feature_ids: Explicit feature-ID values to include.
        layer_slice: A ``slice`` expanded over observed layer values.
        position_slice: A ``slice`` expanded over observed position values.
        triples: Exact ``(layer, position, feature_id)`` tuples to include.
        layer_feature_pairs: Exact ``(layer, feature_id)`` pairs to include across any position.
        activation_overrides: Optional override activation values keyed by ``(layer, feature_id)``.
    """

    layers: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    feature_ids: list[int] = field(default_factory=list)
    layer_slice: slice | None = None
    position_slice: slice | None = None
    triples: list[tuple[int, int, int]] = field(default_factory=list)
    layer_feature_pairs: list[tuple[int, int]] = field(default_factory=list)
    activation_overrides: dict[tuple[int, int], float] = field(default_factory=dict)


def _expand_slice(s: slice, observed: torch.Tensor) -> list[int]:
    """Expand a numeric ``slice`` into concrete observed values."""
    unique_vals = sorted(observed.unique().tolist())
    if not unique_vals:
        return []

    filtered = [
        int(value)
        for value in unique_vals
        if (s.start is None or value >= s.start) and (s.stop is None or value < s.stop)
    ]
    if s.step not in (None, 1):
        filtered = filtered[:: int(s.step)]
    return filtered


def apply_feature_selection_filter(
    active_features: torch.Tensor,
    spec: FeatureSelectionSpec,
) -> torch.Tensor:
    """Return a boolean mask (length *N*) selecting rows of *active_features* that match *spec*.

    ``active_features`` has shape ``(N, 3)`` with columns ``[layer, position, feature_id]``.
    """
    n = active_features.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.bool)

    mask = torch.zeros(n, dtype=torch.bool)

    layers_col = active_features[:, 0]
    positions_col = active_features[:, 1]
    features_col = active_features[:, 2]

    # Explicit layer list
    if spec.layers:
        layer_set = torch.tensor(spec.layers, dtype=layers_col.dtype)
        mask |= torch.isin(layers_col, layer_set)

    # Layer slice
    if spec.layer_slice is not None:
        expanded = _expand_slice(spec.layer_slice, layers_col)
        if expanded:
            layer_set = torch.tensor(expanded, dtype=layers_col.dtype)
            mask |= torch.isin(layers_col, layer_set)

    # Explicit position list
    if spec.positions:
        pos_set = torch.tensor(spec.positions, dtype=positions_col.dtype)
        mask |= torch.isin(positions_col, pos_set)

    # Position slice
    if spec.position_slice is not None:
        expanded = _expand_slice(spec.position_slice, positions_col)
        if expanded:
            pos_set = torch.tensor(expanded, dtype=positions_col.dtype)
            mask |= torch.isin(positions_col, pos_set)

    # Explicit feature IDs
    if spec.feature_ids:
        fid_set = torch.tensor(spec.feature_ids, dtype=features_col.dtype)
        mask |= torch.isin(features_col, fid_set)

    # Exact (layer, position, feature_id) triples
    if spec.triples:
        triple_tensor = torch.tensor(spec.triples, dtype=active_features.dtype)  # (T, 3)
        # Compare every row against every triple: (N, 1, 3) == (T, 3) → (N, T, 3)
        matches = (active_features.unsqueeze(1) == triple_tensor.unsqueeze(0)).all(dim=2)  # (N, T)
        mask |= matches.any(dim=1)

    # Exact (layer, feature_id) pairs across any position
    if spec.layer_feature_pairs:
        pair_tensor = torch.tensor(spec.layer_feature_pairs, dtype=active_features.dtype)  # (P, 2)
        pair_rows = torch.stack((layers_col, features_col), dim=1)
        matches = (pair_rows.unsqueeze(1) == pair_tensor.unsqueeze(0)).all(dim=2)  # (N, P)
        mask |= matches.any(dim=1)

    return mask


class BackendCapability(Enum):
    """Capabilities that a model backend may support.

    Ops and the dispatcher can query ``backend.capabilities`` to check support before
    calling optional methods.  Backends that do not support a capability should fall back
    to a simpler code path (e.g., looping instead of batching).
    """

    BATCHED_HOOKS = "batched_hooks"
    """Backend can run multiple forward passes with different hook configs in a single batched execution (e.g.,
    NNsight multi-invoke within one trace)."""

    GRADIENTS = "gradients"
    """Backend supports forward + backward with gradient caching."""


class AnalysisBackendCapability(Enum):
    """Capabilities exposed by analysis adapters/backends rather than model execution backends."""

    ATTRIBUTION_GRAPH = "attribution_graph"
    """Module exposes attribution graph analysis support via an attached analysis backend."""

    FEATURE_INTERVENTION = "feature_intervention"
    """Module exposes feature intervention support via an attached analysis backend."""

    # Future capabilities (reserved):
    # REMOTE_EXECUTION = "remote_execution"
    # SOURCE_TRACING = "source_tracing"


Capability: TypeAlias = BackendCapability | AnalysisBackendCapability


@dataclass(frozen=True)
class ModuleCapabilities:
    """Execution and analysis capabilities exposed by a module."""

    model: frozenset[BackendCapability]
    analysis: frozenset[AnalysisBackendCapability]

    @property
    def all(self) -> frozenset[Capability]:
        return frozenset({*self.model, *self.analysis})

    @property
    def values(self) -> frozenset[str]:
        return frozenset(cap.value for cap in self.all)

    def supports(self, capability: Capability) -> bool:
        if isinstance(capability, BackendCapability):
            return capability in self.model
        return capability in self.analysis


def normalize_backend_capability(capability: Any) -> Capability:
    """Normalize capability-like values to the local execution or analysis capability enums."""

    if isinstance(capability, (BackendCapability, AnalysisBackendCapability)):
        return capability

    raw_value = getattr(capability, "value", capability)
    normalized_value = str(raw_value)
    if normalized_value == "attribution":
        normalized_value = AnalysisBackendCapability.ATTRIBUTION_GRAPH.value

    try:
        return BackendCapability(normalized_value)
    except ValueError:
        pass

    try:
        return AnalysisBackendCapability(normalized_value)
    except ValueError:
        if isinstance(raw_value, str) and "." in raw_value:
            suffix = raw_value.split(".")[-1].lower()
            if suffix == "attribution":
                suffix = AnalysisBackendCapability.ATTRIBUTION_GRAPH.value
            try:
                return BackendCapability(suffix)
            except ValueError:
                return AnalysisBackendCapability(suffix)
        raise


def get_model_backend(module: Any) -> ModelBackend | None:
    """Return the module's model backend while avoiding mock-created private attrs."""

    module_dict = getattr(module, "__dict__", None)
    backend = module_dict.get("_model_backend") if isinstance(module_dict, dict) else None
    if backend is None and hasattr(module, "model_backend"):
        try:
            backend = module.model_backend
        except (AssertionError, AttributeError):
            backend = None
    return backend


def get_analysis_backend(module: Any) -> AnalysisBackend | None:
    module_dict = getattr(module, "__dict__", None)
    backend = module_dict.get("_analysis_backend") if isinstance(module_dict, dict) else None
    if backend is None and hasattr(module, "analysis_backend"):
        try:
            backend = module.analysis_backend
        except (AssertionError, AttributeError):
            backend = None
    return backend


def require_analysis_backend(module: Any) -> AnalysisBackend:
    """Return the module's analysis backend or raise if it is unavailable."""

    backend = get_analysis_backend(module)
    if backend is None:
        raise ValueError("Target module must expose an analysis_backend for this operation")
    return backend


def get_module_capabilities(module: Any) -> ModuleCapabilities:
    """Aggregate execution and analysis capabilities exposed by a module."""

    model_capabilities: set[BackendCapability] = set()
    analysis_capabilities: set[AnalysisBackendCapability] = set()
    backend = get_model_backend(module)

    if backend is not None and hasattr(backend, "capabilities"):
        model_capabilities.update(
            capability
            for capability in (normalize_backend_capability(raw_capability) for raw_capability in backend.capabilities)
            if isinstance(capability, BackendCapability)
        )

    analysis_backend = get_analysis_backend(module)
    if analysis_backend is not None and hasattr(analysis_backend, "capabilities"):
        analysis_capabilities.update(
            capability
            for capability in (
                normalize_backend_capability(raw_capability) for raw_capability in analysis_backend.capabilities
            )
            if isinstance(capability, AnalysisBackendCapability)
        )

    legacy_analysis_capabilities = getattr(module, "analysis_capabilities", None)
    if legacy_analysis_capabilities:
        analysis_capabilities.update(
            capability
            for capability in (
                normalize_backend_capability(raw_capability) for raw_capability in legacy_analysis_capabilities
            )
            if isinstance(capability, AnalysisBackendCapability)
        )

    return ModuleCapabilities(model=frozenset(model_capabilities), analysis=frozenset(analysis_capabilities))


@runtime_checkable
class AnalysisBackend(Protocol):
    """Protocol defining analysis-adapter functionality layered above model execution backends."""

    @property
    def capabilities(self) -> frozenset[AnalysisBackendCapability]:
        """Return the set of analysis capabilities this backend supports."""
        ...

    def supports(self, capability: AnalysisBackendCapability) -> bool:
        """Check whether this backend supports a given analysis capability."""
        ...

    def get_tokenizer(self, module: Any) -> Any: ...

    def get_embedding_weight(self, module: Any) -> torch.Tensor: ...

    def token_strings_to_ids(self, tokenizer: Any, token_strings: list[str]) -> list[int]: ...

    def resolve_prompt(self, module: Any, analysis_batch: Any, batch: Any) -> str: ...

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
    ) -> list[Any] | None: ...

    def resolve_feature_intervention_settings(
        self,
        module: Any,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def build_feature_interventions(
        self,
        analysis_batch: Any,
        settings: dict[str, Any],
    ) -> tuple[list[tuple[int, int, int, float]], dict[str, Any]]: ...

    def feature_intervention_call_kwargs(self, settings: dict[str, Any]) -> dict[str, Any]: ...

    def decompose_graph(self, graph: Any, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def hydrate_graph_from_batch(self, analysis_batch: Any) -> Any: ...

    def build_pruned_graph(self, graph: Any, node_threshold: float, edge_threshold: float) -> Any: ...

    def select_feature_rows(self, active_features: torch.Tensor, selected_features: torch.Tensor) -> torch.Tensor: ...

    def compute_node_influence_scores(self, graph: Any) -> tuple[torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol defining the interface for model execution backends.

    Each backend wraps a specific framework's model execution API (e.g., TransformerLens hook-based execution, nnsight
    trace-based execution) behind a uniform interface used by analysis op implementations.

    .. note:: ``hook=True`` evaluation

        NNsight's ``hook=True`` parameter (on ``tracer.invoke()``) enables ``.output`` /
        ``.input`` access on auxiliary modules like SAEs.  Our current architecture calls
        ``sae.encode()`` / ``sae.decode()`` explicitly within the trace, giving direct
        proxy access to feature activations.  If SAEs were registered as model sub-modules,
        ``hook=True`` could replace explicit encode/decode calls, but the current external
        ``latent_model_handles`` design makes ``hook=True`` unnecessary.  Adding
        ``hook=True`` would require architectural changes to how SAEs are attached and is
        best evaluated in a future session.
    """

    @property
    def capabilities(self) -> frozenset[BackendCapability]:
        """Return the set of capabilities this backend supports.

        Backends must override this property to declare their capabilities. Analysis ops can check capabilities before
        calling optional methods.
        """
        ...

    def supports(self, capability: BackendCapability) -> bool:
        """Check whether this backend supports a given capability.

        Default implementation checks ``capability in self.capabilities``.
        """
        ...

    def fwd(
        self,
        model: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Run a minimal forward pass and return logits.

        Each backend handles any necessary batch-key mapping (e.g., the NNsight
        backend wraps the call in a trace context so that
        ``LanguageModel._prepare_input`` correctly routes ``input`` →
        ``input_ids`` for HuggingFace models).

        Args:
            model: The model to run.
            batch: Input batch dict.

        Returns:
            Model output logits tensor.
        """
        ...

    def fwd_w_cache_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        names_filter: NamesFilter,
    ) -> tuple[torch.Tensor, Any]:
        """Run a forward pass with activation caching and latent model hooks.

        Args:
            model: The model to run (e.g., HookedSAETransformer, SAETransformerBridge).
            batch: Input batch dict (unpacked as ``**batch`` for the model call).
            latent_model_handles: Latent model handles (e.g., SAE objects) to attach.
            names_filter: Filter specifying which hook activations to cache.

        Returns:
            Tuple of (logits, activation_cache).
        """
        ...

    def fwd_w_cache(
        self,
        model: Any,
        batch: dict[str, Any],
        names_filter: NamesFilter,
    ) -> tuple[torch.Tensor, Any]:
        """Run a forward pass with activation caching but without latent model hooks.

        Args:
            model: The model to run.
            batch: Input batch dict.
            names_filter: Filter specifying which hook activations to cache.

        Returns:
            Tuple of (logits, activation_cache).
        """
        ...

    def fwd_w_hooks_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        fwd_hooks: list[tuple[str, Any]],
        clear_contexts: bool = True,
    ) -> torch.Tensor:
        """Run a forward pass with custom forward hooks and latent model hooks.

        Args:
            model: The model to run.
            batch: Input batch dict (unpacked as ``**batch`` for the model call).
            latent_model_handles: Latent model handles to attach.
            fwd_hooks: List of (hook_name, hook_fn) tuples for forward hooks.
            clear_contexts: Whether to clear hook contexts after the forward pass.

        Returns:
            Model output logits.
        """
        ...

    def fwd_w_hooks_batched(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        hook_configs: Sequence[list[tuple[str, Any]]],
        clear_contexts: bool = True,
        configs_per_pass: int | None = None,
    ) -> list[torch.Tensor]:
        """Run multiple forward passes with different hook configurations, batched when possible.

        Each element of ``hook_configs`` is a ``fwd_hooks`` list (as passed to
        ``fwd_w_hooks_and_latent_models``).  Backends that support
        :attr:`BackendCapability.BATCHED_HOOKS` may batch all configs into a single
        execution context (e.g., NNsight multi-invoke within one trace) for efficiency.
        Other backends loop over configs sequentially.

        ``configs_per_pass`` limits how many configs are batched per execution context.
        When ``None`` (default), the entire ``hook_configs`` list is batched in one context.
        Setting a value (e.g., 64) chunks the work to avoid OOM with very large alive-latent
        counts.

        .. note::

            TODO: evaluate the possibility of releasing memory after each invoke within a
            trace if OOMs become a problem (would require nnsight-level support).

        Args:
            model: The model to run.
            batch: Input batch dict (unpacked as ``**batch`` for the model call).
            latent_model_handles: Latent model handles to attach.
            hook_configs: Sequence of ``fwd_hooks`` lists, one per forward pass.
            clear_contexts: Whether to clear hook contexts (for TL backend compatibility).
            configs_per_pass: Maximum number of configs to batch per execution context.
                ``None`` means unbounded (all configs in one trace).

        Returns:
            List of logits tensors, one per element in ``hook_configs``.
        """
        ...

    def fwd_w_grads_and_latent_models(
        self,
        model: Any,
        batch: dict[str, Any],
        latent_model_handles: list[Any],
        fwd_hooks: list[tuple[Any, Any]],
        bwd_hooks: list[tuple[Any, Any]],
        backward_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run forward + backward with latent model hooks and gradient caching.

        The backend owns the entire forward+backward execution flow.  This enables both
        eager execution (TransformerLens) and deferred/traced execution (NNsight) to
        use the same op-level code.

        The ``backward_fn`` closure is provided by the analysis op and computes a scalar
        metric from raw model logits.  The backend calls ``backward_fn(logits)`` to obtain
        the scalar, then runs ``.backward()`` on it (eager for TL, deferred via
        ``with scalar.backward():`` for NNsight).

        Forward and backward cache hooks (``fwd_hooks``, ``bwd_hooks``) are structured as
        ``[(names_filter, cache_fn), ...]`` and are invoked by the backend to populate
        ``analysis_cfg.cache_dict``.  For TL, hooks fire during execution.  For NNsight,
        the backend calls them after the trace completes with materialized tensors.

        Args:
            model: The model to run.
            batch: Input batch dict (unpacked as ``**batch`` for the model call).
            latent_model_handles: Latent model handles (e.g., SAE objects) to attach.
            fwd_hooks: Forward cache hooks ``[(names_filter, cache_fn), ...]``.
            bwd_hooks: Backward cache hooks ``[(names_filter, cache_fn), ...]``.
            backward_fn: ``raw_logits -> scalar``.  Takes the full model output logits and
                returns a scalar tensor to call ``.backward()`` on.  Must be compatible with
                both real tensors (TL) and NNsight proxy objects.

        Returns:
            Raw model output logits (always a real tensor, even for NNsight).
        """
        ...

    def wrap_activation_cache(
        self,
        cache_dict: dict[str, Any],
        model: Any,
    ) -> Any:
        """Wrap a raw activation dict into a backend-specific activation cache object.

        For TransformerLens, wraps in ``ActivationCache``. Other backends may return the dict
        as-is or wrap in their own cache type.

        Args:
            cache_dict: Raw dict mapping hook names to activation tensors.
            model: The model instance (may be needed for cache construction).

        Returns:
            A cache object suitable for indexed access by hook name.
        """
        ...

    def fwd_w_intervention(
        self,
        model: Any,
        batch: dict[str, Any],
        interventions: InterventionDict | Mapping[str, InterventionValue],
        latent_model_handles: list[Any] | None = None,
    ) -> tuple[Any, Any]:
        """Run baseline + intervention forward passes using the given hook specs.

        Performs two forward passes:

        1. **Baseline**: captures pre-intervention logits.
        2. **Intervention**: for each key in *interventions*, matches the key (which may
           contain ``*`` wildcards) against available hook names, then applies each
           ``InterventionSpec`` at the last sequence position according to its ``mode``
           (``"replace"``, ``"add"``, or ``"project"``).

        Args:
            model: The model to run.
            batch: Input batch dict.
            interventions: Either a canonical :class:`InterventionDict` keyed by concrete hook
                names or a raw mapping from hook-name patterns to intervention payloads. Raw
                payloads may be tensors, ``InterventionSpec`` instances, mapping-style specs, or
                sequences of those values. Patterns may use ``*`` as a glob-style wildcard.
            latent_model_handles: Optional latent model handles to enable latent-hook-aware
                resolution and execution.

        Returns:
            ``(pre_intervention_logits, post_intervention_logits)`` — both real tensors.
        """
        ...


__all__ = [
    "BackendCapability",
    "AnalysisBackend",
    "AnalysisBackendCapability",
    "apply_intervention_to_last_token",
    "build_intervention_dict",
    "Capability",
    "expand_intervention_patterns",
    "FeatureSelectionSpec",
    "get_intervention_target_shape",
    "HOOK_ALIAS_GROUPS",
    "InterventionDict",
    "InterventionSpec",
    "apply_feature_selection_filter",
    "get_module_capabilities",
    "ModelBackend",
    "ModuleCapabilities",
    "normalize_backend_capability",
    "resolve_interventions",
]
