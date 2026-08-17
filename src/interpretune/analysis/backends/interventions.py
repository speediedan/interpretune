"""Intervention specification, expansion, and application for analysis backends.

Backend-agnostic: specs are declarative and resolved against a concrete model's hook names by the
individual backends. Part of the sanctioned :mod:`interpretune.analysis.backends` seam that op
implementations (bundled, local, or hub) may import.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any, Callable, NamedTuple, TypeAlias

import torch

from interpretune.analysis.backends.hook_mapping import SUBHOOK_SUFFIXES


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
