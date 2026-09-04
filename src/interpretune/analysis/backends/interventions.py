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

from interpretune.analysis.backends.capabilities import (
    BackendCapability,
    InterventionMode,
    PositionScope,
)
from interpretune.analysis.backends.hook_mapping import SUBHOOK_SUFFIXES

__all__ = [
    "PositionScope",
    "InterventionMode",
    "InterventionSpec",
    "InterventionDict",
    "InterventionValue",
    "HOOK_ALIAS_GROUPS",
    "expand_intervention_patterns",
    "build_intervention_dict",
    "resolve_interventions",
    "apply_intervention",
    "get_intervention_target_shape",
    "normalize_position_scope",
    "normalize_intervention_mode",
    "require_position_scope",
    "require_intervention_mode",
    "require_intervention_support",
    "iter_intervention_axes",
]


class InterventionSpec(NamedTuple):
    """Specification for a single hook-point intervention.

    Attributes:
        intervention_tensor: Intervention tensor with any shape broadcast-compatible with the
            targeted hook-point slice at the intervention position. This is not restricted to
            ``(d_model,)`` and may instead match higher-rank activations such as
            ``(n_heads, d_head)`` or latent feature activations.
        mode: ``"replace"`` overwrites the activation at the target position with the tensor;
              ``"add"`` adds ``intervention_tensor * scale_factor`` to the activation;
              ``"patch"`` swaps a pair of concepts in lens coordinates, leaving the component
              orthogonal to that pair untouched (see :func:`_apply_lens_coordinate_patch`);
              ``"project"`` replaces the activation with a projection result. By default, the
              current hook input is projected onto the span of ``intervention_tensor``, so the
              intervention tensor acts as the projection basis. When
              ``use_intervention_tensor_as_basis`` is ``False``, the direction is reversed and
              ``intervention_tensor`` is projected onto the span of the current hook input.
        scale_factor: Scalar multiplier applied to *intervention_tensor* before the intervention
            (not used in ``"replace"`` mode and applied to the projected activation in
            ``"project"`` mode).
        position_scope: WHICH POSITIONS the intervention edits -- ``"last_token"`` (default) or
            ``"all_positions"``. This is part of the specification rather than a backend setting
            because the CALLER knows which operation they meant: steering a final-token prediction
            and steering a whole prompt are different experiments, not different implementations of
            one. A backend that cannot honour the requested scope must REFUSE
            (:func:`require_position_scope`) rather than substitute the other one, since the two
            produce equally plausible activations and a silent substitution is undetectable
            downstream.
        use_intervention_tensor_as_basis: Controls which vector defines the projection basis in
            ``"project"`` mode. ``True`` means project the current hook input onto the span of
            ``intervention_tensor``. ``False`` means project ``intervention_tensor`` onto the
            span of the current hook input instead.
    """

    intervention_tensor: torch.Tensor
    mode: InterventionMode | str = InterventionMode.REPLACE
    scale_factor: float = 1.0
    use_intervention_tensor_as_basis: bool = True
    position_scope: PositionScope | str = PositionScope.LAST_TOKEN


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
    # NOT grouped with `hook_mlp_in`: `mlp.hook_in` is the sublayer's argument, i.e. the block norm's
    # OUTPUT, while TransformerLens fires the legacy `blocks.{i}.hook_mlp_in` on `resid_mid.clone()`
    # BEFORE that norm (`components/transformer_block.py:195-197`; the bridge matches legacy, see
    # `model_bridge/bridge.py:3870`). Measured cos 0.088 apart on gemma-3-1b-it layer 5. Grouping them
    # applies an intervention one norm away from where the caller asked for it.
    ("mlp.hook_in",),
    ("hook_mlp_in",),
    ("mlp.hook_out", "hook_mlp_out"),
    ("embed.hook_out", "hook_embed"),
    ("pos_embed.hook_out", "hook_pos_embed"),
    ("attn.hook_pattern", "attn.hook_attention_weights"),
    ("attn.hook_hidden_states", "attn.hook_result"),
    # The three norm tensors are NOT aliases; see NOTE [Norm hooks are three tensors] in
    # `hook_mapping.py`. `hook_normalized` fires before the learned gain and `hook_scale` is the
    # per-token denominator, shape [batch, pos, 1], which cannot alias a [batch, pos, d_model] tensor
    # at all. Measured on gemma-3-1b-it layer 5: `ln2.hook_out` matches the norm module's output at
    # cos 1.000000, `ln2.hook_normalized` at 0.181. TransformerLens' `model_structure.md` documents
    # them as aliases of `.hook_out`; its own implementation disagrees, and this table followed the
    # documentation.
    ("ln1.hook_out",),
    ("ln2.hook_out",),
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
        """Mapping-style access to the underlying hook map."""
        return self.hook_map.items()

    def keys(self):
        """The hook names carrying interventions."""
        return self.hook_map.keys()

    def values(self):
        """The intervention specs, per hook."""
        return self.hook_map.values()

    @classmethod
    def from_mapping(
        cls,
        hook_map: Mapping[str, InterventionSpec | Sequence[InterventionSpec]],
    ) -> InterventionDict:
        """Build from a plain mapping, normalizing single specs to sequences.

        Callers may write one spec per hook or several; normalizing at construction means every consumer downstream
        iterates uniformly instead of re-checking the shape.
        """
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
            **_shared_spec_fields(value, default_mode=default_mode, default_scale_factor=default_scale_factor),
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
    mode = normalize_intervention_mode(spec.mode)
    if mode is InterventionMode.PATCH:
        # A swap needs a PAIR, so this is the one mode whose tensor is legitimately larger than the slice
        # it targets: `(2, *target_shape)`. The general check below demands the tensor broadcast INTO the
        # target, which `(2, d_model)` against `(d_model,)` does not -- so validating patch under it would
        # make the mode unreachable through the op surface while low-level calls kept working.
        if tensor.ndim < 1 or tensor.shape[0] != 2:
            raise ValueError(
                "intervention mode 'patch' requires exactly two lens vectors stacked on a leading axis of "
                f"size 2 (source, target); got tensor with shape {tuple(tensor.shape)}"
            )
        _ensure_shape_compatible(tuple(tensor.shape[1:]), target_shape, hook_name)
    else:
        _ensure_shape_compatible(tuple(tensor.shape), target_shape, hook_name)
    # Rebuild with EVERY field. An earlier version omitted `position_scope` here, and since this is the
    # canonicalization every backend runs, an `all_positions` request silently became last-token on both
    # bundled backends: plausible logits, right arithmetic, wrong position set, nothing raised.
    return InterventionSpec(
        intervention_tensor=tensor,
        mode=mode,
        scale_factor=spec.scale_factor,
        use_intervention_tensor_as_basis=spec.use_intervention_tensor_as_basis,
        position_scope=normalize_position_scope(spec.position_scope),
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


def _shared_spec_fields(
    value: Mapping[str, Any], *, default_mode: InterventionMode | str, default_scale_factor: float
) -> dict[str, Any]:
    """The non-tensor fields of a mapping payload, normalized, for every spec the mapping expands to.

    One reader for the per-hook (`intervention_tensors`) and single-tensor mapping shapes, so a field
    added to one cannot be silently absent from the other -- which is how `position_scope` was dropped
    on the per-hook path while the single-tensor path carried it.
    """
    return {
        "mode": normalize_intervention_mode(value.get("mode", default_mode)),
        "scale_factor": float(value.get("scale_factor", default_scale_factor)),
        "use_intervention_tensor_as_basis": bool(value.get("use_intervention_tensor_as_basis", True)),
        "position_scope": normalize_position_scope(value.get("position_scope", PositionScope.LAST_TOKEN)),
    }


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
        shared = _shared_spec_fields(raw_value, default_mode=default_mode, default_scale_factor=default_scale_factor)
        return [
            (
                _validate_intervention_spec(
                    InterventionSpec(torch.as_tensor(tensor), **shared),
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
    # The scope is part of the shorthand surface too. Without it, the only way to ask for `all_positions`
    # through the op was an explicit `interventions` mapping, and a caller using the shorthand fields could
    # not express the second scope at all.
    position_scope = _first_defined(
        resolve_field("intervention_position_scope"),
        kwargs.get("position_scope"),
        PositionScope.LAST_TOKEN,
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
        "mode": normalize_intervention_mode(intervention_mode or InterventionMode.REPLACE),
        "scale_factor": float(scale_factor),
        "use_intervention_tensor_as_basis": bool(use_intervention_tensor_as_basis),
        "position_scope": normalize_position_scope(position_scope),
    }
    if intervention_tensors is not None:
        payload["intervention_tensors"] = intervention_tensors
    else:
        payload["intervention_tensor"] = intervention_tensor

    return {hook_qualifier: payload}


def _apply_lens_coordinate_patch(
    spec: InterventionSpec,
    *,
    input_value: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Swap a concept pair in lens coordinates, preserving everything orthogonal to the pair.

    Implements ``h <- h + V(sigma(c) - c)`` from the J-space workspace paper, where ``V = [v_s v_t]``
    holds the two lens vectors as columns, ``c = V^+ h`` are the activation's coordinates in their span,
    and ``sigma`` swaps the two entries. Because the update lies entirely in ``span(V)``, the orthogonal
    component of ``h`` is mathematically untouched -- which is the property that distinguishes this from
    naive steering (``h <- h + alpha * v``), where the added vector perturbs every component it overlaps.

    ``spec.intervention_tensor`` must supply exactly two vectors, stacked on a leading axis of size 2:
    index 0 is the source concept ``v_s`` and index 1 the target ``v_t``. A single vector cannot express
    a swap, so this mode rejects one rather than guessing a partner.

    ``spec.scale_factor`` scales the swapped coordinates, the paper's optional ``alpha``. It defaults to
    1.0, a pure exchange. The paper reports oversteering as a real failure mode and uses alpha=2 only
    where needed, so values above 1 should be justified per-case rather than tuned by default.

    The pseudoinverse is used rather than a transpose because lens vectors are not orthonormal, and it is
    worth being precise about what that buys. Orthogonal preservation holds EITHER way: the update lies in
    ``span(V)`` by construction, so nothing outside that span can move no matter how the coordinates are
    computed. What ``pinv`` buys is that the swap is a swap. ``V^+`` gives the true oblique coordinates, so
    the patched activation satisfies ``V^+ h' == sigma(c)`` exactly. With ``V^T`` on a non-orthonormal pair
    the result lands somewhere else entirely -- measured on a correlated pair, target ``sigma(c)`` of
    ``[3.37, -3.49]`` came out as ``[-2.98, 2.86]``. The failure is silent, because the orthogonal component
    still looks untouched and the activation still moved.
    """
    if target.ndim < 1 or target.shape[0] != 2:
        raise ValueError(
            "intervention mode 'patch' requires exactly two lens vectors stacked on a leading axis of "
            f"size 2 (source, target); got tensor with shape {tuple(target.shape)}"
        )

    batch = input_value.shape[0]
    flat = input_value.reshape(batch, -1).to(dtype=torch.float32)
    basis = target.reshape(2, -1).to(dtype=torch.float32)
    if basis.shape[1] != flat.shape[1]:
        raise ValueError(
            f"intervention mode 'patch' lens vectors have width {basis.shape[1]} but the hook activation "
            f"is {flat.shape[1]}-dimensional"
        )

    v_matrix = basis.transpose(0, 1)  # (d, 2), lens vectors as columns
    coords = flat @ torch.linalg.pinv(v_matrix).transpose(0, 1)  # (batch, 2) == (V^+ h)^T
    swapped = coords.flip(-1) * spec.scale_factor
    patched = flat + (swapped - coords) @ v_matrix.transpose(0, 1)

    return patched.reshape(input_value.shape).to(dtype=input_value.dtype)


def _apply_mode_to_region(input_value: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
    """Apply one intervention mode to a selected REGION, returning the edited region.

    The region is whatever the scope selected, flattened so that its leading axis indexes independent
    rows. That flattening is what lets one implementation serve both scopes: ``last_token`` passes a
    single position per batch row, ``all_positions`` passes every position as its own row, and the
    per-row mathematics is identical in both cases -- which is the property that makes the two scopes
    the SAME operation applied to different position sets, rather than two operations that happen to
    share a name.
    """
    target = torch.as_tensor(spec.intervention_tensor, device=input_value.device, dtype=input_value.dtype)
    mode = normalize_intervention_mode(spec.mode)

    if mode is InterventionMode.REPLACE:
        return torch.broadcast_to(target, input_value.shape).to(dtype=input_value.dtype).clone()

    if mode is InterventionMode.ADD:
        return input_value + target * spec.scale_factor

    if mode is InterventionMode.PATCH:
        return _apply_lens_coordinate_patch(spec, input_value=input_value, target=target)

    assert mode is InterventionMode.PROJECT, f"unreachable: normalize_intervention_mode admitted {mode!r}"

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

    return projected.to(dtype=input_value.dtype) * spec.scale_factor


def apply_intervention(
    value: torch.Tensor,
    spec: InterventionSpec,
    *,
    last_pos: int,
) -> torch.Tensor:
    """Apply one intervention spec to the positions ``spec.position_scope`` selects.

    **The name carries no scope, deliberately.** This replaced ``apply_intervention``,
    and the rename is the point rather than tidying: a function whose name asserts one scope cannot
    honestly implement two, and leaving the old name would guarantee every future reader has to work
    out whether the name or the parameter is lying.

    ``last_pos`` is still required, because it identifies the final real token under left padding and
    is therefore not derivable from the tensor shape. It is simply unused when the scope is
    ``all_positions``.

    The existing hook value is treated as the projection input and ``spec.intervention_tensor`` as the
    projection target. In ``"project"`` mode the target defines the default projection basis; when
    ``spec.use_intervention_tensor_as_basis`` is ``False`` the direction is reversed.
    """
    scope = normalize_position_scope(spec.position_scope)

    if scope == PositionScope.LAST_TOKEN:
        value[:, last_pos, ...] = _apply_mode_to_region(value[:, last_pos, ...], spec)
        return value

    if scope == PositionScope.ALL_POSITIONS:
        batch, seq = value.shape[0], value.shape[1]
        flat = value.reshape(batch * seq, *value.shape[2:])
        value[...] = _apply_mode_to_region(flat, spec).reshape(value.shape)
        return value

    raise AssertionError(f"unreachable: normalize_position_scope admitted {scope!r}")


def normalize_position_scope(scope: PositionScope | str) -> PositionScope:
    """Coerce a spec's scope to the enum, raising with the valid set if it is not one.

    Exists so exactly ONE place converts. A spec can legitimately arrive carrying a plain string -- from
    YAML, or a notebook literal -- and ``PositionScope`` is a ``str`` enum so those compare equal at
    runtime. Leaving each consumer to handle both forms is how one of them eventually handles only one.
    """
    try:
        return PositionScope(scope)
    except ValueError:
        raise ValueError(
            f"Unknown position_scope {scope!r}; expected one of {[m.value for m in PositionScope]!r}. "
            "An unrecognised scope is refused rather than defaulted, because both valid scopes produce "
            "plausible activations and guessing between them is undetectable downstream."
        ) from None


def normalize_intervention_mode(mode: InterventionMode | str) -> InterventionMode:
    """Coerce a spec's mode to the enum, raising with the valid set if it is not one.

    One converter, as for scope.
    """
    try:
        return InterventionMode(mode)
    except ValueError:
        raise ValueError(
            f"Unknown intervention mode {mode!r}; expected one of {[m.value for m in InterventionMode]!r}."
        ) from None


def _declared_axes(backend: Any, *, backend_name: str) -> tuple[set[str], set[str]]:
    """The backend's declared scope and mode VALUES, or a refusal naming what is missing.

    Read by attribute and compared by value rather than by ``isinstance``, for the same reason
    ``require_backend_capability`` compares capability values: this repository's test infrastructure can
    load the capabilities module twice, leaving value-equal, identity-distinct enum members and dataclasses
    on either side of the gate. A backend that claims ``INTERVENTION`` without a record is refused outright
    rather than assumed last-token: the assumption is exactly the silent narrowing the scope field exists
    to remove, and a hub-delivered backend written before the scope existed is the case most likely to
    hit it.
    """
    record = getattr(backend, "intervention_support", None)
    if record is None:
        raise NotImplementedError(
            f"backend {backend_name!r} claims {BackendCapability.INTERVENTION.name} but attaches no "
            "`intervention_support` record, so nothing says which position scopes or modes it can honour. "
            "Declare an InterventionSupport(position_scopes=..., modes=...) on the backend."
        )
    scopes = {getattr(s, "value", s) for s in getattr(record, "position_scopes", ()) or ()}
    modes = {getattr(m, "value", m) for m in getattr(record, "modes", ()) or ()}
    if not scopes or not modes:
        raise TypeError(
            f"backend {backend_name!r}.intervention_support must be an InterventionSupport declaring at least one "
            f"position scope and one mode; got {record!r}"
        )
    return scopes, modes


def require_position_scope(backend: Any, spec: InterventionSpec, *, backend_name: str) -> None:
    """Refuse, with a reason, when a backend cannot honour the spec's position scope.

    **Refusing is the whole point of the scope field.** The two scopes produce equally plausible
    activations -- a whole-prompt intervention yields sensible logits, and so does a last-token one --
    so a backend that silently substituted the scope it supports would be undetectable downstream by
    any value comparison. That is precisely the failure that went unnoticed while interpretune had
    only one name for the operation.
    """
    scope = normalize_position_scope(spec.position_scope)
    scopes, _modes = _declared_axes(backend, backend_name=backend_name)
    if scope.value not in scopes:
        raise NotImplementedError(
            f"backend {backend_name!r} cannot apply an intervention with position_scope={scope.value!r}; it declares "
            f"{sorted(scopes)!r}. This is refused rather than narrowed or widened "
            "to a supported scope, because both scopes produce plausible activations and a silent substitution "
            "cannot be detected from the result."
        )


def require_intervention_mode(backend: Any, spec: InterventionSpec, *, backend_name: str) -> None:
    """Refuse, with the axis named, when a backend cannot express the spec's mode.

    Modes are where the semantics live: ``replace``, ``patch`` and ``project`` all need the CURRENT
    activation, and an additive steering primitive never observes it. A backend applying an undeclared
    mode as the one it has returns plausible logits for an intervention nobody requested, so the gate
    refuses before the backend is reached.
    """
    mode = normalize_intervention_mode(spec.mode)
    _scopes, modes = _declared_axes(backend, backend_name=backend_name)
    if mode.value not in modes:
        raise NotImplementedError(
            f"backend {backend_name!r} cannot apply an intervention with mode={mode.value!r}; it declares "
            f"{sorted(modes)!r}. Refused on the MODE axis rather than applied as "
            "another mode, since every mode returns plausible logits and the substitution is undetectable."
        )


def iter_intervention_axes(
    interventions: InterventionDict | Mapping[str, InterventionValue],
    *,
    default_mode: InterventionMode | str = InterventionMode.REPLACE,
) -> Iterator[tuple[str, InterventionSpec]]:
    """Yield ``(hook_pattern, spec)`` for every intervention in a RAW or canonical payload.

    Gating happens before canonicalization, because canonicalization needs concrete hook shapes and therefore runs
    inside the backend, which is exactly the place a refusal must not depend on. Only the scope and mode of each entry
    are needed to gate, and both are readable without a shape: a bare tensor carries the defaults, a mapping carries
    whatever it names, a spec carries itself.
    """
    items = interventions.items() if hasattr(interventions, "items") else interventions
    for pattern, raw in items:
        if isinstance(raw, Mapping) and "intervention_tensors" in raw:
            # The per-hook shape carries one set of axes for every tensor, and the tensors are not needed
            # to gate, so a placeholder stands in for them rather than coercing a shape the gate never reads.
            shared = _shared_spec_fields(raw, default_mode=default_mode, default_scale_factor=1.0)
            yield pattern, InterventionSpec(torch.empty(0), **shared)
            continue
        for spec in _coerce_shared_intervention_specs(raw, default_mode=normalize_intervention_mode(default_mode)):
            yield pattern, spec


def require_intervention_support(
    backend: Any,
    interventions: InterventionDict | Mapping[str, InterventionValue],
    *,
    backend_name: str,
) -> None:
    """Gate every entry of an intervention payload on both axes before it reaches ``fwd_w_intervention``.

    The single entry point ops call. It reads the payload the way the backend will (same coercion), so the axes gated
    here are the axes the backend will see, and a backend never has to re-derive them to refuse; its own refusal table,
    if it keeps one, is the second line of defence for a declaration that has drifted from the implementation.
    """
    for _pattern, spec in iter_intervention_axes(interventions):
        require_position_scope(backend, spec, backend_name=backend_name)
        require_intervention_mode(backend, spec, backend_name=backend_name)
