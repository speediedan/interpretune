"""TransformerLens hook name -> HuggingFace module path, for the nnsight backend.

This module is the nnsight-facing ADAPTER over the activation-point vocabulary (:mod:`interpretune.analysis.points`).
The per-architecture tables that used to live here are now data: one ``ComponentMap`` document per
architecture under ``analysis/points/data/``, and every TL hook name this resolver accepts is derived from
it by parsing the name into an :class:`~interpretune.analysis.points.ActivationPoint` and resolving it. The
``HookMapping`` / ``ArchitectureMapping`` records survive as the derived, nnsight-shaped view (a module path
template, an io selector, a tuple-ness default) because the backend and its tests consume them.

Follows the pattern of circuit-tracer's ``tl_nnsight_mapping.py``: a table keyed by HF architecture, one row
per addressable point, resolved per layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

from interpretune.analysis.backends.hook_mapping_constants import SUBHOOK_SUFFIXES

__all__ = [
    "ArchitectureMapping",
    "HookMapping",
    "HookNameResolver",
    "ResolvedHook",
    "SUBHOOK_SUFFIXES",
]


@dataclass(frozen=True)
class HookMapping:
    """Mapping from a TL base hook name to an HF module path and input/output selector.

    Attributes:
        envoy_path: HF module path template with ``{layer}`` placeholder.
            E.g., ``"transformer.h.{layer}"`` or ``"model.layers.{layer}.ln_2"``.
        io_type: Whether to read/write the module's ``"input"`` or ``"output"``.
        tuple_output: Whether the target module returns a tuple (e.g., transformer blocks,
            attention) or a single tensor (e.g., MLP, LayerNorm). Only relevant when
            ``io_type="output"``. Defaults to ``True`` (tuple output).
    """

    envoy_path: str
    io_type: Literal["input", "output"]
    tuple_output: bool = True


@dataclass(frozen=True)
class ArchitectureMapping:
    """Complete hook mapping configuration for a specific HF model architecture.

    Attributes:
        model_architecture: HuggingFace model class name (e.g., ``"GPT2LMHeadModel"``).
        hook_mappings: Dict mapping TL base hook names to :class:`HookMapping` instances.
    """

    model_architecture: str
    hook_mappings: dict[str, HookMapping] = field(default_factory=dict)


# NOTE [Norm hooks are three tensors]:
# A norm is two operations, so it exposes three tensors, and conflating any two of them silently
# reads an artifact off activations it was not trained on:
#
#   (1) hook_in          the norm's input                       (resid_pre / resid_mid)
#   (2) hook_normalized  x / scale, BEFORE the learned gain     no HF module emits this
#   (3) hook_out         the module's output, gain included     the HF norm module's output
#
# Measured on google/gemma-3-1b-it layer 5 against the HF module output: `ln2.hook_out` matches at
# cosine 1.000000 (rel_l2 6.9e-07) and `ln2.hook_normalized` at 0.181. They are not the same tensor.
# TransformerLens' `model_structure.md` calls (2) an alias of (3); its `normalization.py` fires three
# distinct tensors. The vocabulary resolves (2) and the scale as DERIVED tensor references (computable
# from the norm's input and parameters); this nnsight-facing resolver has no envoy for a derived tensor
# and refuses them with the reason below rather than pointing at the nearest module.
_DERIVED_REASONS = {
    "pre_gain_normalized": (
        "TransformerLens fires it on `x / scale`, before the norm's learned gain, so no HF module "
        "output equals it. Capture `{norm}.hook_out` for the gain-included tensor the sublayer "
        "actually receives, or recompute `x / scale` from `{norm}.hook_in`."
    ),
    "norm_scale": (
        "it is the norm's per-token denominator, shape [batch, pos, 1], an intermediate of the "
        "norm's arithmetic that no module returns. Recompute it from `{norm}.hook_in`."
    ),
}


def _derived_table(architecture: str) -> tuple[dict[str, HookMapping], dict[str, str]]:
    """Every TL hook name the vocabulary resolves on ``architecture``, as nnsight-shaped rows.

    Returns the mappings plus, for the derived norm tensors, the refusal reason keyed by base name.
    Semantic names (``hook_resid_pre``, ``hook_mlp_out``, ...) and component names (``ln2.hook_out``,
    ``attn.o.hook_in``, ``unembed.hook_out``) are enumerated together, so an alias never has to be
    listed by hand and cannot disagree with the point it names.
    """
    from interpretune.analysis.points import (
        TensorRef,
        Unresolvable,
        component_map_for,
        parse,
        resolve,
        semantic_names,
    )

    cmap = component_map_for(architecture)
    candidates: list[str] = list(semantic_names())
    for component in cmap.block_components():
        prefix = f"{component}." if component else ""
        candidates += [f"{prefix}hook_in", f"{prefix}hook_out"]
        if cmap.kind_of(component, 0) == "norm":
            candidates += [f"{prefix}hook_normalized", f"{prefix}hook_scale"]
    for component in cmap.global_components():
        candidates += [f"{component}.hook_in", f"{component}.hook_out"]
        if cmap.kind_of(component, None) == "norm":
            candidates += [f"{component}.hook_normalized", f"{component}.hook_scale"]

    mappings: dict[str, HookMapping] = {}
    derived: dict[str, str] = {}
    globals_ = set(cmap.global_components())
    # concrete layer-0 path -> `{layer}` template, from the map's own rows, so the template is never
    # reconstructed from a string (a block path ends in the index and a replace on ".0." would miss it)
    templates = {e.module.replace("{i}", "0"): e.module.replace("{i}", "{layer}") for e in cmap.components.values()}
    for base in dict.fromkeys(candidates):
        bare = parse(base)
        # A block-relative base is resolved at layer 0 and the concrete index is turned back into the
        # `{layer}` template below; a global one resolves as itself.
        is_global = bare.component in globals_ or base in ("hook_embed", "hook_pos_embed")
        point_for_resolve = bare if is_global else parse(f"blocks.0.{base}")
        if bare.caution is not None:
            # A precise name that callers routinely read as the OTHER tensor (`hook_mlp_in` meaning the
            # post-norm input the artifact was trained on). Resolving it either way is a silent substitution
            # for half its callers, so this consumer refuses it and names both spellings.
            pre = f"{bare.component}.hook_in"
            post = f"{'mlp' if bare.component == 'ln2' else 'attn'}.hook_in"
            derived[base] = (
                f"{bare.caution} Ask for `{post}` (the sublayer's argument) or `{pre}` (the pre-norm residual)."
            )
            continue
        res = resolve(point_for_resolve, cmap)
        if isinstance(res, Unresolvable):
            continue
        assert isinstance(res, TensorRef)
        if res.derived:
            derived[base] = _DERIVED_REASONS[res.derivation or ""].replace("{norm}", base.split(".")[0])
            continue
        template = templates.get(res.module_path, res.module_path)
        mappings[base] = HookMapping(envoy_path=template, io_type=res.io, tuple_output=res.tuple_output)  # type: ignore[arg-type]
    return mappings, derived


class ResolvedHook(NamedTuple):
    """Result of resolving a TL hook name for NNsight envoy navigation.

    Attributes:
        module_path: Concrete HF module path (e.g., ``"transformer.h.5"``).
        io_type: ``"input"`` or ``"output"``.
        tuple_output: Whether the module returns a tuple output.
    """

    module_path: str
    io_type: str
    tuple_output: bool


#: Hand-registered mappings (``register_architecture``) take precedence over the derived ones.
_ARCHITECTURE_REGISTRY: dict[str, ArchitectureMapping] = {}
_DERIVED_CACHE: dict[str, tuple[ArchitectureMapping, dict[str, str]]] = {}


def _mapping_for(architecture: str) -> tuple[ArchitectureMapping, dict[str, str]]:
    """The mapping for an architecture: hand-registered if any, else derived from its component map."""
    if architecture in _ARCHITECTURE_REGISTRY:
        return _ARCHITECTURE_REGISTRY[architecture], {}
    if architecture not in _DERIVED_CACHE:
        from interpretune.analysis.points import known_architectures

        if architecture not in known_architectures():
            raise KeyError(architecture)
        mappings, derived = _derived_table(architecture)
        _DERIVED_CACHE[architecture] = (ArchitectureMapping(architecture, mappings), derived)
    return _DERIVED_CACHE[architecture]


def _supported_architectures() -> list[str]:
    from interpretune.analysis.points import known_architectures

    return sorted(set(known_architectures()) | set(_ARCHITECTURE_REGISTRY))


# Regex for parsing TL hook names: "blocks.{layer}.{rest}"
_TL_HOOK_NAME_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


class HookNameResolver:
    """Resolves TransformerLens hook names to HuggingFace module paths.

    Maps TL-style hook names (e.g., ``"blocks.5.hook_resid_post"``) to HF module paths
    (e.g., ``"transformer.h.5"``) and input/output selectors. Supports SAE sub-hook suffix
    stripping (e.g., ``"blocks.5.hook_resid_post.hook_sae_acts_post"`` resolves to the same
    module as ``"blocks.5.hook_resid_post"``).

    Args:
        model_architecture: HuggingFace model class name (e.g., ``"GPT2LMHeadModel"``).

    Raises:
        ValueError: If the model architecture is not supported.

    Example::

        resolver = HookNameResolver("GPT2LMHeadModel")
        path, io_type = resolver.resolve("blocks.5.hook_resid_post")
        # path = "transformer.h.5", io_type = "output"

        path, io_type = resolver.resolve("blocks.3.hook_resid_post.hook_sae_acts_post")
        # path = "transformer.h.3", io_type = "output"
    """

    def __init__(self, model_architecture: str) -> None:
        try:
            self._mapping, self._derived_reasons = _mapping_for(model_architecture)
        except KeyError:
            raise ValueError(
                f"Unsupported model architecture: {model_architecture!r}. Supported architectures: "
                f"{_supported_architectures()}"
            ) from None
        self._architecture = model_architecture
        # Measured per-base-hook tuple-ness, filled by calibrate_tuple_outputs(). The static
        # `tuple_output` flags describe a transformers version, not a law: transformers 5.x decoder
        # blocks return plain tensors where 4.x returned tuples, and against a plain tensor
        # `envoy.output[0]` silently reads (and on the write path, OVERWRITES) batch row 0.
        self._measured_tuple_outputs: dict[str, bool] = {}

    def _require_mapped(self, base_name: str, layer: int | None = None) -> None:
        if base_name in self._mapping.hook_mappings:
            if layer is not None:
                self._require_layer_shape(base_name, layer)
            return
        reason = self._derived_reasons.get(base_name)
        if reason is not None:
            raise ValueError(f"Hook {base_name!r} has no module counterpart on {self._architecture!r}: {reason}")
        raise ValueError(
            f"Unknown hook name {base_name!r} for architecture {self._architecture!r}. "
            f"Supported hooks: {self.supported_hooks}"
        )

    def _require_layer_shape(self, base_name: str, layer: int) -> None:
        """A block-relative point needs a layer and a global point must not carry one.

        Accepting either the other way produced nonsense paths (`transformer.h.-1.attn.c_proj`,
        `blocks.5.unembed.hook_out`) that a caller probing the resolver for its vocabulary took as valid names;
        as the vocabulary grew, the junk grew with it and a backend refusing partial selections refused everything.
        """
        templated = "{layer}" in self._mapping.hook_mappings[base_name].envoy_path
        if templated and layer < 0:
            raise ValueError(f"Hook {base_name!r} is block-relative and needs a layer: 'blocks.<layer>.{base_name}'")
        if not templated and layer >= 0:
            raise ValueError(f"Hook {base_name!r} is a global point and takes no layer; ask for {base_name!r} bare")

    @property
    def architecture(self) -> str:
        """The model architecture this resolver is configured for."""
        return self._architecture

    @property
    def supported_hooks(self) -> list[str]:
        """List of TL base hook names supported by this architecture."""
        return sorted(self._mapping.hook_mappings.keys())

    def resolve(self, tl_hook_name: str) -> tuple[str, str]:
        """Resolve a TL hook name to an HF module path and input/output selector.

        Handles fully-qualified names including SAE sub-hook suffixes by stripping them
        before resolution.

        Args:
            tl_hook_name: TL-style hook name, e.g., ``"blocks.5.hook_resid_post"`` or
                ``"blocks.5.hook_resid_post.hook_sae_acts_post"``.

        Returns:
            Tuple of ``(module_path, io_type)`` where ``module_path`` uses concrete
            layer indices (e.g., ``"transformer.h.5"``) and ``io_type`` is ``"input"``
            or ``"output"``.

        Raises:
            ValueError: If the hook name cannot be parsed or the base hook is not supported.
        """
        layer, base_name, _ = self.parse_hook_name(tl_hook_name)
        self._require_mapped(base_name, layer)
        hook_mapping = self._mapping.hook_mappings[base_name]
        resolved_path = hook_mapping.envoy_path.format(layer=layer)
        return resolved_path, hook_mapping.io_type

    def calibrate_tuple_outputs(self, hf_model: Any) -> dict[str, bool]:
        """Measure, per output-io base hook, whether its module returns a tuple in THIS environment.

        One tiny eager forward with hooks on layer-0 modules. The static ``tuple_output`` flags encode
        the transformers version the mapping was written against; decoder blocks stopped returning
        tuples in the 5.x line, which turns ``envoy.output[0]`` from tuple-unwrapping into silently
        reading -- and on the write path, overwriting -- batch row 0. Measuring per hook rather than
        version-gating keeps this true across forks and future changes; a hook the probe cannot reach
        keeps its static flag.
        """
        import torch

        hooks = []
        measured: dict[str, bool] = {}
        for base_name, mapping in self._mapping.hook_mappings.items():
            if mapping.io_type != "output":
                continue
            try:
                module = hf_model.get_submodule(mapping.envoy_path.format(layer=0))
            except AttributeError:
                continue

            def _make(base: str):
                def _record(_mod: Any, _args: Any, out: Any) -> None:
                    measured[base] = isinstance(out, tuple)

                return _record

            hooks.append(module.register_forward_hook(_make(base_name)))
        try:
            device = next(hf_model.parameters()).device
            with torch.no_grad():
                hf_model(torch.zeros(1, 2, dtype=torch.long, device=device))
        except Exception:  # probe failure leaves the static flags in force rather than guessing
            measured = {}
        finally:
            for handle in hooks:
                handle.remove()
        self._measured_tuple_outputs = measured
        return dict(measured)

    def resolve_for_envoy(self, tl_hook_name: str) -> ResolvedHook:
        """Resolve a TL hook name to full NNsight envoy navigation information.

        Like :meth:`resolve`, but also returns the ``tuple_output`` flag needed by the
        NNsight backend to correctly read/write module activations through envoys.

        Args:
            tl_hook_name: TL-style hook name (may include SAE sub-hook suffixes).

        Returns:
            :class:`ResolvedHook` with ``module_path``, ``io_type``, and ``tuple_output``.

        Raises:
            ValueError: If the hook name cannot be parsed or the base hook is not supported.
        """
        layer, base_name, _ = self.parse_hook_name(tl_hook_name)
        self._require_mapped(base_name, layer)
        hook_mapping = self._mapping.hook_mappings[base_name]
        resolved_path = hook_mapping.envoy_path.format(layer=layer)
        # A measured flag (calibrate_tuple_outputs) beats the static default: the static value
        # describes a transformers version, and against a plain-tensor output `envoy.output[0]`
        # reads -- and on the write path overwrites -- batch row 0 instead of unwrapping a tuple.
        tuple_output = self._measured_tuple_outputs.get(base_name, hook_mapping.tuple_output)
        return ResolvedHook(
            module_path=resolved_path,
            io_type=hook_mapping.io_type,
            tuple_output=tuple_output,
        )

    def resolve_transcoder_hooks(
        self,
        hook_name: str,
        hook_name_out: str | None = None,
    ) -> tuple[tuple[str, str], tuple[str, str] | None]:
        """Resolve both input and output hooks for transcoders.

        Transcoders may have different read and write hook points (``hook_name`` for reading
        activations, ``hook_name_out`` for writing reconstructed activations).

        Args:
            hook_name: TL-style hook name for reading (the transcoder's input).
            hook_name_out: TL-style hook name for writing (the transcoder's output).
                If ``None``, defaults to the same module as ``hook_name``.

        Returns:
            Tuple of ``(read_info, write_info)`` where each is ``(module_path, io_type)``
            or ``write_info`` is ``None`` if ``hook_name_out`` is ``None``.
        """
        read_info = self.resolve(hook_name)
        write_info = self.resolve(hook_name_out) if hook_name_out else None
        return read_info, write_info

    @staticmethod
    def parse_hook_name(tl_hook_name: str) -> tuple[int, str, str | None]:
        """Parse layer index, base hook name, and optional SAE sub-hook from a TL-style hook name.

        SAE sub-hook suffixes (e.g., ``hook_sae_acts_post``) are separated from the base hook
        name and returned as the third element, rather than being discarded. This allows callers
        to distinguish between base hooks and SAE sub-hooks.

        Examples:
            - ``"blocks.5.hook_resid_post"`` → ``(5, "hook_resid_post", None)``
            - ``"blocks.5.attn.hook_z"`` → ``(5, "attn.hook_z", None)``
            - ``"blocks.5.hook_resid_post.hook_sae_acts_post"`` →
              ``(5, "hook_resid_post", "hook_sae_acts_post")``
            - ``"blocks.3.mlp.hook_pre"`` → ``(3, "mlp.hook_pre", None)``

        Args:
            tl_hook_name: TL-style hook name.

        Returns:
            Tuple of ``(layer_index, base_hook_name, sae_subhook_or_none)``.

        Raises:
            ValueError: If the hook name does not match the expected ``"blocks.{N}.{rest}"`` format.
        """
        match = _TL_HOOK_NAME_RE.match(tl_hook_name)
        if match is None:
            if "." not in tl_hook_name or tl_hook_name.startswith("blocks."):
                raise ValueError(
                    "Cannot parse TL hook name "
                    f"{tl_hook_name!r}. Expected format: 'blocks.{{layer}}.{{hook_name}}' or a "
                    "supported global hook name like 'unembed.hook_in'"
                )
            layer = -1
            rest = tl_hook_name
        else:
            layer = int(match.group(1))
            rest = match.group(2)

        # Separate base hook name from SAE sub-hook suffix
        parts = rest.split(".")
        base_parts: list[str] = []
        sae_subhook: str | None = None
        for i, part in enumerate(parts):
            if part in SUBHOOK_SUFFIXES:
                sae_subhook = ".".join(parts[i:])
                break
            base_parts.append(part)

        base_name = ".".join(base_parts)
        if not base_name:
            raise ValueError(
                f"Cannot parse TL hook name {tl_hook_name!r}. Expected format: 'blocks.{{layer}}.{{hook_name}}' or a "
                "supported global hook name like 'unembed.hook_in'"
            )
        return layer, base_name, sae_subhook

    @staticmethod
    def get_supported_architectures() -> list[str]:
        """Return list of all supported model architecture names (component maps plus hand registrations)."""
        return _supported_architectures()

    @staticmethod
    def register_architecture(mapping: ArchitectureMapping) -> None:
        """Register a hand-built mapping, which takes precedence over a derived one for its architecture.

        The supported way to add an architecture is a component-map document (``interpretune.analysis.points``);
        this remains for callers that need to override a row.
        """
        _ARCHITECTURE_REGISTRY[mapping.model_architecture] = mapping
