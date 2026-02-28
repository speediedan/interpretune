"""Hook name resolver for mapping between TransformerLens and nnsight/HuggingFace hook names.

Provides the ``HookNameResolver`` class and architecture-specific mappings that translate TL-style hook names
(e.g., ``"blocks.5.hook_resid_post"``) to HuggingFace module paths (e.g., ``"transformer.h.5"``) and
input/output selectors.

Architecture mappings follow the same pattern as ``circuit-tracer``'s ``tl_nnsight_mapping.py``, using
``{layer}`` placeholders in path templates that are resolved to concrete layer indices at resolution time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, NamedTuple


# Known SAE sub-hook suffixes that should be stripped when resolving to the base model hook
_SAE_SUBHOOK_SUFFIXES = frozenset(
    {
        "hook_sae_input",
        "hook_sae_acts_pre",
        "hook_sae_acts_post",
        "hook_sae_output",
        "hook_sae_error",
    }
)


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


# ==============================================================================
# Architecture-specific mappings
# ==============================================================================


GPT2_HOOK_MAPPINGS: dict[str, HookMapping] = {
    # Residual stream hooks
    "hook_resid_pre": HookMapping(envoy_path="transformer.h.{layer}", io_type="input"),
    "hook_resid_post": HookMapping(envoy_path="transformer.h.{layer}", io_type="output"),
    "hook_resid_mid": HookMapping(envoy_path="transformer.h.{layer}.ln_2", io_type="input"),
    # Component output hooks
    "hook_attn_out": HookMapping(envoy_path="transformer.h.{layer}.attn", io_type="output"),
    "hook_mlp_out": HookMapping(envoy_path="transformer.h.{layer}.mlp", io_type="output", tuple_output=False),
    # Attention internal hooks (hook_z = attention output before output projection)
    "attn.hook_z": HookMapping(envoy_path="transformer.h.{layer}.attn.c_proj", io_type="input"),
    # Component input hooks
    "mlp.hook_pre": HookMapping(envoy_path="transformer.h.{layer}.mlp", io_type="input"),
}

GPT2_MAPPING = ArchitectureMapping(
    model_architecture="GPT2LMHeadModel",
    hook_mappings=GPT2_HOOK_MAPPINGS,
)


# Llama family
LLAMA_HOOK_MAPPINGS: dict[str, HookMapping] = {
    "hook_resid_pre": HookMapping(envoy_path="model.layers.{layer}", io_type="input"),
    "hook_resid_post": HookMapping(envoy_path="model.layers.{layer}", io_type="output"),
    "hook_resid_mid": HookMapping(envoy_path="model.layers.{layer}.post_attention_layernorm", io_type="input"),
    "hook_mlp_out": HookMapping(envoy_path="model.layers.{layer}.mlp", io_type="output", tuple_output=False),
    "hook_attn_out": HookMapping(envoy_path="model.layers.{layer}.self_attn", io_type="output"),
    # Attention internal hooks (hook_z = attention output before output projection)
    "attn.hook_z": HookMapping(envoy_path="model.layers.{layer}.self_attn.o_proj", io_type="input"),
    "mlp.hook_pre": HookMapping(envoy_path="model.layers.{layer}.mlp", io_type="input"),
    "mlp.hook_in": HookMapping(
        envoy_path="model.layers.{layer}.post_attention_layernorm", io_type="output", tuple_output=False
    ),
    "mlp.hook_out": HookMapping(envoy_path="model.layers.{layer}.mlp", io_type="output", tuple_output=False),
}

LLAMA_MAPPING = ArchitectureMapping(
    model_architecture="LlamaForCausalLM",
    hook_mappings=LLAMA_HOOK_MAPPINGS,
)


# Gemma 2
GEMMA2_HOOK_MAPPINGS: dict[str, HookMapping] = {
    "hook_resid_pre": HookMapping(envoy_path="model.layers.{layer}", io_type="input"),
    "hook_resid_post": HookMapping(envoy_path="model.layers.{layer}", io_type="output"),
    "hook_resid_mid": HookMapping(envoy_path="model.layers.{layer}.pre_feedforward_layernorm", io_type="input"),
    "hook_mlp_out": HookMapping(
        envoy_path="model.layers.{layer}.post_feedforward_layernorm", io_type="output", tuple_output=False
    ),
    "hook_attn_out": HookMapping(envoy_path="model.layers.{layer}.self_attn", io_type="output"),
    # Attention internal hooks (hook_z = attention output before output projection)
    "attn.hook_z": HookMapping(envoy_path="model.layers.{layer}.self_attn.o_proj", io_type="input"),
    "ln2.hook_normalized": HookMapping(
        envoy_path="model.layers.{layer}.pre_feedforward_layernorm", io_type="output", tuple_output=False
    ),
}

GEMMA2_MAPPING = ArchitectureMapping(
    model_architecture="Gemma2ForCausalLM",
    hook_mappings=GEMMA2_HOOK_MAPPINGS,
)


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


# Registry of all supported architectures
_ARCHITECTURE_REGISTRY: dict[str, ArchitectureMapping] = {
    mapping.model_architecture: mapping for mapping in [GPT2_MAPPING, LLAMA_MAPPING, GEMMA2_MAPPING]
}


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
        if model_architecture not in _ARCHITECTURE_REGISTRY:
            supported = sorted(_ARCHITECTURE_REGISTRY.keys())
            raise ValueError(
                f"Unsupported model architecture: {model_architecture!r}. Supported architectures: {supported}"
            )
        self._architecture = model_architecture
        self._mapping = _ARCHITECTURE_REGISTRY[model_architecture]

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
        if base_name not in self._mapping.hook_mappings:
            raise ValueError(
                f"Unknown hook name {base_name!r} for architecture {self._architecture!r}. "
                f"Supported hooks: {self.supported_hooks}"
            )
        hook_mapping = self._mapping.hook_mappings[base_name]
        resolved_path = hook_mapping.envoy_path.format(layer=layer)
        return resolved_path, hook_mapping.io_type

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
        if base_name not in self._mapping.hook_mappings:
            raise ValueError(
                f"Unknown hook name {base_name!r} for architecture {self._architecture!r}. "
                f"Supported hooks: {self.supported_hooks}"
            )
        hook_mapping = self._mapping.hook_mappings[base_name]
        resolved_path = hook_mapping.envoy_path.format(layer=layer)
        return ResolvedHook(
            module_path=resolved_path,
            io_type=hook_mapping.io_type,
            tuple_output=hook_mapping.tuple_output,
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
            raise ValueError(
                f"Cannot parse TL hook name {tl_hook_name!r}. Expected format: 'blocks.{{layer}}.{{hook_name}}'"
            )

        layer = int(match.group(1))
        rest = match.group(2)

        # Separate base hook name from SAE sub-hook suffix
        parts = rest.split(".")
        base_parts: list[str] = []
        sae_subhook: str | None = None
        for i, part in enumerate(parts):
            if part in _SAE_SUBHOOK_SUFFIXES:
                sae_subhook = ".".join(parts[i:])
                break
            base_parts.append(part)

        base_name = ".".join(base_parts)
        return layer, base_name, sae_subhook

    @staticmethod
    def get_supported_architectures() -> list[str]:
        """Return list of all supported model architecture names."""
        return sorted(_ARCHITECTURE_REGISTRY.keys())

    @staticmethod
    def register_architecture(mapping: ArchitectureMapping) -> None:
        """Register a new architecture mapping.

        Args:
            mapping: The architecture mapping to register.
        """
        _ARCHITECTURE_REGISTRY[mapping.model_architecture] = mapping


__all__ = [
    "ArchitectureMapping",
    "HookMapping",
    "HookNameResolver",
    "ResolvedHook",
]
