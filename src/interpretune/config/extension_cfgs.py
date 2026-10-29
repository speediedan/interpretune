"""Configuration classes of the bundled extensions, kept apart from the extensions themselves (#624).

``interpretune.config`` builds ``ExtensionConf`` from these classes when it is imported, so this module imports nothing
an extension's implementation needs (``datasets``, ``transformers.generation``, the analysis stack). It sits inside the
config package so that importing an extension module first does not re-enter a partially initialized config package.
The extension modules re-export their configuration classes, so ``class_path`` references to the old locations keep
resolving.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

import numpy as np
import torch

from interpretune.config.shared import ITSerializableCfg
from interpretune.protocol import AutoStrEnum, CoreSteps
from interpretune.utils import rank_zero_warn


@dataclass(kw_only=True)
class DebugLMConfig(ITSerializableCfg):
    """Configuration for the generation-debugging extension."""

    enabled: bool = False
    debug_raw_preds: np.ndarray | None = None
    debug_raw_labels: np.ndarray | None = None
    debug_raw_sequences: list[str] | None = None
    raw_debug_sequences: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.raw_debug_sequences) == 0 and self.enabled:
            self.raw_debug_sequences = [
                "What is the color of a banana?",
                "List the first 5 letters in the alphabet.",
                "How many days in a week?",
                "How old is Barack Obama?",
            ]


class DefaultMemHooks(AutoStrEnum):
    """Import paths of the built-in memory hooks, used as defaults in :class:`MemProfilerHooks`."""

    pre_forward = "interpretune.extensions.memprofiler._hook_npp_pre_forward"
    post_forward = "interpretune.extensions.memprofiler._hook_npp_post_forward"
    reset_state = "interpretune.extensions.memprofiler._reset_memory_hooks_state"


@dataclass(kw_only=True)
class MemProfilerHooks(ITSerializableCfg):
    """Hooks the profiler installs, each given as an import path or a callable.

    ``reset_state_hooks`` differ in signature from the forward hooks: they are called with the model
    and the configured ``save_hook_attrs`` list, so they can clear exactly the attributes collection
    reads.
    """

    pre_forward_hooks: list[str | Callable] = field(default_factory=lambda: [DefaultMemHooks.pre_forward.value])
    post_forward_hooks: list[str | Callable] = field(default_factory=lambda: [DefaultMemHooks.post_forward.value])
    # the provided reset_state_hooks will be called with the model and the `save_hook_attrs` list
    reset_state_hooks: list[str | Callable] = field(default_factory=lambda: [DefaultMemHooks.reset_state.value])


@dataclass(kw_only=True)
class MemProfilerFuncs(ITSerializableCfg):
    """Which ``memprofilable``-decorated functions to profile, per collection type.

    Defaults to every core step. The three lists are independent so, for instance, CUDA allocator history (which is
    expensive) can be narrowed without also narrowing cheap CPU sampling.
    """

    # can specify arbitrary list of `memprofilable` decorated function names
    cuda: list[str | Enum] = field(default_factory=lambda: list(step.name for step in CoreSteps))
    cpu: list[str | Enum] = field(default_factory=lambda: list(step.name for step in CoreSteps))
    cuda_allocator_history: list[str | Enum] = field(default_factory=lambda: list(step.name for step in CoreSteps))


@dataclass(kw_only=True)
class MemProfilerSchedule(ITSerializableCfg):
    """When profiling is active: skip ``warmup_steps``, then collect until ``max_step``."""

    # keeping schedule simple as possible for now, may expand to accommodate more flexible schedules in the future
    warmup_steps: int = 0
    max_step: int | None = None


@dataclass(kw_only=True)
class MemProfilerCfg(ITSerializableCfg):
    """Configuration for the memory profiler extension."""

    enabled: bool = False
    cuda_allocator_history: bool = False
    schedule: MemProfilerSchedule = field(default_factory=MemProfilerSchedule)
    save_dir: str | os.PathLike | None = None
    enabled_funcs: MemProfilerFuncs = field(default_factory=MemProfilerFuncs)
    enable_memory_hooks: bool = True
    enable_saved_tensors_hooks: bool = True
    memory_hooks: MemProfilerHooks = field(default_factory=MemProfilerHooks)
    saved_tensors_funcs: list = field(
        default_factory=lambda: list(("interpretune.extensions.memprofiler._npp_hook", lambda x: x))
    )
    # if you add custom hooks, make sure to add the desired module state attributes to save to `save_hook_attrs`
    save_hook_attrs: list = field(
        default_factory=lambda: [
            "rss_pre_forward",
            "rss_post_forward",
            "rss_diff",
            "npp_pre_forward",
            "npp_post_forward",
            "npp_diff",
        ]
    )
    # since we cannot reliably ascertain when all MemProfilerFuncs will be executed, memory hooks will
    # only be removed once the funcs in this list have reached `max_step`
    retain_hooks_for_funcs: list[str | Enum] = field(default_factory=lambda: list(step.name for step in CoreSteps))

    def __post_init__(self) -> None:
        if (
            not torch.cuda.is_available()
            and self.enabled
            and any((self.enabled_funcs.cuda_allocator_history, self.enabled_funcs.cuda, self.cuda_allocator_history))
        ):
            rank_zero_warn("Disabling CUDA memory profiling functionality since no CUDA device detected.")
            self.enabled_funcs.cuda, self.enabled_funcs.cuda_allocator_history = [], []
            self.cuda_allocator_history = False
        has_hooks = any(getattr(self.memory_hooks, ht.name) for ht in fields(self.memory_hooks))
        if self.enabled and not has_hooks:
            rank_zero_warn(
                "MemProfilerCfg is configured to enable memory hooks but MemProfilerHooks does not have any specified."
            )


# defined whether or not neuronpedia is installed; `ExtensionConf` includes it only when it is
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
    default_metadata: dict[str, Any] = field(
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
