"""Configuration classes of the bundled extensions, kept apart from the extensions themselves (#624).

``interpretune.config`` builds ``ExtensionConf`` from these classes when it is imported, so this module imports nothing
an extension's implementation needs (``datasets``, ``transformers.generation``, the analysis stack). It sits inside the
config package so that importing an extension module first does not re-enter a partially initialized config package.
The extension modules re-export their configuration classes, so ``class_path`` references to the old locations keep
resolving.
"""

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

import numpy as np

from interpretune.config.shared import ITSerializableCfg
from interpretune.protocol import AutoStrEnum, CoreSteps
from interpretune.utils import MisconfigurationException, _FTS_AVAILABLE


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
    """Import paths of finetuning-scheduler's built-in memory hooks, the defaults in :class:`MemProfilerHooks`."""

    pre_forward = "finetuning_scheduler.profiling.npp_hooks._hook_npp_pre_forward"
    post_forward = "finetuning_scheduler.profiling.npp_hooks._hook_npp_post_forward"
    reset_state = "finetuning_scheduler.profiling.npp_hooks._reset_memory_hooks_state"


# The memory profiler is finetuning-scheduler's framework-neutral core (`finetuning_scheduler.profiling`), and these
# classes carry its configuration schema field for field, so a configuration reads the same in both projects. They are
# defined here rather than imported because finetuning-scheduler is an optional dependency (the `tune` extra) and
# `ExtensionConf` is built whenever `interpretune.config` is imported. `MemProfilerCfg.to_core` builds the core's own
# configuration, which validates it; a test keeps the two schemas' field names identical.
MEMPROFILER_EXTRA = "tune"


def _all_core_steps() -> list[str | Enum]:
    return [step.name for step in CoreSteps]


@dataclass(kw_only=True)
class MemProfilerHooks(ITSerializableCfg):
    """Hooks the profiler installs, each given as an import path or a callable.

    ``reset_state_hooks`` differ in signature from the forward hooks: they are called with the model and the configured
    ``save_hook_attrs`` list, so they can clear exactly the attributes collection reads.
    """

    pre_forward_hooks: list[str | Callable] = field(default_factory=lambda: [DefaultMemHooks.pre_forward.value])
    post_forward_hooks: list[str | Callable] = field(default_factory=lambda: [DefaultMemHooks.post_forward.value])
    reset_state_hooks: list[str | Callable] = field(default_factory=lambda: [DefaultMemHooks.reset_state.value])


@dataclass(kw_only=True)
class MemProfilerFuncs(ITSerializableCfg):
    """Which ``memprofilable``-decorated functions to profile, per collection type.

    ``default`` is added to every collection type. It defaults to every core step, since an interpretune session runs
    test and analysis steps as often as training ones.
    """

    default: list[str | Enum] = field(default_factory=_all_core_steps)
    cpu: list[str | Enum] = field(default_factory=list)
    accelerator: list[str | Enum] = field(default_factory=list)
    allocator_history: list[str | Enum] = field(default_factory=list)
    fsdp: list[str | Enum] = field(default_factory=list)


@dataclass(kw_only=True)
class MemProfilerSchedule(ITSerializableCfg):
    """When profiling is active: skip ``warmup_iters`` iterations, then collect through iteration ``max_iter``.

    Iterations count from 1. An unset ``max_iter`` collects one iteration after the warmup.
    """

    warmup_iters: int = 0
    max_iter: int | None = None


@dataclass(kw_only=True)
class MemProfilerCfg(ITSerializableCfg):
    """Configuration for the memory profiler extension, in finetuning-scheduler's ``MemProfilerCfg`` schema.

    Enabling it requires finetuning-scheduler, installed with interpretune's ``tune`` extra; without it the
    configuration is refused naming the extra.
    """

    enabled: bool = False
    collect_funcs: MemProfilerFuncs = field(default_factory=MemProfilerFuncs)
    allocator_history: bool = False
    track_fsdp_mem: bool = False
    fsdp_mem_track_module_depth: int = 2
    fsdp_mem_tracker_tabulate: bool = False
    fsdp_mem_tracker_units: str = "MiB"
    fsdp_mem_tracker_root_module: str = ""
    dump_fsdp_mem_json: bool = True
    aggregate_fsdp_mem_reports: bool = False
    dump_memorystats_pickle: bool = False
    dump_memorystats_yaml: bool = True
    schedule: MemProfilerSchedule = field(default_factory=MemProfilerSchedule)
    save_dir: str | os.PathLike | None = None
    enable_memory_hooks: bool = True
    enable_saved_tensors_hooks: bool = True
    memory_hooks: MemProfilerHooks = field(default_factory=MemProfilerHooks)
    saved_tensors_funcs: list = field(
        default_factory=lambda: ["finetuning_scheduler.profiling.npp_hooks._npp_hook", "identity_lambda"]
    )
    # if you add custom hooks, add the module state attributes they set to `save_hook_attrs`
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
    retain_hooks_for_funcs: list[str | Enum] = field(default_factory=_all_core_steps)

    def __post_init__(self) -> None:
        if self.enabled and not _FTS_AVAILABLE:
            raise MisconfigurationException(
                "The memory profiler runs on finetuning-scheduler's profiling core, which is not installed. Install "
                f"the `{MEMPROFILER_EXTRA}` extra (`pip install 'interpretune[{MEMPROFILER_EXTRA}]'`) to enable "
                "`memprofiler_cfg`."
            )

    def to_core(self) -> Any:
        """Build finetuning-scheduler's ``MemProfilerCfg`` from this configuration.

        The core's configuration validates on construction: it composes ``collect_funcs.default`` into every collection
        type, sets an unset ``schedule.max_iter``, and disables accelerator collection (with a warning when it was
        requested) on a host without a supported accelerator.
        """
        from finetuning_scheduler.profiling import config as core_cfg

        def as_names(funcs: list) -> set[str]:
            return {f.name if isinstance(f, Enum) else str(f) for f in funcs}

        values = {f.name: getattr(self, f.name) for f in fields(self)}
        funcs = self.collect_funcs
        values["collect_funcs"] = core_cfg.MemProfilerFuncs(
            **{f.name: as_names(getattr(funcs, f.name)) for f in fields(funcs)}
        )
        values["schedule"] = core_cfg.MemProfilerSchedule(**asdict(self.schedule))
        values["memory_hooks"] = core_cfg.MemProfilerHooks(
            **{f.name: list(getattr(self.memory_hooks, f.name)) for f in fields(self.memory_hooks)}
        )
        values["saved_tensors_funcs"] = list(self.saved_tensors_funcs)
        values["save_hook_attrs"] = list(self.save_hook_attrs)
        values["retain_hooks_for_funcs"] = as_names(self.retain_hooks_for_funcs)
        return core_cfg.MemProfilerCfg(**values)


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
                "creator_url": "https://github.com/interpretune/interpretune",
                "generator": {
                    "name": "Interpretune Circuit Tracer",
                    "version": "latest",
                    "url": "https://github.com/interpretune/interpretune",
                },
            },
        }
    )
