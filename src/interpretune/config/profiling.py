"""Profiler selection for modules run by interpretune's core runners (interpretune#11).

The core runners (session train/test and analysis) wrap their batch loops in ``torch.profiler.profile`` when
``profiler_cfg.which == "pytorch"``, stepping the profiler once per batch. The bundled MemProfiler extension keeps its
own configuration (``it_cfg.memprofiler_cfg``); ``ITConfig`` refuses enabling both, so exactly one profiler observes a
module. Lightning compositions use Lightning's native ``--trainer.profiler`` and refuse this selection by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from interpretune.config.shared import ITSerializableCfg
from interpretune.utils import MisconfigurationException

#: ``torch.profiler.profile`` options accepted from configuration. Each is serializable, so a profiler selection
#: round-trips through YAML and the CLI; ``trace_dir`` replaces the non-serializable ``on_trace_ready`` callable.
PYTORCH_PROFILER_KEYS = frozenset(
    {
        "activities",
        "schedule",
        "record_shapes",
        "profile_memory",
        "with_stack",
        "with_flops",
        "with_modules",
        "trace_dir",
    }
)
_ACTIVITIES = ("cpu", "cuda", "xpu", "mtia")


@dataclass(kw_only=True)
class ProfilerCfg(ITSerializableCfg):
    """Whether the core runners profile a module with the PyTorch profiler, and with what settings.

    Example (module config YAML)::

        profiler_cfg:
          which: pytorch
          pytorch_profiler_cfg:
            activities: [cpu, cuda]
            schedule: {wait: 1, warmup: 1, active: 3}
            record_shapes: true

    ``schedule`` takes the keyword arguments of ``torch.profiler.schedule``. Traces are written with
    ``torch.profiler.tensorboard_trace_handler`` to ``trace_dir``, defaulting to ``<core_log_dir>/pytorch_profiler``.
    The core CLI accepts the same settings as ``--profiler_cfg.which pytorch`` and friends.
    """

    which: Literal["none", "pytorch"] = "none"
    # Options for torch.profiler.profile, restricted to PYTORCH_PROFILER_KEYS.
    pytorch_profiler_cfg: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pytorch_profiler_cfg and self.which != "pytorch":
            raise MisconfigurationException(
                f"profiler_cfg.pytorch_profiler_cfg is set but which={self.which!r}: the settings would configure a "
                "profiler that never runs. Set which='pytorch' or drop the settings."
            )
        unknown = sorted(set(self.pytorch_profiler_cfg) - PYTORCH_PROFILER_KEYS)
        if unknown:
            raise MisconfigurationException(
                f"profiler_cfg.pytorch_profiler_cfg has unsupported keys {unknown}; accepted: "
                f"{sorted(PYTORCH_PROFILER_KEYS)}."
            )
        bad_activities = sorted(set(self.pytorch_profiler_cfg.get("activities", ())) - set(_ACTIVITIES))
        if bad_activities:
            raise MisconfigurationException(
                f"profiler_cfg.pytorch_profiler_cfg.activities has unknown entries {bad_activities}; accepted: "
                f"{list(_ACTIVITIES)}."
            )

    def torch_profiler_kwargs(self, default_trace_dir: str | Path) -> dict[str, Any]:
        """Keyword arguments for ``torch.profiler.profile`` built from these settings.

        Args:
            default_trace_dir: Where traces are written when ``trace_dir`` is not configured.
        """
        import torch.profiler as tp

        opts = dict(self.pytorch_profiler_cfg)
        kwargs: dict[str, Any] = {k: v for k, v in opts.items() if k not in ("activities", "schedule", "trace_dir")}
        if "activities" in opts:
            kwargs["activities"] = [getattr(tp.ProfilerActivity, a.upper()) for a in opts["activities"]]
        if "schedule" in opts:
            kwargs["schedule"] = tp.schedule(**opts["schedule"])
        kwargs["on_trace_ready"] = tp.tensorboard_trace_handler(str(opts.get("trace_dir", default_trace_dir)))
        return kwargs
