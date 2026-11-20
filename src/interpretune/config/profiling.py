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

#: Options for the text summary written after a profiled loop, named as finetuning-scheduler's example
#: ``ExtendedPyTorchProfiler`` (and Lightning's ``PyTorchProfiler``) name them, so one set of settings reads the same on
#: both interfaces. The summary is written only when ``filename`` is set.
SUMMARY_KEYS = frozenset({"filename", "sort_by_key", "row_limit", "max_name_column_width"})
#: ``torch.profiler.profile`` options accepted from configuration, plus the summary options. Each is serializable, so a
#: profiler selection round-trips through YAML and the CLI; ``trace_dir`` replaces the non-serializable
#: ``on_trace_ready`` callable.
PYTORCH_PROFILER_KEYS = (
    frozenset(
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
    | SUMMARY_KEYS
)
_ACTIVITIES = ("cpu", "cuda", "xpu", "mtia")
_SUMMARY_DEFAULTS = {"sort_by_key": "cpu_time_total", "row_limit": 20, "max_name_column_width": 55}


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
            filename: rte_profile
            sort_by_key: self_cpu_time_total

    ``schedule`` takes the keyword arguments of ``torch.profiler.schedule``. Traces are written with
    ``torch.profiler.tensorboard_trace_handler`` to ``trace_dir``, defaulting to ``<core_log_dir>/pytorch_profiler``.
    With ``filename`` set, the loop also writes a table of the last profiled window's operators to
    ``<trace_dir>/<phase>-<filename>.txt`` (``-<rank>`` appended in a distributed run), sorted by ``sort_by_key``
    (default ``cpu_time_total``) and cut to ``row_limit`` rows (default 20) and ``max_name_column_width`` characters
    (default 55). The core CLI accepts the same settings as ``--profiler_cfg.which pytorch`` and friends.
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
        self._check_summary_options()

    def _check_summary_options(self) -> None:
        opts = self.pytorch_profiler_cfg
        shaping = sorted((SUMMARY_KEYS - {"filename"}) & set(opts))
        if shaping and not opts.get("filename"):
            raise MisconfigurationException(
                f"profiler_cfg.pytorch_profiler_cfg sets {shaping} but no `filename`: those options shape a summary "
                "that is written only when `filename` names it. Set `filename` or drop them."
            )
        for key in ("row_limit", "max_name_column_width"):
            value = opts.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise MisconfigurationException(
                    f"profiler_cfg.pytorch_profiler_cfg.{key} must be a positive integer, got {value!r}."
                )
        sort_by_key = opts.get("sort_by_key")
        if sort_by_key is not None:
            from torch.autograd.profiler_util import FunctionEventAvg

            # torch reads the sort key as an attribute of each averaged event, after mapping an accelerator's name to
            # `device` (`cuda_time_total` -> `device_time_total`); an unknown key would fail only once the run ends.
            attr = str(sort_by_key)
            for accelerator in ("cuda", "xpu", "privateuse1"):
                attr = attr.replace(accelerator, "device")
            if not hasattr(FunctionEventAvg(), attr):
                raise MisconfigurationException(
                    f"profiler_cfg.pytorch_profiler_cfg.sort_by_key={sort_by_key!r} is not a column of the profiler's "
                    "summary; use one such as 'cpu_time_total', 'self_cpu_time_total', 'cuda_time_total', "
                    "'self_cuda_time_total', 'cpu_memory_usage' or 'count'."
                )

    def summary_options(self) -> dict[str, Any] | None:
        """The summary table's settings with their defaults filled in, or ``None`` when no summary is written."""
        filename = self.pytorch_profiler_cfg.get("filename")
        if not filename:
            return None
        return {
            "filename": str(filename),
            **{k: self.pytorch_profiler_cfg.get(k, default) for k, default in _SUMMARY_DEFAULTS.items()},
        }

    def torch_profiler_kwargs(self, default_trace_dir: str | Path) -> dict[str, Any]:
        """Keyword arguments for ``torch.profiler.profile`` built from these settings.

        Args:
            default_trace_dir: Where traces are written when ``trace_dir`` is not configured.
        """
        import torch.profiler as tp

        opts = dict(self.pytorch_profiler_cfg)
        passthrough = PYTORCH_PROFILER_KEYS - SUMMARY_KEYS - {"activities", "schedule", "trace_dir"}
        kwargs: dict[str, Any] = {k: v for k, v in opts.items() if k in passthrough}
        if "activities" in opts:
            kwargs["activities"] = [getattr(tp.ProfilerActivity, a.upper()) for a in opts["activities"]]
        if "schedule" in opts:
            kwargs["schedule"] = tp.schedule(**opts["schedule"])
        kwargs["on_trace_ready"] = tp.tensorboard_trace_handler(str(opts.get("trace_dir", default_trace_dir)))
        return kwargs
