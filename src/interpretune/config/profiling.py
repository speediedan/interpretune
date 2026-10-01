"""Unified profiler selection for modules (interpretune#11, slice 1: config + invariant).

Two profiler backends exist (the PyTorch profiler and the bundled MemProfiler extension) with no unified selection
surface today. This config names which one runs and carries its settings; the exactly-one-active invariant is enforced
here, at construction, rather than discovered as two profilers fighting over the same module at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from interpretune.config import ITSerializableCfg
from interpretune.utils import MisconfigurationException

if TYPE_CHECKING:
    # Kept import-light: the extension pulls torch/psutil at module level.
    from interpretune.extensions.memprofiler import MemProfilerCfg


@dataclass(kw_only=True)
class ProfilerCfg(ITSerializableCfg):
    """Which profiler runs on a module, and with what settings.

    Exactly one may be active.
    """

    which: Literal["none", "pytorch", "memprofiler"] = "none"
    # Passthrough kwargs for the extended PyTorch profiler (activities, schedule, trace handler...).
    pytorch_profiler_cfg: dict[str, Any] = field(default_factory=dict)
    # MemProfiler section; None means unconfigured. An `enabled: False` section also counts as off.
    memprofiler_cfg: MemProfilerCfg | None = None

    def __post_init__(self) -> None:
        mem_active = self.memprofiler_cfg is not None and bool(getattr(self.memprofiler_cfg, "enabled", True))
        torch_active = self.which == "pytorch" or bool(self.pytorch_profiler_cfg)
        if mem_active and torch_active:
            raise MisconfigurationException(
                "ProfilerCfg refuses two active profilers: memprofiler section is configured "
                f"({self.memprofiler_cfg!r}) and the PyTorch profiler is also requested "
                f"(which={self.which!r}, pytorch_profiler_cfg keys={sorted(self.pytorch_profiler_cfg)}). "
                "Configure exactly one."
            )
        if self.which == "memprofiler" and not mem_active:
            raise MisconfigurationException(
                "ProfilerCfg which='memprofiler' needs a memprofiler_cfg section (or pick 'none')."
            )
