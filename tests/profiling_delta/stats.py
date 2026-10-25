# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Memory statistics for the interpretune-versus-upstream profiling deltas.

Two layers live here. The collector (``memory_stats``, ``StepRecorder``) is the only part that touches the
accelerator, and it does so device-generically through ``torch.accelerator`` so one code path serves CUDA and ROCm.
Everything below it (per-phase slopes, pairing an interpretune arm with its upstream arm, tolerances) is plain Python
over recorded numbers, so it is unit-tested on CPU without any device.
"""

import gc
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

import torch

# The metrics an arm reports per repeat. Peak allocated is the tight-gate metric; reserved depends far more on the
# allocator's configuration than allocated does, so it is recorded for the loose band only.
GATED_METRICS = ("peak_alloc", "end_current")
RECORDED_METRICS = ("peak_alloc", "peak_reserved", "end_current", "start_current")

MIB = 1 << 20
KIB = 1 << 10


def accelerator_type() -> str | None:
    """The current accelerator's device type (``cuda`` on both NVIDIA and ROCm builds), or None without one."""
    accel = getattr(torch, "accelerator", None)
    if accel is None or not accel.is_available():
        return None
    return accel.current_accelerator().type


def vendor() -> str:
    """The vendor key thresholds are frozen under: ``rocm``, ``cuda`` or ``cpu``."""
    if accelerator_type() is None:
        return "cpu"
    return "rocm" if getattr(torch.version, "hip", None) else accelerator_type() or "cpu"


def memory_stats() -> dict[str, Any]:
    """The accelerator's allocator statistics: the generic ``torch.accelerator`` view merged with the backend's
    own.

    The backend's ``memory_stats`` contributes keys the generic view lacks on some builds (``num_oom_rejections`` on
    CUDA, for one). Without an accelerator the result is empty, which the recorder reports as ``available: False``
    rather than as zero bytes.
    """
    dev_type = accelerator_type()
    if dev_type is None:
        return {}
    merged = dict(torch.accelerator.memory_stats())
    backend = getattr(torch, dev_type, None)
    if backend is not None and hasattr(backend, "memory_stats"):
        for key, val in backend.memory_stats().items():
            merged.setdefault(key, val)
    return merged


def reset_peak() -> None:
    if accelerator_type() is not None:
        torch.accelerator.reset_peak_memory_stats()


def release_cached() -> None:
    """Collect garbage and return cached blocks, so a repeat starts from the same allocator state as the last."""
    gc.collect()
    if accelerator_type() is not None:
        empty_cache = getattr(torch.accelerator, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()
        torch.accelerator.synchronize()


def _alloc_current(stats: dict[str, Any]) -> int:
    return int(stats.get("allocated_bytes.all.current", 0))


def device_tensor_inventory() -> list[dict[str, Any]]:
    """Live accelerator tensors the garbage collector can reach, grouped by shape and dtype, largest first.

    A diagnostic for a residual delta: two arms' inventories at the window's end, diffed, name the tensors one arm
    keeps and the other does not. Tensors reachable only from C++ (autograd's saved tensors, allocator workspaces) are
    not seen, so an inventory explains a delta only when its difference adds up to it.
    """
    groups: dict[tuple, list[int]] = {}
    seen: set[int] = set()
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor) or obj.device.type == "cpu" or obj.device.type == "meta":
                continue
            storage = obj.untyped_storage()
            key_ptr = storage.data_ptr()
        except Exception:
            continue
        if key_ptr in seen:
            continue
        seen.add(key_ptr)
        key = (tuple(obj.shape), str(obj.dtype), obj.requires_grad)
        entry = groups.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += storage.nbytes()
    rows = [
        {"shape": list(k[0]), "dtype": k[1], "requires_grad": k[2], "count": v[0], "bytes": v[1]}
        for k, v in groups.items()
    ]
    return sorted(rows, key=lambda r: -r["bytes"])


@dataclass
class StepRecorder:
    """Records ``allocated_bytes.all.current`` at the end of every step, and the peaks over the measured window.

    The first ``warmup`` steps settle the allocator and kernel caches and are not measured: the peak counters are
    reset when the last warmup step ends. Phases (an FTS schedule's depths, for one) each get their own peak, taken by
    resetting the counters at the phase boundary, so the window's peak is the largest phase peak. The window closes
    when step ``total`` ends, so whatever the loop does afterwards (validation, teardown) is outside it.
    """

    warmup: int = 2
    total: int | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    phase_peaks: dict[str, dict[str, int]] = field(default_factory=dict)
    inputs: list[list[int]] = field(default_factory=list)
    structure: dict[str, dict[str, int]] = field(default_factory=dict)
    take_inventory: bool = False
    inventory: list[dict[str, Any]] = field(default_factory=list)
    start_current: int = 0
    available: bool = False
    _phase: str = "0"
    _closed: bool = False

    def begin(self) -> None:
        stats = memory_stats()
        self.available = bool(stats)
        self.start_current = _alloc_current(stats)
        if self.warmup == 0:
            reset_peak()

    def set_phase(self, phase: Any) -> None:
        """Start a new phase.

        The open phase's peaks are recorded first, when the window has started.
        """
        phase = str(phase)
        if phase == self._phase or self._closed:
            return
        if self.measured_steps:
            self._close_phase()
            reset_peak()
        self._phase = phase

    @property
    def measured_steps(self) -> list[dict[str, Any]]:
        return [s for s in self.steps if s["measured"]]

    def note_input(self, batch: Any) -> None:
        """Record a step's input shape, so a pair can prove both arms consumed the same batches."""
        key = next((k for k in ("input_ids", "input") if k in batch), None)
        if key is None:  # the widest tensor is the token batch; labels are one value per example
            key = max((k for k, v in batch.items() if hasattr(v, "shape")), key=lambda k: batch[k].dim())
        self.inputs.append([key, *batch[key].shape])

    def note_structure(self, counts: dict[str, int]) -> None:
        """Record structural counts for the current phase; the last step of each phase is the one kept."""
        self.structure[self._phase] = counts

    def step_end(self) -> None:
        if accelerator_type() is not None:
            torch.accelerator.synchronize()
        idx = len(self.steps)
        measured = idx >= self.warmup and (self.total is None or idx < self.total)
        self.steps.append(
            {"idx": idx, "phase": self._phase, "current": _alloc_current(memory_stats()), "measured": measured}
        )
        if idx + 1 == self.warmup:
            reset_peak()
        if self.total is not None and idx + 1 == self.total:
            self._close_phase()
            self._closed = True
            if self.take_inventory:
                self.inventory = device_tensor_inventory()

    def _close_phase(self) -> None:
        stats = memory_stats()
        self.phase_peaks[self._phase] = {
            "peak_alloc": int(stats.get("allocated_bytes.all.peak", 0)),
            "peak_reserved": int(stats.get("reserved_bytes.all.peak", 0)),
        }

    def finish(self) -> dict[str, Any]:
        """The repeat's record, as written to the arm's JSON output."""
        if self.measured_steps and not self._closed:
            self._close_phase()
        peaks = list(self.phase_peaks.values())
        return {
            "available": self.available,
            "start_current": self.start_current,
            "peak_alloc": max((p["peak_alloc"] for p in peaks), default=0),
            "peak_reserved": max((p["peak_reserved"] for p in peaks), default=0),
            "end_current": self.measured_steps[-1]["current"] if self.measured_steps else 0,
            "phase_peaks": self.phase_peaks,
            "inputs": self.inputs,
            "structure": self.structure,
            "inventory": self.inventory,
            "steps": self.steps,
        }


########################################################################################################################
# Analysis: plain Python over recorded numbers
########################################################################################################################


def slope(values: list[float]) -> float:
    """Least-squares slope of ``values`` against their index, in units per step (0.0 for fewer than two points)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den


# Steps after a phase's transition step that are still allocator settling, not steady state: measured on gpt2 under
# FTSController, both arms move by up to ~0.8 MiB at the step after a thaw and are flat from the one after.
PHASE_SETTLE_STEPS = 1


def phase_slopes(steps: list[dict[str, Any]], settle: int = PHASE_SETTLE_STEPS) -> dict[str, float]:
    """Per-phase leak slope over each phase's steady-state steps.

    A later phase's transition step and the ``settle`` steps after it are excluded. A transition legitimately steps
    bytes up (new parameter groups, and optimizer state at the first step after a thaw), so a slope fitted across
    phases reads a healthy thaw as a leak, and the allocator takes a step to settle after it. The transition step is
    reported by ``transition_steps`` instead.
    """
    measured = [s for s in steps if s["measured"]]
    by_phase: dict[str, list[float]] = {}
    since_transition = None  # None until the first transition: the window's own warmup already covers phase 0
    for i, step in enumerate(measured):
        if i > 0 and measured[i - 1]["phase"] != step["phase"]:
            since_transition = 0
        elif since_transition is not None:
            since_transition += 1
        if since_transition is None or since_transition > settle:
            by_phase.setdefault(step["phase"], []).append(step["current"])
    return {phase: slope(vals) for phase, vals in by_phase.items()}


def transition_steps(steps: list[dict[str, Any]]) -> dict[str, int]:
    """Bytes stepped up at each measured phase boundary, keyed ``"<from>-><to>"``."""
    measured = [s for s in steps if s["measured"]]
    out = {}
    for prev, cur in zip(measured, measured[1:]):
        if prev["phase"] != cur["phase"]:
            out[f"{prev['phase']}->{cur['phase']}"] = cur["current"] - prev["current"]
    return out


def summarize_repeat(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one repeat's record into the scalar metrics the pairing compares."""
    summary = {m: record[m] for m in RECORDED_METRICS}
    summary.update({f"slope[{p}]": v for p, v in phase_slopes(record["steps"]).items()})
    summary.update({f"transition[{k}]": v for k, v in transition_steps(record["steps"]).items()})
    return summary


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, which stays an observed value for the handful of repeats a pair runs."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[rank]


def pair_deltas(it_repeats: list[dict[str, Any]], upstream_repeats: list[dict[str, Any]]) -> dict[str, dict]:
    """Interpretune minus upstream, per metric: each repeat's delta, their median, and the repeat spread.

    Repeats pair by index (the arms run the same seeds in the same order). The spread is the largest absolute deviation
    of a repeat's delta from the median, the noise a frozen tolerance has to clear.
    """
    if len(it_repeats) != len(upstream_repeats):
        raise ValueError(f"arms ran {len(it_repeats)} and {len(upstream_repeats)} repeats; they must match")
    it_sum = [summarize_repeat(r) for r in it_repeats]
    up_sum = [summarize_repeat(r) for r in upstream_repeats]
    metrics = sorted(set(it_sum[0]) & set(up_sum[0])) if it_sum else []
    out = {}
    for metric in metrics:
        deltas = [i[metric] - u[metric] for i, u in zip(it_sum, up_sum)]
        med = statistics.median(deltas)
        out[metric] = {
            "deltas": deltas,
            "median": med,
            "spread": max(abs(d - med) for d in deltas),
            "it_median": statistics.median(i[metric] for i in it_sum),
            "upstream_median": statistics.median(u[metric] for u in up_sum),
        }
    return out


def default_floor(metric: str) -> float:
    """The floor a tolerance never drops below: 8 MiB on byte metrics, 64 KiB per step on slopes."""
    return 64 * KIB if metric.startswith("slope[") else 8 * MIB


def derive_tolerance(spreads: list[float], metric: str, multiplier: float = 3.0) -> float:
    """``max(multiplier x p95(spread), floor)`` over the spreads observed across measurement runs (and vendors)."""
    return max(multiplier * percentile(spreads, 95), default_floor(metric))


def gated(metric: str) -> bool:
    return metric in GATED_METRICS or metric.startswith(("slope[", "transition["))


def evaluate(deltas: dict[str, dict], frozen: dict[str, dict] | None) -> list[str]:
    """Failures of a measured pair against its frozen record (``{metric: {"baseline": b, "tolerance": t}}``).

    A gated metric fails when its median delta exceeds the frozen baseline by more than the tolerance. Only growth
    fails: interpretune using less memory than it used to is not a regression. Without a frozen record nothing is
    evaluated, which is the report-only state every composition starts in.
    """
    failures = []
    for metric, rec in (frozen or {}).items():
        if not gated(metric) or metric not in deltas:
            continue
        excess = deltas[metric]["median"] - rec["baseline"]
        if excess > rec["tolerance"]:
            failures.append(
                f"{metric}: delta {deltas[metric]['median']:.0f} exceeds baseline {rec['baseline']:.0f} by "
                f"{excess:.0f} (tolerance {rec['tolerance']:.0f})"
            )
    return failures
