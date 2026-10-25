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
"""CPU tests for the profiling delta analysis: no accelerator, recorded numbers only."""

import json

import pytest

from tests.profiling_delta import freeze, runner
from tests.profiling_delta.stats import (
    MIB,
    StepRecorder,
    derive_tolerance,
    evaluate,
    pair_deltas,
    percentile,
    phase_slopes,
    slope,
    transition_steps,
)


def _steps(currents, phases=None, warmup=0):
    phases = phases or ["0"] * len(currents)
    return [
        {"idx": i, "phase": p, "current": c, "measured": i >= warmup} for i, (c, p) in enumerate(zip(currents, phases))
    ]


def _record(currents, peak, phases=None, warmup=0, end=None, start=0):
    steps = _steps(currents, phases, warmup)
    return {
        "available": True,
        "start_current": start,
        "peak_alloc": peak,
        "peak_reserved": peak * 2,
        "end_current": end if end is not None else currents[-1],
        "phase_peaks": {},
        "steps": steps,
    }


def test_slope():
    assert slope([]) == 0.0
    assert slope([5]) == 0.0
    assert slope([0, 2, 4, 6]) == pytest.approx(2.0)
    assert slope([3, 3, 3]) == 0.0


def test_phase_slopes_exclude_transition_step():
    # a thaw steps memory up once; per-phase slopes stay flat where an across-run fit would report a leak
    currents = [100, 100, 100, 200, 200, 200]
    steps = _steps(currents, ["0", "0", "0", "1", "1", "1"])
    assert slope(currents) > 0
    assert phase_slopes(steps) == {"0": 0.0, "1": 0.0}
    assert transition_steps(steps) == {"0->1": 100}


def test_phase_slopes_skip_warmup():
    steps = _steps([0, 50, 100, 101, 102, 103], warmup=2)
    assert phase_slopes(steps) == {"0": pytest.approx(1.0)}
    assert transition_steps(steps) == {}


def test_percentile_nearest_rank():
    assert percentile([], 95) == 0.0
    assert percentile([1.0], 95) == 1.0
    assert percentile(list(range(1, 21)), 95) == 19
    assert percentile([3, 1, 2], 50) == 2


def test_pair_deltas_and_evaluate():
    it_reps = [_record([10, 10, 10], peak=100 + d) for d in (0, 2, 4)]
    up_reps = [_record([10, 10, 10], peak=90) for _ in range(3)]
    deltas = pair_deltas(it_reps, up_reps)
    assert deltas["peak_alloc"]["deltas"] == [10, 12, 14]
    assert deltas["peak_alloc"]["median"] == 12
    assert deltas["peak_alloc"]["spread"] == 2
    assert deltas["peak_alloc"]["upstream_median"] == 90
    assert deltas["slope[0]"]["median"] == 0.0
    # report-only without a frozen record
    assert evaluate(deltas, None) == []
    assert evaluate(deltas, {"peak_alloc": {"baseline": 10, "tolerance": 5}}) == []
    failures = evaluate(deltas, {"peak_alloc": {"baseline": 0, "tolerance": 5}})
    assert len(failures) == 1 and failures[0].startswith("peak_alloc")
    # shrinking is never a regression, and recorded-only metrics are never gated
    assert evaluate(deltas, {"peak_alloc": {"baseline": 100, "tolerance": 0}}) == []
    assert evaluate(deltas, {"peak_reserved": {"baseline": -1000, "tolerance": 0}}) == []


def test_pair_deltas_require_matching_repeats():
    with pytest.raises(ValueError, match="must match"):
        pair_deltas([_record([1], 1)], [])


def test_derive_tolerance_floors():
    assert derive_tolerance([0, 0], "peak_alloc") == 8 * MIB
    assert derive_tolerance([0], "slope[0]") == 64 * 1024
    assert derive_tolerance([4 * MIB], "peak_alloc") == 12 * MIB


def test_recorder_without_accelerator_reports_unavailable():
    rec = StepRecorder(warmup=1, total=3)
    rec.begin()
    for _ in range(4):  # one step past the window is recorded but never measured
        rec.step_end()
    out = rec.finish()
    assert out["available"] is False
    assert [s["measured"] for s in out["steps"]] == [False, True, True, False]
    assert out["end_current"] == 0


def test_run_arm_synthetic_cpu(tmp_path):
    out = tmp_path / "it.json"
    runner.main(["--composition", "synthetic", "--arm", "it", "--repeats", "2", "--window", "3", "--out", str(out)])
    record = json.loads(out.read_text())
    assert record["vendor"] == "cpu"
    assert len(record["repeats"]) == 2
    assert all(len(r["steps"]) == 5 for r in record["repeats"])


def test_freeze_requires_both_vendors(tmp_path):
    def pair(vendor, median, spread):
        return {
            "composition": "c",
            "vendor": vendor,
            "it": {"plant": {}},
            "deltas": {
                "peak_alloc": {"median": median, "spread": spread},
                "peak_reserved": {"median": 1, "spread": 1},
            },
        }

    with pytest.raises(SystemExit, match="one vendor only"):
        freeze.freeze([pair("cuda", 10, 1)])
    planted = pair("rocm", 10**9, 10**9) | {"it": {"plant": {"leak_bytes": 1}}}
    bounds = freeze.freeze([pair("cuda", 10, 1), pair("cuda", 20, 1), pair("rocm", 30, 4 * MIB), planted])
    assert bounds["cuda"]["c"]["peak_alloc"] == {"baseline": 15.0, "tolerance": 12.0 * MIB}
    assert bounds["rocm"]["c"]["peak_alloc"]["baseline"] == 30.0
    assert "peak_reserved" not in bounds["cuda"]["c"]
