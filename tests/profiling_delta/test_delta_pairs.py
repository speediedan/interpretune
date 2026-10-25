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
"""Interpretune-versus-upstream memory deltas, one pair of subprocess arms per sampled composition.

These run in the extended profiling tier (``IT_RUN_PROFILING_TESTS=2``). A composition without a frozen bound for the
running vendor is report-only: its artifact is written and nothing is asserted beyond the pair being well formed.
Set ``IT_PROFILING_DELTA_DIR`` to keep the artifacts (``<composition>.pair.json``) for ``freeze``.
"""

import json
import os
import warnings
from pathlib import Path

import pytest
import yaml

from tests.profiling_delta.arms import COMPOSITIONS
from tests.profiling_delta.runner import measure_pair
from tests.profiling_delta.stats import KIB, MIB, evaluate
from tests.runif import RunIf

BOUNDS_FILE = Path(__file__).with_name("delta_bounds.yaml")
PAIRED = sorted(name for name in COMPOSITIONS if name != "synthetic")


def _artifact_dir(tmp_path: Path) -> Path:
    return Path(os.environ["IT_PROFILING_DELTA_DIR"]) if os.environ.get("IT_PROFILING_DELTA_DIR") else tmp_path


def _frozen(vendor: str, composition: str) -> dict | None:
    bounds = yaml.safe_load(BOUNDS_FILE.read_text()) or {}
    return (bounds.get(vendor) or {}).get(composition)


@RunIf(min_cuda_gpus=1, profiling=True)
def test_harness_detects_planted_defects(tmp_path):
    """The instrument's positive and null controls: a planted leak and a planted overhead in the interpretune arm
    of the synthetic pair must surface as exactly those deltas, and a clean pair must show none."""
    leak, overhead = 1 * MIB, 32 * MIB
    clean = measure_pair("synthetic", tmp_path / "clean", repeats=3)
    leaky = measure_pair("synthetic", tmp_path / "leak", repeats=3, plant={"leak_bytes": leak})
    heavy = measure_pair("synthetic", tmp_path / "overhead", repeats=3, plant={"overhead_bytes": overhead})

    for metric in ("peak_alloc", "end_current", "slope[0]"):
        assert abs(clean.deltas[metric]["median"]) < 64 * KIB, (metric, clean.deltas[metric])
    assert leaky.deltas["slope[0]"]["median"] == pytest.approx(leak, rel=0.05)
    assert heavy.deltas["peak_alloc"]["median"] == pytest.approx(overhead, rel=0.05)
    assert abs(heavy.deltas["slope[0]"]["median"]) < 64 * KIB

    # and the gate built on them fails each planted defect while passing the clean pair
    zero = {m: {"baseline": 0.0, "tolerance": 8 * MIB} for m in ("peak_alloc", "end_current")}
    zero["slope[0]"] = {"baseline": 0.0, "tolerance": 64 * KIB}
    assert evaluate(clean.deltas, zero) == []
    assert any(f.startswith("slope[0]") for f in evaluate(leaky.deltas, zero))
    assert any(f.startswith("peak_alloc") for f in evaluate(heavy.deltas, zero))


@RunIf(min_cuda_gpus=1, profiling=True)
@pytest.mark.parametrize("composition", PAIRED)
def test_delta_pair(tmp_path, composition):
    out_dir = _artifact_dir(tmp_path)
    pair = measure_pair(composition, out_dir / "arms")
    (out_dir / f"{composition}.pair.json").write_text(json.dumps(pair.to_json(), indent=1))

    for arm in (pair.it, pair.upstream):
        assert all(r["available"] for r in arm["repeats"]), f"{arm['arm']} arm saw no accelerator statistics"
        assert all(r["peak_alloc"] > 0 for r in arm["repeats"])
    # bind the delta to matched arms: both consumed the same batches, step for step
    assert [r["inputs"] for r in pair.it["repeats"]] == [r["inputs"] for r in pair.upstream["repeats"]]

    frozen = _frozen(pair.vendor, composition)
    if frozen is None:
        summary = {m: pair.deltas[m]["median"] for m in ("peak_alloc", "end_current", "slope[0]") if m in pair.deltas}
        warnings.warn(f"{composition} on {pair.vendor}: report-only (no frozen bound); median deltas {summary}")
        return
    failures = evaluate(pair.deltas, frozen)
    assert not failures, f"{composition} on {pair.vendor}: " + "; ".join(failures)
