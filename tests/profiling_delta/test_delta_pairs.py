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
``IT_PROFILING_DELTA_REPORT_ONLY`` makes every composition report-only (release and tag builds set it). Set
``IT_PROFILING_DELTA_DIR`` to keep the artifacts (``<composition>.pair.json``) for ``freeze``.
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
# device total each pair needs, in GiB: the larger arm's measured reserved peak (NVIDIA and AMD, 2026-10-25 freeze
# runs: train pairs up to 2.78, bf16 1.36, phased 0.88, test 0.77) plus about 0.4 of context, rounded up; the arms run
# one at a time
PAIR_GPU_MEM_GB = {"core_hf_train_bf16": 2.0, "core_hf_fts_train": 1.5}
PAIR_GPU_MEM_GB.update({name: 3.5 for name in PAIRED if name.endswith("_train") and name not in PAIR_GPU_MEM_GB})
PAIR_GPU_MEM_GB.update({name: 1.5 for name in PAIRED if name.endswith("_test")})


def _artifact_dir(tmp_path: Path) -> Path:
    return Path(os.environ["IT_PROFILING_DELTA_DIR"]) if os.environ.get("IT_PROFILING_DELTA_DIR") else tmp_path


def _frozen(vendor: str, composition: str) -> dict | None:
    bounds = yaml.safe_load(BOUNDS_FILE.read_text()) or {}
    return (bounds.get(vendor) or {}).get(composition)


@RunIf(min_cuda_gpus=1, min_gpu_mem_gb=1.0, profiling=True)
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


@pytest.mark.parametrize(
    "composition",
    [
        pytest.param(name, marks=RunIf(min_cuda_gpus=1, min_gpu_mem_gb=PAIR_GPU_MEM_GB[name], profiling=True))
        for name in PAIRED
    ],
)
def test_delta_pair(tmp_path, composition):
    out_dir = _artifact_dir(tmp_path)
    pair = measure_pair(composition, out_dir / "arms")
    (out_dir / f"{composition}.pair.json").write_text(json.dumps(pair.to_json(), indent=1))

    for arm in (pair.it, pair.upstream):
        assert all(r["available"] for r in arm["repeats"]), f"{arm['arm']} arm saw no accelerator statistics"
        assert all(r["peak_alloc"] > 0 for r in arm["repeats"])
    # bind the delta to matched arms: both consumed the same batches, step for step
    assert [r["inputs"] for r in pair.it["repeats"]] == [r["inputs"] for r in pair.upstream["repeats"]]
    # phased compositions: the structural invariants are exact, so they are asserted whether or not bounds are frozen
    for it_rep, up_rep in zip(pair.it["repeats"], pair.upstream["repeats"]):
        for phase, counts in it_rep["structure"].items():
            leaks = {k: counts[k] for k in ("frozen_with_grad", "frozen_in_groups", "state_frozen") if counts[k]}
            assert not leaks, f"{composition} phase {phase}: frozen parameters hold state {leaks}"
        assert it_rep["structure"] == up_rep["structure"], "the arms thawed or optimized different parameters"

    frozen = _frozen(pair.vendor, composition)
    if frozen is None:
        summary = {m: pair.deltas[m]["median"] for m in ("peak_alloc", "end_current", "slope[0]") if m in pair.deltas}
        warnings.warn(f"{composition} on {pair.vendor}: report-only (no frozen bound); median deltas {summary}")
        return
    failures = evaluate(pair.deltas, frozen)
    if failures and os.environ.get("IT_PROFILING_DELTA_REPORT_ONLY"):
        # release and tag builds record profiling without gating on it (the pipeline sets this there)
        warnings.warn(f"{composition} on {pair.vendor}, report-only: " + "; ".join(failures))
        return
    assert not failures, f"{composition} on {pair.vendor}: " + "; ".join(failures)
