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
"""Run the arms of a composition, one subprocess per arm, and pair their results.

Each arm runs in its own process so allocator state, hooks and caches from one arm cannot reach the other: absolute
footprints measured in a shared process were found to depend on what ran before them. Repeats run inside the arm's
process, since process startup (interpreter, imports, accelerator context) costs far more than a measured window.

Usage, for one arm (what ``measure_pair`` launches)::

    python -m tests.profiling_delta.runner --composition core_hf_train --arm it --repeats 5 --out it.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tests.profiling_delta.arms import ARMS, COMPOSITIONS, with_plant
from tests.profiling_delta.stats import StepRecorder, pair_deltas, release_cached, vendor

REPO_ROOT = Path(__file__).resolve().parents[2]

# Both arms run under the same, explicit allocator configuration on every host; ROCm builds read the HIP variable.
ALLOC_CONF = "expandable_segments:False"
DEFAULT_WARMUP = 2
DEFAULT_WINDOW = 8
DEFAULT_REPEATS = 5


def run_arm(
    composition: str,
    arm: str,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
    window: int = DEFAULT_WINDOW,
    plant: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run ``repeats`` repeats of one arm in this process and return its record."""
    comp = COMPOSITIONS[composition]
    if plant:
        comp = with_plant(comp, plant)
    arm_fn = comp.arm(arm)
    steps = warmup + window
    records = []
    t0 = time.perf_counter()
    for seed in range(repeats):
        release_cached()
        recorder = StepRecorder(warmup=warmup, total=steps)
        recorder.begin()
        arm_fn(recorder, steps, seed)
        record = recorder.finish()
        if len(record["steps"]) != steps:
            raise RuntimeError(
                f"{composition}/{arm} reported {len(record['steps'])} step ends, expected {steps}: the arm's loop "
                "and the recorder disagree, so its window is not the one measured"
            )
        records.append(record)
    return {
        "composition": composition,
        "arm": arm,
        "vendor": vendor(),
        "torch": torch.__version__,
        "warmup": warmup,
        "window": window,
        "plant": plant or {},
        "seconds": round(time.perf_counter() - t0, 3),
        "repeats": records,
    }


def _arm_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF)
    env.setdefault("PYTORCH_HIP_ALLOC_CONF", env["PYTORCH_CUDA_ALLOC_CONF"])
    env.setdefault("IT_TRUST_REMOTE_CODE", "1")  # the RTE example module is hub-resident, as in the test suite
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), str(REPO_ROOT / "tests"), env.get("PYTHONPATH", "")) if p
    )
    return env


def run_arm_subprocess(composition: str, arm: str, out: Path, timeout: float = 1800, **kwargs) -> dict[str, Any]:
    """Run one arm in a fresh interpreter and return its record."""
    cmd = [sys.executable, "-m", "tests.profiling_delta.runner", "--composition", composition, "--arm", arm]
    cmd += ["--out", str(out)]
    for key in ("repeats", "warmup", "window"):
        if key in kwargs:
            cmd += [f"--{key}", str(kwargs[key])]
    if kwargs.get("plant"):
        cmd += ["--plant", ",".join(f"{k}={v}" for k, v in kwargs["plant"].items())]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=_arm_env(), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"arm {composition}/{arm} failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-8000:]}"
        )
    record = json.loads(out.read_text())
    record["process_seconds"] = round(time.perf_counter() - t0, 3)
    return record


@dataclass
class PairResult:
    composition: str
    vendor: str
    it: dict[str, Any]
    upstream: dict[str, Any]
    deltas: dict[str, dict] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        def brief(rec):
            return {k: v for k, v in rec.items() if k != "repeats"} | {
                "absolute": {m: [r[m] for r in rec["repeats"]] for m in ("peak_alloc", "peak_reserved", "end_current")}
            }

        return {
            "composition": self.composition,
            "vendor": self.vendor,
            "it": brief(self.it),
            "upstream": brief(self.upstream),
            "deltas": self.deltas,
        }


def measure_pair(composition: str, workdir: Path, **kwargs) -> PairResult:
    """Run the interpretune arm, then the upstream arm, back to back on the same device, and pair them."""
    workdir.mkdir(parents=True, exist_ok=True)
    it_rec = run_arm_subprocess(composition, "it", workdir / f"{composition}.it.json", **kwargs)
    up_kwargs = {k: v for k, v in kwargs.items() if k != "plant"}
    up_rec = run_arm_subprocess(composition, "upstream", workdir / f"{composition}.upstream.json", **up_kwargs)
    return PairResult(
        composition=composition,
        vendor=it_rec["vendor"],
        it=it_rec,
        upstream=up_rec,
        deltas=pair_deltas(it_rec["repeats"], up_rec["repeats"]),
    )


def _parse_plant(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    return {k: int(v) for k, v in (item.split("=") for item in text.split(","))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--composition", required=True, choices=sorted(COMPOSITIONS))
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--plant", help="synthetic only: comma-separated leak_bytes=N,overhead_bytes=N")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    record = run_arm(args.composition, args.arm, args.repeats, args.warmup, args.window, plant=_parse_plant(args.plant))
    args.out.write_text(json.dumps(record))


if __name__ == "__main__":
    main()
