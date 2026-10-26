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
"""Derive frozen delta bounds from pair artifacts measured on both vendors.

Each artifact is a ``<composition>.pair.json`` written by the profiling delta tests. The baseline of a metric is the
median of the measured medians on that vendor; the tolerance pools the repeat spread of every run on every vendor,
because the noise a gate has to clear is the instrument's, and a vendor with fewer runs should not get a looser bound
for it. A composition measured on only one vendor is refused unless ``--allow-single-vendor`` is passed: the
expectation that deltas are vendor-independent is tested here, not assumed.

Usage::

    python -m tests.profiling_delta.freeze --artifacts nvidia/ amd/ --out tests/profiling_delta/delta_bounds.yaml
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from tests.profiling_delta.stats import derive_tolerance, gated


def load_pairs(paths: list[Path]) -> list[dict[str, Any]]:
    files = [f for p in paths for f in (sorted(p.rglob("*.pair.json")) if p.is_dir() else [p])]
    if not files:
        raise SystemExit(f"no *.pair.json artifacts under {[str(p) for p in paths]}")
    return [json.loads(f.read_text()) for f in files]


def freeze(pairs: list[dict[str, Any]], allow_single_vendor: bool = False, multiplier: float = 3.0) -> dict:
    medians: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # vendor -> comp -> metric -> [m]
    spreads: dict = defaultdict(lambda: defaultdict(list))  # comp -> metric -> [spread]
    for pair in pairs:
        if pair["it"].get("plant"):
            continue  # planted controls measure the instrument, never interpretune
        for metric, rec in pair["deltas"].items():
            if gated(metric):
                medians[pair["vendor"]][pair["composition"]][metric].append(rec["median"])
                spreads[pair["composition"]][metric].append(rec["spread"])
    vendors_by_comp = defaultdict(set)
    for vend, comps in medians.items():
        for comp in comps:
            vendors_by_comp[comp].add(vend)
    single = sorted(c for c, v in vendors_by_comp.items() if len(v) < 2)
    if single and not allow_single_vendor:
        raise SystemExit(f"measured on one vendor only, not frozen: {single} (pass --allow-single-vendor to override)")
    out: dict = {"cuda": {}, "rocm": {}}
    for vend, comps in medians.items():
        for comp, metrics in comps.items():
            out.setdefault(vend, {})[comp] = {
                metric: {
                    "baseline": float(statistics.median(vals)),
                    "tolerance": float(derive_tolerance(spreads[comp][metric], metric, multiplier)),
                }
                for metric, vals in sorted(metrics.items())
            }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--multiplier", type=float, default=3.0)
    ap.add_argument("--allow-single-vendor", action="store_true")
    args = ap.parse_args(argv)
    bounds = freeze(load_pairs(args.artifacts), args.allow_single_vendor, args.multiplier)
    header = [ln for ln in args.out.read_text().splitlines() if ln.startswith("#")] if args.out.exists() else []
    args.out.write_text("\n".join(header + [yaml.safe_dump(bounds, sort_keys=True)]))


if __name__ == "__main__":
    main()
