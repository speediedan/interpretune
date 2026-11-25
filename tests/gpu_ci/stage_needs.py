"""Size the large and multi stages of GPU placement from the tests this run will execute.

``tests/conftest.py`` writes each collecting session's GPU tests (declared device count, memory and bf16 need) into
``IT_GPU_STAGE_NEEDS_DIR`` when the job's collect step collects the phases, narrowed by the run's selection. Before
the large stage takes its lease, this reads them and prints shell assignments for ``gpu_stage_lease.sh``:

    found=1 large_gb=14 multi=0

``large_gb`` is the most memory a single-device test outside the small class declares (0 when none does), so the
large stage leases one device that holds it rather than the whole server; ``multi`` says whether any test needs more
than one device, so the whole-server stage runs only then. ``found=0`` (no records: an older collect step, a
collection that wrote nothing) means the caller keeps the old two-stage plan, the whole server for everything above
the small class.

    python -m tests.gpu_ci.stage_needs <dir> <small_max_gb>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def stage_needs(directory: str | Path, small_max_gb: float) -> dict[str, int]:
    """``{found, large_gb, multi}`` from every session file in ``directory`` (see the module docstring)."""
    rows: list[dict] = []
    for f in sorted(Path(directory).glob("session-*.json")) if Path(directory).is_dir() else []:
        try:
            got = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows += [r for r in got if isinstance(r, dict)] if isinstance(got, list) else []
    if not rows:
        return {"found": 0, "large_gb": 0, "multi": 0}
    single = [r for r in rows if int(r.get("gpus") or 1) <= 1]
    large = [float(r.get("mem_gb") or 0) for r in single if r.get("bf16") or float(r.get("mem_gb") or 0) > small_max_gb]
    return {
        "found": 1,
        "large_gb": math.ceil(max(large)) if large else 0,
        "multi": int(any(int(r.get("gpus") or 1) > 1 for r in rows)),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m tests.gpu_ci.stage_needs <dir> <small_max_gb>", file=sys.stderr)
        return 2
    try:
        small = float(argv[1])
    except ValueError:
        print(f"stage_needs: small_max_gb must be a number, got {argv[1]!r}", file=sys.stderr)
        return 2
    print(" ".join(f"{k}={v}" for k, v in stage_needs(argv[0], small).items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
