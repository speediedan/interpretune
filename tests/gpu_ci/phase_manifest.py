"""Phase-specific collection manifests and ROCm failure triage (#794).

Collection is not execution. This module records, per named test phase, exactly which tests the
phase selector in ``tests/conftest.py`` selects (node ids and counts), together with the source
and environment pins the collection ran under. It never executes a selected test, so a manifest
is evidence about selection only: it is not a pass on any hardware, NVIDIA or AMD.

The phases and their selectors mirror the ones CI actually runs
(``.azure-pipelines/scripts/gpu_phase.sh`` and ``tests/special_tests.sh``), and collection reuses
the same collect-only shape as ``tests/gpu_ci/calibrate.py``: a CPU-only
``pytest --collect-only`` subprocess with ``CUDA_VISIBLE_DEVICES`` emptied, so producing a
manifest needs no GPU and no lease.

Usage (from the repository root):

    python tests/gpu_ci/phase_manifest.py --out /tmp/manifest.json
    python tests/gpu_ci/phase_manifest.py --phase cuda --phase standalone --out /tmp/manifest.json
    python tests/gpu_ci/phase_manifest.py --classify-file /tmp/a-failing-phase.log

ROCm triage: ``classify_failure`` sorts one failure's text into the buckets the build-day run
needs. The buckets, in the order they are checked:

- ``cuda-only-dependency``: an import or shared library that only ships for CUDA (a wheel or
  ``.so`` that no ROCm build provides). Fix shape: replace or gate the dependency.
- ``rocm-kernel-gap``: the op reached HIP and the kernel/library does not implement it
  (rocBLAS, MIOpen, an unimplemented HIP kernel). Fix shape: upstream gap; record and skip
  narrowly.
- ``backend-gap``: a backend capability is missing rather than a kernel (a distributed backend,
  a device type a backend refuses, a capability probe that says no). Fix shape: backend work
  or a capability-gated skip.
- ``hip-compatible-cuda-interface``: the failure is in test or harness code that assumed an
  NVIDIA-flavoured surface (a device name, a CUDA version string, ``nvidia-smi``) while the
  ``torch.cuda`` interface itself is the HIP compatibility layer and works. Fix shape: make
  the assumption vendor-neutral; the test is expected to run on ROCm.
- ``unknown``: none of the above matched. Triage by hand; do not guess a bucket.

Actual ROCm execution stays pending until AMD hardware is available: every manifest records
``rocm_execution: pending-hardware`` so the distinction cannot be lost downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLLECT_DIRS = ("tests", "src/it_examples/tests")

# phase name -> the env selector tests/conftest.py keys on (empty dict: the baseline suite).
# Keep in step with tests/conftest.py::pytest_collection_modifyitems and gpu_phase.sh.
PHASES: dict[str, dict[str, str]] = {
    "baseline": {},
    "cuda": {"IT_RUN_CUDA_TESTS": "1"},
    "standalone": {"IT_RUN_STANDALONE_TESTS": "1"},
    "profile_ci": {"IT_RUN_PROFILING_TESTS": "1"},
    "profiling": {"IT_RUN_PROFILING_TESTS": "2"},
    "optional": {"IT_RUN_OPTIONAL_TESTS": "1"},
    "benchmark": {"IT_RUN_BENCHMARK_TESTS": "1"},
}
_SELECTOR_VARS = tuple({var for env in PHASES.values() for var in env})

_NODEID_RE = re.compile(r"^(\S+::\S+)\s*$")


def parse_nodeids(collect_output: str) -> list[str]:
    """The node ids from ``pytest --collect-only -q`` output, in collection order."""
    nodeids = []
    for line in collect_output.splitlines():
        match = _NODEID_RE.match(line.strip())
        if match and not match.group(1).startswith("<"):
            nodeids.append(match.group(1))
    return nodeids


def collect_phase(phase: str, extra: list[str] | None = None) -> dict:
    """Collect one phase (CPU only, executes nothing) and describe the selection."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; known phases: {sorted(PHASES)}")
    env = {k: v for k, v in os.environ.items() if k not in _SELECTOR_VARS}
    env.update(PHASES[phase])
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *COLLECT_DIRS,
        "-q",
        "--collect-only",
        "-p",
        "no:cacheprovider",
        *(extra or []),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    nodeids = parse_nodeids(proc.stdout)
    return {
        "phase": phase,
        "selector_env": dict(PHASES[phase]),
        "collect_dirs": list(COLLECT_DIRS),
        "collected": len(nodeids),
        "executed": 0,
        "nodeids": nodeids,
        "collect_exit_code": proc.returncode,
    }


def env_pins() -> dict:
    """The pins a reader needs to reproduce a collection: source, interpreter, salient packages."""
    pins: dict = {"python": sys.version.split()[0], "platform": platform.platform()}
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    pins["git_sha"] = proc.stdout.strip() if proc.returncode == 0 else "unknown"
    try:
        import torch

        pins["torch"] = torch.__version__
        pins["torch_cuda"] = torch.version.cuda
        pins["torch_hip"] = getattr(torch.version, "hip", None)
    except Exception as exc:  # a manifest from a broken env should say so, not crash
        pins["torch"] = f"unavailable: {exc}"
    try:
        import interpretune

        pins["interpretune_file"] = interpretune.__file__
    except Exception as exc:
        pins["interpretune_file"] = f"unavailable: {exc}"
    return pins


def build_manifest(phases: list[str], extra: list[str] | None = None) -> dict:
    """A manifest over ``phases``: collected beside phase, executed nowhere, ROCm pending."""
    return {
        "kind": "phase-collection-manifest",
        "execution": "not-run-collection-only",
        "rocm_execution": "pending-hardware",
        "pins": env_pins(),
        "phases": {phase: collect_phase(phase, extra) for phase in phases},
    }


# ---------------------------------------------------------------------------
# ROCm failure triage. Patterns are checked in bucket order; first match wins.
# Each pattern set is deliberately narrow: a miss lands in ``unknown`` for a
# human, which is the honest outcome, rather than in a guessed bucket.
# ---------------------------------------------------------------------------

_CUDA_ONLY_DEPENDENCY = (
    r"no module named ['\"]?(apex|flash_attn|bitsandbytes|cuml|nvidia\.)",
    r"(libcudart|libcuda\.so|libnccl|\.so[^\n]*cuda)[^\n]*(cannot open|not found|no such file)",
    r"could not load[^\n]*(cuda|nvidia)",
    r"nvidia-[a-z0-9_-]+[^\n]*(not installed|no matching distribution)",
)
_ROCM_KERNEL_GAP = (
    r"hiperror\w*(notyetimplemented|notsupported)",
    r"miopen\w*(notimplemented|notsupported)",
    r"(not implemented|unsupported)[^\n]*(hip|rocm|rocblas|miopen|hipblas|hipsparse)",
    r"(rocblas|miopen|hipblas|hipsparse)[^\n]*(not implemented|unsupported|status_not_implemented)",
    r"no kernel image[^\n]*(hip|gfx\d)",
    r"invalid device function[^\n]*gfx\d",
)
_BACKEND_GAP = (
    r"(backend|capability)[^\n]*(not supported|does not support|unavailable|missing)",
    r"(nccl|rccl)[^\n]*(not available|unavailable|not built|unsupported)",
    r"distributed[^\n]*backend[^\n]*(not|unavailable)",
    r"device type[^\n]*(not supported|unsupported)",
    r"torch\.backends[^\n]*(not|unavailable|false)",
)
_HIP_COMPATIBLE_INTERFACE = (
    r"(nvidia-smi|nvml)[^\n]*(not found|no such file|failed)",
    r"(expected|assert)[^\n]*(nvidia|geforce|rtx|tesla|cuda capability|compute capability)",
    r"device[^\n]*name[^\n]*(nvidia|geforce|rtx)",
    r"torch\.version\.cuda[^\n]*(none|assert|expected)",
    r"cuda[^\n]*version[^\n]*(mismatch|expected)[^\n]*hip",
)

_BUCKETS = (
    ("cuda-only-dependency", _CUDA_ONLY_DEPENDENCY),
    ("rocm-kernel-gap", _ROCM_KERNEL_GAP),
    ("backend-gap", _BACKEND_GAP),
    ("hip-compatible-cuda-interface", _HIP_COMPATIBLE_INTERFACE),
)


def classify_failure(text: str) -> dict:
    """Sort one failure's text into a ROCm triage bucket (see the module docstring)."""
    lowered = text.lower()
    for bucket, patterns in _BUCKETS:
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return {"bucket": bucket, "evidence": match.group(0)[:160]}
    return {"bucket": "unknown", "evidence": ""}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", action="append", choices=sorted(PHASES), help="repeatable; default: cuda, standalone")
    ap.add_argument("--out", type=Path, default=None, help="write the manifest JSON here (default: stdout summary)")
    ap.add_argument("--classify-file", type=Path, default=None, help="classify a failure log instead of collecting")
    ap.add_argument("-k", default=None, help="pytest -k expression narrowing every collection")
    args = ap.parse_args(argv)

    if args.classify_file is not None:
        print(json.dumps(classify_failure(args.classify_file.read_text()), indent=2))
        return 0

    phases = args.phase or ["cuda", "standalone"]
    manifest = build_manifest(phases, ["-k", args.k] if args.k else None)
    if args.out is not None:
        args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    for phase, entry in manifest["phases"].items():
        # collected beside phase, always: a count without its phase reads as a suite result.
        print(f"phase={phase} collected={entry['collected']} executed=0 (collection only)")
    print(f"rocm_execution={manifest['rocm_execution']} git_sha={manifest['pins']['git_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
