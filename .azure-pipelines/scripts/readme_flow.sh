#!/usr/bin/env bash
# The steps of readme-flow-gpu.yml (Tier 2 of the user-install validation lane, #270), run INSIDE the any-vendor
# template's container (#703) through ctr.sh, with the checkout at the working directory. readme-flow-gpu.yml says
# why each step exists; the comments here say only what is particular to the script.
#
#   readme_flow.sh lease | extract | install | provenance | residue | cli | notebooks
#
# The install command extract finds is exported as IT_README_INSTALL_CMD, which ctr.sh forwards to the later steps.
set -euo pipefail

VENV="${PWD}/.readme-venv"
PY="${VENV}/bin/python"

case "${1:?command}" in
    lease)
        # The whole server and host memory, through the same one-stage path as gpu-tests.yml (released by the
        # template's final step with every other lease this job may hold).
        IT_GPU_ONE_STAGE=1 bash .azure-pipelines/scripts/gpu_stage_lease.sh acquire small
        ;;
    extract)
        # Extracted, never duplicated, and exactly one match: see readme-flow-gpu.yml.
        cmd="$(grep -E '^uv pip install -e .*--group test' README.md || true)"
        count="$(printf '%s\n' "${cmd}" | grep -c . || true)"
        if [[ "${count}" != 1 ]]; then
            echo "ERROR: expected exactly one 'uv pip install -e ... --group test' line in README.md," >&2
            echo "found ${count}. The install recipe moved or was reshaped; update this extraction" >&2
            echo "deliberately, keeping it aligned with what the provenance assertion imports." >&2
            exit 1
        fi
        echo "Extracted: ${cmd}"
        echo "##vso[task.setvariable variable=IT_README_INSTALL_CMD]${cmd}"
        ;;
    install)
        : "${IT_README_INSTALL_CMD:?run extract first}"
        # `env -u PYTHONPATH` on BOTH install and run: the image sets PYTHONPATH, and an inherited one would let a fresh
        # env silently resolve from elsewhere and report success.
        env -u PYTHONPATH uv venv "$VENV" --python 3.13
        env -u PYTHONPATH VIRTUAL_ENV="$VENV" bash -c "$IT_README_INSTALL_CMD"
        ;;
    provenance)
        env -u PYTHONPATH "$PY" - <<'PY'
import sys, pathlib, importlib
venv = pathlib.Path(sys.prefix).resolve()
failures = []
for name in ("torch", "transformer_lens", "nnsight", "lightning", "sae_lens",
             "circuit_tracer", "jsonargparse", "papermill"):
    path = pathlib.Path(importlib.import_module(name).__file__ or "").resolve()
    ok = venv in path.parents
    print(f"{'ok ' if ok else 'BAD'} {name:18} {path}")
    if not ok:
        failures.append(f"{name} resolved outside the README venv: {path}")
import torch
print(f"    cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    failures.append("CUDA unavailable; the GPU-gated tests would silently SKIP and this "
                    "pipeline would pass while validating nothing")
if failures:
    raise SystemExit("provenance failures:\n  " + "\n  ".join(failures))
PY
        ;;
    residue)
        env -u PYTHONPATH "$PY" - <<'PY'
from importlib.metadata import distributions
installed = {(d.metadata["Name"] or "").lower().replace("_", "-") for d in distributions()}
leaked = sorted(installed & {"pytest-timeout", "docstr-coverage"})
if leaked:
    raise SystemExit(
        f"dev-flow residue in a README-resolved env: {leaked}. These reach environments only "
        "via requirements/ci/sl_uv_requirements.txt, so this pipeline is testing the dev flow "
        "rather than the documented user flow."
    )
print("no dev-flow residue; the env is what README.md produces")
PY
        ;;
    cli)
        # IT_RUN_CUDA_TESTS is deliberately NOT set: it SELECTS the cuda-marked set rather than adding to it.
        set +e
        env -u PYTHONPATH PATH="${VENV}/bin:${PATH}" PYTHONPATH="${PWD}/src:${PWD}" "$PY" -m pytest \
            tests/core/test_components_cli.py tests/parity_acceptance/test_it_cli.py -v -rs --durations=25 2>&1 \
            | tee /tmp/cli_out.txt
        rc=${PIPESTATUS[0]}
        set -e
        bash .azure-pipelines/assert_surface.sh /tmp/cli_out.txt 45 0 "CLI instantiation" "$rc"
        ;;
    notebooks)
        # 5 expected skips, measured from a real run: see readme-flow-gpu.yml.
        set +e
        env -u PYTHONPATH PATH="${VENV}/bin:${PATH}" PYTHONPATH="${PWD}/src:${PWD}" "$PY" -m pytest \
            src/it_examples/tests/test_notebooks.py -v -rs --durations=25 2>&1 | tee /tmp/nb_out.txt
        rc=${PIPESTATUS[0]}
        set -e
        rm -rf .pytest_cache || true
        bash .azure-pipelines/assert_surface.sh /tmp/nb_out.txt 16 5 "published notebooks" "$rc"
        ;;
    *)
        echo "unknown command ${1}" >&2; exit 2 ;;
esac
