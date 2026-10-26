#!/usr/bin/env bash
# The non-phase steps of gpu-tests.yml's any-vendor job (#703), run INSIDE its container (.azure-pipelines/templates/
# gpu-job.yml) through ctr.sh, with the checkout at the working directory. The GPU phases and leases have scripts of
# their own beside this one (gpu_phase.sh, gpu_stage_lease.sh, check_gpu_phases.sh).
#
#   gpu_job_container.sh hf-cache-check | hf-cache-probe | hf-cache-reset | install | env | plan | collect | warm
#                        | report | extended-profile | extended-optional | coverage | hf-cache-fix
#
# Parameters arrive as IT_* variables, which ctr.sh forwards: IT_RESET_HF_CACHE, IT_GPU_MODE and IT_GPU_EXTENDED_PARAM
# (the pipeline's parameters as Azure renders them), and IT_FORCE_CACHE_RESET (set by hf-cache-probe).
set -uo pipefail

CACHE=/opt/interpretune/hf_cache
VENV=/tmp/venvs/it_dev
# Phase manifests, records and reports (tests/gpu_ci/phase_report.py), on the template's evidence mount.
PHASES_OUT=/evidence/it-phases
PHASES=(cuda standalone profile_ci)
venv() { source "${VENV}/bin/activate"; }
# The checkout belongs to the host's uid, which git inside the container may refuse as "dubious ownership". Trust it
# for each git call only: GIT_CONFIG_* is command-line scope, which git honours for safe.directory.
trusted_git_env() { export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"; }
# Lock files group-writable and directories setgid, so the next run (whichever uid it maps to) can take the locks.
fix_cache_perms() {
    sudo find "${CACHE}/datasets" -type f -name "*.lock" -exec chmod 664 {} \; 2>/dev/null || true
    sudo find "${CACHE}/datasets" -type d -exec chmod g+ws {} \; 2>/dev/null || true
}

case "${1:?command}" in
    hf-cache-check)
        id || true
        ls -ld /opt/interpretune "$CACHE" 2>/dev/null || true
        if touch "${CACHE}/.write_test" 2>/dev/null; then
            echo WRITE_OK
        else
            echo "ERROR: Unable to write to ${CACHE}" >&2; exit 1
        fi
        ;;
    hf-cache-probe)
        lock=$(find "${CACHE}/datasets" -name "*super_glue*rte*.lock" -type f 2>/dev/null | head -1)
        if [[ -z "$lock" ]]; then echo "No super_glue RTE lock files found in cache"; exit 0; fi
        echo "Found lock file: ${lock}"; ls -l "$lock"
        if touch "$lock" 2>/dev/null; then echo "Write access OK"; exit 0; fi
        echo "Lock file not writable: fixing the cache's permissions"
        fix_cache_perms
        if touch "$lock" 2>/dev/null; then
            echo "Write access restored after the permission fix"
        else
            echo "Write access still failed: the cache will be reset"
            echo "##vso[task.setvariable variable=IT_FORCE_CACHE_RESET]1"
        fi
        ;;
    hf-cache-reset)
        if [[ "${IT_FORCE_CACHE_RESET:-0}" == 1 || "${IT_RESET_HF_CACHE:-False}" == True ]]; then
            echo "Clearing the HF cache ($([[ "${IT_FORCE_CACHE_RESET:-0}" == 1 ]] && echo "inaccessible lock files" || echo "manual reset parameter"))"
            rm -rf "${CACHE:?}"/* || true
        else
            echo "HF cache reset not required"
        fi
        ;;
    install)
        set -e
        venv
        # A ROCm image carries the exclusion file, which keeps its torch build from being replaced by the lock's PyPI
        # (CUDA) wheel; the CUDA image has none and installs as it always has.
        excludes=()
        [[ -f "${VENV}/torch-excludes.txt" ]] && excludes=(--excludes "${VENV}/torch-excludes.txt")
        uv pip install -e . --group git-deps "${excludes[@]}"
        uv pip install -r requirements/ci/requirements.txt "${excludes[@]}"
        uv pip list
        for pkg in coverage pytest torch transformers; do
            uv pip show "$pkg" >/dev/null 2>&1 || { echo "ERROR: ${pkg} is NOT installed" >&2; exit 1; }
        done
        # The image carries torch and the steps above resolve against it. A resolve that moved torch (or an image
        # built from a different pin) would test a torch nobody asked for while the run reads as a run of the pin, so
        # refuse by name. overrides.txt is the source of truth for the pin; the build's local tag (+rocm...) is not
        # part of it.
        python - <<'PY'
import re, sys
import torch
pin = re.search(r"^torch==(\S+)", open("requirements/ci/overrides.txt").read(), re.M).group(1)
installed = torch.__version__.split("+")[0]
print(f"torch {torch.__version__} (git {torch.version.git_version}, cuda {torch.version.cuda}, hip {torch.version.hip}); pinned {pin}")
if installed != pin:
    sys.exit(f"overrides.txt pins torch {pin}, but {torch.__version__} is installed")
PY
        ;;
    env)
        venv
        python requirements/utils/collect_env_details.py
        ;;
    plan)
        set -e
        venv
        trusted_git_env
        python -m tests.gpu_ci.plan_gpu_run --mode "${IT_GPU_MODE:-auto}" --extended "${IT_GPU_EXTENDED_PARAM:-False}"
        ;;
    collect)
        # CPU only, executes nothing: what each phase's paths collect and its selection keeps. A collection error fails
        # here, before any GPU work, rather than surfacing as a phase that silently ran less.
        set -e
        venv
        trusted_git_env
        mkdir -p "$PHASES_OUT"
        args=(); for phase in "${PHASES[@]}"; do args+=(--phase "$phase"); done
        python -m tests.gpu_ci.phase_report collect "${args[@]}" --out-dir "$PHASES_OUT"
        echo "##vso[task.setvariable variable=IT_PHASES_COLLECTED]true"
        ;;
    report)
        # One report per collected phase, under the vendor the agent reported, gated by its class (phase_report.gate).
        venv
        lease_mode=$(cat /tmp/azp_gpu_large.pid.mode 2>/dev/null || echo none)
        status=0
        for phase in "${PHASES[@]}"; do
            [[ -f "${PHASES_OUT}/manifests/${phase}.json" ]] || continue
            python -m tests.gpu_ci.phase_report report --phase "$phase" --out-dir "$PHASES_OUT" \
                --vendor "${IT_GPU_VENDOR:?}" --evidence-class pipeline --job-class "${IT_JOB_CLASS:-ci-any-vendor}" \
                --lease-mode "$lease_mode" --image "${IT_IMAGE:?}" --reviewed-exclusions tests/gpu_ci/reviewed_exclusions.json \
                || { echo "##vso[task.logissue type=error]phase ${phase}: report does not pass a ${IT_JOB_CLASS:-ci-any-vendor} job"; status=1; }
        done
        exit "$status"
        ;;
    warm)
        set -e
        venv
        # Suite resolves Hub experiment seeds offline from this mount, and nothing else populates components on it:
        # models/datasets persist across builds, pinned component revisions do not. Warm them (small, pinned,
        # idempotent) so the Testing phases never inherit another build's refs/main roulette.
        python scripts/warm_hf_cache.py --components-only
        # GPU-only artifacts (the circuit-tracer transcoders), fetched here with retries rather than lazily inside a
        # notebook kernel, and only when the plan runs the notebook tests that load them
        if python -m tests.gpu_ci.plan_gpu_run --covers src/it_examples/tests/test_notebooks.py; then
            python scripts/warm_hf_cache.py --manifest tests/gpu_ci/hf_warm_manifest_gpu.yaml
        fi
        ;;
    extended-profile)
        venv
        mkdir -p /tmp/kernel_cache
        PYTORCH_KERNEL_CACHE_PATH=/tmp/kernel_cache bash ./tests/special_tests.sh --mark_type=profile --allow-failures
        ;;
    extended-optional)
        venv
        bash ./tests/special_tests.sh --mark_type=optional --allow-failures
        ;;
    coverage)
        set -e
        venv
        python -m coverage report
        python -m coverage xml
        python -m coverage html
        if [[ -z "${CODECOV_TOK:-}" ]]; then
            echo "##vso[task.logissue type=warning]no Codecov token in this step: coverage written, not uploaded"
            exit 0
        fi
        # The CLI is verified before it runs: its checksum file must carry a good signature from Codecov's signing key, by
        # fingerprint (keybase.io, which used to serve the key, no longer does: it returns no key data), and the binary
        # must match the checksum. As in finetuning-scheduler's GPU job.
        codecov_key=27034E7FDB850E0BBC2C62FF806BB28AED779869
        curl -fsS https://uploader.codecov.io/verification.gpg | gpg --no-default-keyring --keyring trustedkeys.gpg --import
        curl -fsSO https://cli.codecov.io/latest/linux/codecov
        curl -fsSO https://cli.codecov.io/latest/linux/codecov.SHA256SUM
        curl -fsSO https://cli.codecov.io/latest/linux/codecov.SHA256SUM.sig
        sig=$(gpg --no-default-keyring --keyring trustedkeys.gpg --status-fd 1 --verify codecov.SHA256SUM.sig \
            codecov.SHA256SUM 2>/dev/null) || true
        grep -q "^\[GNUPG:\] VALIDSIG ${codecov_key} " <<< "$sig" \
            || { echo "##vso[task.logissue type=error]the Codecov CLI checksums are not signed by ${codecov_key}"; exit 1; }
        sha256sum -c codecov.SHA256SUM
        chmod +x codecov
        # A selected run covers only part of the GPU set, so it uploads under its own flag: under `gpu` it would read
        # as a coverage drop. Either vendor covers the same lines, so the vendor is named only in the run name.
        flags=(-F gpu -F pytest); name="GPU-coverage-${IT_GPU_VENDOR:-unknown}"
        if [[ "${IT_GPU_SELECTION_MODE:-}" == selected ]]; then
            flags=(-F gpu-selected); name="GPU-selected-coverage-${IT_GPU_VENDOR:-unknown}"
        fi
        ./codecov upload-process --slug 'speediedan/interpretune' -t "$CODECOV_TOK" --commit-sha "${BUILD_SOURCEVERSION:?}" \
            --git-service 'github' -n "$name" "${flags[@]}" --env 'linux,azure' -f 'coverage.xml'
        ;;
    hf-cache-fix)
        rm -rf .pytest_cache || true
        fix_cache_perms
        ;;
    *)
        echo "unknown command ${1}" >&2; exit 2 ;;
esac
