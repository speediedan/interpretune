#!/bin/bash
#
# Utility script to generate local IT coverage for a given environment
#
# The environment half (building or activating the venv, torch backend selection, taking the GPU lease) stays in
# shell, because it runs before a usable Python environment exists. The test phases then run through gpu-epsilon's
# phase driver (python -m gpu_epsilon.phases), which reads them from [tool.gpu-epsilon.phases] in pyproject.toml,
# records each phase's outcome and reports one final status.
#
# Torch handling:
# - Uses torch-pre.txt if present (for nightly/test builds)
# - Otherwise uses stable torch with automatic backend detection
# - Use --torch-backend=cpu to force CPU-only torch
set -eo pipefail

# Source shared infrastructure utilities
source "$(dirname "${BASH_SOURCE[0]}")/infra_utils.sh"

# Serialize the whole coverage run against other GPU work on the host (no-op unless GPU_LEASE_CMD is set).
# Held once for the entire run; the special-test runs inside it inherit it.
# A full suite is HOST-MEMORY-heavy as well as GPU-heavy, so it takes both host leases (gpu, then cpu-heavy in the
# host tool's order) and stays serialized with every phase of a CI job, whose CPU-only phase holds only cpu-heavy.
# GPU-only work (a notebook, a benchmark leg) takes the plain lease and may overlap that phase. An explicit
# GPU_LEASE_ARGS wins.
GPU_LEASE_ARGS="${GPU_LEASE_ARGS:---cpu-heavy}"
gpu_lease_reexec "$0" "$@"

unset repo_home
unset target_env_name
unset no_rebuild_base
unset from_source_spec
unset run_all_and_examples
unset no_export_cov_xml
unset pip_install_flags
unset self_test_only
unset it_build_flags
unset venv_dir
unset python_version
unset torch_backend
unset no_reruns
unset reruns_count
unset reruns_delay
unset allow_failures
unset dry_run
unset resource_debug
declare -a from_source_specs

# Default rerun settings (for transient httpx read timeouts with HF transformers v5)
reruns_count=2
reruns_delay=5

usage(){
>&2 cat << EOF
Usage: $0
    [ --repo-home input]
    [ --target-env-name input ]
    [ --python-version input ]  (uv --python spec, e.g. 3.12 or python3.12; default: 3.13)
    [ --torch-backend input ]  (cpu, auto, cu128, etc. default: auto)
    [ --no-rebuild-base ]
    [ --from-source "package:path[:extras][:env_var=value...]" ] (can be specified multiple times)
    [ --venv-dir "/path/to/venv/base" ]
    [ --run-all-and-examples ]
    [ --no-export-cov-xml ]
    [ --pip-install-flags "flags" ]
    [ --self-test-only ]
    [ --it-build-flags "flags" ]
    [ --no-reruns ]  (disable test reruns for transient failures)
    [ --reruns N ]  (number of reruns for transient failures, default: 2)
    [ --reruns-delay N ]  (delay in seconds between reruns, default: 5)
    [ --allow-failures ]  (run every phase after a failure; the run still exits 1, naming each failed phase)
    [ --dry-run ]  (print the phase plan and run nothing)
    [ --resource-debug ]  (enable opt-in CPU/CUDA per-test and per-fixture resource diagnostics)
    [ --help ]

    The --from-source flag can be specified multiple times for clarity, or use semicolons to separate specs.
    Format: "package:path[:extras][:env_var=value...]"
    - extras: optional, e.g., "all" or "dev,test"
    - env_var=value: optional, multiple env vars separated by colons

    Test Rerun Options (for transient httpx read timeouts with HF transformers v5):
    - By default, tests are rerun up to 2 times with a 5-second delay on failure
    - Use --no-reruns to disable automatic retries
    - Use --reruns N and --reruns-delay N to customize retry behavior

    Phases ([tool.gpu-epsilon.phases] in pyproject.toml): main (no GPU visible), cuda (cuda-marked tests, skipped
    without CUDA), standalone and profile_ci; --run-all-and-examples adds profile, optional and benchmark. Without
    --allow-failures the run stops at the first failed phase; the coverage report is written either way.

    Torch Handling:
    - Uses torch-pre.txt if present in requirements/ci/ (for nightly/test builds)
    - Otherwise uses stable torch with automatic backend detection
    - Use --torch-backend=cpu to force CPU-only torch (e.g., for CI environments)

    Examples:
    # generate it_latest coverage without rebuilding the it_latest base environment:
    #   ./gen_it_coverage.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --no-rebuild-base
    # generate it_latest coverage with CPU-only torch:
    #   ./gen_it_coverage.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --torch-backend=cpu
    # generate it_latest coverage with FTS from source:
    #   ./gen_it_coverage.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1"
    # generate it_latest coverage with multiple packages from source (multiple --from-source flags):
    #   ./gen_it_coverage.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1" --from-source="circuit_tracer:${HOME}/repos/circuit-tracer" --from-source="transformer_lens:${HOME}/repos/TransformerLens"
    # generate it_latest coverage with multiple packages from source (semicolon separator):
    #   ./gen_it_coverage.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1;circuit_tracer:${HOME}/repos/circuit-tracer;transformer_lens:${HOME}/repos/TransformerLens"
    # Build with custom venv directory:
    #   ./gen_it_coverage.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --venv-dir=/mnt/cache/${USER}/.venvs --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1;circuit_tracer:${HOME}/repos/circuit-tracer;transformer_lens:${HOME}/repos/TransformerLens"
    # generate it_latest coverage with no pip cache:
    #   ./gen_it_coverage.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --pip-install-flags="--no-cache-dir"
    # print the phase plan of a full run without running it:
    #   ./gen_it_coverage.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --no-rebuild-base --run-all-and-examples --dry-run
    # generate it_latest coverage with self_test_only:
    #   ./gen_it_coverage.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --self-test-only
EOF
exit 1
}

args=$(getopt -o '' --long repo-home:,target-env-name:,python-version:,torch-backend:,no-rebuild-base,from-source:,venv-dir:,run-all-and-examples,no-export-cov-xml,pip-install-flags:,self-test-only,it-build-flags:,no-reruns,reruns:,reruns-delay:,allow-failures,dry-run,resource-debug,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo-home)  repo_home=$2    ; shift 2  ;;
    --target-env-name)  target_env_name=$2  ; shift 2 ;;
    --python-version)  python_version=$2  ; shift 2 ;;
    --torch-backend)   torch_backend=$2   ; shift 2 ;;
    --no-rebuild-base)   no_rebuild_base=1 ; shift  ;;
    --from-source)   from_source_specs+=("$2") ; shift 2 ;;
    --venv-dir)   venv_dir=$2 ; shift 2 ;;
    --run-all-and-examples)   run_all_and_examples=1 ; shift  ;;
    --no-export-cov-xml)   no_export_cov_xml=1 ; shift ;;
    --pip-install-flags)   pip_install_flags=$2 ; shift 2 ;;
    --self-test-only)   self_test_only=1 ; shift ;;
    --it-build-flags)   it_build_flags=$2 ; shift 2 ;;
    --no-reruns)   no_reruns=1 ; shift ;;
    --reruns)   reruns_count=$2 ; shift 2 ;;
    --reruns-delay)   reruns_delay=$2 ; shift 2 ;;
    --allow-failures)  allow_failures=1 ; shift ;;
    --dry-run)  dry_run=1 ; shift ;;
    --resource-debug)  resource_debug=1 ; shift ;;
    --help)    usage      ; shift   ;;
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

# Combine multiple --from-source flags into single spec with semicolon separator
if [[ ${#from_source_specs[@]} -gt 0 ]]; then
    from_source_spec=$(IFS=';'; echo "${from_source_specs[*]}")
fi

d=`date +%Y%m%d%H%M%S`
tmp_coverage_dir="/tmp"
coverage_session_log="${tmp_coverage_dir}/gen_it_coverage_${target_env_name}_${d}.log"
resource_summary_json="${coverage_session_log%.log}_resource_summary.json"
# the phase driver's JSON record of the run, beside the session log
phase_record_json="${coverage_session_log%.log}.json"
echo "Use 'tail -f ${coverage_session_log}' to monitor progress"

# Expand leading ~ in common path arguments
repo_home=$(expand_tilde "${repo_home}")
venv_dir=$(expand_tilde "${venv_dir}")

# Determine venv path
# Priority: --venv-dir > IT_VENV_BASE > default ~/.venvs
venv_path=$(determine_venv_path "${venv_dir}" "${target_env_name}")

# Report to the terminal as well as the session log. Everything here used to exit into the log
# only, which is indistinguishable from a run still in progress to whoever launched it.
fail(){
    echo "gen_it_coverage: $1" >&2
    echo "gen_it_coverage: $1" >> "$coverage_session_log" 2>/dev/null || true
    exit 1
}

# With --no-rebuild-base there is no build step to surface a bad env name, so check it up front
# rather than failing later inside `source .../bin/activate`.
require_existing_venv(){
    [[ -x "${venv_path}/bin/activate" || -f "${venv_path}/bin/activate" ]] && return 0
    fail "no venv at '${venv_path}' for --target-env-name='${target_env_name}'. Build it first with scripts/build_it_env.sh, or drop --no-rebuild-base."
}

# Strip leading/trailing quotes from string variables if present
if [[ -n "${from_source_spec}" ]]; then
    from_source_spec=$(strip_quotes "$from_source_spec")
fi
if [[ -n "${it_build_flags}" ]]; then
    it_build_flags=$(strip_quotes "$it_build_flags")
fi


check_self_test_only(){
    local message=$1
    if [[ $self_test_only -eq 1 ]]; then
        echo "Self-test only mode enabled. $message" >> $coverage_session_log
        return 0
    fi
    return 1
}

env_rebuild(){
    cd ${repo_home}

    # Any env name is accepted. build_it_env.sh is what actually validates the target, and the
    # build command never varied by name -- an allowlist here only refused unfamiliar names, which
    # made a pins-only env (the one CLAUDE.md requires for anything gating a release or an upstream
    # PR) impossible to run without clobbering the integrated dev env.
    echo "Rebuilding environment with build_it_env.sh..." >> $coverage_session_log
    # Build command with conditional flags
    build_cmd="${repo_home}/scripts/build_it_env.sh --repo-home=${repo_home} --target-env-name=$1"
    [[ -n ${venv_dir} ]] && build_cmd="${build_cmd} --venv-dir=${venv_dir}"
    [[ -n ${python_version} ]] && build_cmd="${build_cmd} --python-version=${python_version}"
    [[ -n ${torch_backend} ]] && build_cmd="${build_cmd} --torch-backend=${torch_backend}"

    # Handle multiple --from-source flags
    if [[ ${#from_source_specs[@]} -gt 0 ]]; then
        for spec in "${from_source_specs[@]}"; do
            build_cmd="${build_cmd} --from-source='${spec}'"
        done
    fi

    [[ -n ${pip_install_flags} ]] && build_cmd="${build_cmd} --uv-install-flags='${pip_install_flags}'"

    echo "Running: ${build_cmd}" >> $coverage_session_log
    if ! eval ${build_cmd} >> $coverage_session_log 2>&1; then
        fail "environment build failed for '$1'; see ${coverage_session_log}"
    fi
}

collect_env_coverage(){
    temp_special_log="${tmp_coverage_dir}/special_test_output_$1_${d}.log"
    cd ${repo_home}
    maybe_deactivate
    source ${venv_path}/bin/activate
    check_self_test_only "Skipping all tests and examples." && return
    if ! python -c "import gpu_epsilon.phases" 2>/dev/null; then
        fail "gpu-epsilon's phase driver is not installed: uv pip install -r requirements/ci/gpu_epsilon.txt"
    fi

    # Reruns (for transient httpx read timeouts with HF transformers v5) and resource-debug's per-test output reach
    # the main-suite phases through PYTEST_ADDOPTS, and the special-test phases as the runner's options.
    local -a special_args=()
    local addopts=""
    if [[ $no_reruns -eq 1 ]]; then
        special_args+=(--no-reruns)
    else
        special_args+=("--reruns=${reruns_count}" "--reruns-delay=${reruns_delay}")
        addopts="--reruns ${reruns_count} --reruns-delay ${reruns_delay}"
    fi
    if [[ ${resource_debug:-0} -eq 1 ]]; then
        enable_resource_debug_env
        special_args+=(--resource-debug)
        addopts="${addopts:+${addopts} }-s"
        # resource snapshots at every phase boundary, and at every special-test boundary inside the runner
        export GPU_EPSILON_EVENT_CMD="${GPU_EPSILON_EVENT_CMD:-bash ${repo_home}/scripts/resource_snapshot.sh}"
        log_shell_resource_snapshot "coverage:bootstrap:$1" >> "$coverage_session_log" 2>&1 || true
    fi
    [[ -n "${addopts}" ]] && export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }${addopts}"

    # The phases run in pyproject.toml order: the default ones, plus the optional ones a full run adds. The driver
    # erases the coverage data first and writes the report last, even after a failed phase. Without --allow-failures
    # it stops at the first failed phase; with it every phase runs and the run still exits 1, naming each failed phase.
    local -a driver_args=(--log_file "${coverage_session_log}" --log-dir "${tmp_coverage_dir}")
    driver_args+=(--special-tests-args "${special_args[*]}")
    [[ $run_all_and_examples -eq 1 ]] && driver_args+=(--with profile,optional,benchmark)
    [[ $allow_failures -eq 1 ]] && driver_args+=(--allow-failures)
    [[ $dry_run -eq 1 ]] && driver_args+=(--dry-run)
    echo "Running the coverage phases through gpu-epsilon's phase driver: python -m gpu_epsilon.phases ${driver_args[*]}" >> "$coverage_session_log"
    # each special-test phase's console output goes to the temp log; the session log carries every phase's outcome
    python -m gpu_epsilon.phases "${driver_args[@]}" > "${temp_special_log}" 2>&1 || phase_status=$?
}

env_rebuild_collect(){
    if [[ $no_rebuild_base -eq 1 ]]; then
        require_existing_venv
        echo "Skipping rebuild of the base IT env ${target_env_name}" >> $coverage_session_log
    else
        echo "Beginning IT env rebuild for $1" >> $coverage_session_log
        check_self_test_only "Skipping all tests and examples." && return
        env_rebuild "$1"
    fi
    echo "Collecting coverage for the IT env $1" >> $coverage_session_log
    printf "\n"  >> $coverage_session_log
    collect_env_coverage "$1"
}

## Main coverage collection logic
start_time=$(date +%s)
echo "IT coverage collection executing at ${d} PT" > $coverage_session_log
echo "Generating base coverage for the IT env ${target_env_name}" >> $coverage_session_log
env_rebuild_collect "${target_env_name}"
# Hook for envs that need an EXTRA coverage pass beyond the standard phases. Unknown names are not
# an error here: the default is "no additional pass required", which is true of every env today.
case ${target_env_name} in
    # it_release_pt2_2_x)  # special path to be used when releasing a previous patch version after a new minor version available
    #     echo "Generating env-specific coverage for the IT env it_release_pt2_0_1" >> $coverage_session_log
    #     env_rebuild_collect "it_release_pt2_0_1"
    #     ;;
    *)
        echo "No env-specific additional coverage currently required for ${target_env_name}" >> $coverage_session_log
        ;;
esac
if [[ ${resource_debug:-0} -eq 1 && ${dry_run:-0} -ne 1 && ${self_test_only:-0} -ne 1 ]]; then
    # the session log carries the main-suite phases' output; each special-test phase keeps its own log
    summary_logs=("--log-file" "$coverage_session_log")
    if [[ -f "$phase_record_json" ]]; then
        while IFS= read -r phase_log; do
            [[ -f "$phase_log" ]] && summary_logs+=("--log-file" "$phase_log")
        done < <(python -c 'import json, sys; [print(p["log"]) for p in json.load(open(sys.argv[1]))["phases"] if p.get("log")]' "$phase_record_json")
    fi
    printf "\n" >> "$coverage_session_log"
    python "${repo_home}/scripts/resource_debug_summary.py" \
        "${summary_logs[@]}" \
        --json-output "$resource_summary_json" \
        >> "$coverage_session_log"
    echo "Resource summary JSON: ${resource_summary_json}" >> "$coverage_session_log"
fi
show_elapsed_time $coverage_session_log "IT coverage collection"
if [[ ${dry_run:-0} -eq 1 ]]; then
    echo "Dry run: the phase plan is in ${coverage_session_log}; nothing was executed." | tee -a "$coverage_session_log"
elif [[ ${phase_status:-0} -ne 0 ]]; then
    echo "Coverage phases failed (exit ${phase_status}); the phase summary in ${coverage_session_log} names each failed phase." | tee -a "$coverage_session_log" >&2
fi
exit ${phase_status:-0}
