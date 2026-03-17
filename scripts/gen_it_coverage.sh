#!/bin/bash
#
# Utility script to generate local IT coverage for a given environment
#
# Torch handling:
# - Uses torch-pre.txt if present (for nightly/test builds)
# - Otherwise uses stable torch with automatic backend detection
# - Use --torch-backend=cpu to force CPU-only torch
set -eo pipefail

# Source shared infrastructure utilities
source "$(dirname "${BASH_SOURCE[0]}")/infra_utils.sh"

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
    [ --python-version input ]  (e.g., python3.12, default: python3.13)
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
    [ --allow-failures ]  (continue collecting coverage even if tests fail; useful for debugging)
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
    # generate it_latest coverage with self_test_only:
    #   ./gen_it_coverage.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --self-test-only
EOF
exit 1
}

args=$(getopt -o '' --long repo-home:,target-env-name:,python-version:,torch-backend:,no-rebuild-base,from-source:,venv-dir:,run-all-and-examples,no-export-cov-xml,pip-install-flags:,self-test-only,it-build-flags:,no-reruns,reruns:,reruns-delay:,allow-failures,resource-debug,help -- "$@")
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
echo "Use 'tail -f ${coverage_session_log}' to monitor progress"

# Expand leading ~ in common path arguments
repo_home=$(expand_tilde "${repo_home}")
venv_dir=$(expand_tilde "${venv_dir}")

# Determine venv path
# Priority: --venv-dir > IT_VENV_BASE > default ~/.venvs
venv_path=$(determine_venv_path "${venv_dir}" "${target_env_name}")

# Strip leading/trailing quotes from string variables if present
if [[ -n "${from_source_spec}" ]]; then
    from_source_spec=$(strip_quotes "$from_source_spec")
fi
if [[ -n "${it_build_flags}" ]]; then
    it_build_flags=$(strip_quotes "$it_build_flags")
fi

# Build rerun args string (for transient httpx read timeouts with HF transformers v5)
if [[ $no_reruns -eq 1 ]]; then
    rerun_args=""
else
    rerun_args="--reruns ${reruns_count} --reruns-delay ${reruns_delay}"
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

    case $1 in
        it_latest | it_release )
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
            eval ${build_cmd} >> $coverage_session_log 2>&1
            ;;
        *)
            echo "no matching environment found, exiting..." >> $coverage_session_log
            exit 1
            ;;
    esac
}

collect_env_coverage(){
    temp_special_log="${tmp_coverage_dir}/special_test_output_$1_${d}.log"
    cd ${repo_home}
    maybe_deactivate
    source ${venv_path}/bin/activate
    export PYTHONFAULTHANDLER=1
    if [[ ${resource_debug:-0} -eq 1 ]]; then
        enable_resource_debug_env
        log_shell_resource_snapshot "coverage:bootstrap:$1" >> "$coverage_session_log" 2>&1 || true
    fi

    cuda_phase_available(){
        python - <<'PY'
import sys

import torch

raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    }

    run_logged_phase(){
        local phase_name=$1
        shift
        log_shell_resource_snapshot "coverage-phase:start:${phase_name}" >> "$coverage_session_log" 2>&1 || true
        set +e
        "$@"
        local phase_status=$?
        set -e
        echo "${phase_name} exit code: ${phase_status}" >> "$coverage_session_log"
        log_shell_resource_snapshot "coverage-phase:end:${phase_name}" >> "$coverage_session_log" 2>&1 || true
        if [[ $phase_status -ne 0 && $allow_failures -ne 1 ]]; then
            return $phase_status
        fi
        return 0
    }

    run_logged_cuda_phase(){
        local phase_name=$1
        shift
        if cuda_phase_available; then
            run_logged_phase "$phase_name" "$@"
        else
            echo "${phase_name} skipped: CUDA unavailable in current environment" >> "$coverage_session_log"
            return 0
        fi
    }

    # Build special_tests.sh rerun args to pass through
    special_tests_rerun_args=""
    [[ $no_reruns -eq 1 ]] && special_tests_rerun_args="--no-reruns"
    [[ -n "${reruns_count}" && $no_reruns -ne 1 ]] && special_tests_rerun_args="${special_tests_rerun_args} --reruns=${reruns_count}"
    [[ -n "${reruns_delay}" && $no_reruns -ne 1 ]] && special_tests_rerun_args="${special_tests_rerun_args} --reruns-delay=${reruns_delay}"
    pytest_resource_args=""
    if [[ ${resource_debug:-0} -eq 1 ]]; then
        pytest_resource_args="-s"
    fi

    # Prepare allow_failures flag for special_tests.sh
    local failures_flag=""
    if [[ $allow_failures -eq 1 ]]; then
        failures_flag="--allow-failures"
        echo "Running in --allow-failures mode: coverage collection will continue past test failures." >> $coverage_session_log
    fi
    case $1 in
        it_latest )
            check_self_test_only "Skipping all tests and examples." && return
            python -m coverage erase
            if [[ $run_all_and_examples -eq 1 ]]; then
                # Using pytest-cov ensures coverage starts before test collection imports
                run_logged_phase \
                    "base pytest" \
                    env CUDA_VISIBLE_DEVICES='' python -X faulthandler -m pytest --cov=src/interpretune --cov-report= src/interpretune src/it_examples tests -v ${pytest_resource_args} ${rerun_args} \
                    >> "$coverage_session_log" 2>&1
                run_logged_cuda_phase \
                    "base pytest cuda-marked" \
                    env -u CUDA_VISIBLE_DEVICES IT_RUN_CUDA_TESTS=1 python -X faulthandler -m pytest --cov=src/interpretune --cov-append --cov-report= tests -v ${pytest_resource_args} ${rerun_args} \
                    >> "$coverage_session_log" 2>&1
                run_logged_phase \
                    "special tests standalone" \
                    bash -lc "./tests/special_tests.sh --mark_type=standalone --log_file=${coverage_session_log} ${special_tests_rerun_args} ${failures_flag} ${resource_debug:+--resource-debug} >> ${temp_special_log} 2>&1"
                run_logged_phase \
                    "special tests profile_ci" \
                    bash -lc "./tests/special_tests.sh --mark_type=profile_ci --log_file=${coverage_session_log} ${special_tests_rerun_args} ${failures_flag} ${resource_debug:+--resource-debug} >> ${temp_special_log} 2>&1"
                run_logged_phase \
                    "special tests profile" \
                    bash -lc "./tests/special_tests.sh --mark_type=profile --log_file=${coverage_session_log} ${special_tests_rerun_args} ${failures_flag} ${resource_debug:+--resource-debug} >> ${temp_special_log} 2>&1"
                run_logged_phase \
                    "special tests optional" \
                    bash -lc "./tests/special_tests.sh --mark_type=optional --log_file=${coverage_session_log} ${special_tests_rerun_args} ${failures_flag} ${resource_debug:+--resource-debug} >> ${temp_special_log} 2>&1"
            else
                # Using pytest-cov ensures coverage starts before test collection imports
                run_logged_phase \
                    "base pytest" \
                    env CUDA_VISIBLE_DEVICES='' python -X faulthandler -m pytest --cov=src/interpretune --cov-append --cov-report= tests -v ${pytest_resource_args} ${rerun_args} \
                    >> "$coverage_session_log" 2>&1
                run_logged_cuda_phase \
                    "base pytest cuda-marked" \
                    env -u CUDA_VISIBLE_DEVICES IT_RUN_CUDA_TESTS=1 python -X faulthandler -m pytest --cov=src/interpretune --cov-append --cov-report= tests -v ${pytest_resource_args} ${rerun_args} \
                    >> "$coverage_session_log" 2>&1
                run_logged_phase \
                    "special tests standalone" \
                    bash -lc "./tests/special_tests.sh --mark_type=standalone --log_file=${coverage_session_log} ${special_tests_rerun_args} ${failures_flag} ${resource_debug:+--resource-debug} >> ${temp_special_log} 2>&1"
                run_logged_phase \
                    "special tests profile_ci" \
                    bash -lc "./tests/special_tests.sh --mark_type=profile_ci --log_file=${coverage_session_log} ${special_tests_rerun_args} ${failures_flag} ${resource_debug:+--resource-debug} >> ${temp_special_log} 2>&1"
            fi
            ;;
        *)
            echo "no matching environment found, exiting..."  >> $coverage_session_log
            exit 1
            ;;
    esac
}

env_rebuild_collect(){
    if [[ $no_rebuild_base -eq 1 ]]; then
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
case ${target_env_name} in
    it_latest)
        echo "No env-specific additional coverage currently required for ${target_env_name}" >> $coverage_session_log
        ;;
    it_release)
        echo "No env-specific additional coverage currently required for ${target_env_name}" >> $coverage_session_log
        ;;
    # it_release_pt2_2_x)  # special path to be used when releasing a previous patch version after a new minor version available
    #     echo "Generating env-specific coverage for the IT env it_release_pt2_0_1" >> $coverage_session_log
    #     env_rebuild_collect "it_release_pt2_0_1"
    #     ;;
    *)
        echo "no matching environment found, exiting..."  >> $coverage_session_log
        exit 1
        ;;
esac
echo "Writing collected coverage stats for IT env ${target_env_name}" >> $coverage_session_log
# Reactivate the environment for coverage report (in case it was deactivated)
source ${venv_path}/bin/activate
python -m coverage report -m >> $coverage_session_log
if [[ ${resource_debug:-0} -eq 1 ]]; then
    printf "\n" >> "$coverage_session_log"
    python "${repo_home}/scripts/resource_debug_summary.py" \
        --log-file "$coverage_session_log" \
        --json-output "$resource_summary_json" \
        >> "$coverage_session_log"
    echo "Resource summary JSON: ${resource_summary_json}" >> "$coverage_session_log"
fi
show_elapsed_time $coverage_session_log "IT coverage collection"
