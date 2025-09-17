#!/bin/bash
#
# Utility script to generate local IT coverage for a given environment
set -eo pipefail

unset repo_home
unset target_env_name
unset torch_dev_ver
unset torch_test_channel
unset no_rebuild_base
unset fts_from_source
unset ct_from_source
unset run_all_and_examples
unset no_export_cov_xml
unset pip_install_flags
unset self_test_only
unset ct_commit_pin
unset apply_post_upgrades

usage(){
>&2 cat << EOF
Usage: $0
    [ --repo-home input]
    [ --target-env-name input ]
    [ --torch-dev-ver input ]
    [ --torch-test-channel ]
    [ --no-rebuild-base ]
    [ --fts-from-source "path" ]
    [ --ct-from-source "path" ]
    [ --run-all-and-examples ]
    [ --no-export-cov-xml ]
    [ --pip-install-flags "flags" ]
    [ --self-test-only ]
    [ --ct-commit-pin ]
     [ --apply-post-upgrades ]
   [ --help ]
   Examples:
    # generate it_latest coverage without rebuilding the it_latest base environment:
    #   ./gen_it_coverage.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --no_rebuild_base
    # generate it_latest coverage with a given torch_dev_version, rebuilding base it_latest and with FTS from source:
    #   ./gen_it_coverage.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --torch_dev_ver=dev20240201 --fts_from_source=${HOME}/repos/finetuning-scheduler
    # generate it_latest coverage, rebuilding base it_latest with PyTorch test channel and FTS from source:
    #   ./gen_it_coverage.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --torch_test_channel --fts_from_source=${HOME}/repos/finetuning-scheduler
    # generate it_latest coverage with circuit-tracer from source:
    #   ./gen_it_coverage.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --ct_from_source=${HOME}/repos/circuit-tracer
    # generate it_latest coverage with no pip cache:
    #   ./gen_it_coverage.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --pip_install_flags="--no-cache-dir"
    # generate it_latest coverage with self_test_only:
    #   ./gen_it_coverage.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --self_test_only
    # generate it_latest coverage using CT commit pinning:
    #   ./gen_it_coverage.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --ct_commit_pin
EOF
exit 1
}

args=$(getopt -o '' --long repo-home:,target-env-name:,torch-dev-ver:,torchvision-dev-ver:,torch-test-channel,no-rebuild-base,fts-from-source:,ct-from-source:,run-all-and-examples,no-export-cov-xml,pip-install-flags:,self-test-only,ct-commit-pin,apply-post-upgrades,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo-home)  repo_home=$2    ; shift 2  ;;
    --target-env-name)  target_env_name=$2  ; shift 2 ;;
    --torch-dev-ver)   torch_dev_ver=$2   ; shift 2 ;;
    --torch-test-channel)   torch_test_channel=1 ; shift  ;;
    --no-rebuild-base)   no_rebuild_base=1 ; shift  ;;
    --fts-from-source)   fts_from_source=$2 ; shift 2 ;;
    --ct-from-source)   ct_from_source=$2 ; shift 2 ;;
    --run-all-and-examples)   run_all_and_examples=1 ; shift  ;;
    --no-export-cov-xml)   no_export_cov_xml=1 ; shift ;;
    --pip-install-flags)   pip_install_flags=$2 ; shift 2 ;;
    --self-test-only)   self_test_only=1 ; shift ;;
    --ct-commit-pin)   ct_commit_pin=1 ; shift ;;
    --apply-post-upgrades)   apply_post_upgrades=1 ; shift ;;
    --help)    usage      ; shift   ;;
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

d=`date +%Y%m%d%H%M%S`
tmp_coverage_dir="/tmp"
coverage_session_log="${tmp_coverage_dir}/gen_it_coverage_${target_env_name}_${d}.log"
echo "Use 'tail -f ${coverage_session_log}' to monitor progress"

# Robustness: strip leading/trailing quotes from path variables if present
strip_quotes() {
    # Remove both single and double quotes from start/end
    local val="$1"
    val="${val%\' }"; val="${val#\' }"
    val="${val%\"}"; val="${val#\"}"
    echo "$val"
}

# Only strip if set
if [[ -n "${ct_from_source}" ]]; then
    ct_from_source=$(strip_quotes "$ct_from_source")
fi
if [[ -n "${fts_from_source}" ]]; then
    fts_from_source=$(strip_quotes "$fts_from_source")
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
    # Prepare pip_install_flags parameter if set
    pip_flags_param=""
    if [[ -n "${pip_install_flags}" ]]; then
        pip_flags_param="--pip-install-flags=\"${pip_install_flags}\""
    fi

    fts_from_source_param=""
    if [[ -n "${fts_from_source}" ]]; then
        fts_from_source_param="--fts-from-source=${fts_from_source}"
    fi

    ct_from_source_param=""
    if [[ -n "${ct_from_source}" ]]; then
        ct_from_source_param="--ct-from-source=${ct_from_source}"
    fi

    ct_commit_pin_param=""
    if [[ -n "${ct_commit_pin}" ]]; then
        ct_commit_pin_param="--no-commit-pin"
    fi

    apply_post_upgrades_param=""
    if [[ -n "${apply_post_upgrades}" ]]; then
        apply_post_upgrades_param="--apply-post-upgrades"
    fi

    case $1 in
        it_latest )
            if [[ -n ${torch_dev_ver} ]]; then
                ${repo_home}/scripts/build_it_env.sh --repo-home=${repo_home} --target-env-name=$1 --torch-dev-ver=${torch_dev_ver} ${fts_from_source_param} ${ct_from_source_param} ${pip_flags_param} ${ct_commit_pin_param} ${apply_post_upgrades_param}
            elif [[ $torch_test_channel -eq 1 ]]; then
                ${repo_home}/scripts/build_it_env.sh --repo-home=${repo_home} --target-env-name=$1 --torch-test-channel  ${fts_from_source_param} ${ct_from_source_param} ${pip_flags_param} ${ct_commit_pin_param} ${apply_post_upgrades_param}
            else
                ${repo_home}/scripts/build_it_env.sh --repo-home=${repo_home} --target-env-name=$1 ${fts_from_source_param} ${ct_from_source_param} ${pip_flags_param} ${ct_commit_pin_param} ${apply_post_upgrades_param}
            fi
            ;;
        it_release )
            ${repo_home}/scripts/build_it_env.sh --repo-home=${repo_home} --target-env-name=$1 ${fts_from_source_param} ${ct_from_source_param} ${pip_flags_param} ${ct_commit_pin_param} ${apply_post_upgrades_param}
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
    source ./scripts/infra_utils.sh
    maybe_deactivate
    source ~/.venvs/$1/bin/activate
    case $1 in
        it_latest | it_latest_pt_2_4 )
            check_self_test_only "Skipping all tests and examples." && return
            python -m coverage erase
            if [[ $run_all_and_examples -eq 1 ]]; then
                #check_self_test_only "Skipping all tests and examples." && return
                python -m coverage run --source src/interpretune -m pytest src/interpretune src/it_examples tests -v 2>&1 >> $coverage_session_log
                (./tests/special_tests.sh --mark_type=standalone --log_file=${coverage_session_log} 2>&1 >> ${temp_special_log}) > /dev/null
                (./tests/special_tests.sh --mark_type=profile_ci --log_file=${coverage_session_log} 2>&1 >> ${temp_special_log}) > /dev/null
                (./tests/special_tests.sh --mark_type=profile --log_file=${coverage_session_log} 2>&1 >> ${temp_special_log}) > /dev/null
                (./tests/special_tests.sh --mark_type=optional --log_file=${coverage_session_log} 2>&1 >> ${temp_special_log}) > /dev/null
            else
                #check_self_test_only "Skipping all tests and examples." && return
                # if check_self_test_only "Skipping CI tests."; then
                #     return
                # fi
                python -m coverage run --append --source src/interpretune -m pytest tests -v 2>&1 >> $coverage_session_log
                (./tests/special_tests.sh --mark_type=standalone --log_file=${coverage_session_log} 2>&1 >> ${temp_special_log}) > /dev/null
                (./tests/special_tests.sh --mark_type=profile_ci --log_file=${coverage_session_log} 2>&1 >> ${temp_special_log}) > /dev/null
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
        # if check_self_test_only "Skipping rebuild."; then
        #     return
        # fi
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
python -m coverage report -m >> $coverage_session_log
show_elapsed_time $coverage_session_log "IT coverage collection"
