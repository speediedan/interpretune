#!/bin/bash
#
# Utility script to update profiling memory stats for interpretune
set -eo pipefail

unset repo_home
unset target_env_name
unset working_dir

usage(){
>&2 cat << EOF
Usage: $0
  [ --repo-home input ]
  [ --target-env-name input ]
  [ --working-dir input ]
   [ --help ]
   Examples:
	# update profiling memory stats for it_latest:
	#   ./update_profiling_memory_stats.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --working-dir=/tmp
EOF
exit 1
}

args=$(getopt -o '' --long repo-home:,target-env-name:,working-dir:,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo-home)  repo_home=$2    ; shift 2  ;;
    --target-env-name)  target_env_name=$2  ; shift 2 ;;
    --working-dir)  working_dir=$2    ; shift 2  ;;
    --help)    usage      ; shift   ;;
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

d=`date +%Y%m%d%H%M%S`
profiling_session_log="${working_dir}/update_profiling_memory_stats_${target_env_name}_${d}.log"
echo "Use 'tail -f ${profiling_session_log}' to monitor progress"
source ${repo_home}/scripts/infra_utils.sh

update_profiling_memory_stats(){
    # Source common utility functions
    start_time=$(date +%s)

    cd ${repo_home}
    source ~/.venvs/${target_env_name}/bin/activate
    declare -a mark_types=("profile_ci" "profile" "optional")
    for mark_type in "${mark_types[@]}"; do
        echo "Running tests with ${mark_type} marker" >> $profiling_session_log
        # Save original memory footprints before updating
        orig_memory_footprint_path="${working_dir}/profile_memory_footprints_before_${mark_type}_update_${d}.yaml"
        echo "Saving original memory footprints to ${orig_memory_footprint_path}" >> $profiling_session_log
        (cd ${repo_home} && cp tests/parity_acceptance/profile_memory_footprints.yaml $orig_memory_footprint_path)

        # Run tests with memory profiling
        export IT_GLOBAL_STATE_LOG_MODE=1
        temp_per_type_out="${working_dir}/update_profiling_memory_stats_output_${mark_type}_${d}.out"
        echo "Executing tests and updating memory footprint yaml" >> $profiling_session_log
        (./tests/special_tests.sh --mark_type=${mark_type} --log_file=${profiling_session_log}) >> $temp_per_type_out 2>&1
        unset IT_GLOBAL_STATE_LOG_MODE

        # Generate diff between original and updated memory footprints
        mem_footprint_diff_path="${working_dir}/mem_footprint_changes_${mark_type}_${d}.out"
        echo "Generating memory footprint diff for ${mark_type} to ${mem_footprint_diff_path}" >> $profiling_session_log
        (cd ${repo_home} && diff -u tests/parity_acceptance/profile_memory_footprints.yaml $orig_memory_footprint_path > $mem_footprint_diff_path || true)

        # Output the diff to the log
        echo "Memory footprint changes for ${mark_type}:" >> $profiling_session_log
        cat $mem_footprint_diff_path >> $profiling_session_log
        echo "" >> $profiling_session_log
    done

    show_elapsed_time $profiling_session_log "Profiling memory stats update"
    echo "Results available in $profiling_session_log"
}

# Main execution
update_profiling_memory_stats
