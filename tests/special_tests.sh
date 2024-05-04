#!/bin/bash
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
# TODO: add parameters supporting conditional running of example tests

set -eo pipefail

unset mark_type
unset log_file
unset IT_RUN_PROFILING_TESTS
unset IT_RUN_STANDALONE_TESTS

usage(){
>&2 cat << EOF
Usage: $0
   [ --mark_type input]
   [ --log_file input]
   [ --help ]
   Examples:
	# run profile tests marked to run with CI:
	#   ./special_tests.sh --mark_type=profile
	# run all profile tests:
	#   ./special_tests.sh --mark_type=profile_ci --log_file=/tmp/some_parent_process_file_to_append_to.log
  # run all standalone tests:
  #   ./special_tests.sh --mark_type=standalone
EOF
exit 1
}

args=$(getopt -o '' --long mark_type:,log_file:,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --mark_type)  mark_type=$2    ; shift 2  ;;
    --log_file)  log_file=$2    ; shift 2  ;;
    --help)    usage      ; shift   ;;
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

d=`date +%Y%m%d%H%M%S`
tmp_log_dir="/tmp"
special_test_session_log=${log_file:-"${tmp_log_dir}/special_tests_${mark_type}_${d}.log"}
special_test_session_tmp_out="${tmp_log_dir}/special_tests_${mark_type}_${d}.out"

collect_tests(){
  printf "Collected the following tests: \n" >> $special_test_session_log
  special_tests=$(python3 -m pytest tests -q --collect-only --pythonwarnings ignore | tee -a $special_test_session_log)
  # echo "Collected tests: \n ${special_tests}" >> $special_test_session_log
  # match only lines with tests
  declare -a -g parameterizations=($(grep -oP '\S+::test_\S+' <<< "$special_tests"))
}

execute_tests(){
  # hardcoded tests to skip - space separated
  blocklist=''
  export report=''

  for i in "${!parameterizations[@]}"; do
    parameterization=${parameterizations[$i]}

    # check blocklist
    if echo $blocklist | grep -F "${parameterization}"; then
      report+="Skipped\t$parameterization\n"
      continue
    fi

    # run the test
    echo "Running ${parameterization}" >> $special_test_session_log
    python ${defaults} ${parameterization} >> $special_test_session_tmp_out

    report+="Ran\t$parameterization\n"
  done
}

show_test_results(){

  if [ -f ${special_test_session_tmp_out} ]; then  # if exists
    cat $special_test_session_tmp_out
    if grep --quiet --ignore-case --extended-regexp 'error|exception|traceback|failed' ${special_test_session_tmp_out} ; then
      echo "Potential error!"
      rm ${special_test_session_tmp_out}
      exit 1
    fi
    rm ${special_test_session_tmp_out}
  elif [ -f ${special_test_session_log} ]; then  # if the log but not the out exists, check for collection errors
    if grep --ignore-case --extended-regexp 'traceback|failed' ${special_test_session_log} ; then
      echo "Potential collection error!"
      exit 1
    fi
  fi
}
trap show_test_results EXIT  # show the output on exit


## Main coverage collection logic
start_time=$(date +%s)
echo "IT special tests beginning execution at ${d} PT" >> $special_test_session_log

case ${mark_type} in
  profile)
    echo "Collecting and running profile tests..." >> $special_test_session_log
    export IT_RUN_PROFILING_TESTS=2
    ;;
  profile_ci)
    echo "Collecting and running only profile tests marked for CI..." >> $special_test_session_log
    export IT_RUN_PROFILING_TESTS=1
    ;;
  standalone)
    echo "Collecting and running standalone tests..." >> $special_test_session_log
    export IT_RUN_STANDALONE_TESTS=1
    ;;
  *)
    echo "no matching `mark_type` found, exiting..."  >> $special_test_session_log
    exit 1
    ;;
esac

# default python coverage arguments
defaults='-m coverage run --source src/interpretune --append -m pytest --capture=no --no-header -v -s'
collect_tests
execute_tests
show_test_results

## write elapsed time in user-friendly fashion
end_time=$(date +%s)
elapsed_seconds=$(($end_time-$start_time))
if (( $elapsed_seconds/60 == 0 )); then
            echo "Script completed in $elapsed_seconds seconds" >> $special_test_session_log
elif (( $elapsed_seconds%60 == 0 )); then
    echo "Script completed in $(($elapsed_seconds/60)) minutes" >> $special_test_session_log
else
    echo "Script completed in $(($elapsed_seconds/60)) minutes and $(($elapsed_seconds%60)) seconds" >> $special_test_session_log
fi

# summarize test report
printf '=%.s' {1..80} >> $special_test_session_log
printf "\n$report" >> $special_test_session_log
printf '=%.s' {1..80} >> $special_test_session_log
printf '\n' >> $special_test_session_log
