#!/bin/bash
#
# Utility script to collect recently generated pytest expected dev state for interpretune profiling
# Usage examples:
# collect data from most recent 10 tests (the default number of tests):
#   ./collect_intepretune_profiling_results.sh
# collect data from most recent 5 tests:
#   ./collect_intepretune_profiling_results.sh 5
# collect data from most recent 10 tests for a specific user:
#   ./collect_intepretune_profiling_results.sh 10 username
default_num_tests=10
default_username=$(whoami)
tests_to_collect=${1:-$default_num_tests}
username=${2:-$default_username}
pytest_target_dir="/tmp/pytest-of-${username}"
pytest_pat=".*/pytest\-[0-9]*\/*/.*dev_state_log.yaml"
d=`date +%Y%m%d%H%M%S`
collected_state_log="/tmp/collected_state_${d}.log"
cd $pytest_target_dir
state_logs=$(find . -regextype sed -regex "$pytest_pat" -print0 | xargs -r -0 ls -1 -t | head -${tests_to_collect})
echo "Collecting requested state of most recent ${tests_to_collect} tests beginning at ${d}:" > $collected_state_log
for statelog in $state_logs; do
    cat $statelog >> $collected_state_log
done
cat $collected_state_log
