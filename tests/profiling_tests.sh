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
# Originally based on https://bit.ly/3AZGVVT
set -e

# use to run only tests marked as `profiling`
export IT_RUN_PROFILING_TESTS=1
# python arguments
defaults='-m coverage run --source src/interpretune --append -m pytest --capture=no --no-header -v -s'

parametrizations=$(pytest --collect-only --quiet "$@" | head -n -2)
parametrizations_arr=($parametrizations)

# tests to skip - space separated
blocklist=''
report=''

rm -f profiling_test_output.txt  # in case it exists, remove it
function show_output {
  if [ -f profiling_test_output.txt ]; then  # if exists
    cat profiling_test_output.txt
    # heuristic: stop if there are errors mentioned. this can prevent false negatives when only some of the ranks fail
    if grep --quiet --ignore-case --extended-regexp 'error|exception|traceback|failed' profiling_test_output.txt; then
      echo "Potential error! Stopping."
      rm profiling_test_output.txt
      exit 1
    fi
    rm profiling_test_output.txt
  fi
}
trap show_output EXIT  # show the output on exit

for i in "${!parametrizations_arr[@]}"; do
  parametrization=${parametrizations_arr[$i]}

  # check blocklist
  if echo $blocklist | grep -F "${parametrization}"; then
    report+="Skipped\t$parametrization\n"
    continue
  fi

  # run the test
  echo "Running ${parametrization}"
  python ${defaults} "${parametrization}"

  report+="Ran\t$parametrization\n"
done

show_output

# echo test report
printf '=%.s' {1..80}
printf "\n$report"
printf '=%.s' {1..80}
printf '\n'
