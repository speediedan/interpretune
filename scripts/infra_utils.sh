#!/bin/bash
# infra utility functions

show_elapsed_time(){
  local test_log="$1"
  script_name=${2:-$(basename "$0")}
  ## write elapsed time in user-friendly fashion
  end_time=$(date +%s)
  elapsed_seconds=$(($end_time-$start_time))
  if (( $elapsed_seconds/60 == 0 )); then
      printf "${script_name} completed in $elapsed_seconds seconds \n" | tee -a $test_log
  elif (( $elapsed_seconds%60 == 0 )); then
      printf  "${script_name} completed in $(($elapsed_seconds/60)) minutes \n" | tee -a $test_log
  else
      printf "${script_name} completed in $(($elapsed_seconds/60)) minutes and $(($elapsed_seconds%60)) seconds \n" | tee -a $test_log
  fi
  printf "\n" | tee -a $test_log
}

# Function to safely deactivate a virtual environment if one is active
maybe_deactivate(){
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    # maybe replace with this in future:
    # deactivate 2>/dev/null || true
}
