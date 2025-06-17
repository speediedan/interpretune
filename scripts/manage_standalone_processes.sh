#!/bin/bash
#
# Generic script to manage standalone processes
# Usage examples:
#   ./manage_standalone_processes.sh /path/to/script.sh --arg1 value1 --arg2 value2
#   ./manage_standalone_processes.sh python -m pytest tests/some_test.py
#   ./manage_standalone_processes.sh --use_nohup /path/to/script.sh --arg1 value1
set -eo pipefail

# Default configuration
USE_NOHUP=false

# Process flags
while [[ "$1" == --* ]]; do
  case "$1" in
    --use_nohup)
      USE_NOHUP=true
      shift
      ;;
    *)
      break
      ;;
  esac
done

# Get current date/time for log file naming
d=`date +%Y%m%d%H%M%S`

# Validate no conflicting processes are running
validate_process_not_running() {
    # Exclude the current script's PID from the search
    current_pid=$$
    if pgrep -f "gen_it_coverage.sh|build_it_env.sh|special_tests.sh|update_profiling_memory_stats.sh|python -m (coverage|pytest)" | grep -v "^${current_pid}$" > /dev/null; then
        echo "Error: Found running processes that may conflict:"
        pgrep -fa "gen_it_coverage.sh|build_it_env.sh|special_tests.sh|update_profiling_memory_stats.sh|python -m (coverage|pytest)" | grep -v "^${current_pid}$"
        exit 1
    fi
    echo "No conflicting processes found, proceeding..."
}

if [ -z "$1" ]; then
    echo "Usage: $0 command [arguments]"
    echo "Example: $0 ./gen_it_coverage.sh --arg1 value1"
    exit 1
fi

# Use all arguments, not just the first one
CMD="$@"
SCRIPT_NAME="$1"
BASE_NAME=$(basename "$SCRIPT_NAME" .sh)
WRAPPER_OUT="/tmp/${BASE_NAME}_${d}_wrapper.out"

# Validate no processes are running
validate_process_not_running

echo "Starting command: $CMD"
# Check if we should use nohup (VSCode kills nohup jobs: https://github.com/microsoft/vscode/issues/231216)
if [ "$USE_NOHUP" = true ]; then
    echo "Running in background with nohup..."
    nohup $CMD > "$WRAPPER_OUT" 2>&1 &
    echo "Wrapper process started with PID: $!" | tee -a "$WRAPPER_OUT"
    echo "Wrapper output logged to: $WRAPPER_OUT"
    echo "First 5 lines of output will be displayed in 3 seconds as a sanity check..."
    echo `printf "%0.s-" {1..120} && printf "\n"`
    sleep 3
    cat "$WRAPPER_OUT" | head -n 5
else
    echo "Running in foreground (default for VSCode compatibility)..."
    $CMD > "$WRAPPER_OUT" 2>&1
    echo "Wrapper process completed, wrapper output saved to: $WRAPPER_OUT"
fi
