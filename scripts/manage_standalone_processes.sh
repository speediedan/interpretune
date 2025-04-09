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
LOG_FILE="/tmp/${BASE_NAME}_${d}.log"

# Validate no processes are running
validate_process_not_running

echo "Starting command: $CMD"
echo "Output will be logged to: $LOG_FILE"

# Check if we should use nohup (VSCode kills nohup jobs: https://github.com/microsoft/vscode/issues/231216)
if [ "$USE_NOHUP" = true ]; then
    echo "Running in background with nohup..."
    nohup $CMD > "$LOG_FILE" 2>&1 &
    echo "Process started with PID: $!"
    echo "Use 'tail -f $LOG_FILE' to monitor progress"
else
    echo "Running in foreground (default for VSCode compatibility)..."
    $CMD > "$LOG_FILE" 2>&1
    echo "Process completed, output saved to: $LOG_FILE"
fi
