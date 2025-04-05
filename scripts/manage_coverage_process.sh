#!/bin/bash
# Script to manage interpretune coverage process
# Ensures only one coverage collection process is running at a time

set -eo pipefail

usage() {
    echo "Usage: $0 [--coverage_args \"arguments for gen_it_coverage.sh\"]"
    echo "Example: $0 --coverage_args \"--repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --no_rebuild_base\""
    exit 1
}

COVERAGE_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --coverage_args)
            COVERAGE_ARGS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$COVERAGE_ARGS" ]]; then
    echo "Error: coverage_args is required"
    usage
fi

# Expand HOME variable in the arguments
COVERAGE_ARGS="${COVERAGE_ARGS//\${HOME\}/$HOME}"

# Extract target environment name for logging
if [[ "$COVERAGE_ARGS" =~ --target_env_name=([^ ]+) ]]; then
    TARGET_ENV="${BASH_REMATCH[1]}"
else
    TARGET_ENV="unknown"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_HOME="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/${TARGET_ENV}_coverage_nohup.out"

# Function to check if coverage processes are running
check_running_processes() {
    pids=$(pgrep -f "gen_it_coverage.sh|build_it_env.sh|special_tests.sh|python -m (coverage|pytest)" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "Found existing coverage processes:"
        ps -f -p "$pids"
        return 0
    else
        return 1
    fi
}

# Function to kill running coverage processes
kill_running_processes() {
    echo "Stopping existing coverage processes..."
    pkill -f "gen_it_coverage.sh|build_it_env.sh|special_tests.sh|python -m (coverage|pytest)" || true
    # Wait a moment to ensure processes are terminated
    sleep 2

    # Check if any processes are still running
    if check_running_processes; then
        echo "Some processes are still running. Attempting to force kill..."
        pkill -9 -f "gen_it_coverage.sh|build_it_env.sh|special_tests.sh|python -m (coverage|pytest)" || true
        sleep 1
    fi
}

# Main execution
echo "=== Starting coverage management $(date) ==="

# Check if coverage processes are already running
if check_running_processes; then
    echo "Coverage processes are already running. Stopping them before starting a new run."
    kill_running_processes
else
    echo "No existing coverage processes found."
fi

# Run the coverage script with nohup
echo "Starting coverage collection with args: $COVERAGE_ARGS"
echo "Output will be logged to $LOG_FILE"
nohup "$SCRIPT_DIR/gen_it_coverage.sh" $COVERAGE_ARGS > "$LOG_FILE" 2>&1 &

NOHUP_PID=$!
echo "Coverage process started with PID: $NOHUP_PID"
echo "=== Coverage management complete $(date) ==="
