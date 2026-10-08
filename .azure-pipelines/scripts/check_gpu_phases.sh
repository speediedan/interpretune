#!/usr/bin/env bash
# Fail a FULL GPU run whose phase ran nothing in every memory class. Each class may be empty on its own (gpu_phase.sh
# lets it pass), but a full run whose whole phase was empty means the collector broke, and passing it would report a
# green phase that tested nothing. A selected run may legitimately select no test in a phase.
set -uo pipefail
COUNTS="${AGENT_TEMPDIRECTORY:-/tmp}/gpu_phase_counts"
if [[ "${IT_GPU_SELECTION_MODE:-full}" != full ]]; then
    echo "A ${IT_GPU_SELECTION_MODE} run: an empty phase is allowed."; exit 0
fi
bad=0
for phase in cuda standalone profile_ci; do
    states=$(cat "$COUNTS/${phase}".* 2>/dev/null | grep -E '^(ran|empty)$' | sort -u | tr '\n' ' ')
    if [[ -z "$states" ]]; then
        echo "ERROR: phase ${phase} recorded no class at all (did its steps run?)." >&2; bad=1
    elif [[ "$states" != *ran* ]]; then
        echo "ERROR: phase ${phase} ran no test in any memory class on a full run; the collector is suspect." >&2; bad=1
    else
        echo "phase ${phase}: $(for f in "$COUNTS/${phase}".*; do [[ "$f" == *.log ]] || printf '%s=%s ' "${f##*.}" "$(cat "$f")"; done)"
    fi
done
exit "$bad"
