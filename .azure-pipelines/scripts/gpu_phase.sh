#!/usr/bin/env bash
# Run one GPU phase for one memory class of two-stage placement, on the device the stage's lease granted.
#
#   gpu_phase.sh cuda|standalone|profile_ci small|large
#
# A class may legitimately select nothing (a change touching only small-class tests, or a phase whose tests are all
# one class), so an empty class passes here and records itself as empty. check_gpu_phases.sh, beside this, then fails
# a FULL run whose phase came out empty in every class, which on a full run means the collector broke.
#
# Strict mode is on: a GPU test placed on a device smaller than it declares fails rather than skips, since a skip
# would hide exactly the misplacement this placement exists to avoid.
set -uo pipefail

PHASE=${1:?cuda, standalone or profile_ci}
CLASS=${2:?small or large}
COUNTS="${AGENT_TEMPDIRECTORY:-/tmp}/gpu_phase_counts"
mkdir -p "$COUNTS"
# shellcheck source=/dev/null
source /tmp/venvs/it_dev/bin/activate

if [[ "${IT_GPU_TWO_STAGE:-0}" != 1 ]]; then
    # One-stage run (no lease tool): the first class runs the whole phase, the second has nothing left.
    [[ "$CLASS" == large ]] && { echo "One-stage run: the whole phase ran in the first stage."; exit 0; }
    CLASS=all
else
    export IT_GPU_MEM_CLASS="$CLASS" IT_GPU_SMALL_MAX_GB="${IT_GPU_SMALL_MAX_GB:?the small stage exports it}"
fi
if [[ -n "${IT_GPU_DEVICE:-}" && "${IT_GPU_DEVICE}" != all ]]; then
    export CUDA_VISIBLE_DEVICES="$IT_GPU_DEVICE"
fi
export IT_GPU_STRICT=1
echo "Phase ${PHASE}, class ${CLASS}, device ${IT_GPU_DEVICE:-all}"

record() { echo "$1" > "${COUNTS}/${PHASE}.${CLASS}"; }

case "$PHASE" in
    cuda)
        IT_RUN_CUDA_TESTS=1 python -m pytest --cov=src/interpretune --cov-append --cov-report= tests src/it_examples/tests \
            -v --durations=50 --reruns 2 --reruns-delay 5
        rc=$?
        if (( rc == 5 )); then
            echo "No cuda-marked test in this class."; record empty; exit 0
        fi
        record ran; exit "$rc"
        ;;
    standalone|profile_ci)
        log="${COUNTS}/${PHASE}.${CLASS}.log"
        bash ./tests/special_tests.sh --mark_type="$PHASE" --allow-failures --allow-empty 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        if grep -q "No tests were selected by" "$log"; then record empty; else record ran; fi
        exit "$rc"
        ;;
    *)
        echo "usage: gpu_phase.sh cuda|standalone|profile_ci small|large" >&2; exit 2 ;;
esac
