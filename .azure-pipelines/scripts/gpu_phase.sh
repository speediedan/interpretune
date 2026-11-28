#!/usr/bin/env bash
# Run one GPU phase for one memory class of staged placement, on the device the stage's lease granted.
#
#   gpu_phase.sh cuda|standalone|profile_ci small|large|multi
#
# A large stage sized by the collected declarations (IT_GPU_STAGES=3) runs the `large` class there and `multi` on the
# whole server; one sized without them (IT_GPU_STAGES=2) runs both as `rest` in the large stage, and its multi pass is
# a no-op.
#
# A class may legitimately select nothing (a change touching only small-class tests, or a phase whose tests are all
# one class), so an empty class passes here and records itself as empty. check_gpu_phases.sh, beside this, then fails
# a FULL run whose phase came out empty in every class, which on a full run means the collector broke.
#
# Strict mode is on: a GPU test placed on a device smaller than it declares fails rather than skips, since a skip
# would hide exactly the misplacement this placement exists to avoid.
set -uo pipefail

PHASE=${1:?cuda, standalone or profile_ci}
CLASS=${2:?small, large or multi}
COUNTS="${AGENT_TEMPDIRECTORY:-/tmp}/gpu_phase_counts"
mkdir -p "$COUNTS"
# shellcheck source=/dev/null
source /tmp/venvs/it_dev/bin/activate
# shellcheck source=/dev/null
source "$(dirname "$0")/telemetry.sh"

if [[ "${IT_GPU_TWO_STAGE:-0}" != 1 ]]; then
    # One-stage run (no lease tool): the first class runs the whole phase, the later ones have nothing left.
    [[ "$CLASS" != small ]] && { echo "One-stage run: the whole phase ran in the first stage."; exit 0; }
    CLASS=all
else
    if [[ "${IT_GPU_STAGES:-2}" != 3 ]]; then
        [[ "$CLASS" == multi ]] && { echo "Two-stage run: the multi-device tests ran in the large stage."; exit 0; }
        [[ "$CLASS" == large ]] && CLASS=rest
    fi
    export IT_GPU_MEM_CLASS="$CLASS" IT_GPU_SMALL_MAX_GB="${IT_GPU_SMALL_MAX_GB:?the small stage exports it}"
fi
if [[ -n "${IT_GPU_DEVICE:-}" && "${IT_GPU_DEVICE}" != all ]]; then
    export CUDA_VISIBLE_DEVICES="$IT_GPU_DEVICE"
fi
export IT_GPU_STRICT=1
# Record every pytest process the phase starts (gpu-epsilon's records plugin, through phase_evidence run-env) for the
# job's phase reports, once the job has collected the phases' manifests.
if [[ -f "/evidence/it-phases/manifests/${PHASE}.json" ]]; then
    eval "$(python -m gpu_epsilon.phase_evidence run-env --phase "$PHASE" --out-dir /evidence/it-phases)"
fi
echo "Phase ${PHASE}, class ${CLASS}, device ${IT_GPU_DEVICE:-all}"

record() { echo "$1" > "${COUNTS}/${PHASE}.${CLASS}"; }
# The phase's own interval, measured where it runs: the step's elapsed time also holds the container round trip and
# the venv activation, and a phase that never starts (its stage's lease not held) leaves no start event at all.
telemetry_event kind=phase phase="$PHASE" class="$CLASS" action=start device="${IT_GPU_DEVICE:-all}"
finish() { telemetry_event kind=phase phase="$PHASE" class="$CLASS" action=end rc="$1" ran="$2"; exit "$1"; }

case "$PHASE" in
    cuda)
        IT_RUN_CUDA_TESTS=1 python -m pytest --cov=src/interpretune --cov-append --cov-report= tests \
            -v --durations=50 --reruns 2 --reruns-delay 5
        rc=$?
        if (( rc == 5 )); then
            echo "No cuda-marked test in this class."; record empty; finish 0 empty
        fi
        record ran; finish "$rc" ran
        ;;
    standalone|profile_ci)
        log="${COUNTS}/${PHASE}.${CLASS}.log"
        # under the job's own host leases, so without --lease
        python -m gpu_epsilon.special_tests --config pyproject.toml --mark_type="$PHASE" --allow-failures --allow-empty 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        if grep -q "No tests were selected by" "$log"; then record empty; finish "$rc" empty; fi
        record ran; finish "$rc" ran
        ;;
    *)
        echo "usage: gpu_phase.sh cuda|standalone|profile_ci small|large|multi" >&2; exit 2 ;;
esac
