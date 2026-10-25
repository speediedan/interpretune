#!/usr/bin/env bash
# Take or release the host lease for one stage of two-stage GPU placement.
#
#   gpu_stage_lease.sh acquire small|large
#   gpu_stage_lease.sh release small|large|all
#
# small  one device with at least the smallest device's memory, the smallest free one that fits (best fit), plus
#        cpu-heavy. The small memory class (single-device tests declaring no more than that, and no bf16) runs here,
#        so it can run on the small card while someone else holds the large one.
# large  the whole server, plus cpu-heavy. Everything else runs here: larger declarations, multi-device and bf16
#        tests, and the extended tiers.
#
# Each stage holds its lease through ci_host_lease_hold (scripts/gpu_lease_wrap.sh, shared with the other
# repositories on this agent), which uses the host's lease tool, mounted read-only at /gpu_lease_tool, so this job
# queues in the same fair order as every local caller, is placed by the same best-fit rule, and waits for cpu-heavy
# before taking a GPU. The tool is host infrastructure behind a seam: when it is absent the job falls back to the
# plain whole-server locks (one stage, no placement), and the second stage is a no-op. The lease directory being
# absent means the host does not use leases.
#
# Wait policy, from the GPU run planner: IT_GPU_LEASE_WAIT seconds, and on a timeout either fail
# (IT_GPU_LEASE_ON_TIMEOUT=fail, a pull request gate) or cancel the build (cancel, a scheduled run). A scheduled run
# shares one deadline across both stages, so a busy host costs it at most that long in all.
#
# IT_GPU_ONE_STAGE=1 (the any-vendor template sets it on an AMD agent, whose two cards are identical, so placing a small
# class buys nothing there) takes the one whole-server stage even when the tool is usable.
#
# Exports to later steps: IT_GPU_TWO_STAGE (1 or 0), IT_GPU_DEVICE (a device UUID, or all), IT_GPU_SMALL_MAX_GB.
set -uo pipefail

ACTION=${1:?acquire or release}
STAGE=${2:?small, large or all}
DIR=${GPU_STAGE_LEASE_DIR:-/gpu_leases}       # overridable for tests; the pipeline mounts these two paths
TOOL=${GPU_STAGE_LEASE_TOOL:-/gpu_lease_tool}
LABEL="azure-it-${BUILD_BUILDID:-local}"
PIDDIR=${GPU_STAGE_LEASE_PIDDIR:-/tmp}         # overridable for tests, so they never write the host's /tmp
PIDFILE() { echo "${PIDDIR}/azp_gpu_${1}.pid"; }
DEADLINE_FILE=${PIDDIR}/azp_gpu_lease_deadline
setvar() { echo "##vso[task.setvariable variable=$1]$2"; }

# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/gpu_lease_wrap.sh"

tool_usable() { [[ -f "$TOOL" && -x "$TOOL" ]]; }

wait_secs() {
    local wait=${IT_GPU_LEASE_WAIT:-2400}
    if [[ "${IT_GPU_LEASE_ON_TIMEOUT:-fail}" == cancel ]]; then
        [[ -f "$DEADLINE_FILE" ]] || echo $(( $(date +%s) + wait )) > "$DEADLINE_FILE"
        wait=$(( $(cat "$DEADLINE_FILE") - $(date +%s) ))
        (( wait > 60 )) || wait=60
    fi
    echo "$wait"
}

# A timed-out scheduled run cancels itself; a timed-out gate fails. A cancel request that fails or does not take
# effect fails closed, so the GPU phases never run unleased.
on_timeout() {
    if [[ "${IT_GPU_LEASE_ON_TIMEOUT:-fail}" == cancel ]]; then
        echo "##vso[task.logissue type=warning]The host leases stayed busy past this run's wait; cancelling this scheduled run rather than failing it."
        curl -sS -f -X PATCH -H "Authorization: Bearer ${SYSTEM_ACCESSTOKEN:-}" -H "Content-Type: application/json" \
            -d '{"status": "cancelling"}' \
            "${SYSTEM_COLLECTIONURI:-}${SYSTEM_TEAMPROJECT:-}/_apis/build/builds/${BUILD_BUILDID:-}?api-version=7.1" >/dev/null \
            && sleep 300
        echo "ERROR: the cancel request failed or did not take effect; failing closed instead." >&2
    fi
    exit 1
}

# The smallest device's total memory in whole GiB: the largest declaration the small class takes. GPU_LEASE_DEVICES
# ("<uuid>:<MiB>,...") stands in for nvidia-smi, as it does for the lease tool itself.
small_max_gb() {
    if [[ -n "${GPU_LEASE_DEVICES:-}" ]]; then
        tr ',' '\n' <<< "$GPU_LEASE_DEVICES" | cut -d: -f2
    else
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null
    fi | sort -n | head -1 | awk '{printf "%d", $1 / 1024}'
}

# Hold the lease for one stage; the request is the lease tool's own options. A timeout follows the wait policy.
hold() {
    local stage=$1 rc
    shift
    ci_host_lease_hold "$DIR" "$TOOL" "$(PIDFILE "$stage")" "$(wait_secs)" "$LABEL" "$@"
    rc=$?
    (( rc == 75 )) && on_timeout
    (( rc == 0 )) || exit "$rc"
}

case "$ACTION/$STAGE" in
    acquire/small)
        if [[ ! -d "$DIR" ]]; then
            echo "gpu_stage_lease: ${DIR} not mounted; this host does not use GPU leases. Proceeding unleased, one stage."
            setvar IT_GPU_TWO_STAGE 0; setvar IT_GPU_DEVICE all; exit 0
        fi
        if ! tool_usable || [[ "${IT_GPU_ONE_STAGE:-0}" == 1 ]]; then
            # Without the tool a device cannot be placed (and a one-stage host asks for none), so the run is one
            # whole-server stage. It is held under the large stage's pidfile: the pipeline's "release small" step must
            # leave it held for the later phases, and "release all" frees it at the end.
            hold large --gpus all --cpu-heavy
            setvar IT_GPU_TWO_STAGE 0; setvar IT_GPU_DEVICE all; exit 0
        fi
        small=$(small_max_gb)
        [[ "$small" =~ ^[0-9]+$ && "$small" -gt 0 ]] || { echo "ERROR: cannot read the devices' memory from nvidia-smi." >&2; exit 1; }
        hold small --gpus 1 --min-vram "$small" --cpu-heavy
        device=$(cat "$(PIDFILE small).device")
        echo "gpu_stage_lease: small stage holds ${device} (classes declaring up to ${small} GiB, no bf16)."
        setvar IT_GPU_TWO_STAGE 1; setvar IT_GPU_DEVICE "$device"; setvar IT_GPU_SMALL_MAX_GB "$small"
        ;;
    acquire/large)
        if [[ "${IT_GPU_TWO_STAGE:-0}" != 1 ]]; then
            echo "gpu_stage_lease: one-stage run; the first stage's lease already covers this."; exit 0
        fi
        hold large --gpus all --cpu-heavy
        echo "gpu_stage_lease: large stage holds the whole server."
        setvar IT_GPU_DEVICE all
        ;;
    release/small|release/large)
        ci_host_lease_release "$DIR" "$TOOL" "$(PIDFILE "$STAGE")"
        ;;
    release/all)
        # Belt-and-braces for the always() cleanup: if this job dies or its container is torn down, every process
        # inside dies too and the kernel frees the leases anyway.
        for s in small large; do ci_host_lease_release "$DIR" "$TOOL" "$(PIDFILE "$s")"; done
        ;;
    *)
        echo "usage: gpu_stage_lease.sh acquire small|large | release small|large|all" >&2; exit 2 ;;
esac
exit 0
