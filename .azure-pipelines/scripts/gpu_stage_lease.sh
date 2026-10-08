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
# The lease is taken through the host's own lease tool, mounted read-only at /gpu_lease_tool, so this job queues in
# the same fair order as every local caller and is placed by the same best-fit rule. The tool is host infrastructure
# behind a seam: when it is absent the job falls back to the plain whole-server flock it always took (one stage, no
# placement), and the second stage is a no-op. The lease directory being absent means the host does not use leases.
#
# Wait policy, from the GPU run planner: IT_GPU_LEASE_WAIT seconds, and on a timeout either fail
# (IT_GPU_LEASE_ON_TIMEOUT=fail, a pull request gate) or cancel the build (cancel, a scheduled run). A scheduled run
# shares one deadline across both stages, so a busy host costs it at most that long in all.
#
# Exports to later steps: IT_GPU_TWO_STAGE (1 or 0), IT_GPU_DEVICE (a device UUID, or all), IT_GPU_SMALL_MAX_GB.
set -uo pipefail

ACTION=${1:?acquire or release}
STAGE=${2:?small, large or all}
DIR=${GPU_STAGE_LEASE_DIR:-/gpu_leases}       # overridable for tests; the pipeline mounts these two paths
TOOL=${GPU_STAGE_LEASE_TOOL:-/gpu_lease_tool}
LABEL="azure-it-${BUILD_BUILDID:-local}"
PIDFILE() { echo "/tmp/azp_gpu_${1}.pid"; }
DEADLINE_FILE=/tmp/azp_gpu_lease_deadline
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

# Wait for cpu-heavy to be free BEFORE taking the gpu key. The tool takes gpu first and only then waits on cpu-heavy
# (the fixed order that keeps two callers from deadlocking), so without this a long local CPU-only suite holding
# cpu-heavy would leave this job holding a GPU it cannot use. Racy by nature: another caller can take cpu-heavy
# between this wait and the hold, which only costs that wait again; the fixed order still rules out deadlock.
prewait_cpu_heavy() {
    GPU_LEASE_DIR="$DIR" "$TOOL" --wait-only --lease cpu-heavy --timeout "$(wait_secs)" --project "$LABEL"
    local rc=$?
    (( rc == 75 )) && on_timeout
    (( rc == 0 )) || exit "$rc"
}

acquire_fallback() {
    # The plain whole-server lease this job took before per-device placement: gpu, then cpu-heavy.
    local rc
    ci_gpu_lease_acquire "$DIR" /tmp/azp_gpu_lease.pid "$(wait_secs)" "$LABEL" gpu; rc=$?
    if (( rc == 0 )); then
        ci_gpu_lease_acquire "$DIR" /tmp/azp_cpu_heavy_lease.pid "$(wait_secs)" "$LABEL" cpu-heavy; rc=$?
    fi
    (( rc == 0 )) && return 0
    (( rc == 75 )) && on_timeout
    exit "$rc"
}

case "$ACTION/$STAGE" in
    acquire/small)
        if [[ ! -d "$DIR" ]]; then
            echo "gpu_stage_lease: ${DIR} not mounted; this host does not use GPU leases. Proceeding unleased, one stage."
            setvar IT_GPU_TWO_STAGE 0; setvar IT_GPU_DEVICE all; exit 0
        fi
        if ! tool_usable; then
            echo "##vso[task.logissue type=warning]The host lease tool is not mounted at ${TOOL}; running one whole-server stage without device placement."
            acquire_fallback
            setvar IT_GPU_TWO_STAGE 0; setvar IT_GPU_DEVICE all; exit 0
        fi
        small=$(small_max_gb)
        [[ "$small" =~ ^[0-9]+$ && "$small" -gt 0 ]] || { echo "ERROR: cannot read the devices' memory from nvidia-smi." >&2; exit 1; }
        prewait_cpu_heavy
        GPU_LEASE_DIR="$DIR" "$TOOL" --hold --pidfile "$(PIDFILE small)" --gpus 1 --min-vram "$small" --cpu-heavy \
            --timeout "$(wait_secs)" --project "$LABEL"
        rc=$?
        (( rc == 75 )) && on_timeout
        (( rc == 0 )) || exit "$rc"
        device=$(cat "$(PIDFILE small).device")
        echo "gpu_stage_lease: small stage holds ${device} (classes declaring up to ${small} GiB, no bf16)."
        setvar IT_GPU_TWO_STAGE 1; setvar IT_GPU_DEVICE "$device"; setvar IT_GPU_SMALL_MAX_GB "$small"
        ;;
    acquire/large)
        if [[ "${IT_GPU_TWO_STAGE:-0}" != 1 ]]; then
            echo "gpu_stage_lease: one-stage run; the first stage's lease already covers this."; exit 0
        fi
        prewait_cpu_heavy
        GPU_LEASE_DIR="$DIR" "$TOOL" --hold --pidfile "$(PIDFILE large)" --gpus all --cpu-heavy \
            --timeout "$(wait_secs)" --project "$LABEL"
        rc=$?
        (( rc == 75 )) && on_timeout
        (( rc == 0 )) || exit "$rc"
        echo "gpu_stage_lease: large stage holds the whole server."
        setvar IT_GPU_DEVICE all
        ;;
    release/small|release/large)
        if [[ -f "$(PIDFILE "$STAGE")" ]] && tool_usable; then
            GPU_LEASE_DIR="$DIR" "$TOOL" --release --pidfile "$(PIDFILE "$STAGE")"
        fi
        ;;
    release/all)
        # Belt-and-braces for the always() cleanup: if this job dies or its container is torn down, every process
        # inside dies too and the kernel frees the leases anyway.
        for s in small large; do
            [[ -f "$(PIDFILE "$s")" ]] && tool_usable && GPU_LEASE_DIR="$DIR" "$TOOL" --release --pidfile "$(PIDFILE "$s")"
        done
        ci_gpu_lease_release "$DIR" /tmp/azp_gpu_lease.pid gpu
        ci_gpu_lease_release "$DIR" /tmp/azp_cpu_heavy_lease.pid cpu-heavy
        ;;
    *)
        echo "usage: gpu_stage_lease.sh acquire small|large | release small|large|all" >&2; exit 2 ;;
esac
exit 0
