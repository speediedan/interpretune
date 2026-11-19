# Sourced, not run: append one timestamped event to the GPU job's telemetry log.
#
#   telemetry_event kind=lease stage=small action=acquired device=GPU-...
#
# Each call writes one JSON object to ${IT_TELEMETRY_DIR}/events.jsonl (default /evidence/it-phases/telemetry, on the
# evidence mount the template publishes), with the UTC time and epoch seconds the event happened at. Values are
# recorded as strings; they are names, ids and exit codes, so none carries a quote or a backslash, and one that does is
# dropped rather than written as broken JSON. tests/gpu_ci/build_telemetry.py joins these events to the build's plan,
# manifests and phase reports.
#
# Telemetry never changes what the job does: without the evidence mount (a contributor's machine, a test) and without
# IT_TELEMETRY_DIR, nothing is written, and a write that fails is ignored.
telemetry_event() {
    local dir=${IT_TELEMETRY_DIR:-}
    if [[ -z "$dir" ]]; then
        [[ -d /evidence ]] || return 0
        dir=/evidence/it-phases/telemetry
    fi
    local json kv key value
    json="{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"epoch\":$(date +%s)"
    for kv in "$@"; do
        key=${kv%%=*} value=${kv#*=}
        [[ "$key$value" == *[\"\\]* ]] && continue
        json+=",\"${key}\":\"${value}\""
    done
    { mkdir -p "$dir" && echo "${json}}" >> "${dir}/events.jsonl"; } 2>/dev/null || true
    return 0
}

# The device UUIDs a stage's grant covers: the UUID itself for one device, every device for "all". GPU_LEASE_DEVICES
# ("<uuid>:<MiB>,...") stands in for nvidia-smi, as it does for the lease tool. Prints "unknown" when the devices
# cannot be enumerated (an AMD agent, where the grant is still recorded as "all").
telemetry_devices() {
    local grant=${1:-}
    if [[ -n "$grant" && "$grant" != all ]]; then
        echo "$grant"
    elif [[ -n "${GPU_LEASE_DEVICES:-}" ]]; then
        tr ',' '\n' <<< "$GPU_LEASE_DEVICES" | cut -d: -f1 | paste -sd';' -
    else
        local uuids
        uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | paste -sd';' -)
        echo "${uuids:-unknown}"
    fi
}
