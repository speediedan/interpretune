#!/usr/bin/env bash
# interpretune's own part of infra_utils.sh, sourced at its end: the venv-base variable
# determine_venv_path reads, and the resource-debug helpers only this project uses.

INFRA_VENV_BASE_VAR=IT_VENV_BASE


resource_debug_enabled(){
    [[ "${IT_RESOURCE_DEBUG:-0}" == "1" ]]
}


enable_resource_debug_env(){
    export IT_RESOURCE_DEBUG=1
}


log_shell_resource_snapshot(){
    local label="$1"

    if ! resource_debug_enabled; then
        return 0
    fi

    python - <<'PY' "$label"
import os
import sys

import psutil

label = sys.argv[1]
process = psutil.Process(os.getpid())
memory = process.memory_info()
parts = [
    "context=shell",
    f"rss_gb={memory.rss / (1024**3):.2f}",
    f"vms_gb={memory.vms / (1024**3):.2f}",
]

try:
    import torch
except Exception as exc:  # pragma: no cover - defensive shell helper
    parts.append(f"torch_import_error={type(exc).__name__}")
else:
    cuda_available = torch.cuda.is_available()
    parts.append(f"cuda_available={str(cuda_available).lower()}")
    parts.append(f"cuda_device_count={torch.cuda.device_count() if cuda_available else 0}")
    if cuda_available:
        for device_idx in range(torch.cuda.device_count()):
            prefix = f"cuda_gpu{device_idx}"
            props = torch.cuda.get_device_properties(device_idx)
            parts.extend(
                [
                    f"{prefix}_total_gb={props.total_memory / (1024**3):.2f}",
                    f"{prefix}_current_allocated_gb={torch.cuda.memory_allocated(device_idx) / (1024**3):.2f}",
                    f"{prefix}_current_reserved_gb={torch.cuda.memory_reserved(device_idx) / (1024**3):.2f}",
                    f"{prefix}_peak_allocated_gb={torch.cuda.max_memory_allocated(device_idx) / (1024**3):.2f}",
                    f"{prefix}_peak_reserved_gb={torch.cuda.max_memory_reserved(device_idx) / (1024**3):.2f}",
                ]
            )

print(f"[shell_resource_debug] {label}: " + " ".join(parts))
PY
}
