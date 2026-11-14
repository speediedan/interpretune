"""Analysis points for the attention-aware attribution notebook.

Each function runs inside circuit-tracer at the line its hook in ``attention_attribution_injection_config.yaml`` names,
with that frame's locals, and appends what it reads to a trace in the shared hook context. The notebook reads a trace
with ``read_trace`` and empties it with ``reset_trace`` before a call whose values it wants on their own, so a trace
never mixes two calls.
"""

from __future__ import annotations

from typing import Any

from interpretune.analysis.injection.analysis_hook_patcher import HOOK_REGISTRY
from interpretune.analysis.injection.orchestrator import analysis_log_point

GRADIENT_FLOW = "aa_gradient_flow"
NEUMANN_RUNS = "aa_neumann_runs"
HEAD_LOADING_ROWS = "aa_head_loading_rows"
CONDUCTANCE_POINTS = "aa_conductance_points"
CONDUCTANCE_RUNNING = "aa_conductance_running"
TRACE_KEYS = (GRADIENT_FLOW, NEUMANN_RUNS, HEAD_LOADING_ROWS, CONDUCTANCE_POINTS, CONDUCTANCE_RUNNING)


def reset_trace(*keys: str) -> None:
    """Empty the named traces (every trace when none is named)."""
    for key in keys or TRACE_KEYS:
        HOOK_REGISTRY.set_context(**{key: {} if key == CONDUCTANCE_RUNNING else []})


def read_trace(key: str) -> Any:
    """The trace stored under ``key``, created empty if no point has written it yet."""
    if HOOK_REGISTRY.get_context(key) is None:
        reset_trace(key)
    return HOOK_REGISTRY.get_context(key)


def ap_gradient_flow_regime(local_vars: dict[str, Any]) -> None:
    """Record which locations ``configure_gradient_flow`` is about to detach for this regime.

    The locations are read as the replacement model's location patterns, one match per layer, rather than through
    ``attention_locs``: resolving those envoys reaches nnsight's ``.source``, which works only inside a trace.
    """
    model = local_vars["self"]
    freeze_attention, freeze_norms = bool(local_vars["freeze_attention"]), bool(local_vars["freeze_norms"])
    n_layers = int(model.cfg.n_layers)
    norm_patterns = list(model._layernorm_scale_patterns)
    record = {
        "freeze_attention": freeze_attention,
        "freeze_norms": freeze_norms,
        "attention_pattern": model._attention_pattern,
        "norm_patterns": norm_patterns,
        "attention_locations": n_layers,
        "norm_scale_locations": n_layers * len(norm_patterns),
        "detached": n_layers * freeze_attention + n_layers * len(norm_patterns) * freeze_norms,
    }
    read_trace(GRADIENT_FLOW).append(record)
    analysis_log_point("Gradient flow configured for an attribution pass", record)


def ap_node_influence_init(local_vars: dict[str, Any]) -> None:
    """Start a new Neumann-series trace with the first term, ``logit_weights @ A``."""
    term = local_vars["current_influence"].detach().float().cpu().clone()
    read_trace(NEUMANN_RUNS).append([term])


def ap_node_influence_step(local_vars: dict[str, Any]) -> None:
    """Append each further term, ``current_influence @ A``, to the newest trace."""
    runs = read_trace(NEUMANN_RUNS)
    if runs:
        runs[-1].append(local_vars["current_influence"].detach().float().cpu().clone())


def ap_head_loadings_rows(local_vars: dict[str, Any]) -> None:
    """Keep one layer's per-copy edge rows: one copy per head, then the bypass, then every head live."""
    record = {
        "layer": int(local_vars["layer"]),
        "n_heads": int(local_vars["n_heads"]),
        "freeze_attention": bool(local_vars["freeze_attention"]),
        "rows": local_vars["rows"].detach().float().cpu(),
    }
    read_trace(HEAD_LOADING_ROWS).append(record)
    analysis_log_point(
        f"Head loadings rows for layer {record['layer']}",
        {"rows_shape": tuple(record["rows"].shape), "n_heads": record["n_heads"]},
    )


def ap_conductance_path_points(local_vars: dict[str, Any]) -> None:
    """Keep the target and its path derivative at one batch of path points."""
    read_trace(CONDUCTANCE_POINTS).append(
        {
            "alphas": local_vars["chunk"].detach().float().cpu(),
            "t": local_vars["t"].detach().float().cpu(),
            "dt_dalpha": local_vars["dt_dalpha"].detach().float().cpu(),
        }
    )


def ap_conductance_interval(local_vars: dict[str, Any]) -> None:
    """Keep layer ``i``'s running pattern and value credit per head after one more trapezoid interval."""
    i = int(local_vars["i"])
    running = read_trace(CONDUCTANCE_RUNNING)
    running.setdefault(i, []).append(
        (
            local_vars["qk"][i].detach().cpu().clone(),
            local_vars["ov"][i].detach().cpu().clone(),
            float(local_vars["resid"][i]),
        )
    )


AP_FUNCTIONS = {
    "ap_gradient_flow_regime": ap_gradient_flow_regime,
    "ap_node_influence_init": ap_node_influence_init,
    "ap_node_influence_step": ap_node_influence_step,
    "ap_head_loadings_rows": ap_head_loadings_rows,
    "ap_conductance_path_points": ap_conductance_path_points,
    "ap_conductance_interval": ap_conductance_interval,
}

__all__ = ["AP_FUNCTIONS", "TRACE_KEYS", "read_trace", "reset_trace"]
