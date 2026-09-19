"""First-order intervention-validation instrument (#539).

Predicts the metric change an intervention should produce, from the edit it actually applied::

    Δm ≈ ∇_h mᵀ Δh

where ``m`` is a caller-supplied scalar metric of the output logits, ``h`` the activation at the
intervened site, and ``Δh`` the edit the intervention really wrote (not the tensor it was
configured with: those differ for every mode except ``add``, and the difference is what is being
validated).

The instrument is eager and backend-agnostic by construction: it drives the module's own forward
with plain forward hooks, and it computes the applied edit with :func:`apply_intervention`, the
same pure function every backend calls. No backend interface was extended.

Run once per basis and compare the runs: the report names its basis, because a result that does
not name its basis cannot be compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from interpretune.analysis.backends.interventions import InterventionSpec, apply_intervention


@dataclass(frozen=True)
class FirstOrderReport:
    """Predicted against measured metric change for one basis run."""

    basis: str
    predicted: float
    measured: float
    residual: float
    mode: str
    scale_factor: float


def first_order_check(
    model: torch.nn.Module,
    target_module: torch.nn.Module,
    input_ids: torch.Tensor,
    spec: InterventionSpec,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    basis: str,
    last_pos: int,
) -> FirstOrderReport:
    """Validate one intervention run to first order (issue #539).

    Args:
        model: The eager module to drive (eval mode; gradients enabled locally).
        target_module: The submodule whose output activation is intervened on.
        input_ids: Integer token ids, ``(batch, seq)``.
        spec: The intervention specification, applied verbatim.
        metric_fn: Caller-supplied scalar metric over output logits. Must return a
            zero-dimensional tensor and use only differentiable ops.
        basis: Name of the basis this run belongs to (``embed`` / ``store`` /
            ``jlens_paper`` / ``jlens_norm_aware``). Recorded, never interpreted.
        last_pos: Index of the final real token (needed for ``last_token`` scope).

    Returns:
        A :class:`FirstOrderReport` with the predicted change (``gradᵀ Δh``), the measured
        change, and their residual. A large residual is a finding (the edit left the linear
        regime), not an error.
    """
    was_training = model.training
    model.eval()
    try:
        return _first_order_check_evaled(model, target_module, input_ids, spec, metric_fn, basis, last_pos)
    finally:
        model.train(was_training)


def _first_order_check_evaled(
    model: torch.nn.Module,
    target_module: torch.nn.Module,
    input_ids: torch.Tensor,
    spec: InterventionSpec,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    basis: str,
    last_pos: int,
) -> FirstOrderReport:
    captured: dict[str, torch.Tensor] = {}

    def capture(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h"] = hidden

    handle = target_module.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            logits = model(input_ids).logits
            h_pre = captured["h"]
            metric_pre = metric_fn(logits)
            (grad,) = torch.autograd.grad(metric_pre, h_pre)
    finally:
        handle.remove()
    pre_value = float(metric_pre.detach())

    h_pre_detached = h_pre.detach()
    h_post = apply_intervention(h_pre_detached.clone(), spec, last_pos=last_pos)
    delta_h = (h_post - h_pre_detached).to(dtype=torch.float32)
    predicted = float((grad.detach().to(dtype=torch.float32) * delta_h).sum())

    def intervene(_module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        edited = apply_intervention(hidden.clone(), spec, last_pos=last_pos)
        if isinstance(output, tuple):
            return (edited,) + tuple(output[1:])
        return edited

    hook = target_module.register_forward_hook(intervene)
    try:
        with torch.no_grad():
            post_logits = model(input_ids).logits
    finally:
        hook.remove()

    with torch.no_grad():
        measured = float(metric_fn(post_logits) - pre_value)

    return FirstOrderReport(
        basis=basis,
        predicted=predicted,
        measured=measured,
        residual=measured - predicted,
        mode=str(spec.mode),
        scale_factor=float(spec.scale_factor),
    )
