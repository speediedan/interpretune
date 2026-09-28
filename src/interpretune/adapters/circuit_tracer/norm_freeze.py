r"""Hold residual-stream norm denominators at their clean values during a feature intervention.

An attribution graph treats every norm denominator as a constant (circuit-tracer detaches them), so an edge predicts
the effect of an intervention in which they do not move. A feature intervention that lets the MLPs recompute but holds
the denominators fixed measures exactly the response the graph's feature paths describe; one that lets them move adds
the rescaling the graph does not represent. For a target read far downstream of its strongest sources (a lens read at a
middle layer attributed mostly to the first few layers) that rescaling can cancel most of the predicted effect, so the
frozen measurement is the one a graph is validated against.

Freezing works on the norm's OUTPUT, so it needs no knowledge of the weight convention: with $s(x)$ the norm's
inverse scale ($1 / \operatorname{rms}(x)$ for an RMSNorm, $1 / \operatorname{std}(x)$ for a LayerNorm) and $b$ its
bias, the frozen output is $(y - b) s_{\mathrm{clean}} / s(x) + b$. The norms frozen are those that act on the residual
stream: every norm module that is a direct child of a decoder block, and the final norm. Norms inside attention (query
and key norms) act on the attention pattern, which ``freeze_attention`` governs.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn


def _unwrap(module: Any) -> Any:
    """The torch module an nnsight envoy wraps, or ``module`` itself."""
    return getattr(module, "_module", module)


def _norm_kind(module: nn.Module) -> str | None:
    """``"rms"`` or ``"layer"`` for a norm this module can freeze exactly, ``None`` for a module that is not a
    norm."""
    name = type(module).__name__
    if name.endswith("RMSNorm"):
        return "rms"
    if name.endswith("LayerNorm"):
        return "layer"
    if "norm" in name.lower():
        raise ValueError(
            f"{name} is a normalization module whose denominator this freeze cannot compute; only RMSNorm and "
            "LayerNorm classes are supported. Run without freeze_norms, or extend norm_freeze for this class."
        )
    return None


def _eps(module: nn.Module) -> float:
    for attr in ("eps", "variance_epsilon", "epsilon"):
        value = getattr(module, attr, None)
        if value is not None:
            return float(value)
    raise ValueError(f"{type(module).__name__} carries no epsilon attribute (eps, variance_epsilon or epsilon)")


def _inverse_scale(kind: str, x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    if kind == "rms":
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return torch.rsqrt(x.var(-1, keepdim=True, unbiased=False) + eps)


def residual_norm_modules(replacement_model: Any) -> list[tuple[nn.Module, str]]:
    """Every residual-stream norm of the model with its kind: the decoder blocks' own norms, then the final norm."""
    location = replacement_model.pre_logit_location
    norms: list[tuple[nn.Module, str]] = []
    for block in _unwrap(getattr(location, "layers")):
        for child in _unwrap(block).children():
            kind = _norm_kind(child)
            if kind is not None:
                norms.append((child, kind))
    final = _unwrap(getattr(location, "norm", None))
    if final is not None:
        kind = _norm_kind(final)
        if kind is not None:
            norms.append((final, kind))
    if not norms:
        raise ValueError(
            "freeze_norms found no RMSNorm or LayerNorm module on the residual stream of this model, so it cannot "
            "hold any denominator fixed; run without freeze_norms."
        )
    return norms


@contextmanager
def frozen_norm_denominators(replacement_model: Any, prompt: Any) -> Iterator[list[nn.Module]]:
    """Within the context, every residual-stream norm divides by its clean-pass scale for ``prompt``.

    The clean scales are captured on an unintervened pass over the same input, so the context applies to calls on that
    input only; a call on a different sequence length is refused by the shape check in the hook.
    """
    norms = residual_norm_modules(replacement_model)
    clean: dict[nn.Module, torch.Tensor] = {}
    handles = []

    def _capture(kind: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            clean[module] = _inverse_scale(kind, inputs[0], _eps(module)).detach()

        return hook

    try:
        for module, kind in norms:
            handles.append(module.register_forward_hook(_capture(kind)))
        replacement_model.get_activations(prompt)
    finally:
        for handle in handles:
            handle.remove()
    missing = [type(m).__name__ for m, _ in norms if m not in clean]
    if missing:
        raise RuntimeError(f"the clean pass never reached {len(missing)} residual norm(s) ({missing[:3]}...)")

    def _freeze(kind: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            reference = clean[module]
            live = _inverse_scale(kind, inputs[0], _eps(module))
            if live.shape != reference.shape:
                raise ValueError(
                    f"freeze_norms captured clean scales of shape {tuple(reference.shape)} but the intervened pass "
                    f"has {tuple(live.shape)}: the clean pass must run on the same input as the intervention"
                )
            bias = getattr(module, "bias", None)
            out = output.float()
            if bias is not None:
                out = out - bias.float()
            out = out * (reference.to(out.device) / live)
            if bias is not None:
                out = out + bias.float()
            return out.to(output.dtype)

        return hook

    handles = [module.register_forward_hook(_freeze(kind)) for module, kind in norms]
    try:
        yield [module for module, _ in norms]
    finally:
        for handle in handles:
            handle.remove()
