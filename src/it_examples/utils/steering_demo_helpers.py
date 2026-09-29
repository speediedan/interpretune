"""Shared maths for the concept-steering demos' embed-basis, J-lens validation and readout cells.

The two steering demos (public gemma-2-2b and local-Neuronpedia gemma-3-1b-it) run the same sections 4a, 4c and 4d.
Keeping the constructions here rather than in each notebook means the demos cannot drift apart, and the logic is
importable by tests without a kernel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import torch

import interpretune as it
from interpretune.analysis.backends import require_analysis_backend
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.bundled.jlens.jlens_ops import jlens_readout_logits
from interpretune.analysis.optools import UnembedNormInfo
from interpretune.config import AnalysisCfg, init_analysis_cfgs


def embed_pole_pair(module: Any, tokenizer: Any, group_a: Sequence[str], group_b: Sequence[str]) -> torch.Tensor:
    """Stack the per-group mean token embeddings of two concept groups into a ``(2, d_model)`` pole pair.

    Every token each word tokenizes to contributes to its group's mean, so a multi-token word is not silently reduced to
    its first piece. A group that tokenizes to nothing is refused by name rather than yielding a zero pole, which would
    make the pair rank-deficient and every coordinate read from it meaningless.
    """
    embed = require_analysis_backend(module).get_embedding_weight(module).detach().float().cpu()

    def _pole(words: Sequence[str], name: str) -> torch.Tensor:
        ids = [t for w in words for t in tokenizer(w, add_special_tokens=False)["input_ids"]]
        if not ids:
            raise ValueError(f"concept group {name} ({list(words)!r}) tokenized to no ids")
        return embed[torch.tensor(ids)].mean(dim=0)

    return torch.stack([_pole(group_a, "a"), _pole(group_b, "b")])


def outside_span_share(direction: torch.Tensor, poles: torch.Tensor) -> float:
    """Norm of ``direction``'s component outside ``span(poles)``, as a fraction of a unit direction.

    Uses the pseudoinverse projector: the pole rows are not orthonormal by construction, so projecting onto each
    row separately would double-count their overlap. ``direction`` is expected to be unit norm (concept
    directions are), which makes the returned norm a share.
    """
    poles = poles.float().cpu()
    projector = torch.linalg.pinv(poles.double()).float() @ poles
    residual = (torch.eye(poles.shape[1]) - projector) @ direction.float().cpu().reshape(-1)
    return float(torch.linalg.vector_norm(residual))


class PoleSwapPrediction(NamedTuple):
    """A ``patch``-mode pole swap's clean coordinates and its first-order effect on the target gap."""

    coordinates: tuple[float, float]
    predicted_gap_delta: float


def pole_swap_prediction(
    activation: torch.Tensor, poles: torch.Tensor, unembed_a: torch.Tensor, unembed_b: torch.Tensor
) -> PoleSwapPrediction:
    """Predict what exchanging the two pole coordinates of ``activation`` does to the ``a`` minus ``b`` logit gap.

    A swap maps ``c = V^+ x`` to its reverse, a displacement of ``(c_b - c_a)(v_a - v_b)``, so under a linear
    readout the gap moves by ``(c_b - c_a) (v_a - v_b) . (u_a - u_b)``. Its SIGN is therefore set by which pole the
    clean activation already leans toward: a swap pushes toward ``a`` only when the clean state sits nearer ``b``.
    Exact where the readout is linear in ``activation``, as it is at the unembed's input.
    """
    x = activation.detach().double().cpu().reshape(-1)
    v = poles.detach().double().cpu()
    c = x @ torch.linalg.pinv(v)
    gap_direction = unembed_a.detach().double().cpu() - unembed_b.detach().double().cpu()
    predicted = (c[1] - c[0]) * ((v[0] - v[1]) @ gap_direction)
    return PoleSwapPrediction(coordinates=(float(c[0]), float(c[1])), predicted_gap_delta=float(predicted))


def final_norm_output(module: Any, batch: Any) -> torch.Tensor:
    """The final norm's output at the last position (the unembed's input), from one forward pass, on CPU."""
    hf = getattr(module.model, "_model", None) or getattr(module.model, "_module", None)
    if hf is not None:
        target = hf.model.norm

        def forward() -> Any:
            return hf(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
    else:
        target = module.model.ln_final

        def forward() -> Any:
            return module.model(batch["input"])

    cache: dict[str, torch.Tensor] = {}

    def _capture(_mod: Any, _inputs: Any, output: Any) -> None:
        cache["x"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = target.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            forward()
    finally:
        handle.remove()
    return cache["x"][0, -1].float().cpu()


class SiteGradient(NamedTuple):
    """The clean activation at a residual site and the target gap's gradient there, both on CPU."""

    activation: torch.Tensor
    gradient: torch.Tensor
    last_pos: int


def capture_site_gap_gradient(module: Any, batch: Any, layer: int, target_a_id: int, target_b_id: int) -> SiteGradient:
    """One forward and backward pass giving ``h`` at ``blocks.<layer>.hook_resid_post`` and ``d(gap)/dh``.

    The gap is the last-position logit of ``target_a_id`` minus that of ``target_b_id``. The backend's gradient
    seam saves only SAE-spliced sub-hooks, so a bare residual site is read through a forward hook on the module
    that produces it: the HF decoder layer under an nnsight model, the TransformerLens hook point otherwise. The
    captured activation is made a graph leaf so autograd reaches it whatever the parameters' grad state.
    """
    # nnsight wraps the HF module (as `_model`, `_module` on older releases); TL-shaped models carry `blocks`.
    hf = getattr(module.model, "_model", None) or getattr(module.model, "_module", None)
    forward: Callable[[], torch.Tensor]
    if hf is not None:
        target = hf.model.layers[layer]
        ids = batch["input_ids"]

        def forward() -> torch.Tensor:
            return hf(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")).logits
    else:
        target = module.model.blocks[layer].hook_resid_post
        ids = batch["input"]

        def forward() -> torch.Tensor:
            return module.model(batch["input"])

    cache: dict[str, torch.Tensor] = {}

    def _capture(_mod: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        leaf = hidden.detach().requires_grad_(True)
        cache["h"] = leaf
        return (leaf,) + tuple(output[1:]) if isinstance(output, tuple) else leaf

    handle = target.register_forward_hook(_capture)
    try:
        with torch.enable_grad():
            logits = forward()
            metric = logits[0, -1, target_a_id] - logits[0, -1, target_b_id]
            (grad,) = torch.autograd.grad(metric, cache["h"])
    finally:
        handle.remove()
    return SiteGradient(
        activation=cache["h"].detach().float().cpu(),
        gradient=grad.detach().float().cpu(),
        last_pos=int(ids.shape[1]) - 1,
    )


def gap_after_add(
    module: Any,
    prompt: str,
    batch: Any,
    site: str,
    tensor: torch.Tensor,
    scale: float,
    target_a_id: int,
    target_b_id: int,
) -> float:
    """The target gap after an ``add``-mode intervention of ``scale * tensor`` at ``site``."""
    module.analysis_cfg = AnalysisCfg(target_op=it.model_fwd_intervention, ignore_manual=True, save_tokens=False)
    init_analysis_cfgs(module, [module.analysis_cfg])
    probe = AnalysisBatch(
        prompts=[prompt],
        logit_target_ids=torch.tensor([target_a_id], dtype=torch.long),
        concept_group_a_token_ids=[target_a_id],
        concept_group_b_token_ids=[target_b_id],
        intervention_hook_pattern=site,
        intervention_mode="add",
        intervention_tensor=tensor,
        intervention_scale_factor=scale,
    )
    out = it.model_fwd_intervention(module, probe, batch, 0)
    logits = out.post_intervention_logits.float().cpu().reshape(-1)
    return float(logits[target_a_id] - logits[target_b_id])


def central_difference_gap_slopes(
    module: Any,
    prompt: str,
    batch: Any,
    site: str,
    directions: torch.Tensor,
    target_a_id: int,
    target_b_id: int,
    eps: float = 1.0,
) -> list[float]:
    """Central-difference slope of the target gap along each row of ``directions``, through ``add`` probes.

    An independent check on the gradient readouts ``w = V^T g``: the two agree to the extent the gap is locally
    linear at the site, and they share no code path (this one runs the intervention machinery, the readout reads
    autograd).
    """

    def _gap(row: torch.Tensor, scale: float) -> float:
        return gap_after_add(module, prompt, batch, site, row, scale, target_a_id, target_b_id)

    return [(_gap(row, eps) - _gap(row, -eps)) / (2 * eps) for row in directions]


def jlens_readout_top_tokens(
    activation: torch.Tensor, lens: torch.Tensor, info: UnembedNormInfo, tokenizer: Any, k: int = 5
) -> list[tuple[str, float]]:
    """Top-``k`` decoded tokens and scores of the J-lens readout of one activation vector.

    The same readout the ``jlens_read`` op computes, with the input-dependent RMS divisor off: within one
    position it is a positive scalar and cannot change rankings.
    """
    logits = jlens_readout_logits(
        activation.detach().float().cpu(), lens.detach().float().cpu(), _cpu_info(info), include_rms_scale=False
    )
    scores, ids = torch.topk(logits, k=k)
    return [(tokenizer.decode([int(i)]), float(s)) for s, i in zip(scores, ids)]


def _cpu_info(info: UnembedNormInfo) -> UnembedNormInfo:
    return info._replace(
        **{
            name: value.detach().cpu()
            for name, value in info._asdict().items()
            if isinstance(value, torch.Tensor) and value.device.type != "cpu"
        }
    )


class SteeringScalePoint(NamedTuple):
    """One steering arm at one scale: the two target tokens' logits and probabilities before and after."""

    arm: str
    scale: float
    pre_logits: tuple[float, float]
    post_logits: tuple[float, float]
    pre_probs: tuple[float, float]
    post_probs: tuple[float, float]

    @property
    def pre_gap(self) -> float:
        return self.pre_logits[0] - self.pre_logits[1]

    @property
    def post_gap(self) -> float:
        return self.post_logits[0] - self.post_logits[1]


def steering_scale_point(
    arm: str, scale: float, pre_logits: torch.Tensor, post_logits: torch.Tensor, target_a_id: int, target_b_id: int
) -> SteeringScalePoint:
    """Summarize one arm at one scale from its full last-position logits (probabilities over the whole
    vocabulary)."""
    pre = pre_logits.detach().float().cpu().reshape(-1)
    post = post_logits.detach().float().cpu().reshape(-1)
    pre_p, post_p = torch.softmax(pre, dim=-1), torch.softmax(post, dim=-1)
    return SteeringScalePoint(
        arm=arm,
        scale=float(scale),
        pre_logits=(float(pre[target_a_id]), float(pre[target_b_id])),
        post_logits=(float(post[target_a_id]), float(post[target_b_id])),
        pre_probs=(float(pre_p[target_a_id]), float(pre_p[target_b_id])),
        post_probs=(float(post_p[target_a_id]), float(post_p[target_b_id])),
    )
