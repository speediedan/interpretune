"""Bundled Jacobian-lens READ family: readout, concept probe, and sparse inventory.

A Jacobian lens answers what an activation is disposed to make the model say:
``lens(h_l) = softmax(W_U . norm(J_l h_l))``. Interpretune could already STEER in that basis (the
``patch`` intervention mode swaps a pair's lens coordinates) and could not READ it, so every
representation-level probe built on the technique was blocked on this family.

Three ops, in dependency order of what they let you ask:

- ``jlens_read`` applies the readout and ranks vocabulary tokens.
- ``jlens_concept_probe`` measures alignment against chosen concept tokens' lens directions.
- ``jlens_sparse_inventory`` decomposes an activation over the lens dictionary, which is what makes
  "the J-space component of this vector" computable rather than merely describable.

Self-contained modulo the sanctioned op-authoring surfaces (:mod:`interpretune.analysis.optools`,
:mod:`interpretune.analysis.backends`); no backend-specific imports live here.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import BatchEncoding

from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.optools import (
    UnembedNormInfo,
    fold_norm_into_unembed_rows,
    jlens_layer_for_percentile,
    resolve_jlens,
    resolve_tokenizer,
    resolve_unembed_and_norm_scale,
)

DEFAULT_LAYER_PERCENTILE = 0.85


def _resolve_lens_layer(module: Any, analysis_batch: AnalysisBatch, kwargs: dict) -> tuple[torch.Tensor, int, Any]:
    """The ``J`` matrix and the fitted layer this call reads at, plus the artifact for provenance."""
    artifact = resolve_jlens(
        module,
        repo_id=kwargs.get("jlens_repo_id") or analysis_batch.get("jlens_repo_id") or "neuronpedia/jacobian-lens",
        model_id=kwargs.get("jlens_model_id") or analysis_batch.get("jlens_model_id"),
        path=kwargs.get("jlens_lens_path") or analysis_batch.get("jlens_lens_path"),
    )
    layer = kwargs.get("jlens_layer", analysis_batch.get("jlens_layer"))
    if layer is None:
        percentile = kwargs.get("jlens_layer_percentile", analysis_batch.get("jlens_layer_percentile"))
        layer = jlens_layer_for_percentile(
            artifact, DEFAULT_LAYER_PERCENTILE if percentile is None else float(percentile)
        )
    layer = int(layer)
    if layer not in artifact.j_by_layer:
        raise ValueError(
            f"the lens at {artifact.repo_id}:{artifact.path} was fit at layers {artifact.source_layers}, "
            f"which does not include {layer}. Interpolating between fitted layers is not the same lens, "
            "so it is refused rather than approximated."
        )
    return artifact.j_by_layer[layer].float(), layer, artifact


def _readout_device(info: UnembedNormInfo) -> torch.device:
    """Where the readout runs: the unembed's device.

    Activations arrive on CPU (they are detached out of the cache) while the unembed sits wherever the
    model does, so the two have to be reconciled somewhere. Moving the small tensors to the unembed is
    the cheap direction: activations and the ``d x d`` lens are kilobytes, while an unembed is hundreds
    of megabytes and would be copied on every call. Results come back to CPU, which is where the rest
    of an analysis batch lives.
    """
    return info.w_u.device


def _activations(analysis_batch: AnalysisBatch, cache_key: str) -> torch.Tensor:
    """The cached activations at ``cache_key`` as float, shape ``(batch, position, d_model)``."""
    cache = analysis_batch.get("cache")
    if cache is None:
        raise ValueError("the J-lens read ops require a populated activation `cache`")
    if cache_key not in cache:
        raise ValueError(f"activation cache has no key {cache_key!r}; it holds {sorted(cache)[:12]}")
    tensor = torch.as_tensor(cache[cache_key]).detach().cpu().float()
    if tensor.dim() != 3:
        raise ValueError(f"expected cached activations at {cache_key!r} to be (batch, position, d_model)")
    return tensor


def _apply_readout_norm(y: torch.Tensor, info: UnembedNormInfo, include_rms_scale: bool) -> torch.Tensor:
    """Normalize lens output exactly as the model's final norm would, per norm kind.

    ``include_rms_scale`` controls only the input-dependent divisor. It changes no ranking WITHIN a
    position, because it is a positive scalar, and it does change magnitude comparisons ACROSS
    positions. That is why it is an explicit flag: a default would be silently wrong for exactly the
    cross-position comparison someone eventually makes.
    """
    if info.norm_kind == "layernorm":
        y = y - y.mean(dim=-1, keepdim=True)
    if info.norm_scale is not None:
        y = y * info.norm_scale.float().to(y.device)
    if not include_rms_scale or info.norm_kind == "none":
        return y
    if info.norm_kind == "layernorm":
        denom = y.var(dim=-1, keepdim=True, unbiased=False).sqrt()
    else:
        denom = y.pow(2).mean(dim=-1, keepdim=True).sqrt()
    return y / denom.clamp_min(torch.finfo(y.dtype).tiny)


def _lens_readout(h: torch.Tensor, j: torch.Tensor, info: UnembedNormInfo, include_rms_scale: bool) -> torch.Tensor:
    """``W_U .

    norm(J h)`` for activations ``h`` shaped ``(..., d_model)``.
    """
    if h.shape[-1] != j.shape[0]:
        raise ValueError(f"activation width {h.shape[-1]} does not match lens d_model {j.shape[0]}")
    y = h @ j.transpose(0, 1)
    return _apply_readout_norm(y, info, include_rms_scale) @ info.w_u.float().transpose(0, 1)


def _selected_positions(analysis_batch: AnalysisBatch, kwargs: dict, n_positions: int) -> torch.Tensor:
    """Positions to read, defaulting to the last one."""
    raw = kwargs.get("jlens_positions", analysis_batch.get("jlens_positions"))
    if raw is None:
        return torch.tensor([n_positions - 1], dtype=torch.long)
    positions = torch.as_tensor(raw, dtype=torch.long).reshape(-1)
    if positions.numel() == 0:
        raise ValueError("jlens_positions was given but empty")
    if int(positions.max()) >= n_positions or int(positions.min()) < -n_positions:
        raise ValueError(f"jlens_positions {positions.tolist()} out of range for {n_positions} positions")
    return positions % n_positions


def jlens_read_impl(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Rank vocabulary tokens by the J-lens readout at the selected layer and positions."""
    j, layer, artifact = _resolve_lens_layer(module, analysis_batch, kwargs)
    info = resolve_unembed_and_norm_scale(module)
    cache_key = kwargs.get("jlens_cache_key", analysis_batch.get("jlens_cache_key")) or f"blocks.{layer}.hook_in"
    include_rms_scale = bool(kwargs.get("jlens_include_rms_scale", analysis_batch.get("jlens_include_rms_scale")))
    top_k = int(kwargs.get("jlens_top_k", analysis_batch.get("jlens_top_k") or 10))

    device = _readout_device(info)
    activations = _activations(analysis_batch, cache_key)
    positions = _selected_positions(analysis_batch, kwargs, activations.shape[1])
    selected = activations[:, positions, :].to(device)
    logits = _lens_readout(selected, j.to(device), info, include_rms_scale)
    scores, ids = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
    scores, ids = scores.detach().cpu(), ids.detach().cpu()

    tokenizer = resolve_tokenizer(module)
    strings = [[[tokenizer.decode([int(i)]) for i in row] for row in example] for example in ids]
    analysis_batch.update(
        jlens_top_token_ids=ids,
        jlens_top_token_scores=scores,
        jlens_top_token_strings=strings,
        jlens_layer=layer,
        jlens_positions=positions,
        jlens_include_rms_scale=include_rms_scale,
        jlens_provenance=artifact.provenance,
    )
    return analysis_batch


def jlens_concept_probe_impl(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Cosine of activations against concept tokens' J-lens directions.

    The direction is the readout-faithful one, ``v_c = C(W_U[c] * s) @ J``, built through the shared
    seam rather than re-derived here: the paper's "rows of ``W_U J``" shorthand drops the norm its own
    readout formula contains, and the two differ enough to change whether a probe finds anything.
    """
    j, layer, artifact = _resolve_lens_layer(module, analysis_batch, kwargs)
    info = resolve_unembed_and_norm_scale(module)
    cache_key = kwargs.get("jlens_cache_key", analysis_batch.get("jlens_cache_key")) or f"blocks.{layer}.hook_in"
    apply_norm = kwargs.get("jlens_apply_final_norm", analysis_batch.get("jlens_apply_final_norm"))
    token_ids = kwargs.get("jlens_concept_token_ids", analysis_batch.get("jlens_concept_token_ids"))
    if token_ids is None:
        raise ValueError("jlens_concept_probe requires jlens_concept_token_ids")

    device = _readout_device(info)
    rows = fold_norm_into_unembed_rows(info, token_ids, apply_norm=True if apply_norm is None else bool(apply_norm))
    directions = (rows @ j.to(device)).detach().cpu()  # (n_concepts, d_model), in the residual basis
    activations = _activations(analysis_batch, cache_key)
    positions = _selected_positions(analysis_batch, kwargs, activations.shape[1])
    selected = activations[:, positions, :]

    normed_acts = selected / selected.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(selected.dtype).tiny)
    normed_dirs = directions / directions.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(directions.dtype).tiny)
    analysis_batch.update(
        jlens_concept_cosine=normed_acts @ normed_dirs.transpose(0, 1),
        jlens_concept_token_ids=torch.as_tensor(token_ids, dtype=torch.long).reshape(-1),
        jlens_layer=layer,
        jlens_positions=positions,
        jlens_provenance=artifact.provenance,
    )
    return analysis_batch


def _gradient_pursuit(h: torch.Tensor, atom_of, correlate, k: int, iters: int = 200) -> tuple[list[int], torch.Tensor]:
    """Greedy nonnegative sparse pursuit; returns the selected atom indices and their coefficients.

    Atoms are materialized only when selected, because the dictionary is the whole vocabulary and
    correlating against it is a single readout rather than a matrix the size of ``vocab x d_model``.
    """
    residual, chosen, coefficients = h.detach().clone(), [], torch.zeros(0)
    tolerance = 1e-6 * h.norm().clamp_min(torch.finfo(h.dtype).tiny)
    for _ in range(k):
        if residual.norm() <= tolerance:
            break  # the selected atoms already explain the vector; further picks would fit only noise
        scores = correlate(residual)
        scores[torch.tensor(chosen, dtype=torch.long)] = -torch.inf if chosen else scores[0] * 0 - torch.inf
        best = int(torch.argmax(scores))
        if not torch.isfinite(scores[best]) or scores[best] <= 0:
            break  # no atom is positively correlated with what remains; adding one would only fit noise
        chosen.append(best)
        atoms = torch.stack([atom_of(c) for c in chosen])  # (n_chosen, d_model)
        coefficients = _nnls(atoms, h, iters)
        residual = h - coefficients @ atoms
    return chosen, coefficients


def _nnls(atoms: torch.Tensor, h: torch.Tensor, iters: int) -> torch.Tensor:
    """Nonnegative least squares of ``h`` on ``atoms``, exact when the solution is interior.

    The unconstrained solution is tried first because it is the answer whenever every coefficient
    comes out nonnegative, which is the common case and is exact rather than iterative. Projected
    gradient handles the rest, stepping at ``1/L`` from the true Lipschitz constant: a step derived
    from the Frobenius norm is a valid bound but a loose one, and it leaves a visible residual on
    inputs whose exact decomposition the selected atoms can represent.
    """
    solution = torch.linalg.lstsq(atoms.transpose(0, 1), h.unsqueeze(-1)).solution.squeeze(-1)
    if bool((solution >= 0).all()):
        return solution
    gram = atoms @ atoms.transpose(0, 1)
    lipschitz = torch.linalg.matrix_norm(gram, ord=2).clamp_min(torch.finfo(atoms.dtype).tiny)
    coefficients = solution.clamp_min(0.0)
    target = atoms @ h
    for _ in range(iters):
        coefficients = (coefficients - (gram @ coefficients - target) / lipschitz).clamp_min(0.0)
    return coefficients


def jlens_sparse_inventory_impl(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Sparse nonnegative decomposition of an activation over the J-lens vocabulary dictionary.

    Reports the reconstruction residual alongside the coefficients, deliberately. A decomposition quoted without its
    residual cannot distinguish an account that captured the vector from one that captured a quarter of it, and "the
    J-space component of this vector" is exactly a claim about that ratio.
    """
    j, layer, artifact = _resolve_lens_layer(module, analysis_batch, kwargs)
    info = resolve_unembed_and_norm_scale(module)
    cache_key = kwargs.get("jlens_cache_key", analysis_batch.get("jlens_cache_key")) or f"blocks.{layer}.hook_in"
    k = int(kwargs.get("jlens_inventory_k", analysis_batch.get("jlens_inventory_k") or 25))

    activations = _activations(analysis_batch, cache_key)
    positions = _selected_positions(analysis_batch, kwargs, activations.shape[1])
    device = _readout_device(info)
    j = j.to(device)
    w_u = info.w_u.float().detach()

    def atom_of(token_id: int) -> torch.Tensor:
        return (fold_norm_into_unembed_rows(info, [token_id])[0] @ j).detach().cpu()

    ids_out, coefficients_out, residual_out = [], [], []
    for example in range(activations.shape[0]):
        for position in positions.tolist():
            h = activations[example, position]

            # correlation with every atom is `(W_U * s) @ J @ r`, which is the folded readout of r
            def correlate(r: torch.Tensor, _h=h) -> torch.Tensor:
                y = r.to(device) @ j.transpose(0, 1)
                if info.norm_kind == "layernorm":
                    y = y - y.mean(dim=-1, keepdim=True)
                if info.norm_scale is not None:
                    y = y * info.norm_scale.float().to(device)
                return (w_u @ y).detach().cpu()

            chosen, coefficients = _gradient_pursuit(h, atom_of, correlate, k)
            reconstruction = coefficients @ torch.stack([atom_of(c) for c in chosen]) if chosen else torch.zeros_like(h)
            ids_out.append(chosen)
            coefficients_out.append(coefficients.tolist())
            residual_out.append(float((h - reconstruction).norm() / h.norm().clamp_min(1e-12)))

    analysis_batch.update(
        jlens_inventory_token_ids=ids_out,
        jlens_inventory_coefficients=coefficients_out,
        jlens_inventory_residual_ratio=torch.tensor(residual_out),
        jlens_layer=layer,
        jlens_positions=positions,
        jlens_provenance=artifact.provenance,
    )
    return analysis_batch
