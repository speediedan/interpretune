"""Notebook input helpers that several example tutorials share.

Supported notebook-support API: checking the environment a tutorial needs (``required_os_env``), locating concept
tokens in a prompt, and profiling transcoder features' input and output directions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


EnvVarSpec = tuple[str, str | Callable[[str], bool]]


def validate_env_vars(env_specs: Sequence[EnvVarSpec]) -> bool:
    """Validate environment variables based on specified modes.

    Args:
        env_specs: Sequence of (env_var_name, mode) tuples where mode can be:
            - 'present': Variable must be present in os.environ (allows empty string)
            - 'non-empty': Variable must be present and not empty
            - callable: A lambda/function that takes the env value and returns bool

    Returns:
        True if all validations pass, False otherwise
    """
    for env_var, mode in env_specs:
        value = os.environ.get(env_var)
        if mode == "present":
            if value is None:
                return False
        elif mode == "non-empty":
            if not value:
                return False
        elif callable(mode):
            if value is None or not mode(value):
                return False
        else:
            raise ValueError(f"Invalid mode for {env_var}: {mode}")
    return True


def required_os_env(env_path: str | Path | None = None, env_reqs: Sequence[EnvVarSpec] | None = None) -> bool:
    """Load environment variables from .env file and validate required env vars.

    Args:
        env_path: Optional path to .env file. If None, load_dotenv auto-discovers.
        env_reqs: Optional sequence of (env_var, mode) tuples to validate.

    Returns:
        True if loading succeeds and all validations pass (or no reqs), False otherwise.
    """
    logger = logging.getLogger(__name__)

    if load_dotenv is None:
        logger.debug("dotenv not available, skipping env loading")
    else:
        loaded = False
        if env_path is not None:
            p = Path(env_path)
            if p.exists():
                loaded = load_dotenv(str(p))
            else:
                # Fall back to auto-discover
                loaded = load_dotenv()
        else:
            loaded = load_dotenv()

        if loaded:
            logger.debug("load_dotenv found and loaded a .env file")
        else:
            logger.debug("load_dotenv did not find a .env file")

    # Validate required env vars if provided
    if env_reqs:
        return validate_env_vars(env_reqs)
    return True


@dataclass(frozen=True)
class FeatureIOProfile:
    """Input/output profile for one ``(layer, feature)`` pair from a hydrated attribution graph.

    ``input_concept_share`` is the fraction of the feature's absolute activation mass that occurs at
    concept-token positions (input side: where the feature fires); ``output_projection`` is the
    signed projection of the feature's decoder vector onto a target direction (output side: what
    amplifying the feature writes toward). Large ``|output_projection|`` with a near-zero
    ``input_concept_share`` is the input/output decoupling signature; a high share with a
    projection *against* the concept marks a suppressor-motif exemplar.

    ``jlens_signature`` is the same output side read through the J-lens instead of a single
    unembed difference: top-(token, score) pairs from the readout of the decoder vector, always in
    the folded (norm-aware) basis — an unfolded signature is not the direction the readout
    computes, so there is no flag to ask for one. ``jlens_concept_mass`` is the share of signature
    mass on the concept tokens; alongside ``input_concept_share`` it turns the decoupling
    observation into a computed comparison rather than a hand-read table. Both stay ``None`` when
    the caller supplies no lens, so existing callers are unaffected.
    """

    layer: int
    feature: int
    input_concept_share: float
    activation_mass: float
    output_projection: float
    jlens_signature: tuple[tuple[str, float], ...] | None = None
    jlens_concept_mass: float | None = None


def concept_token_positions(tokenizer: Any, token_ids: Sequence[int], concept_words: Sequence[str]) -> set[int]:
    """Return prompt positions whose decoded token matches one of ``concept_words`` (case/space-insensitive)."""
    normalized = {w.strip().lower() for w in concept_words}
    positions: set[int] = set()
    for pos, tid in enumerate(token_ids):
        token_text = tokenizer.decode([int(tid)]).strip().lower().lstrip("▁")
        if token_text in normalized:
            positions.add(pos)
    return positions


def feature_io_profiles(
    graph: Any,
    feature_pairs: Sequence[tuple[int, int]],
    target_direction: torch.Tensor,
    transcoder_set: Any,
    concept_positions: set[int],
    jlens_artifact: Any | None = None,
    unembed_info: Any | None = None,
    tokenizer: Any | None = None,
    concept_token_ids: Sequence[int] | None = None,
    signature_top_k: int = 10,
) -> list[FeatureIOProfile]:
    """Compute :class:`FeatureIOProfile` rows from a hydrated circuit-tracer graph.

    Args:
        graph: Hydrated attribution graph (``analysis_backend.hydrate_graph_from_batch(...)``);
            uses ``active_features`` rows ``(layer, pos, feature)`` and ``activation_values``.
        feature_pairs: ``(layer, feature)`` pairs to profile (order preserved).
        target_direction: Unit direction in unembed space (e.g. target-token unembed difference).
        transcoder_set: Circuit-tracer transcoder set (NNsight ``Envoy`` wrappers are unwrapped);
            decoder rows are read lazily via ``_get_decoder_vectors``.
        concept_positions: Prompt positions counted as concept-token positions
            (see :func:`concept_token_positions`).
        jlens_artifact: Resolved :class:`JLensArtifact` (its per-layer ``j_by_layer``); when
            supplied together with ``unembed_info`` and ``tokenizer``, each profile also carries
            the folded J-lens signature of its decoder vector at its own layer. A feature whose
            layer has no fitted lens gets ``None`` rather than an interpolated signature.
        unembed_info: :class:`UnembedNormInfo` from ``resolve_unembed_and_norm_scale``.
        tokenizer: Used only to decode signature token ids to strings.
        concept_token_ids: Token ids counted as concept tokens for ``jlens_concept_mass``.
        signature_top_k: Ranked signature entries kept per feature.
    """
    transcoder_set = getattr(transcoder_set, "_module", transcoder_set)
    active_rows = graph.active_features.cpu()
    active_vals = graph.activation_values.detach().float().cpu()
    target_direction = target_direction.detach().float().cpu().reshape(-1)
    concept_ids = {int(t) for t in (concept_token_ids or [])}

    profiles: list[FeatureIOProfile] = []
    for layer, feature in feature_pairs:
        mask = (active_rows[:, 0] == int(layer)) & (active_rows[:, 2] == int(feature))
        if mask.any():
            vals = active_vals[mask].abs()
            positions = active_rows[mask][:, 1].tolist()
            total = float(vals.sum())
            concept_mass = float(sum(v for v, p in zip(vals.tolist(), positions) if int(p) in concept_positions))
            share = concept_mass / total if total > 0 else 0.0
        else:
            share, total = 0.0, 0.0
        decoder_vec = transcoder_set._get_decoder_vectors(int(layer), torch.tensor([int(feature)]))[0]
        out_proj = float(decoder_vec.detach().float().cpu() @ target_direction)
        signature, concept_mass_j = _jlens_signature_for_decoder(
            decoder_vec, int(layer), jlens_artifact, unembed_info, tokenizer, concept_ids, signature_top_k
        )
        profiles.append(
            FeatureIOProfile(
                layer=int(layer),
                feature=int(feature),
                input_concept_share=share,
                activation_mass=total,
                output_projection=out_proj,
                jlens_signature=signature,
                jlens_concept_mass=concept_mass_j,
            )
        )
    return profiles


def _jlens_signature_for_decoder(
    decoder_vec: torch.Tensor,
    layer: int,
    jlens_artifact: Any | None,
    unembed_info: Any | None,
    tokenizer: Any | None,
    concept_ids: set[int],
    top_k: int,
) -> tuple[tuple[tuple[str, float], ...] | None, float | None]:
    """Folded J-lens readout of one decoder vector: ranked ``(token, score)`` pairs plus concept mass.

    The readout is always folded (norm-aware): with no resolvable norm the fold is the identity
    and the basis is the paper's, otherwise it is the readout's own direction. There is no flag
    for an unfolded signature because that would not be the direction the readout computes.
    """
    if jlens_artifact is None or unembed_info is None or tokenizer is None:
        return None, None
    j_by_layer = getattr(jlens_artifact, "j_by_layer", None) or {}
    if layer not in j_by_layer:
        return None, None

    d = decoder_vec.detach().float().cpu().reshape(-1)
    w_u = unembed_info.w_u.float().detach().cpu()
    j = torch.as_tensor(j_by_layer[layer], dtype=torch.float32)
    y = (j @ d).reshape(-1)
    if unembed_info.norm_kind == "layernorm":
        y = y - y.mean()
    scale = unembed_info.norm_scale
    if scale is not None:
        y = y * scale.float().cpu().reshape(-1)
    # Scores omit the readout's input-dependent 1/rms scalar: it is a positive factor shared by
    # every token, so rankings and mass fractions are exact without it; only absolute scale differs.
    scores = (w_u @ y).reshape(-1)
    top = torch.topk(scores, k=min(int(top_k), scores.numel()))
    pairs = tuple((tokenizer.decode([int(t)]), float(s)) for t, s in zip(top.indices.tolist(), top.values.tolist()))
    mass = sum(abs(s) for _, s in pairs)
    concept = sum(abs(s) for (_, s), tok in zip(pairs, top.indices.tolist()) if int(tok) in concept_ids)
    return pairs, (concept / mass if mass > 0 else 0.0)
