"""The Interpretune op-authoring toolkit.

This module is the sanctioned shared surface for analysis-op implementations. Bundled, local, and
hub op collections may all import from it: interpretune is by definition installed wherever an op
runs, so depending on this module is a declared, supported contract rather than a reach into package
internals. Op YAML ``importable_params`` entries may also reference callables in this namespace.

Everything exported here is expected to remain stable for op authors (within the project's pre-MVP
caveats); anything not exported is internal. Backend-specific behavior stays behind the
:mod:`interpretune.analysis.backends` capability seam, which op implementations may also use.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Literal, NamedTuple

import torch
from jaxtyping import Float

from interpretune.analysis.backends import get_analysis_backend, get_model_backend
from interpretune.analysis.inputs import _resolve_attr_path
from interpretune.protocol import DefaultAnalysisBatchProtocol

FEATURE_SCORE_SOURCE_ALIASES: dict[str, str] = {
    "influence": "node_influence_scores",
    "absolute_influence": "node_influence_scores",
    "signed_influence": "node_signed_influence_scores",
    "gradient": "node_logit_diff_gradient_scores",
    "gradients": "node_logit_diff_gradient_scores",
    "logit_diff_gradient": "node_logit_diff_gradient_scores",
    "target_logit_diff_gradient": "node_logit_diff_gradient_scores",
}


def resolve_feature_score_source(score_source: str | None) -> str | None:
    """Normalize user-facing score-source aliases to analysis-batch field names."""
    if score_source is None:
        return None
    return FEATURE_SCORE_SOURCE_ALIASES.get(score_source, score_source)


# ---------------------------------------------------------------------------
# Tensor / logits utilities
# ---------------------------------------------------------------------------


def extract_logits(output: Any) -> torch.Tensor:
    """Extract a logits tensor from framework-specific model outputs."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    raise TypeError(f"Cannot extract logits from model output of type {type(output).__name__}")


def last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return the final-token logits as a detached CPU tensor."""
    if logits.dim() == 1:
        return logits.detach().cpu()
    if logits.dim() == 2:
        return logits[-1].detach().cpu()
    if logits.dim() >= 3:
        return logits[0, -1].detach().cpu()
    raise ValueError(f"Unsupported logits rank for feature intervention output: {logits.dim()}")


def mean_target_logit_delta(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    target_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Return the mean delta over requested target ids, or over all logits if none are given."""
    if target_ids is not None and torch.numel(target_ids) > 0:
        target_ids = target_ids.to(dtype=torch.long).reshape(-1)
        vocab_size = pre_logits.size(0)
        oob = target_ids >= vocab_size
        if oob.any():
            raise ValueError(
                f"logit_target_ids contain out-of-bounds indices (>= vocab_size {vocab_size}): "
                f"{target_ids[oob].tolist()}. Virtual IDs from concept-direction targets must be "
                "resolved before intervention."
            )
        return (post_logits.index_select(0, target_ids) - pre_logits.index_select(0, target_ids)).mean()
    return (post_logits - pre_logits).mean()


def stack_column_tensors(values: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Normalize dataset or run-input column values into a tensor."""

    def _combine_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
        if tensors[0].ndim > 1:
            try:
                return torch.cat(tensors, dim=0)
            except RuntimeError:
                return torch.stack(tensors)
        return torch.stack(tensors)

    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype) if dtype is not None else values
    if isinstance(values, list | tuple):
        values = list(values)
        if not values:
            target_dtype = dtype if dtype is not None else torch.float32
            return torch.empty((0,), dtype=target_dtype)
        if all(isinstance(value, torch.Tensor) for value in values):
            tensors = [value.detach().cpu() for value in values]
            stacked = _combine_tensors(tensors)
            return stacked.to(dtype=dtype) if dtype is not None else stacked
        tensor_values = []
        for value in values:
            tensor_value = torch.as_tensor(value)
            tensor_values.append(tensor_value.detach().cpu())
        stacked = _combine_tensors(tensor_values)
        return stacked.to(dtype=dtype) if dtype is not None else stacked
    return torch.as_tensor(values, dtype=dtype)


def weighted_mean(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute a stable weighted mean over state rows."""
    weights = weights.to(device=states.device, dtype=states.dtype).reshape(-1)
    weight_sum = weights.sum().clamp_min(1e-12)
    return (states * weights.unsqueeze(-1)).sum(dim=0) / weight_sum


# ---------------------------------------------------------------------------
# Classification-logit utilities (op ``importable_params`` targets)
# ---------------------------------------------------------------------------


def boolean_logits_to_avg_logit_diff(
    logits: Float[torch.Tensor, "batch seq 2"],  # type: ignore
    target_indices: torch.Tensor,
    reduction: Literal["mean", "sum"] | None = None,
) -> torch.Tensor:
    """Returns the avg logit diff on a set of prompts, with fixed s2 pos and stuff."""
    incorrect_indices = 1 - target_indices
    correct_logits = torch.gather(logits, 2, torch.reshape(target_indices, (-1, 1, 1))).squeeze()
    incorrect_logits = torch.gather(logits, 2, torch.reshape(incorrect_indices, (-1, 1, 1))).squeeze()
    logit_diff = correct_logits - incorrect_logits
    if reduction is not None:
        logit_diff = logit_diff.mean() if reduction == "mean" else logit_diff.sum()
    return logit_diff


def get_loss_preds_diffs(
    module: torch.nn.Module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    answer_logits: torch.Tensor,
    logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implementation for computing loss, predictions, and logit differences.

    Args:
        module: The module containing loss_fn and standardize_logits methods
        analysis_batch: The analysis batch containing labels and orig_labels
        answer_logits: The logits to analyze
        logit_diff_fn: Function to compute logit differences

    Returns:
        Tuple of (loss, logit_diffs, preds, answer_logits)
    """
    loss = module.loss_fn(answer_logits, analysis_batch.label_ids)  # type: ignore[attr-defined]
    answer_logits = module.standardize_logits(answer_logits)  # type: ignore[attr-defined]
    per_example_answers, _ = torch.max(answer_logits, dim=-2)
    preds = torch.argmax(per_example_answers, axis=-1)  # type: ignore[call-arg]
    logit_diffs = logit_diff_fn(answer_logits, target_indices=analysis_batch.orig_labels)
    return loss, logit_diffs, preds, answer_logits


# ---------------------------------------------------------------------------
# Model-access resolution (backend-agnostic, per the composition guide)
# ---------------------------------------------------------------------------


def require_model_backend(module: Any) -> Any:
    """Return a model backend from either ``_model_backend`` or ``model_backend``."""
    backend = get_model_backend(module)
    if backend is None:
        raise ValueError("Target module must expose a model backend for this operation")
    return backend


def require_backend_capability(backend: Any, capability: Any, op_name: str) -> None:
    """Raise a uniform, actionable error when ``backend`` does not claim ``capability``.

    Optional ``ModelBackend`` method groups are capability-gated (``SupportsLatentModels`` et al.), so
    ops call this before an optional method rather than letting a partial backend fail as an
    ``AttributeError`` deep inside execution.
    """
    # Compare by VALUE, not enum identity: this repo's test infrastructure can load the capabilities
    # module twice (the documented importlib double-loading class), leaving value-equal enum members
    # that fail an identity-based ``in``. A backend that REPORTS the capability must pass the gate.
    supported = {getattr(c, "value", c) for c in backend.capabilities}
    if getattr(capability, "value", capability) not in supported:
        raise ValueError(
            f"{op_name} requires a model backend with {capability.name}; "
            f"{type(backend).__name__} reports {sorted(c.name for c in backend.capabilities)}"
        )


def resolve_tokenizer(module: Any) -> Any:
    """Resolve a tokenizer from a generic module or its analysis backend."""
    analysis_backend = get_analysis_backend(module)
    if analysis_backend is not None:
        try:
            return analysis_backend.get_tokenizer(module)
        except (AttributeError, ValueError):
            pass

    for path in (
        ("replacement_model", "tokenizer"),
        ("model", "tokenizer"),
        ("datamodule", "tokenizer"),
        ("tokenizer",),
    ):
        value = _resolve_attr_path(module, *path)
        if value is not None:
            return value

    raise ValueError("A tokenizer is required for this analysis operation")


# HF RMSNorm families that apply ``(1 + weight)`` rather than ``weight``. Membership is EXACT, not a
# prefix test, because the convention is not a property of the name: `gemma3` carries the offset and
# `gemma3n` does not, so any prefix wide enough to catch the first also catches the second. An earlier
# `startswith("gemma")` rule was correct for every family that existed when it was written and silently
# wrong for `gemma3n` and the whole `gemma4` line, producing a plausible direction rather than an error.
_RMSNORM_OFFSET_MODEL_TYPES = frozenset({"gemma", "gemma2", "gemma3", "gemma3_text"})
# Families in the same namespace that are KNOWN to apply weight directly. Listed rather than left to the
# default so that an unrecognized `gemma*` type is distinguishable from a checked one.
_RMSNORM_NO_OFFSET_MODEL_TYPES = frozenset(
    {
        "gemma3n",
        "gemma3n_text",
        "gemma3n_audio",
        "gemma3n_vision",
        "gemma4",
        "gemma4_text",
        "gemma4_audio",
        "gemma4_vision",
        "gemma4_assistant",
        "gemma4_unified",
        "gemma4_unified_text",
        "gemma4_unified_audio",
        "gemma4_unified_vision",
        "gemma4_unified_assistant",
    }
)


def _rmsnorm_scale(weight: torch.Tensor, model_type: str) -> torch.Tensor:
    """The elementwise scale an RMSNorm APPLIES, given its stored weight and the model's family.

    Most families apply ``weight``; the gemma line splits, and the split does not follow the name. An
    unrecognized family in that namespace warns rather than guessing, because both guesses are wrong for
    some member of it and neither failure is visible in the output: the direction stays plausible and
    only the answer changes.
    """
    if model_type in _RMSNORM_OFFSET_MODEL_TYPES:
        return 1.0 + weight
    if model_type not in _RMSNORM_NO_OFFSET_MODEL_TYPES and model_type.startswith("gemma"):
        from interpretune.utils.logging import rank_zero_warn

        rank_zero_warn(
            f"unrecognized gemma-family model_type {model_type!r}: the gemma line splits on whether its "
            "RMSNorm applies `(1 + weight)` or `weight`, and this one is in neither list, so `weight` is "
            "assumed. If that is wrong every readout-faithful direction built here is silently off; check "
            "the family's `RMSNorm.forward` and add it to the right set in `interpretune.analysis.optools`."
        )
    return weight


class UnembedNormInfo(NamedTuple):
    """Unembed matrix plus the final norm's elementwise scale, in readout orientation.

    Attributes:
        w_u: The unembed/LM-head matrix as ``(vocab, d_model)`` regardless of backend (TL's
            ``W_U`` is stored transposed and is normalized here).
        norm_scale: The final norm's effective elementwise scale in the convention the model
            APPLIES it -- ``(1 + weight)`` for HF gemma RMSNorms, ``weight`` for other HF RMSNorms
            and for TL models (TL's gemma conversion folds the ``+1`` at load), ``weight`` for
            LayerNorms. ``None`` when no final-norm weight is resolvable.
        norm_kind: ``"rmsnorm"``, ``"layernorm"``, or ``"none"`` -- callers constructing readout
            DIRECTIONS need this because LayerNorm additionally centers, so its readout-faithful
            direction is ``C(W_U[c] * scale)`` with the centering projector ``C = I - 11^T/d``,
            while RMSNorm's is ``W_U[c] * scale`` unchanged. Unlike a uniform rescaling, which
            cancels in patch mode because scaling ``V`` scales its pseudoinverse inversely, centering
            removes an ADDITIVE uniform component, so it moves the direction and with it the plane a
            swap happens in. :func:`fold_norm_into_unembed_rows` applies the right one per kind.
    """

    w_u: torch.Tensor
    norm_scale: torch.Tensor | None
    norm_kind: str


def resolve_unembed_and_norm_scale(module: Any) -> UnembedNormInfo:
    """Resolve the unembed matrix and final-norm scale a READOUT-faithful direction needs.

    The J-lens readout is ``softmax(W_U . norm(J h))``: pushing the norm's elementwise scale
    through the dot product gives token directions ``(W_U[c] * scale) @ J``, and dropping the scale
    is not cosmetic, and it is not uniformly beneficial either: measured on gemma-3-1b-it unfolded
    vectors fail to steer at all where folded ones flip the answer at scale 1.0, while on gemma-2-2b
    the unfolded vector is the stronger of the two. Folding aims the direction at the coordinates the
    norm amplifies, which helps exactly when the model stores the task contrast there, so it is a
    per-model choice rather than a default to hard-code. This is the single sanctioned home for the
    per-family conventions.

    Resolution order mirrors :func:`resolve_embedding_weight`: HF-style models first
    (``lm_head.weight`` or GPT-NeoX's ``embed_out.weight``, shape ``(vocab, d)``), then
    TransformerLens (``W_U``, stored ``(d, vocab)`` and transposed here). Raises when no unembed
    surface is found, because silently returning an embedding matrix would only be correct for
    tied-weight models and wrong without warning everywhere else.
    """
    for attr_name in ("model", "replacement_model"):
        model = getattr(module, attr_name, None)
        if model is None:
            continue
        for head_attr in ("lm_head", "embed_out"):
            head = getattr(model, head_attr, None)
            if head is not None and isinstance(getattr(head, "weight", None), torch.Tensor):
                w_u = head.weight
                # Selected by `is not None` rather than by truthiness, and that is load-bearing rather
                # than stylistic: a module is not a safe thing to truth-test. `nn.Sequential` and
                # `nn.ModuleList` define `__len__`, so an empty one is FALSY and an `or` chain would
                # silently skip a real submodule; and a wrapper whose `__len__` delegates to a module
                # that has none (nnsight's `Envoy` over an HF model) makes the truth-test itself raise
                # `TypeError: object of type '...' has no len()`. Both failures land far from here.
                inner = next(
                    (
                        candidate
                        for attr in ("model", "transformer", "gpt_neox")
                        if (candidate := getattr(model, attr, None)) is not None
                    ),
                    model,
                )
                for norm_attr in ("norm", "ln_f", "final_layer_norm"):
                    norm = getattr(inner, norm_attr, None)
                    weight = getattr(norm, "weight", None) if norm is not None else None
                    if isinstance(weight, torch.Tensor):
                        kind = "layernorm" if "LayerNorm" in type(norm).__name__ else "rmsnorm"
                        model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
                        scale = weight if kind != "rmsnorm" else _rmsnorm_scale(weight, model_type)
                        return UnembedNormInfo(w_u=w_u, norm_scale=scale, norm_kind=kind)
                return UnembedNormInfo(w_u=w_u, norm_scale=None, norm_kind="none")
        w_u = getattr(model, "W_U", None)
        if isinstance(w_u, torch.Tensor):
            ln_final = getattr(model, "ln_final", None)
            weight = getattr(ln_final, "w", None) if ln_final is not None else None
            if isinstance(weight, torch.Tensor):
                kind = "layernorm" if hasattr(ln_final, "b") else "rmsnorm"
                return UnembedNormInfo(w_u=w_u.transpose(0, 1), norm_scale=weight, norm_kind=kind)
            return UnembedNormInfo(w_u=w_u.transpose(0, 1), norm_scale=None, norm_kind="none")
    raise ValueError(
        "resolve_unembed_and_norm_scale: module exposes neither an HF-style `.model.lm_head/.embed_out` "
        "nor a TransformerLens-style `.model.W_U`"
    )


DEFAULT_JLENS_REPO = "neuronpedia/jacobian-lens"
# Every published artifact lives at `{np_model_id}/jlens/{fitting-corpus}/{stem}.pt`. The corpus segment is
# NOT constant (one model was fit on pile-10k rather than wikitext) and the stem is NOT the directory name
# (it tracks the HF model name, so `gpt2-small/` holds `gpt2_...` and `gemma-4-e2b/` holds `gemma-4-E2B_...`).
# Measured over the default repo, the obvious `{dir}/jlens/Salesforce-wikitext/{dir}_jacobian_lens.pt` shape
# matches 13 of 40 artifacts, so this resolver discovers rather than formats.
_JLENS_KIND_DIR = "jlens"
_JLENS_STEM_SUFFIX = "_jacobian_lens"
_JLENS_SIDECAR = "config.yaml"


class JLensArtifact(NamedTuple):
    """One resolved Jacobian lens, with the provenance a reader needs to interpret a weak result.

    Attributes:
        j_by_layer: ``{layer_index: Tensor[d_model, d_model]}``, the averaged causal Jacobians.
        source_layers: The layer indices the lens was fit at, ascending.
        d_model: Residual width the lens expects; a mismatch against the model is a hard error, not a
            broadcast.
        repo_id: Repository the artifact came from.
        path: Repo-relative path of the artifact actually loaded, so a caller can quote what it used.
        hf_model_name: The HF model the lens was fit against, read from the sidecar rather than inferred.
        provenance: The sidecar's ``fit`` and ``results`` blocks when present. ``results.prompts_fitted``
            is the count actually used (fitting stops on a convergence delta, so it is usually well below
            the configured ``n_prompts``), and ``results.final_identity_distance`` says how far the lens
            ended up from the identity, the degenerate case where a J-lens reduces to a logit lens. A weak
            probe against a barely-converged lens is a different finding from a weak probe against a
            well-separated one, which a caller can only tell if these travel with the artifact.
    """

    j_by_layer: dict[int, torch.Tensor]
    source_layers: list[int]
    d_model: int
    repo_id: str
    path: str
    hf_model_name: str | None
    provenance: dict[str, Any]


def _jlens_repo_artifacts(repo_id: str, revision: str | None, token: str | None) -> dict[str, list[str]]:
    """``{np_model_id: [artifact paths]}`` for a lens repository, discovered from its file listing."""
    from huggingface_hub import HfApi

    grouped: dict[str, list[str]] = {}
    for name in HfApi().list_repo_files(repo_id, revision=revision, token=token):
        parts = name.split("/")
        if len(parts) == 4 and parts[1] == _JLENS_KIND_DIR and name.endswith(".pt"):
            grouped.setdefault(parts[0], []).append(name)
    return {k: sorted(v) for k, v in grouped.items()}


def _select_jlens_artifact(candidates: list[str], model_dir: str) -> str:
    """Pick one artifact from a model directory, or refuse naming the alternatives.

    Prefers the unsuffixed stem, accepts a lone candidate whatever its suffix, and raises when several
    remain. Both halves are load-bearing on the published repository: one directory carries a default and
    an ``_n1000`` variant, so "take the only one" is wrong there, and two directories carry ONLY a
    suffixed artifact, so "require the unsuffixed name" is wrong there. Picking by sort order would
    silently prefer whichever name sorts first, which is not a property anyone chose.
    """
    plain = [c for c in candidates if c.rsplit("/", 1)[-1].endswith(f"{_JLENS_STEM_SUFFIX}.pt")]
    if len(plain) == 1:
        return plain[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"{model_dir!r} publishes {len(candidates)} lens artifacts and none is unambiguously the default: "
        f"{[c.rsplit('/', 1)[-1] for c in candidates]}. Pass `path=` to choose one."
    )


def _read_jlens_sidecar(repo_id: str, artifact_path: str, revision: str | None, token: str | None) -> dict:
    """The ``config.yaml`` beside an artifact, or ``{}`` when the repository publishes none."""
    import yaml
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    sidecar = f"{artifact_path.rsplit('/', 1)[0]}/{_JLENS_SIDECAR}"
    try:
        text = pathlib.Path(hf_hub_download(repo_id, sidecar, revision=revision, token=token)).read_text(
            encoding="utf-8"
        )
    except (EntryNotFoundError, OSError):
        return {}
    # The sidecar leads with a provenance comment block; strip it so the YAML body parses on its own.
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    return yaml.safe_load(body) or {}


def _resolve_model_name_for_lens(module: Any) -> str | None:
    """Best-effort HF model name for the module, used only to PROPOSE a lens directory."""
    for path in (
        ("model", "config", "_name_or_path"),
        ("model", "config", "name_or_path"),
        ("model", "cfg", "model_name"),
        ("model", "name_or_path"),
    ):
        value = _resolve_attr_path(module, *path)
        if isinstance(value, str) and value:
            return value
    return None


def _model_names_agree(declared: str | None, model_name: str) -> bool:
    """Whether a sidecar's ``hf_model_name`` corroborates the model we are resolving a lens for.

    Sidecars record the fully qualified name (``openai-community/gpt2``), while a model loaded by its
    short name self-reports ``gpt2``, so requiring string equality rejects the correct lens for every
    model loaded the short way. When the caller's name carries no organization there is no
    organization to compare, so the basenames are compared instead; that is weaker evidence, but it is
    the strongest available and still far more than a formatted path, which checks nothing at all.

    ``declared is None`` means the directory publishes no sidecar (one does not), which is accepted
    rather than refused: the caller sees it as ``hf_model_name=None`` on the artifact.
    """
    if declared is None:
        return True
    if "/" in model_name:
        return declared.lower() == model_name.lower()
    return declared.rsplit("/", 1)[-1].lower() == model_name.lower()


def resolve_jlens(
    module: Any,
    *,
    repo_id: str = DEFAULT_JLENS_REPO,
    model_id: str | None = None,
    path: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> JLensArtifact:
    """Resolve a pre-fitted Jacobian lens for ``module``, verifying the match rather than assuming it.

    Resolution is deliberately propose-then-verify. A candidate directory comes from the model's HF name,
    matched against artifact STEMS rather than directory names (the stem tracks the HF name and the
    directory does not), and the candidate is then checked against the sidecar's ``hf_model_name`` before
    anything loads. If it disagrees, every sidecar is scanned and the right directory is used.

    The verification is the point. A mismatched lens does not fail: it produces a readout that is entirely
    plausible and quietly wrong, which a formatted path cannot detect and a checked one cannot miss.

    ``model_id`` names the repository directory directly and ``path`` names the artifact outright; both
    skip discovery, so a lens this resolver cannot place never blocks a caller.

    A sidecar is not guaranteed: 38 of the 39 model directories in the default repository publish one,
    and the remaining directory publishes none. A resolution that could not be confirmed is not refused,
    because refusing would make one model unreachable for no safety gain, but it is not silent either:
    the returned artifact's ``hf_model_name`` is ``None`` exactly when nothing corroborated the match, so
    a caller that cares can tell a verified resolution from a merely plausible one.
    """
    if path is None:
        artifacts = _jlens_repo_artifacts(repo_id, revision, token)
        if not artifacts:
            raise ValueError(f"{repo_id!r} publishes no `*/{_JLENS_KIND_DIR}/*/*.pt` lens artifacts")
        if model_id is not None:
            if model_id not in artifacts:
                raise ValueError(f"{repo_id!r} has no lens directory {model_id!r}; it publishes {sorted(artifacts)}")
            path = _select_jlens_artifact(artifacts[model_id], model_id)
        else:
            path = _discover_jlens_path(module, repo_id, artifacts, revision, token)
    sidecar = _read_jlens_sidecar(repo_id, path, revision, token)
    return _load_jlens_artifact(repo_id, path, revision, token, sidecar)


def _discover_jlens_path(
    module: Any, repo_id: str, artifacts: dict[str, list[str]], revision: str | None, token: str | None
) -> str:
    """Propose a directory from the model name, then confirm it against the sidecar before loading."""
    model_name = _resolve_model_name_for_lens(module)
    if not model_name:
        raise ValueError(
            f"could not determine a model name to match against {repo_id!r}; pass `model_id=` (one of "
            f"{sorted(artifacts)}) or `path=`"
        )
    basename = model_name.rsplit("/", 1)[-1].lower()
    for model_dir, candidates in sorted(artifacts.items()):
        for candidate in candidates:
            stem = candidate.rsplit("/", 1)[-1].lower()
            if stem.startswith(f"{basename}{_JLENS_STEM_SUFFIX}"):
                sidecar = _read_jlens_sidecar(repo_id, candidate, revision, token)
                if _model_names_agree(sidecar.get("hf_model_name"), model_name):
                    return _select_jlens_artifact(candidates, model_dir)
    for model_dir, candidates in sorted(artifacts.items()):
        chosen = candidates[0]
        sidecar = _read_jlens_sidecar(repo_id, chosen, revision, token)
        if sidecar.get("hf_model_name") and _model_names_agree(sidecar["hf_model_name"], model_name):
            return _select_jlens_artifact(candidates, model_dir)
    raise ValueError(
        f"{repo_id!r} publishes no lens whose `hf_model_name` is {model_name!r}; it covers "
        f"{sorted(artifacts)}. Pass `model_id=` or `path=` to select one explicitly."
    )


def _load_jlens_artifact(
    repo_id: str, path: str, revision: str | None, token: str | None, sidecar: dict
) -> JLensArtifact:
    """Load a resolved artifact, memory-mapped so a single layer does not materialize the whole lens."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id, path, revision=revision, token=token)
    try:
        ckpt = torch.load(local, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, ValueError):  # not a zipfile-serialized checkpoint; mmap is unavailable
        ckpt = torch.load(local, map_location="cpu", weights_only=True)
    j_by_layer = {int(k): v for k, v in (ckpt.get("J") or {}).items()}
    if not j_by_layer:
        raise ValueError(f"{repo_id}:{path} carries no `J` layer mapping, so it is not a usable lens")
    source_layers = sorted(int(x) for x in (ckpt.get("source_layers") or j_by_layer))
    d_model = int(ckpt.get("d_model") or next(iter(j_by_layer.values())).shape[0])
    provenance = {k: sidecar[k] for k in ("fit", "results", "dataset") if k in sidecar}
    if "n_prompts" in ckpt:
        provenance.setdefault("checkpoint", {})["n_prompts"] = int(ckpt["n_prompts"])
    return JLensArtifact(
        j_by_layer=j_by_layer,
        source_layers=source_layers,
        d_model=d_model,
        repo_id=repo_id,
        path=path,
        hf_model_name=sidecar.get("hf_model_name"),
        provenance=provenance,
    )


def jlens_layer_for_percentile(artifact: JLensArtifact, percentile: float) -> int:
    """The fitted layer at ``percentile`` through the fitted set, ``percentile`` in [0, 1].

    Lenses are fit at a sampled subset of layers, so a caller asking for "85%" needs a layer that was
    actually fit rather than the nearest layer index.

    This indexes POSITION IN THE FITTED SET, not depth in the model, and the two differ once rounding
    enters: five layers fit at 0, 6, 12, 18, 24 put ``percentile=0.85`` at index 3, which is layer 18 and
    therefore 75% of the way down. Position is the definition the published steering recipes were tuned
    against, so it is kept deliberately rather than quietly upgraded to true depth: changing it would move
    which layer a validated demo patches. A caller who means depth should compute the layer directly.
    """
    if not 0.0 <= percentile <= 1.0:
        raise ValueError(f"percentile must be in [0, 1], got {percentile}")
    layers = artifact.source_layers
    return layers[round(percentile * (len(layers) - 1))]


def fold_norm_into_unembed_rows(info: UnembedNormInfo, token_ids: Any, *, apply_norm: bool) -> torch.Tensor:
    """Readout-faithful unembed rows for ``token_ids``, with the final norm folded in per kind.

    Returns ``(n_tokens, d_model)`` float rows, one per id, in the order given. Composing a lens
    direction is the caller's job (``rows @ J`` for a J-lens, ``rows`` alone for a logit lens); this
    function owns only the part that is easy to get subtly wrong.

    Two conventions, both exact rather than heuristic. For an RMSNorm the readout
    ``W_U[c] . norm(x)`` equals ``(W_U[c] * scale) . x / rms(x)``, so folding the elementwise scale
    into the row reproduces the readout's own direction and the input-dependent ``1/rms(x)`` scales
    magnitude only. A LayerNorm additionally subtracts the mean, and pushing that through the dot
    product moves a centering onto the row: ``(W_U[c] * scale) . (x - mean(x)1) = C(W_U[c] * scale) . x``
    with ``C = I - 11^T/d``. The learned bias contributes an input-independent logit offset and drops
    out of a direction.

    ``apply_norm=False`` returns the raw rows, which is the paper's probing shorthand ("rows of
    ``W_U J``") rather than its readout formula. The two agree only when the scale is uniform, and
    which one steers better is model-dependent, so neither is a safe default to hard-code.

    **`apply_norm` is required, and that is what this paragraph used to only assert.** The signature
    carried `= True` while the sentence above said no default is safe, so a caller could pick a basis by
    omission and one did: an op composed its atoms through this function without passing the flag, taking
    the basis from a default two layers away, with nothing in its schema or output recording the choice.
    Changing that default would have silently changed that op's basis. A required parameter makes the
    omission unrepresentable rather than merely wrong.
    """
    ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1)
    if ids.numel() == 0:
        raise ValueError("fold_norm_into_unembed_rows requires at least one token id")
    rows = info.w_u[ids].float()
    if not apply_norm or info.norm_scale is None:
        return rows
    rows = rows * info.norm_scale.float().to(rows.device)
    if info.norm_kind == "layernorm":
        rows = rows - rows.mean(dim=-1, keepdim=True)
    return rows


#: The two named bases, as the values an artifact records and #420's selector resolves to. Keyed by
#: whether the final norm is folded in, because that is the only thing that distinguishes them.
JLENS_BASIS_NAMES = {True: "jlens_norm_aware", False: "jlens_paper"}


def jlens_basis_name(apply_norm: bool) -> str:
    """The name of the basis ``apply_norm`` selects, for recording in an op's output.

    A result that cannot say which basis produced it is not interpretable: the folded and unfolded
    bases are not related by a coefficient, so two results differing only in this choice are not
    comparable and are indistinguishable after the fact.
    """
    return JLENS_BASIS_NAMES[bool(apply_norm)]


def jlens_direction_rows(
    info: UnembedNormInfo, token_ids: Any, j_matrix: torch.Tensor, *, apply_norm: bool
) -> torch.Tensor:
    """``(n_tokens, d_model)`` J-lens directions for ``token_ids``, in the basis ``apply_norm`` names.

    THE one construction. It existed in three places that did not agree on how the basis was chosen: one passed the flag
    through, one omitted it and inherited a default two layers away, and a third lived in a published op collection
    where the choice was decided by whichever revision a caller had pulled. Composing `rows @ J` is two lines, which is
    exactly why it was rewritten rather than shared, and why the three drifted on the part that is not the arithmetic.

    Callers wanting a single direction take row 0; callers wanting a group's mean direction may average the rows or the
    result, since the composition is linear and the two agree exactly.
    """
    rows = fold_norm_into_unembed_rows(info, token_ids, apply_norm=apply_norm)
    return rows @ j_matrix.to(rows.device)


def resolve_embedding_weight(module: Any) -> torch.Tensor:
    """Resolve an embedding weight matrix from a generic module or its analysis backend."""
    analysis_backend = get_analysis_backend(module)
    if analysis_backend is not None:
        try:
            return analysis_backend.get_embedding_weight(module)
        except (AttributeError, ValueError):
            pass

    for path in (
        ("replacement_model", "unembed_weight"),
        ("model", "unembed_weight"),
        ("replacement_model", "embed_weight"),
        ("model", "embed_weight"),
        ("replacement_model", "W_E"),
        ("model", "W_E"),
        ("replacement_model", "embed", "W_E"),
        ("model", "embed", "W_E"),
    ):
        value = _resolve_attr_path(module, *path)
        if isinstance(value, torch.Tensor):
            return value

    for attr_name in ("replacement_model", "model"):
        model = getattr(module, attr_name, None)
        get_input_embeddings = getattr(model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            embedding_layer = get_input_embeddings()
            weight = getattr(embedding_layer, "weight", None)
            if isinstance(weight, torch.Tensor):
                return weight

    raise ValueError("An embedding weight matrix is required for concept_direction")


# ---------------------------------------------------------------------------
# Tokenization utilities
# ---------------------------------------------------------------------------


def _flatten_token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, torch.Tensor):
        return [int(value) for value in tokenized.reshape(-1).tolist()]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, list):
        if tokenized and isinstance(tokenized[0], list):
            return [int(value) for sublist in tokenized for value in sublist]
        return [int(value) for value in tokenized]
    return [int(tokenized)]


def token_strings_to_ids(tokenizer: Any, token_strings: list[str]) -> list[int]:
    """Resolve token strings to token ids using either the vocab or tokenizer call path."""
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    token_ids: list[int] = []
    for token_str in token_strings:
        if token_str in vocab:
            token_ids.append(int(vocab[token_str]))
            continue
        tokenized = tokenizer(token_str, add_special_tokens=False)["input_ids"]
        token_ids.extend(_flatten_token_ids(tokenized))
    if not token_ids:
        raise ValueError("Unable to resolve any token ids for the provided concept groups")
    return token_ids


def token_strings_to_last_ids(tokenizer: Any, token_strings: list[str]) -> list[int]:
    """Resolve each token string to its terminal token id.

    This preserves one id per input token string, which is required for paired concept-direction constructions such as
    vector rejection.
    """

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    token_ids: list[int] = []
    for token_str in token_strings:
        if token_str in vocab:
            token_ids.append(int(vocab[token_str]))
            continue
        tokenized = tokenizer(token_str, add_special_tokens=False)["input_ids"]
        flattened = _flatten_token_ids(tokenized)
        if not flattened:
            raise ValueError(f"Unable to resolve a terminal token id for {token_str!r}")
        token_ids.append(int(flattened[-1]))
    if not token_ids:
        raise ValueError("Unable to resolve any token ids for the provided concept groups")
    return token_ids


def decode_token_ids(tokenizer: Any, token_ids: torch.Tensor | list[int]) -> list[str]:
    """Decode individual token ids to token strings when possible."""
    ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return [str(tokenizer.convert_ids_to_tokens(int(token_id))) for token_id in ids]
    return [str(tokenizer.decode([int(token_id)], skip_special_tokens=False)) for token_id in ids]


# ---------------------------------------------------------------------------
# Scoped-input conveniences for op implementations
# ---------------------------------------------------------------------------
# Sanctioned bridge for whole-column aggregate inputs until the declared op-state /
# declared-inputs work lands; new op code should prefer the bound ``AnalysisBatch``
# access surface for batch-scoped values.


def resolve_aggregate_input(module: Any, analysis_batch: Any, field_name: str) -> Any:
    """Resolve whole-column aggregation inputs, preferring explicit run or batch values over the input store."""
    analysis_cfg = getattr(module, "analysis_cfg", None)
    batch_inputs = getattr(analysis_cfg, "batch_inputs", {}) or {}
    run_inputs = getattr(analysis_cfg, "run_inputs", {}) or {}

    for scoped_values in (batch_inputs, run_inputs):
        if field_name in scoped_values and scoped_values[field_name] is not None:
            return scoped_values[field_name]

    if hasattr(analysis_batch, "keys") and field_name in analysis_batch.keys():
        return getattr(analysis_batch, field_name)

    input_store = getattr(analysis_cfg, "input_store", None)
    if input_store is not None:
        dataset = getattr(input_store, "dataset", None)
        raw_column_names = getattr(dataset, "column_names", None) if dataset is not None else None
        column_names = list(raw_column_names) if raw_column_names is not None else []
        if field_name in column_names:
            return input_store[field_name]
        store_value = getattr(input_store, field_name, None)
        if store_value is not None:
            return store_value

    return None


def load_json_field(module: Any, analysis_batch: Any, field_name: str) -> Any:
    """Resolve an aggregate input field and decode JSON string payloads when present."""

    raw_value = resolve_aggregate_input(module, analysis_batch, field_name)
    if isinstance(raw_value, str):
        return json.loads(raw_value)
    return raw_value


__all__ = [
    "boolean_logits_to_avg_logit_diff",
    "decode_token_ids",
    "extract_logits",
    "resolve_jlens",
    "jlens_layer_for_percentile",
    "JLensArtifact",
    "DEFAULT_JLENS_REPO",
    "fold_norm_into_unembed_rows",
    "jlens_basis_name",
    "jlens_direction_rows",
    "JLENS_BASIS_NAMES",
    "FEATURE_SCORE_SOURCE_ALIASES",
    "get_loss_preds_diffs",
    "last_token_logits",
    "load_json_field",
    "mean_target_logit_delta",
    "require_backend_capability",
    "require_model_backend",
    "resolve_aggregate_input",
    "resolve_embedding_weight",
    "resolve_unembed_and_norm_scale",
    "UnembedNormInfo",
    "resolve_feature_score_source",
    "resolve_tokenizer",
    "stack_column_tensors",
    "token_strings_to_ids",
    "token_strings_to_last_ids",
    "weighted_mean",
]
