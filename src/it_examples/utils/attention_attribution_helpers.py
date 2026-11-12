"""Helpers for the attention-aware attribution notebook: a prompt sanity check, regime comparisons, and figures.

The figures read either an analysis batch the notebook already holds (an attribution graph's adjacency matrix, a head
report) or a trace the notebook's analysis points recorded inside circuit-tracer. They are shown through
``display_plotly_figure`` so they render on the docs site as well as in a live notebook.
"""

from __future__ import annotations

import copy
import html

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from it_examples.utils.nb_ui_utils import (
    NotebookLink,
    display_notebook_table,
    display_plotly_figure,
    resolve_feature_explanations,
)

_PALETTE = ("#2471a3", "#c0392b", "#27ae60", "#8e44ad", "#e67e22", "#16a085", "#7f8c8d", "#d4ac0d")
_LAYOUT: dict[str, Any] = dict(template="simple_white", autosize=True, margin=dict(l=60, r=30, t=60, b=60))


@dataclass(frozen=True)
class NextTokenCheck:
    """The model's next-token answer to one prompt, against the answer the experiment needs.

    Attributes:
        label: A name for the prompt.
        prompt: The prompt.
        expected: The token the experiment needs the model to answer.
        rank: The expected token's rank among the next-token logits (1 is top).
        prob: The expected token's probability.
        margin: The expected token's logit minus the best other token's.
        copy_token: A token the prompt states that a model could copy instead (the swapped city), if any.
        copy_prob: That token's probability.
        top: The top tokens and their probabilities.
    """

    label: str
    prompt: str
    expected: str
    rank: int
    prob: float
    margin: float
    copy_token: str | None
    copy_prob: float | None
    top: list[tuple[str, float]]


def _first_token_id(tokenizer: Any, word: str) -> int:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is {len(ids)} tokens; a next-token check needs a single-token answer")
    return int(ids[0])


def check_next_token(
    model: Any,
    tokenizer: Any,
    label: str,
    prompt: str,
    expected: str,
    copy_token: str | None = None,
    k: int = 5,
) -> NextTokenCheck:
    """Read the model's next-token distribution for ``prompt`` and place ``expected`` in it.

    ``expected`` and ``copy_token`` are single tokens, written as the tokenizer spells them (``"▁Austin"``). The logits
    come from the replacement model's underlying model (``model.get_activations``), the forward pass every attribution
    on this prompt starts from.
    """
    expected_id = _first_token_id(tokenizer, expected)
    logits, _ = model.get_activations(prompt)
    last = logits.reshape(-1, logits.shape[-1])[-1].float()
    probs = last.softmax(-1)
    order = last.argsort(descending=True)
    rank = int((order == expected_id).nonzero()[0, 0]) + 1
    best_other = order[0] if int(order[0]) != expected_id else order[1]
    copy_prob = None
    if copy_token is not None:
        copy_prob = float(probs[_first_token_id(tokenizer, copy_token)])
    return NextTokenCheck(
        label=label,
        prompt=prompt,
        expected=expected,
        rank=rank,
        prob=float(probs[expected_id]),
        margin=float(last[expected_id] - last[best_other]),
        copy_token=copy_token,
        copy_prob=copy_prob,
        top=[(tokenizer.decode([int(t)]), float(probs[t])) for t in order[:k]],
    )


def print_next_token_checks(checks: Sequence[NextTokenCheck]) -> None:
    """Print one row per check: the expected answer's rank, probability and margin, and the copy probability."""
    print(f"{'prompt':<10} {'expected':<12} {'rank':>4} {'prob':>6} {'margin':>7} {'copy prob':>14}  top tokens")
    for c in checks:
        copy = f"{c.copy_prob:.3f} ({c.copy_token.lstrip('▁')})" if c.copy_token is not None else "-"
        top = ", ".join(f"{tok!r} {p:.3f}" for tok, p in c.top)
        print(
            f"{c.label:<10} {c.expected.lstrip('▁'):<12} {c.rank:>4} {c.prob:>6.3f} {c.margin:>+7.2f} {copy:>14}  {top}"
        )


def display_next_token_checks(checks: Sequence[NextTokenCheck]) -> None:
    """Show answer, confidence, competing city and top alternatives with explicit units."""
    display_notebook_table(
        [
            [
                c.label,
                c.expected.lstrip("▁"),
                c.rank,
                f"{c.prob:.1%}",
                f"{c.margin:+.2f}",
                f"{c.copy_prob:.1%} ({c.copy_token.lstrip('▁')})"
                if c.copy_prob is not None and c.copy_token is not None
                else "Not measured",
                ", ".join(f"{tok.strip()}: {p:.1%}" for tok, p in c.top),
            ]
            for c in checks
        ],
        ["Prompt city", "Expected capital", "Rank", "Probability", "Lead (logits)", "Copy probability", "Top tokens"],
        title="Does the model answer the intended question?",
        caption="Lead is the expected capital's logit minus the strongest competing token's logit.",
    )


@dataclass(frozen=True)
class FeatureContext:
    """What a reader needs beside a feature id: the token it fires on, a description, and where to look further.

    Attributes:
        tokens: The prompt's tokens as text, by position.
        explanations: ``(layer, feature) -> description`` for the features that have one.
        model_id: Neuronpedia's model id (``"gemma-2-2b"``).
        source_set: Neuronpedia's source set for the transcoders (``"gemmascope-transcoder-16k"``).
        base_url: The Neuronpedia instance the links point at.
    """

    tokens: Sequence[str]
    explanations: Mapping[tuple[int, int], str]
    model_id: str
    source_set: str
    base_url: str = "https://www.neuronpedia.org"

    def link(self, layer: int, feature: int) -> NotebookLink:
        """A link to the feature's Neuronpedia page, labelled with its layer and index."""
        url = f"{self.base_url.rstrip('/')}/{self.model_id}/{layer}-{self.source_set}/{feature}"
        return NotebookLink(f"L{layer} #{feature}", url)

    def token(self, pos: int) -> str:
        """The token at ``pos`` as quoted text, so a leading space stays visible."""
        return repr(self.tokens[pos])

    def explanation(self, layer: int, feature: int) -> str:
        """The feature's description, or a dash when Neuronpedia has none (or could not be reached)."""
        return self.explanations.get((int(layer), int(feature)), "-")

    def describe(self, layer: int, pos: int, feature: int) -> str:
        """One line for prose: ``L20 #15589 on ' is' (pos 10), "Texas"``."""
        text = self.explanations.get((int(layer), int(feature)))
        suffix = f', "{text}"' if text else ""
        return f"L{layer} #{feature} on {self.token(pos)} (pos {pos}){suffix}"


def resolve_feature_context(
    features: Sequence[tuple[int, int, int]],
    tokens: Sequence[str],
    *,
    model_id: str,
    source_set: str,
    preferred_type_name: str | None = "oai_token-act-pair",
    base_url: str = "https://www.neuronpedia.org",
) -> FeatureContext:
    """Fetch descriptions for the ``(layer, pos, feature)`` triples a notebook displays, and nothing more.

    One request per distinct feature, so pass the handful a table shows rather than a graph's whole selection. The
    preferred type is Neuronpedia's autointerpretation; a feature without one falls back to another listed explanation.
    A description is a hypothesis about what a feature responds to, not a measured causal role.
    """
    pairs = list(dict.fromkeys((int(layer), int(feat)) for layer, _, feat in features))
    explanations = resolve_feature_explanations(
        model_id=model_id,
        source_set=source_set,
        feature_tuples=pairs,
        base_url=base_url,
        preferred_type_name=preferred_type_name,
    )
    return FeatureContext(list(tokens), explanations, model_id, source_set, base_url)


@dataclass(frozen=True)
class AlignedRegimes:
    """Index maps that put two attribution graphs of one prompt on the nodes they share.

    Each graph selects its own feature nodes, by attribution, so two graphs of the same prompt built under different
    regimes have adjacency matrices of the same shape whose feature columns name different features. Comparing them
    index by index compares unrelated nodes. ``first`` and ``second`` index each graph's nodes so that position ``j``
    names the same node in both: first the features both graphs selected (as rows of ``active_features``), then every
    error, token and logit node, which the two graphs lay out identically.

    Attributes:
        first: Node indices into the first graph.
        second: Node indices into the second graph.
        common_features: ``active_features`` rows of the shared feature nodes, in aligned order.
        n_first_only: Features only the first graph selected.
        n_second_only: Features only the second graph selected.
    """

    first: torch.Tensor
    second: torch.Tensor
    common_features: torch.Tensor
    n_first_only: int
    n_second_only: int

    @property
    def n_common(self) -> int:
        """The number of shared feature nodes, which come first in the aligned order."""
        return len(self.common_features)


def align_regime_nodes(first: Any, second: Any) -> AlignedRegimes:
    """Align two attribution graphs of one prompt on their shared nodes, refusing graphs that are not comparable.

    Both must share the prompt's active features (the same prompt and transcoders) and the logit targets, and lay out
    the same number of non-feature nodes; otherwise no node correspondence exists and a comparison would be a guess.
    """
    if not torch.equal(first.active_features, second.active_features):
        raise ValueError("the two graphs have different active features, so they are not graphs of one prompt")
    if [int(t) for t in first.logit_target_ids] != [int(t) for t in second.logit_target_ids]:
        raise ValueError("the two graphs have different logit targets, so their logit nodes do not correspond")
    sel_a, sel_b = first.selected_features.tolist(), second.selected_features.tolist()
    n_nodes_a, n_nodes_b = first.adjacency_matrix.shape[1], second.adjacency_matrix.shape[1]
    if n_nodes_a - len(sel_a) != n_nodes_b - len(sel_b):
        raise ValueError("the two graphs lay out different numbers of error, token and logit nodes")
    pos_b = {feature: i for i, feature in enumerate(sel_b)}
    common = [feature for feature in sel_a if feature in pos_b]
    pos_a = {feature: i for i, feature in enumerate(sel_a)}
    rest_a = list(range(len(sel_a), n_nodes_a))
    rest_b = list(range(len(sel_b), n_nodes_b))
    return AlignedRegimes(
        first=torch.tensor([pos_a[f] for f in common] + rest_a, dtype=torch.long),
        second=torch.tensor([pos_b[f] for f in common] + rest_b, dtype=torch.long),
        common_features=first.active_features[torch.tensor(common, dtype=torch.long)],
        n_first_only=len(sel_a) - len(common),
        n_second_only=len(sel_b) - len(common),
    )


def node_kinds(n_features: int, n_layers: int, n_pos: int, n_logits: int) -> list[str]:
    """The kind of every adjacency-matrix node, in circuit-tracer's order: features, errors, tokens, logits."""
    return ["feature"] * n_features + ["error"] * (n_layers * n_pos) + ["token"] * n_pos + ["logit"] * n_logits


def source_layers(active_features: torch.Tensor, n_layers: int, n_pos: int, n_logits: int) -> np.ndarray:
    """The layer of every source node (features and errors by their layer, tokens -1, logits ``n_layers``)."""
    feature_layers = active_features[:, 0].cpu().numpy()
    error_layers = np.repeat(np.arange(n_layers), n_pos)
    return np.concatenate([feature_layers, error_layers, np.full(n_pos, -1), np.full(n_logits, n_layers)])


def node_positions(features: torch.Tensor, n_layers: int, n_pos: int, n_logits: int) -> np.ndarray:
    """The prompt position of every node (features, errors and tokens by their position; logits ``-1``)."""
    feature_pos = features[:, 1].cpu().numpy()
    error_pos = np.tile(np.arange(n_pos), n_layers)
    return np.concatenate([feature_pos, error_pos, np.arange(n_pos), np.full(n_logits, -1)])


def influence_by_kind(influence: torch.Tensor, kinds: Sequence[str]) -> dict[str, float]:
    """The summed influence of each node kind, and the error nodes' share of the summed influence.

    Influence is linear in the logit weights, so the sums scale with the targets' total probability; the share is the
    quantity that can be compared between graphs whose targets differ.
    """
    values = influence.float().cpu().numpy()
    kinds_arr = np.asarray(kinds)
    sums = {kind: float(values[kinds_arr == kind].sum()) for kind in ("feature", "error", "token")}
    total = sum(sums.values())
    return {**sums, "total": total, "error_share": sums["error"] / total if total else float("nan")}


def position_change_mass(delta: torch.Tensor, positions: np.ndarray, kinds: Sequence[str], n_pos: int) -> np.ndarray:
    """``(n_pos, n_pos)``: summed ``|delta|`` of the feature-to-feature edges, by (target position, source
    position).

    An edge between two positions can only run through attention, so mass off the diagonal is credit that moves between
    positions; mass on it stays within one position's residual stream.
    """
    features = np.asarray(kinds) == "feature"
    block = delta.abs()[np.flatnonzero(features)][:, np.flatnonzero(features)].float().cpu().numpy()
    pos = positions[features]
    out = np.zeros((n_pos, n_pos))
    np.add.at(out, (pos[:, None].repeat(len(pos), 1), pos[None, :].repeat(len(pos), 0)), block)
    return out


@dataclass(frozen=True)
class ResidualPartition:
    """The change of a logit difference between two prompts, split over the components that wrote the final
    residual.

    The final residual at the last position is the embedding plus every attention and MLP output, so the change of any
    function of it is split exactly by integrating its gradient along the straight line between the two residuals
    (integrated gradients over a sum is additive over its terms). Each part is a DIRECT path: what that component
    wrote into the final residual. An effect an attention layer has through a later MLP is credited to the MLP.

    Attributes:
        t_from: The logit difference on the first prompt (recomputed in fp32 from the captured residual).
        t_to: The same on the second prompt.
        parts: Component name (``embed``, ``attn<L>``, ``mlp<L>``) to its part of the change, in logits.
        recon_error: The relative error of rebuilding the first prompt's final residual from the components, the
            control that the components really are the residual's terms.
    """

    t_from: float
    t_to: float
    parts: dict[str, float]
    recon_error: float

    @property
    def change(self) -> float:
        return self.t_to - self.t_from

    @property
    def unexplained(self) -> float:
        """The change the parts do not account for (the integral's discretization)."""
        return self.change - sum(self.parts.values())

    def total(self, prefix: str) -> float:
        return sum(v for k, v in self.parts.items() if k.startswith(prefix))


def _hf_causal_lm(model: Any) -> Any:
    """The Hugging Face model behind an nnsight wrapper (``_model``), or the model itself."""
    inner = getattr(model, "_model", None)
    return inner if isinstance(inner, torch.nn.Module) else model


def capture_residual_terms(module: Any, prompt: str, n_layers: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Next-token logits and every term of the residual stream for ``prompt``, through the model backend."""
    from interpretune.analysis.optools import require_model_backend

    hooks = (
        ["blocks.0.hook_resid_pre"]
        + [f"blocks.{layer}.hook_{kind}_out" for layer in range(n_layers) for kind in ("attn", "mlp")]
        + [f"blocks.{n_layers - 1}.hook_resid_post"]
    )
    model = module.model
    device = next(_hf_causal_lm(model).parameters()).device
    batch = {k: v.to(device) for k, v in dict(module.replacement_model.tokenizer(prompt, return_tensors="pt")).items()}
    logits, cache = require_model_backend(module).fwd_w_cache(model=model, batch=batch, names_filter=hooks)
    return torch.as_tensor(logits).float(), {h: torch.as_tensor(cache[h]).detach() for h in hooks}


def partition_final_residual(
    module: Any, prompt: str, counterfactual: str, token_a: int, token_b: int, n_steps: int = 64
) -> ResidualPartition:
    """Split the change of ``logit[token_a] - logit[token_b]`` between two prompts over the final residual's terms.

    The readout (final norm, unembedding and, where the model has one, the logit soft cap) is evaluated in fp32 so the
    partition's own arithmetic is not limited by the model's dtype.
    """
    hf = _hf_causal_lm(module.model)
    decoder, head = hf.get_decoder(), hf.get_output_embeddings()
    norm = getattr(decoder, "norm", None)
    if norm is None or head is None:
        raise ValueError(f"{type(hf).__name__}: no final norm and output embedding to read the residual through")
    norm = copy.deepcopy(norm).float()
    w_u = head.weight.float()
    cap = getattr(hf.config, "final_logit_softcapping", None)
    n_layers = int(hf.config.num_hidden_layers)
    final = f"blocks.{n_layers - 1}.hook_resid_post"
    _, first = capture_residual_terms(module, prompt, n_layers)
    _, second = capture_residual_terms(module, counterfactual, n_layers)
    terms = {"embed": "blocks.0.hook_resid_pre"}
    terms.update(
        {f"{kind}{layer}": f"blocks.{layer}.hook_{kind}_out" for layer in range(n_layers) for kind in ("attn", "mlp")}
    )
    rebuilt = sum(first[h][0, -1].float() for h in terms.values())
    recon = float((rebuilt - first[final][0, -1].float()).norm() / first[final][0, -1].float().norm())

    def readout(r: torch.Tensor) -> torch.Tensor:
        z = norm(r) @ w_u.T
        if cap:
            z = cap * torch.tanh(z / cap)
        return z[..., token_a] - z[..., token_b]

    r0, r1 = first[final][0, -1].float(), second[final][0, -1].float()
    alphas = torch.linspace(0, 1, n_steps + 1, device=r0.device)
    weights = torch.full((n_steps + 1,), 1.0 / n_steps, device=r0.device)
    weights[0] = weights[-1] = 0.5 / n_steps
    path = (r0 + alphas[:, None] * (r1 - r0)).requires_grad_(True)
    values = readout(path)
    values.sum().backward()
    grad = (path.grad * weights[:, None]).sum(0)
    parts = {name: float(grad @ (second[h][0, -1].float() - first[h][0, -1].float())) for name, h in terms.items()}
    return ResidualPartition(float(values[0].detach()), float(values[-1].detach()), parts, recon)


def last_token_edits(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor], hooks: Sequence[str]
) -> dict[str, torch.Tensor]:
    """The second prompt's minus the first prompt's output at the last position, per hook: an additive edit."""
    return {h: (second[h][0, -1] - first[h][0, -1]).float() for h in hooks}


def apply_last_token_edits(
    module: Any,
    prompt: str,
    edits: Mapping[str, torch.Tensor],
    dose: float,
    token_a: int,
    token_b: int,
    mode: str = "add",
) -> dict[str, Any]:
    """Edit each site at the last position through ``model_fwd_intervention`` and read the answer.

    ``mode="add"`` adds ``dose`` times each tensor; ``mode="replace"`` overwrites the site with the tensor (an
    interchange, when the tensor is another prompt's value there; ``dose`` is then ignored). The model runs with its
    own attention patterns and norms, so every site after an edited one responds to it. Returns the logit difference,
    both tokens' probabilities, the top token and, for ``add``, the edit's size (the L2 norm of every site's added
    vector together, in residual units).
    """
    import interpretune as it
    from interpretune.analysis.backends.interventions import InterventionSpec
    from interpretune.analysis.ops.base import AnalysisBatch

    hf = _hf_causal_lm(module.model)
    device, dtype = next(hf.parameters()).device, next(hf.parameters()).dtype
    tokenizer = module.replacement_model.tokenizer
    batch = {k: v.to(device) for k, v in dict(tokenizer(prompt, return_tensors="pt")).items()}
    specs = {
        h: InterventionSpec(
            intervention_tensor=d.to(device, dtype),
            mode=mode,
            scale_factor=dose if mode == "add" else 1.0,
            position_scope="last_token",
        )
        for h, d in edits.items()
    }
    out = it.model_fwd_intervention(
        module, AnalysisBatch(prompts=[prompt], interventions=specs, logit_target_ids=torch.tensor([token_a])), batch, 0
    )
    post = out.post_intervention_logits.float().reshape(-1)
    probs = torch.softmax(post, -1)
    top = int(post.argmax())
    norm = float(torch.sqrt(sum((d.float() ** 2).sum() for d in edits.values()))) if edits else 0.0
    size = dose * norm if mode == "add" else float("nan")
    return {
        "dose": dose,
        "gap": float(post[token_a] - post[token_b]),
        "p_a": float(probs[token_a]),
        "p_b": float(probs[token_b]),
        "top": tokenizer.decode([top]),
        "top_p": float(probs[top]),
        "edit_norm": size,
        "pre_gap": float(
            out.pre_intervention_logits.float().reshape(-1)[token_a]
            - out.pre_intervention_logits.float().reshape(-1)[token_b]
        ),
    }


def _asinh_axis(title: str, scale: float) -> dict[str, Any]:
    ticks = [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0]
    return dict(
        title=title,
        tickvals=[float(np.arcsinh(t / scale)) for t in ticks],
        ticktext=[f"{t:g}" for t in ticks],
        zeroline=True,
    )


def plot_regime_edges(
    frozen_row: torch.Tensor,
    live_row: torch.Tensor,
    kinds: Sequence[str],
    labels: Sequence[str],
    target: str,
    annotate: int = 6,
    scale: float = 0.01,
    hover: Sequence[str] | None = None,
) -> go.Figure:
    """Scatter every source's edge into one target, frozen regime against live, one colour per node kind.

    ``hover`` (one per source, defaulting to ``labels``) is the hover text, so a description can ride on the points
    without crowding the annotations.

    Both axes use $\\operatorname{asinh}(x / s)$ so small edges stay visible beside large ones while the sign is kept;
    tick labels are raw edge values. Points on the diagonal are sources the regime does not change. The ``annotate``
    sources whose edge changes most are labelled.
    """
    frozen_np, live_np = frozen_row.float().cpu().numpy(), live_row.float().cpu().numpy()
    keep = (frozen_np != 0) | (live_np != 0)
    fig = go.Figure()
    for kind in dict.fromkeys(kinds):
        mask = keep & (np.asarray(kinds) == kind)
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        fig.add_trace(
            go.Scatter(
                x=np.arcsinh(frozen_np[idx] / scale),
                y=np.arcsinh(live_np[idx] / scale),
                mode="markers",
                name=f"{kind} ({len(idx)})",
                marker=dict(size=5, opacity=0.6),
                text=[html.escape((hover or labels)[int(i)]) for i in idx],
                customdata=np.stack([frozen_np[idx], live_np[idx]], axis=1),
                hovertemplate="%{text}<br>frozen %{customdata[0]:.4f}<br>live %{customdata[1]:.4f}<extra></extra>",
            )
        )
    lim = float(np.arcsinh(max(np.abs(frozen_np).max(), np.abs(live_np).max()) / scale))
    fig.add_trace(
        go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", line=dict(color="gray", dash="dash"), name="no change")
    )
    change = np.abs(live_np - frozen_np)
    for i in np.argsort(change)[::-1][:annotate]:
        fig.add_annotation(
            x=float(np.arcsinh(frozen_np[i] / scale)),
            y=float(np.arcsinh(live_np[i] / scale)),
            text=labels[int(i)],
            showarrow=True,
            arrowhead=1,
            font=dict(size=10),
        )
    fig.update_layout(
        title=f"Edges into {target}: frozen against live patterns",
        xaxis=_asinh_axis("frozen-pattern edge", scale),
        yaxis=_asinh_axis("live-pattern edge", scale),
        height=560,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
        **_LAYOUT,
    )
    return fig


def plot_position_change_mass(mass: np.ndarray, tokens: Sequence[str], title: str) -> go.Figure:
    """The feature-to-feature change mass by (target position, source position), as a lower-triangular heatmap."""
    labels = [f"{i} {tok!r}" for i, tok in enumerate(tokens)]
    fig = go.Figure(
        go.Heatmap(
            z=mass,
            x=labels,
            y=labels,
            colorscale="Viridis",
            colorbar=dict(title="summed |live - frozen|"),
            hovertemplate="target %{y}<br>source %{x}<br>%{z:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title="source position"),
        yaxis=dict(title="target position", autorange="reversed"),
        height=520,
        **_LAYOUT,
    )
    return fig


def plot_dose_response(arms: Mapping[str, Sequence[Mapping[str, Any]]], label_a: str, label_b: str) -> go.Figure:
    """Each arm's logit difference against its dose, and against the edit's size, with the sign change marked."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("against the dose", "against the edit's size"))
    for i, (name, rows) in enumerate(arms.items()):
        color = _PALETTE[i % len(_PALETTE)]
        text = [f"{name}, dose {r['dose']}: top {r['top']!r} ({r['top_p']:.0%})" for r in rows]
        for col, key in ((1, "dose"), (2, "edit_norm")):
            fig.add_trace(
                go.Scatter(
                    x=[r[key] for r in rows],
                    y=[r["gap"] for r in rows],
                    mode="lines+markers",
                    name=name,
                    legendgroup=name,
                    showlegend=col == 1,
                    line=dict(color=color),
                    text=[html.escape(t) for t in text],
                    hovertemplate="%{text}<br>gap %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=col,
            )
    for col in (1, 2):
        fig.add_hline(y=0.0, line=dict(color="gray", dash="dash"), row=1, col=col)
    fig.update_xaxes(title_text="dose (1: the full Dallas-to-Detroit change)", row=1, col=1)
    fig.update_xaxes(title_text="edit size (L2 norm over every edited site, residual units)", row=1, col=2)
    fig.update_yaxes(title_text=f"{label_a} minus {label_b} (logits)", row=1, col=1)
    fig.update_layout(height=460, legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0), **_LAYOUT)
    return fig


def plot_layer_bars(series: Mapping[str, Sequence[float]], title: str, yaxis_title: str) -> go.Figure:
    """Grouped bars per layer, one bar series per entry of ``series``."""
    fig = go.Figure()
    for i, (name, values) in enumerate(series.items()):
        fig.add_trace(
            go.Bar(
                x=[f"L{layer}" for layer in range(len(values))],
                y=list(values),
                name=name,
                marker_color=_PALETTE[i % len(_PALETTE)],
            )
        )
    fig.update_layout(
        title=title,
        barmode="group",
        yaxis=dict(title=yaxis_title),
        height=420,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        **_LAYOUT,
    )
    return fig


def plot_ridgeline(
    series: Sequence[tuple[str, np.ndarray]],
    title: str,
    x_title: str,
    bins: int = 40,
    floor: float = 1e-8,
) -> go.Figure:
    """One density curve of $\\log_{10}$ of each series' magnitudes, stacked bottom to top in the order given.

    Values at or below ``floor`` are left out of a curve and counted in its hover text, so a series that is mostly
    zeros reads as such rather than as a spike at the floor.
    """
    fig = go.Figure()
    palette = [
        f"rgba({int(68 + 185 * t)}, {int(1 + 230 * t)}, {int(84 + 30 * (1 - t))}, 0.55)"
        for t in np.linspace(0, 1, max(len(series), 2))
    ]
    for row, (name, values) in enumerate(series):
        values = np.abs(np.asarray(values, dtype=np.float64))
        nonzero = values[values > floor]
        if nonzero.size < 2:
            continue
        logs = np.log10(nonzero)
        counts, edges = np.histogram(logs, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        height = 0.9 * counts / counts.max()
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([centers, centers[::-1]]),
                y=np.concatenate([row + height, np.full(len(centers), row)]),
                fill="toself",
                fillcolor=palette[row],
                line=dict(color="black", width=0.6),
                name=name,
                hoverinfo="text",
                text=(
                    f"{name}<br>{nonzero.size} of {values.size} above {floor:g}<br>"
                    f"median {np.median(nonzero):.2e}, max {nonzero.max():.2e}"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_title),
        yaxis=dict(tickvals=list(range(len(series))), ticktext=[name for name, _ in series], title=None),
        height=max(360, 38 * len(series) + 120),
        showlegend=False,
        **_LAYOUT,
    )
    return fig


def neumann_term_mass(trace: Sequence[torch.Tensor]) -> np.ndarray:
    """Each Neumann-series term's total mass: the influence the series still adds at that order."""
    return np.array([float(term.abs().sum()) for term in trace])


def plot_neumann_convergence(traces: Mapping[str, Sequence[torch.Tensor]]) -> go.Figure:
    """Per-term mass of each regime's node-influence series, on a log scale, one line per regime."""
    fig = go.Figure()
    for name, trace in traces.items():
        mass = neumann_term_mass(trace)
        fig.add_trace(go.Scatter(x=np.arange(1, len(mass) + 1), y=mass, mode="lines+markers", name=name))
    fig.update_layout(
        title="Node influence as a Neumann series: the mass each further term adds",
        xaxis=dict(title="term (path length in edges)"),
        yaxis=dict(title="term mass", type="log", exponentformat="power"),
        height=420,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        **_LAYOUT,
    )
    return fig


def top_k_overlap(
    a: torch.Tensor, b: torch.Tensor, ks: Sequence[int], ids_a: Sequence[Any], ids_b: Sequence[Any]
) -> dict[int, float]:
    """Jaccard overlap of the top-``k`` nodes by two score vectors, for each ``k``, matched by node identity.

    ``ids_a`` and ``ids_b`` name each score's node (for a feature, its row of ``active_features``), since two graphs
    index the same node differently.
    """
    if len(ids_a) != len(a) or len(ids_b) != len(b):
        raise ValueError("every score needs a node identity")
    overlap = {}
    for k in ks:
        top_a = {ids_a[i] for i in a.topk(k).indices.tolist()}
        top_b = {ids_b[i] for i in b.topk(k).indices.tolist()}
        overlap[k] = len(top_a & top_b) / len(top_a | top_b)
    return overlap


def plot_head_heatmap(
    values: torch.Tensor,
    layers: Sequence[int],
    title: str,
    colorbar_title: str,
    extra_column: tuple[str, torch.Tensor] | None = None,
) -> go.Figure:
    """A layer-by-head heatmap on a diverging scale centred on zero, optionally with one more named column."""
    z = values.float().cpu().numpy()
    columns = [f"h{h}" for h in range(z.shape[1])]
    if extra_column is not None:
        z = np.concatenate([z, extra_column[1].float().cpu().numpy().reshape(-1, 1)], axis=1)
        columns.append(extra_column[0])
    bound = float(np.abs(z).max()) or 1.0
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=columns,
            y=[f"L{layer}" for layer in layers],
            colorscale="RdBu_r",
            zmid=0.0,
            zmin=-bound,
            zmax=bound,
            colorbar=dict(title=colorbar_title),
            hovertemplate="%{y} %{x}: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=max(360, 22 * len(layers) + 140), yaxis=dict(autorange="reversed"), **_LAYOUT)
    return fig


def plot_head_split(record: Mapping[str, Any], source: int, source_label: str, target: str) -> go.Figure:
    """One source's edge into the target at one layer, built from the head-loadings copies.

    ``record`` is one entry of the notebook's head-loadings trace: ``rows`` holds one copy per head (the bypass plus
    that head), then the bypass alone, then every head live. Each head's part is its copy minus the bypass, and the bar
    labelled "sum of parts" is what those parts and the bypass add to, beside the independently computed edge.
    """
    rows, n_heads = record["rows"], record["n_heads"]
    bypass, total = float(rows[n_heads, source]), float(rows[-1, source])
    parts = [float(rows[h, source]) - bypass for h in range(n_heads)]
    names = ["bypass"] + [f"head {h}" for h in range(n_heads)]
    fig = go.Figure(
        go.Waterfall(
            x=names + ["sum of parts"],
            measure=["relative"] * (n_heads + 1) + ["total"],
            y=[bypass] + parts + [0.0],
            connector=dict(line=dict(color="lightgray")),
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=total, line_dash="dash", line_color="gray", annotation_text=f"edge with every head live {total:.4f}"
    )
    fig.update_layout(
        title=f"{source_label} into {target}, split across layer {record['layer']}'s heads",
        yaxis=dict(title="edge into the target"),
        height=420,
        showlegend=False,
        **_LAYOUT,
    )
    return fig


def conductance_path(points: Sequence[Mapping[str, torch.Tensor]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(alpha, target, d target / d alpha)`` at every path point, from the notebook's conductance trace."""
    alphas = np.concatenate([p["alphas"].numpy() for p in points])
    t = np.concatenate([p["t"].numpy() for p in points])
    dt = np.concatenate([p["dt_dalpha"].numpy() for p in points])
    return alphas, t, dt


def plot_conductance_path(
    points: Sequence[Mapping[str, torch.Tensor]], target: str, units: str = "logits"
) -> go.Figure:
    """The target along the interpolation path, beside the straight line a linearization at the start predicts.

    The left panel is the target in ``units``; the right is its derivative in ``alpha``, in ``units`` per unit
    ``alpha``, whose area over the path is the measured change.
    """
    alphas, t, dt = conductance_path(points)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{target} along the path", "its derivative along the path"))
    fig.add_trace(go.Scatter(x=alphas, y=t, mode="lines+markers", name="measured"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[t[0], t[0] + dt[0]],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="first-order prediction from the start",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=alphas, y=dt, mode="lines+markers", name="d target / d alpha"), row=1, col=2)
    fig.update_xaxes(title_text="alpha (0: first prompt, 1: second)")
    fig.update_yaxes(title_text=f"{target} ({units})", row=1, col=1)
    fig.update_yaxes(title_text=f"d({target}) / d alpha ({units} per unit alpha)", row=1, col=2)
    fig.update_layout(height=420, legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0), **_LAYOUT)
    return fig


def running_head_credit(running: Mapping[int, Sequence[tuple[torch.Tensor, torch.Tensor, float]]]) -> np.ndarray:
    """``(intervals, layers, heads)`` running pattern-plus-value credit per head, from the notebook's trace."""
    layers = sorted(running)
    return np.stack([np.stack([(qk + ov).numpy() for qk, ov, _ in running[layer]]) for layer in layers], axis=1)


def plot_running_credit(
    running: Mapping[int, Sequence[tuple[torch.Tensor, torch.Tensor, float]]], n_steps: int, top: int = 6
) -> go.Figure:
    """How the largest heads' conductance accumulates along the path, one line per head."""
    credit = running_head_credit(running)  # (intervals, layers, heads)
    alphas = np.arange(1, credit.shape[0] + 1) / n_steps
    final = credit[-1]
    order = np.argsort(np.abs(final).ravel())[::-1][:top]
    fig = go.Figure()
    for flat in order:
        layer, head = divmod(int(flat), final.shape[1])
        fig.add_trace(
            go.Scatter(
                x=alphas, y=credit[:, layer, head], mode="lines", name=f"L{layer} h{head} ({final[layer, head]:+.2f})"
            )
        )
    fig.update_layout(
        title="Running head conductance along the path (pattern plus value part)",
        xaxis=dict(title="alpha reached"),
        yaxis=dict(title="credit so far (target units)"),
        height=420,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        **_LAYOUT,
    )
    return fig


def plot_residual_convergence(steps: Sequence[int], residuals: Sequence[float]) -> go.Figure:
    """Largest completeness residual against the step count, with a $1/n^2$ guide through the first point."""
    steps_np, res_np = np.asarray(steps, dtype=float), np.asarray(residuals, dtype=float)
    guide = res_np[0] * (steps_np[0] / steps_np) ** 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps_np, y=res_np, mode="lines+markers", name="largest completeness residual"))
    fig.add_trace(
        go.Scatter(x=steps_np, y=guide, mode="lines", line=dict(dash="dash", color="gray"), name="1/n² guide")
    )
    fig.update_layout(
        title="Conductance completeness residual against the number of path steps",
        xaxis=dict(title="path steps", type="log"),
        yaxis=dict(title="largest residual (fraction of the change)", type="log", exponentformat="power"),
        height=420,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        **_LAYOUT,
    )
    return fig


__all__ = [
    "AlignedRegimes",
    "FeatureContext",
    "ResidualPartition",
    "apply_last_token_edits",
    "capture_residual_terms",
    "influence_by_kind",
    "last_token_edits",
    "node_positions",
    "partition_final_residual",
    "position_change_mass",
    "resolve_feature_context",
    "display_next_token_checks",
    "NextTokenCheck",
    "align_regime_nodes",
    "check_next_token",
    "conductance_path",
    "display_plotly_figure",
    "neumann_term_mass",
    "node_kinds",
    "plot_conductance_path",
    "plot_dose_response",
    "plot_head_heatmap",
    "plot_layer_bars",
    "plot_position_change_mass",
    "plot_head_split",
    "plot_neumann_convergence",
    "plot_regime_edges",
    "plot_residual_convergence",
    "plot_ridgeline",
    "plot_running_credit",
    "print_next_token_checks",
    "running_head_credit",
    "source_layers",
    "top_k_overlap",
]
