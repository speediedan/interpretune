"""Helpers for the attention-aware attribution notebook: a prompt sanity check, regime comparisons, and figures.

The figures read either an analysis batch the notebook already holds (an attribution graph's adjacency matrix, a head
report) or a trace the notebook's analysis points recorded inside circuit-tracer. They are shown through
``display_plotly_figure`` so they render on the docs site as well as in a live notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from it_examples.utils.nb_ui_utils import display_plotly_figure

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
) -> go.Figure:
    """Scatter every source's edge into one target, frozen regime against live, one colour per node kind.

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
                text=[labels[int(i)] for i in idx],
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


def plot_conductance_path(points: Sequence[Mapping[str, torch.Tensor]], target: str) -> go.Figure:
    """The target along the interpolation path, beside the straight line a linearization at the start predicts."""
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
    "NextTokenCheck",
    "align_regime_nodes",
    "check_next_token",
    "conductance_path",
    "display_plotly_figure",
    "neumann_term_mass",
    "node_kinds",
    "plot_conductance_path",
    "plot_head_heatmap",
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
