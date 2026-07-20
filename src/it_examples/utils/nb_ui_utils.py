"""Notebook display utilities for Interpretune demos.

Adapted from ``circuit_tracer.utils.demo_utils`` for use within Interpretune
example notebooks.  See the upstream ``demo_utils.py`` for additional display
functions (attribution config, generation comparisons).
"""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

import torch
from IPython.display import HTML, display


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_prob(p: float, *, precision: int = 4) -> str:
    """Format a probability with automatic scientific notation for small values.

    Values >= 1e-3 are shown as percentages; smaller values use scientific notation.
    """
    if p >= 1e-3:
        return f"{p * 100:.{precision}f}%"
    return f"{p:.{min(precision, 2)}e}"


def format_score(v: float, *, precision: int = 4) -> str:
    """Format a raw score with automatic scientific notation for small magnitudes.

    Values with ``|v| >= 1e-3`` use fixed-point notation; smaller magnitudes use
    scientific notation so that near-zero attribution scores remain readable.
    """
    if abs(v) >= 1e-3:
        return f"{v:.{precision}f}"
    return f"{v:.{min(precision, 2)}e}"


def _ensure_1d_logits(logits: torch.Tensor) -> torch.Tensor:
    """Squeeze logits to a 1-D vocab vector, taking the last sequence position if multi-dimensional."""
    if logits.dim() > 1:
        return logits.squeeze(0)[-1].float()
    return logits.float()


def get_topk(
    logits: torch.Tensor,
    tokenizer,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Return the top-*k* ``(token_str, probability)`` pairs.

    Accepts logits of any shape — 1-D ``(vocab,)``, 2-D ``(seq, vocab)``, or 3-D ``(batch, seq, vocab)`` — and always
    resolves to the final position.
    """

    def _token_to_display(token_id: int) -> str:
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            if isinstance(token, list):
                if len(token) == 1:
                    return str(token[0])
                return str(token[-1])
            if token is not None:
                return str(token)
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)

    probs = torch.softmax(_ensure_1d_logits(logits), dim=-1)
    topk = torch.topk(probs, k)
    return [(_token_to_display(int(topk.indices[i])), topk.values[i].item()) for i in range(k)]


# ---------------------------------------------------------------------------
# Candidate example review
# ---------------------------------------------------------------------------


def display_candidate_examples(
    examples: list[dict],
    title: str = "Candidate Examples",
) -> None:
    """Display Phase A candidate examples in a styled HTML table with full prompts.

    Each dict in *examples* should have at least:
    ``batch_idx``, ``prompt``, ``label``, ``predicted``, ``yes_logit``, ``no_logit``,
    ``gap``, ``is_correct``.

    Args:
        examples: List of example dicts from the validation loop.
        title: Heading for the table.
    """
    n_correct = sum(1 for ex in examples if ex["is_correct"])
    n_total = len(examples)
    accuracy_pct = 100 * n_correct / n_total if n_total else 0

    rows = ""
    for i, ex in enumerate(examples):
        mark = "&#x2713;" if ex["is_correct"] else "&#x2717;"
        mark_color = "#27AE60" if ex["is_correct"] else "#C0392B"
        row_bg = "rgba(240,240,240,0.1)" if i % 2 == 0 else "rgba(255,255,255,0.1)"
        rows += (
            f'<tr style="background:{row_bg};">'
            f'<td style="text-align:center;">{ex["batch_idx"]}</td>'
            f"<td>{html.escape(str(ex['label']))}</td>"
            f"<td>{html.escape(str(ex['predicted']))}</td>"
            f'<td style="text-align:right;font-family:monospace;">{ex["gap"]:+.4f}</td>'
            f'<td style="text-align:center;color:{mark_color};font-weight:bold;">{mark}</td>'
            f'<td style="white-space:pre-wrap;word-break:break-word;font-family:monospace;'
            f'font-size:12px;max-width:500px;">{html.escape(str(ex["prompt"]))}</td>'
            f"</tr>\n"
        )

    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;margin-bottom:12px;font-size:13px;">
        <div style="font-weight:bold;font-size:14px;margin-bottom:6px;padding:4px 8px;
            border-radius:3px;background:#555;color:white;display:inline-block;">
            {html.escape(title)}</div>
        <div style="margin-bottom:6px;">
            Accuracy: <b>{n_correct}/{n_total}</b> ({accuracy_pct:.1f}%)
            &mdash; Correct examples available for intervention: <b>{n_correct}</b>
        </div>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="text-align:center;padding:4px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);width:50px;">Batch</th>
                    <th style="text-align:left;padding:4px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);width:60px;">Label</th>
                    <th style="text-align:left;padding:4px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);width:60px;">Pred</th>
                    <th style="text-align:right;padding:4px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);width:90px;">Yes&minus;No</th>
                    <th style="text-align:center;padding:4px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);width:30px;">&#10003;</th>
                    <th style="text-align:left;padding:4px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Prompt</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    display(HTML(markup))


# ---------------------------------------------------------------------------
# Feature display
# ---------------------------------------------------------------------------


def display_top_features_comparison(
    feature_sets: dict[str, list[tuple[int, int, int]]],
    scores_sets: dict[str, list[float]] | None = None,
    neuronpedia_model: str | None = None,
    neuronpedia_set: str = "gemmascope-transcoder-16k",
    neuronpedia_base_url: str = "https://www.neuronpedia.org",
    show_score_sign: bool = False,
) -> None:
    """Display top features from multiple attribution configurations side by side.

    Args:
        feature_sets: Mapping from config label to list of ``(layer, pos, feat_idx)`` tuples.
        scores_sets: Optional mapping from config label to list of attribution scores.
        neuronpedia_model: Neuronpedia model slug (e.g. ``"gemma-2-2b"``).
            When provided, feature indices become clickable links.
        neuronpedia_set: Neuronpedia set name.
        neuronpedia_base_url: Base URL for Neuronpedia feature links.
    """
    labels = list(feature_sets.keys())
    colors = ["#2471A3", "#27AE60", "#8E44AD", "#E67E22", "#C0392B", "#16A085"]
    resolved_base_url = neuronpedia_base_url.rstrip("/")

    style = """
    <style>
    .features-cmp {
        font-family: system-ui, -apple-system, sans-serif;
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 12px;
    }
    .features-cmp .col {
        flex: 1;
        min-width: 220px;
    }
    .features-cmp .col-header {
        font-weight: bold;
        font-size: 14px;
        padding: 4px 8px;
        border-radius: 3px;
        color: white;
        margin-bottom: 6px;
    }
    .features-cmp table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .features-cmp th, .features-cmp td {
        text-align: left;
        padding: 3px 6px;
        border: 1px solid rgba(150,150,150,0.5);
    }
    .features-cmp th {
        background-color: rgba(200,200,200,0.3);
        font-weight: bold;
    }
    .features-cmp .monospace { font-family: monospace; }
    .features-cmp a.np-link {
        color: inherit;
        text-decoration: none;
        border-bottom: 1px dashed rgba(150,150,150,0.6);
    }
    .features-cmp a.np-link:hover {
        color: #2980B9;
        border-bottom-style: solid;
    }
    </style>
    """

    body = '<div class="features-cmp">'
    for i, label in enumerate(labels):
        color = colors[i % len(colors)]
        features = feature_sets[label]
        scores = scores_sets.get(label) if scores_sets else None
        body += '<div class="col">'
        body += f'<div class="col-header" style="background-color: {color};">{html.escape(label)}</div>'
        body += "<table><thead><tr><th>#</th><th>Node</th>"
        if scores is not None:
            if show_score_sign:
                body += "<th>Sign</th><th>|Score|</th>"
            else:
                body += "<th>Score</th>"
        body += "</tr></thead><tbody>"
        for j, (layer, pos, feat_idx) in enumerate(features):
            score_cell = ""
            if scores is not None:
                score_value = float(scores[j])
                if show_score_sign:
                    score_sign = "+" if score_value > 0 else "−" if score_value < 0 else "0"
                    score_cell = f"<td>{score_sign}</td><td>{format_score(abs(score_value))}</td>"
                else:
                    score_cell = f"<td>{format_score(score_value)}</td>"
            if neuronpedia_model is not None:
                np_url = (
                    f"{html.escape(resolved_base_url)}/{html.escape(neuronpedia_model)}/"
                    f"{layer}-{html.escape(neuronpedia_set)}/{feat_idx}"
                )
                feat_link = (
                    f'<a class="np-link" href="{np_url}" target="_blank" title="View on Neuronpedia">{feat_idx}</a>'
                )
            else:
                feat_link = str(feat_idx)
            node_cell = f'<td class="monospace">({layer},&#8239;{pos},&#8239;{feat_link})</td>'
            body += f"<tr><td>{j + 1}</td>{node_cell}{score_cell}</tr>"
        body += "</tbody></table></div>"
    body += "</div>"

    display(HTML(style + body))


# ---------------------------------------------------------------------------
# Token probability display
# ---------------------------------------------------------------------------


def display_token_probs(
    logits: torch.Tensor,
    token_ids: list[int],
    labels: list[str],
    title: str = "",
) -> None:
    """Display softmax probabilities for specific tokens as a styled HTML table.

    Accepts logits of any shape (1-D through 3-D); the last sequence position is
    always used.

    Args:
        logits: Raw logits tensor.
        token_ids: Vocabulary indices to display.
        labels: Human-readable label for each token.
        title: Optional heading rendered above the table.
    """
    logits_1d = _ensure_1d_logits(logits)
    probs = torch.softmax(logits_1d, dim=-1)

    rows = ""
    for i, (tid, label) in enumerate(zip(token_ids, labels)):
        p = probs[tid].item()
        logit_val = logits_1d[tid].item()
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        rows += (
            f'<tr class="{row_class}">'
            f'<td class="monospace">{html.escape(label)}</td>'
            f'<td style="text-align:right;">{format_prob(p, precision=3)}</td>'
            f'<td style="text-align:right;">{logit_val:.4f}</td>'
            f"</tr>\n"
        )

    title_html = (
        f'<div style="font-weight:bold;font-size:14px;margin-bottom:4px;padding:4px 6px;'
        f'border-radius:3px;background:#555;color:white;display:inline-block;">'
        f"{html.escape(title)}</div>"
        if title
        else ""
    )

    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:420px;margin-bottom:10px;font-size:13px;">
        {title_html}
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="text-align:left;padding:3px 6px;
                        border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Token</th>
                    <th style="text-align:right;padding:3px 6px;
                        border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Probability</th>
                    <th style="text-align:right;padding:3px 6px;
                        border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Logit</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    display(HTML(markup))


# ---------------------------------------------------------------------------
# Pre/post intervention comparison
# ---------------------------------------------------------------------------


def display_topk_token_predictions(
    sentence: str,
    original_logits: torch.Tensor,
    new_logits: torch.Tensor,
    tokenizer,
    k: int = 5,
    key_tokens: list[tuple[str, int]] | None = None,
) -> None:
    """Display top-*k* token predictions before and after an intervention.

    Renders side-by-side probability bars, and optionally a *Key Tokens*
    table tracking specific tokens of interest across both distributions.

    Accepts logits of any shape (1-D through 3-D).

    Args:
        sentence: Input prompt string.
        original_logits: Logits before the intervention.
        new_logits: Logits after the intervention.
        tokenizer: Tokenizer for decoding token IDs.
        k: Number of top tokens to show per section.
        key_tokens: Optional list of ``(token_label, token_id)`` pairs for the
            Key Tokens comparison.
    """
    original_tokens = get_topk(original_logits, tokenizer, k)
    new_tokens = get_topk(new_logits, tokenizer, k)

    max_prob = max(
        max(prob for _, prob in original_tokens),
        max(prob for _, prob in new_tokens),
    )

    markup = f"""
    <style>
    .token-viz {{
        font-family: system-ui, -apple-system,
            BlinkMacSystemFont, 'Segoe UI', Roboto,
            Oxygen, Ubuntu, Cantarell, sans-serif;
        margin-bottom: 10px;
        max-width: 700px;
    }}
    .token-viz .header {{
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 3px;
        padding: 4px 6px;
        border-radius: 3px;
        color: white;
        display: inline-block;
    }}
    .token-viz .sentence {{
        background-color: rgba(200, 200, 200, 0.2);
        padding: 4px 6px;
        border-radius: 3px;
        border: 1px solid rgba(100, 100, 100, 0.5);
        font-family: monospace;
        margin-bottom: 8px;
        font-weight: 500;
        font-size: 14px;
    }}
    .token-viz table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 8px;
        font-size: 13px;
        table-layout: fixed;
    }}
    .token-viz th {{
        text-align: left;
        padding: 4px 6px;
        font-weight: bold;
        border: 1px solid rgba(150, 150, 150, 0.5);
        background-color: rgba(200, 200, 200, 0.3);
    }}
    .token-viz td {{
        padding: 3px 6px;
        border: 1px solid rgba(150, 150, 150, 0.5);
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .token-viz .token-col {{ width: 20%; }}
    .token-viz .prob-col {{ width: 15%; }}
    .token-viz .dist-col {{ width: 65%; }}
    .token-viz .monospace {{ font-family: monospace; }}
    .token-viz .bar-container {{ display: flex; align-items: center; }}
    .token-viz .bar {{ height: 12px; min-width: 2px; }}
    .token-viz .bar-text {{ margin-left: 6px; font-weight: 500; font-size: 12px; }}
    .token-viz .even-row {{ background-color: rgba(240, 240, 240, 0.1); }}
    .token-viz .odd-row {{ background-color: rgba(255, 255, 255, 0.1); }}
    </style>

    <div class="token-viz">
        <div class="header" style="background-color: #555555;">Input Sentence:</div>
        <div class="sentence">{html.escape(sentence)}</div>

        <div>
            <div class="header" style="background-color: #2471A3;">Original Top {k} Tokens</div>
            <table>
                <thead>
                    <tr>
                        <th class="token-col">Token</th>
                        <th class="prob-col" style="text-align: right;">Probability</th>
                        <th class="dist-col">Distribution</th>
                    </tr>
                </thead>
                <tbody>
    """

    for i, (token, prob) in enumerate(original_tokens):
        bar_width = int(prob / max_prob * 100)
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        markup += (
            f'<tr class="{row_class}">'
            f'<td class="monospace token-col" title="{html.escape(token)}">{html.escape(token)}</td>'
            f'<td class="prob-col" style="text-align: right;">{format_prob(prob, precision=3)}</td>'
            f'<td class="dist-col"><div class="bar-container">'
            f'<div class="bar" style="background-color: #2471A3; width: {bar_width}%;"></div>'
            f'<span class="bar-text">{prob * 100:.1f}%</span>'
            f"</div></td></tr>\n"
        )

    markup += f"""
                </tbody>
            </table>

            <div class="header" style="background-color: #27AE60;">New Top {k} Tokens</div>
            <table>
                <thead>
                    <tr>
                        <th class="token-col">Token</th>
                        <th class="prob-col" style="text-align: right;">Probability</th>
                        <th class="dist-col">Distribution</th>
                    </tr>
                </thead>
                <tbody>
    """

    for i, (token, prob) in enumerate(new_tokens):
        bar_width = int(prob / max_prob * 100)
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        markup += (
            f'<tr class="{row_class}">'
            f'<td class="monospace token-col" title="{html.escape(token)}">{html.escape(token)}</td>'
            f'<td class="prob-col" style="text-align: right;">{format_prob(prob, precision=3)}</td>'
            f'<td class="dist-col"><div class="bar-container">'
            f'<div class="bar" style="background-color: #27AE60; width: {bar_width}%;"></div>'
            f'<span class="bar-text">{prob * 100:.1f}%</span>'
            f"</div></td></tr>\n"
        )

    markup += """
                </tbody>
            </table>
        </div>
    """

    # Optional key-tokens section
    if key_tokens:
        orig_probs = torch.softmax(_ensure_1d_logits(original_logits), dim=-1)
        new_probs = torch.softmax(_ensure_1d_logits(new_logits), dim=-1)

        # Sort key tokens by influence: most positive change first, most negative last
        key_token_data = []
        for label, tid in key_tokens:
            p_orig = orig_probs[tid].item()
            p_new = new_probs[tid].item()
            change = p_new - p_orig
            key_token_data.append((label, tid, p_orig, p_new, change))
        key_token_data.sort(key=lambda x: x[4], reverse=True)
        max_change = max((abs(change) for *_prefix, change in key_token_data), default=0.0)

        markup += """
        <div>
            <div class="header" style="background-color: #8E44AD;">Key Tokens</div>
            <table>
                <thead>
                    <tr>
                        <th class="token-col">Token</th>
                        <th class="prob-col" style="text-align: right;">Original</th>
                        <th class="prob-col" style="text-align: right;">New</th>
                        <th class="dist-col">Change</th>
                    </tr>
                </thead>
                <tbody>
        """
        for i, (label, tid, p_orig, p_new, change) in enumerate(key_token_data):
            sign = "+" if change >= 0 else ""
            bar_color = "#27AE60" if change >= 0 else "#C0392B"
            bar_width = 0 if max_change <= 0 else int(abs(change) / max(max_change, 1e-9) * 100)
            row_class = "even-row" if i % 2 == 0 else "odd-row"
            markup += (
                f'<tr class="{row_class}">'
                f'<td class="monospace token-col" title="{html.escape(label)}">{html.escape(label)}</td>'
                f'<td class="prob-col" style="text-align: right;">{format_prob(p_orig)}</td>'
                f'<td class="prob-col" style="text-align: right;">{format_prob(p_new)}</td>'
                f'<td class="dist-col"><div class="bar-container">'
                f'<div class="bar" style="background-color: {bar_color}; width: {bar_width}%;"></div>'
                f'<span class="bar-text">{sign}{format_score(abs(change))}</span>'
                f"</div></td></tr>\n"
            )
        markup += """
                </tbody>
            </table>
        </div>
        """

    markup += """
    </div>
    """

    display(HTML(markup))


# ---------------------------------------------------------------------------
# Ablation chart
# ---------------------------------------------------------------------------


def display_ablation_chart(
    groups: dict[str, dict[str, float]],
    logit_diffs: dict[str, float] | None = None,
    title: str = "",
    colors: list[str] | None = None,
) -> None:
    """Display ablation results as a grouped bar chart with optional logit-difference line.

    Adapted from ``circuit_tracer.utils.demo_utils.display_ablation_chart``.

    Args:
        groups: Mapping from condition label (e.g. ``"baseline"``, ``"top-10"``)
            to a dict of ``{token_label: probability}``.
        logit_diffs: Optional mapping from condition label to a scalar logit
            difference value.  When provided, a dashed line overlay is drawn on
            a secondary y-axis.
        title: Chart title.
        colors: Optional list of bar colors (one per token label).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Close any leaked figures to avoid corrupted renderer state.
    plt.close("all")

    group_labels = list(groups.keys())
    token_labels = list(next(iter(groups.values())).keys())
    n_groups = len(group_labels)
    n_tokens = len(token_labels)

    if colors is None:
        colors = ["#2471A3", "#E67E22", "#27AE60", "#C0392B", "#8E44AD"][:n_tokens]

    def _safe_probability(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    x = np.arange(n_groups)
    width = 0.8 / n_tokens

    fig, ax1 = plt.subplots(figsize=(8, 5.0))

    for i, tok in enumerate(token_labels):
        vals = [_safe_probability(groups[g].get(tok, 0.0)) for g in group_labels]
        offset = (i - (n_tokens - 1) / 2) * width
        bars = ax1.bar(x + offset, vals, width * 0.9, label=tok, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.set_ylabel("Probability")
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels)
    max_prob = max(max(_safe_probability(groups[g].get(t, 0.0)) for t in token_labels) for g in group_labels)
    ax1.set_ylim(0, max(0.05, max_prob * 1.4))

    if logit_diffs is not None:
        ax2 = ax1.twinx()
        diff_vals = [
            float(logit_diffs.get(g, 0.0)) if np.isfinite(logit_diffs.get(g, 0.0)) else 0.0 for g in group_labels
        ]
        ax2.plot(x, diff_vals, "k--o", label="Logit diff", linewidth=1.5, markersize=5)
        ax2.set_ylabel("Logit difference")
        ax2.legend(loc="upper right")

    ax1.legend(loc="upper left")
    if title:
        ax1.set_title(title, fontsize=13, fontweight="bold")
    try:
        fig.tight_layout()
    except Exception:
        fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.18)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Key-token logit table
# ---------------------------------------------------------------------------


def display_key_token_logits(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    token_ids: list[int],
    token_labels: list[str],
    title: str = "Key-Token Logit Analysis",
) -> None:
    """Display a styled HTML table of per-token logit changes before/after intervention.

    Accepts 1-D logit vectors (vocab dimension).

    Args:
        pre_logits: Logits before intervention (1-D, vocab size).
        post_logits: Logits after intervention (1-D, vocab size).
        token_ids: Vocabulary indices to display.
        token_labels: Human-readable label for each token.
        title: Heading rendered above the table.
    """
    pre = pre_logits.float().cpu()
    post = post_logits.float().cpu()

    # Compute ranks
    _, pre_sorted = torch.sort(pre, descending=True)
    _, post_sorted = torch.sort(post, descending=True)
    pre_rank = {int(idx): r for r, idx in enumerate(pre_sorted.tolist())}
    post_rank = {int(idx): r for r, idx in enumerate(post_sorted.tolist())}

    # Sort by delta: most positive first, most negative last
    token_data = []
    for tid, label in zip(token_ids, token_labels):
        pre_v = pre[tid].item()
        post_v = post[tid].item()
        delta = post_v - pre_v
        pr = pre_rank.get(tid, -1)
        por = post_rank.get(tid, -1)
        token_data.append((tid, label, pre_v, post_v, delta, pr, por))
    token_data.sort(key=lambda x: x[4], reverse=True)

    rows = ""
    for i, (tid, label, pre_v, post_v, delta, pr, por) in enumerate(token_data):
        delta_color = "#27AE60" if delta > 0 else "#C0392B" if delta < 0 else "#888"
        row_bg = "rgba(240,240,240,0.1)" if i % 2 == 0 else "rgba(255,255,255,0.1)"
        rows += (
            f'<tr style="background:{row_bg};">'
            f'<td style="font-family:monospace;">{html.escape(label)}</td>'
            f'<td style="text-align:right;">{pre_v:.4f}</td>'
            f'<td style="text-align:center;">{pr}</td>'
            f'<td style="text-align:right;">{post_v:.4f}</td>'
            f'<td style="text-align:center;">{por}</td>'
            f'<td style="text-align:right;color:{delta_color};font-weight:bold;">{delta:+.4f}</td>'
            f"</tr>\n"
        )

    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:650px;margin-bottom:12px;font-size:13px;">
        <div style="font-weight:bold;font-size:14px;margin-bottom:6px;padding:4px 8px;
            border-radius:3px;background:#555;color:white;display:inline-block;">
            {html.escape(title)}</div>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="text-align:left;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Token</th>
                    <th style="text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Pre Logit</th>
                    <th style="text-align:center;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Rank</th>
                    <th style="text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Post Logit</th>
                    <th style="text-align:center;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Rank</th>
                    <th style="text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Δ Logit</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    display(HTML(markup))


def _summary_value(item: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key)


def display_layer_divergence_summary(
    layer_summaries: Sequence[Mapping[str, Any] | Any],
    title: str = "Retained-Feature Divergence by Layer",
) -> None:
    """Display per-layer retained-feature drift summaries as an HTML table."""

    rows = ""

    def _layer_sort_key(summary: Mapping[str, Any] | Any) -> tuple[int, int | str]:
        layer = _summary_value(summary, "layer")
        try:
            return (1, int(layer))
        except (TypeError, ValueError):
            return (0, str(layer))

    for index, summary in enumerate(sorted(layer_summaries, key=_layer_sort_key, reverse=True)):
        layer = _summary_value(summary, "layer")
        divergent_feature_count = _summary_value(summary, "divergent_feature_count")
        total_feature_count = _summary_value(summary, "total_feature_count")
        max_abs_error = float(_summary_value(summary, "max_abs_error") or 0.0)
        mean_abs_error = float(_summary_value(summary, "mean_abs_error") or 0.0)
        expected_abs_delta_sum = float(_summary_value(summary, "expected_abs_delta_sum") or 0.0)
        actual_abs_delta_sum = float(_summary_value(summary, "actual_abs_delta_sum") or 0.0)
        top_error_feature_row = _summary_value(summary, "top_error_feature_row")
        top_error_display = "-" if not top_error_feature_row else html.escape(str(tuple(top_error_feature_row)))
        row_bg = "rgba(240,240,240,0.1)" if index % 2 == 0 else "rgba(255,255,255,0.1)"
        rows += (
            f'<tr style="background:{row_bg};">'
            f'<td style="text-align:center;">{layer}</td>'
            f'<td style="text-align:center;">{divergent_feature_count}</td>'
            f'<td style="text-align:center;">{total_feature_count}</td>'
            f'<td style="text-align:right;">{format_score(max_abs_error)}</td>'
            f'<td style="text-align:right;">{format_score(mean_abs_error)}</td>'
            f'<td style="text-align:right;">{format_score(expected_abs_delta_sum)}</td>'
            f'<td style="text-align:right;">{format_score(actual_abs_delta_sum)}</td>'
            f'<td style="font-family:monospace;">{top_error_display}</td>'
            f"</tr>\n"
        )

    center_header_style = (
        "text-align:center;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);background:rgba(200,200,200,0.3);"
    )
    right_header_style = (
        "text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);background:rgba(200,200,200,0.3);"
    )
    left_header_style = (
        "text-align:left;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);background:rgba(200,200,200,0.3);"
    )

    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:980px;margin-bottom:14px;font-size:13px;">
        <div style="font-weight:bold;font-size:14px;margin-bottom:6px;padding:4px 8px;
            border-radius:3px;background:#555;color:white;display:inline-block;">
            {html.escape(title)}</div>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="{center_header_style}">Layer</th>
                    <th style="{center_header_style}">Diverged</th>
                    <th style="{center_header_style}">Total</th>
                    <th style="{right_header_style}">Max |Error|</th>
                    <th style="{right_header_style}">Mean |Error|</th>
                    <th style="{right_header_style}">Σ|Expected Δ|</th>
                    <th style="{right_header_style}">Σ|Actual Δ|</th>
                    <th style="{left_header_style}">Top Error Row</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    display(HTML(markup))


def display_logit_drift_summary(
    logit_summary: Mapping[str, Any] | Any,
    title: str = "Tracked Graph-Logit Divergence",
) -> None:
    """Display tracked logit drift details as an HTML table."""

    top_errors = _summary_value(logit_summary, "top_errors") or []
    header = (
        f"Diverged: {_summary_value(logit_summary, 'divergent_logit_count')} / "
        f"{_summary_value(logit_summary, 'total_logit_count')}"
        f" | Max |Error|: {format_score(float(_summary_value(logit_summary, 'max_abs_error') or 0.0))}"
    )
    rows = ""
    for index, error in enumerate(top_errors):
        token_label = _summary_value(error, "token_label") or ""
        token_id = _summary_value(error, "token_id")
        actual_delta = float(_summary_value(error, "actual_delta") or 0.0)
        expected_delta = float(_summary_value(error, "expected_delta") or 0.0)
        abs_error = float(_summary_value(error, "abs_error") or 0.0)
        row_bg = "rgba(240,240,240,0.1)" if index % 2 == 0 else "rgba(255,255,255,0.1)"
        rows += (
            f'<tr style="background:{row_bg};">'
            f'<td style="font-family:monospace;">{html.escape(str(token_label))}</td>'
            f'<td style="text-align:center;">{token_id}</td>'
            f'<td style="text-align:right;">{format_score(expected_delta)}</td>'
            f'<td style="text-align:right;">{format_score(actual_delta)}</td>'
            f'<td style="text-align:right;">{format_score(abs_error)}</td>'
            f"</tr>\n"
        )

    center_header_style = (
        "text-align:center;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);background:rgba(200,200,200,0.3);"
    )
    right_header_style = (
        "text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);background:rgba(200,200,200,0.3);"
    )
    left_header_style = (
        "text-align:left;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);background:rgba(200,200,200,0.3);"
    )

    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:760px;margin-bottom:14px;font-size:13px;">
        <div style="font-weight:bold;font-size:14px;margin-bottom:6px;padding:4px 8px;
            border-radius:3px;background:#555;color:white;display:inline-block;">
            {html.escape(title)}</div>
        <div style="margin:4px 0 8px 2px;color:#444;">{html.escape(header)}</div>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="{left_header_style}">Token</th>
                    <th style="{center_header_style}">Token ID</th>
                    <th style="{right_header_style}">Expected Δ</th>
                    <th style="{right_header_style}">Actual Δ</th>
                    <th style="{right_header_style}">|Error|</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    display(HTML(markup))
