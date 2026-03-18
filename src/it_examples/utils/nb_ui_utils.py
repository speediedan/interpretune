"""Notebook display utilities for Interpretune demos.

Adapted from ``circuit_tracer.utils.demo_utils`` for use within Interpretune
example notebooks.  See the upstream ``demo_utils.py`` for additional display
functions (attribution config, ablation charts, generation comparisons).
"""

from __future__ import annotations

import html

import torch
from IPython.display import HTML, display


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    probs = torch.softmax(_ensure_1d_logits(logits), dim=-1)
    topk = torch.topk(probs, k)
    return [(tokenizer.decode([topk.indices[i]]), topk.values[i].item()) for i in range(k)]


# ---------------------------------------------------------------------------
# Feature display
# ---------------------------------------------------------------------------


def display_top_features_comparison(
    feature_sets: dict[str, list[tuple[int, int, int]]],
    scores_sets: dict[str, list[float]] | None = None,
    neuronpedia_model: str | None = None,
    neuronpedia_set: str = "gemmascope-transcoder-16k",
) -> None:
    """Display top features from multiple attribution configurations side by side.

    Args:
        feature_sets: Mapping from config label to list of ``(layer, pos, feat_idx)`` tuples.
        scores_sets: Optional mapping from config label to list of attribution scores.
        neuronpedia_model: Neuronpedia model slug (e.g. ``"gemma-2-2b"``).
            When provided, feature indices become clickable links.
        neuronpedia_set: Neuronpedia set name.
    """
    labels = list(feature_sets.keys())
    colors = ["#2471A3", "#27AE60", "#8E44AD", "#E67E22", "#C0392B", "#16A085"]

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
            body += "<th>Score</th>"
        body += "</tr></thead><tbody>"
        for j, (layer, pos, feat_idx) in enumerate(features):
            score_cell = f"<td>{scores[j]:.4f}</td>" if scores is not None else ""
            if neuronpedia_model is not None:
                np_url = (
                    f"https://www.neuronpedia.org/{html.escape(neuronpedia_model)}/"
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

    def _fmt(p: float) -> str:
        return f"{p * 100:.3f}%" if p >= 1e-3 else f"{p:.2e}"

    rows = ""
    for i, (tid, label) in enumerate(zip(token_ids, labels)):
        p = probs[tid].item()
        logit_val = logits_1d[tid].item()
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        rows += (
            f'<tr class="{row_class}">'
            f'<td class="monospace">{html.escape(label)}</td>'
            f'<td style="text-align:right;">{_fmt(p)}</td>'
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
            f'<td class="prob-col" style="text-align: right;">{prob:.3f}</td>'
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
            f'<td class="prob-col" style="text-align: right;">{prob:.3f}</td>'
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
        for i, (label, tid) in enumerate(key_tokens):
            p_orig = orig_probs[tid].item()
            p_new = new_probs[tid].item()
            relative = (p_new - p_orig) / max(p_orig, 1e-9)
            sign = "+" if relative >= 0 else ""
            bar_width = int(p_new / max(max_prob, 1e-9) * 100)
            row_class = "even-row" if i % 2 == 0 else "odd-row"
            markup += (
                f'<tr class="{row_class}">'
                f'<td class="monospace token-col" title="{html.escape(label)}">{html.escape(label)}</td>'
                f'<td class="prob-col" style="text-align: right;">{p_orig:.4f}</td>'
                f'<td class="prob-col" style="text-align: right;">{p_new:.4f}</td>'
                f'<td class="dist-col"><div class="bar-container">'
                f'<div class="bar" style="background-color: #8E44AD; width: {bar_width}%;"></div>'
                f'<span class="bar-text">{sign}{relative * 100:.1f}%</span>'
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
