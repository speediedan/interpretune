"""Notebook display utilities that a single example tutorial uses.

The displays several tutorials share live in :mod:`interpretune.harness.notebook_ui` and are re-exported here so the
notebooks keep their imports until they move to the examples repository. Adapted from
``circuit_tracer.utils.demo_utils``.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence
from urllib.parse import urlsplit

import torch
from IPython.display import HTML, display
from interpretune.harness.notebook_ui import (  # noqa: F401 - promoted; re-exported for the notebooks until they move
    display_target_gap,
    resolve_feature_explanations,
    display_top_features_comparison,
    display_steering_results,
    SteeringDisplaySummary,
    best_variant_token_ids,
    display_concept_positions,
    display_feature_decoupling_table,
    plot_decoder_projection_map,
    display_jlens_readout_comparison,
    build_jlens_readout_comparison_html,
    JlensReadoutComparisonSummary,
)

# what the single-tutorial displays below build on
from interpretune.harness.display import format_score
from interpretune.harness.notebook_ui import _ensure_1d_logits, _plotly_figure_html, format_prob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NotebookLink:
    """A labelled HTTP(S) link in a notebook table; both label and URL are escaped on rendering."""

    label: str
    url: str


def build_notebook_table_html(
    rows: Sequence[Sequence[Any]],
    columns: Sequence[str],
    *,
    title: str,
    caption: str = "",
) -> str:
    """Build a scrollable, accessible report table from already formatted cells.

    Callers own numerical precision and units, so the same helper can show probabilities, logits, checks or
    configuration. Ordinary cells are escaped text, never HTML; use :class:`NotebookLink` for a feature reference.
    A caption can state the denominator or limitation alongside the measurement. No notebook kernel is required.
    """

    def cell(value: Any) -> str:
        if isinstance(value, NotebookLink):
            parsed = urlsplit(value.url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("notebook table links require an absolute HTTP(S) URL")
            return f'<a href="{html.escape(value.url, quote=True)}">{html.escape(value.label)}</a>'
        return html.escape(str(value))

    if not columns:
        raise ValueError("a notebook table needs at least one column")
    if any(len(row) != len(columns) for row in rows):
        raise ValueError("every notebook table row must match the column count")
    header = "".join(f'<th scope="col">{html.escape(c)}</th>' for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>" for row in rows)
    return f"""<style>
    .it-report {{font-family:system-ui,sans-serif; font-size:14px; margin:12px 0; overflow-x:auto;}}
    .it-report table {{border-collapse:collapse; width:max-content; min-width:55%; max-width:none;}}
    .it-report caption {{text-align:left; caption-side:top; padding:8px 0; color:inherit;}}
    .it-report th {{background:#2471a3; color:white; text-align:left;}}
    .it-report th,.it-report td {{padding:7px 10px; border:1px solid #aaa6; vertical-align:top;}}
    .it-report tr:nth-child(even) {{background:#78b4f018;}}
    .it-report td {{font-variant-numeric:tabular-nums; max-width:36em; white-space:pre-wrap;}}
    </style><div class="it-report"><table>
    <caption><strong>{html.escape(title)}</strong><br>{html.escape(caption)}</caption>
    <thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"""


def display_notebook_table(
    rows: Sequence[Sequence[Any]], columns: Sequence[str], *, title: str, caption: str = ""
) -> None:
    """Display :func:`build_notebook_table_html` in a notebook or its rendered documentation."""
    display(HTML(build_notebook_table_html(rows, columns, title=title, caption=caption)))


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


# ---------------------------------------------------------------------------
# Feature display
# ---------------------------------------------------------------------------


class JlensDirectionValidationSummary(NamedTuple):
    """What :func:`display_jlens_direction_validation` rendered, for assertions without re-deriving it."""

    direction_labels: tuple[str, ...]
    direction_shares: tuple[float, ...]
    direction_fractions: tuple[float, ...]
    attribution_total: float
    predicted_delta: float
    unexplained_remainder: float


def build_jlens_direction_validation_html(
    attribution: Mapping[str, Any],
    direction_labels: Sequence[str],
    *,
    finite_difference_slopes: Sequence[float] | None = None,
    measured_delta: float | None = None,
    title: str = "J-lens direction validation",
) -> tuple[str, JlensDirectionValidationSummary]:
    """Build the J-lens direction validation table as HTML; :func:`display_jlens_direction_validation` shows it.

    One row per J-lens direction from ``attribution`` (the dict
    :func:`~interpretune.analysis.ops.bundled.jlens.jlens_ops.subspace_attribution_scores` returns), with the
    coordinate change ``Δc_i``, the gradient readout ``w_i``, their product ``a_i``, the direction's fraction of
    the explained mass and, when given, an independent central-difference slope; a footer row each for the
    explained total, the remainder, the first-order prediction ``gᵀΔh`` and, when given, the measured logit change.

    This validates the J-lens directions against the edit they produced. It deliberately carries no feature
    column: SAE-feature influence and J-lens gap shares are different vocabularies, and setting them side by side
    implies a comparison this machinery cannot yet make. Pure (no display), so a test can check the markup and the
    summary without a kernel.
    """
    shares = [float(x) for x in attribution["attribution_shares"]]
    labels = [str(lbl) for lbl in direction_labels]
    if len(labels) != len(shares):
        raise ValueError(f"{len(labels)} direction labels for {len(shares)} attribution shares")
    delta_coords = [float(x) for x in attribution.get("delta_coords") or [float("nan")] * len(shares)]
    readouts = [float(x) for x in attribution.get("readouts") or [float("nan")] * len(shares)]
    slopes = [float(x) for x in finite_difference_slopes] if finite_difference_slopes is not None else None
    if slopes is not None and len(slopes) != len(shares):
        raise ValueError(f"{len(slopes)} finite-difference slopes for {len(shares)} attribution shares")
    total = float(attribution["attribution_total"])
    predicted = float(attribution["predicted_delta"])
    remainder = float(attribution["unexplained_remainder"])
    share_mass = sum(abs(a) for a in shares)
    fractions = [abs(a) / share_mass if share_mass else 0.0 for a in shares]

    # Sized to its content and scrolling inside the widget when the page is narrower, so the docs theme never
    # squeezes the table into its own box or scrolls the page sideways.
    style = """
    <style>
    .jl-val { font-family: system-ui, -apple-system, sans-serif; margin-bottom: 12px; font-size: 13px; }
    .jl-val .title { font-weight: bold; font-size: 14px; margin-bottom: 6px; padding: 4px 6px;
        border-radius: 3px; background: #2471A3; color: white; display: inline-block; }
    .jl-val .scroll { overflow-x: auto; padding-bottom: 6px; }
    .jl-val table { width: max-content; max-width: none; border-collapse: collapse; }
    .jl-val th, .jl-val td { padding: 3px 6px; border: 1px solid rgba(150,150,150,0.5); text-align: right; }
    .jl-val td { white-space: nowrap; }
    .jl-val th { white-space: normal; }
    .jl-val td.lbl, .jl-val th.lbl { text-align: left; }
    .jl-val tr.total td { background: rgba(120,180,240,0.15); font-weight: bold; }
    .jl-val .footnote { font-size: 12px; color: #666; margin-top: 6px; max-width: 900px; }
    </style>
    """

    def _signed(v: float, precision: int = 4) -> str:
        return "n/a" if v != v else f"{v:+.{precision}f}"  # NaN when a factor was not supplied

    # Headers are short because the table is sized to its content: a header long enough to want wrapping simply
    # widens the table instead. The symbols are spelled out in the legend.
    slope_header = "<th>dGap/ds</th>" if slopes is not None else ""
    table = (
        '<table><thead><tr><th class="lbl">Direction</th><th>&#916;c</th><th>Readout w</th>'
        f"<th>Share a</th><th>Fraction</th>{slope_header}</tr></thead><tbody>"
    )
    for i, lbl in enumerate(labels):
        slope_cell = f"<td>{_signed(slopes[i])}</td>" if slopes is not None else ""
        table += (
            f'<tr><td class="lbl">{html.escape(lbl)}</td><td>{_signed(delta_coords[i])}</td>'
            f"<td>{_signed(readouts[i])}</td><td>{shares[i]:+.4f}</td><td>{fractions[i]:.1%}</td>{slope_cell}</tr>"
        )
    # A footer value is a share-space quantity, so it must not land under the slope header, where it would read as
    # a slope: one left-aligned cell spanning every column bar the label keeps it beside its label and out of
    # every numeric column. Six columns with the slope column present, five without.
    span = 5 if slopes is not None else 4

    def _footer(label: str, value: float) -> str:
        return f'<tr class="total"><td class="lbl">{label}</td><td class="lbl" colspan="{span}">{value:+.4f}</td></tr>'

    table += _footer("Explained &#931;a", total)
    table += _footer("Remainder", remainder)
    table += _footer("First-order g&#7488;&#916;h", predicted)
    if measured_delta is not None:
        table += _footer("Measured &#916;gap", float(measured_delta))
    table += "</tbody></table>"

    # The legend defines only the columns actually rendered: naming a column the reader cannot see is the same
    # defect in prose that an empty column would be in the table.
    legend = (
        "&#916;c = V&#8314;&#916;h, the displacement's coordinates in the pair; w = V&#7488;g, the gradient's "
        "readout of each direction; a = w&#183;&#916;c, the direction's share of the first-order prediction "
        "g&#7488;&#916;h. Fraction is |a| over &#931;|a|."
    )
    if slopes is not None:
        legend += " dGap/ds is the central-difference slope of the gap along that direction, an independent check on w."
    footnote = (
        f"{legend} The first-order prediction is the gradient's linear estimate of the patch's effect; the measured "
        "change includes everything nonlinear downstream of the site."
    )
    markup = (
        f'{style}<div class="jl-val"><div class="title">{html.escape(title)}</div>'
        f'<div class="scroll">{table}</div><div class="footnote">{footnote}</div></div>'
    )
    summary = JlensDirectionValidationSummary(
        direction_labels=tuple(labels),
        direction_shares=tuple(shares),
        direction_fractions=tuple(fractions),
        attribution_total=total,
        predicted_delta=predicted,
        unexplained_remainder=remainder,
    )
    return markup, summary


def display_jlens_direction_validation(
    attribution: Mapping[str, Any], direction_labels: Sequence[str], **kwargs: Any
) -> JlensDirectionValidationSummary:
    """Render :func:`build_jlens_direction_validation_html` and return its summary (see that function for the
    columns)."""
    markup, summary = build_jlens_direction_validation_html(attribution, direction_labels, **kwargs)
    display(HTML(markup))
    return summary


# ---------------------------------------------------------------------------
# Token probability display
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Key-token logit table
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Input/output decoupling displays (feature IO profiles)
# ---------------------------------------------------------------------------


def display_plotly_figure(fig: Any, id_prefix: str = "it-figure") -> None:
    """Show a plotly figure so it renders on the docs site as well as in a live notebook.

    ``fig.show()`` writes only the ``application/vnd.plotly.v1+json`` output, which Jupyter renders and the docs build
    drops without a warning, so a figure shown that way is missing from every rendered page. This embeds the figure as
    HTML instead (see ``_plotly_figure_html``), which both render.
    """
    display(HTML(_plotly_figure_html(fig, id_prefix)))


def build_steering_scale_sweep_html(points: Sequence[Any], token_labels: Sequence[str]) -> str:
    """Build the per-arm, per-scale table of logit and probability shifts for the two target tokens.

    ``points`` are :class:`~it_examples.utils.steering_demo_helpers.SteeringScalePoint` rows. Each shift is signed
    and coloured by the repository convention (green positive, red negative), and the last column says whether the
    answer flipped, which is what the chart above it shows. Pure (no display), so a test can check the markup.
    """
    if len(token_labels) != 2:
        raise ValueError(f"expected two target token labels, got {len(token_labels)}")
    a, b = (html.escape(str(t)) for t in token_labels)

    def _signed(v: float, fmt: str) -> str:
        colour = "#1a7f37" if v > 0 else ("#d1242f" if v < 0 else "inherit")
        return f'<td style="color:{colour};font-weight:600">{v:{fmt}}</td>'

    rows = ""
    for p in points:
        # A strict sign change: a gap that lands exactly on zero is a tie, not a flip.
        flipped = p.pre_gap * p.post_gap < 0
        rows += (
            f'<tr><td class="lbl">{html.escape(str(p.arm))}</td><td>{p.scale:g}</td>'
            + _signed(p.post_logits[0] - p.pre_logits[0], "+.3f")
            + _signed(p.post_logits[1] - p.pre_logits[1], "+.3f")
            + _signed(p.post_gap - p.pre_gap, "+.3f")
            + _signed(100 * (p.post_probs[0] - p.pre_probs[0]), "+.2f")
            + _signed(100 * (p.post_probs[1] - p.pre_probs[1]), "+.2f")
            + f"<td>{'yes' if flipped else 'no'}</td></tr>"
        )
    style = """
    <style>
    .sweep { font-family: system-ui, -apple-system, sans-serif; font-size: 13px; margin-bottom: 12px; }
    .sweep .scroll { overflow-x: auto; }
    .sweep table { width: max-content; border-collapse: collapse; }
    .sweep th, .sweep td { padding: 3px 8px; border: 1px solid rgba(150,150,150,0.5); text-align: right; }
    .sweep td.lbl, .sweep th.lbl { text-align: left; }
    .sweep .footnote { font-size: 12px; color: #666; margin-top: 6px; max-width: 900px; }
    </style>
    """
    header = (
        '<tr><th class="lbl">Arm</th><th>Scale</th>'
        f"<th>&#916; logit {a}</th><th>&#916; logit {b}</th><th>&#916; gap ({a} &#8722; {b})</th>"
        f"<th>&#916; prob {a} (pp)</th><th>&#916; prob {b} (pp)</th><th>Answer flipped</th></tr>"
    )
    footnote = (
        "Shifts are post minus pre at the answer position; probabilities are over the full vocabulary, in percentage "
        "points. Scales are each arm's own multiplier, not matched displacements across arms."
    )
    return (
        f'{style}<div class="sweep"><div class="scroll"><table><thead>{header}</thead><tbody>{rows}</tbody>'
        f'</table></div><div class="footnote">{footnote}</div></div>'
    )


def display_steering_scale_sweep(points: Sequence[Any], token_labels: Sequence[str], title: str) -> None:
    """Chart each arm's post-intervention gap against scale, with the answer-flip line, then show the shift table.

    The chart plots the gap after intervention rather than its change, so the flip threshold is one line at zero shared
    by every arm (all arms start from the same clean gap), and the smallest scale at which each arm crosses it is read
    off the chart rather than asserted in prose.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[plotly unavailable] scale-sweep chart skipped; the table below carries the same numbers")
    else:
        fig = go.Figure()
        for arm in dict.fromkeys(p.arm for p in points):
            arm_points = sorted((p for p in points if p.arm == arm), key=lambda p: p.scale)
            fig.add_trace(
                go.Scatter(
                    x=[p.scale for p in arm_points], y=[p.post_gap for p in arm_points], mode="lines+markers", name=arm
                )
            )
        fig.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"answer flips above ({token_labels[0]} > {token_labels[1]})",
            annotation_position="top left",
        )
        fig.update_layout(
            title=title,
            autosize=True,
            height=420,
            template="simple_white",
            xaxis=dict(title="intervention scale (each arm's own multiplier)", type="log"),
            yaxis=dict(title=f"gap after intervention ({token_labels[0]} − {token_labels[1]}, logits)"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        )
        display(HTML(_plotly_figure_html(fig, "it-scale-sweep")))
    display(HTML(build_steering_scale_sweep_html(points, token_labels)))
