"""Drift-summary display helpers for the notebook harness."""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

from IPython.display import HTML, display


def format_score(v: float, *, precision: int = 4) -> str:
    """Format a raw score with automatic scientific notation for small magnitudes.

    Values with ``|v| >= 1e-3`` use fixed-point notation; smaller magnitudes use
    scientific notation so that near-zero attribution scores remain readable.
    """
    if abs(v) >= 1e-3:
        return f"{v:.{precision}f}"
    return f"{v:.{min(precision, 2)}e}"


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


# ---------------------------------------------------------------------------
# Input/output decoupling displays (feature IO profiles)
# ---------------------------------------------------------------------------
