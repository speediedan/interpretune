"""Notebook display utilities for Interpretune demos.

Adapted from ``circuit_tracer.utils.demo_utils`` for use within Interpretune
example notebooks.  See the upstream ``demo_utils.py`` for additional display
functions (attribution config, generation comparisons).
"""

from __future__ import annotations

import html
from typing import Any, Mapping, NamedTuple, Sequence

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


def resolve_feature_explanations(
    *,
    model_id: str,
    source_set: str,
    feature_tuples: Sequence[tuple[int, int]],
    base_url: str = "https://www.neuronpedia.org",
    preferred_type_name: str | None = None,
    timeout_seconds: int = 15,
) -> dict[tuple[int, int], str]:
    """Best-effort explanation text for ``(layer, feature_index)`` tuples via the feature API.

    Works against both public neuronpedia.org and a local Neuronpedia dev webapp ``base_url``.
    Fetch failures and features without explanations simply yield no entry, so callers can
    always render (an empty cell) without guarding.
    """
    from interpretune.utils.neuronpedia_explanations import feature_tuples_to_feature_refs, fetch_feature_payload

    unique_tuples = list(dict.fromkeys((int(layer), int(feat)) for layer, feat in feature_tuples))
    if not unique_tuples:
        return {}
    feature_refs = feature_tuples_to_feature_refs(
        model_id=model_id,
        source_set=source_set,
        feature_tuples=unique_tuples,
        base_url=base_url,
    )
    explanations: dict[tuple[int, int], str] = {}
    for feature_tuple, feature_ref in zip(unique_tuples, feature_refs):
        try:
            payload = fetch_feature_payload(feature_ref, timeout_seconds=timeout_seconds)
        except Exception:
            continue
        rows = payload.get("explanations")
        if not isinstance(rows, list):
            continue
        candidate_rows = [row for row in rows if isinstance(row, Mapping)]
        if preferred_type_name:
            preferred_rows = [
                row for row in candidate_rows if str(row.get("typeName") or "").strip() == preferred_type_name
            ]
            candidate_rows = preferred_rows or candidate_rows
        for row in candidate_rows:
            description = row.get("description")
            if isinstance(description, str) and description.strip():
                explanations[feature_tuple] = " ".join(description.split())
                break
    return explanations


def display_top_features_comparison(
    feature_sets: dict[str, list[tuple[int, int, int]]],
    scores_sets: dict[str, list[float]] | None = None,
    neuronpedia_model: str | None = None,
    neuronpedia_set: str = "gemmascope-transcoder-16k",
    neuronpedia_base_url: str = "https://www.neuronpedia.org",
    show_score_sign: bool = False,
    feature_explanations: Mapping[tuple[int, int], str] | None = None,
) -> None:
    """Display top features from multiple attribution configurations side by side.

    Args:
        feature_sets: Mapping from config label to list of ``(layer, pos, feat_idx)`` tuples.
        scores_sets: Optional mapping from config label to list of attribution scores.
        neuronpedia_model: Neuronpedia model slug (e.g. ``"gemma-2-2b"``).
            When provided, feature indices become clickable links.
        neuronpedia_set: Neuronpedia set name.
        neuronpedia_base_url: Base URL for Neuronpedia feature links.
        show_score_sign: Render directional scores as separate Sign / ``|Score|`` columns
            (use with signed score sources such as ``signed_influence``).
        feature_explanations: Optional ``(layer, feature_index) -> explanation`` mapping
            (e.g. from :func:`resolve_feature_explanations`); adds an Explanation column
            with empty cells for unmapped features.
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
        if feature_explanations is not None:
            body += "<th>Explanation</th>"
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
            explanation_cell = ""
            if feature_explanations is not None:
                explanation = feature_explanations.get((int(layer), int(feat_idx)), "")
                explanation_cell = f"<td>{html.escape(explanation)}</td>"
            body += f"<tr><td>{j + 1}</td>{node_cell}{score_cell}{explanation_cell}</tr>"
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


def best_variant_token_ids(
    tokenizer: Any,
    tokens: Sequence[str],
    reference_logits: torch.Tensor,
    *,
    use_best_variant: bool = True,
) -> list[int]:
    """Resolve tokens to the plain-vs-leading-space variant the model favors under reference logits.

    SentencePiece/BPE vocabularies carry distinct ids for ``"Fruit"`` vs ``" Fruit"`` (``▁Fruit``);
    which variant a model concentrates mass on depends on the prompt position (post-newline chat
    answer positions favor the bare variant, mid-sentence positions the space-prefixed one). By
    default each token resolves to whichever variant carries the higher reference (typically
    pre-intervention) logit; pass ``use_best_variant=False`` to always take the bare spelling.
    """
    reference_1d = _ensure_1d_logits(reference_logits)
    resolved: list[int] = []
    for token in tokens:
        variant_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in (token, " " + token)]
        if use_best_variant:
            resolved.append(max(variant_ids, key=lambda i: float(reference_1d[i])))
        else:
            resolved.append(variant_ids[0])
    return resolved


def display_target_gap(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    target_a: tuple[str, int],
    target_b: tuple[str, int],
    title: str = "Target gap",
) -> tuple[float, float]:
    """Render a consolidated pre/post table for a two-token target contrast and return the gaps.

    One reusable display for any pre/post logit (or activation-derived logit) gap — feature-mediated
    interventions, direct hook-tensor steering, ablations, etc. Shows each target token's pre/post
    probability and logit with its per-token delta, plus a highlighted ``A − B`` gap row. Returns
    ``(pre_gap, post_gap)`` so callers can assert on steering outcomes without recomputing.
    """
    pre_1d = _ensure_1d_logits(pre_logits)
    post_1d = _ensure_1d_logits(post_logits)
    pre_probs = torch.softmax(pre_1d, dim=-1)
    post_probs = torch.softmax(post_1d, dim=-1)

    (label_a, id_a), (label_b, id_b) = target_a, target_b
    pre_gap = float(pre_1d[id_a] - pre_1d[id_b])
    post_gap = float(post_1d[id_a] - post_1d[id_b])

    def _cell(value: str, *, align: str = "right", bold: bool = False) -> str:
        weight = "font-weight:bold;" if bold else ""
        cell_style = f"text-align:{align};padding:3px 6px;border:1px solid rgba(150,150,150,0.5);{weight}"
        return f'<td style="{cell_style}">{value}</td>'

    rows = ""
    for i, (label, tid) in enumerate(((label_a, id_a), (label_b, id_b))):
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        delta = float(post_1d[tid] - pre_1d[tid])
        rows += (
            f'<tr class="{row_class}">'
            + _cell(html.escape(label), align="left")
            + _cell(format_prob(float(pre_probs[tid]), precision=3))
            + _cell(format_prob(float(post_probs[tid]), precision=3))
            + _cell(f"{float(pre_1d[tid]):.4f}")
            + _cell(f"{float(post_1d[tid]):.4f}")
            + _cell(f"{delta:+.4f}")
            + "</tr>\n"
        )
    rows += (
        '<tr style="background:rgba(120,180,240,0.15);">'
        + _cell(f"Gap ({html.escape(label_a)} − {html.escape(label_b)})", align="left", bold=True)
        + _cell("")
        + _cell("")
        + _cell(f"{pre_gap:+.4f}", bold=True)
        + _cell(f"{post_gap:+.4f}", bold=True)
        + _cell(f"{post_gap - pre_gap:+.4f}", bold=True)
        + "</tr>\n"
    )

    title_html = (
        f'<div style="font-weight:bold;font-size:14px;margin-bottom:4px;padding:4px 6px;'
        f'border-radius:3px;background:#555;color:white;display:inline-block;">'
        f"{html.escape(title)}</div>"
        if title
        else ""
    )
    header_cell = (
        'style="text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);'
        'background:rgba(200,200,200,0.3);"'
    )
    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:640px;margin-bottom:10px;font-size:13px;">
        {title_html}
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="text-align:left;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Token</th>
                    <th {header_cell}>Pre prob</th>
                    <th {header_cell}>Post prob</th>
                    <th {header_cell}>Pre logit</th>
                    <th {header_cell}>Post logit</th>
                    <th {header_cell}>Δ</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    display(HTML(markup))
    return pre_gap, post_gap


class SteeringDisplaySummary(NamedTuple):
    """Values extracted/derived from an ``intervention_from_concept`` result by ``display_steering_results``."""

    steered_features: list[tuple[int, int, int]]
    target_ids: tuple[int, int]
    pre_gap: float
    post_gap: float
    direction: torch.Tensor
    pre_logits: torch.Tensor
    post_logits: torch.Tensor


def display_steering_results(
    pipeline_results: Any,
    tokenizer: Any,
    target_tokens: Sequence[str],
    *,
    neuronpedia_model: str | None = None,
    neuronpedia_set: str = "gemmascope-transcoder-16k",
    neuronpedia_base_url: str = "https://www.neuronpedia.org",
    feature_explanations: Mapping[tuple[int, int], str] | None = None,
    min_layer: int | None = None,
    use_best_variant: bool = True,
    features_label: str = "Steered Features (signed influence)",
    gap_title: str = "Feature-mediated steering — target gap",
) -> SteeringDisplaySummary:
    """One-call display + extraction for an ``it.intervention_from_concept`` result.

    Composes the underlying helpers — the linked/signed steered-features table
    (:func:`display_top_features_comparison`) and the consolidated pre/post target-gap table
    (:func:`display_target_gap`) — after extracting the steered features, unit concept direction,
    flattened pre/post logits, and best-variant target token ids from ``pipeline_results``. Returns a
    :class:`SteeringDisplaySummary` so callers can assert on the gaps (``post_gap > pre_gap``) and
    reuse the direction/target ids in later phases without re-deriving anything.

    Args:
        pipeline_results: ``AnalysisBatch`` returned by ``it.intervention_from_concept`` (needs
            ``top_feature_ids``, ``top_feature_scores``, ``concept_direction``, and
            ``pre/post_intervention_logits``).
        tokenizer: Tokenizer used for best-variant target-token resolution.
        target_tokens: Two-token target contrast ``(A, B)`` — the displayed/returned gap is ``A − B``.
        min_layer: When provided, validates every steered feature's layer is ``>= min_layer``
            (the ``FeatureSelectionSpec.layer_slice`` contract).
        use_best_variant: Resolve each target token to the plain-vs-leading-space variant the model
            favors pre-intervention (see :func:`best_variant_token_ids`).
    """
    steered_features = [tuple(f.tolist()) for f in pipeline_results.top_feature_ids]
    if min_layer is not None:
        assert all(f[0] >= min_layer for f in steered_features), (
            f"layer_slice constraint violated: steered features {steered_features} include layers < {min_layer}"
        )
    direction = pipeline_results.concept_direction.detach().float().cpu().reshape(-1)
    pre_logits = _ensure_1d_logits(pipeline_results.pre_intervention_logits.float().cpu())
    post_logits = _ensure_1d_logits(pipeline_results.post_intervention_logits.float().cpu())

    target_a_id, target_b_id = best_variant_token_ids(
        tokenizer, list(target_tokens)[:2], pre_logits, use_best_variant=use_best_variant
    )
    display_top_features_comparison(
        {features_label: steered_features},
        {features_label: pipeline_results.top_feature_scores.tolist()},
        neuronpedia_model=neuronpedia_model,
        neuronpedia_set=neuronpedia_set,
        neuronpedia_base_url=neuronpedia_base_url,
        show_score_sign=True,
        feature_explanations=feature_explanations,
    )
    pre_gap, post_gap = display_target_gap(
        pre_logits,
        post_logits,
        (str(target_tokens[0]), target_a_id),
        (str(target_tokens[1]), target_b_id),
        title=gap_title,
    )
    return SteeringDisplaySummary(
        steered_features=steered_features,
        target_ids=(target_a_id, target_b_id),
        pre_gap=pre_gap,
        post_gap=post_gap,
        direction=direction,
        pre_logits=pre_logits,
        post_logits=post_logits,
    )


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


# ---------------------------------------------------------------------------
# Input/output decoupling displays (feature IO profiles)
# ---------------------------------------------------------------------------


def display_concept_positions(tokenizer: Any, token_ids: Sequence[int], concept_positions: set[int]) -> None:
    """Render the prompt as token chips with concept-token positions highlighted."""
    chips = []
    for pos, tid in enumerate(token_ids):
        token_text = html.escape(tokenizer.decode([int(tid)]))
        if pos in concept_positions:
            style = "background:#2471A3;color:white;font-weight:bold;"
        else:
            style = "background:rgba(200,200,200,0.25);"
        chips.append(
            f'<span title="pos {pos}" style="{style}padding:2px 5px;margin:1px;border-radius:3px;'
            f'font-family:monospace;font-size:12px;display:inline-block;">{token_text}</span>'
        )
    markup = (
        '<div style="font-family:system-ui,-apple-system,sans-serif;font-size:13px;margin-bottom:10px;">'
        '<div style="font-weight:bold;font-size:14px;margin-bottom:4px;padding:4px 6px;border-radius:3px;'
        'background:#555;color:white;display:inline-block;">Prompt tokens — concept positions highlighted</div>'
        f'<div style="line-height:2;">{"".join(chips)}</div>'
        f"<div>concept-token positions: <b>{sorted(concept_positions)}</b></div></div>"
    )
    display(HTML(markup))


def display_feature_decoupling_table(
    profiles: Sequence[Any],
    feature_explanations: Mapping[tuple[int, int], str] | None = None,
    *,
    target_label: str = "A−B",
    title: str = "Feature input/output decoupling",
) -> None:
    """Render :class:`~it_examples.utils.example_helpers.FeatureIOProfile` rows as a styled table.

    Rows sort by ``|output_projection|`` (the steering-relevant quantity). The decoupling signature
    is a large ``|output proj|`` with a near-zero input concept share; a high input share with a
    projection *against* the concept marks a suppressor-motif exemplar.
    """
    header_cell = 'style="padding:3px 6px;border:1px solid rgba(150,150,150,0.5);background:rgba(200,200,200,0.3);"'

    def _cell(value: str, *, align: str = "right", bold: bool = False) -> str:
        weight = "font-weight:bold;" if bold else ""
        cell_style = f"text-align:{align};padding:3px 6px;border:1px solid rgba(150,150,150,0.5);{weight}"
        return f'<td style="{cell_style}">{value}</td>'

    rows = ""
    for i, prof in enumerate(sorted(profiles, key=lambda r: -abs(r.output_projection))):
        explanation = (feature_explanations or {}).get((prof.layer, prof.feature), "")
        decoupled = abs(prof.output_projection) >= 0.03 and prof.input_concept_share < 0.05
        suppressor = prof.input_concept_share >= 0.3 and prof.output_projection < -0.03
        marker = "decoupled" if decoupled else ("suppressor-motif" if suppressor else "")
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        rows += (
            f'<tr class="{row_class}">'
            + _cell(f"L{prof.layer}/{prof.feature}", align="left")
            + _cell(f"{prof.input_concept_share:.3f}", bold=prof.input_concept_share == 0.0)
            + _cell(f"{prof.activation_mass:.2f}")
            + _cell(f"{prof.output_projection:+.4f}", bold=abs(prof.output_projection) >= 0.03)
            + _cell(html.escape(marker), align="left", bold=bool(marker))
            + _cell(html.escape(str(explanation))[:80], align="left")
            + "</tr>\n"
        )

    title_html = (
        f'<div style="font-weight:bold;font-size:14px;margin-bottom:4px;padding:4px 6px;'
        f'border-radius:3px;background:#555;color:white;display:inline-block;">{html.escape(title)}</div>'
    )
    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:880px;margin-bottom:10px;font-size:13px;">
        {title_html}
        <table style="width:100%;border-collapse:collapse;">
            <thead><tr>
                <th {header_cell}>Feature</th>
                <th {header_cell}>Input concept share</th>
                <th {header_cell}>Act mass</th>
                <th {header_cell}>Output proj ({html.escape(target_label)})</th>
                <th {header_cell}>Signature</th>
                <th {header_cell}>Explanation</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <div style="font-size:12px;color:#666;">decoupled = large |output proj| with ~zero input concept share;
        suppressor-motif = fires on concept contexts yet projects against the concept token.</div>
    </div>
    """
    display(HTML(markup))


_LABEL_CANDIDATE_POSITIONS = (
    "top center",
    "bottom center",
    "middle right",
    "middle left",
    "top right",
    "top left",
    "bottom right",
    "bottom left",
)
# matplotlib analogue of each plotly textposition: (offset-points, ha, va)
_MPL_LABEL_ANCHORS = {
    "top center": ((0, 6), "center", "bottom"),
    "bottom center": ((0, -6), "center", "top"),
    "middle right": ((8, 0), "left", "center"),
    "middle left": ((-8, 0), "right", "center"),
    "top right": ((6, 6), "left", "bottom"),
    "top left": ((-6, 6), "right", "bottom"),
    "bottom right": ((6, -6), "left", "top"),
    "bottom left": ((-6, -6), "right", "top"),
}


def _collision_aware_label_positions(
    points: Any,
    labels: Sequence[str],
    marker_px: Sequence[float],
    extent: tuple[float, float, float, float] | None = None,
    plot_px: tuple[float, float] = (600.0, 400.0),
    char_px: float = 5.5,
    label_h_px: float = 12.0,
) -> list[str]:
    """Choose a plotly ``textposition`` per labeled point so labels stay individually legible.

    Approximates each candidate label's bounding box in plot pixels (accounting for text length and the point's
    marker diameter) and greedily picks the first anchor position whose box intersects neither previously placed
    labels nor other analyzed markers — so near-coincident markers fan their labels out instead of rendering them
    on top of each other. Falls back to round-robin if every candidate collides (dense worst case). Pass ``extent``
    as the FULL data extent backing the axes (e.g. including background points) — normalizing by the labeled
    points' own extent alone overestimates the pixel space between them.
    """
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    if not xs:
        return []
    x0, x1, y0, y1 = extent if extent is not None else (min(xs), max(xs), min(ys), max(ys))
    span_x = (x1 - x0) or 1.0
    span_y = (y1 - y0) or 1.0
    pts = [((x - x0) / span_x * plot_px[0], (y - y0) / span_y * plot_px[1]) for x, y in zip(xs, ys)]

    def label_box(i: int, pos: str) -> tuple[float, float, float, float]:
        w, h, r = char_px * len(labels[i]), label_h_px, marker_px[i] / 2.0
        dx = -(r + w / 2.0) if "left" in pos else (r + w / 2.0) if "right" in pos else 0.0
        dy = (r + h / 2.0) if "top" in pos else -(r + h / 2.0) if "bottom" in pos else 0.0
        cx, cy = pts[i][0] + dx, pts[i][1] + dy
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def overlaps(b1, b2) -> bool:
        return not (b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1])

    marker_boxes = [(x - m / 2.0, y - m / 2.0, x + m / 2.0, y + m / 2.0) for (x, y), m in zip(pts, marker_px)]
    placed: list[tuple[float, float, float, float]] = []
    chosen_positions: list[str] = []
    for i in range(len(pts)):
        obstacles = placed + [b for j, b in enumerate(marker_boxes) if j != i]
        chosen = next(
            (pos for pos in _LABEL_CANDIDATE_POSITIONS if not any(overlaps(label_box(i, pos), b) for b in obstacles)),
            _LABEL_CANDIDATE_POSITIONS[i % len(_LABEL_CANDIDATE_POSITIONS)],
        )
        placed.append(label_box(i, chosen))
        chosen_positions.append(chosen)
    return chosen_positions


def plot_decoder_projection_map(
    analyzed_profiles: Sequence[Any],
    analyzed_vectors: torch.Tensor,
    background_vectors: torch.Tensor,
    *,
    feature_explanations: Mapping[tuple[int, int], str] | None = None,
    target_label: str = "A−B",
    title: str = "Decoder-vector projection map",
    static_companion: bool = False,
) -> None:
    """Interactive decoder-space projection (UMAP w/ PCA fallback) with per-feature hover details.

    Uses plotly for hover tooltips (feature id, input concept share, activation mass, signed output projection,
    explanation) with the analyzed features colored by signed output projection over a gray active-feature background.
    Axis tick labels are hidden deliberately: UMAP coordinates are non-metric (arbitrary rotation/scale; only local
    neighborhood structure is meaningful), so raw axis numbers invite over-reading. The interactive figure is embedded
    as an HTML div bootstrapping plotly.js from CDN (``fig.to_html(include_plotlyjs="cdn")``) rather than via
    ``fig.show()`` — renderer autodetection can silently pick the browser renderer under headless kernels (spawning a
    local server and writing nothing usable into the notebook), while the embedded div renders with full hover in any
    HTML-capable notebook viewer, matching the latent-dynamics analysis charts. SVG traces only (``go.Scatter``):
    WebGL ``Scattergl`` traces render blank in several viewers. Set ``static_companion=True`` to ALSO render a static
    matplotlib PNG (useful for offline viewers without CDN access); matplotlib likewise serves as the fallback when
    plotly is unavailable. The trace legend renders horizontally BELOW the plot so it cannot collide with the
    colorbar on the right, and per-point labels use collision-aware anchor placement
    (``_collision_aware_label_positions``) so near-coincident analyzed features remain individually legible.
    """
    from tests.nb_experiments.concept_direction.analysis.latent_state_projection import project_embeddings

    matrix = torch.cat(
        [
            torch.as_tensor(analyzed_vectors, dtype=torch.float32),
            torch.as_tensor(background_vectors, dtype=torch.float32),
        ],
        dim=0,
    )
    proj_result = project_embeddings(matrix, method="umap", n_components=2)
    coords = proj_result.coordinates
    n_analyzed = len(analyzed_profiles)
    fg, bg = coords[:n_analyzed], coords[n_analyzed:]
    method_label = f"{proj_result.method.upper()} ({proj_result.backend})"
    axes_extent = (
        float(coords[:, 0].min()),
        float(coords[:, 0].max()),
        float(coords[:, 1].min()),
        float(coords[:, 1].max()),
    )

    try:
        import plotly.graph_objects as go

        hover_text = []
        for prof in analyzed_profiles:
            explanation = (feature_explanations or {}).get((prof.layer, prof.feature), "") or "—"
            hover_text.append(
                f"L{prof.layer}/{prof.feature}<br>input concept share: {prof.input_concept_share:.3f}"
                f"<br>act mass: {prof.activation_mass:.2f}"
                f"<br>output proj ({target_label}): {prof.output_projection:+.4f}"
                f"<br>explanation: {html.escape(str(explanation))[:80]}"
            )
        point_labels = [f"L{p.layer}/{p.feature}" for p in analyzed_profiles]
        marker_px = [10 + 22 * p.input_concept_share for p in analyzed_profiles]
        label_positions = _collision_aware_label_positions(fg, point_labels, marker_px, extent=axes_extent)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bg[:, 0],
                y=bg[:, 1],
                mode="markers",
                name="active-feature background",
                marker=dict(size=4, color="lightgray"),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fg[:, 0],
                y=fg[:, 1],
                mode="markers+text",
                name="analyzed features",
                text=point_labels,
                textposition=label_positions,
                textfont=dict(size=9),
                hovertext=hover_text,
                hoverinfo="text",
                marker=dict(
                    size=marker_px,
                    color=[p.output_projection for p in analyzed_profiles],
                    colorscale="RdBu_r",
                    cmid=0.0,
                    showscale=True,
                    colorbar=dict(title=dict(text=f"output proj<br>({target_label})", side="right"), len=0.85),
                    line=dict(width=1, color="black"),
                ),
            )
        )
        fig.update_layout(
            title=f"{title} — {method_label}; marker size = input concept share",
            width=820,
            height=560,
            template="simple_white",
            xaxis=dict(showticklabels=False, title=None),
            yaxis=dict(showticklabels=False, title=None),
            legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="left", x=0),
        )
        display(HTML(fig.to_html(full_html=False, include_plotlyjs="cdn")))
        plotly_rendered = True
    except Exception:
        plotly_rendered = False

    if static_companion or not plotly_rendered:
        import matplotlib.pyplot as plt

        static_fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(bg[:, 0], bg[:, 1], s=8, c="lightgray", label="active-feature background")
        sc = ax.scatter(
            fg[:, 0],
            fg[:, 1],
            s=[80 + 320 * p.input_concept_share for p in analyzed_profiles],
            c=[p.output_projection for p in analyzed_profiles],
            cmap="coolwarm",
            edgecolors="black",
            linewidths=0.8,
            label="analyzed features",
        )
        static_labels = [f"L{p.layer}/{p.feature}" for p in analyzed_profiles]
        static_marker_px = [10 + 22 * p.input_concept_share for p in analyzed_profiles]
        for prof, (x, y), pos in zip(
            analyzed_profiles,
            fg,
            _collision_aware_label_positions(fg, static_labels, static_marker_px, extent=axes_extent),
        ):
            (dx, dy), ha, va = _MPL_LABEL_ANCHORS[pos]
            ax.annotate(
                f"L{prof.layer}/{prof.feature}",
                (x, y),
                fontsize=8,
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
            )
        static_fig.colorbar(sc, ax=ax, label=f"decoder proj onto {target_label} (signed)")
        static_suffix = " (static companion)" if plotly_rendered else ""
        ax.set_title(f"{title} — {method_label}; marker size = input concept share{static_suffix}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.show()
